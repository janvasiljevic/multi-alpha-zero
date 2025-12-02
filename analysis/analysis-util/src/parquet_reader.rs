use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, GenericListArray, RecordBatch,
    StringArray, UInt32Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone)]
pub struct Sample {
    pub player_index: usize,
    pub current_move_count: u32,
    pub mcts_policy: Vec<f32>,
    pub depth_range: (u32, u32),
    pub encoded_board: Vec<bool>,
    pub q_values: Vec<f32>,
    pub legal_moves_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct FinalizedSample {
    pub inner: Sample,
    pub z_values: Vec<f32>,
    pub moves_left: u32,
    pub sim_id: String,
}

#[derive(Error, Debug)]
pub enum DataReaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parquet processing error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("Arrow processing error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Schema mismatch or unexpected data type for column '{0}'")]
    SchemaMismatch(String),
}

type ReaderResult<T> = Result<T, DataReaderError>;

pub struct ParquetDataReader;

pub struct AllSamples {
    pub samples: Vec<FinalizedSample>,
}

impl ParquetDataReader {
    pub fn read_file(path: &Path) -> ReaderResult<AllSamples> {
        info!("Reading data from '{}'", path.display());

        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut all_samples = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;
            let samples_in_batch = Self::process_record_batch(&batch)?;
            all_samples.extend(samples_in_batch);
        }

        info!(
            "Successfully read {} samples from '{}'",
            all_samples.len(),
            path.display()
        );

        Ok(AllSamples {
            samples: all_samples,
        })
    }

    fn process_record_batch(batch: &RecordBatch) -> ReaderResult<Vec<FinalizedSample>> {
        let sim_id_arr = Self::get_array::<StringArray>(batch, "sim_id")?;
        let state_arr = Self::get_array::<FixedSizeListArray>(batch, "state")?;
        let policy_arr = Self::get_array::<FixedSizeListArray>(batch, "policy")?;
        let masked_policy_arr = Self::get_array::<FixedSizeListArray>(batch, "masked_policy")?;
        let value_arr = Self::get_list_array(batch, "value")?;
        let moves_left_arr = Self::get_array::<UInt32Array>(batch, "moves_left")?;
        let player_arr = Self::get_array::<UInt32Array>(batch, "player")?;
        let move_count_arr = Self::get_array::<UInt32Array>(batch, "move_count")?;
        let q_values_pov = Self::get_list_array(batch, "q_value")?;
        let tree_depth_0_arr = Self::get_array::<UInt32Array>(batch, "tree_depth_0")?;
        let tree_depth_1_arr = Self::get_array::<UInt32Array>(batch, "tree_depth_1")?;

        let mut samples = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let sample = FinalizedSample {
                sim_id: sim_id_arr.value(i).to_string(),
                moves_left: moves_left_arr.value(i),
                z_values: Self::get_f32_list_from_array(value_arr, i)?,
                inner: Sample {
                    player_index: player_arr.value(i) as usize,
                    current_move_count: move_count_arr.value(i),
                    depth_range: (tree_depth_0_arr.value(i), tree_depth_1_arr.value(i)),
                    encoded_board: Self::get_bool_fsl_from_array(state_arr, i)?,
                    mcts_policy: Self::get_f32_fsl_from_array(policy_arr, i)?,
                    legal_moves_mask: Self::get_bool_fsl_from_array(masked_policy_arr, i)?,
                    q_values: Self::get_f32_list_from_array(q_values_pov, i)?,
                },
            };
            samples.push(sample);
        }

        Ok(samples)
    }

    fn get_array<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> ReaderResult<&'a T> {
        batch
            .column_by_name(name)
            .ok_or_else(|| DataReaderError::SchemaMismatch(name.to_string()))?
            .as_any()
            .downcast_ref::<T>()
            .ok_or_else(|| DataReaderError::SchemaMismatch(name.to_string()))
    }

    fn get_list_array<'a>(
        batch: &'a RecordBatch,
        name: &str,
    ) -> ReaderResult<&'a GenericListArray<i32>> {
        Self::get_array::<GenericListArray<i32>>(batch, name)
    }

    fn get_f32_list_from_array(
        arr: &GenericListArray<i32>,
        index: usize,
    ) -> ReaderResult<Vec<f32>> {
        let list_val = arr.value(index);
        let float_arr = list_val
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DataReaderError::SchemaMismatch("inner list value".to_string()))?;
        Ok(float_arr.values().to_vec())
    }

    fn get_f32_fsl_from_array(arr: &FixedSizeListArray, index: usize) -> ReaderResult<Vec<f32>> {
        let list_val = arr.value(index);
        let float_arr = list_val
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DataReaderError::SchemaMismatch("inner fsl f32".to_string()))?;
        Ok(float_arr.values().to_vec())
    }

    fn get_bool_fsl_from_array(arr: &FixedSizeListArray, index: usize) -> ReaderResult<Vec<bool>> {
        let list_val = arr.value(index);
        let bool_arr = list_val
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| DataReaderError::SchemaMismatch("inner fsl bool".to_string()))?;
        Ok(bool_arr.values().iter().collect())
    }
}

impl AllSamples {
    pub fn get_unique_sim_ids(&self) -> Vec<String> {
        let mut sim_ids = self
            .samples
            .iter()
            .map(|s| s.sim_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        sim_ids.sort();
        sim_ids
    }

    pub fn get_all_samples_for_sim_id(&self, sim_id: &str) -> Vec<&FinalizedSample> {
        self.samples
            .iter()
            .filter(|s| s.sim_id == sim_id)
            .collect::<Vec<_>>()
    }
}
