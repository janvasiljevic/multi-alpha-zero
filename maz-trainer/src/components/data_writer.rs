use crate::self_play_game::FinalizedSample;
use arrow::array::BooleanBuilder;
use arrow::array::Float32Builder;
use arrow::array::GenericListBuilder;
use arrow::array::ListBuilder;
use arrow::array::StringBuilder;
use arrow::array::UInt32Builder;
use arrow::array::{Array, FixedSizeListBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use maz_core::mapping::{Board, BoardMapper};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression::SNAPPY;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{error, info, warn};

fn create_f32_builder(capacity: usize) -> ListBuilder<Float32Builder> {
    GenericListBuilder::with_capacity(Float32Builder::new(), capacity)
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)))
}

fn create_fixed_length_boolean_builder(
    capacity: usize,
    list_size: usize,
) -> FixedSizeListBuilder<BooleanBuilder> {
    FixedSizeListBuilder::with_capacity(BooleanBuilder::new(), list_size as i32, capacity)
        .with_field(Arc::new(Field::new("item", DataType::Boolean, false)))
}

fn create_fixed_length_f32_builder(
    capacity: usize,
    list_size: usize,
) -> FixedSizeListBuilder<Float32Builder> {
    FixedSizeListBuilder::with_capacity(Float32Builder::new(), list_size as i32, capacity)
        .with_field(Arc::new(Field::new("item", DataType::Float32, false)))
}

#[derive(Error, Debug)]
pub enum DataWriterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parquet processing error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("Arrow processing error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}
type WriterResult<T> = Result<T, DataWriterError>;

pub struct BufferedDataWriter {
    length: usize,
    chunk_size: usize,
    output_dir: PathBuf,
    arrow_schema: SchemaRef,
    file_name_fn: Arc<dyn Fn(&Path) -> PathBuf + Send + Sync>,

    state_builder: FixedSizeListBuilder<BooleanBuilder>,
    policy_builder: FixedSizeListBuilder<Float32Builder>,
    masked_policy_builder: FixedSizeListBuilder<BooleanBuilder>,
    z_value_builder: ListBuilder<Float32Builder>,
    moves_left_builder: UInt32Builder,

    move_count_builder: UInt32Builder,
    player_builder: UInt32Builder,
    tree_depth_0_builder: UInt32Builder,
    tree_depth_1_builder: UInt32Builder,
    q_values_builder: ListBuilder<Float32Builder>,
    sim_id_builder: StringBuilder,
}

impl BufferedDataWriter {
    pub fn new<B: Board, BM: BoardMapper<B>>(
        chunk_size: usize,
        output_dir: PathBuf,
        file_name_fn: Arc<dyn Fn(&Path) -> PathBuf + Send + Sync>,
        mapper: BM,
    ) -> WriterResult<Self> {
        fs::create_dir_all(&output_dir)?;

        let input_len = mapper.input_board_shape().iter().product::<usize>();
        let policy_len = mapper.policy_len();

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("sim_id", DataType::Utf8, false),
            // Most important fields for the SPGPosition: For training!
            Field::new(
                "state",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Boolean, false)),
                    input_len as i32,
                ),
                false,
            ),
            Field::new(
                "policy",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    policy_len as i32,
                ),
                false,
            ),
            Field::new(
                "masked_policy",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Boolean, false)),
                    policy_len as i32,
                ),
                false,
            ),
            Field::new("value", DataType::new_list(DataType::Float32, false), false),
            Field::new("moves_left", DataType::UInt32, false),
            // Needed for reconstruction of the game state
            Field::new("player", DataType::UInt32, false),
            // Additional metadata
            Field::new("move_count", DataType::UInt32, false),
            // Q values for all players at this position. Not the same as 'value' which
            // is the final outcome of the game (z).
            // q_value is great for training an auxiliary head, but suffers from the horizon
            // effect.
            Field::new(
                "q_value",
                DataType::new_list(DataType::Float32, false),
                false,
            ),
            Field::new("tree_depth_0", DataType::UInt32, false),
            Field::new("tree_depth_1", DataType::UInt32, false),
        ]));

        let sim_id_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 10);

        let state_builder = create_fixed_length_boolean_builder(chunk_size, input_len);
        let value_builder = create_f32_builder(chunk_size);
        let policy_builder = create_fixed_length_f32_builder(chunk_size, policy_len);

        let masked_policy_builder = create_fixed_length_boolean_builder(chunk_size, policy_len);
        let moves_left_builder = UInt32Builder::with_capacity(chunk_size);
        let move_count_builder = UInt32Builder::with_capacity(chunk_size);
        let player_builder = UInt32Builder::with_capacity(chunk_size);
        let tree_depth_0_builder = UInt32Builder::with_capacity(chunk_size);
        let tree_depth_1_builder = UInt32Builder::with_capacity(chunk_size);
        let abs_mcts_values_builder = create_f32_builder(chunk_size);

        info!(
            "Initialized BufferedDataWriter. Flushing to '{}' every {} samples.",
            output_dir.display(),
            chunk_size
        );

        Ok(Self {
            length: 0,
            chunk_size,
            output_dir,
            arrow_schema,
            file_name_fn,

            // Arrow builders for each field
            sim_id_builder,
            state_builder,
            z_value_builder: value_builder,
            policy_builder,
            masked_policy_builder,
            moves_left_builder,
            move_count_builder,
            player_builder,
            tree_depth_0_builder,
            tree_depth_1_builder,
            q_values_builder: abs_mcts_values_builder,
        })
    }

    /// Adds a batch of samples to the internal buffer.
    /// If the buffer size exceeds the `chunk_size`, it automatically flushes the data to a file.
    pub fn add_full_simulation<B: Board>(
        &mut self,
        samples: Vec<FinalizedSample<B>>,
        simulation_name: &str,
        auto_flush: bool,
    ) -> WriterResult<Option<PathBuf>> {
        self.length += samples.len();

        for sample in samples {
            self.sim_id_builder.append_value(simulation_name);

            for v in sample.inner.encoded_board.into_iter() {
                self.state_builder.values().append_value(v);
            }
            self.state_builder.append(true);

            for v in sample.inner.mcts_policy.into_iter() {
                self.policy_builder.values().append_value(v);
            }

            self.policy_builder.append(true);

            self.z_value_builder
                .append_value(sample.z_values.clone().iter().copied().map(Some));

            for v in sample.inner.legal_moves_mask.into_iter() {
                self.masked_policy_builder.values().append_value(v);
            }
            self.masked_policy_builder.append(true);

            self.moves_left_builder.append_value(sample.moves_left);

            self.move_count_builder
                .append_value(sample.inner.current_move_count);
            self.player_builder
                .append_value(sample.inner.player_index as u32);

            self.q_values_builder
                .append_value(sample.inner.aux_values.iter().copied().map(Some));
            self.tree_depth_0_builder
                .append_value(sample.inner.depth_range.0 as u32);
            self.tree_depth_1_builder
                .append_value(sample.inner.depth_range.1 as u32);
        }

        if self.length >= self.chunk_size && auto_flush {
            return self.flush();
        }

        Ok(None)
    }

    /// Writes all currently buffered samples to a new Parquet file.
    /// The internal buffer is cleared on success.
    pub fn flush(&mut self) -> WriterResult<Option<PathBuf>> {
        if self.length == 0 {
            return Ok(None);
        }

        // We could have buffers at precisely the chunk size,
        // but it's better to have a full game in one file, so the total sample in one file
        // are usually around chunk_size + up to max game length.
        let samples_to_process = self.length;

        let benchmark_start_time = Instant::now();

        let arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(self.sim_id_builder.finish()),
            Arc::new(self.state_builder.finish()),
            Arc::new(self.policy_builder.finish()),
            Arc::new(self.masked_policy_builder.finish()),
            Arc::new(self.z_value_builder.finish()),
            Arc::new(self.moves_left_builder.finish()),
            Arc::new(self.player_builder.finish()),
            Arc::new(self.move_count_builder.finish()),
            Arc::new(self.q_values_builder.finish()),
            Arc::new(self.tree_depth_0_builder.finish()),
            Arc::new(self.tree_depth_1_builder.finish()),
        ];

        let batch = RecordBatch::try_new(self.arrow_schema.clone(), arrays)?;

        let file_path = (self.file_name_fn)(&self.output_dir);

        let file = File::create(&file_path)?;

        let writer_props = WriterProperties::builder()
            .set_column_dictionary_enabled("policy".into(), false)
            .set_compression(SNAPPY)
            .set_write_batch_size(4096) // Experiment with this
            .set_max_row_group_size(5000) // Your chunk size
            .build();

        let mut writer = ArrowWriter::try_new(file, self.arrow_schema.clone(), Some(writer_props))?;

        let start_of_write = Instant::now();

        writer.write(&batch)?;
        writer.close()?;

        let write_duration = start_of_write.elapsed();

        let file_size = file_path.metadata()?.len();

        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("This should be run in the sub-project");

        let relative_path = file_path
            .strip_prefix(workspace_root)
            .unwrap_or(file_path.as_path());

        info!(
            "Wrote {} samples to '{}' in {:.2?} (Actual write took {:.2?}. File size: {:.2} MB",
            samples_to_process,
            relative_path.display(),
            benchmark_start_time.elapsed(),
            write_duration,
            file_size as f64 / (1024.0 * 1024.0)
        );

        self.length = 0;

        Ok(Some(file_path))
    }

    /// Trims the data directory by removing the oldest files.
    ///
    /// The process stops when deleting the next oldest file would cause the total
    /// sample count to drop below `min_samples_target`. This ensures the final
    /// sample count is the smallest possible value that is still greater than
    /// or equal to the target.
    ///
    /// For example, with a `min_samples_target` of 30,000 and files containing
    /// [20k, 20k, 20k] samples (from oldest to newest), the oldest file will be
    /// deleted, leaving a total of 40,000 samples.
    pub fn trim_by_oldest(
        &self,
        min_samples_target: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        struct FileInfo {
            path: PathBuf,
            file_size_mb: f64,
            sample_count: i64,
        }

        let mut files_to_consider: Vec<FileInfo> = Vec::new();
        let mut total_samples: u64 = 0;

        for entry in fs::read_dir(&self.output_dir)?.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                match File::open(&path) {
                    Ok(file) => {
                        if let Ok(reader) = SerializedFileReader::new(file) {
                            let num_rows = reader.metadata().file_metadata().num_rows();
                            let file_meta = entry.metadata()?;
                            total_samples += num_rows as u64;

                            files_to_consider.push(FileInfo {
                                path: path.clone(),
                                // It's better to just use the file name convention
                                // created: file_meta.created().unwrap_or(SystemTime::UNIX_EPOCH),
                                file_size_mb: file_meta.len() as f64 / (1024.0 * 1024.0),
                                sample_count: num_rows,
                            });
                        } else {
                            warn!(
                                "Could not create Parquet reader for '{}'. Skipping.",
                                path.display()
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Could not open file '{}' to read metadata: {}. Skipping.",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }

        if total_samples <= min_samples_target {
            info!(
                "No trim needed. Current sample count {total_samples} is not above the target of {min_samples_target}."
            );
            return Ok(());
        }

        // This only works with the correct closure passed to the constructor
        files_to_consider.sort_by_key(|f| {
            let filename = f.path.file_name().unwrap().to_string_lossy();
            let parts: Vec<&str> = filename.split('_').collect();
            assert_eq!(parts.len(), 2, "Unexpected file name format: {}", filename);

            let first: u64 = parts[0].parse().expect("Invalid first number");
            let second_str = parts[1].trim_end_matches(".parquet");
            let second: u128 = second_str.parse().expect("Invalid timestamp");

            (first, second)
        });

        let mut freed_mb = 0.0;
        let mut samples_deleted = 0;
        let mut files_deleted = 0;

        let mut files_names_deleted = Vec::new();

        for file_info in files_to_consider {
            let samples_after_delete = total_samples - file_info.sample_count as u64;

            if samples_after_delete < min_samples_target {
                break;
            }

            match fs::remove_file(&file_info.path) {
                Ok(_) => {
                    total_samples = samples_after_delete;

                    freed_mb += file_info.file_size_mb;
                    samples_deleted += file_info.sample_count as u64;
                    files_deleted += 1;

                    files_names_deleted.push(
                        file_info
                            .path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .to_string(),
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to delete old data file '{}': {}. Skipping.",
                        file_info.path.display(),
                        e
                    );
                }
            }
        }

        info!(
            "Trim finished. Final sample count: {total_samples}. Deleted {files_deleted} files, freed {freed_mb:.2} MB, removed {samples_deleted} samples. Files deleted: {files_names_deleted:?}",
        );

        Ok(())
    }
}

impl Drop for BufferedDataWriter {
    fn drop(&mut self) {
        if self.length > 0 {
            warn!(
                "BufferedDataWriter is being dropped with {} samples still in buffer. Flushing...",
                self.length
            );
            if let Err(e) = self.flush() {
                error!("Failed to flush buffer on drop: {}", e);
            }
        }
    }
}
