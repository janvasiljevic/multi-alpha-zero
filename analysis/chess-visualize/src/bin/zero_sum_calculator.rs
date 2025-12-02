use analysis_util::parquet_reader::{FinalizedSample, ParquetDataReader};
use game_tri_chess::basics::COLORS;
use game_tri_chess::phase::Phase::Normal;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::{Board, InputMapper, ReverseInputMapper};
use maz_core::values_const::ValuesPov;
use ndarray::Array2;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
// Import necessary traits
use std::path::PathBuf;
use walkdir::WalkDir;

fn check_sample_correctness(sample: &FinalizedSample) -> Option<f32> {
    let domain_maper = ChessDomainMapper;
    let input_shape = domain_maper.input_board_shape();

    let input_array = Array2::from_shape_vec(
        (input_shape[0], input_shape[1]),
        sample.inner.encoded_board.clone(),
    )
    .unwrap();

    let board =
        domain_maper.decode_input(&input_array.view(), &vec![sample.inner.player_index as f32]);

    if board.player_num_of_active() == 2 {
        return match board.state.phase {
            Normal(state) => {
                let abs_values = ValuesPov::<3>::from_slice(
                    &sample.z_values,
                    0.0,
                    sample.inner.player_index,
                );
                let abs_values = abs_values.abs().value_abs;

                let mut sum = 0.0;

                for color in COLORS {
                    if state.is_present(color) {
                        sum += abs_values[color as usize];
                    }
                }
                Some(sum)
            }
            _ => None,
        };
    }
    None
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let folder_path = "./testing/samples_new";

    let parquet_files: Vec<PathBuf> = WalkDir::new(&folder_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file() && e.path().extension().map_or(false, |ext| ext == "parquet")
        })
        .map(|e| e.into_path())
        .collect();

    if parquet_files.is_empty() {
        println!("No .parquet files found. Exiting.");
        return Ok(());
    }

    println!(
        "Found {} parquet files. Starting processing...",
        parquet_files.len()
    );

    let all_sums: Vec<f32> = parquet_files
        .par_iter()
        .flat_map(|file_path| {
            match ParquetDataReader::read_file(file_path) {
                Ok(all_samples) => {
                    all_samples
                        .samples
                        .iter()
                        .filter_map(check_sample_correctness)
                        .collect::<Vec<f32>>() // Explicitly collect into a Vec.
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Skipping file due to error '{}': {}",
                        file_path.display(),
                        e
                    );
                    vec![]
                }
            }
        })
        .collect();

    println!("Finished processing. Got {} sums.", all_sums.len());

    println!("Writing sums to sums.txt...");
    let file_txt = File::create("sums.txt")?;
    let mut writer_txt = BufWriter::new(file_txt);
    for sum in &all_sums {
        writeln!(writer_txt, "{}", sum)?;
    }
    println!("Successfully wrote to sums.txt");

    Ok(())
}
