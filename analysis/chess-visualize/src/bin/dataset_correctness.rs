use analysis_util::parquet_reader::{FinalizedSample, ParquetDataReader};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;

/// A struct to hold detailed information about a validation error.
#[derive(Debug)]
struct ValidationError {
    file_path: PathBuf,
    sim_id: String,
    move_count: u32,
    error_message: String,
}

fn check_sample_correctness(sample: &FinalizedSample) -> Vec<String> {
    let mut errors = Vec::new();
    const EPS: f32 = 1e-5;

    // Rule 1: All individual scalars of the z values must be between -1 and 1
    for &z in &sample.z_values {
        if !(-1.0..=1.0).contains(&z) {
            errors.push(format!("z_value out of range [-1, 1]: {}", z));
        }
    }

    // Rule 2: All individual scalars of the Q values must be between 0 and 1
    for &q in &sample.inner.q_values {
        if !(0.0..=1.0).contains(&q) {
            errors.push(format!("q_value out of range [0, 1]: {}", q));
        }
    }

    // Rule 3: All policy elements must be between 0 and 1
    for &p in &sample.inner.mcts_policy {
        if !(0.0..=1.0).contains(&p) {
            errors.push(format!("Policy element out of range [0, 1]: {}", p));
        }
    }

    // Rule 4: The policy must sum to 1
    let policy_sum: f32 = sample.inner.mcts_policy.iter().sum();
    if (policy_sum - 1.0).abs() > EPS {
        errors.push(format!(
            "Policy does not sum to 1 (sum: {}, difference: {})",
            policy_sum,
            policy_sum - 1.0
        ));
    }

    // Rule 5: Each move mask must have at least one true boolean and at max 384
    let num_legal_moves = sample.inner.legal_moves_mask.iter().filter(|&&b| b).count();

    if num_legal_moves == 0 {
        errors.push("Legal moves mask has no true values (0 legal moves found).".to_string());
    }

    if num_legal_moves > 384 {
        errors.push(format!(
            "Legal moves mask has more than 384 true values (found: {})",
            num_legal_moves
        ));
    }

    errors
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let folder_path = "./testing/samples_hybrid_bad";

    // Recursively find all parquet files in the specified directory.
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
        "Found {} parquet files. Starting validation...",
        parquet_files.len()
    );

    let all_errors: Mutex<Vec<ValidationError>> = Mutex::new(Vec::new());
    let total_samples_checked = AtomicUsize::new(0);
    let total_files_processed = AtomicUsize::new(0);

    parquet_files.par_iter().for_each(|file_path| {
        println!("Processing file: {}", file_path.display());

        match ParquetDataReader::read_file(file_path) {
            Ok(all_samples) => {
                total_samples_checked.fetch_add(all_samples.samples.len(), Ordering::Relaxed);

                for sample in &all_samples.samples {
                    let errors = check_sample_correctness(sample);
                    if !errors.is_empty() {
                        let mut guard = all_errors.lock().unwrap();
                        for error_message in errors {
                            guard.push(ValidationError {
                                file_path: file_path.clone(),
                                sim_id: sample.sim_id.clone(),
                                move_count: sample.inner.current_move_count,
                                error_message,
                            });
                        }
                    }
                }
            }
            Err(e) => {
                // Collect errors for files that fail to be read or parsed.
                let mut guard = all_errors.lock().unwrap();
                guard.push(ValidationError {
                    file_path: file_path.clone(),
                    sim_id: "N/A".to_string(),
                    move_count: 0,
                    error_message: format!("Failed to read or process parquet file: {}", e),
                });
            }
        }
        total_files_processed.fetch_add(1, Ordering::Relaxed);
    });

    let final_errors = all_errors.into_inner()?;
    let num_errors = final_errors.len();

    println!("\n--- Validation Complete ---");
    println!(
        "Files processed: {}",
        total_files_processed.load(Ordering::Relaxed)
    );
    println!(
        "Samples checked: {}",
        total_samples_checked.load(Ordering::Relaxed)
    );
    println!("Errors found: {}", num_errors);
    println!("---------------------------\n");

    if num_errors > 0 {
        println!("Error Details:");
        for error in &final_errors {
            println!(
                "- File: {}\n  SimID: {}, Move: {}\n  Error: {}",
                error.file_path.display(),
                error.sim_id,
                error.move_count,
                error.error_message
            );
        }
        std::process::exit(1);
    } else {
        println!("✅ All files and samples are correct.");
        Ok(())
    }
}
