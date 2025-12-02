use crate::self_play_game::{FinalizedSample, Simulation};
use maz_config::config::CycleCollectionSettings;
use maz_util::convert::hashmap_to_histogram;
use maz_util::math::{safe_div, RunningAverage};
use std::collections::HashMap;
use std::error::Error;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Endpoint;
use tonic::Request;
use tracing::{error, info};
use maz_core::mapping::{Board, BoardMapper};

mod training {
    tonic::include_proto!("training");
}

use crate::components::collector::training::TrainingSamplesUpdate;
use crate::components::data_writer::BufferedDataWriter;
use crate::components::reporter::create_reporter;
use crate::global_stats::CollectorStats;
use training::training_coordinator_client::TrainingCoordinatorClient;
use training::SendTrainingDataRequest;

enum WriterCommand<B: Board>{
    AddSimulation(Vec<FinalizedSample<B>>, String),
    FlushAndShutdown,
}

struct TaskAborter(tokio::task::JoinHandle<()>);

impl Drop for TaskAborter {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Collects all the samples from the generators, writes them to Parquet files, updates the Python
/// trainer via gRPC, reports stats periodically and in the end writes global stats to Tensorboard.
///
/// Spawns two additional tasks:
/// 1. A background writer task that handles writing simulations to Parquet files.
///     This needs to be in a separate thread, because the writes with more complex boards can
///     take up to 3 seconds (e.g. Chess). For smaller games like Hex this wouldn't be necessary,
///     because the writes are very fast (<50ms). If it isn't in a separate thread, it can block
///     the async runtime and cause delays in processing incoming simulations - leading to reporter
///     stats being off, meaning the stop
/// 2. A reporting task that periodically logs stats about the collection process.
///     The reporter is also responsible for setting the stop_signal!
#[bon::builder]
pub async fn run_collector_task<B: Board, BM: BoardMapper<B> + 'static>(
    board_mapper: BM,
    mut samples_rx: mpsc::UnboundedReceiver<Simulation<B>>,
    server_addr: &'static str,
    base_dir: PathBuf,
    max_samples: u64,
    loop_count: usize,
    cycle_settings: CycleCollectionSettings,
    stop_signal: Arc<AtomicBool>,
    global_stats: CollectorStats,
    parquest_chunk_size: usize,
) -> Result<CollectorStats, Box<dyn Error + Send + Sync>>
where
    B: Send + 'static,
{
    let samples_dir = base_dir.join("samples_new");

    if !samples_dir.exists() {
        create_dir_all(&samples_dir).expect("Failed to create samples directory");
    }

    let samples_archive_dir = base_dir.join("archive_samples");

    if !samples_archive_dir.exists() {
        create_dir_all(&samples_archive_dir).expect("Failed to create samples archive directory");
    }

    let channel = Endpoint::from_static(server_addr)
        .connect()
        .await?;

    let mut client = TrainingCoordinatorClient::new(channel);

    let total_samples_sent = Arc::new(AtomicUsize::new(0));
    let total_simulation = Arc::new(AtomicUsize::new(0));

    let reporting_handle = create_reporter(
        total_samples_sent.clone(),
        total_simulation.clone(),
        cycle_settings,
        stop_signal.clone(),
    );

    let _task_aborter = TaskAborter(reporting_handle);

    // Create a channel for gRPC stream.
    // The sender (grpc_tx) will be used in this task to send data.
    // The receiver (grpc_rx) will be consumed by the gRPC client call.
    let (grpc_tx, grpc_rx) = mpsc::channel(300);

    // Spawn a new task to handle the long-lived gRPC call.
    let mut stream_client = client.clone();

    let stream_handle = tokio::spawn(async move {
        let request_stream = ReceiverStream::new(grpc_rx);
        stream_client
            .send_training_data(Request::new(request_stream))
            .await
    });

    info!("Collector task started. gRPC stream is open. Waiting for training samples...");

    let start = Instant::now();

    // Channel to send commands TO the writer thread.
    let (writer_cmd_tx, mut writer_cmd_rx) = mpsc::unbounded_channel::<WriterCommand<B>>();

    //  Channel to receive flushed file paths FROM the writer thread.
    let (flushed_path_tx, mut flushed_path_rx) = mpsc::unbounded_channel::<PathBuf>();

    let writer_samples_dir = samples_dir.clone();
    let writer_loop_count = loop_count;

    let thread_board_mapper = board_mapper.clone();

    let writer_handle = tokio::task::spawn_blocking(move || {
        let mut writer = BufferedDataWriter::new(
            parquest_chunk_size,
            writer_samples_dir,
            Arc::new(move |dir: &Path| {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                dir.join(format!("{}_{}.parquet", writer_loop_count, timestamp))
            }),
            thread_board_mapper,
        )
        .expect("Writer thread failed to create BufferedDataWriter");

        while let Some(command) = writer_cmd_rx.blocking_recv() {
            match command {
                WriterCommand::AddSimulation(positions, sim_name) => {
                    match writer.add_full_simulation(positions, &sim_name, true) {
                        Ok(Some(path)) => {
                            if let Err(e) = flushed_path_tx.send(path) {
                                error!(
                                    "Writer thread: Failed to send flushed path back. Main task may have shut down. Error: {}",
                                    e
                                );
                                break;
                            }
                        }
                        Ok(None) => { /* Buffer was filled but not yet flushed, do nothing. */ }
                        Err(e) => error!("Writer thread encountered an error: {}", e),
                    }
                }
                WriterCommand::FlushAndShutdown => {
                    info!("Writer thread received shutdown command. Flushing final data...");
                    if let Ok(Some(path)) = writer.flush() {
                        if let Err(e) = flushed_path_tx.send(path) {
                            error!(
                                "Writer thread: Failed to send FINAL flushed path. Error: {}",
                                e
                            );
                        }
                    }
                    break;
                }
            }
        }

        writer
    });

    let mut simulations_done = 0u64; // Used for tracking filename
    let mut has_saved_to_archive = false;

    // Stats
    let mut game_lengths: HashMap<u32, usize> = HashMap::new();
    let mut total_entropy = RunningAverage::new();
    let mut move_count_to_entropy: HashMap<u32, RunningAverage> = HashMap::new();
    let mut winning_ratios: HashMap<usize, RunningAverage> = HashMap::new();

    'main_loop: loop {
        // `try_recv` to drain the channel of all pending paths
        // without ever blocking the main task.
        loop {
            match flushed_path_rx.try_recv() {
                Ok(path) => {
                    if !has_saved_to_archive {
                        let archive_path = samples_archive_dir.join(path.file_name().unwrap());
                        if let Err(e) = std::fs::copy(&path, &archive_path) {
                            error!("Failed to copy file to archive directory: {}", e);
                        }
                        has_saved_to_archive = true;
                    }

                    if let Err(e) = grpc_tx.try_send(SendTrainingDataRequest {
                        update: Some(TrainingSamplesUpdate {
                            parquet_file_path: path.to_string_lossy().to_string(),
                        }),
                    }) {
                        error!(
                            "gRPC channel full or closed when sending path. Error: {}",
                            e
                        );
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    // No more paths are waiting. Break the inner loop.
                    break;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // The writer thread has shut down. We shouldn't receive any more paths.
                    // We also can't receive any more simulations, so break the main loop.
                    info!("Flushed path channel disconnected. Shutting down collector loop.");
                    break 'main_loop;
                }
            }
        }

        let maybe_sim = samples_rx.recv().await;

        let game_sim = match maybe_sim {
            Some(sim) => sim,
            None => {
                // Input channel closed, break the main loop.
                break 'main_loop;
            }
        };

        // Stats increment, logging and collection
        total_samples_sent.fetch_add(game_sim.samples.len(), Ordering::Relaxed);
        total_simulation.fetch_add(1, Ordering::Relaxed);

        let values_abs = game_sim.outcome;

        for (player, count) in values_abs.value_abs().iter().enumerate() {
            winning_ratios.entry(player).or_default().add_sample(*count);
        }

        for sample in &game_sim.samples {
            total_entropy.add_sample(sample.inner.mcts_policy_entropy);
            move_count_to_entropy
                .entry(sample.inner.current_move_count)
                .or_default()
                .add_sample(sample.inner.mcts_policy_entropy);
        }
        game_lengths
            .entry(game_sim.actual_game_length)
            .and_modify(|e| *e += 1)
            .or_insert(1);

        // Send the simulation to the writer thread (non-blocking)
        let cmd = WriterCommand::AddSimulation(
            game_sim.samples,
            format!("{loop_count}_{simulations_done}"),
        );

        if let Err(e) = writer_cmd_tx.send(cmd) {
            error!(
                "Main task: Failed to send simulation to writer thread. It may have panicked. Error: {}",
                e
            );
            break 'main_loop;
        }
        simulations_done += 1;
    }

    let elapsed = start.elapsed();

    let total_samples = total_samples_sent.load(Ordering::Relaxed);
    let total_games = total_simulation.load(Ordering::Relaxed);

    let avg_samples_per_sec = safe_div(total_samples as f64, elapsed.as_secs_f64());
    let avg_games_per_sec = safe_div(total_games as f64, elapsed.as_secs_f64());

    if let Err(e) = writer_cmd_tx.send(WriterCommand::FlushAndShutdown) {
        error!("Could not send shutdown command to writer: {}", e);
    }

    info!(
        "Collector input channel closed. Closing gRPC stream. Collected {} samples and {} games in {:.2}s. Avg {:.2} samples/sec and {:.2} games/sec",
        total_samples,
        total_games,
        elapsed.as_secs_f64(),
        avg_samples_per_sec,
        avg_games_per_sec,
    );

    drop(writer_cmd_tx);

    // Process any final file paths that the writer flushes upon shutdown.
    while let Some(path) = flushed_path_rx.recv().await {
        info!(
            "Processing final flushed file from writer: {}",
            path.display()
        );
        if let Err(e) = grpc_tx.try_send(SendTrainingDataRequest {
            update: Some(TrainingSamplesUpdate {
                parquet_file_path: path.to_string_lossy().to_string(),
            }),
        }) {
            error!("Failed to send final flushed path to gRPC: {}", e);
        }
    }

    // Drop the gRPC sender and wait for the stream to close.
    drop(grpc_tx);

    let step = loop_count as u64;

    client
        .tensorboard_histogram(Request::new(training::TensorboardHistogramRequest {
            tag: "stats/game_length_to_mcts_policy_entropy".to_string(),
            values: hashmap_to_histogram(
                &move_count_to_entropy
                    .iter()
                    .map(|(k, v)| (*k, v.get_average()))
                    .collect(),
            ),
            step,
        }))
        .await
        .expect("Failed to send Tensorboard scalar");

    client
        .tensorboard_histogram(Request::new(training::TensorboardHistogramRequest {
            tag: "stats/game_lengths".to_string(),
            values: hashmap_to_histogram(
                &game_lengths.iter().map(|(k, v)| (*k, *v as f32)).collect(),
            ),
            step,
        }))
        .await
        .expect("Failed to send Tensorboard histogram");

    client
        .tensorboard_scalar(Request::new(training::TensorboardScalarRequest {
            tag: "stats/total_entropy".to_string(),
            value: total_entropy.get_average(),
            step,
        }))
        .await
        .expect("Failed to send Tensorboard scalar");

    client
        .tensorboard_multiple_scalars(Request::new(training::TensorboardMultipleScalarsRequest {
            tag: "stats/game_outcomes".to_string(),
            values: winning_ratios
                .iter()
                .map(|(k, v)| (k.to_string(), v.get_average()))
                .collect(),
            step,
        }))
        .await
        .expect("Failed to send Tensorboard scalars");

    client
        .tensorboard_scalar(Request::new(training::TensorboardScalarRequest {
            tag: "stats/total_games_played".to_string(),
            value: (global_stats.total_games_played + total_games as u64) as f32,
            step,
        }))
        .await
        .expect("Failed to send total games played scalar");

    client
        .tensorboard_scalar(Request::new(training::TensorboardScalarRequest {
            tag: "stats/total_samples_generated".to_string(),
            value: (global_stats.total_samples_generated + total_samples as u64) as f32,
            step,
        }))
        .await
        .expect("Failed to send total samples generated scalar");

    match stream_handle.await {
        // Unrecoverable errors or panics in the gRPC task.
        Err(e) => {
            error!("gRPC task panicked or was aborted: {}", e);
            return Err(Box::new(e));
        }

        Ok(result) => match result {
            Ok(response) => {
                info!(
                    "gRPC stream finished gracefully. Server response: {:?}",
                    response.into_inner()
                );
            }

            Err(status) => {
                if status.code() == tonic::Code::Unknown
                    && status.message().contains("transport error")
                {
                    info!(
                        "gRPC stream closed with an expected transport error during shutdown. This is likely okay."
                    );
                } else {
                    error!("gRPC task failed with an unexpected status: {}", status);
                    return Err(Box::new(status));
                }
            }
        },
    }

    let final_writer = writer_handle.await.expect("Writer thread panicked!");

    // Trim the oldest samples if we exceeded max_samples
    final_writer
        .trim_by_oldest(max_samples)
        .expect("Failed to trim BufferedDataWriter by oldest samples");

    Ok(CollectorStats {
        total_games_played: total_games as u64,
        total_samples_generated: total_samples as u64,
    })
}
