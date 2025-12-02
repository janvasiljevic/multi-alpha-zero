mod cli;
mod components;
mod dynamic_self_play_game;
mod global_stats;
mod gpu;
mod inference_protocol;
mod learning_target_modifier;
mod move_selector;
mod node;
mod range;
mod search_settings;
mod self_play_game;
mod sharded_sender;
mod shared_cache;
mod stable_dirichlet;
mod steppable_mcts;
mod trainer;
mod tree;

use clap::Parser;
use crossbeam_channel::bounded;
use game_hex::game_hex::HexGame;
use log::warn;
use maz_core::mapping::hex_absolute_mapper::HexAbsoluteMapper;
use maz_core::mapping::{Board, MetaBoardMapper, OptionalSharedOracle};
use quick_cache::sync::Cache;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::cli::Cli;
use crate::components::collector::run_collector_task;
use crate::components::generator::{generator_worker_task, GeneratorStats};
use crate::components::inference::spawn_gpu_worker_threads;
use crate::global_stats::{CsvLogger, Record};
use crate::gpu::gpu_provider::{automatic_gpu_spawner, DummyArgs, GenericModelArgs, GpuModelArgs};
use crate::inference_protocol::{InferenceRequest, InferenceResult};
use crate::search_settings::SearchSettings;
use crate::self_play_game::Simulation;
use crate::sharded_sender::new_sharded_sender;
use crate::shared_cache::{shared_cache_actor, CacheWriterMessage, SharedCache};
use crate::trainer::protocol::{send_initial_settings, train_model};
use dynamic_self_play_game::DynamicSelfPlayGame;
use game_tri_chess::chess_game::TriHexChess;
use maz_config::config::{AppConfig, GameConfig, GameNames, GpuSpawnerChoice};
use maz_core::mapping::chess_canonical_mapper::ChessCanonicalMapper;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_util::logging::setup_logging;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::info;
use trainer::process_manager::launch_python_server;

use crate::components::standalone_collector::run_standalone_collector_task;
use crate::trainer::process_manager::ManagedChildProcess;
use crate::trainer::protocol::training::InitialSettingsResponse;
use maz_core::mapping::aux_values::set_num_of_aux_features;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::chess_extended_canonical_mapper::ChessExtendedCanonicalMapper;
use maz_core::mapping::chess_hybrid_canonical_mapper::ChessHybridCanonicalMapper;
use oracle_tri_chess::oracle::TriHexEndgameOracle;
use tikv_jemallocator::Jemalloc;
use crate::learning_target_modifier::{LearningModifier, MaterialAdvantageModifier};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn create_run_folder(config: &AppConfig) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let project_root = manifest_dir
        .parent()
        .ok_or("Failed to get parent directory of CARGO_MANIFEST_DIR")?;

    let run_folder = project_root.join(&config.base_dir);

    if !run_folder.exists() || !run_folder.is_dir() {
        return Err(format!("Run folder does not exist: {run_folder:?}").into());
    }

    let run_subdir = run_folder.join(&config.run_name);

    if config.overwrite_run {
        if run_subdir.exists() {
            info!("Removing existing run directory: {:?}", run_subdir);
            std::fs::remove_dir_all(&run_subdir)?;
        }
        std::fs::create_dir_all(&run_subdir)?;

        Ok(run_subdir)
    } else if !run_subdir.exists() {
        std::fs::create_dir_all(&run_subdir)?;

        Ok(run_subdir)
    } else {
        Ok(run_subdir)
    }
}

pub struct BoardWithMapper<B: Board, M: MetaBoardMapper<B>> {
    pub board: B,
    pub mapper: M,
}

pub enum SelfPlayState {
    Hex5Absolute(BoardWithMapper<HexGame, HexAbsoluteMapper>),
    Hex5Canonical(BoardWithMapper<HexGame, HexCanonicalMapper>),
    Hex4Absolute(BoardWithMapper<HexGame, HexAbsoluteMapper>),
    Hex4Canonical(BoardWithMapper<HexGame, HexCanonicalMapper>),
    Hex3Absolute(BoardWithMapper<HexGame, HexAbsoluteMapper>),
    Hex3Canonical(BoardWithMapper<HexGame, HexCanonicalMapper>),
    Chess(BoardWithMapper<TriHexChess, ChessCanonicalMapper>),
    ChessAdvanced(BoardWithMapper<TriHexChess, ChessExtendedCanonicalMapper>),
    ChessHybrid(BoardWithMapper<TriHexChess, ChessHybridCanonicalMapper>),
    ChessDomain(BoardWithMapper<TriHexChess, ChessDomainMapper>),
}

fn get_game_with_mapper(game_config: &GameConfig) -> SelfPlayState {
    let name = game_config.name;

    match name {
        GameNames::Hex5Absolute
        | GameNames::Hex5Canonical
        | GameNames::Hex5CanonicalAxiomBias
        | GameNames::Hex5CanonicalAxiomBiasBertCls => {
            let board = HexGame::new(5).unwrap();
            if matches!(name, GameNames::Hex5Absolute) {
                let mapper = HexAbsoluteMapper::new(&board);
                SelfPlayState::Hex5Absolute(BoardWithMapper { mapper, board })
            } else {
                let mapper = HexCanonicalMapper::new(&board);
                SelfPlayState::Hex5Canonical(BoardWithMapper { mapper, board })
            }
        }
        GameNames::Hex4Absolute | GameNames::Hex4Canonical => {
            let board = HexGame::new(4).unwrap();
            if matches!(name, GameNames::Hex4Absolute) {
                let mapper = HexAbsoluteMapper::new(&board);
                SelfPlayState::Hex4Absolute(BoardWithMapper { mapper, board })
            } else {
                let mapper = HexCanonicalMapper::new(&board);
                SelfPlayState::Hex4Canonical(BoardWithMapper { mapper, board })
            }
        }
        // This is meant purely for testing and debugging with very fast games.
        GameNames::Hex2Absolute | GameNames::Hex2Canonical => {
            let board = HexGame::new(2).unwrap();
            if matches!(name, GameNames::Hex2Absolute) {
                let mapper = HexAbsoluteMapper::new(&board);
                SelfPlayState::Hex3Absolute(BoardWithMapper { mapper, board })
            } else {
                let mapper = HexCanonicalMapper::new(&board);
                SelfPlayState::Hex3Canonical(BoardWithMapper { mapper, board })
            }
        }
        GameNames::ChessAxiomBias | GameNames::ChessAbsPositionalEnc => {
            let board = TriHexChess::default_with_grace_period();
            let mapper = ChessCanonicalMapper;
            SelfPlayState::Chess(BoardWithMapper { mapper, board })
        }
        GameNames::ChessBigBert | GameNames::ChessBigBertV2 | GameNames::ChessShaw => {
            let board = TriHexChess::default_with_grace_period();
            let mapper = ChessExtendedCanonicalMapper;
            SelfPlayState::ChessAdvanced(BoardWithMapper { mapper, board })
        }
        GameNames::ChessHybrid => {
            let board = TriHexChess::default_with_grace_period();
            let mapper = ChessHybridCanonicalMapper;
            SelfPlayState::ChessHybrid(BoardWithMapper { mapper, board })
        }
        GameNames::ChessDomain => {
            let board = TriHexChess::default_with_grace_period();
            let mapper = ChessDomainMapper;
            SelfPlayState::ChessDomain(BoardWithMapper { mapper, board })
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(true);

    let config = AppConfig::load(&cli.config_path).unwrap_or_else(|e| {
        panic!(
            "Failed to load configuration file from '{}': {}",
            &cli.config_path, e
        )
    });

    let base_dir = create_run_folder(&config).unwrap_or_else(|e| {
        panic!("Failed to create run folder: {e}");
    });

    let resaved_config_for_reference = base_dir.join("saved_config.yaml");

    config
        .save_to_file(resaved_config_for_reference)
        .unwrap_or_else(|e| {
            panic!("Failed to save configuration file: {e}");
        });

    let any_game = get_game_with_mapper(&config.game);

    set_num_of_aux_features(config.training.number_of_aux_features);

    let oracle_singleton: OptionalSharedOracle<TriHexChess> = match any_game {
        // For all chess variants, we load the oracle.
        SelfPlayState::Chess(_)
        | SelfPlayState::ChessAdvanced(_)
        | SelfPlayState::ChessHybrid(_)
        | SelfPlayState::ChessDomain(_) => {
            info!("Chess game selected, initializing endgame oracle...");
            match TriHexEndgameOracle::new(
                "./tablebases/kqk_tablebase.bin",
                "./tablebases/krk_tablebase.bin",
            ) {
                Ok(oracle) => Some(Arc::new(oracle)),
                Err(e) => {
                    // This is a fatal error if we expect an oracle for chess.
                    panic!("Failed to initialize TriHexEndgameOracle: {:?}", e);
                }
            }
        }
        // For all Hex variants, we do not create an oracle.
        _ => {
            info!("Hex game selected, endgame oracle is disabled.");
            None
        }
    };

    match any_game {
        SelfPlayState::Hex5Absolute(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Hex5Canonical(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Hex4Absolute(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Hex4Canonical(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Hex3Absolute(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Hex3Canonical(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, None).await;
        }
        SelfPlayState::Chess(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, oracle_singleton).await;
        }
        SelfPlayState::ChessAdvanced(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, oracle_singleton).await;
        }
        SelfPlayState::ChessHybrid(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, oracle_singleton).await;
        }
        SelfPlayState::ChessDomain(board_with_mapper) => {
            run_app_logic(board_with_mapper, config, &base_dir, oracle_singleton).await;
        }
    }
}

async fn run_app_logic<B, M>(
    board_with_mapper: BoardWithMapper<B, M>,
    config: AppConfig,
    base_dir: &std::path::Path,
    oracle: OptionalSharedOracle<B>,
) where
    B: Board + Send + 'static,
    M: MetaBoardMapper<B> + Send + 'static,
    <B as Board>::Move: Send,
{
    let mut csv_logger = CsvLogger::new(base_dir.join("training_log.csv")).unwrap_or_else(|e| {
        panic!("Failed to initialize CSV logger: {e}");
    });

    let server_addr: &str = "http://[::1]:50051";

    let (initial_settings_res, _handler) = if !config.standalone {
        let _python_server_handle =
            match launch_python_server(server_addr, config.python_interpreter_path.clone()).await {
                Ok(handle) => {
                    info!("Python server launched successfully.");
                    handle
                }
                Err(e) => {
                    tracing::error!("FATAL: Failed to start Python server: {}", e);
                    panic!()
                }
            };

        let current_training_step = csv_logger.last_training_steps();

        if current_training_step > 0 {
            warn!(
                "Resuming from previous training step: {}",
                current_training_step
            );
        }

        let initial_settings_res = send_initial_settings(
            server_addr,
            &config,
            base_dir.to_string_lossy().to_string(),
            current_training_step,
        )
        .await
        .expect("Failed to send initial settings to the training coordinator");

        let model_path = base_dir.join("onnx_current.onnx");

        // copy the initial model to onnx_current
        std::fs::copy(&initial_settings_res.model_path, &model_path).unwrap_or_else(|e| {
            panic!(
                "Failed to copy initial model file from {} to {model_path:?}: {e}",
                initial_settings_res.model_path
            );
        });

        (Some(initial_settings_res), Some(_python_server_handle))
    } else {
        warn!(
            "Running in standalone mode. No model training will be performed and no connection to a training server will be made."
        );

        // find the latest model in the base_dir if it exists in /onnx/ folder
        // get all files in the onnx folder
        let onnx_dir = base_dir.join("onnx");

        if onnx_dir.exists() && onnx_dir.is_dir() {
            let mut onnx_files: Vec<_> = std::fs::read_dir(&onnx_dir)
                .unwrap()
                .filter_map(|entry| {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect();

            // files are all of form 0.onnx, 1.onnx, ..., sort them by number
            onnx_files.sort_by_key(|path| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0)
            });

            info!("Found {} .onnx files in {:?}", onnx_files.len(), onnx_dir);

            let tuple = if let Some(latest_model) = onnx_files.pop() {
                info!("Found latest model in standalone mode: {:?}", latest_model);

                (
                    Some(InitialSettingsResponse {
                        model_path: latest_model.to_string_lossy().to_string(),
                    }),
                    None::<ManagedChildProcess>,
                )
            } else {
                (None, None)
            };

            tuple
        } else {
            (None, None)
        }
    };

    let BoardWithMapper { board, mapper } = board_with_mapper;
    let board = board.new_varied();
    let search_settings = SearchSettings::from(config.self_play_settings);

    let gpu_spawner = automatic_gpu_spawner(
        match config.performance.gpu {
            GpuSpawnerChoice::Automatic(gpu_config) => GpuModelArgs::Generic(GenericModelArgs {
                model_path: initial_settings_res
                    .expect("Initial settings must be provided")
                    .model_path,
                sessions_per_gpu: gpu_config.sessions_per_gpu,
                batch_size: config.performance.gpu_batch_size as usize,
                run_folder: base_dir.to_path_buf(),
            }),
            GpuSpawnerChoice::Dummy(args) => GpuModelArgs::Dummy(DummyArgs::new(
                args.inferences_per_second,
                args.num_of_gpus,
                config.performance.gpu_batch_size as usize,
            )),
        },
        &board,
        &mapper,
    )
    .unwrap();

    let num_of_generators = config.performance.cpu_generator_threads;
    let games_per_generator = config.performance.games_per_cpu_generator;
    let gpu_batch_size = config.performance.gpu_batch_size as usize;

    let total_games = num_of_generators * games_per_generator;
    let total_gpu_ingress =
        gpu_spawner.get_gpu_count() * gpu_spawner.sessions_per_gpu() * gpu_batch_size;

    let safety_ingress_factor = 1.6;
    let actual_factor = total_games as f64 / total_gpu_ingress as f64;

    if actual_factor < safety_ingress_factor {
        tracing::warn!(
            "{total_games} are being played at the same time.\
             GPUs will ingress {total_gpu_ingress} positions per 'tick'.\
             To be sure to keep the GPU busy, it's recommend to have at least {:.1} times more GPU ingress capacity than total games.\
             Current ratio is {:.2}.\
             Consider increasing the total amount of concurrent games to at least {}.",
            safety_ingress_factor,
            actual_factor,
            (safety_ingress_factor * total_gpu_ingress as f64).ceil() as usize
        );
    } else {
        info!(
            "Total games being generated: {}. GPU ingress capacity: {}. Safety factor: {:.2} (>= {}).",
            total_games, total_gpu_ingress, actual_factor, safety_ingress_factor
        );
    }

    if gpu_spawner.model_path().is_some() {
        let [board_size, field_size] = mapper.input_board_shape();

        gpu_spawner
            .verify_dimensions(
                board_size,
                field_size,
                mapper.policy_len(),
                board.player_num(),
            )
            .expect("Failed to verify model dimensions");
    }

    let num_of_gpus = gpu_spawner.get_gpu_count();

    let mut loop_count = csv_logger.last_cycle_number().map_or_else(
        || {
            info!("Starting training from cycle 0");
            0
        },
        |last_cycle| {
            info!("Resuming training from cycle {}", last_cycle + 1);
            last_cycle + 1
        },
    );

    let mut collector_stats_global = csv_logger.generate_collector_stats();

    let mut previous_game_pools_backup: Vec<Option<Vec<DynamicSelfPlayGame<B>>>> =
        vec![None; num_of_generators];

    let cache_size = config.performance.cache_size as u64;
    let shared_cache: SharedCache = Arc::new(Cache::new(cache_size as usize));

    let cache_actors = config.performance.cache_writers;

    loop {
        if let Some(stop_count) = config.number_of_cycles {
            if loop_count >= stop_count {
                info!("Reached the configured number of cycles: {}", stop_count);
                break;
            }
        }

        // Some providers (TensorRT) cache the engine build.
        gpu_spawner.warmup();

        let start_time = Instant::now();

        let stop_signal = Arc::new(AtomicBool::new(false));

        // Vector of channels for the GPU to send replies back to specific CPUs.
        // Position in the vector corresponds to the CPU thread ID!
        let mut txs = Vec::with_capacity(num_of_generators);
        let mut rxs = Vec::with_capacity(num_of_generators);

        for _ in 0..num_of_generators {
            let (tx, rx) = crossbeam_channel::unbounded::<InferenceResult>();
            txs.push(tx);
            rxs.push(rx);
        }

        let (samples_tx, samples_rx) = mpsc::unbounded_channel::<Simulation<B>>();

        let stop_signal_thread = stop_signal.clone();
        let cycle_settings = config.cycle_collection.clone();
        let base_dir_collector = base_dir.to_path_buf();
        let collector_stats = collector_stats_global.clone();
        let board_mapper = mapper.clone();

        let collector_handle = if config.standalone {
            tokio::spawn(async move {
                match run_standalone_collector_task()
                    .board_mapper(board_mapper)
                    .samples_rx(samples_rx)
                    .base_dir(base_dir_collector)
                    .max_samples(config.training.samples)
                    .loop_count(loop_count as usize)
                    .cycle_settings(cycle_settings)
                    .stop_signal(stop_signal_thread)
                    .parquest_chunk_size(config.performance.parquest_chunk_size)
                    .call()
                    .await
                {
                    Ok(global_stats) => global_stats,
                    Err(e) => {
                        tracing::error!("Collector task failed: {}", e);
                        panic!("Collector task failed: {}", e);
                    }
                }
            })
        } else {
            tokio::spawn(async move {
                match run_collector_task()
                    .board_mapper(board_mapper)
                    .samples_rx(samples_rx)
                    .server_addr(server_addr)
                    .base_dir(base_dir_collector)
                    .max_samples(config.training.samples)
                    .loop_count(loop_count as usize)
                    .cycle_settings(cycle_settings)
                    .stop_signal(stop_signal_thread)
                    .global_stats(collector_stats)
                    .parquest_chunk_size(config.performance.parquest_chunk_size)
                    .call()
                    .await
                {
                    Ok(global_stats) => global_stats,
                    Err(e) => {
                        tracing::error!("Collector task failed: {}", e);
                        panic!("Collector task failed: {}", e);
                    }
                }
            })
        };

        // We have multiple channels for requests, one for each GPU.
        // CPU threads cycle the request channels to send requests to the GPUs (round-robin).
        // This is called sharded MPSC (Multiple Producer Single Consumer) pattern.
        let mut generator_to_gpu_tx = Vec::new();
        let mut generator_to_batcher_rx = Vec::new();

        for _ in 0..num_of_gpus * gpu_spawner.sessions_per_gpu() {
            let (tx, rx) = bounded::<InferenceRequest>(total_games);
            generator_to_gpu_tx.push(tx);
            generator_to_batcher_rx.push(rx);
        }

        let mut receiver_iterator = generator_to_batcher_rx.into_iter();

        // Create generators
        let mut cpu_threads = vec![];
        let mut cache_writer_rxs = Vec::with_capacity(cache_actors);
        let mut cache_writer_txs = Vec::with_capacity(cache_actors);
        for _actor_id in 0..cache_actors {
            let (tx, rx) = bounded::<CacheWriterMessage>(total_games * 2);
            cache_writer_txs.push(tx);
            cache_writer_rxs.push(rx);
        }

        let mut cache_handles = Vec::with_capacity(cache_actors);

        for actor_id in 0..cache_actors {
            let cache_for_actor = shared_cache.clone();
            let rx = cache_writer_rxs.remove(0);
            cache_handles.push(
                thread::Builder::new()
                    .name(format!("cache-actor-{actor_id}"))
                    .spawn(move || {
                        shared_cache_actor(cache_for_actor, rx);
                        info!("Cache actor {actor_id} thread - exiting gracefully.");
                    })
                    .unwrap(),
            )
        }

        info!(
            "Estimated total memory usage for cache: {} MB",
            shared_cache::estimate_total_memory_usage_mb(cache_size, &board, &mapper)
        );

        let material_advantage = MaterialAdvantageModifier::new(9, 0.35, 0.3, 0.35);

        // TODO: Make this configurable
        let learning_rate_modifier = LearningModifier::MaterialAdvantage(material_advantage);

        for i in 0..num_of_generators {
            let mapper_thread = mapper.clone();

            // Each CPU gets access to all GPUs, but must load balance requests across them.
            let gpu_sharded_sender = new_sharded_sender(generator_to_gpu_tx.clone());
            let cache_sharded_sender = new_sharded_sender(cache_writer_txs.clone());

            let rx_thread = rxs.remove(0);
            let samples_tx_thread = samples_tx.clone();

            let stop_signal_thread = stop_signal.clone();
            let board = board.clone();
            let cache_clone = shared_cache.clone();
            let oracle_for_thread = oracle.clone();

            let previous_game_pool_thread = previous_game_pools_backup[i].take();
            let learning_modifier = learning_rate_modifier.clone();

            let handle = thread::Builder::new()
                .name(format!("CPU_{i}"))
                .spawn(move || {
                    (
                        i,
                        generator_worker_task(
                            i,
                            board,
                            games_per_generator,
                            mapper_thread,
                            search_settings,
                            gpu_sharded_sender,
                            cache_sharded_sender,
                            rx_thread,
                            samples_tx_thread,
                            stop_signal_thread,
                            cache_clone,
                            previous_game_pool_thread,
                            oracle_for_thread,
                            learning_modifier,
                        ),
                    )
                })
                .unwrap();
            cpu_threads.push(handle);
        }

        let gpu_init_locks: Vec<Arc<Mutex<()>>> =
            (0..num_of_gpus).map(|_| Arc::new(Mutex::new(()))).collect();

        let gpu_joined_handles = (0..num_of_gpus)
            .map(|gpu_index| {
                let mapper_thread = mapper.clone();

                let gpu_requests_receivers = (&mut receiver_iterator)
                    .take(gpu_spawner.sessions_per_gpu())
                    .collect::<Vec<_>>();

                // Since each GPU can receive requests from all CPU threads,
                // it needs to have a vector of all reply channels.
                let txs_thread = txs.clone();

                spawn_gpu_worker_threads(
                    mapper_thread,
                    config.performance.gpu_batch_size,
                    gpu_index as i32,
                    gpu_spawner.clone(),
                    txs_thread,
                    gpu_requests_receivers,
                    stop_signal.clone(),
                    &gpu_init_locks, // Pass the shared locks here
                )
            })
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        drop(cache_writer_txs);

        // Drop the transmitters so the only remaining references are in the threads.
        drop(samples_tx);
        drop(generator_to_gpu_tx);

        for tx in txs {
            drop(tx);
        }

        let mut generator_stats = GeneratorStats::default();

        // Joins all CPU threads and collects their game pools for the next iteration.
        for handle in cpu_threads {
            let (cpu_index, (stats, game_pool)) = handle.join().expect("CPU thread panicked");
            generator_stats += stats;

            previous_game_pools_backup[cpu_index] = Some(game_pool);
        }

        let mut total_positions_saved = 0;

        for game_pool in &previous_game_pools_backup {
            if let Some(games) = game_pool {
                for game in games {
                    total_positions_saved += game.get_position_count();
                }
            }
        }

        info!(
            "CPU Generators off. Total positions saved in backup: {}",
            total_positions_saved
        );

        for cache_handle in cache_handles {
            cache_handle.join().expect("Cache actor thread panicked");
        }

        info!("Cache actor threads - completed successfully.");

        let mut gpu_stats = vec![];

        for joined_handle in gpu_joined_handles {
            gpu_stats.push(joined_handle.join());
        }

        info!("GPU Worker tasks - completed successfully.");

        let collector_stats_cycle = collector_handle.await.expect("Collector task failed");

        collector_stats_global += collector_stats_cycle.clone();

        info!("CPU Generator/ GPU Worker(s)/ Collector tasks - completed successfully.");

        let self_play_time = start_time.elapsed().as_secs_f64();

        let training_future = async {
            if !config.standalone {
                let res = train_model(server_addr, &config.training)
                    .await
                    .expect("Failed to train model");

                // Get the model path, and copy the file to onnx_new (sibling of onnx_current)
                let new_file = res.new_model_path.clone();
                let current_file = base_dir.join("onnx_current.onnx");

                std::fs::copy(&new_file, &current_file).unwrap_or_else(|e| {
                    panic!(
                        "Failed to copy new model file from {new_file} to {current_file:?}: {e}"
                    );
                });

                gpu_spawner.set_model_path(res.new_model_path.clone());

                Some(res)
            } else {
                None
            }
        };

        let cache_for_eviction = shared_cache.clone();

        let cache_eviction_future = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            cache_for_eviction.clear();
            start.elapsed()
        });

        let (training_result, cache_eviction_result) =
            tokio::join!(training_future, cache_eviction_future);

        let eviction_duration = cache_eviction_result.expect("Cache eviction task panicked");

        info!(
            "Cache eviction completed in {} seconds.",
            eviction_duration.as_secs_f64()
        );

        let res = training_result;

        let training_time = res.as_ref().map_or(0f64, |r| r.duration_seconds as f64);

        let total_time = self_play_time + training_time;

        let inferences_per_second = gpu_stats
            .iter()
            .fold(0f64, |acc, stat| acc + stat.inferences_per_second);

        let record = Record {
            cycle_number: loop_count,
            total_games_played: collector_stats_cycle.total_games_played,
            total_samples_generated: collector_stats_cycle.total_samples_generated,
            self_play_time_seconds: self_play_time,
            model_training_time_seconds: training_time,
            total_time_seconds: total_time,
            nodes_backed_up: total_positions_saved as u64,
            inferences_per_second,
            cache_hits: generator_stats.cache_hits,
            nodes_processed: generator_stats.nodes_processed,
            nodes_per_second: generator_stats.nodes_per_second,
            training_steps: res.as_ref().map_or(0, |r| r.total_training_steps),
        };

        info!("Record: {:#?}", record);

        if config.standalone {
            warn!("Exiting after one cycle in standalone mode.");
            break;
        }

        csv_logger.append(record).unwrap_or_else(|e| {
            panic!("Failed to append record to CSV log: {e}");
        });

        loop_count += 1;
    }
}
