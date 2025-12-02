use clap::Parser;
use maz_arena::arena_config::{ArenaConfigList, ConfigGame, PlayerConfig};
use maz_arena::chess_arena::run_arena_chess;
use maz_arena::chess_arena_2_player::run_arena_chess_2_player;
use maz_arena::hex_arena::run_arena_hex;
use maz_util::logging::setup_logging;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use tracing::{error, info};
use maz_trainer::gpu::cuda_provider::detect_cuda;

/// Arena for evaluating and ranking multiple models by playing games against each other.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the configuration file
    #[arg(default_value = "maz-arena/arena-config.yaml")]
    pub(crate) config_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    setup_logging(true);

    let config_list = ArenaConfigList::load(&cli.config_path)?;

    info!("Starting pre-flight checks for all configurations...");
    for config in &config_list.configs {
        for player_config in &config.players {
            if let PlayerConfig::AlphaZeroPlayer { onnx_path: path } = player_config {
                if !std::path::Path::new(path).exists() {
                    return Err(format!("Model file does not exist: {}", path).into());
                }
                // // This warm-up is useful for some backends like TensorRT.
                // // We'll do it once on the primary GPU (device 0).
                // info!("Pre-loading and validating model: {}", path);
                // let session = auto_non_cpu_model(path, Some(0))?;
                // drop(session);
            }
        }
    }
    info!("Pre-flight checks passed successfully.");

    let total_matches: usize = config_list
        .configs
        .iter()
        .map(|c| {
            let num_of_board_players = match c.game {
                ConfigGame::Hex | ConfigGame::Chess => 3,
                ConfigGame::Chess2Player => 2,
            };

            c.heads_up(num_of_board_players)
        })
        .sum();

    info!(
        "All model paths exist. Will run a total of {} matches across all configurations.",
        total_matches
    );

    let gpu_count = detect_cuda().unwrap_or(1);
    info!(
        "Detected {} GPUs. Creating a thread pool of that size.",
        gpu_count
    );

    let pool = ThreadPoolBuilder::new()
        .num_threads(gpu_count)
        .build()
        .expect("Failed to create Rayon thread pool.");

    let session_per_gpu = config_list.sessions_per_gpu;

    pool.install(|| {
        config_list.configs.par_iter().for_each(|config| {
            let gpu_id = rayon::current_thread_index()
                .expect("Should be running inside a Rayon thread pool.");

            info!(
                "Assigning config '{}' to worker on GPU {}",
                config.config_unique_name, gpu_id
            );

            match &config.game {
                ConfigGame::Hex => {
                    if let Err(e) = run_arena_hex(config, Some(gpu_id), session_per_gpu) {
                        error!(
                            "Error running hex arena for '{}' on GPU {} (skipping): {}",
                            config.config_unique_name, gpu_id, e
                        );
                    }
                }
                ConfigGame::Chess => {
                    if let Err(e) = run_arena_chess(config, Some(gpu_id), session_per_gpu) {
                        error!(
                            "Error running chess arena for '{}' on GPU {} (skipping): {}",
                            config.config_unique_name, gpu_id, e
                        );
                    }
                }
                &ConfigGame::Chess2Player => {
                    if let Err(e) = run_arena_chess_2_player(config, Some(gpu_id), session_per_gpu)
                    {
                        error!(
                            "Error running 2-player chess arena for '{}' on GPU {} (skipping): {}",
                            config.config_unique_name, gpu_id, e
                        );
                    }
                }
            }
        });
    });

    Ok(())
}
