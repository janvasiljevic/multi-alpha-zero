use itertools::Itertools;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub enum ConfigGame {
    Hex,
    Chess,
    Chess2Player,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(tag = "type")]
pub enum PlayerConfig {
    AlphaZeroPlayer {
        onnx_path: String,
    },
    Random {
        unique_name: String,
    },
    Mcts {
        unique_name: String,

        num_of_playouts: usize,
    },
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArenaConfig {
    /// Name of the configuration.
    pub config_unique_name: String,

    /// Output file, must end with .csv
    pub out_file: String,

    /// The game to be played in the arena.
    /// Specifics such as the mapper and board size will be inferred from models themselves.
    pub game: ConfigGame,

    /// List of model file paths to be used in the arena.
    pub players: Vec<PlayerConfig>,

    /// Number of games to be played per matchup between three models.
    pub games_per_matchup: usize,

    /// Number of rollouts per move for MCTS-based models.
    #[serde(default = "default_rollouts_per_move")]
    pub mcts_rollouts_per_move: usize,

    /// Initial Elo rating for models.
    #[serde(default = "default_initial_elo")]
    pub initial_elo: f64,
}

fn default_rollouts_per_move() -> usize {
    200
}

fn default_initial_elo() -> f64 {
    600.0
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ArenaConfigList {
    pub sessions_per_gpu: usize,

    /// List of arena configurations.
    pub configs: Vec<ArenaConfig>,
}

impl ArenaConfig {
    pub fn get_elo_uncertainty(&self) -> f64 {
        self.initial_elo / 3.0
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.out_file.ends_with(".csv") {
            return Err("Output file must have a .csv extension.".to_string());
        }

        let min_players = match self.game {
            ConfigGame::Hex | ConfigGame::Chess => 3,
            ConfigGame::Chess2Player => 2,
        };

        if self.players.len() < min_players {
            return Err(format!(
                "At least {} players are required for the selected game.",
                min_players
            ));
        }

        // Validate player configurations
        for player in &self.players {
            if let PlayerConfig::AlphaZeroPlayer { onnx_path: path } = player {
                if !std::path::Path::new(path).exists() {
                    return Err(format!("Model file does not exist: {}", path));
                }
            }
        }

        // Ensure unique player names
        let names: Vec<_> = self
            .players
            .iter()
            .map(|p| match p {
                PlayerConfig::AlphaZeroPlayer { onnx_path: path } => {
                    path.split('/').last().unwrap_or("").to_string()
                }
                PlayerConfig::Random { unique_name: name } => name.clone(),
                PlayerConfig::Mcts {
                    unique_name: name, ..
                } => name.clone(),
            })
            .collect();
        let unique_names: std::collections::HashSet<_> = names.iter().cloned().collect();
        if names.len() != unique_names.len() {
            return Err("Player names must be unique.".to_string());
        }

        Ok(())
    }

    pub fn lint(&self) {
        if self.mcts_rollouts_per_move <= 100 {
            warn!(
                "Current rollouts_per_move ({}) is quite low. Consider increasing it for more robust evaluations.",
                self.mcts_rollouts_per_move
            );
        }
    }

    pub fn heads_up(&self, num_of_board_players: usize) -> usize {
        let num_players = self.players.len();
        let total_matchups = (0..num_players).combinations(num_of_board_players).count();
        let permutations_per_matchup = (1..=num_of_board_players)
            .permutations(num_of_board_players)
            .count();
        let total_games = total_matchups * permutations_per_matchup * self.games_per_matchup;

        info!(
            "With {} board players and {} total players, there will be {} matchups, {} permutations per matchup, and {} games per matchup, resulting in a total of {} games.",
            num_of_board_players,
            num_players,
            total_matchups,
            permutations_per_matchup,
            self.games_per_matchup,
            total_games,
        );

        total_games
    }
}

impl ArenaConfigList {
    /// Loads configuration from a YAML file.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_yaml::from_reader(file)?;

        for cfg in &config.configs {
            cfg.validate()?;

            let player_num = match cfg.game {
                ConfigGame::Hex | ConfigGame::Chess => 3,
                ConfigGame::Chess2Player => 2,
            };

            cfg.heads_up(player_num);
        }

        info!("Configuration from '{}' loaded:\n{:#?}", &path, config);

        Ok(config)
    }

    pub fn save_to_file(&self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        serde_yaml::to_writer(file, self)?;
        Ok(())
    }
}
