use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use strum_macros::AsRefStr;
use tracing::info;

/// Configuration for the real ONNX-based GPU spawner.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AutomaticGpuConfig {
    /// How many sessions to spawn per GPU. May help with latency by interleaving requests.
    pub sessions_per_gpu: usize,
}

/// Configuration for the dummy spawner, used for testing CPU-bound performance.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DummyGpuConfig {
    /// The simulated number of inferences per second.
    pub inferences_per_second: f64,

    /// The number of fake GPU instances to spawn.
    pub num_of_gpus: usize,
}

/// Enum to select the type of GPU spawner to use.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type")] // This is the key! It tells serde to use a field named "type" to decide which enum variant to use.
#[serde(rename_all = "PascalCase")] // Makes YAML use "Automatic" and "Dummy"
pub enum GpuSpawnerChoice {
    Automatic(AutomaticGpuConfig),
    Dummy(DummyGpuConfig),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct UctSettings {
    #[serde(default = "default_exploration_weight")]
    pub exploration_weight: f32,

    #[serde(default = "default_moves_left_weight")]
    pub moves_left_weight: f32,

    #[serde(default = "default_moves_left_clip")]
    pub moves_left_clip: f32,

    #[serde(default = "default_moves_left_sharpness")]
    pub moves_left_sharpness: f32,
}

fn default_exploration_weight() -> f32 {
    4.0
}
fn default_moves_left_weight() -> f32 {
    0.03
}
fn default_moves_left_clip() -> f32 {
    20.0
}
fn default_moves_left_sharpness() -> f32 {
    0.5
}

impl Default for UctSettings {
    fn default() -> Self {
        UctSettings {
            exploration_weight: default_exploration_weight(),
            moves_left_weight: default_moves_left_weight(),
            moves_left_clip: default_moves_left_clip(),
            moves_left_sharpness: default_moves_left_sharpness(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoveSelectionSettings {
    #[serde(default = "default_move_selection_temp")]
    pub temperature: f32,

    /// Number of moves after which the temperature is set to zero.
    #[serde(default = "default_zero_temp_move_count")]
    pub zero_temp_move_count: u32,
}

fn default_move_selection_temp() -> f32 {
    1.0
}

fn default_zero_temp_move_count() -> u32 {
    40
}

impl Default for MoveSelectionSettings {
    fn default() -> Self {
        MoveSelectionSettings {
            temperature: default_move_selection_temp(),
            zero_temp_move_count: default_zero_temp_move_count(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirichletSettings {
    /// Dirichlet alpha parameter, controls the noise level in the policy.
    pub alpha: f32,

    /// Dirichlet epsilon parameter, controls the amount of noise added to the policy.
    #[serde(default = "default_dirichlet_eps")]
    pub eps: f32,
}

fn default_dirichlet_eps() -> f32 {
    0.25
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RolloutSettings {
    /// Probability of performing a full search instead of a partial one.
    #[serde(default = "default_full_search_prob")]
    pub full_search_prob: f64,

    /// Number of full iterations to perform when a full search is chosen.
    #[serde(default = "default_full_iterations")]
    pub full_iterations: u64,

    /// Number of partial iterations to perform when a partial search is chosen.
    #[serde(default = "default_part_iterations")]
    pub part_iterations: u64,

    /// When doing partial rollouts, how much to explore. Keep it low, we want greedy moves.
    #[serde(default = "default_part_cpuct_exploration")]
    pub part_cpuct_exploration: f32,
}

fn default_full_search_prob() -> f64 {
    1.0
}

fn default_full_iterations() -> u64 {
    200
}

fn default_part_iterations() -> u64 {
    20
}

fn default_part_cpuct_exploration() -> f32 {
    1.0
}

impl Default for RolloutSettings {
    fn default() -> Self {
        RolloutSettings {
            full_search_prob: default_full_search_prob(),
            full_iterations: default_full_iterations(),
            part_iterations: default_part_iterations(),
            part_cpuct_exploration: default_part_cpuct_exploration(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelativeFpu {
    pub relative: f32,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FixedFpu {
    pub fixed: f32,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "PascalCase")]
pub enum ConfigFpuMode {
    Relative(RelativeFpu),
    Fixed(FixedFpu),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelfPlaySettings {
    /// Maximum length of a game in moves. If None, the games will always run to the natural end.
    /// Beware that for some games (e.g., Chess) this parameter is essential to avoid infinite games.
    #[serde(default = "default_max_game_length")]
    pub max_game_length: Option<u64>,

    #[serde(default)]
    pub uct: UctSettings,

    #[serde(default)]
    pub move_selection: MoveSelectionSettings,

    pub dirichlet: DirichletSettings,

    #[serde(default = "default_search_policy_temperature_root")]
    pub search_policy_temperature_root: f32,

    #[serde(default = "default_search_policy_temperature_child")]
    pub search_policy_temperature_child: f32,

    #[serde(default = "default_search_fpu_root")]
    pub search_fpu_root: ConfigFpuMode,

    #[serde(default = "default_search_fpu_child")]
    pub search_fpu_child: ConfigFpuMode,

    #[serde(default)]
    pub rollouts: RolloutSettings,

    pub contempt: f32,
}

fn default_max_game_length() -> Option<u64> {
    None
}

fn default_search_fpu_root() -> ConfigFpuMode {
    ConfigFpuMode::Fixed(FixedFpu { fixed: 0.0 })
}

fn default_search_fpu_child() -> ConfigFpuMode {
    ConfigFpuMode::Relative(RelativeFpu { relative: 0.2 })
}

fn default_search_policy_temperature_root() -> f32 {
    1.4
}

fn default_search_policy_temperature_child() -> f32 {
    1.0
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct CycleCollectionSettings {
    /// Minimum number of games to **collect** in a training cycle.
    #[serde(default = "default_min_games_per_cycle")]
    pub min_games_per_cycle: Option<u64>,

    /// Minimum number of samples to **collect** in a training cycle.
    #[serde(default = "default_min_samples_per_cycle")]
    pub min_samples_per_cycle: Option<u64>,

    /// Cut-off time for a training cycle in seconds.
    #[serde(default = "default_max_time_per_cycle")]
    pub max_time_per_cycle_s: Option<u64>,
}

fn default_min_games_per_cycle() -> Option<u64> {
    None
}

fn default_min_samples_per_cycle() -> Option<u64> {
    None
}

fn default_max_time_per_cycle() -> Option<u64> {
    Some(5 * 60)
}

impl Default for CycleCollectionSettings {
    fn default() -> Self {
        CycleCollectionSettings {
            min_games_per_cycle: default_min_games_per_cycle(),
            min_samples_per_cycle: default_min_samples_per_cycle(),
            max_time_per_cycle_s: default_max_time_per_cycle(),
        }
    }
}

impl CycleCollectionSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.min_games_per_cycle.is_none()
            && self.min_samples_per_cycle.is_none()
            && self.max_time_per_cycle_s.is_none()
        {
            Err("At least one of 'min_games_per_cycle', 'min_samples_per_cycle', or 'max_time_per_cycle_s' must be set.".to_string())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// How many samples to keep in the training buffer (and save to disk).
    pub samples: u64,

    pub batch_size: usize,

    pub epochs: usize,

    pub learning_rate: f32,

    pub weight_decay: f32,

    pub policy_loss_weight: f32,

    pub z_value_loss_weight: f32,

    pub q_value_loss_weight: f32,

    pub number_of_aux_features: usize,

    pub warmup_steps: usize,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize, JsonSchema)]
#[derive(AsRefStr)]
pub enum GameNames {
    Hex2Absolute,
    Hex2Canonical,
    Hex4Absolute,
    Hex4Canonical,
    Hex5Absolute,
    Hex5Canonical,
    Hex5CanonicalAxiomBias,
    Hex5CanonicalAxiomBiasBertCls,
    ChessAbsPositionalEnc,
    ChessAxiomBias,
    ChessBigBert,
    ChessBigBertV2,
    ChessShaw,
    ChessHybrid,
    ChessDomain
}

impl GameNames {
    pub fn to_internal_name(&self) -> &str {
        self.as_ref()
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GameConfig {
    pub name: GameNames,

}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct PerformanceSettings {
    /// How big chunks are used when saving and loading Parquet files, that the Python server
    /// uses for training. Higher sizes use less disk space, but take longer to write.
    /// Note that this can be a bottleneck with games with huge actions spaces or states.
    #[serde(default = "default_parquest_chunk_size")]
    pub parquest_chunk_size: usize,

    /// GPU cache size.
    pub cache_size: usize,

    /// How many threads to use for writing to the GPU cache.
    pub cache_writers: usize,

    /// Size of the arena for retaining the search tree between moves.
    /// Higher values use more memory, but improve performance, since every time
    /// the size is exceeded, the tree is de-fragmented, which is expensive.
    #[serde(default = "default_tree_arena_size")]
    pub tree_arena_size: usize,

    /// The batch size for GPU inference.
    /// For best performance, perform a benchmark to find the optimal size.
    pub gpu_batch_size: i32,

    /// Number of CPU threads to spawn for game generation.
    pub cpu_generator_threads: usize,

    /// Number of games each CPU thread will manage.
    pub games_per_cpu_generator: usize,

    /// Configuration for the GPU spawner and inference workers.
    pub gpu: GpuSpawnerChoice,
}

fn default_parquest_chunk_size() -> usize {
    5_000
}

fn default_tree_arena_size() -> usize {
    100_000
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    /// The game configuration, which defines the game name and whether it uses absolute mapping.
    pub game: GameConfig,

    /// Whether to just generate games for 1 cycle and exit, without connecting to the training server.
    pub standalone: bool,

    #[serde(default = "default_python_interpreter_path")]
    pub python_interpreter_path: String,

    /// Name of the run, used for storing models, buffers, and other artifacts.
    pub run_name: String,

    /// Whether to overwrite existing run artifacts. Be careful with this!
    #[serde(default = "default_overwrite_run")]
    pub overwrite_run: bool,

    /// Base directory for 'run_name' artifacts.
    #[serde(default = "default_base_dir")]
    pub base_dir: String,

    /// Performance-related settings.
    pub performance: PerformanceSettings,

    /// How many 'self-play'-'train' cycles to perform.
    /// If null, runs indefinitely, till the user stops the process.
    pub number_of_cycles: Option<u64>,

    /// Settings for training the neural network (PyTorch).
    pub training: TrainingConfig,

    /// Settings for the self-play process.
    pub self_play_settings: SelfPlaySettings,

    /// Settings for the self-play cycle, which controls how often to collect games and samples.
    /// At least one of the fields must be set to a non-null value.
    #[serde(default)]
    pub cycle_collection: CycleCollectionSettings,
}

fn default_python_interpreter_path() -> String {
    "python/.venv/bin/python".to_string()
}

fn default_base_dir() -> String {
    "runs".to_string()
}

fn default_overwrite_run() -> bool {
    false
}

impl AppConfig {
    /// Loads configuration from a YAML file.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_yaml::from_reader(file)?;
        config.validate()?;

        info!("Configuration from '{}' loaded:\n{:#?}", &path, config);

        Ok(config)
    }

    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.cycle_collection.validate()?;

        Ok(())
    }

    pub fn save_to_file(&self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        serde_yaml::to_writer(file, self)?;
        Ok(())
    }
}
