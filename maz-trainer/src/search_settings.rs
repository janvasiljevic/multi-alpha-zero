use crate::node::{FpuMode, UctWeights};
use maz_config::config::SelfPlaySettings;
use rand::Rng;

/// Settings for the AZ algorithm.
/// For temperature:
/// 0 ~ deterministic, 1 ~ uniform random, >1 ~ more random, ~ inf ~ completely random.
#[derive(Copy, Clone)]
pub struct SearchSettings {
    /// Cut-off for the maximum game length. If none, no limit is applied.
    pub max_game_length: Option<u64>,
    pub weights: UctWeights,

    /// Temperature for picking moves in **Move selection**.
    /// (So when we have an actual improved policy, not just priors from the network.)
    pub move_selection_temp: f32,

    /// After this number of moves, use temperature zero to always select the best move.
    pub zero_temp_move_count: u32,

    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,

    /// Temperature for the root and child nodes in the search policy (**not the move selection!**).
    /// (So `policy_softmax_temperature_in_place` kind of thing!)
    pub search_policy_temperature_root: f32,
    pub search_policy_temperature_child: f32,

    pub search_fpu_root: FpuMode,
    pub search_fpu_child: FpuMode,
    pub search_virtual_loss_weight: f32,

    // How much to evaluate draws
    pub contempt: f32,

    /// An idea taken from KataGo. Play some rollouts with a full search, and some with a partial search.
    /// This should in theory help with better quality sampling for the value network.
    /// Make `full_search_prob` 1.0 to always use full search (aka standard AZ).
    pub full_search_prob: f64,
    pub full_iterations: u64,
    pub part_iterations: u64,
    pub part_cpuct_exploration: f32,
}

impl SearchSettings {
    /// Playout cap randomization from KataGo.
    pub fn do_full_search(&self, rng: &mut impl Rng) -> (bool, u64) {
        let use_full_search = rng.random_bool(self.full_search_prob);
        let target_rollouts = if use_full_search {
            self.full_iterations
        } else {
            self.part_iterations
        };
        (use_full_search, target_rollouts)
    }
}

impl From<SelfPlaySettings> for SearchSettings {
    fn from(settings: SelfPlaySettings) -> Self {
        SearchSettings {
            max_game_length: settings.max_game_length,
            weights: UctWeights {
                exploration_weight: settings.uct.exploration_weight,
                moves_left_weight: settings.uct.moves_left_weight,
                moves_left_clip: settings.uct.moves_left_clip,
                moves_left_sharpness: settings.uct.moves_left_sharpness,
            },
            move_selection_temp: settings.move_selection.temperature,
            zero_temp_move_count: settings.move_selection.zero_temp_move_count,

            dirichlet_alpha: settings.dirichlet.alpha,
            dirichlet_eps: settings.dirichlet.eps,

            search_policy_temperature_root: settings.search_policy_temperature_root,
            search_policy_temperature_child: settings.search_policy_temperature_child,
            search_fpu_root: FpuMode::from(settings.search_fpu_root),
            search_fpu_child: FpuMode::from(settings.search_fpu_child),
            search_virtual_loss_weight: 0.0, // We don't use virtual loss in self-play.

            contempt: settings.contempt,

            full_search_prob: settings.rollouts.full_search_prob,
            full_iterations: settings.rollouts.full_iterations,
            part_iterations: settings.rollouts.part_iterations,
            part_cpuct_exploration: settings.rollouts.part_cpuct_exploration,
        }
    }
}

impl Default for SearchSettings {
    // The defaults are optimized for inference, not for training.
    fn default() -> Self {
        SearchSettings {
            max_game_length: None,
            weights: UctWeights {
                exploration_weight: 3.0,
                moves_left_weight: 0.0,
                moves_left_clip: 0.0,
                moves_left_sharpness: 0.0
            },
            move_selection_temp: 0.0,
            zero_temp_move_count: 0,

            dirichlet_alpha: 0.0,
            dirichlet_eps: 0.0,

            search_policy_temperature_root: 1.0,
            search_policy_temperature_child: 1.0,
            search_fpu_root: FpuMode::Fixed(0.0),
            search_fpu_child: FpuMode::Fixed(0.0),
            search_virtual_loss_weight: 1.0,
            
            contempt: -0.1,

            full_search_prob: 1.0,
            full_iterations: 200,
            part_iterations: 0,
            part_cpuct_exploration: f32::NAN,
        }
    }
}
