use crate::range::IdxRange;
use maz_core::values_const::ValuesAbs;
use maz_config::config::ConfigFpuMode;
use std::fmt::Display;
use maz_core::mapping::BoardPlayer;

#[derive(Debug, Clone)]
pub struct Node<M: Display, const N: usize> {
    // Potentially update Tree::keep_moves when this struct gets new fields.
    /// The parent node.
    pub parent: Option<usize>,

    /// The move that was just made to get to this node. Is `None` only for the root node.
    pub last_move: Option<M>,

    /// The index of the last move in the policy vector that was used to expand this node.
    /// Created by BoardMapper.
    /// If root, this is 0
    pub last_move_policy_index: usize,

    /// The children of this node. Is `None` if this node has not been visited yet.
    pub children: Option<IdxRange>,

    /// The number of non-virtual visits for this node and its children.
    pub complete_visits: u64,

    /// The number of virtual visits for this node and its children.
    pub virtual_visits: u64,

    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value.
    /// Does not include virtual visits.
    pub sum_values: ValuesAbs<N>,

    /// The data returned by the network for this position.
    /// If `None` and the node has children this means this is a virtual node with data to be filled in later.
    pub net_values: Option<ValuesAbs<N>>,

    /// The policy/prior probability as evaluated by the network when the parent node was expanded.
    pub net_policy: f32,

    /// This is needed when playing games in canonical form, where the player is not always 0,
    /// so we can transform PovValues into ValuesAbs used in the node.
    /// We can only set player_index once the node is visited - if we would
    /// set it earlier it meeans we would actually need to perform the move to get the player index!
    /// In 2 player games you could obviously just get the next player,
    /// however in multiplayer games where players can be skipped (they are eliminated),
    /// this is not possible.
    pub player_index: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
pub struct Uct {
    /// value, range -1..1
    pub q: f32,
    /// exploration, range 0..inf
    pub u: f32,
    /// moves left delta, range -inf..inf
    ///   positive means this node has more moves left than its siblings
    pub m: f32,
}

// #[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[derive(Debug, Copy, Clone)]
pub struct UctWeights {
    pub exploration_weight: f32,

    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
}

#[derive(Debug, Clone)]
pub struct UctContext<const N: usize> {
    pub complete_visits: u64,
    pub virtual_visits: u64,

    pub total_visits: u64,
    pub values: ValuesAbs<N>,

    pub visited_policy_mass: f32,
}

/// Which value to use as `Q` for unvisited nodes in the PUCT formula.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum FpuMode {
    /// Use the given fixed value.
    Fixed(f32),
    /// Use the parent value (from the current POV) with an offset added to it.
    Relative(f32),
}

impl From<ConfigFpuMode> for FpuMode {
    fn from(mode: ConfigFpuMode) -> Self {
        match mode {
            ConfigFpuMode::Fixed(fpu) => FpuMode::Fixed(fpu.fixed),
            ConfigFpuMode::Relative(fpu) => FpuMode::Relative(fpu.relative),
        }
    }
}

impl Default for UctWeights {
    fn default() -> Self {
        UctWeights {
            exploration_weight: 3.5,
            moves_left_weight: 0.00, // TODO: Test this when implementing moves left
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        }
    }
}

impl Uct {
    pub fn nan() -> Uct {
        Uct {
            q: f32::NAN,
            u: f32::NAN,
            m: f32::NAN,
        }
    }

    pub fn total(self, weights: UctWeights) -> f32 {
        let Uct { q, u, m } = self;

        let m_unit = if weights.moves_left_weight == 0.0 {
            0.0
        } else {
            let m_clipped = m.clamp(-weights.moves_left_clip, weights.moves_left_clip);
            (weights.moves_left_sharpness * m_clipped * -q).clamp(-1.0, 1.0)
        };

        q + weights.exploration_weight * u + weights.moves_left_weight * m_unit
    }
}

impl<M: Display, const N: usize> Node<M, N> {
    pub(super) fn new(
        parent: Option<usize>,
        last_move: Option<M>,
        last_move_policy_index: usize,
        p: f32,
        player_index: Option<usize>,
    ) -> Self {
        Node {
            parent,
            last_move,
            last_move_policy_index,
            children: None,

            complete_visits: 0,
            virtual_visits: 0,
            sum_values: ValuesAbs::<N>::default(),

            net_values: None,
            net_policy: p,
            player_index,
        }
    }

    /// Used when retaining the subtree.
    pub fn set_root_properties(&mut self) {
        self.parent = None;
        self.last_move = None;
        self.last_move_policy_index = usize::default();
        self.net_policy = f32::NAN;
    }

    pub fn total_visits(&self) -> u64 {
        self.complete_visits + self.virtual_visits
    }

    /// The (normalized) values of this node.
    pub fn values(&self) -> ValuesAbs<N> {
        self.sum_values / self.complete_visits as f32
    }

    pub fn uct_context(&self, visited_policy_mass: f32) -> UctContext<N> {
        UctContext {
            complete_visits: self.complete_visits,
            virtual_visits: self.virtual_visits,
            total_visits: self.complete_visits + self.virtual_visits,
            values: self.values(),
            visited_policy_mass,
        }
    }

    pub fn is_terminal_node(&self) -> bool {
        self.children.is_none() && self.total_visits() > 0
    }

    pub fn uct(
        &self,
        parent: &UctContext<N>,
        fpu_mode: FpuMode,
        // q_mode: QMode, // We don't really have this to use yet - our current network is jut 3x1 value, so it doesn't really matter yet
        virtual_loss_weight: f32,
        player: impl BoardPlayer,
    ) -> Uct {
        if parent.total_visits == 0 {
            return Uct::nan();
        }

        let total_visits = self.total_visits();

        let fpu = match fpu_mode {
            FpuMode::Relative(_scalar) => {
                panic!("This isn't supported and not in use");

                // TODO: Men se tle iskreno zdi da je bil bug v originalni kodi, ker se je uporabljal `parent.values.pov(pov)` namesto `parent.values()`
                // pac ne gledat iz pov? Idk, weird stuff tle
                // let parent_values_pov = parent.values.pov(pov);
                // let parent_value = q_mode
                //     .select(parent_values_pov.value, parent_values_pov.wdl)
                //     .value;

                // TODO: Ni pravilno semizdi, ker 'pov' gleda na napacnega igralca....
                // let parent_value = parent.values.val_for_player(player);
                //
                // parent_value - scalar * parent.visited_policy_mass.sqrt()
            }
            FpuMode::Fixed(fpu) => fpu,
        };

        debug_assert!(
            virtual_loss_weight >= 0.0,
            "Virtual loss weight should not be negative"
        );

        let total_visits_virtual =
            self.complete_visits as f32 + virtual_loss_weight * self.virtual_visits as f32;

        let q = if total_visits_virtual == 0.0 {
            fpu
        } else {
            let total_value = self.sum_values.value_abs[player.into()];

            let total_value_virtual =
                total_value - virtual_loss_weight * self.virtual_visits as f32;

            total_value_virtual / total_visits_virtual
            // Just testing stuff with copilot
            // --- 1. Calculate Q for the current player (`q_me`) ---
            // We incorporate virtual loss into "my" value, as this is the standard
            // way to discourage other threads from exploring this same node.
            // let my_total_value = self.sum_values.value_abs[player.into()];
            // let my_total_value_virtual = my_total_value - virtual_loss_weight * self.virtual_visits as f32;
            // let q_me = my_total_value_virtual / total_visits_virtual;
            //
            // // --- 2. Find the maximum Q among all opponents (`max_q_opponent`) ---
            // let mut max_q_opponent = f32::NEG_INFINITY;
            // let player_idx = player.into();
            //
            // // Assuming N::PLAYER_COUNT exists. Replace with the actual way you get the player count.
            // for i in 0..N {
            //     if i == player_idx {
            //         // This is me, not an opponent.
            //         continue;
            //     }
            //
            //     // For opponents, we calculate their raw Q-value without virtual loss.
            //     // Virtual loss is a penalty from the perspective of the current searcher.
            //     let opponent_total_value = self.sum_values.value_abs[i];
            //     let q_opponent = opponent_total_value / total_visits_virtual;
            //
            //     if q_opponent > max_q_opponent {
            //         max_q_opponent = q_opponent;
            //     }
            // }
            //
            // // Handle the case of a 2-player game where the loop finds no opponents.
            // // This is unlikely in your 3+ player scenario, but good for robustness.
            // if max_q_opponent == f32::NEG_INFINITY {
            //     max_q_opponent = 0.0; // Or some other neutral value
            // }
            //
            // // println!("My q value: {}, Max opponent q value: {}", q_me, max_q_opponent);
            //
            // // --- 3. The final Max-n Q-value ---
            // // This value is high if my prospects are good AND my strongest opponent's prospects are poor.
            // q_me - max_q_opponent
        };

        let u =
            self.net_policy * ((parent.total_visits - 1) as f32).sqrt() / (1 + total_visits) as f32;

        let m = if self.complete_visits == 0 {
            // don't even bother with moves_left if we don't have any information
            0.0
        } else {
            // this node has been visited, so we know parent_moves_left is also a useful value
            self.values().moves_left - (parent.values.moves_left - 1.0)
        };

        Uct { q, u, m }
    }
}
