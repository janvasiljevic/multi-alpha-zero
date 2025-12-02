use decorum::R32;
use itertools::Itertools;
use num_traits::FromPrimitive;
use rand::distr::weighted::WeightedIndex;

#[derive(Debug, Copy, Clone)]
pub struct MoveSelector {
    /// The temperature applied to the policy before sampling. Can be any positive value.
    /// * `0.0`: always pick the move with the highest policy
    /// * `inf`: pick a completely random move
    pub temperature: f32,

    /// After this number of moves, use temperature zero to always select the best move.
    pub zero_temp_move_count: u32,
}

impl MoveSelector {
    pub fn new(temperature: f32, zero_temp_move_count: u32) -> Self {
        MoveSelector {
            temperature,
            zero_temp_move_count,
        }
    }

    /// Always select the move with the maximum policy, temperature 0.
    pub fn zero_temp() -> Self {
        Self::new(0.0, 0)
    }
}

impl MoveSelector {
    pub fn select(&self, move_count: u32, policy: &[f32], rng: &mut impl rand::Rng) -> usize {
        let temperature = if move_count >= self.zero_temp_move_count {
            0.0
        } else {
            self.temperature
        };

        assert!(temperature >= 0.0);

        // we handle the extreme cases separately, in theory that would not be necessary but they're degenerate
        if temperature == 0.0 {
            // pick the best move
            policy
                .iter()
                .copied()
                .map::<Option<R32>, _>(R32::from_f32)
                .position_max()
                .unwrap()
        } else if temperature == f32::INFINITY {
            // pick a random move
            rng.random_range(0..policy.len())
        } else {
            // pick according to `policy ** (1/temperature)`
            let policy_temp = policy.iter().map(|p| p.powf(1.0 / temperature));
            let distribution = WeightedIndex::new(policy_temp).unwrap();
            rng.sample(distribution)
        }
    }
}
