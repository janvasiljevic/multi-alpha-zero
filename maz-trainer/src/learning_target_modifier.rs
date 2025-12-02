use game_tri_chess::chess_game::TriHexChess;
use maz_core::mapping::Board;
use maz_core::values_const::ValuesAbs;
use std::any::Any;
use std::fmt::Debug;

pub struct PlayerRanges {
    pub num_of_players: usize,
    pub start_move_count: usize,
    pub end_move_count: usize,
}

pub trait LearningTargetModifier<B: Board + Any>: Debug + Send + Sync {
    fn modify_target(
        &self,
        board: &B,
        z_target: &Vec<f32>,
        q_target: &Vec<f32>,
        is_absolute: bool,
        current_move_count: u32,
        ranges: &Vec<PlayerRanges>,
    ) -> Vec<f32>;
}

impl<B: Board + Any> LearningTargetModifier<B> for LearningModifier {
    fn modify_target(
        &self,
        board: &B,
        z_target: &Vec<f32>,
        q_target: &Vec<f32>,
        is_absolute: bool,
        current_move_count: u32,
        ranges: &Vec<PlayerRanges>,
    ) -> Vec<f32> {
        match self {
            LearningModifier::NoOp(modifier) => modifier.modify_target(
                board,
                z_target,
                q_target,
                is_absolute,
                current_move_count,
                ranges,
            ),
            LearningModifier::MaterialAdvantage(modifier) => modifier.modify_target(
                board,
                z_target,
                q_target,
                is_absolute,
                current_move_count,
                ranges,
            ),
        }
    }
}

// 2. Create a new enum that can be cloned and debugged
#[derive(Debug, Clone)]
pub enum LearningModifier {
    NoOp(NoOpModifier),
    MaterialAdvantage(MaterialAdvantageModifier),
}

#[derive(Debug, Clone)]
pub struct NoOpModifier;

impl<B: Board + Any> LearningTargetModifier<B> for NoOpModifier {
    fn modify_target(
        &self,
        _board: &B,
        z_target: &Vec<f32>,
        _q_target: &Vec<f32>,
        _is_absolute: bool,
        _current_move_count: u32,
        _ranges: &Vec<PlayerRanges>,
    ) -> Vec<f32> {
        z_target.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MaterialAdvantageModifier {
    transition_duration_moves: u32,
    w_z: f32,
    w_q: f32,
    w_material: f32,
}

impl MaterialAdvantageModifier {
    pub(crate) fn new(transition_duration_moves: u32, w_z: f32, w_q: f32, w_material: f32) -> Self {
        // Ensure the weights sum to 1.0
        let total_weight = w_z + w_q + w_material;
        let w_z = w_z / total_weight;
        let w_q = w_q / total_weight;
        let w_material = w_material / total_weight;

        Self {
            transition_duration_moves,
            w_z,
            w_q,
            w_material,
        }
    }
}

impl<B: Board + Any> LearningTargetModifier<B> for MaterialAdvantageModifier {
    fn modify_target(
        &self,
        board: &B,
        z_target: &Vec<f32>,
        q_target: &Vec<f32>,
        is_absolute: bool,
        current_move_count: u32,
        ranges: &Vec<PlayerRanges>,
    ) -> Vec<f32> {
        // Only modify if more than 2 players are active
        if board.player_num_of_active() <= 2 {
            return z_target.clone();
        }

        let board_any = board as &dyn Any;

        match board_any.downcast_ref::<TriHexChess>() {
            Some(chess_board) => {
                let material_count = chess_board.material_count();

                const NORMALIZATION_FACTOR: f32 = 10.0;

                let mut heuristic_values = [0.0f32; 3];

                for p_idx in 0..3 {
                    let mut max_opponent_material = 0.0f32;

                    for opp_idx in 0..3 {
                        if p_idx == opp_idx {
                            continue;
                        }
                        max_opponent_material = max_opponent_material.max(material_count[opp_idx] as f32);
                    }

                    let my_material = material_count[p_idx] as f32;

                    // The core logic: my material vs. the MAX of my opponents.
                    let material_advantage = my_material - max_opponent_material;

                    // Use tanh to squash the value into the [-1, 1] range for the training target.
                    heuristic_values[p_idx] = (material_advantage / NORMALIZATION_FACTOR).tanh();
                }

                let material_reward = ValuesAbs::<3> {
                    value_abs: heuristic_values,
                    moves_left: 0.0,
                }
                    .to_values_vec(is_absolute, board.player_current().into());

                let hybrid_reward: Vec<f32> = z_target
                    .iter()
                    .zip(q_target.iter())
                    .zip(material_reward.iter())
                    .map(|((z, q), m)| self.w_z * z + self.w_q * q + self.w_material * m)
                    .collect();

                if let Some(r) = ranges.iter().find(|r| r.num_of_players == 3) {
                    let phase_end_move = r.end_move_count as u32;

                    // Define the start of our transition window.
                    // It's N moves before the phase ends.
                    let transition_start_move =
                        phase_end_move.saturating_sub(self.transition_duration_moves);

                    // Check if we are currently inside this "fade-to-Z" window.
                    if current_move_count >= transition_start_move
                        && current_move_count <= phase_end_move
                    {
                        // Calculate how far we are into the transition (0.0 to 1.0).
                        let progress = if self.transition_duration_moves == 0 {
                            1.0 // If duration is 0, transition is instant.
                        } else {
                            (current_move_count - transition_start_move) as f32
                                / self.transition_duration_moves as f32
                        };

                        let clamped_progress = progress.clamp(0.0, 1.0);

                        // Interpolate FROM the hybrid reward TO the pure Z target.
                        // As `clamped_progress` -> 1.0, the weight of `z` increases.
                        return hybrid_reward
                            .iter()
                            .zip(z_target.iter())
                            .map(|(hybrid_r, z)| {
                                hybrid_r * (1.0 - clamped_progress) + z * clamped_progress
                            })
                            .collect();
                    }
                }

                // If not in the transition window, return the standard hybrid reward.
                hybrid_reward
            }
            None => z_target.clone(),
        }
    }
}


// tests#[cfg(test)]
#[cfg(test)]
mod tests {
    // test the MaterialAdvantageModifier
    use super::*;

    #[test]
    fn test_material_advantage_modifier() {
        let modifier = MaterialAdvantageModifier::new(10, 0.4, 0.4, 0.8);

        // Create a mock board with 3 players active
        let board = TriHexChess::default();

        let z_target = vec![0.0, 0.0, 0.0];
        let q_target = vec![0.0, 0.0, 0.0];

        let is_absolute = true;
        let current_move_count = 15;
        let ranges = vec![
            PlayerRanges {
                num_of_players: 3,
                start_move_count: 0,
                end_move_count: 20,
            }
        ];

        let modified_target = modifier.modify_target(
            &board,
            &z_target,
            &q_target,
            is_absolute,
            current_move_count,
            &ranges,
        );

        // Check that the modified target is as expected
        assert_eq!(modified_target.len(), 3);

        println!("Modified target: {:?}", modified_target);
    }
}