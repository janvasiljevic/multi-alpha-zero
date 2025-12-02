use crate::learning_target_modifier::{LearningModifier, LearningTargetModifier, PlayerRanges};
use crate::move_selector::MoveSelector;
use crate::node::UctWeights;
use crate::search_settings::SearchSettings;
use crate::steppable_mcts::{CacheShapeMissmatch, ConsumeValues, DirichletError, SteppableMCTS};
use crate::tree::ROOT_NODE_ID;
use log::warn;
use maz_core::mapping::{
    Board, MetaBoardMapper, MoveStore, OptionalSharedOracle, OracleAnalysis, OracleError,
};
use maz_core::values_const::{ValuesAbs, ValuesAbsLike};
use ndarray::Array2;
use rand::Rng;
use std::num::NonZeroUsize;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct Sample<B: Board> {
    pub encoded_board: Vec<bool>,
    pub current_move_count: u32,
    pub board: B,

    pub number_of_active_players: usize,

    pub player_index: usize,

    pub mcts_policy: Vec<f32>,
    pub legal_moves_mask: Vec<bool>,

    pub aux_values: Vec<f32>,

    pub root_q_values: Vec<f32>,

    pub best_action_q_value: Vec<f32>,

    pub depth_range: (usize, usize),
    pub mcts_policy_entropy: f32,
}

#[derive(Debug, Clone)]
pub struct FinalizedSample<B: Board> {
    pub inner: Sample<B>,

    /// Final outcome of the game (z)
    /// Either POV or Absolute, depending on the board mapper.
    pub z_values: Vec<f32>,

    pub moves_left: u32,
}

#[derive(Debug, Clone)]
pub struct Simulation<B: Board> {
    pub samples: Vec<FinalizedSample<B>>,
    pub actual_game_length: u32,
    pub outcome: Box<dyn ValuesAbsLike>,
}

pub struct ProduceOutput<B: Board> {
    pub board: B,
    pub node_id: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SPGStats {
    pub total_retained_nodes: u64,
}

fn calculate_entropy(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0) // Avoid log(0), which is -inf.
        .map(|&p| -p * p.log2())
        .sum()
}

#[derive(Debug, Clone)]
pub struct SelfPlayGame<B: Board + 'static, const N: usize> {
    pub mcts: SteppableMCTS<B, N>,
    pub stats: SPGStats,

    /// Samples we have collected so far in this game.
    samples: Vec<Sample<B>>,

    /// Game length is not the same as the number of positions, because
    /// we might be using playout cap randomization.
    game_length: usize,

    /// Playout cap randomization (KataGo style).
    is_full_search: bool,

    /// Allocated once, reused for each position.
    state_buffer: Array2<bool>,

    /// Allocated once, reused for each position.
    mask_buffer: Vec<bool>,

    /// Allocated once, reused for each position.
    policy_buffer: Vec<f32>,

    /// Optional oracle for additional evaluations.
    pub oracle: OptionalSharedOracle<B>,

    oracle_takeover: Option<OracleAnalysis<B::Move>>,

    learning_target_modifier: LearningModifier,
}

impl<B: Board, const N: usize> SelfPlayGame<B, N> {
    pub fn new(
        game: &B,
        mapper: &impl MetaBoardMapper<B>,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        oracle: OptionalSharedOracle<B>,
        learning_target_modifier: LearningModifier,
    ) -> Self {
        let (is_full_search, number_of_rollouts) = settings.do_full_search(rng);

        let mcts = SteppableMCTS::new_with_capacity(game, mapper, number_of_rollouts, false);
        Self {
            mcts,
            samples: vec![],
            game_length: 0,
            stats: SPGStats::default(),
            is_full_search,

            policy_buffer: vec![0.0; mapper.policy_len()],
            state_buffer: Array2::from_elem(mapper.input_board_shape(), false),
            mask_buffer: vec![false; mapper.policy_len()],
            oracle,
            oracle_takeover: None,
            learning_target_modifier,
        }
    }

    /// Reset the game state for a new game,
    pub fn reset_mut(
        &mut self,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        board_mapper: &impl MetaBoardMapper<B>,
    ) {
        // Clear previous samples and reset game length.
        self.samples.clear();
        self.game_length = 0;
        self.oracle_takeover = None;

        let (is_full_search, number_of_rollouts) = settings.do_full_search(rng);
        self.is_full_search = is_full_search;

        let new_board = self.mcts.board.new_varied();

        self.mcts.board = new_board.clone();

        self.mcts = SteppableMCTS::new_with_capacity(
            &new_board,
            board_mapper,
            number_of_rollouts,
            self.mcts.using_virtual_loss,
        );
    }

    pub fn get_position_count(&self) -> usize {
        self.game_length
    }

    pub fn receive(
        &mut self,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        node_id: usize,
        values: ConsumeValues,
    ) -> Result<(), CacheShapeMissmatch> {
        // Only apply dirichlet, if this is a full search.
        self.mcts
            .consume(rng, settings, node_id, values, self.is_full_search)
    }

    pub fn advance(
        &mut self,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        samples_tx: &UnboundedSender<Simulation<B>>,
        board_mapper: &impl MetaBoardMapper<B>,
    ) -> Option<ProduceOutput<B>> {
        if let Some(oracle_plan) = &self.oracle_takeover {
            self.execute_oracle_plan(oracle_plan.clone(), rng, settings, samples_tx, board_mapper);
            // Since we're in oracle mode, we never need a network evaluation.
            return None;
        }

        // Probe Oracle BEFORE any MCTS work
        if let Some(oracle) = &self.oracle {
            match oracle.probe(&self.mcts.board) {
                Ok(analysis) => {
                    self.oracle_takeover = Some(analysis.clone());
                    self.execute_oracle_plan(analysis, rng, settings, samples_tx, board_mapper);
                    // No network evaluation needed.
                    return None;
                }
                Err(OracleError::NotApplicable) => {
                    // This is the normal case: not an endgame. Proceed to MCTS.
                }
            }
        }

        if self.mcts.is_enough_rollouts() {
            self.mcts
                .tree
                .improved_policy_vector(&mut self.policy_buffer, &mut self.mask_buffer);

            board_mapper.encode_input(&mut self.state_buffer.view_mut(), &self.mcts.board);

            let player_index: usize = self.mcts.board.player_current().into();

            let aux_values = self
                .mcts
                .board
                .get_aux_values(board_mapper.is_absolute())
                .unwrap_or_else(|| vec![0.0; N]);

            let abs_root_values = self.mcts.tree.root_q_abs_values();
            let abs_best_child_q_values = self.mcts.tree.best_child_q_abs_values();

            // If we did a full search, store the sample for training and do the normal
            // move selection with temperature.
            // If we did a fast search, we discard the sample and greedily pick the best move.
            let move_selector = if self.is_full_search {
                self.samples.push(Sample {
                    player_index,
                    current_move_count: self.get_position_count() as u32,
                    mcts_policy: self.policy_buffer.clone(),
                    depth_range: self.mcts.tree.depth_range(0),
                    encoded_board: self.state_buffer.as_slice().unwrap().to_vec(),
                    mcts_policy_entropy: calculate_entropy(&self.policy_buffer),
                    board: self.mcts.board.clone(),
                    aux_values,
                    number_of_active_players: self.mcts.board.player_num_of_active(),
                    legal_moves_mask: self.mask_buffer.to_vec(),
                    root_q_values: abs_root_values
                        .to_values_vec(board_mapper.is_absolute(), player_index),
                    best_action_q_value: abs_best_child_q_values
                        .to_values_vec(board_mapper.is_absolute(), player_index),
                });

                MoveSelector::new(settings.move_selection_temp, settings.zero_temp_move_count)
            } else {
                MoveSelector::zero_temp()
            };

            self.game_length += 1;

            let root_children = self.mcts.tree[ROOT_NODE_ID]
                .children
                .expect("Root node should have children");

            // Instead of using the full improved policy vector which can be several thousands of moves
            // with a small % of non-zero entries, we compact it down to just the children of the root node.
            let compact_improved_policy = root_children
                .iter()
                .map(|c| self.mcts.tree.get_node_improved_policy(c))
                .collect::<Vec<f32>>();

            // This will give an index from 0 to number of children - 1
            let selected_child_sequential_id = move_selector.select(
                self.get_position_count() as u32,
                &compact_improved_policy,
                rng,
            );

            let picked_child_node_id = root_children.get(selected_child_sequential_id);

            let picked_move = self.mcts.tree[picked_child_node_id]
                .last_move
                .expect("Child node should have a move");

            let mut move_store = <B as Board>::MoveStore::default();

            // Some games check the move store when playing a move, to validate
            // it - but only in debug mode.
            #[cfg(debug_assertions)]
            {
                self.mcts.board.fill_move_store(&mut move_store);
            }

            self.mcts
                .board
                .play_move_mut_with_store(&picked_move, &mut move_store, None);

            let is_terminal = self.mcts.board.is_terminal();

            let total_moves = self.get_position_count() as u32;

            let force_end = if let Some(max_len) = settings.max_game_length {
                !is_terminal && (total_moves >= max_len as u32)
            } else {
                false
            };

            let can_end_early = !is_terminal && self.mcts.board.can_game_end_early();

            if is_terminal || force_end || can_end_early {
                self.finalize_and_send_simulation(samples_tx, board_mapper, force_end);

                self.reset_mut(rng, settings, board_mapper);
            } else {
                let retained_nodes = self.mcts.tree.nodes[picked_child_node_id].total_visits();

                self.stats.total_retained_nodes += retained_nodes;

                let picked_child_node_id = NonZeroUsize::new(picked_child_node_id)
                    .expect("Picked child node ID must be non-zero");

                // TODO: Add a fine tune for the tree size - something like de-frag memory limit
                if self.mcts.tree.nodes.len() > 100_000 {
                    self.mcts.tree.retain_subtree_with_defrag(
                        self.mcts.board.clone(),
                        picked_child_node_id,
                        board_mapper.average_number_of_moves() as u64,
                    )
                } else {
                    self.mcts
                        .tree
                        .retain_subtree(self.mcts.board.clone(), picked_child_node_id);
                }

                let (use_full_search, target_rollouts) = settings.do_full_search(rng);

                self.is_full_search = use_full_search;
                self.mcts.min_number_of_rollouts = target_rollouts;

                // We need to 'forcefully' inject dirichlet noise here, because
                // the root node is almost definitely already expanded (has visit count > 0).
                // This means it will never receive a policy network evaluation again,
                // and thus never apply dirichlet noise in the normal way.
                if self.is_full_search {
                    match self.mcts.apply_dirichlet(rng, settings, ROOT_NODE_ID) {
                        Ok(_) => {}
                        Err(DirichletError::ChildrenNotExpanded { .. }) => {
                            warn!(
                                "Board {:?} didn't have children to apply dirichlet noise to. This is a super rare case and might be indicative of a bug.",
                                self.mcts.board
                            );
                        }
                        Err(DirichletError::ZeroChildren { .. }) => {
                            panic!(
                                "Board {:?} had zero children in the root node. This means it should be terminal, but it isn't...",
                                self.mcts.board
                            );
                        }
                    }
                }
            }
        }

        self.mcts.step(
            rng,
            settings.search_fpu_root,
            settings.search_fpu_child,
            settings.search_virtual_loss_weight,
            // Use a lower exploration for partial search, to make it more greedy.
            if self.is_full_search {
                settings.weights
            } else {
                UctWeights {
                    exploration_weight: settings.part_cpuct_exploration,
                    moves_left_weight: 0.0,
                    moves_left_clip: 0.0,
                    moves_left_sharpness: 0.0,
                }
            },
            board_mapper,
            settings.contempt,
        )
    }

    fn execute_oracle_plan(
        &mut self,
        oracle_plan: OracleAnalysis<B::Move>,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        samples_tx: &UnboundedSender<Simulation<B>>,
        board_mapper: &impl MetaBoardMapper<B>,
    ) {
        self.policy_buffer.fill(0.0);
        self.mask_buffer.fill(false);

        let move_idx =
            board_mapper.move_to_index(self.mcts.board.player_current(), oracle_plan.best_move);
        self.policy_buffer[move_idx] = 1.0;

        let mut move_store = <B as Board>::MoveStore::default();

        self.mcts.board.fill_move_store(&mut move_store);

        for mv in move_store.iter() {
            let idx = board_mapper.move_to_index(self.mcts.board.player_current(), mv);
            self.mask_buffer[idx] = true;
        }

        // Generate a perfect training sample if it's a full search turn.
        board_mapper.encode_input(&mut self.state_buffer.view_mut(), &self.mcts.board);

        let player_index: usize = self.mcts.board.player_current().into();

        let aux_values = self
            .mcts
            .board
            .get_aux_values(board_mapper.is_absolute())
            .unwrap_or_else(|| vec![0.0; N]);

        let abs_win = ValuesAbs::<N> {
            value_abs: oracle_plan
                .outcome_abs
                .clone()
                .try_into()
                .expect("Oracle outcome length mismatch"),
            moves_left: 0.0,
        };

        self.samples.push(Sample {
            player_index,
            current_move_count: self.get_position_count() as u32,
            encoded_board: self.state_buffer.as_slice().unwrap().to_vec(),
            mcts_policy: self.policy_buffer.clone(),
            legal_moves_mask: self.mask_buffer.clone(),
            aux_values,
            board: self.mcts.board.clone(),
            number_of_active_players: self.mcts.board.player_num_of_active(),
            depth_range: (0, 0),
            mcts_policy_entropy: 0.0,
            root_q_values: abs_win.to_values_vec(board_mapper.is_absolute(), player_index),
            best_action_q_value: abs_win.to_values_vec(board_mapper.is_absolute(), player_index),
        });

        self.game_length += 1;

        // Play the oracle's perfect move.
        let best_move = oracle_plan.best_move;

        self.mcts.board.play_move_mut_with_store(
            &best_move,
            &mut <B as Board>::MoveStore::default(),
            None,
        );

        // Check if the game is over.
        if self.mcts.board.is_terminal() {
            self.finalize_and_send_simulation(samples_tx, board_mapper, false);
            self.reset_mut(rng, settings, board_mapper);
            // After reset, oracle_takeover is already None, so we are done.
        } else {
            // The game is not over. Get a new plan from the oracle for the *next* turn.
            if let Some(oracle) = &self.oracle {
                self.oracle_takeover = oracle.probe(&self.mcts.board).ok();

                if self.oracle_takeover.is_none() {
                    tracing::warn!(
                        "Oracle takeover ended unexpectedly. Returning to MCTS for next turn. Pretty board: {}. Is terminal: {}",
                        self.mcts.board.fancy_debug(),
                        self.mcts.board.is_terminal()
                    );
                }
            }
        }
    }

    fn finalize_and_send_simulation(
        &self,
        samples_tx: &UnboundedSender<Simulation<B>>,
        board_mapper: &impl MetaBoardMapper<B>,
        forced_end: bool, // For games that hit max_game_length
    ) {
        // If no samples were collected (e.g., all turns were fast searches or game was too short),
        // there's nothing to send.
        if self.samples.is_empty() {
            return;
        }

        let total_moves = self.get_position_count() as u32;

        // Determine the absolute outcome of the game.
        // This logic is identical to your original `advance` function's terminal state handling.

        // First, check if the board provides a heuristic value for early-terminated games.
        let heuristic_values = if forced_end || self.mcts.board.can_game_end_early() {
            self.mcts.board.get_heuristic_vector()
        } else {
            None
        };

        let values_abs = if let Some(heuristic_values) = heuristic_values {
            // Use the heuristic value provided by the board.
            ValuesAbs::<N> {
                value_abs: heuristic_values
                    .try_into()
                    .expect("Heuristic values length mismatch"),
                moves_left: 0.0,
            }
        } else {
            // No heuristic, so the game must have a definitive outcome (win/loss/draw).
            let outcome = self
                .mcts
                .board
                .outcome(forced_end)
                .expect("Board should be terminal or have a heuristic if ending early");

            ValuesAbs::<N>::from_outcome(outcome, 0.0, 0.0)
        };

        let mut num_of_players_ranges: Vec<PlayerRanges> = vec![];

        // Find the move ranges for phases with different numbers of players (e.g., 3-player phase, 2-player phase)
        for n_players in (2..=self.mcts.board.player_num()).rev() {
            let start_move = self
                .samples
                .iter()
                .find(|s| s.number_of_active_players == n_players)
                .map(|s| s.current_move_count as usize);

            let end_move = self
                .samples
                .iter()
                .rev()
                .find(|s| s.number_of_active_players == n_players)
                .map(|s| s.current_move_count as usize);

            if let (Some(start), Some(end)) = (start_move, end_move) {
                num_of_players_ranges.push(PlayerRanges {
                    num_of_players: n_players,
                    start_move_count: start,
                    end_move_count: end,
                });
            }
        }

        // Now, create a FinalizedSample for each Sample collected during the game.
        let finalized_samples: Vec<FinalizedSample<B>> = self
            .samples
            .iter()
            .map(|sample| {
                // The `z_values` are the final outcome of the game from the perspective
                // of the player at that specific point in the game.
                let z_target = if board_mapper.is_absolute() {
                    // For absolute mappers, the z-value is always the final absolute outcome.
                    values_abs.value_abs.as_slice().to_vec()
                } else {
                    // For POV mappers, we convert the absolute outcome to the player's perspective.
                    let pov_values = values_abs.pov(sample.player_index);
                    pov_values.value_pov.as_slice().to_vec()
                };

                let outcome_vector = self.learning_target_modifier.modify_target(
                    &sample.board,
                    &z_target,
                    &sample.root_q_values,
                    board_mapper.is_absolute(),
                    sample.current_move_count,
                    &num_of_players_ranges,
                );

                FinalizedSample {
                    inner: sample.clone(),
                    z_values: outcome_vector,
                    moves_left: total_moves.saturating_sub(sample.current_move_count),
                }
            })
            .collect();

        // Finally, package everything into a Simulation object.
        let simulation = Simulation {
            samples: finalized_samples,
            actual_game_length: total_moves,
            outcome: Box::new(values_abs),
        };

        // Send the completed simulation to the collector thread.
        // If this fails, it's a critical error, as the collector has likely panicked.
        if samples_tx.send(simulation).is_err() {
            tracing::error!(
                "Failed to send final simulation to collector. The collector task may have panicked."
            );
        };
    }
}
