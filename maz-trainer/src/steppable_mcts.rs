use crate::node::{FpuMode, Node, UctWeights};
use crate::range::IdxRange;
use crate::search_settings::SearchSettings;
use crate::self_play_game::ProduceOutput;
use crate::shared_cache::NetCachedEvaluation;
use crate::stable_dirichlet::StableDirichletFast;
use crate::tree::{ROOT_NODE_ID, Tree};
use decorum::R32;
use half::f16;
use log::warn;
use maz_core::mapping::{Board, MetaBoardMapper};
use maz_core::mapping::{BoardMapper, MoveStore};
use maz_core::values_const::{ValuesAbs, ValuesPov};
use maz_util::iter::choose_max_by_key;
use num_traits::FromPrimitive;
use std::cmp::Reverse;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct SteppableMCTS<B: Board, const N: usize> {
    pub board: B,
    pub tree: Tree<B, N>,

    /// Minimum number of rollouts to consider the search finished.
    pub min_number_of_rollouts: u64,

    /// This is more metadata that performs additional assertions and checks.
    pub using_virtual_loss: bool,

    /// Do we use absolute state representation or canonical?
    /// This affects how we interpret the network values and how they are propagated.
    is_absolute: bool,

    /// Reusable move store for the board.
    move_store: B::MoveStore,

    /// Reusable buffer for Dirichlet noise sampling.
    dirichlet_noise_buffer: Vec<f32>,
}

pub enum ConsumeValues<'a> {
    ReadFromCache(&'a NetCachedEvaluation),
    ConsumeWithOptionalCallback {
        policy_net_f16: &'a [f16],
        values_net: &'a [f32],
        callback: Option<Box<dyn FnOnce(NetCachedEvaluation) -> () + 'a>>,
    },
}

#[derive(Debug, Error)]
pub enum DirichletError {
    #[error("Node {node_id} hasn't been expanded yet, can't apply Dirichlet noise")]
    ChildrenNotExpanded { node_id: usize },

    #[error("Node {node_id} has zero children - an illegal state / problem with move gen.?")]
    ZeroChildren { node_id: usize },
}

#[derive(Debug)]
pub struct CacheShapeMissmatch {
    pub expected_len: u16,
}

impl<B: Board, const N: usize> SteppableMCTS<B, N> {
    pub fn new_with_capacity(
        game: &B,
        mapper: &impl MetaBoardMapper<B>,
        number_of_rollouts: u64,
        using_virtual_loss: bool,
    ) -> Self {
        Self {
            board: game.clone(),
            tree: Tree::new_with_capacity(
                game.clone(),
                number_of_rollouts as usize * mapper.average_number_of_moves(),
            ),
            min_number_of_rollouts: number_of_rollouts,
            is_absolute: mapper.is_absolute(),
            using_virtual_loss,
            move_store: <B as Board>::MoveStore::default(),
            dirichlet_noise_buffer: Vec::new(),
        }
    }
}

// Traverses back to the root, recording the path.
// Then it goes back down the path, printing the state of each node.
pub fn get_weird_state<B: Board, const N: usize>(
    mcts: &SteppableMCTS<B, N>,
    mapper: &impl BoardMapper<B>,
    node: usize,
) -> String {
    let mut path = Vec::new();
    let mut curr_index = node;
    let mut string = String::new();

    while let Some(parent) = mcts.tree[curr_index].parent {
        path.push(curr_index);
        curr_index = parent;
    }
    path.push(0); // add root
    path.reverse();

    let mut board = mcts.board.clone();
    let mut move_store = mcts.move_store.clone();

    move_store.clear();
    board.fill_move_store(&mut move_store);

    for &index in &path {
        let node = &mcts.tree[index];

        string.push_str(&format!(
            "> Node {}. Move: {:?} (index: {:?}). Visits: {}. Player: {:?}. Mapper idx->move: {:?} (index: {}).\nBoard:\n{}\n\n",
            index,
            node.last_move,
            node.last_move.map(|m| mapper.move_to_index(board.player_current(), m)),
            node.total_visits(),
            node.player_index
                .map(|p| B::Player::from(usize::from_u32(p as u32).unwrap())),
            mapper.index_to_move(
                &board,
                &move_store,
                node.last_move_policy_index,
            ),
            node.last_move_policy_index,
            board.fancy_debug()
        ));

        if let Some(mv) = &node.last_move {
            board.play_move_mut_with_store(mv, &mut move_store, None);
        }
    }

    string
}

impl<B: Board, const N: usize> SteppableMCTS<B, N> {
    pub fn step(
        &mut self,
        rng: &mut impl rand::Rng,
        fpu_root: FpuMode,
        fpu_child: FpuMode,
        virtual_loss_weight: f32,
        weights: UctWeights,
        board_mapper: &impl BoardMapper<B>,
        contempt: f32,
    ) -> Option<ProduceOutput<B>>
    where
        <B as Board>::Player: From<usize>,
    {
        assert!(!self.is_finished(), "Game is already finished");

        let mut curr_node = 0;
        let mut curr_board = self.tree.root_board().clone();

        self.move_store.clear();

        loop {
            // count each node as visited
            self.tree[curr_node].virtual_visits += 1;

            let current_player = curr_board.player_current();

            // if the board is done backpropagate the real value
            // DON'T apply current player logic here, outcome is abstract
            if let Some(outcome) = curr_board.outcome(false) {
                tree_propagate_values(
                    &mut self.tree,
                    curr_node,
                    ValuesAbs::<N>::from_outcome(outcome, 0.0, contempt),
                );

                return None;
            }

            // Either:
            // - Expand the node with legal moves and generate the GPU Request
            // - Get children if the node has already been expanded
            let children = match self.tree[curr_node].children {
                None => {
                    // Move store should be filled up with `play_move_mut_with_store`
                    // except for the root node.
                    if self.move_store.is_empty() {
                        curr_board.fill_move_store(&mut self.move_store);
                    }

                    debug_assert!(
                        !self.move_store.is_empty(),
                        "Node has no children and the move store is empty - this should not happen unless the board is done. Debug travel:\n{}\n",
                        get_weird_state(self, board_mapper, curr_node)
                    );

                    // So this node is being expanded for the first time.
                    // We need to set the player index for the node, so when we get the
                    // inference result we can correctly apply the values.
                    self.tree[curr_node].player_index = Some(current_player.into());

                    // Initialize the children with uniform policy
                    // There are as many children as there are legal moves
                    // Since tree is a Vec we just count how many new nodes we will add
                    // and then create an IdxRange for them
                    let mv_count = self.move_store.len();
                    let p = 1.0 / mv_count as f32;

                    let start = self.tree.len();

                    self.move_store.iter().for_each(|mv| {
                        let policy_index = board_mapper.move_to_index(current_player, mv);

                        let node = Node::new(Some(curr_node), Some(mv), policy_index, p, None);

                        self.tree.nodes.push(node);
                    });

                    let end = self.tree.len();

                    self.tree[curr_node].children = Some(IdxRange::new(start, end));
                    self.tree[curr_node].net_values = None;

                    // Return the board position we want to evaluate
                    return Some(ProduceOutput {
                        board: curr_board,
                        node_id: curr_node,
                    });
                }
                Some(children) => children,
            };

            debug_assert!(
                self.tree[curr_node].player_index.unwrap() == current_player.into(),
                "Node player index does not match current board player. Debug travel:{}\n",
                get_weird_state(self, board_mapper, curr_node)
            );

            #[cfg(debug_assertions)]
            {
                let mut temp_move_store = <B as Board>::MoveStore::default();

                curr_board.fill_move_store(&mut temp_move_store);

                debug_assert_eq!(
                    children.length as usize,
                    temp_move_store.len(),
                    "Children length ({}) must match move store length ({}). Debug board:\n{}. Children differences: {:?}\n",
                    children.length,
                    temp_move_store.len(),
                    curr_board.fancy_debug(),
                    temp_move_store
                        .iter()
                        .filter(|mv| !self.tree[curr_node]
                            .children
                            .unwrap()
                            .iter()
                            .any(|c| self.tree[c].last_move == Some(*mv)))
                        .collect::<Vec<_>>()
                );
            }

            let selected = if self.tree[curr_node].complete_visits == 0 {
                // This branch only gets executed if the node has not been 'expanded' by the network yet.
                // This can only happen if we are using virtual loss (batching requests together for the GPU)
                // and that the node is in 'pending' state.
                // This means we have no real information about the node yet, so we just pick
                // a random child (but taking into account their virtual visits).
                assert!(self.using_virtual_loss && self.tree[curr_node].virtual_visits > 0);

                debug_assert!(
                    children.iter().all(|c| self.tree[c].complete_visits == 0),
                    "All children of an unvisited node should have 0 visits"
                );

                choose_max_by_key(
                    children,
                    |&child| Reverse(self.tree[child].total_visits()),
                    rng,
                )
            } else {
                // As said in "Accelerating Self-Play Learning in Go" footnotes 3
                // '...except C_fpu = 0 at the root if Dirichlet noise is used'
                let fpu_mode = if curr_node == 0 { fpu_root } else { fpu_child };
                let uct_context = self.tree.uct_context(curr_node);

                choose_max_by_key(
                    children,
                    |&child| -> Option<R32> {
                        let uct = self.tree[child]
                            .uct(&uct_context, fpu_mode, virtual_loss_weight, current_player)
                            .total(weights);
                        R32::from_f32(uct) // Must be 32 Non-NAN, because it implements Ord
                    },
                    rng,
                )
            };

            let selected_child =
                selected.expect("Board is not done, this node should have a child");

            curr_node = selected_child;

            // If the next node has children, we can use its player index as a hint
            // Giving this to the board implementation it can use some optimizations
            // to skip some internal calculations.
            let next_player_hint = match self.tree[curr_node].children {
                Some(_) => self.tree[curr_node].player_index.map(|p| p.into()),
                None => None,
            };

            curr_board.play_move_mut_with_store(
                &self.tree[curr_node].last_move.unwrap(),
                &mut self.move_store,
                next_player_hint,
            );
        }
    }

    pub fn consume(
        &mut self,
        rng: &mut impl rand::Rng,
        settings: &SearchSettings,
        node_id: usize,
        values: ConsumeValues,
        apply_dirichlet: bool,
    ) -> Result<(), CacheShapeMissmatch> {
        if !self.using_virtual_loss {
            assert!(
                self.tree[node_id].net_values.is_none(),
                "Node {node_id} was already evaluated by the network - this can't happen when not using virtual loss"
            );
        }

        assert!(
            self.tree[node_id].player_index.is_some(),
            "Node {node_id} should have a player index set before applying the network values"
        );

        let values_net = match values {
            ConsumeValues::ConsumeWithOptionalCallback { values_net, .. } => values_net,
            ConsumeValues::ReadFromCache(cached) => &cached.value,
        };

        // policy
        let children_indices = self.tree[node_id]
            .children
            .expect("Applied node should have initialized children");

        let is_root = node_id == ROOT_NODE_ID;

        match values {
            ConsumeValues::ConsumeWithOptionalCallback {
                policy_net_f16,
                values_net,
                mut callback,
            } => {
                let max_logit = children_indices
                    .iter()
                    .map(|child_id| {
                        policy_net_f16[self.tree[child_id].last_move_policy_index].to_f32()
                    })
                    .fold(f32::NEG_INFINITY, f32::max);

                let net_error = !max_logit.is_finite();

                if net_error {
                    warn!(
                        "Max logit before softmax was not finite, assigning uniform policy. This can happen because of instability in fp16 calculations."
                    );
                    panic!(
                        "Max logit before softmax was not finite, assigning uniform policy. This can happen because of instability in fp16 calculations."
                    );

                    // let uniform_policy = 1.0 / children_indices.length as f32;
                    // for child_id in children_indices.iter() {
                    //     self.tree[child_id].net_policy = uniform_policy;
                    // }
                } else {
                    // Calculate exp. values and their sum. Store directly in nodes to avoid extra allocations.
                    let mut policy_sum = 0.0;
                    for child_id in children_indices.iter() {
                        let node = &mut self.tree[child_id];
                        let logit = policy_net_f16[node.last_move_policy_index].to_f32();
                        let exp_val = (logit - max_logit).exp();

                        node.net_policy = exp_val; // Store un-normalized value
                        policy_sum += exp_val;
                    }

                    assert!(
                        policy_sum > 0.0,
                        "Softmax sum must be positive, was {policy_sum}. Policy logits: {:?}. Max logit: {}. Node.net_policy values: {:?}. Value head: {:?}",
                        policy_net_f16,
                        max_logit,
                        children_indices
                            .iter()
                            .map(|child_id| self.tree[child_id].net_policy)
                            .collect::<Vec<_>>(),
                        values_net
                    );
                    let inv_policy_sum = 1.0 / policy_sum;

                    for child_id in children_indices.iter() {
                        self.tree[child_id].net_policy *= inv_policy_sum;
                    }
                }

                // If the cache callback is provided, call it with the softmax policy and values.
                if let Some(cache_callback) = callback.take() {
                    let softmax_policy: Vec<f32> = children_indices
                        .iter()
                        .map(|child_id| self.tree[child_id].net_policy)
                        .collect();

                    let cached_eval = NetCachedEvaluation {
                        softmax_policy: softmax_policy.iter().map(|p| f16::from_f32(p)).collect(),
                        value: if net_error {
                            vec![0.0; N] // If the network produced NaNs, we store a zero value to avoid catastrophic cache errors.
                        } else {
                            values_net.to_vec()
                        },
                    };

                    cache_callback(cached_eval);
                }
            }
            ConsumeValues::ReadFromCache(cached_softmax) => {
                // This happens extremely rarely, but we can get a hash collision.
                // Prevent catastrophic behavior by checking the shape.
                if children_indices.length as usize != cached_softmax.softmax_policy.len() {
                    return Err(CacheShapeMissmatch {
                        expected_len: children_indices.length,
                    });
                }

                for (i, child_id) in children_indices.iter().enumerate() {
                    let node = &mut self.tree[child_id];
                    node.net_policy = cached_softmax.softmax_policy[i].to_f32();
                }
            }
        }

        // Next step is to apply temperature. Mathematically, this should be applied directly to
        // the logits before softmax, but then we would be submitting an already 'flattened' policy
        // to the cache. This maybe wouldn't be a big deal, but if policy_child >> policy_root,
        // and the cached result of a child gets applied to another root node of a different search,
        // the policy would be too flat / incorrect and could potentially bias the search.
        let temperature = if is_root {
            settings.search_policy_temperature_root
        } else {
            settings.search_policy_temperature_child
        };

        if temperature != 1.0 {
            // Apply temperature with powf and calculate the new sum for re-normalization.
            let inv_temp = 1.0 / temperature;

            let mut temp_scaled_sum = 0.0;

            for child_id in children_indices.iter() {
                let node = &mut self.tree[child_id];
                node.net_policy = node.net_policy.powf(inv_temp);
                temp_scaled_sum += node.net_policy;
            }

            // Re-normalize the temperature-scaled policies.
            if temp_scaled_sum > 0.0 {
                let inv_temp_scaled_sum = 1.0 / temp_scaled_sum;
                for child_id in children_indices.iter() {
                    self.tree[child_id].net_policy *= inv_temp_scaled_sum;
                }
            }
        }

        // Apply Dirichlet noise to the root node policy. Also applies to the root of the
        // retained subtree! This detail is skipped in the OG paper, so this is a bit of a guess.
        if is_root && apply_dirichlet {
            self.apply_dirichlet(rng, settings, node_id)
                .expect("When consuming, we should always have children expanded already");
        }

        // TODO: The network must generate moves left as well, so we can use it here.
        // Super important this is after the network values are applied,
        // because of potential Cache errors and then asserts...
        let values_abs: ValuesAbs<N> = if self.is_absolute {
            ValuesAbs::<N>::from_slice(values_net, 1.0)
        } else {
            let player = self.tree[node_id]
                .player_index
                .expect("Node should have a player index set before applying the network values");

            ValuesPov::<N>::from_slice(values_net, 1.0, player).abs()
        };

        self.tree[node_id].net_values = Some(values_abs);
        tree_propagate_values(&mut self.tree, node_id, values_abs);

        Ok(()) // TODO: Maybe we could return the cache here if requested? Instead of doing that callback madness
    }

    pub fn apply_dirichlet(
        &mut self,
        rng: &mut impl rand::Rng,
        settings: &SearchSettings,
        node_id: usize,
    ) -> Result<(), DirichletError> {
        let children_indices = match self.tree[node_id].children {
            Some(children) => children,
            None => return Err(DirichletError::ChildrenNotExpanded { node_id }),
        };

        let eps = settings.dirichlet_eps;
        let num_children = children_indices.length as usize;

        if num_children == 0 {
            return Err(DirichletError::ZeroChildren { node_id });
        }

        if num_children > 1 && eps > 0.0 {
            self.dirichlet_noise_buffer.resize(num_children, 0.0);
            let distribution =
                StableDirichletFast::new(settings.dirichlet_alpha, num_children).unwrap();
            distribution.sample_into(rng, &mut self.dirichlet_noise_buffer);

            // Apply noise and find the new sum for re-normalization.
            let mut final_sum = 0.0;
            for (i, child_id) in children_indices.iter().enumerate() {
                let node = &mut self.tree[child_id];
                let noise = self.dirichlet_noise_buffer[i];

                node.net_policy = (1.0 - eps) * node.net_policy + eps * noise;
                final_sum += node.net_policy;
            }

            // Re-normalize the final policy.
            if final_sum > 0.0 {
                let inv_final_sum = 1.0 / final_sum;
                for child_id in children_indices.iter() {
                    self.tree[child_id].net_policy *= inv_final_sum;
                }
            }
        }

        Ok(())
    }

    /// Return true if the tree has enough rollouts to be considered finished.
    pub fn is_enough_rollouts(&self) -> bool {
        self.tree.root_visits() >= self.min_number_of_rollouts
    }

    pub fn is_finished(&self) -> bool {
        self.board.is_terminal()
    }
}

fn tree_propagate_values<B: Board, const N: usize>(
    tree: &mut Tree<B, N>,
    node: usize,
    values: ValuesAbs<N>,
) {
    let mut curr_index = node;

    loop {
        let curr_node = &mut tree[curr_index];
        assert!(curr_node.virtual_visits > 0);

        curr_node.complete_visits += 1;
        curr_node.virtual_visits -= 1;
        curr_node.sum_values += &values;

        curr_index = match curr_node.parent {
            Some(parent) => parent,
            None => break,
        };
    }
}
