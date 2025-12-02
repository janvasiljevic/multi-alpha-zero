use crate::arena_config::{ArenaConfig, PlayerConfig};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::{HexGame, HexPlayer};
use log::warn;
use maz_core::mapping::hex_wrapper_mapper::HexWrapperMapper;
use maz_core::mapping::{Board, InputMapper, Outcome, PolicyMapper};
use maz_core::values_const::ValuesAbs;
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};
use maz_util::network::{batcher_from, prediction};
use rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use std::sync::Mutex;
use std::time::Instant;

pub trait HexArenaPlayer {
    /// Returns the name of the player.
    fn name(&self) -> &str;

    /// Determines the best move for the given board state.
    fn get_best_move(&self, board: &HexGame, config: &ArenaConfig) -> AxialCoord;
}

/// An enum to dispatch to different concrete player implementations.
/// This allows us to have a heterogeneous collection of players.
pub enum HexInnerPlayer {
    MctsModel(AlphaZeroPlayer),
    Random(RandomPlayer),
    Mcts(MctsPlayer),
}

impl HexArenaPlayer for HexInnerPlayer {
    fn name(&self) -> &str {
        match self {
            HexInnerPlayer::MctsModel(p) => p.name(),
            HexInnerPlayer::Random(p) => p.name(),
            HexInnerPlayer::Mcts(p) => p.name(),
        }
    }

    fn get_best_move(&self, board: &HexGame, config: &ArenaConfig) -> AxialCoord {
        match self {
            HexInnerPlayer::MctsModel(p) => p.get_best_move(board, config),
            HexInnerPlayer::Random(p) => p.get_best_move(board, config),
            HexInnerPlayer::Mcts(p) => p.get_best_move(board, config),
        }
    }
}

pub struct RandomPlayer {
    name: String,
    rng: Mutex<StdRng>,
}

impl RandomPlayer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            rng: Mutex::new(StdRng::from_os_rng()),
        }
    }
}

impl HexArenaPlayer for RandomPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_best_move(&self, board: &HexGame, _config: &ArenaConfig) -> AxialCoord {
        let mut move_store = <HexGame as Board>::MoveStore::default();
        let mut board = board.clone();
        board.fill_move_store(&mut move_store);
        let mut rng = self.rng.lock().unwrap();

        // `choose` will panic if `move_store` is empty, but this function
        // should only be called on non-terminal boards.
        *move_store.iter().choose(&mut *rng).unwrap()
    }
}

pub struct MctsPlayer {
    name: String,
    rng: Mutex<StdRng>,
    num_of_simulations: usize,
}

#[derive(Debug, Clone)]
struct MctsNode {
    last_move: Option<AxialCoord>,
    parent: Option<usize>,
    children: Vec<usize>,
    values: ValuesAbs<3>,
    visits: u32,
    unexplored_moves: Vec<AxialCoord>,
}

impl MctsPlayer {
    pub(crate) fn new(name: String, num_of_simulations: usize) -> Self {
        Self {
            name,
            rng: Mutex::new(StdRng::from_os_rng()),
            num_of_simulations,
        }
    }

    /// Performs a random playout from the given board state.
    fn run_playout(&self, board: &HexGame, rng: &mut StdRng) -> Outcome {
        let mut temp_board = board.clone();

        let mut move_store = <HexGame as Board>::MoveStore::default();

        while !temp_board.is_terminal() {
            temp_board.fill_move_store(&mut move_store);
            let random_move = move_store.iter().choose(rng).unwrap().clone();

            temp_board.play_move_mut_with_store(&random_move, &mut move_store, None);
        }
        temp_board.outcome(false).unwrap()
    }
}

impl HexArenaPlayer for MctsPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_best_move(&self, board: &HexGame, config: &ArenaConfig) -> AxialCoord {
        let mut rng_guard = self.rng.lock().unwrap();
        let mut rng = &mut *rng_guard;
        let mut board = board.clone();

        let mut move_store = <HexGame as Board>::MoveStore::default();
        board.fill_move_store(&mut move_store);

        let mut tree = Vec::new();
        let root_node = MctsNode {
            last_move: None,
            parent: None,
            children: Vec::new(),
            values: ValuesAbs::default(),
            visits: 0,
            unexplored_moves: move_store.iter().cloned().collect(),
        };
        tree.push(root_node);

        let start_time = Instant::now();
        let time_limit = std::time::Duration::from_secs(30);

        for _ in 0..config.mcts_rollouts_per_move {
            if start_time.elapsed() > time_limit {
                warn!("MCTS for player '{}' reached time limit", self.name);
                break;
            }

            let mut current_node_idx = 0;
            let mut current_board = board.clone();
            let mut traversal_move_store = move_store.clone();
            let mut path = vec![0];

            // 1. Selection
            while tree[current_node_idx].unexplored_moves.is_empty()
                && !tree[current_node_idx].children.is_empty()
            {
                let current_node = &tree[current_node_idx];
                let current_player_idx = current_board.player_current() as usize;

                let best_child = current_node
                    .children
                    .iter()
                    .max_by(|a, b| {
                        let ucb1 = |node_idx: &usize| {
                            let node = &tree[*node_idx];
                            if node.visits == 0 {
                                f32::INFINITY
                            } else {
                                let exploitation =
                                    node.values.value_abs[current_player_idx] / node.visits as f32;
                                let exploration = (2.0 * (current_node.visits as f32).ln()
                                    / node.visits as f32)
                                    .sqrt();
                                exploitation + exploration
                            }
                        };
                        ucb1(a).partial_cmp(&ucb1(b)).unwrap()
                    })
                    .unwrap();

                current_node_idx = *best_child;
                current_board.play_move_mut_with_store(
                    &tree[current_node_idx].last_move.unwrap(),
                    &mut traversal_move_store,
                    None,
                );
                path.push(current_node_idx);
            }

            // 2. Expansion
            if !current_board.is_terminal() && !tree[current_node_idx].unexplored_moves.is_empty() {
                let mv_to_explore = tree[current_node_idx]
                    .unexplored_moves
                    .pop()
                    .expect("Checked not empty");

                current_board.play_move_mut_with_store(
                    &mv_to_explore,
                    &mut traversal_move_store,
                    None,
                );

                let new_node = MctsNode {
                    last_move: Some(mv_to_explore),
                    parent: Some(current_node_idx),
                    children: Vec::new(),
                    values: ValuesAbs::default(),
                    visits: 0,
                    unexplored_moves: if current_board.is_terminal() {
                        vec![]
                    } else {
                        traversal_move_store.iter().cloned().collect()
                    },
                };

                let new_node_idx = tree.len();
                tree.push(new_node);
                tree[current_node_idx].children.push(new_node_idx);
                current_node_idx = new_node_idx;
                path.push(current_node_idx);
            }

            // 3. Simulation
            let outcome = if current_board.is_terminal() {
                current_board.outcome(false).unwrap()
            } else {
                self.run_playout(&current_board, &mut rng)
            };

            let outcome_values = ValuesAbs::<3>::from_outcome(outcome, 0.0, 0.0);

            for node_idx in path {
                tree[node_idx].visits += 1;
                tree[node_idx].values += &outcome_values;
            }
        }

        tree[0]
            .children
            .iter()
            .max_by_key(|c| tree[**c].visits)
            .map(|c| tree[*c].last_move.unwrap())
            .expect("Root node must have children after rollouts")
    }
}

pub struct AlphaZeroPlayer {
    pub(crate) name: String,
    pub(crate) model: Mutex<ort::session::Session>,
    pub(crate) mapper: HexWrapperMapper,
}

impl HexArenaPlayer for AlphaZeroPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_best_move(&self, board: &HexGame, config: &ArenaConfig) -> AxialCoord {
        // This logic is moved directly from your original `get_best_move` function.
        let num_of_rollouts = config.mcts_rollouts_per_move as u64;
        let settings = SearchSettings::default();
        let mut rng = StdRng::from_os_rng();

        let mut steppable_mcts =
            SteppableMCTS::<_, 3>::new_with_capacity(board, &self.mapper, num_of_rollouts, true);

        let model_guard = self.model.lock().unwrap();
        let mut batcher = batcher_from(&*model_guard, &self.mapper);
        drop(model_guard);

        let mut requests: Vec<ProduceOutput<HexGame>> =
            Vec::with_capacity(batcher.get_batch_size());
        let mut i = 0;
        let virtual_size_batch = 4;

        loop {
            if let Some(request) = steppable_mcts.step(
                &mut rng,
                settings.search_fpu_root,
                settings.search_fpu_child,
                settings.search_virtual_loss_weight,
                settings.weights,
                &self.mapper,
                settings.contempt,
            ) {
                self.mapper
                    .encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
                requests.push(request);
            }

            i += 1;

            if requests.len() >= batcher.get_batch_size() || i >= virtual_size_batch {
                let mut model_guard = self.model.lock().unwrap();
                let (policy_data, value_data) = prediction(&mut *model_guard, &batcher);

                drop(model_guard);

                for (i, request) in requests.iter().enumerate() {
                    let policy_net_f16 = &policy_data
                        [i * self.mapper.policy_len()..(i + 1) * self.mapper.policy_len()];
                    let values_net = &value_data[i * 3..(i + 1) * 3];

                    steppable_mcts
                        .consume(
                            &mut rng,
                            &settings,
                            request.node_id,
                            ConsumeValues::ConsumeWithOptionalCallback {
                                policy_net_f16,
                                values_net,
                                callback: None,
                            },
                            false,
                        )
                        .expect("Impossible to error out here, we aren't taking in Cache results.");
                }
                requests.clear();
                i = 0;

                if steppable_mcts.is_enough_rollouts() {
                    break;
                }
            }
        }

        let best_child_id = steppable_mcts
            .tree
            .best_child(0)
            .expect("MCTS root should have at least one child after rollouts");

        steppable_mcts.tree[best_child_id]
            .last_move
            .expect("Best child node must have a move associated with it")
    }
}
