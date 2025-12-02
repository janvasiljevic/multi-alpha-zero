use crate::server::ServerImpl;
use api_autogen::models::Move;
use chess_tourney_server::OnnxModelConfig;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, PseudoLegalMove};
use log::warn;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::{Board, MetaBoardMapper};
use maz_util::network::{auto_non_cpu_model, batcher_from, prediction};
use ndarray::Array2;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};

pub enum ChessModelActorRequest {
    Think(ThinkRequestChess),
}

pub struct ThinkRequestChess {
    pub board: TriHexChess,
    pub num_of_rollouts: u64,
    pub exploration_factor: f32,
    pub contempt: f32,
    pub virtual_loss_weight: f32,
    pub responder: oneshot::Sender<ThinkResult>,
}

pub struct MoveWithStats {
    pub inner: PseudoLegalMove,
    pub prior: f32,
    pub confidence: f32,
}

pub struct ThinkResult {
    pub root_board: TriHexChess,
    pub root_position_eval: Vec<f32>,
    pub duration_ms: u64,
    pub moves: Vec<MoveWithStats>,
}

pub struct ModelHandle {
    pub tx: mpsc::Sender<ChessModelActorRequest>,
}

pub struct ModelService {
    pub models: HashMap<String, ModelHandle>,
    pub fallback: ModelHandle,
}

impl ModelService {
    pub fn new(models: &Vec<OnnxModelConfig>) -> anyhow::Result<Self> {
        let mut result = HashMap::new();

        // Spawn dummy actor
        let (tx_dummy, rx_dummy) = mpsc::channel(32);
        tokio::spawn(async move {
            dummy_model_actor(rx_dummy).await;
        });

        let fallback = ModelHandle { tx: tx_dummy };

        // Build real model actors
        for cfg in models {
            let file_path = cfg.file_path.clone();
            let key = cfg.key.clone();

            if !std::path::Path::new(&file_path).exists() {
                warn!("Model file missing, skipping: {}", file_path);
                continue;
            }

            let session = auto_non_cpu_model(&file_path, Some(0))
                .map_err(|e| anyhow::anyhow!("Failed to load model {}: {}", file_path, e))?;

            let (tx, rx) = mpsc::channel(32);

            let session_clone = session; // moved into actor

            tokio::spawn(async move {
                model_actor_chess(rx, session_clone).await;
            });

            result.insert(key, ModelHandle { tx });
        }

        Ok(Self {
            models: result,
            fallback,
        })
    }

    pub fn get_all_model_keys(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn get_model(&self, key: &str) -> &ModelHandle {
        self.models.get(key).unwrap_or(&self.fallback)
    }
}

pub async fn model_actor_chess(
    mut rx: mpsc::Receiver<ChessModelActorRequest>,
    mut model: ort::session::Session,
) {
    let mapper = ChessDomainMapper;

    while let Some(outer) = rx.recv().await {
        match outer {
            ChessModelActorRequest::Think(request) => {
                let result = think_about_position(
                    &mut model,
                    request.board,
                    request.num_of_rollouts,
                    request.exploration_factor,
                    request.contempt,
                    request.virtual_loss_weight,
                    &mapper,
                );

                let _ = request.responder.send(result);
            }
        }
    }
}

pub fn think_about_position<MB: MetaBoardMapper<TriHexChess>>(
    model: &mut ort::session::Session,
    board: TriHexChess,
    num_of_rollouts: u64,
    exploration_factor: f32,
    contempt: f32,
    virtual_loss_weight: f32,
    mapper: &MB,
) -> ThinkResult {
    let start_time = std::time::Instant::now();

    let mut batcher = batcher_from(&model, mapper);

    let mut settings = SearchSettings::default();

    settings.part_cpuct_exploration = exploration_factor;

    let mut rng = StdRng::from_os_rng();

    let mut steppable_mcts =
        SteppableMCTS::<_, 3>::new_with_capacity(&board.clone(), mapper, num_of_rollouts, true);

    let mut requests: Vec<ProduceOutput<TriHexChess>> = Vec::with_capacity(batcher.get_batch_size());

    let mut is_first_round = true;

    let mut terminator = 0;


    let start_time = Instant::now();

    loop {
        if let Some(request) = steppable_mcts.step(
            &mut rng,
            settings.search_fpu_root,
            settings.search_fpu_child,
            virtual_loss_weight,
            settings.weights,
            mapper,
            contempt,
        ) {
            if is_first_round {
                // preety print the input tensor
                let mut input = Array2::from_elem(mapper.input_board_shape(), false);
                mapper.encode_input(&mut input.view_mut(), &request.board);
                // info!("Input tensor:\n{}", pretty_print_extended_chess_tensor(&input));
            }
            mapper.encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
            requests.push(request);
        }

        if requests.len() >= batcher.get_batch_size() || (num_of_rollouts == 0) {
            is_first_round = false;
            let (policy_data, value_data) = prediction(model, &batcher);

            for (i, request) in requests.iter().enumerate() {
                let policy_data =
                    policy_data[i * mapper.policy_len()..(i + 1) * mapper.policy_len()].as_ref();

                let value_data = value_data[i * 3..(i + 1) * 3].as_ref();

                steppable_mcts
                    .consume(
                        &mut rng,
                        &settings,
                        request.node_id,
                        ConsumeValues::ConsumeWithOptionalCallback {
                            values_net: value_data,
                            policy_net_f16: policy_data,
                            callback: None,
                        },
                        false,
                    )
                    .expect("Failed to consume values");
            }

            requests.clear();

            if steppable_mcts.is_enough_rollouts() {
                break;
            }
        }

        terminator += 1;

        if start_time.elapsed().as_secs() > 10 {
            warn!("Terminating MCTS early due to timeout (stuck loop)");
            break;
        }

        if terminator > num_of_rollouts * 2 {
            warn!("Terminating MCTS early due to excessive iterations");
            break;
        }
    }

    let search_duration = start_time.elapsed();

    let best_child = steppable_mcts
        .tree
        .best_child(0)
        .expect("Expected at least one child in the root node");
    let best_move = steppable_mcts.tree[best_child]
        .last_move
        .expect("Expected a last move in the best child node");

    info!(
        "Finished MCTS with {} rollouts, but {} root visits. Search took {:?}. Best move: {:?}",
        num_of_rollouts,
        steppable_mcts.tree.root_visits(),
        search_duration,
        best_move
    );

    // Sanity check!
    assert_eq!(steppable_mcts.tree.root_virtual_visits(), 0u64);

    let root_nodes = steppable_mcts.tree.root_nodes();
    let root_visits = steppable_mcts.tree.root_visits();

    // sort root nodes by visits descending
    let mut root_nodes = root_nodes.into_iter().collect::<Vec<_>>();

    root_nodes.sort_by(|a, b| {
        // first compare by complete visits, then by policy
        b.complete_visits
            .cmp(&a.complete_visits)
            .then_with(|| b.net_policy.partial_cmp(&a.net_policy).unwrap())
    });

    let moves = root_nodes
        .into_iter()
        .map(|node| {
            let last_move = node.last_move.unwrap();
            MoveWithStats {
                inner: last_move,
                prior: node.net_policy,
                confidence: node.complete_visits as f32 / root_visits as f32,
            }
        })
        .collect::<Vec<_>>();

    ThinkResult {
        root_board: board,
        root_position_eval: steppable_mcts.tree.root_q_abs_values().value_abs.to_vec(),
        moves,
        duration_ms: search_duration.as_millis() as u64,
    }
}

pub async fn dummy_model_actor(mut rx: mpsc::Receiver<ChessModelActorRequest>) {
    while let Some(msg) = rx.recv().await {
        match msg {
            ChessModelActorRequest::Think(mut req) => {
                // Pick a random pseudo-legal move from the board
                let mut rng = rand::rng();

                let mut move_store = ChessMoveStore::default();

                req.board.fill_move_store(&mut move_store);

                let pick = if move_store.len() == 0 {
                    None
                } else {
                    move_store.get(rng.random_range(0..move_store.len()))
                };

                let result = ThinkResult {
                    root_board: req.board.clone(),
                    root_position_eval: vec![0.0, 0.0, 0.0],
                    duration_ms: 5,
                    moves: vec![MoveWithStats {
                        inner: pick.expect("No legal moves available"),
                        prior: 1.0,
                        confidence: 0.5,
                    }],
                };

                let _ = req.responder.send(result);
            }
        }
    }
}
