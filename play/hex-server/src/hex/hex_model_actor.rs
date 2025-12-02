use crate::hex::hex_routes_logic::BoardState;
use crate::hex::tree_serializer;
use crate::hex::tree_serializer::ApiNode;
use game_hex::game_hex::HexGame;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_core::mapping::MetaBoardMapper;
use maz_util::network::{auto_non_cpu_model, batcher_from, prediction};
use rand::prelude::StdRng;
use rand::SeedableRng;
use schemars::JsonSchema;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};

pub struct ThinkRequest {
    pub board: HexGame,
    pub num_of_rollouts: u64,
    // The sender part of a one-shot channel.
    pub responder: oneshot::Sender<ThinkResult>,
    pub key: String,
}

pub async fn model_actor(mut receiver: mpsc::Receiver<ThinkRequest>) {
    let start_time = std::time::Instant::now();

    let board = HexGame::new(5).expect("Failed to create Hex game");

    let mut model_new = auto_non_cpu_model("./testing/arena/Hex5Bert/Hex5Bert_140.onnx", Some(0))
        .expect("Failed to load model");

    let mapper_old = HexCanonicalMapper::new(&board);

    let mapper_new = HexCanonicalMapper::new(&board);

    info!(
        "Model Actor: Model loaded successfully in {:?}",
        start_time.elapsed()
    );

    while let Some(request) = receiver.recv().await {
        let think_result = match request.key.as_str() {
            "CanonicalHex4" => think_about_position(
                &mut model_new,
                request.board,
                request.num_of_rollouts,
                &mapper_old,
            ),
            "RelativeCanonicalHex4" => think_about_position(
                &mut model_new,
                request.board,
                request.num_of_rollouts,
                &mapper_new,
            ),
            _ => {
                panic!("Unknown model key: {}", request.key);
            }
        };

        // If client disconnects, it's okay to fail silently.
        let _ = request.responder.send(think_result);
    }
}

#[derive(Serialize, JsonSchema)]
pub struct ThinkResultItem {
    q: i32,
    r: i32,
    score: f32,              // 0 - 1 (based on percentage of visits)
    renormalized_score: f32, // 0 - 1 (based on which is the best move)
}

#[derive(Serialize, JsonSchema)]
pub struct ThinkResult {
    root_board: BoardState,
    moves: Vec<ThinkResultItem>,
    root_position_eval: Vec<f32>,
    tree: ApiNode,
    duration_ms: u64,
}

pub fn think_about_position(
    model: &mut ort::session::Session,
    board: HexGame,
    num_of_rollouts: u64,
    mapper: &impl MetaBoardMapper<HexGame>,
) -> ThinkResult {
    let start_time = std::time::Instant::now();

    let mut batcher = batcher_from(&model, mapper);

    let settings = SearchSettings::default();

    let mut rng = StdRng::from_os_rng();

    let mut steppable_mcts =
        SteppableMCTS::<_, 3>::new_with_capacity(&board.clone(), mapper, num_of_rollouts, true);

    let mut requests: Vec<ProduceOutput<HexGame>> = Vec::with_capacity(batcher.get_batch_size());

    loop {
        if let Some(request) = steppable_mcts.step(
            &mut rng,
            settings.search_fpu_root,
            settings.search_fpu_child,
            settings.search_virtual_loss_weight,
            settings.weights,
            mapper,
            0.0
        ) {
            mapper.encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
            requests.push(request);
        }

        if requests.len() >= batcher.get_batch_size() {
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
                    .expect("Cant fail here");
            }

            requests.clear();

            if steppable_mcts.is_enough_rollouts() {
                break;
            }
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
    let best_child_id = steppable_mcts.tree.best_child(0).unwrap();
    let best_child = &steppable_mcts.tree[best_child_id];
    let root_visits = steppable_mcts.tree.root_visits();

    info!("Root has visit count: {}", root_visits);

    let results = root_nodes
        .map(|node| {
            let last_move = node.last_move.unwrap();
            ThinkResultItem {
                q: last_move.q,
                r: last_move.r,
                score: (node.complete_visits as f32 / root_visits as f32),
                renormalized_score: node.complete_visits as f32 / best_child.complete_visits as f32,
            }
        })
        .collect::<Vec<_>>();

    let api_tree = tree_serializer::build_api_tree(
        &steppable_mcts.tree,
        10, // Max depth to prevent huge responses
        5,  // Max children per node (sorted by visits)
    );

    ThinkResult {
        root_board: BoardState::new_from_internal(board),
        moves: results,
        root_position_eval: steppable_mcts.tree.root_q_abs_values().value_abs.to_vec(),
        tree: api_tree.unwrap(),
        duration_ms: search_duration.as_millis() as u64,
    }
}
