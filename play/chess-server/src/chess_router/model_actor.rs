use crate::chess_router::chess_routes_dto::{ChessBoardDto, ChessMoveWrapperDto};
use crate::chess_router::dto_traits::{ApiBoard, ApiMove};
use crate::chess_router::tree_serializer;
use crate::chess_router::tree_serializer::ApiNode;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{MoveType, PseudoLegalMove};
use game_tri_chess::pos::MemoryPos;
use maz_core::mapping::chess_domain_canonical_mapper::{pretty_print_domain_chess_tensor, ChessDomainMapper};
use maz_core::mapping::{Board, InputMapper, MetaBoardMapper, PolicyMapper};
use maz_util::network::{auto_non_cpu_model, batcher_from, prediction};
use ndarray::Array2;
use rand::prelude::StdRng;
use rand::SeedableRng;
use schemars::JsonSchema;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tracing::info;
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};

pub enum ChessModelActorRequest {
    Think(ThinkRequestChess),
    Policy(PolicyRequestChess),
}

pub struct ThinkRequestChess {
    pub board: TriHexChess,
    pub num_of_rollouts: u64,
    pub exploration_factor: f32,
    pub contempt: f32,
    pub virtual_loss_weight: f32,
    pub responder: oneshot::Sender<ThinkResult<ChessMoveWrapperDto, ChessBoardDto>>,
}

pub struct PolicyRequestChess {
    pub board: TriHexChess,
    pub number_of_moves: usize,
    pub responder: oneshot::Sender<Vec<ChessMoveWrapperDto>>,
}

pub async fn model_actor_chess(mut rx: mpsc::Receiver<ChessModelActorRequest>) {
    let start_time = std::time::Instant::now();

    // r1bqkb1r/ppp2ppp/3p4/4p2n/X/8/8/8/5n2 X/r2qkbnr/p1p1ppp1/3p4/np5p/X X/X/r1bqkbnr/pppp2p1/5p2/n3p2p B qkqkqk -2- 0 5
    let mut model_new = auto_non_cpu_model("./testing/play/ChessDomain32/ChessDomain_850.onnx", Some(0))
    // let mut model_new = auto_non_cpu_model("./testing/play/ChessDomain32/ChessDomain_249.onnx", Some(0))
    // let mut model_new = auto_non_cpu_model("./testing/play/v2/ChessDomain_435.onnx", Some(0))
        .expect("Failed to load model");

    let mapper = ChessDomainMapper;

    // r3k1r1/1p3p1p/1bnppp2/p7/X/8/5q2/7b/8 X/r5nr/1p2p1k1/1qpp2b1/p4p1p/X X/X/X W ------ --- 0 1

    info!(
        "Model Actor: Model loaded successfully in {:?}",
        start_time.elapsed()
    );

    while let Some(outer) = rx.recv().await {
        match outer {
            ChessModelActorRequest::Think(request) => {
                let think_result = think_about_position(
                    &mut model_new,
                    request.board,
                    request.num_of_rollouts,
                    request.exploration_factor,
                    request.contempt,
                    request.virtual_loss_weight,
                    &mapper,
                );

                // If client disconnects, it's okay to fail silently.
                let _ = request.responder.send(think_result);
            }
            ChessModelActorRequest::Policy(request) => {
                // process a single request and send back top N moves by policy
                // dont do any search, just run the model once
                let mut batcher = batcher_from(&model_new, &mapper);
                let mut input = Array2::from_elem(mapper.input_board_shape(), false);

                mapper.encode_input(&mut input.view_mut(), &request.board);

                info!("Pretty: \n{}", pretty_print_domain_chess_tensor(&input));

                batcher.get_mut_item(0).assign(&input);
                let (policy_data, _value_data) = prediction(&mut model_new, &batcher);
                let policy_data = &policy_data[0..mapper.policy_len()];

                let mut move_probs: Vec<(usize, f32)> = policy_data
                    .iter()
                    .enumerate()
                    .map(|(i, &prob)| (i, prob.to_f32()))
                    .collect();

                move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let offset = request.board.get_turn().unwrap().get_offset();

                let top_moves = move_probs
                    .into_iter()
                    .take(request.number_of_moves)
                    .map(|(i, prob)| {
                        let mv = PseudoLegalMove {
                            from: MemoryPos((i / 96) as u8).to_global(offset),
                            to: MemoryPos(((i % 96)) as u8).to_global(offset),
                            move_type: MoveType::Move,
                        };

                        assert_eq!(
                            mapper.move_to_index(request.board.get_turn().unwrap(), mv),
                            i,
                            "Mapping error between move and index"
                        );

                        ApiMove::new_from_internal(mv, 0.0, prob)
                    })
                    .collect::<Vec<_>>();

                let _ = request.responder.send(top_moves);
            }
        }
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
pub struct ThinkResult<M, AP> {
    root_board: AP,
    root_position_eval: Vec<f32>,
    tree: ApiNode,
    duration_ms: u64,
    moves: Vec<M>,
}

// 7k/8/7q/8/X/X X/X/X 8/8/8/3q4/X/4k3/8/8/8 W qkqkqk --- 0 1

pub fn think_about_position<B: Board, MB: MetaBoardMapper<B>, M: ApiMove<B>, AP: ApiBoard<B>>(
    model: &mut ort::session::Session,
    board: B,
    num_of_rollouts: u64,
    exploration_factor: f32,
    contempt: f32,
    virtual_loss_weight: f32,
    mapper: &MB,
) -> ThinkResult<M, AP> {
    let start_time = std::time::Instant::now();

    let mut batcher = batcher_from(&model, mapper);

    let mut settings = SearchSettings::default();

    settings.part_cpuct_exploration = exploration_factor;

    let mut rng = StdRng::from_os_rng();

    let mut steppable_mcts =
        SteppableMCTS::<_, 3>::new_with_capacity(&board.clone(), mapper, num_of_rollouts, true);

    let mut requests: Vec<ProduceOutput<B>> = Vec::with_capacity(batcher.get_batch_size());

    let mut is_first_round = true;

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
            ApiMove::new_from_internal(
                last_move,
                node.complete_visits as f32 / root_visits as f32,
                node.net_policy,
            )
        })
        .collect::<Vec<_>>();

    let api_tree = tree_serializer::build_api_tree(
        &steppable_mcts.tree,
        10, // Max depth to prevent huge responses
        5,  // Max children per node (sorted by visits)
    );

    ThinkResult {
        root_board: AP::new_from_internal(&board),
        root_position_eval: steppable_mcts.tree.root_q_abs_values().value_abs.to_vec(),
        tree: api_tree.unwrap(),
        moves,
        duration_ms: search_duration.as_millis() as u64,
    }
}
