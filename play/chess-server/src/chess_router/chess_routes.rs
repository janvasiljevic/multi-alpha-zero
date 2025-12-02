use crate::chess_router::chess_routes_dto::{ChessBoardDto, ChessMoveWrapperDto};
use crate::chess_router::model_actor::{
    ChessModelActorRequest, PolicyRequestChess, ThinkRequestChess, ThinkResult,
};
use crate::state::AppState;
use aide::axum::routing::post_with;
use aide::axum::{ApiRouter, IntoApiResponse};
use aide::transform::TransformOperation;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{http, Json};
use game_tri_chess::basics::{Piece, COLORS};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use http::StatusCode;
use maz_core::mapping::chess_extended_canonical_mapper::ChessExtendedCanonicalMapper;
use maz_core::mapping::{Board, PolicyMapper};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::oneshot;
use tracing::info;

fn empty_doc(op: TransformOperation) -> TransformOperation {
    op
}

pub fn chess_routes(state: AppState) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/think",
            post_with(post_think_about_position, |op| {
                op.response::<200, Json<ThinkResult<ChessMoveWrapperDto, ChessBoardDto>>>()
            }),
        )
        .api_route("/apply_moves", post_with(post_apply_moves, empty_doc))
        .api_route("/attack_maps", post_with(get_attack_maps, attack_maps_doc))
        .api_route("/policy_moves", post_with(post_policy_moves, policy_moves_doc))
        .with_state(state)
}

#[derive(Deserialize, JsonSchema)]
struct ThinkInput {
    board_state: String,
    number_of_rollouts: u64,
    exploration_factor: f32,
    contempt: f32,
    virtual_loss_weight: f32,
}

async fn post_think_about_position(
    State(state): State<AppState>,
    Json(think_input): Json<ThinkInput>,
) -> impl IntoApiResponse {
    let mut internal_board = match TriHexChess::new_with_fen(think_input.board_state.as_ref(), true)
    {
        Ok(board) => board,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "Invalid board state provided").into_response();
        }
    };

    if internal_board.state.turn_counter == 1 {
        let material_count = internal_board.material_count();

        for count in material_count {
            if count != 39 {
                internal_board.is_using_grace_period = false;

                info!(
                    "First move but material count is off: {:?} -> not using grace period",
                    material_count
                );

                break;
            }
        }
    }

    if internal_board.is_over() {
        return (
            StatusCode::BAD_REQUEST,
            "Cannot think about a terminal board",
        )
            .into_response();
    };

    let number_of_rollouts = think_input.number_of_rollouts;

    let (responder, receiver) = oneshot::channel();

    let request = ThinkRequestChess {
        board: internal_board,
        num_of_rollouts: number_of_rollouts,
        responder,
        exploration_factor: think_input.exploration_factor,
        contempt: think_input.contempt,
        virtual_loss_weight: think_input.virtual_loss_weight,
    };

    // The .send() operation is async and can fail if the receiver is dropped.
    if state
        .think_request_sender
        .send(ChessModelActorRequest::Think(request))
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model service is unavailable",
        )
            .into_response();
    }

    match receiver.await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to receive response from model service",
        )
            .into_response(),
    }
}

#[derive(Deserialize, JsonSchema)]
struct ApplyMoves {
    fen: String,
    moves_policy_indices: Vec<usize>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ApplyMovesOutput {
    fen: String,
}

async fn post_apply_moves(Json(apply_moves): Json<ApplyMoves>) -> impl IntoApiResponse {
    let mut board = match TriHexChess::new_with_fen(apply_moves.fen.as_ref(), false) {
        Ok(board) => board,
        Err(_) => {
            panic!("Invalid board state provided")
        }
    };

    let mapper = ChessExtendedCanonicalMapper;
    let mut move_store = ChessMoveStore::default();

    for policy_index in apply_moves.moves_policy_indices {
        board.fill_move_store(&mut move_store);

        let mv = mapper
            .index_to_move(&board, &move_store, policy_index)
            .expect("Invalid policy index");

        board.play_move_mut_with_store(&mv, &mut move_store, None);
    }

    Json(ApplyMovesOutput {
        fen: board.to_fen(),
    })
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct AttackMapsIn {
    fen: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct AttackMapsOut {
    maps: HashMap<String, Vec<bool>>,
}

fn attack_maps_doc(op: TransformOperation) -> TransformOperation {
    op.response::<200, Json<AttackMapsOut>>()
}

async fn get_attack_maps(Json(input): Json<AttackMapsIn>) -> impl IntoApiResponse {
    let board = match TriHexChess::new_with_fen(input.fen.as_ref(), false) {
        Ok(board) => board,
        Err(_) => {
            panic!("Invalid board state provided")
        }
    };

    let mut out_maps: HashMap<String, Vec<bool>> = HashMap::new();

    let attack_maps = board.calculate_per_piece_bitboard_attack_data();

    // generate to WP, WN, WB, WR, WQ, WK, GP, GN, GB, GR, GQ, GK, BP, BN, BB, BR, BQ, BK
    for p in Piece::all() {
        let idx = p.get_as_zero_based_index();

        for color in COLORS {
            let color_idx = color as usize;
            let attack_bb = attack_maps.attacks[color_idx][idx as usize];
            let mut attack_vec = vec![false; 96];

            for sq in 0..96 {
                if (attack_bb & (1u128 << sq)) != 0 {
                    attack_vec[sq] = true;
                }
            }

            let key = format!("{}{}", color, p.to_char().to_uppercase());
            out_maps.insert(key, attack_vec);
        }
    }

    (StatusCode::OK, Json(AttackMapsOut { maps: out_maps })).into_response()
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct PolicyMovesIn {
    fen: String,
    number_of_moves: usize,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct PolicyMovesOut {
    moves: Vec<ChessMoveWrapperDto>,
}

fn policy_moves_doc(op: TransformOperation) -> TransformOperation {
    op.response::<200, Json<PolicyMovesOut>>()
}

async fn post_policy_moves(
    State(state): State<AppState>,
    Json(input): Json<PolicyMovesIn>,
) -> impl IntoApiResponse {
    let board = match TriHexChess::new_with_fen(input.fen.as_ref(), false) {
        Ok(board) => board,
        Err(_) => {
            panic!("Invalid board state provided")
        }
    };

    let (responder, receiver) = oneshot::channel();

    let requests = PolicyRequestChess {
        board,
        number_of_moves: input.number_of_moves,
        responder,
    };

    if state
        .think_request_sender
        .send(ChessModelActorRequest::Policy(requests))
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model service is unavailable",
        )
            .into_response();
    }

    match receiver.await {
        Ok(moves) => (StatusCode::OK, Json(PolicyMovesOut { moves })).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to receive response from model service",
        )
            .into_response(),
    }
}
