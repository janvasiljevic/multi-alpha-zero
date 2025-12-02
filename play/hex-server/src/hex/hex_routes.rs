use crate::hex::hex_model_actor::{ThinkRequest, ThinkResult};
use crate::hex::hex_routes_logic::{BoardState, BoardTile};
use crate::state::AppState;
use aide::axum::routing::{get_with, post_with};
use aide::axum::{ApiRouter, IntoApiResponse};
use aide::transform::TransformOperation;
use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::{http, Json};
use http::StatusCode;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::oneshot;
use game_hex::coords::AxialCoord;
use game_hex::game_hex::HexGame;
use maz_core::mapping::{Board, PolicyMapper};
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;

fn empty_doc(op: TransformOperation) -> TransformOperation {
    op
}

pub fn hex_routes(state: AppState) -> ApiRouter {
    ApiRouter::new()
        .api_route("/default/{radius}", get_with(get_default_board, empty_doc))
        .api_route(
            "/make_move",
            post_with(post_make_move, |op| {
                op.response::<200, Json<BoardState>>()
                    .response::<400, String>()
            }),
        )
        .api_route(
            "/think",
            post_with(post_think_about_position, |op| {
                op.response::<200, Json<ThinkResult>>()
            }),
        )
        .api_route("/apply_moves", post_with(post_apply_moves, empty_doc))
        .with_state(state)
}

#[derive(Deserialize, JsonSchema)]
struct CommitMove {
    coord: BoardTile,
    board: BoardState,
}

async fn post_make_move(Json(commit_move): Json<CommitMove>) -> impl IntoApiResponse {
    let mut internal_board = commit_move.board.to_internal();

    if internal_board.is_terminal() {
        return (
            StatusCode::BAD_REQUEST,
            "Cannot make a move on a terminal board",
        )
            .into_response();
    }

    internal_board
        .make_move_mut(AxialCoord {
            q: commit_move.coord.q,
            r: commit_move.coord.r,
        })
        .expect("Failed to make move");

    (
        StatusCode::OK,
        Json(BoardState::new_from_internal(internal_board)),
    )
        .into_response()
}

#[derive(Deserialize, JsonSchema)]
struct SelectRadius {
    radius: u32,
}

async fn get_default_board(Path(radius): Path<SelectRadius>) -> impl IntoApiResponse {
    let board = HexGame::new((radius.radius + 1) as i32).unwrap();

    Json(BoardState::new_from_internal(board))
}

#[derive(Deserialize, JsonSchema)]
struct ThinkInput {
    board_state: BoardState,
    number_of_rollouts: u64,
    key: String, // This should match the model key used in the actor
}

async fn post_think_about_position(
    State(state): State<AppState>,
    Json(think_input): Json<ThinkInput>,
) -> impl IntoApiResponse {
    let internal_board = think_input.board_state.to_internal();


    println!("Fancy debug {}", internal_board.fancy_debug());

    if internal_board.is_terminal() {
        return (
            StatusCode::BAD_REQUEST,
            "Cannot think about a terminal board",
        )
            .into_response();
    }

    let number_of_rollouts = think_input.number_of_rollouts;

    let (responder, receiver) = oneshot::channel();

    let request = ThinkRequest {
        board: internal_board,
        num_of_rollouts: number_of_rollouts,
        responder,
        key: think_input.key.clone(),
    };

    // The .send() operation is async and can fail if the receiver is dropped.
    if state.think_request_sender.send(request).await.is_err() {
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
    board: BoardState,
    moves_policy_indices: Vec<usize>,
}

async fn post_apply_moves(Json(apply_moves): Json<ApplyMoves>) -> impl IntoApiResponse {
    let mut internal_board = apply_moves.board.to_internal();

    let mapper = HexCanonicalMapper::new(&internal_board);

    // FIXME: This function needs to be more general purpose, not just for Hex.
    for policy_index in apply_moves.moves_policy_indices {
        let coord = mapper
            .index_to_move(&internal_board, &vec![], policy_index)
            .expect("Invalid policy index");

        internal_board
            .make_move_mut(coord)
            .expect("Failed to apply move");
    }

    Json(BoardState::new_from_internal(internal_board))
}
