use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::game_events::GameEvent;
use crate::server::{Claims, ServerImpl};
use api_autogen::apis::{ApiAuthBasic, BasicAuthKind};
use api_autogen::models::{
    GamesSubscribeToGameEventsPathParams, WsEvent, WsEventJoined, WsEventOnJoin,
    WsEventPlayerUpdate, WsOnJoinMessage, WsPlayerJoined, WsPlayerUpdate,
};
use axum::extract::ws::{Message, Utf8Bytes};
use axum::extract::FromRef;
use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderMap},
};
use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade}, Query,
        State,
    },
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tracing::info;

pub struct AuthError(StatusCode, &'static str);

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, error_message) = (self.0, self.1);
        let body = Json(json!({ "error": error_message }));
        (status, body).into_response()
    }
}

pub struct WsAuthenticatedClaims(pub Claims);

impl<S> FromRequestParts<S> for WsAuthenticatedClaims
where
    // This allows the extractor to access your `ServerImpl` from the router's state
    Arc<ServerImpl>: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let server = Arc::<ServerImpl>::from_ref(state);
        let headers: &HeaderMap = &parts.headers;

        if let Some(claims) = server
            .extract_claims_from_auth_header(BasicAuthKind::Bearer, headers, "unused")
            .await
        {
            Ok(WsAuthenticatedClaims(claims))
        } else {
            // If it fails, Axum will halt and return a 401 Unauthorized response.
            // Your handler will never even be called.
            Err(AuthError(
                StatusCode::UNAUTHORIZED,
                "Authentication token is missing or invalid.",
            ))
        }
    }
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(server): State<Arc<ServerImpl>>,
    Query(params): Query<GamesSubscribeToGameEventsPathParams>,
    WsAuthenticatedClaims(claims): WsAuthenticatedClaims, // <-- Use the extractor here
    headers: HeaderMap,
) -> impl IntoResponse {
    let protocols = headers
        .get(http::header::SEC_WEBSOCKET_PROTOCOL)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    ws.protocols(
        protocols
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<String>>(),
    )
    .on_upgrade(move |socket| ws_handle(socket, server, params, claims))
}

async fn ws_handle(
    socket: WebSocket,
    server: Arc<ServerImpl>,
    params: GamesSubscribeToGameEventsPathParams,
    claims: Claims,
) {
    let user_id = claims.sub;
    let game_id = params.game_id;

    server.game_events.join_room(user_id, game_id).await;

    let (mut sender, mut receiver) = socket.split();

    let game = match server.find_game(game_id).await {
        Ok(Some(game)) => game,
        Ok(None) => {
            let _ = sender
                .send(Message::Close(Some(axum::extract::ws::CloseFrame {
                    code: axum::extract::ws::close_code::NORMAL,
                    reason: "Game not found".into(),
                })))
                .await;
            server.game_events.leave_room(user_id, game_id).await;
            return;
        }
        Err(e) => {
            tracing::error!("Error finding game {}: {}", game_id, e);
            let _ = sender
                .send(Message::Close(Some(axum::extract::ws::CloseFrame {
                    code: axum::extract::ws::close_code::ABNORMAL,
                    reason: "Internal server error".into(),
                })))
                .await;
            server.game_events.leave_room(user_id, game_id).await;
            return;
        }
    };

    let dto_on_join = WsEvent::WsEventOnJoin(Box::new(WsEventOnJoin::new(
        "".into(),
        WsOnJoinMessage {
            game: server.map_game_to_dto(game.clone(), true).await,
        },
    )));

    let to_send = serde_json::to_string(&dto_on_join).unwrap();

    if sender.send(Message::Text(to_send.into())).await.is_err() {
        println!("Client disconnected during initial state send");
        server.game_events.leave_room(user_id, game_id).await;
        return;
    }

    let players = server
        .find_players_for_game(game_id)
        .await
        .unwrap_or_default();
    let map = server
        .players_to_dto(&players, game.names_masked.unwrap_or(false))
        .await;

    server
        .game_events
        .send(GameEvent {
            game_id,
            event: WsEvent::WsEventPlayerUpdate(Box::new(WsEventPlayerUpdate {
                kind: "".into(),
                value: WsPlayerUpdate { players: map },
            })),
        })
        .await;

    // Subscribe to game events
    let mut rx = server.game_events.subscribe(game_id).await;

    // Main loop
    loop {
        tokio::select! {
            // Receive client messages
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        println!("Client says: {}", text);
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => {
                        info!("Client disconnected");
                        break;
                    }
                    _ => {}
                }
            }

            // Receive game events from the server
            Ok(event) = rx.recv() => {
                let text = serde_json::to_string(&event.event).unwrap();
                if sender.send(Message::Text(text.into())).await.is_err() {
                    println!("Client disconnected during send");
                    break;
                }
            }
        }
    }

    // Always leave the room when the loop ends
    server.game_events.leave_room(user_id, game_id).await;
}

pub fn ws_router() -> Router<Arc<ServerImpl>> {
    Router::new().route("/ws", get(websocket_handler))
}
