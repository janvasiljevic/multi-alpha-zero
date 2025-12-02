use async_trait::async_trait;
use axum::extract::*;
use axum_extra::extract::{CookieJar, Host};
use bytes::Bytes;
use http::Method;
use serde::{Deserialize, Serialize};

use crate::{models, types::*};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesCreateGameResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::GameState),
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesFinishGameInResponse {
    /// There is no content to send for this request, but the headers may be useful.
    Status204_ThereIsNoContentToSendForThisRequest,
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesGetGameStateResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::GameState),
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesJoinGameResponse {
    /// There is no content to send for this request, but the headers may be useful.
    Status204_ThereIsNoContentToSendForThisRequest,
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesLeaveGameResponse {
    /// There is no content to send for this request, but the headers may be useful.
    Status204_ThereIsNoContentToSendForThisRequest,
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesListGamesResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::GameState>),
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesMakeMoveResponse {
    /// There is no content to send for this request, but the headers may be useful.
    Status204_ThereIsNoContentToSendForThisRequest,
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesStartGameResponse {
    /// There is no content to send for this request, but the headers may be useful.
    Status204_ThereIsNoContentToSendForThisRequest,
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[must_use]
#[allow(clippy::large_enum_variant)]
pub enum GamesSubscribeToGameEventsResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::WsEvent),
    /// Bad Request
    Status400_BadRequest(models::BadRequestError),
    /// Unauthorized access
    Status401_UnauthorizedAccess(models::UnauthorizedError),
    /// Forbidden
    Status403_Forbidden(models::ForbiddenError),
    /// Not Found
    Status404_NotFound(models::NotFoundError),
    /// Conflict
    Status409_Conflict(models::ConflictError),
    /// ISE
    Status500_ISE(models::InternalError),
}

/// GameManagement
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait GameManagement<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Creates a new game lobby.
    ///
    /// GamesCreateGame - POST /api/games/
    async fn games_create_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &models::CreateGame,
    ) -> Result<GamesCreateGameResponse, E>;

    /// Finish a game in a specific outcome (Admin only).
    ///
    /// GamesFinishGameIn - POST /api/games/{gameId}/admin/finish
    async fn games_finish_game_in(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesFinishGameInPathParams,
        body: &models::FinishGameIn,
    ) -> Result<GamesFinishGameInResponse, E>;

    /// Get the full state of a specific game.
    ///
    /// GamesGetGameState - GET /api/games/{gameId}/state
    async fn games_get_game_state(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesGetGameStatePathParams,
    ) -> Result<GamesGetGameStateResponse, E>;

    /// Join an existing game.
    ///
    /// GamesJoinGame - POST /api/games/{gameId}/join
    async fn games_join_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesJoinGamePathParams,
        body: &models::JoinGamePayload,
    ) -> Result<GamesJoinGameResponse, E>;

    /// Leave a game.
    ///
    /// GamesLeaveGame - POST /api/games/{gameId}/leave
    async fn games_leave_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesLeaveGamePathParams,
    ) -> Result<GamesLeaveGameResponse, E>;

    /// Lists available games to join.
    ///
    /// GamesListGames - GET /api/games/
    async fn games_list_games(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        query_params: &models::GamesListGamesQueryParams,
    ) -> Result<GamesListGamesResponse, E>;

    /// Make a move in a game.
    ///
    /// GamesMakeMove - POST /api/games/{gameId}/move
    async fn games_make_move(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesMakeMovePathParams,
        body: &models::MakeMovePayload,
    ) -> Result<GamesMakeMoveResponse, E>;

    /// Start a game, if all player slots are filled and you are the creator.
    ///
    /// GamesStartGame - POST /api/games/{gameId}/start
    async fn games_start_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesStartGamePathParams,
    ) -> Result<GamesStartGameResponse, E>;

    /// GamesSubscribeToGameEvents - GET /api/games/{gameId}/events
    async fn games_subscribe_to_game_events(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::GamesSubscribeToGameEventsPathParams,
    ) -> Result<GamesSubscribeToGameEventsResponse, E>;
}
