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
pub enum HistoryGetGameHistoryResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::GameHistory),
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

/// GameHistory
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait GameHistory<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Get the move history of a specific game.
    ///
    /// HistoryGetGameHistory - GET /api/history/{gameId}
    async fn history_get_game_history(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::HistoryGetGameHistoryPathParams,
    ) -> Result<HistoryGetGameHistoryResponse, E>;
}
