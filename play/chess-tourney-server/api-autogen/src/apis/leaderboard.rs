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
pub enum LeaderboardGetLeaderboardResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::LeaderboardEntry>),
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

/// Leaderboard
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait Leaderboard<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Get the leaderboard.
    ///
    /// LeaderboardGetLeaderboard - GET /api/leaderboard/
    async fn leaderboard_get_leaderboard(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        query_params: &models::LeaderboardGetLeaderboardQueryParams,
    ) -> Result<LeaderboardGetLeaderboardResponse, E>;
}
