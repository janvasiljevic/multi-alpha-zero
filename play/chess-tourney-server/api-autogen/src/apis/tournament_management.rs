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
pub enum TournamentsCreateTournamentResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::ReadTournament),
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
pub enum TournamentsListTournamentsResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::ReadTournament>),
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

/// TournamentManagement
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait TournamentManagement<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Create a new tournament - Admin only.
    ///
    /// TournamentsCreateTournament - POST /api/tournaments/create
    async fn tournaments_create_tournament(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &models::CreateTournament,
    ) -> Result<TournamentsCreateTournamentResponse, E>;

    /// Get list of all tournaments.
    ///
    /// TournamentsListTournaments - GET /api/tournaments/
    async fn tournaments_list_tournaments(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<TournamentsListTournamentsResponse, E>;
}
