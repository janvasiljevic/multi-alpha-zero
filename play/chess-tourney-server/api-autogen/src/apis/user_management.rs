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
pub enum UsersListBotUsersResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::User>),
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
pub enum UsersListUsersResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::User>),
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
pub enum UsersRemoveUserFromGameResponse {
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

/// UserManagement
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait UserManagement<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Get a list of all bot users.
    ///
    /// UsersListBotUsers - GET /api/users/bots
    async fn users_list_bot_users(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<UsersListBotUsersResponse, E>;

    /// Get a list of all users.
    ///
    /// UsersListUsers - GET /api/users
    async fn users_list_users(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<UsersListUsersResponse, E>;

    /// Remove a user from a specific game (Admin only).
    ///
    /// UsersRemoveUserFromGame - DELETE /api/users/{userId}/games/{gameId}
    async fn users_remove_user_from_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::UsersRemoveUserFromGamePathParams,
    ) -> Result<UsersRemoveUserFromGameResponse, E>;
}
