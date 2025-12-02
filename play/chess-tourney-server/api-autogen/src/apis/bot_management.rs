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
pub enum BotsAssignBotResponse {
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
pub enum BotsCreateResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::ReadBot),
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
pub enum BotsDeleteResponse {
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
pub enum BotsListBotsResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<models::ReadBot>),
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
pub enum BotsModelKeysResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(Vec<String>),
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
pub enum BotsUpdateResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::ReadBot),
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

/// BotManagement
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait BotManagement<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Assign a bot to a game - Admin / Owner only.
    ///
    /// BotsAssignBot - POST /api/bots/assign
    async fn bots_assign_bot(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &models::AssignBotPayload,
    ) -> Result<BotsAssignBotResponse, E>;

    /// Create a new bot user - Admin only.
    ///
    /// BotsCreate - POST /api/bots/create
    async fn bots_create(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &models::CreateBot,
    ) -> Result<BotsCreateResponse, E>;

    /// Delete a bot user (changes the user type back to Regular) - Admin only.
    ///
    /// BotsDelete - DELETE /api/bots/delete/{botId}
    async fn bots_delete(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::BotsDeletePathParams,
    ) -> Result<BotsDeleteResponse, E>;

    /// Get list of all bots.
    ///
    /// BotsListBots - GET /api/bots/
    async fn bots_list_bots(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<BotsListBotsResponse, E>;

    /// Get available bot model keys.
    ///
    /// BotsModelKeys - GET /api/bots/keys
    async fn bots_model_keys(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<BotsModelKeysResponse, E>;

    /// Update an existing bot's configuration - Admin.
    ///
    /// BotsUpdate - PATCH /api/bots/update/{botId}
    async fn bots_update(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &models::BotsUpdatePathParams,
        body: &models::UpdateBot,
    ) -> Result<BotsUpdateResponse, E>;
}
