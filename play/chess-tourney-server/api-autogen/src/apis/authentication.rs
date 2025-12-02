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
pub enum AuthLoginResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::LoginResponse),
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
pub enum AuthMeResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::MeUser),
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
pub enum AuthRegisterResponse {
    /// The request has succeeded and a new resource has been created as a result.
    Status201_TheRequestHasSucceededAndANewResourceHasBeenCreatedAsAResult(models::UserCreated),
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
pub enum AuthUpdateProfileResponse {
    /// The request has succeeded.
    Status200_TheRequestHasSucceeded(models::MeUser),
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

/// Authentication
#[async_trait]
#[allow(clippy::ptr_arg)]
pub trait Authentication<E: std::fmt::Debug + Send + Sync + 'static = ()>:
    super::ErrorHandler<E>
{
    type Claims;

    /// Log in to get an authentication token.
    ///
    /// AuthLogin - POST /api/auth/login
    async fn auth_login(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        body: &models::LoginPayload,
    ) -> Result<AuthLoginResponse, E>;

    /// Get the current authenticated user's profile.
    ///
    /// AuthMe - GET /api/auth/me
    async fn auth_me(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<AuthMeResponse, E>;

    /// Register a new user.
    ///
    /// AuthRegister - POST /api/auth/register
    async fn auth_register(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        body: &models::RegisterPayload,
    ) -> Result<AuthRegisterResponse, E>;

    /// Update the current authenticated user's profile.
    ///
    /// AuthUpdateProfile - PATCH /api/auth/me
    async fn auth_update_profile(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &models::UpdateProfilePayload,
    ) -> Result<AuthUpdateProfileResponse, E>;
}
