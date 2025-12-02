use api_autogen::models;
use std::fmt::Debug;
use api_autogen::apis::game_management::{GamesCreateGameResponse, GamesGetGameStateResponse, GamesJoinGameResponse, GamesStartGameResponse};

pub fn error_401_unauthorized() -> models::UnauthorizedError {
    models::UnauthorizedError {
        message: "Unauthorized access".to_string(),
    }
}

pub fn error_400_bad_request() -> models::BadRequestError {
    models::BadRequestError {
        message: "Bad request".to_string(),
    }
}

pub fn error_400_bad_request_with_msg(msg: &str) -> models::BadRequestError {
    models::BadRequestError {
        message: msg.to_string(),
    }
}

pub fn error_403_forbidden() -> models::ForbiddenError {
    models::ForbiddenError {
        message: "Forbidden".to_string(),
    }
}

pub fn error_404_miss() -> models::NotFoundError {
    models::NotFoundError {
        message: "Not found".to_string(),
    }
}

pub fn error_409_conflict() -> models::ConflictError {
    models::ConflictError {
        message: "Conflict".to_string(),
    }
}

pub fn err_encapsulate(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::Error::msg(format!("{:?}", e))
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(models::BadRequestError),
    Unauthorized(models::UnauthorizedError),
    Forbidden(models::ForbiddenError),
    NotFound(models::NotFoundError),
    Conflict(models::ConflictError),
    Internal(models::InternalError),
}