use crate::entity::users::UserTypeDb;
use crate::game_events::RoomBroker;
use api_autogen::models;
use axum::http;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::error;
pub(crate) use crate::bot_thread::BotGameContext;
use crate::model_provider::ModelService;

pub struct ServerImpl {
    pub(crate) jwt_secret: String,
    pub(crate) db: DatabaseConnection,
    pub(crate) game_events: Arc<RoomBroker>,
    pub(crate) model_service: ModelService,
    pub bot_processing_tx: mpsc::Sender<BotGameContext>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: i64,
    pub exp: usize,
    pub userType: UserTypeDb,
    pub username: String,
}

#[async_trait::async_trait]
impl api_autogen::apis::ErrorHandler<anyhow::Error> for ServerImpl {
    #[allow(unused_variables)]
    #[tracing::instrument(skip_all)]
    async fn handle_error(
        &self,
        method: &http::Method,
        host: &axum_extra::extract::Host,
        cookies: &axum_extra::extract::CookieJar,
        error: anyhow::Error,
    ) -> Result<axum::response::Response, http::StatusCode> {
        tracing::error!("Unhandled error: {:?}", error);

        // TODO: Return 500
        axum::response::Response::builder()
            .status(http::StatusCode::INTERNAL_SERVER_ERROR)
            .body(axum::body::Body::empty())
            .map_err(|_| http::StatusCode::INTERNAL_SERVER_ERROR)
    }
}

fn error_500_ise_old() -> models::InternalError {
    models::InternalError {
        message: "Internal server error".to_string(),
    }
}

fn error_500_ise(e: anyhow::Error) -> models::InternalError {
    error!("Internal server error: {}", e);

    models::InternalError {
        message: "Internal server error".to_string(),
    }
}
