use std::sync::Arc;

use crate::chess_router::chess_routes::chess_routes;
use aide::{
    axum::ApiRouter,
    openapi::{OpenApi, Tag},
    transform::TransformOpenApi,
};
use axum::{http::StatusCode, Extension, Json};
use docs::docs_routes;
use errors::AppError;
use state::AppState;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::chess_router::model_actor::{
    model_actor_chess, ChessModelActorRequest,
};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub mod chess_router;
pub mod docs;
pub mod errors;
pub mod state;

#[tokio::main]
async fn main() {
    aide::generate::on_error(|error| {
        println!("{error}");
    });

    aide::generate::infer_responses(true);
    aide::generate::extract_schemas(true);

    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"))
        .add_directive("ort=warn".parse().unwrap());

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_level(true);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();

    let (tx, rx) = mpsc::channel::<ChessModelActorRequest>(32);

    tokio::spawn(model_actor_chess(rx));

    let state = AppState {
        think_request_sender: tx,
    };

    let mut api = OpenApi::default();

    let cors_layer = CorsLayer::new()
        .allow_origin(Any) // Open access to selected route
        .allow_headers(Any)
        .allow_methods(Any);

    let app = ApiRouter::new()
        .nest_api_service("/chess", chess_routes(state.clone()))
        .nest_api_service("/docs", docs_routes(state.clone()))
        .finish_api_with(&mut api, api_docs)
        .layer(Extension(Arc::new(api))) // Arc is very important here or you will face massive memory and performance issues
        .layer(cors_layer)
        .with_state(state);

    info!("Example docs are accessible at http://127.0.0.1:3000/docs");

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}

fn api_docs(api: TransformOpenApi) -> TransformOpenApi {
    api.title("Aide axum Open API")
        .summary("An example Todo application")
        .description(include_str!("README.md"))
        .tag(Tag {
            name: "todo".into(),
            description: Some("Todo Management".into()),
            ..Default::default()
        })
        .security_scheme(
            "ApiKey",
            aide::openapi::SecurityScheme::ApiKey {
                location: aide::openapi::ApiKeyLocation::Header,
                name: "X-Auth-Key".into(),
                description: Some("A key that is ignored.".into()),
                extensions: Default::default(),
            },
        )
        .default_response_with::<Json<AppError>, _>(|res| {
            res.example(AppError {
                error: "some error happened".to_string(),
                error_details: None,
                error_id: Uuid::nil(),
                // This is not visible.
                status: StatusCode::IM_A_TEAPOT,
            })
        })
}
