mod bot_thread;
mod entity;
mod errors;
mod game_events;
mod global_config;
mod model_provider;
mod routes_auth;
mod routes_bots;
mod routes_games;
mod routes_users;
mod seed;
mod server;
mod ws;
mod util;
mod routes_leaderboard;
mod routes_history;
mod routes_tournament;

use crate::game_events::RoomBroker;
use crate::server::ServerImpl;
use axum::Router;
use sea_orm::{Database, DatabaseConnection};
use sqlx::migrate::MigrateDatabase;
use sqlx::Sqlite;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tracing::info;

use crate::bot_thread::{start_bot_worker, BotGameContext};
use crate::global_config::load_config;
use crate::model_provider::ModelService;
use clap::Parser;
use tokio::sync::mpsc;
#[cfg(feature = "swagger-ui")]
use utoipa_swagger_ui::SwaggerUi;

pub async fn start_server(addr: &str, db: DatabaseConnection) {
    // This is a demo server, never use something like this in prod :)
    // The whole auth system is designed more around 'lets just identify users'
    // rather than 'securely authenticate users'.
    let jwt_secret = "a-very-secret-and-long-key-that-is-at-least-32-bytes".to_string();

    let model_service = ModelService::new(&global_config::CONFIG.get().unwrap().onnx_models)
        .expect("Failed to initialize ModelService");

    let (bot_tx, bot_rx) = mpsc::channel::<BotGameContext>(100);

    let server_impl = Arc::new(ServerImpl {
        jwt_secret,
        db,
        game_events: Arc::new(RoomBroker::new()),
        model_service,
        bot_processing_tx: bot_tx,
    });

    let server_copy = server_impl.clone();

    tokio::spawn(async move {
        start_bot_worker(server_copy, bot_rx).await;
    });

    let api_router = api_autogen::server::new::<_, _, anyhow::Error, _>(server_impl.clone());

    let custom_ws_router = ws::ws_router();

    server_impl.seed_default_user();

    let resolved_ws_router = custom_ws_router.with_state(server_impl);

    let cors_middleware = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    #[allow(unused_mut)]
    let mut app = Router::new()
        .merge(api_router)
        .merge(resolved_ws_router)
        .layer(cors_middleware);

    #[cfg(feature = "swagger-ui")]
    {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let schema_path = std::path::Path::new(&crate_dir).join("./api/schema/openapi.json");
        let openapi_json = std::fs::read_to_string(schema_path).unwrap();
        let openapi_json: serde_json::Value = serde_json::from_str(&openapi_json).unwrap();
        let swagger_ui =
            SwaggerUi::new("/swagger-ui").external_url_unchecked("/swagger.json", openapi_json);
        app = app.merge(swagger_ui);
        info!("Swagger UI available at http://{}/swagger-ui", addr);
    }

    info!("Listening on {}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(default_value = "play/chess-tourney-server/server-config-test.yaml")]
    pub(crate) config_path: String,
}

#[tokio::main]
async fn main() {
    let filter = tracing_subscriber::EnvFilter::new("info")
        .add_directive("sqlx::query=off".parse().unwrap())
        .add_directive("sqlx::migrate=off".parse().unwrap());

    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();

    load_config(&cli.config_path);

    let addr = format!(
        "{}:{}",
        global_config::CONFIG.get().unwrap().host,
        global_config::CONFIG.get().unwrap().port
    );
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR");

    let crate_dir = crate_dir.unwrap_or_else(|_| {
        info!("Could not get CARGO_MANIFEST_DIR env variable");
        ".".to_string()
    });

    info!(
        "swagger-ui feature enabled? {}",
        cfg!(feature = "swagger-ui")
    );

    let db_url = std::path::Path::new(&crate_dir)
        .join(&global_config::CONFIG.get().unwrap().database_file_path)
        .to_str()
        .unwrap()
        .to_string();

    if !Sqlite::database_exists(db_url.as_ref())
        .await
        .unwrap_or(false)
    {
        info!("Creating database {}", db_url);
        match Sqlite::create_database(db_url.as_ref()).await {
            Ok(_) => info!("Create db success"),
            Err(error) => panic!("error: {}", error),
        }
    }

    let db = Database::connect(&format!("sqlite://{}", db_url))
        .await
        .unwrap();

    db.get_schema_registry("chess-tourney-server::entity::*")
        .sync(&db)
        .await
        .unwrap();

    start_server(addr.as_str(), db).await;
}
