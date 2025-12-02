use crate::chess_router::model_actor::ChessModelActorRequest;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct AppState {
    pub think_request_sender: mpsc::Sender<ChessModelActorRequest>,
}
