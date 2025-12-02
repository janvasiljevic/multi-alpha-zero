use crate::hex::hex_model_actor::ThinkRequest;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct AppState {
    pub think_request_sender: mpsc::Sender<ThinkRequest>,
}
