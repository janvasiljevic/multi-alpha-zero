use api_autogen::models::WsEvent;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{info, warn};

#[derive(Clone, Debug, Serialize)]
pub struct GameEvent {
    pub game_id: i64,
    pub event: WsEvent,
}

#[derive(Clone)]
pub struct RoomBroker {
    /// Map game_id -> broadcast sender
    inner: Arc<Mutex<HashMap<i64, Arc<broadcast::Sender<GameEvent>>>>>,
    /// Map game_id -> map of user_id -> connection count
    rooms: Arc<RwLock<HashMap<i64, HashMap<i64, usize>>>>,
}

impl RoomBroker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            rooms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Subscribe to a game's broadcast channel
    pub async fn subscribe(&self, game_id: i64) -> broadcast::Receiver<GameEvent> {
        let mut map = self.inner.lock().await;
        if let Some(sender) = map.get(&game_id) {
            return sender.subscribe();
        }

        let (sender, receiver) = broadcast::channel(100);
        map.insert(game_id, Arc::new(sender));
        receiver
    }

    /// Send a game event to all subscribers
    pub async fn send(&self, event: GameEvent) {
        let map = self.inner.lock().await;
        if let Some(sender) = map.get(&event.game_id) {
            let _ = sender.send(event);
        } else {
            warn!("No sender found for game {}", event.game_id);
        }
    }

    /// Join a room (increments connection count)
    pub async fn join_room(&self, user_id: i64, game_id: i64) {
        let mut rooms = self.rooms.write().await;
        let user_map = rooms.entry(game_id).or_default();
        let count = user_map.entry(user_id).or_insert(0);
        *count += 1;
        info!(
            "User {} joined game {} (connections = {})",
            user_id, game_id, *count
        );
    }

    /// Leave a room (decrements connection count, removes user if 0)
    pub async fn leave_room(&self, user_id: i64, game_id: i64) {
        let mut rooms = self.rooms.write().await;
        if let Some(user_map) = rooms.get_mut(&game_id) {
            if let Some(count) = user_map.get_mut(&user_id) {
                if *count > 1 {
                    *count -= 1;
                    info!(
                        "User {} left one connection in game {} (remaining = {})",
                        user_id, game_id, *count
                    );
                    return;
                }
            }

            // Remove the user entirely if last connection
            user_map.remove(&user_id);
            info!("User {} fully left game {}", user_id, game_id);

            if user_map.is_empty() {
                rooms.remove(&game_id);
            }
        }
    }

    /// Check if a user is in a room
    pub async fn is_in_room(&self, user_id: i64, game_id: i64) -> bool {
        let rooms = self.rooms.read().await;
        rooms
            .get(&game_id)
            .map(|user_map| user_map.contains_key(&user_id))
            .unwrap_or(false)
    }
}
