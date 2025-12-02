use chess_tourney_server::ServerConfig;
use once_cell::sync::OnceCell;
use std::fs;
use log::info;

pub static CONFIG: OnceCell<ServerConfig> = OnceCell::new();

pub fn load_config(path: &str) {
    let data = fs::read_to_string(path).expect("failed to read config file");

    let cfg: ServerConfig = serde_yaml::from_str(&data).expect("invalid config file");

    CONFIG.set(cfg).expect("CONFIG already set");

    info!("Configuration loaded from {}", path);
    info!("Config looks like: {:#?}", CONFIG.get().unwrap());
}
