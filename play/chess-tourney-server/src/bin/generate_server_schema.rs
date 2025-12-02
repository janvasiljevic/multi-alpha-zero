use schemars::schema_for;
use serde_json::to_string_pretty;
use chess_tourney_server::ServerConfig;

fn main() {
    let schema = schema_for!(ServerConfig);
    std::fs::write("play/chess-tourney-server/server.schema.json", to_string_pretty(&schema).unwrap())
        .expect("Unable to write file");
}
