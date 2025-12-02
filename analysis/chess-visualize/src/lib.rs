use schemars::JsonSchema;

pub mod vis_chess_render;
pub mod vis_chess_demo_move;
pub mod vis_chess_board;
pub mod vis_chess_util;
mod vis_chess_check;
mod vis_chess_tensor;

#[derive(serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
pub struct DatasetViewerConfig {
    pub parquet_path: String,
}

impl DatasetViewerConfig {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_yaml::from_reader(file)?;

        Ok(config)
    }
}