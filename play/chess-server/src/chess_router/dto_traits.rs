use maz_core::mapping::Board;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
pub trait ApiMove<B: Board>: Serialize + for<'de> Deserialize<'de> + JsonSchema {
    fn new_from_internal(mv: B::Move, score: f32, prior: f32) -> Self;
}

pub trait ApiBoard<B: Board>: Serialize + for<'de> Deserialize<'de> + JsonSchema {
    fn new_from_internal(board: &B) -> Self;
}

