use ndarray::Array2;
use std::sync::Arc;

#[derive(Debug)]
pub struct InferenceResult {
    pub policy: Arc<[half::f16]>,
    pub value: Arc<[f32]>,

    // We need to tell the consumer which part of the Arc is theirs.
    pub policy_offset: usize,
    pub policy_len: usize,
    pub value_offset: usize,
    pub value_len: usize,

    /// ID of the game in the actual thread! (matches [`InferenceRequest.game_index`])
    pub game_index: usize,

    pub node_id: usize,

    pub input_array2: Array2<bool>,

    pub hash: u64,
}

impl InferenceResult {
    pub fn get_policy_slice(&self) -> &[half::f16] {
        &self.policy[self.policy_offset..self.policy_offset + self.policy_len]
    }
    pub fn get_value_slice(&self) -> &[f32] {
        &self.value[self.value_offset..self.value_offset + self.value_len]
    }
}

#[derive(Debug)]
pub struct InferenceRequest {

    pub input_array2: Array2<bool>,

    /// ID of the game in the actual thread!
    pub pool_index: usize,

    /// ID of the CPU thread sending the request, used to find the correct reply channel.
    pub thread_id: usize,

    /// ID of the node in the tree, to know which node to update.
    pub node_id: usize,

    pub hash: u64,
}
