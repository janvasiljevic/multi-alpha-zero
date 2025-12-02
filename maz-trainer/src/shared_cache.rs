use crossbeam_channel::Receiver;
use half::f16;
use maz_core::mapping::{Board, MetaBoardMapper};
use quick_cache::sync::Cache;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct NetCachedEvaluation {
    /// We only store the softmax-ed policy to save space and computation.
    /// This requires that nodes in the tree are always expanded in the same order,
    /// so we can reliably map 0->0, 1->1, etc.
    pub softmax_policy: Vec<half::f16>,
    pub value: Vec<f32>,
}

pub type SharedCache = Arc<Cache<u64, Arc<NetCachedEvaluation>>>;

pub type CacheWriterMessage = (u64, NetCachedEvaluation);

pub fn estimate_total_memory_usage_mb<B: Board, BM: MetaBoardMapper<B>>(
    capacity: u64,
    board: &B,
    mapper: &BM,
) -> u64 {
    let policy_size = mapper.average_number_of_moves() * size_of::<f16>();
    let value_size = board.player_num() * size_of::<f32>();
    let entry_size = policy_size + value_size;

    (capacity * entry_size as u64) / (1024 * 1024) // Convert to MB
}

pub fn shared_cache_actor(cache: SharedCache, rx: Receiver<CacheWriterMessage>) {
    while let Ok((board, evaluation)) = rx.recv() {
        cache.insert(board, Arc::new(evaluation));
    }
}
