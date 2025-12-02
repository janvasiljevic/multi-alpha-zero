use crate::dynamic_self_play_game::DynamicSelfPlayGame;
use crate::inference_protocol::{InferenceRequest, InferenceResult};
use crate::learning_target_modifier::LearningModifier;
use crate::search_settings::SearchSettings;
use crate::self_play_game::Simulation;
use crate::sharded_sender::ShardedSender;
use crate::shared_cache::{CacheWriterMessage, NetCachedEvaluation, SharedCache};
use crate::steppable_mcts::{CacheShapeMissmatch, ConsumeValues};
use crossbeam_channel::Receiver;
use maz_core::mapping::{Board, MetaBoardMapper, OptionalSharedOracle};
use ndarray::Array2;
use rand::prelude::ThreadRng;
use std::collections::VecDeque;
use std::ops::AddAssign;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{span, Level};

#[derive(Debug, Clone, Default)]
pub struct GeneratorStats {
    pub cache_hits: u64,
    pub nodes_per_second: f64,
    pub nodes_processed: u64,
}

impl AddAssign for GeneratorStats {
    fn add_assign(&mut self, other: Self) {
        self.cache_hits += other.cache_hits;
        self.nodes_per_second += other.nodes_per_second;
        self.nodes_processed += other.nodes_processed;
    }
}

pub fn generator_worker_task<B: Board + 'static, BM: MetaBoardMapper<B>>(
    thread_id: usize,
    init_board: B,
    number_of_parallel_games: usize,
    mapper: BM,
    search_settings: SearchSettings,
    mut gpu_sharded_sender: ShardedSender<InferenceRequest>,
    mut cache_sharded_sender: ShardedSender<CacheWriterMessage>,
    rx: Receiver<InferenceResult>,
    samples_tx: mpsc::UnboundedSender<Simulation<B>>,
    stop_signal: Arc<AtomicBool>,
    cache: SharedCache,
    previous_game_pool: Option<Vec<DynamicSelfPlayGame<B>>>,
    oracle: OptionalSharedOracle<B>,
    learning_modifier: LearningModifier,
) -> (GeneratorStats, Vec<DynamicSelfPlayGame<B>>) {
    let thread_name = std::thread::current()
        .name()
        .expect("Thread name should be set")
        .to_owned();

    let span = span!(Level::INFO, "Generator", id = %thread_id, name = %thread_name);
    let _enter = span.enter();

    let mut rng = ThreadRng::default();

    let mut game_pool: Vec<DynamicSelfPlayGame<B>> = if let Some(previous) = previous_game_pool {
        assert_eq!(
            previous.len(),
            number_of_parallel_games,
            "Previous game pool length does not match the expected number of parallel games"
        );
        previous
    } else {
        (0..number_of_parallel_games)
            .map(|_| {
                DynamicSelfPlayGame::new(
                    &init_board,
                    &mapper,
                    &mut rng,
                    &search_settings,
                    oracle.clone(),
                    learning_modifier.clone()
                )
            })
            .collect()
    };

    let board_shape = mapper.input_board_shape();

    let mut input_array_pool: Vec<Option<Array2<bool>>> =
        vec![Some(Array2::from_elem(board_shape, false)); number_of_parallel_games];

    let mut games_to_advance: VecDeque<usize> = (0..number_of_parallel_games).collect();
    let mut pending_requests_count = 0u16;

    let mut cache_hits = 0;

    let mut nodes_processed = 0u64;

    let start_time = Instant::now();

    let mut cache_writes_dropped = 0;

    let hash_builder = ahash::RandomState::with_seed(42); // All threads use the same seed for hashing or we would not get any cache hits.

    while !stop_signal.load(Ordering::Relaxed) {
        while let Some(i) = games_to_advance.pop_front() {
            let request = game_pool[i].advance(&mut rng, &search_settings, &samples_tx, &mapper);

            nodes_processed += 1;

            if let Some(request_details) = request {
                let mut sender_ndarray = input_array_pool[i]
                    .take()
                    .expect("Array should be available in the pool");

                mapper.encode_input(&mut sender_ndarray.view_mut(), &request_details.board);

                let hash_u64 = hash_builder.hash_one(&sender_ndarray);

                // Check if the specific board is already cached.
                if let Some(cached_eval) = cache.get(&hash_u64) {
                    cache_hits += 1;

                    if let Err(CacheShapeMissmatch { expected_len }) = game_pool[i].receive(
                        &mut rng,
                        &search_settings,
                        request_details.node_id,
                        ConsumeValues::ReadFromCache(&cached_eval),
                    ) {
                        // Extremely rare case, but can happen
                        tracing::warn!(
                            "Cache collision detected. Expected len: {}, but got: {}. Invalidating cache entry.\
                            This should occur extremely rarely, if it happens often, something is wrong with the hashing function.\
                            Board:\n {}",
                            expected_len,
                            cached_eval.softmax_policy.len(),
                            request_details.board.fancy_debug()
                        );

                        cache.remove(&hash_u64);
                    } else {
                        // Successfully used cache, so we can continue to the next game.
                        games_to_advance.push_back(i);

                        // Put back the array for future use.
                        input_array_pool[i] = Some(sender_ndarray);

                        continue;
                    }
                }

                let inference_request = InferenceRequest {
                    input_array2: sender_ndarray,
                    pool_index: i,
                    thread_id,
                    node_id: request_details.node_id,
                    hash: hash_u64,
                };

                gpu_sharded_sender
                    .send(inference_request)
                    .expect("GPU thread has panicked");

                pending_requests_count += 1;
            } else {
                // If no request was generated (e.g., terminal state),
                // the game is still "ready" for another attempt to advance.
                games_to_advance.push_back(i);
            }
        }

        // All games have sent their requests, now we wait for results with blocking recv,
        // to avoid busy waiting in a loop.
        // Need to have a timeout, so the thread can check the stop signal periodically
        if games_to_advance.is_empty() {
            if let Ok(res) = rx.recv_timeout(Duration::from_millis(2000)) {
                process_inference_result::<B, BM>(
                    res,
                    &mut game_pool,
                    &mut rng,
                    &search_settings,
                    &mut input_array_pool,
                    &mut games_to_advance,
                    &mut cache_sharded_sender,
                    &mut cache_writes_dropped,
                );
                pending_requests_count -= 1;
            }
        }

        // Process any available results without blocking.
        while let Ok(res) = rx.try_recv() {
            process_inference_result::<B, BM>(
                res,
                &mut game_pool,
                &mut rng,
                &search_settings,
                &mut input_array_pool,
                &mut games_to_advance,
                &mut cache_sharded_sender,
                &mut cache_writes_dropped,
            );
            pending_requests_count -= 1;
        }
    }

    let to_flush = pending_requests_count;

    while pending_requests_count > 0 {
        if let Ok(res) = rx.recv() {
            process_inference_result::<B, BM>(
                res,
                &mut game_pool,
                &mut rng,
                &search_settings,
                &mut input_array_pool,
                &mut games_to_advance,
                &mut cache_sharded_sender,
                &mut cache_writes_dropped,
            );
            pending_requests_count -= 1;
        }
    }

    let total_retained_nodes: u64 = game_pool
        .iter()
        .map(|game| game.stats().total_retained_nodes)
        .sum();

    let nodes_per_second = nodes_processed as f64 / start_time.elapsed().as_secs_f64();

    tracing::info!(
        "Flushed {} requests. Retained nodes: {}. Cache hits: {}. Speed of {} nodes/sec. Without cache {}. Cached writes dropped: {}",
        to_flush,
        total_retained_nodes,
        cache_hits,
        ((nodes_processed as f64) / start_time.elapsed().as_secs_f64()) as i64,
        ((nodes_processed - cache_hits) as f64 / start_time.elapsed().as_secs_f64()) as i64,
        cache_writes_dropped
    );

    (
        GeneratorStats {
            cache_hits,
            nodes_per_second,
            nodes_processed,
        },
        game_pool,
    )
}

fn process_inference_result<B, BM>(
    res: InferenceResult,
    game_pool: &mut [DynamicSelfPlayGame<B>],
    rng: &mut ThreadRng,
    search_settings: &SearchSettings,
    array_pool: &mut [Option<Array2<bool>>],
    games_to_advance: &mut VecDeque<usize>,
    cache_sharded_sender: &mut ShardedSender<CacheWriterMessage>,
    cache_writes_dropped: &mut u64,
) where
    B: Board + 'static,
    BM: MetaBoardMapper<B>,
{
    let node_id = res.node_id;
    let policy_net = res.get_policy_slice();
    let value_net = res.get_value_slice();
    let game_index = res.game_index;

    let hashkey = res.hash;

    let callback = Box::new(|eval: NetCachedEvaluation| {
        if let Some(_) = cache_sharded_sender.try_send((hashkey, eval)).err() {
            *cache_writes_dropped += 1;
        };
    });

    game_pool[game_index]
        .receive(
            rng,
            search_settings,
            node_id,
            ConsumeValues::ConsumeWithOptionalCallback {
                policy_net_f16: policy_net,
                values_net: value_net,
                callback: Some(callback),
            },
        )
        .unwrap();

    array_pool[game_index] = Some(res.input_array2);
    games_to_advance.push_back(game_index);
}
