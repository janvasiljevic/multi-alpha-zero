use crate::gpu::gpu_provider::{GpuOut, GpuSpawner, TimedGpuPerformanceMetadata};
use crate::inference_protocol::{InferenceRequest, InferenceResult};
use crossbeam_channel::{Receiver, Sender};
use half::f16;
use maz_util::math::RunningAverage;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{info, span, Level};
use maz_core::mapping::{Board, BoardMapper, InputMapper};
use maz_core::net::batch::Batch;

struct PreparedBatch{
    batcher: Batch,
    requests: Vec<InferenceRequest>,
}

pub struct JoinedHandle {
    batcher_handle: JoinHandle<()>,
    inference_handle: JoinHandle<TimedGpuPerformanceMetadata>,
    resender_handle: JoinHandle<()>,
}

struct InferenceOutputBundle {
    gpu_out: GpuOut,
    requests: Vec<InferenceRequest>,
}

impl JoinedHandle {
    pub fn join(self) -> TimedGpuPerformanceMetadata {
        let _ = self.batcher_handle.join().expect("Batcher thread panicked");
        let _ = self
            .resender_handle
            .join()
            .expect("Resender thread panicked");
        self.inference_handle
            .join()
            .expect("Inference thread panicked")
    }
}

/// Spawns (sessions per gpu * 2) dedicated GPU worker threads: one for batching and one for inference.
/// This creates a pipeline to keep the GPU saturated.
/// The point of this setup is to speed up the inference. As GPU is doing the inference,
/// we can prepare the next batch of requests in parallel and even stack them up.
/// Previously preparing the batch and inference were done in the same thread,
/// and this lead to a slowdown.
/// E.g. on Mac M2 Pro there is no difference, however on A100 the inference went up from
/// 180-200k inf/s per second to 270-290k inf/s per second.
/// Additionally, by running multiple sessions on the same GPU, we can increase throughput further,
/// because it allows the GPU to better schedule the work. Ideally this would be done with streams,
/// but not all providers support this kind of low level control.
pub fn spawn_gpu_worker_threads<B, I>(
    mapper: I,
    batch_size: i32,
    gpu_index: i32,
    gpu_spawner: Arc<dyn GpuSpawner>,
    reply_txs: Vec<Sender<InferenceResult>>,
    mut request_receivers: Vec<Receiver<InferenceRequest>>,
    stop_signal: Arc<AtomicBool>,
    // 3. ACCEPT THE SHARED LOCKS AS AN ARGUMENT
    gpu_init_locks: &[Arc<Mutex<()>>],
) -> Vec<JoinedHandle>
where
    B: Board + Send + 'static,
    I: BoardMapper<B> + Send + 'static,
{
    assert_eq!(
        request_receivers.len(),
        gpu_spawner.sessions_per_gpu(),
        "Number of request senders ({}) must match the number of sessions per GPU ({}).",
        request_receivers.len(),
        gpu_spawner.sessions_per_gpu()
    );

    // Channel for 'batcher' -> 'inferrer' threads. High capacity = buffer.
    let mut b2i_txs = Vec::with_capacity(gpu_spawner.sessions_per_gpu());
    let mut b2i_rxs = Vec::with_capacity(gpu_spawner.sessions_per_gpu());

    // Channel for 'inferrer' -> 'resender' threads.
    let mut i2r_txs = Vec::with_capacity(gpu_spawner.sessions_per_gpu());
    let mut i2r_rxs = Vec::with_capacity(gpu_spawner.sessions_per_gpu());

    for _ in 0..gpu_spawner.sessions_per_gpu() {
        let (tx, rx) = crossbeam_channel::bounded::<PreparedBatch>(20);
        b2i_txs.push(tx);
        b2i_rxs.push(rx);

        let (tx, rx) = crossbeam_channel::bounded::<InferenceOutputBundle>(20);
        i2r_txs.push(tx);
        i2r_rxs.push(rx);
    }

    let mut join_handles = Vec::with_capacity(3 * gpu_spawner.sessions_per_gpu());

    let policy_len = mapper.policy_len();

    // In your main setup function
    let num_gpus = gpu_spawner.get_gpu_count();
    // Create one Arc<Mutex> per GPU.
    let gpu_init_locks: Vec<Arc<Mutex<()>>> = (0..num_gpus)
        .map(|_| Arc::new(Mutex::new(())))
        .collect();

    // let sync_lock = Arc::new(std::sync::Mutex::new(()));
    let sync_barrier = Arc::new(std::sync::Barrier::new(gpu_spawner.sessions_per_gpu()));

    // 4. REMOVED: Do NOT create locks here anymore. They are passed in.
    // let num_gpus = gpu_spawner.get_gpu_count();
    // let gpu_init_locks: Vec<Arc<Mutex<()>>> = ...

    for i in 0..gpu_spawner.sessions_per_gpu() {
        let prepared_batch_tx = b2i_txs
            .pop()
            .expect("Not enough tx channels for the number of sessions per GPU");
        let prepared_batch_rx = b2i_rxs
            .pop()
            .expect("Not enough rx channels for the number of sessions per GPU");
        let output_bundle_tx = i2r_txs
            .pop()
            .expect("Not enough tx channels for the number of sessions per GPU");
        let output_bundle_rx = i2r_rxs
            .pop()
            .expect("Not enough rx channels for the number of sessions per GPU");

        let batcher_thread_handle = {
            let span = span!(
                Level::INFO,
                "GpuBatcher",
                gpu_id = gpu_index,
                session_id = i
            );

            let builder =
                std::thread::Builder::new().name(format!("GPU-Batcher-{}-{}", gpu_index, i));
            let cloned_mapper = mapper.clone();

            let request_rx = request_receivers
                .pop()
                .expect("Not enough request receivers for the number of sessions per GPU");

            let stop_signal = stop_signal.clone();

            builder
                .spawn(move || {
                    let _enter = span.enter();
                    batching_task(
                        cloned_mapper,
                        batch_size,
                        request_rx,
                        prepared_batch_tx,
                        stop_signal,
                    );
                })
                .expect("Failed to spawn GPU-Batcher thread")
        };

        let inference_thread_handle = {
            let gpu_spawner_clone = gpu_spawner.clone();
            let sync_lock_clone = gpu_init_locks[gpu_index as usize].clone();

            let sync_barrier_clone = sync_barrier.clone();
            let mapper_clone = mapper.clone();
            let batch_size_usize = batch_size as usize;

            let span = span!(
                Level::INFO,
                "GpuInferrer",
                gpu_id = gpu_index,
                session_id = i
            );
            let builder =
                std::thread::Builder::new().name(format!("GPU-Inferrer-{}-{}", gpu_index, i));

            builder
                .spawn(move || {
                    let _enter = span.enter();
                    inference_task(
                        gpu_spawner_clone,
                        gpu_index,
                        sync_lock_clone,
                        sync_barrier_clone,
                        mapper_clone,
                        batch_size_usize,
                        prepared_batch_rx,
                        output_bundle_tx,
                    )
                })
                .expect("Failed to spawn GPU-Inferrer thread")
        };

        let resender_thread_handle = {
            let reply_txs_clone = reply_txs.clone();
            let span = span!(
                Level::INFO,
                "GpuResender",
                gpu_id = gpu_index,
                session_id = i
            );
            let builder =
                std::thread::Builder::new().name(format!("GPU-Resender-{}-{}", gpu_index, i));

            builder
                .spawn(move || {
                    let _enter = span.enter();
                    resending_task(reply_txs_clone, output_bundle_rx, policy_len);
                })
                .expect("Failed to spawn GPU-Resender thread")
        };

        join_handles.push(JoinedHandle {
            batcher_handle: batcher_thread_handle,
            inference_handle: inference_thread_handle,
            resender_handle: resender_thread_handle,
        });
    }

    join_handles
}

fn batching_task<B: Board, I: InputMapper<B>>(
    input_mapper: I,
    batch_size: i32,
    request_rx: Receiver<InferenceRequest>,
    prepared_batch_tx: Sender<PreparedBatch>,
    stop_signal: Arc<AtomicBool>,
) {
    info!("Started batching task with batch size {}.", batch_size);
    let batch_size_usize = batch_size as usize;

    let [board_size, field_size] = input_mapper.input_board_shape();

    // Pre allocate a batcher with the correct shape.
    // batch_size is static, since models have a fixed batch size.
    // If we do less than batch_size, the rest of the batch will be empty
    // (or just garbage, since it might be leftover from a previous batch).
    let mut batcher = Batch::new(batch_size_usize, board_size, field_size);

    let mut num_of_request_in_queue = RunningAverage::default();

    loop {
        let mut request_batch = Vec::with_capacity(batch_size_usize);

        // Block for first request.
        match request_rx.recv() {
            Ok(first_req) => request_batch.push(first_req),
            Err(_) => {
                break; // Exit thread
            }
        }

        // If there are already multiple batches waiting to be processed,
        // we can take a little more time to wait for more requests.
        let timeout_micros =
            Duration::from_micros((50_000.0 * (1.0 + prepared_batch_tx.len() as f64 * 0.6)) as u64);

        // If there is no batch waiting, we want to immediately send what we have.
        // This kills the performance on the first 1-2 batches, but after that
        // the pipeline is more responsive.
        while request_batch.len() < batch_size_usize {
            match request_rx.recv_timeout(timeout_micros) {
                Ok(req) => request_batch.push(req),
                Err(_) => break, // Timeout or disconnect, proceed with current batch
            }

            // If the timeout was long, we want to check if we should stop.
            // This is we flush the last few requests and exit.
            if stop_signal.load(Ordering::Relaxed) {
                break;
            }
        }

        num_of_request_in_queue.add_sample(request_rx.len() as f32);

        if request_batch.is_empty() {
            assert!(
                stop_signal.load(Ordering::Relaxed),
                "Batch is empty but stop signal is not set."
            );
        }

        for (i, request) in request_batch.iter().enumerate() {
            let mut input_view = batcher.get_mut_item(i);
            input_view.assign(&request.input_array2);
        }

        // Send the prepared batch to the inference thread.
        let prepared_batch = PreparedBatch {
            batcher: batcher.clone(),
            requests: request_batch,
        };

        if prepared_batch_tx.send(prepared_batch).is_err() {
            info!("Inference thread disconnected, exiting batching task.");
            break;
        }
    }

    info!(
        "Batching task finished. Average requests in queue: {:.2}",
        num_of_request_in_queue.get_average()
    );
}

fn inference_task<B: Board>(
    gpu_spawner: Arc<dyn GpuSpawner>,
    gpu_index: i32,
    sync_lock: Arc<std::sync::Mutex<()>>,
    sync_barrier: Arc<std::sync::Barrier>,
    mapper: impl BoardMapper<B> + Send + 'static,
    batch_size_usize: usize,
    prepared_batch_rx: Receiver<PreparedBatch>,
    output_bundle_tx: Sender<InferenceOutputBundle>,
) -> TimedGpuPerformanceMetadata {
    let [board_size, field_size] = mapper.input_board_shape();

    let temp_batcher = Batch::new(batch_size_usize, board_size, field_size);

    let policy_len = mapper.policy_len();

    // Lock the whole prewarm phase, so that only one thread does it at a time.
    // CUDA/ TesnorRT is really picky about this phase.
    let _guard = sync_lock.lock().unwrap();

    let mut gpu_instance = gpu_spawner
        .generate_gpu_instance(gpu_index as usize)
        .expect("Failed to create GPU instance");

    gpu_instance
        .set_sizes(mapper.input_board_shape(), policy_len, 3) // Assuming value has 3 channels
        .expect("Failed to set sizes for GPU instance");

    // We do a prewarm round - needs lock because some providers are not thread safe during first run.
    // E.g. TensorRT or CUDA with CudaGraph (DAG dispatch).
    let num_warmup_runs = 3;
    for _ in 0..num_warmup_runs {
        gpu_instance.run_with_batch(&temp_batcher, batch_size_usize);
    }

    drop(_guard); // Release the lock after prewarm.

    sync_barrier.wait(); // Wait for all inference threads to be ready.

    let start_time = Instant::now();

    let mut inference_time_ms: RunningAverage = RunningAverage::default();

    let mut sum_wait_for_batch = Duration::default();

    loop {
        let start_wait = Instant::now();

        let prepared_batch = match prepared_batch_rx.recv() {
            Ok(batch) => batch,
            Err(_) => {
                info!("Batching thread disconnected, exiting inference task.");
                break;
            }
        };
        sum_wait_for_batch += start_wait.elapsed();

        let PreparedBatch { batcher, requests } = prepared_batch;
        let current_batch_size = requests.len();

        let start_inference = Instant::now();

        let gpu_out = gpu_instance.run_with_batch(&batcher, current_batch_size);

        let inference_duration = start_inference.elapsed();
        inference_time_ms.add_sample(inference_duration.as_millis() as f32);

        let bundle = InferenceOutputBundle { gpu_out, requests };
        if output_bundle_tx.send(bundle).is_err() {
            info!("Resending thread disconnected, exiting inference task.");
            break;
        }
    }

    let duration = start_time.elapsed();

    let perf_metadata = gpu_instance.get_performance_metadata().get_timed(duration);

    drop(gpu_instance);

    info!(
        "Average inf. time: {:.2} ms. Time waiting for batch: {:.2?}. Total time: {:.2?}. {:?}",
        inference_time_ms.get_average(),
        sum_wait_for_batch,
        duration,
        perf_metadata
    );

    perf_metadata
}

fn resending_task(
    reply_txs: Vec<Sender<InferenceResult>>,
    output_bundle_rx: Receiver<InferenceOutputBundle>,
    policy_len: usize,
) {
    info!("Started resending task.");
    let mut resend_time_ms = RunningAverage::default();

    loop {
        let bundle = match output_bundle_rx.recv() {
            Ok(bundle) => bundle,
            Err(_) => {
                info!("Inference thread disconnected, exiting resending task.");
                break;
            }
        };

        let start_resend = Instant::now();
        let InferenceOutputBundle { gpu_out, requests } = bundle;

        let policy_data: Arc<[f16]> = Arc::from(gpu_out.policy_logits.view().to_slice().unwrap());
        let value_data: Arc<[f32]> = Arc::from(gpu_out.value.view().to_slice().unwrap());

        for (i, request) in requests.into_iter().enumerate() {
            let result = InferenceResult {
                game_index: request.pool_index,
                policy: policy_data.clone(),
                value: value_data.clone(),
                policy_offset: i * policy_len,
                policy_len,
                value_offset: i * 3, // Assuming value has 3 channels
                value_len: 3,
                node_id: request.node_id,
                input_array2: request.input_array2,
                hash: request.hash,
            };

            debug_assert!(
                request.thread_id < reply_txs.len(),
                "Received a request with invalid thread ID {}. Available channels: {}",
                request.thread_id,
                reply_txs.len()
            );

            reply_txs[request.thread_id]
                .try_send(result)
                .unwrap_or_else(|_| {
                    tracing::warn!(
                        "Failed to send reply to CPU with thread ID {}. Channel is closed.",
                        request.thread_id
                    );
                });
        }
        resend_time_ms.add_sample(start_resend.elapsed().as_micros() as f32 / 1000.0);
    }

    info!(
        "Resending task finished. Average resend time: {:.2} ms",
        resend_time_ms.get_average()
    );
}
