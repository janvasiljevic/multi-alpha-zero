use crate::gpu::gpu_provider::{GpuInstance, GpuOut, GpuPerformanceMetadata, GpuSpawner};
use half::f16;
use ndarray::ArrayD;
use std::fmt::{Debug, Formatter};
use std::time::{Duration, Instant};
use maz_core::mapping::{Board, BoardMapper};
use maz_core::net::batch::Batch;

pub struct DummyGpuSpawner {
    gpu_count: usize,
    gpus: Vec<DummyGpuInstance>,
    batch_size: usize,
}

#[derive(Clone)]
pub struct DummyGpuInstance {
    artificial_delay: Duration,

    batch_size: usize,
    policy_len: usize,
    value_len: usize,

    // Statistics
    performance_metadata: GpuPerformanceMetadata,

    policy_logits_buffer: ArrayD<f16>,
    value_buffer: ArrayD<f32>,
}

impl Debug for DummyGpuInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("DummyGpuInstance")
            .field("artificial_delay", &self.artificial_delay)
            .field("batch_size", &self.batch_size)
            .field("policy_len", &self.policy_len)
            .field("value_len", &self.value_len)
            .field("performance_metadata", &self.performance_metadata)
            .finish()
    }
}

impl DummyGpuInstance {
    pub fn new(
        batch_size: usize,
        value_len: usize,
        policy_len: usize,
        artificial_delay: Duration,
    ) -> Self {
        let mut rng = fastrand::Rng::new();

        // Precompute random buffers to return on each inference.
        let mut policy_logits_buffer = ArrayD::zeros(vec![batch_size, policy_len]);

        policy_logits_buffer.map_inplace(|x| {
            *x = f16::from_f32((rng.f32() * 2.0 - 1.0) / 20.0);
        });

        let mut value_buffer = ArrayD::zeros(vec![batch_size, value_len]);

        value_buffer.map_inplace(|x| {
            *x = (rng.f32() * 2.0 - 1.0) / 20.0;
        });

        DummyGpuInstance {
            batch_size,
            value_len,
            policy_len,
            artificial_delay,
            performance_metadata: GpuPerformanceMetadata::default(),
            policy_logits_buffer,
            value_buffer,
        }
    }
}

impl GpuInstance for DummyGpuInstance {
    fn get_performance_metadata(&self) -> GpuPerformanceMetadata {
        self.performance_metadata
    }

    fn run_with_batch(&mut self, _: &Batch, count: usize) -> GpuOut {
        let start_time = Instant::now();

        // The target delay for this specific batch.
        let target_delay = self.artificial_delay.as_secs_f32();

        self.performance_metadata.increment(count, self.batch_size);

        let cloned_policy_logits = self.policy_logits_buffer.clone();
        let cloned_value = self.value_buffer.clone();

        let target_duration = Duration::from_secs_f32(target_delay);

        // This will eat up CPU cycles until the target duration is reached, but at least
        // it will be accurate - thread::sleep guarantees minimum duration, not exact timing,
        // and in my testing it was inaccurate by a lot.
        while start_time.elapsed() < target_duration {
            std::hint::spin_loop();
        }

        GpuOut {
            policy_logits: cloned_policy_logits,
            value: cloned_value,
        }
    }
}

impl DummyGpuSpawner {
    pub fn new<B: Board, BM: BoardMapper<B>>(
        desired_inferences_per_second: f64,
        gpu_count: usize,
        batch_size: usize,
        board: &B,
        mapper: &BM,
    ) -> Self {
        let artificial_delay =
            Duration::from_secs_f64(1.0 / (desired_inferences_per_second / batch_size as f64));

        let gpus = (0..gpu_count)
            .map(|_| {
                DummyGpuInstance::new(
                    batch_size,
                    board.player_num(),
                    mapper.policy_len(),
                    artificial_delay,
                )
            })
            .collect();

        DummyGpuSpawner {
            gpu_count,
            gpus,
            batch_size,
        }
    }
}

impl GpuSpawner for DummyGpuSpawner {
    fn get_gpu_count(&self) -> usize {
        self.gpu_count
    }

    fn sessions_per_gpu(&self) -> usize {
        1
    }

    fn generate_gpu_instance(&self, index: usize) -> Option<Box<dyn GpuInstance>> {
        if index < self.gpus.len() {
            Some(Box::new(self.gpus[index].clone()))
        } else {
            None
        }
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn model_path(&self) -> Option<String> {
        None
    }

    // No-op for dummy spawner
    fn set_model_path(&self, _: String) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use maz_core::mapping::hex_absolute_mapper::HexAbsoluteMapper;
    use maz_core::mapping::InputMapper;
    use game_hex::game_hex::HexGame;

    #[test]
    fn test_dummy_gpu_spawner() {
        let board = HexGame::new(5).unwrap();
        let mapper = HexAbsoluteMapper::new(&board);
        let batch_size = 64;

        let inferences_per_second = 70000.0;
        let test_for_seconds = 2.0;

        let spawner = DummyGpuSpawner::new(inferences_per_second, 1, batch_size, &board, &mapper);
        assert_eq!(spawner.get_gpu_count(), 1);

        let mut gpu_instance = spawner.generate_gpu_instance(0).unwrap();

        let [board_size, field_size] = mapper.input_board_shape();

        let batch = Batch::new(batch_size, board_size, field_size);

        for _ in 0..100 {
            let _ = gpu_instance.run_with_batch(&batch, batch_size);
        }

        let now = Instant::now();
        let limit = Duration::from_secs(test_for_seconds as u64);

        loop {
            if now.elapsed() > limit {
                break;
            }

            let _ = gpu_instance.run_with_batch(&batch, batch_size);
        }

        let expected_inferences = (inferences_per_second * test_for_seconds) as usize;
        let metadata = gpu_instance.get_performance_metadata();

        assert!(
            (metadata.inferences_processed as f64 - expected_inferences as f64).abs()
                < expected_inferences as f64 * 0.1,
            "Expected inferences: {}, processed: {}",
            expected_inferences,
            metadata.inferences_processed
        );

        println!("{:?}", metadata.get_timed(now.elapsed()));

        let gpu_out = gpu_instance.run_with_batch(&batch, batch_size);

        println!(
            "First 10 policy logits: {:?}",
            gpu_out.policy_logits.slice(s![0, ..10])
        );
    }
}
