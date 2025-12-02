use crate::gpu::coreml_provider::{detect_coreml, CoreMLGpuSpawner};
use crate::gpu::cuda_provider::{detect_cuda, CudaGpuSpawner};
use crate::gpu::dummy_gpu::DummyGpuSpawner;
use ndarray::ArrayD;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::ValueType;
use std::env;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use maz_core::mapping::{Board, BoardMapper};
use maz_core::net::batch::Batch;

pub struct GpuOut {
    pub policy_logits: ArrayD<half::f16>,
    pub value: ArrayD<f32>,
}

pub fn get_cpu_count() -> usize {
    env::var("SLURM_CPUS_PER_TASK")
        .map(|s| s.parse::<i32>().unwrap_or(1))
        .unwrap_or_else(|_| num_cpus::get() as i32) as usize
}

#[derive(Copy, Clone, Default)]
pub struct GpuPerformanceMetadata {
    pub inferences_processed: usize,
    pub inferences_missed: usize,
    pub batches_processed: usize,
}

impl GpuPerformanceMetadata {
    pub fn increment(&mut self, processed: usize, batch_size: usize) {
        self.inferences_processed += processed;
        self.inferences_missed += batch_size - processed;
        self.batches_processed += 1;
    }

    pub fn fill_percent(&self) -> f64 {
        let total_potential_inferences = self.inferences_missed + self.inferences_processed;

        if total_potential_inferences == 0 {
            return 0.0;
        }

        (self.inferences_processed as f64 / total_potential_inferences as f64) * 100.0
    }

    pub fn get_timed(&self, duration: Duration) -> TimedGpuPerformanceMetadata {
        TimedGpuPerformanceMetadata::new(
            *self,
            duration,
            self.inferences_processed,
            self.batches_processed,
        )
    }
}

impl Debug for GpuPerformanceMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct GpuPerformanceMetadataDebug {
            inferences_processed: usize,
            inferences_missed: usize,
            batches_processed: usize,
            fill_percent: String,
        }

        let debug_info = GpuPerformanceMetadataDebug {
            inferences_processed: self.inferences_processed,
            inferences_missed: self.inferences_missed,
            batches_processed: self.batches_processed,
            fill_percent: format!("{:.2}%", self.fill_percent()),
        };

        Debug::fmt(&debug_info, f)
    }
}

#[derive(Debug)]
#[allow(dead_code)] // This struct is only used for logging with {:?}
pub struct TimedGpuPerformanceMetadata {
    metadata: GpuPerformanceMetadata,
    duration: Duration,

    pub inferences_per_second: f64,
    duration_per_inference: Duration,

    batches_per_second: f64,
    duration_per_batch: Duration,
}

impl TimedGpuPerformanceMetadata {
    pub fn new(
        metadata: GpuPerformanceMetadata,
        duration: Duration,
        inferences_processed: usize,
        batches_processed: usize,
    ) -> Self {
        let inferences_per_second = inferences_processed as f64 / duration.as_secs_f64();

        let seconds_per_inference = if inferences_processed > 0 {
            duration.as_secs_f64() / inferences_processed as f64
        } else {
            0.0
        };

        let batches_per_second = batches_processed as f64 / duration.as_secs_f64();
        let seconds_per_batch = if batches_processed > 0 {
            duration.as_secs_f64() / batches_processed as f64
        } else {
            0.0
        };

        TimedGpuPerformanceMetadata {
            metadata,
            duration,
            inferences_per_second,
            duration_per_inference: Duration::from_secs_f64(seconds_per_inference),
            batches_per_second,
            duration_per_batch: Duration::from_secs_f64(seconds_per_batch),
        }
    }
}

pub trait GpuInstance {
    fn get_performance_metadata(&self) -> GpuPerformanceMetadata;

    fn set_sizes(
        &mut self,
        _input: [usize; 2],
        _policy_len: usize,
        _value_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn run_with_batch(&mut self, batch: &Batch, count: usize) -> GpuOut;
}

pub trait GpuSpawner: Send + Sync {
    fn get_gpu_count(&self) -> usize;
    fn sessions_per_gpu(&self) -> usize;
    fn generate_gpu_instance(&self, index: usize) -> Option<Box<dyn GpuInstance>>;
    fn batch_size(&self) -> usize;
    fn model_path(&self) -> Option<String>;
    fn set_model_path(&self, path: String);
    fn warmup(&self) {}

    fn verify_dimensions(
        &self,
        board_size: usize,
        field_size: usize,
        policy_len: usize,
        player_num: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            self.model_path().is_some(),
            "Model path is not set. Cannot verify dimensions."
        );

        let model = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(get_cpu_count())?
            .with_inter_threads(1)?
            .commit_from_file(
                self.model_path()
                    .ok_or("Model path is not set. Cannot verify dimensions.")?,
            )?;

        let input_shape = [self.batch_size(), board_size, field_size];
        let policy_shape = [self.batch_size(), policy_len];
        let value_shape = [self.batch_size(), player_num];

        for input in model.inputs.iter() {
            if let ValueType::Tensor { shape, .. } = &input.input_type {
                if !input_shape
                    .iter()
                    .zip(shape.iter())
                    .all(|(a, b)| *a == (*b as usize))
                {
                    return Err(format!(
                        "Input shape mismatch for '{}': expected {:?}, got {:?}",
                        input.name, input_shape, shape
                    )
                    .into());
                }
            }
        }

        for output in model.outputs.iter() {
            match output.name.as_str() {
                "policy_logits" => match &output.output_type {
                    ValueType::Tensor { shape, .. } => {
                        if !policy_shape
                            .iter()
                            .zip(shape.iter())
                            .all(|(a, b)| *a == (*b as usize))
                        {
                            return Err(format!(
                                "Output 'policy_logits' shape mismatch: expected {policy_shape:?}, got {shape:?}"
                            )
                                .into());
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Expected 'policy_logits' to be a Tensor, but got {:?}",
                            output.output_type
                        )
                        .into());
                    }
                },
                "value" => match &output.output_type {
                    ValueType::Tensor { shape, .. } => {
                        if !value_shape
                            .iter()
                            .zip(shape.iter())
                            .all(|(a, b)| *a == (*b as usize))
                        {
                            return Err(format!(
                                "Output 'value' shape mismatch: expected {value_shape:?}, got {shape:?}"
                            )
                                .into());
                        }
                    }
                    _ => panic!(
                        "Expected 'value' to be a Tensor, but got {:?}",
                        output.output_type
                    ),
                },
                _ => {
                    return Err(format!(
                        "Unexpected output name: '{}'. Expected 'policy_logits' or 'value'.",
                        output.name
                    )
                    .into());
                }
            }
        }

        let span = tracing::info_span!(
            "GpuSpawner",
            model_path = self.model_path().as_deref().unwrap_or("unknown"),
            batch_size = self.batch_size()
        );
        let _enter = span.enter();

        tracing::info!(
            "Successfully verified model dimensions. Input shape: {:?}, Policy shape: {:?}, Value shape: {:?}",
            input_shape,
            policy_shape,
            value_shape
        );

        Ok(())
    }
}

pub struct DummyArgs {
    pub desired_inference_count: f64,
    pub gpu_count: usize,
    pub batch_size: usize,
}

impl DummyArgs {
    pub fn new(desired_inference_count: f64, gpu_count: usize, batch_size: usize) -> Self {
        assert!(gpu_count > 0, "GPU count must be > 0");
        assert!(batch_size > 0, "Batch size must be > 0");
        assert!(
            desired_inference_count > 0.0,
            "Desired inference count must be > 0"
        );
        Self {
            desired_inference_count,
            gpu_count,
            batch_size,
        }
    }
}

pub struct GenericModelArgs {
    pub model_path: String,
    pub batch_size: usize,
    pub sessions_per_gpu: usize,
    pub run_folder: PathBuf
}

pub enum GpuModelArgs {
    Dummy(DummyArgs),
    Generic(GenericModelArgs),
}

pub fn automatic_gpu_spawner<B: Board, BM: BoardMapper<B>>(
    spawner_args: GpuModelArgs,
    board: &B,
    board_mapper: &BM,
) -> Result<Arc<dyn GpuSpawner>, String> {
    match spawner_args {
        GpuModelArgs::Dummy(args) => Ok(Arc::new(DummyGpuSpawner::new(
            args.desired_inference_count,
            args.gpu_count,
            args.batch_size,
            board,
            board_mapper,
        ))),
        GpuModelArgs::Generic(args) => {
            match detect_cuda() {
                Ok(gpu_count) => {
                    return Ok(Arc::new(CudaGpuSpawner::new(
                        args.model_path,
                        args.batch_size,
                        gpu_count,
                        args.sessions_per_gpu,
                        args.run_folder,
                    )));
                }
                Err(e) => {
                    tracing::info!("CUDA detection failed: {}", e);
                }
            }

            if detect_coreml() {
                return Ok(Arc::new(CoreMLGpuSpawner::new(
                    args.model_path,
                    args.batch_size,
                    args.sessions_per_gpu,
                )));
            }

            Err(
                "Couldn't detect either CoreML or CUDA, and no dummy arguments provided"
                    .to_string(),
            )
        }
    }
}
