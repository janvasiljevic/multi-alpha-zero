use crate::gpu::gpu_provider::{
    get_cpu_count, GpuInstance, GpuOut, GpuPerformanceMetadata, GpuSpawner,
};
use half::f16;
use maz_core::net::batch::Batch;
use ndarray::ArrayD;
use ort::execution_providers::{
    CUDAExecutionProvider, ExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

use tracing::info;

pub fn detect_cuda() -> Result<usize, Box<dyn std::error::Error>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()?;

    if !output.status.success() {
        tracing::error!("nvidia-smi failed: {:?}", output);
        return Err(
            "nvidia-smi command failed. Ensure that NVIDIA drivers and CUDA are installed.".into(),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let device_count = stdout.lines().count();

    if device_count == 0 {
        return Err("Nvidia-smi is present, however no CUDA devices detected.".into());
    }

    Ok(device_count)
}

#[derive(Debug)]
pub struct CudaGpuInstance {
    batch_size: usize,
    performance_metadata: GpuPerformanceMetadata,

    model: Session,
}

fn get_engine_cache_path(run_folder: PathBuf) -> String {
    let cache_dir = run_folder.join("cache");

    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    let tensorrt_engine_cache = cache_dir.join("tensorrt_engine_cache");

    if !tensorrt_engine_cache.exists() {
        std::fs::create_dir_all(&tensorrt_engine_cache)
            .expect("Failed to create TensorRT engine cache directory");
    }

    tensorrt_engine_cache
        .to_str()
        .unwrap_or_default()
        .to_string()
}

fn get_engine_timing_cache_path(run_folder: PathBuf) -> String {
    let cache_dir = run_folder.join("cache");

    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    let timing_cache = cache_dir.join("tensorrt_timing_cache");

    if !timing_cache.exists() {
        std::fs::create_dir_all(&timing_cache)
            .expect("Failed to create TensorRT engine cache directory");
    }

    timing_cache.to_str().unwrap_or_default().to_string()
}

impl CudaGpuInstance {
    pub fn new(
        model_path: String,
        batch_size: usize,
        gpu_id: i32,
        is_tensor_rt_available: bool,
        run_folder: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(batch_size > 0, "Batch size must be greater than 0");

        let execution_provider = if is_tensor_rt_available {
            TensorRTExecutionProvider::default()
                .with_device_id(gpu_id)
                .with_cuda_graph(false)
                .with_builder_optimization_level(5)
                .with_engine_cache(true)
                .with_engine_cache_path(get_engine_cache_path(run_folder.clone()))
                .with_layer_norm_fp32_fallback(true)
                .with_timing_cache(true)
                .with_timing_cache_path(get_engine_timing_cache_path(run_folder))
                .with_fp16(true)
                .build()
                .error_on_failure()
        } else {
            CUDAExecutionProvider::default()
                .with_cuda_graph(true)
                .with_device_id(gpu_id)
                .build()
                .error_on_failure()
        };

        let model = Session::builder()?
            .with_execution_providers([execution_provider])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path.clone())?;

        Ok(Self {
            model,
            batch_size,
            performance_metadata: GpuPerformanceMetadata::default(),
        })
    }
}

impl GpuInstance for CudaGpuInstance {
    fn get_performance_metadata(&self) -> GpuPerformanceMetadata {
        self.performance_metadata
    }

    fn run_with_batch(&mut self, batch: &Batch, count: usize) -> GpuOut {
        let mut outputs = self
            .model
            .run(ort::inputs!["input" => batch.tensor()])
            .expect("Failed to run model");

        let policy_logits_dyn = outputs
            .remove("policy_logits")
            .expect("No 'policy_logits' output found");

        let policy_logits: ArrayD<f16> = policy_logits_dyn
            .try_extract_array::<f16>()
            .expect("Failed to extract 'policy_logits' as ArrayView2")
            .into_owned();

        let value_dyn = outputs.remove("value").expect("No 'value' output found");

        let value: ArrayD<f32> = value_dyn
            .try_extract_array()
            .expect("Failed to extract 'value' as ArrayView2")
            .into_owned();

        self.performance_metadata.increment(count, self.batch_size);

        GpuOut {
            policy_logits,
            value,
        }
    }
}

#[derive(Debug)]
pub struct CudaGpuSpawner {
    model_path: Mutex<String>,
    batch_size: usize,
    device_count: usize,
    is_tensor_rt_available: bool,
    sessions_per_gpu: usize,
    run_folder: PathBuf,
}

impl CudaGpuSpawner {
    pub fn new(
        model_path: String,
        batch_size: usize,
        device_count: usize,
        sessions_per_gpu: usize,
        run_folder: PathBuf,
    ) -> Self {
        assert!(batch_size > 0, "Batch size must be greater than 0");

        let span = tracing::info_span!("CudaGpuSpawner", model_path = model_path,);
        let _enter = span.enter();

        let mut builder = Session::builder()
            .unwrap()
            .with_intra_threads(get_cpu_count())
            .unwrap()
            .with_inter_threads(1)
            .unwrap();

        let rt = TensorRTExecutionProvider::default();

        let is_tensor_rt_available = rt.register(&mut builder).is_ok();

        if is_tensor_rt_available {
            info!("TensorRT=TRUE. TensorRT is available, will use it where possible.");
        } else {
            info!("TensorRT=FALSE. TensorRT is not available, falling back to CUDA.");
        }

        Self {
            model_path: Mutex::new(model_path),
            batch_size,
            device_count,
            is_tensor_rt_available,
            sessions_per_gpu,
            run_folder,
        }
    }
}

impl GpuSpawner for CudaGpuSpawner {
    fn get_gpu_count(&self) -> usize {
        self.device_count
    }

    fn sessions_per_gpu(&self) -> usize {
        self.sessions_per_gpu
    }

    fn generate_gpu_instance(&self, index: usize) -> Option<Box<dyn GpuInstance>> {
        if index < self.device_count {
            Some(Box::new(
                CudaGpuInstance::new(
                    self.model_path.lock().unwrap().clone(),
                    self.batch_size,
                    index as i32,
                    self.is_tensor_rt_available,
                    self.run_folder.clone(),
                )
                .expect("Failed to create CUDA GPU instance"),
            ))
        } else {
            None
        }
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn model_path(&self) -> Option<String> {
        Some(self.model_path.lock().unwrap().clone())
    }

    fn set_model_path(&self, model_path: String) {
        let mut path_lock = self.model_path.lock().unwrap();
        *path_lock = model_path;
    }

    fn warmup(&self) {
        let start = Instant::now();
        info!(
            "Building first CUDA GPU instance for model: {}. TensorRT: {}. Might take multiple minutes on the first run due to engine creation...",
            self.model_path.lock().unwrap(),
            self.is_tensor_rt_available
        );

        let first_model = CudaGpuInstance::new(
            self.model_path.lock().unwrap().clone(),
            self.batch_size,
            0,
            self.is_tensor_rt_available,
            self.run_folder.clone(),
        )
        .expect("Failed to create the first CUDA GPU instance");

        let elapsed = start.elapsed();

        info!("First CUDA GPU instance created in {:?}", elapsed);

        drop(first_model);
    }
}
