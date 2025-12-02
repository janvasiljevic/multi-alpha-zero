use crate::gpu::gpu_provider::{GpuInstance, GpuOut, GpuPerformanceMetadata, GpuSpawner};
use half::f16;
use ndarray::ArrayD;
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLModelFormat, CoreMLSpecializationStrategy,
};
use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::Mutex;
use maz_core::net::batch::Batch;

pub fn detect_coreml() -> bool {
    let coreml_ex_provider = CoreMLExecutionProvider::default();

    coreml_ex_provider.is_available().unwrap_or(false)
}

pub struct CoreMLInstance {
    batch_size: usize,
    model: Session,
    performance_metadata: GpuPerformanceMetadata,
}

impl CoreMLInstance {
    pub fn new(model_path: String, batch_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(batch_size > 0, "Batch size must be greater than 0");

        let model = Session::builder()?
            .with_execution_providers([CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::All)
                .with_subgraphs(false)
                .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction)
                .with_low_precision_accumulation_on_gpu(true)
                .build()
                .error_on_failure()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;

        Ok(Self {
            batch_size,
            model,
            performance_metadata: GpuPerformanceMetadata::default(),
        })
    }
}

impl GpuInstance for CoreMLInstance {
    fn get_performance_metadata(&self) -> GpuPerformanceMetadata {
        self.performance_metadata
    }

    fn run_with_batch(&mut self, batch: &Batch, count: usize) -> GpuOut {
        let mut _outputs = self
            .model
            .run(ort::inputs![
               "input" => TensorRef::from_array_view(batch.view()).unwrap()
            ])
            .expect("Failed to run model");

        let policy_logits_dyn = _outputs
            .remove("policy_logits")
            .expect("No 'policy_logits' output found");

        let policy_logits: ArrayD<f16> = policy_logits_dyn
            .try_extract_array::<f16>()
            .expect("Failed to extract 'policy_logits' as ArrayView2")
            .into_owned();

        let value_dyn = _outputs.remove("value").expect("No 'value' output found");

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

pub struct CoreMLGpuSpawner {
    model_path: Mutex<String>,
    batch_size: usize,
    sessions_per_gpu: usize,
}

impl CoreMLGpuSpawner {
    pub fn new(model_path: String, batch_size: usize, sessions_per_gpu: usize) -> Self {
        assert!(batch_size > 0, "Batch size must be greater than 0");
        Self {
            model_path: Mutex::new(model_path),
            batch_size,
            sessions_per_gpu,
        }
    }
}

impl GpuSpawner for CoreMLGpuSpawner {
    fn get_gpu_count(&self) -> usize {
        1
    }

    fn sessions_per_gpu(&self) -> usize {
        self.sessions_per_gpu
    }

    fn generate_gpu_instance(&self, index: usize) -> Option<Box<dyn GpuInstance>> {
        if index == 0 {
            Some(Box::new(
                CoreMLInstance::new(self.model_path.lock().unwrap().clone(), self.batch_size)
                    .expect("Failed to create CoreMLInstance"),
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
}
