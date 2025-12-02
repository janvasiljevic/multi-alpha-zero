use crate::math::softmax_in_place;
use half::f16;
use maz_core::mapping::{Board, InputMapper, MoveStore, PolicyMapper};
use maz_core::net::batch::Batch;
use ndarray::ArrayD;
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLModelFormat, CoreMLSpecializationStrategy,
};
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

pub fn policy_to_hashmap<B: Board>(
    policy: &[f32],
    board: &B,
    move_store: &B::MoveStore,
    mapper: &impl PolicyMapper<B>,
) -> HashMap<B::Move, f32> {
    policy
        .iter()
        .enumerate()
        .map(|(i, &p)| (mapper.index_to_move(&board, move_store, i), p))
        .filter(|(mv, _)| mv.is_some())
        .map(|(mv, p)| (mv.unwrap(), p))
        .collect()
}

pub fn policy_to_hashmap_f16<B: Board>(
    policy: &[f16],
    board: &B,
    move_store: &B::MoveStore,
    mapper: &impl PolicyMapper<B>,
) -> HashMap<B::Move, f32> {
    policy
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            (
                mapper.index_to_move(&board, move_store, i).unwrap(),
                p.to_f32(),
            )
        })
        .collect()
}

pub fn cpu_model(path: &str, graph_optimization_level: GraphOptimizationLevel) -> Result<Session, Box<dyn Error>> {
    let model = Session::builder()?
        .with_execution_providers([
            ort::execution_providers::CPUExecutionProvider::default().build()
        ])?
        .with_optimization_level(graph_optimization_level)?
        .commit_from_file(path)?;

    Ok(model)
}

fn get_engine_cache_path() -> (String, String) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let project_root = PathBuf::from(manifest_dir).parent().unwrap().to_path_buf();

    let cache_dir = project_root.join("cache");

    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    let tensorrt_engine_cache = cache_dir.join("tensorrt_engine_cache");

    if !tensorrt_engine_cache.exists() {
        std::fs::create_dir_all(&tensorrt_engine_cache)
            .expect("Failed to create TensorRT engine cache directory");
    }

    let tensorrt_timing_cache = cache_dir.join("tensorrt_timing_cache");
    if !tensorrt_timing_cache.exists() {
        std::fs::create_dir_all(&tensorrt_timing_cache)
            .expect("Failed to create TensorRT timing cache directory");
    }

    (
        tensorrt_engine_cache.to_str().unwrap().to_string(),
        tensorrt_timing_cache.to_str().unwrap().to_string(),
    )
}

pub fn auto_non_cpu_model(
    model_path: &str,
    maybe_device_id: Option<usize>,
) -> Result<Session, Box<dyn Error>> {
    let (engine_cache_path, timing_cache_path) = get_engine_cache_path();

    let model = Session::builder()?
        .with_execution_providers([
            TensorRTExecutionProvider::default()
                .with_device_id(maybe_device_id.unwrap_or(0) as i32)
                .with_cuda_graph(false)
                .with_builder_optimization_level(5)
                .with_layer_norm_fp32_fallback(true)
                // .with_engine_cache(true)
                // .with_engine_cache_path(engine_cache_path)
                // .with_timing_cache(true)
                // .with_timing_cache_path(timing_cache_path)
                .with_fp16(true)
                .build(),
            CUDAExecutionProvider::default()
                .with_device_id(maybe_device_id.unwrap_or(0) as i32)
                .build(),
            CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::All)
                .with_subgraphs(false)
                .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction)
                .with_low_precision_accumulation_on_gpu(true)
                .build(),
        ])?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_parallel_execution(false)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    Ok(model)
}

pub fn batcher_from<B: Board>(model: &Session, mapper: &impl InputMapper<B>) -> Batch {
    let [board_size, field_size] = mapper.input_board_shape();

    let mut batch_size: Option<usize> = None;

    for input in model.inputs.iter() {
        if let ValueType::Tensor { shape, .. } = &input.input_type {
            batch_size = shape
                .first()
                .and_then(|&s| if s > 0 { Some(s as usize) } else { None });
        }
    }

    Batch::new(
        batch_size.expect("Model does not have a valid batch size"),
        board_size,
        field_size,
    )
}

pub fn prediction(model: &mut Session, batcher: &Batch) -> (Arc<[f16]>, Arc<[f32]>) {
    let mut _outputs = model
        .run(ort::inputs![
            "input" => TensorRef::from_array_view(batcher.view()).unwrap()
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

    let policy_data: Arc<[f16]> = Arc::from(policy_logits.view().to_slice().unwrap());
    let value_data: Arc<[f32]> = Arc::from(value.view().to_slice().unwrap());

    (policy_data, value_data)
}

pub fn prediction_special(
    model: &mut Session,
    batcher: &Batch,
) -> (Arc<[f16]>, Arc<[f32]>, Arc<[f32]>, Arc<[f32]>, Arc<[f32]>) {
    let mut _outputs = model
        .run(ort::inputs![
            "input" => TensorRef::from_array_view(batcher.view()).unwrap()
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

    let material_dyn = _outputs
        .remove("material")
        .expect("No 'material' output found");
    let material: ArrayD<f32> = material_dyn
        .try_extract_array()
        .expect("Failed to extract 'material' as ArrayView2")
        .into_owned();

    let from_logits_dyn = _outputs
        .remove("from_logits")
        .expect("No 'from_logits' output found");

    let from_logits: ArrayD<f32> = from_logits_dyn
        .try_extract_array::<f32>()
        .expect("Failed to extract 'from_logits' as ArrayView2")
        .into_owned();

    let to_logits_dyn = _outputs
        .remove("to_logits")
        .expect("No 'to_logits' output found");

    let to_logits: ArrayD<f32> = to_logits_dyn
        .try_extract_array::<f32>()
        .expect("Failed to extract 'to_logits' as ArrayView2")
        .into_owned();

    let policy_data: Arc<[f16]> = Arc::from(policy_logits.view().to_slice().unwrap());
    let value_data: Arc<[f32]> = Arc::from(value.view().to_slice().unwrap());
    let material_data: Arc<[f32]> = Arc::from(material.view().to_slice().unwrap());
    let from_logits_data: Arc<[f32]> = Arc::from(from_logits.view().to_slice().unwrap());
    let to_logits_data: Arc<[f32]> = Arc::from(to_logits.view().to_slice().unwrap());

    (
        policy_data,
        value_data,
        material_data,
        from_logits_data,
        to_logits_data,
    )
}

pub fn prediction_with_post_processing<B: Board>(
    model: &mut Session,
    batcher: &Batch,
    board: &mut B,
    mapper: &impl PolicyMapper<B>,
) -> (HashMap<B::Move, f32>, Vec<f32>) {
    let mut move_store = <B as Board>::MoveStore::default();
    board.fill_move_store(&mut move_store);

    let (policy_data, value_data) = prediction(model, batcher);

    let mut policy_slice = policy_data[0..mapper.policy_len()]
        .to_vec()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<f32>>();

    let value_slice = &value_data[0..board.player_num()].to_vec();

    let mut legal_mask = vec![false; mapper.policy_len()];

    for mv in move_store.iter() {
        legal_mask[mapper.move_to_index(board.player_current(), mv)] = true;
    }

    for (i, is_legal) in legal_mask.iter().enumerate() {
        if !is_legal {
            policy_slice[i] = f32::NEG_INFINITY;
        }
    }

    softmax_in_place(policy_slice.as_mut_slice());

    let policy_map = policy_to_hashmap(policy_slice.as_slice(), board, &move_store, mapper);

    (policy_map, value_slice.to_vec())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetShapesInfo {
    pub input_shape: [usize; 3],
    pub policy_logits_shape: [usize; 2],
    pub value_shape: [usize; 2],
    pub batch_size: usize,
}

pub fn alpha_zero_shapes(model: &Session) -> Result<NetShapesInfo, Box<dyn Error>> {
    let input_shape: [usize; 3] = model
        .inputs
        .iter()
        .find(|input| input.name == "input")
        .ok_or("No 'input' found")?
        .input_type
        .tensor_shape()
        .ok_or("Failed to get tensor shape for 'input'")?
        .iter()
        .map(|&dim| dim as usize)
        .collect::<Vec<_>>()
        .as_slice()
        .try_into()
        .map_err(|_| "Input shape must be of length 3")?;

    let policy_logits_shape: [usize; 2] = model
        .outputs
        .iter()
        .find(|output| output.name == "policy_logits")
        .ok_or("No 'policy_logits' found")?
        .output_type
        .tensor_shape()
        .ok_or("Failed to get tensor shape for 'policy_logits'")?
        .iter()
        .map(|&dim| dim as usize)
        .collect::<Vec<_>>()
        .as_slice()
        .try_into()
        .map_err(|_| "Policy logits shape must be of length 2")?;

    let value_shape: [usize; 2] = model
        .outputs
        .iter()
        .find(|output| output.name == "value")
        .ok_or("No 'value' found")?
        .output_type
        .tensor_shape()
        .ok_or("Failed to get tensor shape for 'value'")?
        .iter()
        .map(|&dim| dim as usize)
        .collect::<Vec<_>>()
        .as_slice()
        .try_into()
        .map_err(|_| "Value shape must be of length 2")?;

    if input_shape[0] != policy_logits_shape[0] || input_shape[0] != value_shape[0] {
        return Err("Batch size mismatch between input, policy_logits, and value shapes".into());
    }

    Ok(NetShapesInfo {
        input_shape,
        policy_logits_shape,
        value_shape,
        batch_size: input_shape[0],
    })
}
