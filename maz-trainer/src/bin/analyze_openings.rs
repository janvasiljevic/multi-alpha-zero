use ahash::{HashMap, HashMapExt};
use clap::Parser; // NEW: Using clap for argument parsing
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::PseudoLegalMove;
use maz_core::mapping::chess_wrapper_mapper::{ChessWrapperMapper, ALL_CHESS_MAPPERS};
use maz_core::mapping::{Board, InputMapper, PolicyMapper};
use maz_core::net::batch::Batch;
use maz_util::network::{
    alpha_zero_shapes, auto_non_cpu_model, batcher_from, prediction, NetShapesInfo,
};
use ort::session::Session;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*; // NEW: Import Rayon traits
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Line {
    name: String,
    moves: Vec<String>,
    times_played: usize,
}

// NEW: Command-line arguments struct using clap
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the folder containing .onnx models
    #[arg(long)]
    onnx_folder: PathBuf,

    /// Path to the output YAML file
    #[arg(long)]
    output_file: PathBuf,

    /// Number of games to play per model
    #[arg(long)]
    games_per_model: usize,

    /// Number of models to process concurrently
    #[arg(long, default_value_t = 3)]
    concurrent_models: usize,
}

fn main() {
    // NEW: Parse arguments using clap
    let args = Args::parse();

    // 1. Collect all valid model paths first
    let paths: Vec<PathBuf> = std::fs::read_dir(&args.onnx_folder)
        .expect("Failed to read ONNX models folder")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("onnx"))
        .collect();

    println!(
        "Found {} models. Processing in chunks of {}...",
        paths.len(),
        args.concurrent_models
    );

    let mut model_moves: HashMap<String, HashMap<String, Line>> = HashMap::new();

    // 2. Process paths in chunks to control concurrency
    for chunk in paths.chunks(args.concurrent_models) {
        // 3. Use Rayon to process the current chunk in parallel
        let results: Vec<_> = chunk
            .par_iter()
            .map(|path| {
                println!("Processing model: {:?}", path.file_stem().unwrap());
                process_model(path, args.games_per_model)
            })
            .collect();

        // 4. Collect results from the parallel processing
        for result in results {
            match result {
                Ok((model_name, lines)) => {
                    model_moves.insert(model_name, lines);
                }
                Err(e) => eprintln!("Error processing model: {}", e),
            }
        }
    }

    // Write all to a yaml file
    println!("Writing results to {:?}", &args.output_file);
    let serialized = serde_yaml::to_string(&model_moves).expect("Failed to serialize model moves");
    std::fs::write(&args.output_file, serialized).expect("Failed to write output file");
    println!("Done.");
}

// NEW: Extracted function to process a single model. This is the unit of work for Rayon.
fn process_model(
    model_path: &Path,
    num_games: usize,
) -> Result<(String, HashMap<String, Line>), String> {
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap()
        .to_string();

    let mut model = auto_non_cpu_model(model_path.to_str().unwrap(), Some(0))
        .map_err(|e| format!("Failed to load model {:?}: {}", model_path, e))?;

    let model_shapes =
        alpha_zero_shapes(&model).map_err(|e| format!("Failed to get shapes for {:?}: {}", model_path, e))?;

    let mapper = chess_mapper_from_network_shapes(model_shapes.clone())
        .map_err(|e| format!("Mapper error for {:?}: {}", model_path, e))?;

    let mut batcher = batcher_from(&mut model, &mapper);

    // MODIFIED: Create RNG once per model/thread
    let mut rng = StdRng::from_os_rng();

    let mut lines_for_model = HashMap::new();

    for _ in 0..num_games {
        let moves = play_n_moves(
            &mut model,
            &mapper,
            &mut batcher,
            model_shapes.clone(),
            9,
            &mut rng, // Pass RNG by reference
        );

        let move_strings: Vec<String> = moves.iter().map(|m| m.notation_hyphen()).collect();
        let line_key = move_strings.join(" ");

        lines_for_model
            .entry(line_key.clone())
            .or_insert_with(|| Line {
                name: line_key,
                moves: move_strings,
                times_played: 0,
            })
            .times_played += 1;
    }

    Ok((model_name, lines_for_model))
}

// MODIFIED: Accepts a mutable reference to the RNG
pub fn play_n_moves(
    model: &mut Session,
    mapper: &ChessWrapperMapper,
    batcher: &mut Batch,
    shapes: NetShapesInfo,
    num_moves: usize,
    rng: &mut StdRng,
) -> Vec<PseudoLegalMove> {
    let mut board = TriHexChess::default_with_grace_period();
    let mut move_store = <TriHexChess as Board>::MoveStore::default();
    let settings = SearchSettings::default();

    let mut moves_played = Vec::new();

    for _ in 0..num_moves {
        // Re-initialize MCTS for each move
        let mut steppable_mcts =
            SteppableMCTS::<TriHexChess, 3>::new_with_capacity(&board, mapper, 200, true);

        let mut requests: Vec<ProduceOutput<TriHexChess>> =
            Vec::with_capacity(batcher.get_batch_size());

        let mut i = 0;
        let virtual_size_batch = shapes.batch_size;
        loop {
            if let Some(request) = steppable_mcts.step(
                rng,
                settings.search_fpu_root,
                settings.search_fpu_child,
                settings.search_virtual_loss_weight,
                settings.weights,
                mapper,
                settings.contempt,
            ) {
                mapper.encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
                requests.push(request);
            }

            i += 1;

            if requests.len() >= batcher.get_batch_size() || i >= virtual_size_batch {
                let (policy_data, value_data) = prediction(model, batcher);

                for (i, request) in requests.iter().enumerate() {
                    let policy_net_f16 =
                        &policy_data[i * mapper.policy_len()..(i + 1) * mapper.policy_len()];
                    let values_net = &value_data[i * 3..(i + 1) * 3];

                    steppable_mcts
                        .consume(
                            rng,
                            &settings,
                            request.node_id,
                            ConsumeValues::ConsumeWithOptionalCallback {
                                policy_net_f16,
                                values_net,
                                callback: None,
                            },
                            false,
                        )
                        .expect("Consume failed");
                }
                requests.clear();
                i = 0;

                if steppable_mcts.is_enough_rollouts() {
                    break;
                }
            }
        }

        let nodes = steppable_mcts.tree.root_nodes();
        let nodes_moves_count: Vec<f32> = nodes.map(|n| n.complete_visits as f32).collect();

        let distribution = match WeightedIndex::new(&nodes_moves_count) {
            Ok(dist) => dist,
            Err(_) => {
                eprintln!("Warning: Could not create weighted distribution, possibly no valid moves or rollouts. Ending game early.");
                break;
            }
        };

        let selected_index = rng.sample(distribution);
        let selected_node = steppable_mcts.tree.root_nodes().nth(selected_index).unwrap();
        let chosen_move = selected_node.last_move.unwrap();

        moves_played.push(chosen_move);
        board.fill_move_store(&mut move_store);
        board.play_move_mut_with_store(&chosen_move, &mut move_store, None);
    }

    moves_played
}

pub fn chess_mapper_from_network_shapes(
    input_shape: NetShapesInfo,
) -> Result<ChessWrapperMapper, String> {
    let input_shape = input_shape.input_shape[1..3].to_vec();
    let shape_size = input_shape[0] * input_shape[1];
    for mapper in ALL_CHESS_MAPPERS {
        let mapper_shape = mapper.input_board_shape();
        if mapper_shape[0] * mapper_shape[1] == shape_size {
            return Ok(mapper);
        }
    }

    Err(format!(
        "No suitable ChessWrapperMapper found for input shape {:?}",
        input_shape
    ))
}