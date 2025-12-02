use analysis_util::{VisArrow, convert_svg_to_pdf};
use chess_visualize::vis_chess_render::render_board;
use chess_visualize::vis_chess_util::get_move_color;
use colorgrad::Gradient;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::PseudoLegalMove;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::{Board, InputMapper, PolicyMapper};
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};
use maz_util::network::{auto_non_cpu_model, batcher_from, prediction};
use ort::session::Session;
use rand::SeedableRng;
use rand::prelude::StdRng;
use serde::Deserialize;
use serde_yaml;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct VisConfig {
    name: String,
    playouts: u64,
    exploration: f32,
    model_path: String,
    fen: String,
    output_file: String,
    show_number_of_top_moves: usize,
    color_map: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VisRoot {
    vis: Vec<VisConfig>,
}

fn main() {
    let file_path = "analysis/chess-visualize/vis_mcts.yaml";

    let vis_root: VisRoot = serde_yaml::from_str(
        &fs::read_to_string(file_path).expect("Failed to read visualization config file"),
    )
    .expect("Failed to parse visualization config YAML");

    for config in vis_root.vis {
        println!("Processing visualization: {}", config.name);
        if let Err(e) = run_visualization(&config) {
            eprintln!(
                "Error processing visualization for '{}': {}",
                config.name, e
            );
        }
    }
}

fn run_visualization(config: &VisConfig) -> Result<(), Box<dyn Error>> {
    let model_path = &config.model_path;
    if !Path::new(model_path).exists() {
        return Err(format!(
            "Model file not found at '{}'. Please ensure the model exists.",
            model_path
        )
        .into());
    }
    let mut model = auto_non_cpu_model(model_path, None)
        .map_err(|e| format!("Failed to load model: {:?}", e))?;

    let board = TriHexChess::new_with_fen(&config.fen.as_ref(), false)
        .map_err(|e| format!("Invalid FEN: {:?}", e))?;

    println!("Running {} MCTS playouts...", config.playouts);
    let moves_with_probs = run_mcts_and_get_moves(&mut model, &board, config)?;
    println!("MCTS complete.");

    let top_moves = moves_with_probs
        .into_iter()
        .take(config.show_number_of_top_moves)
        .collect::<Vec<_>>();

    if top_moves.is_empty() {
        println!(
            "No legal moves found or MCTS returned no results for {}.",
            config.name
        );
        return Ok(());
    }

    let max_prob = top_moves
        .iter()
        .map(|(_, prob)| *prob)
        .fold(0. / 0., f32::max); // Get max probability for normalization

    let arrows = top_moves
        .iter()
        .map(|(mv, prob)| {
            let color = if let Some(ref cmap) = config.color_map {
                let at = *prob / max_prob;
                match cmap.as_str() {
                    "viridis" => colorgrad::preset::viridis().at(at).to_css_hex(),
                    "plasma" => colorgrad::preset::plasma().at(at).to_css_hex(),
                    "cool" => colorgrad::preset::cool().at(at).to_css_hex(),
                    _ => {
                        eprintln!("Unknown color map '{}', defaulting to move color.", cmap);
                        get_move_color(*mv, &board)
                    }
                }
            } else {
                get_move_color(*mv, &board)
            };

            VisArrow {
                from_q: mv.from.to_qrs_global().q,
                from_r: mv.from.to_qrs_global().r,
                from_i: mv.from,
                to_i: mv.to,
                to_q: mv.to.to_qrs_global().q,
                to_r: mv.to.to_qrs_global().r,
                opacity: 1.0,
                color,
                text: Some(format!("{:.0}%", prob * 100.0)),
            }
        })
        .collect::<Vec<_>>();

    let document = render_board(&board, &arrows, None, None, None, false);

    let temp_svg_path = format!("{}.tmp.svg", config.output_file);
    svg::save(&temp_svg_path, &document)?;

    convert_svg_to_pdf(&temp_svg_path, &config.output_file)
        .map_err(|e| format!("PDF conversion failed: {}", e))?;

    fs::remove_file(&temp_svg_path)?;

    println!(
        "Successfully generated visualization: {}",
        config.output_file
    );

    Ok(())
}

/// Executes the MCTS algorithm for a given position.
fn run_mcts_and_get_moves(
    model: &mut Session,
    board: &TriHexChess,
    config: &VisConfig,
) -> Result<Vec<(PseudoLegalMove, f32)>, Box<dyn Error>> {
    let mapper = ChessDomainMapper;
    let mut batcher = batcher_from(model, &mapper);
    let mut settings = SearchSettings::default();
    settings.part_cpuct_exploration = config.exploration;

    let virtual_loss_weight = 3.0;
    let contempt = 0.0;

    // Use a fixed seed for reproducible results
    let mut rng = StdRng::from_seed([0; 32]);

    let mut steppable_mcts =
        SteppableMCTS::<_, 3>::new_with_capacity(board, &mapper, config.playouts, true);

    let mut requests: Vec<ProduceOutput<TriHexChess>> =
        Vec::with_capacity(batcher.get_batch_size());

    loop {
        if let Some(request) = steppable_mcts.step(
            &mut rng,
            settings.search_fpu_root,
            settings.search_fpu_child,
            virtual_loss_weight,
            settings.weights,
            &mapper,
            contempt,
        ) {
            mapper.encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
            requests.push(request);
        }

        let is_batch_full = requests.len() >= batcher.get_batch_size();
        let is_done_and_pending = steppable_mcts.is_enough_rollouts() && !requests.is_empty();

        if is_batch_full || is_done_and_pending {
            let (policy_data, value_data) = prediction(model, &batcher);

            for (i, request) in requests.iter().enumerate() {
                let policy_slice =
                    &policy_data[i * mapper.policy_len()..(i + 1) * mapper.policy_len()];
                let value_slice = &value_data[i * 3..(i + 1) * 3];

                steppable_mcts
                    .consume(
                        &mut rng,
                        &settings,
                        request.node_id,
                        ConsumeValues::ConsumeWithOptionalCallback {
                            values_net: value_slice,
                            policy_net_f16: policy_slice,
                            callback: None,
                        },
                        false,
                    )
                    .unwrap();
            }
            requests.clear();
        }

        if steppable_mcts.is_enough_rollouts() {
            break;
        }
    }

    let root_nodes = steppable_mcts.tree.root_nodes();
    let root_visits = steppable_mcts.tree.root_visits();

    if root_visits == 0 {
        return Ok(Vec::new());
    }

    let mut moves_with_probs = root_nodes
        .into_iter()
        .filter_map(|node| {
            node.last_move.map(|mv| {
                let prob = node.complete_visits as f32 / root_visits as f32;
                (mv, prob)
            })
        })
        .collect::<Vec<_>>();

    // Sort moves by their visit probability (descending)
    moves_with_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(moves_with_probs)
}
