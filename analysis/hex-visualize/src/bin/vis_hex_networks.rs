use analysis_util::convert_svg_to_pdf;
use colorgrad::Gradient;
use game_hex::game_hex::HexPlayer::{P1, P2, P3};
use game_hex::game_hex::{HexGame, HexPlayer};
use hex_visualize::vis_hex::{VisHexHighlight, render_hex_board};
use hex_visualize::vis_hex_board::VisHexBoard;
use maz_core::mapping::hex_absolute_mapper::HexAbsoluteMapper;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_core::mapping::hex_wrapper_mapper::HexWrapperMapper;
use maz_core::mapping::{InputMapper, PolicyMapper};
use maz_core::values_const::ValuesAbs;
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};
use maz_util::network::{auto_non_cpu_model, batcher_from, prediction};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng, rng};
use std::fs;

fn create_random_highlights(game: HexGame) -> Vec<VisHexHighlight> {
    let mut highlights = Vec::new();

    let mut rng = rng();

    let valid_hexes = game.get_valid_empty_cells();

    let magma_gradient = colorgrad::preset::viridis();

    for hex in valid_hexes {
        let rand_val: f64 = rng.random_range(0.0..1.0);

        let color = magma_gradient.at(rand_val as f32).to_css_hex();

        highlights.push(VisHexHighlight {
            q: hex.q,
            r: hex.r,
            color,
            opacity: 1.0,
        });
    }

    highlights
}

pub fn default_symmetric_with_player(player: HexPlayer) -> HexGame {
    let mut game = HexGame::new(5).unwrap();
    game.current_turn = player;

    // game.set_to_player(0, 0, P1);

    // test
    // game.set_to_player(-1, 0, P1);
    // game.set_to_player(1, 0, P1);
    //
    game.set_to_player(-2, 0, P1);
    game.set_to_player(2, 0, P1);

    game.set_to_player(-2, 2, P2);
    game.set_to_player(2, -2, P2);

    game.set_to_player(0, -2, P3);
    game.set_to_player(0, 2, P3);

    game
}

struct HexVisNetwork {
    name: String,
    board: HexGame,
    maybe_network_path: Option<String>,
}

fn main() {
    let demos = vec![
        HexVisNetwork {
            name: "random".to_string(),
            board: HexGame::new(5).unwrap(),
            maybe_network_path: None,
        },
        // abs p1, p2, p3 to show that the absolute network learns different strategies
        HexVisNetwork {
            name: "abs_p1".to_string(),
            board: default_symmetric_with_player(P1),
            maybe_network_path: Some("testing/arena/Hex5Abs/Hex5Abs_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "abs_p2".to_string(),
            board: default_symmetric_with_player(P2),
            maybe_network_path: Some("testing/arena/Hex5Abs/Hex5Abs_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "abs_p3".to_string(),
            board: default_symmetric_with_player(P3),
            maybe_network_path: Some("testing/arena/Hex5Abs/Hex5Abs_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "canon_p1".to_string(),
            board: default_symmetric_with_player(P1),
            maybe_network_path: Some("testing/arena/Hex5Canon/Hex5Canon_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "canon_p2".to_string(),
            board: default_symmetric_with_player(P2),
            maybe_network_path: Some("testing/arena/Hex5Canon/Hex5Canon_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "canon_p3".to_string(),
            board: default_symmetric_with_player(P3),
            maybe_network_path: Some("testing/arena/Hex5Canon/Hex5Canon_110.onnx".to_string()),
        },
        HexVisNetwork {
            name: "axiom_p1".to_string(),
            board: default_symmetric_with_player(P1),
            maybe_network_path: Some("testing/arena/Hex5Axiom/Hex5Axiom_140.onnx".to_string()),
        },
        HexVisNetwork {
            name: "axiom_p2".to_string(),
            board: default_symmetric_with_player(P2),
            maybe_network_path: Some("testing/arena/Hex5Axiom/Hex5Axiom_140.onnx".to_string()),
        },
        HexVisNetwork {
            name: "axiom_p3".to_string(),
            board: default_symmetric_with_player(P3),
            maybe_network_path: Some("testing/arena/Hex5Axiom/Hex5Axiom_140.onnx".to_string()),
        },
    ];

    let vis_board = VisHexBoard::new(4);

    let dir = std::env::var("HEX_VIS_NN_RES_DIR").unwrap_or("pdfs/hex/network".to_string());
    let dir = dir.as_str();

    if let Err(_e) = fs::create_dir_all(dir) {
        return;
    }

    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file()
            && (path.extension().unwrap_or_default() == "svg"
                || path.extension().unwrap_or_default() == "pdf")
        {
            fs::remove_file(path).unwrap();
        }
    }

    for demo in demos {
        let (eval, highlights) = if let Some(_network_path) = demo.maybe_network_path {
            // Here you would load the network and generate highlights based on its evaluation
            // For simplicity, we'll just create random highlights
            let mut onnx_session = auto_non_cpu_model(_network_path.as_str(), None).unwrap();

            let mapper = if demo.name.contains("abs") {
                HexWrapperMapper::Absolute(HexAbsoluteMapper::new(&demo.board))
            } else {
                HexWrapperMapper::Canonical(HexCanonicalMapper::new(&demo.board))
            };

            let mut batcher = batcher_from(&onnx_session, &mapper);

            // seeded rng (32 for axiom)
            let mut rng = StdRng::from_seed([2u8; 32]);

            let mut steppable_mcts =
                SteppableMCTS::<_, 3>::new_with_capacity(&demo.board.clone(), &mapper, 1600, true);

            let mut requests: Vec<ProduceOutput<HexGame>> =
                Vec::with_capacity(batcher.get_batch_size());

            let mut settings = SearchSettings::default();

            // settings.weights.exploration_weight = 0.7;
            settings.weights.exploration_weight = 1.4;

            loop {
                if let Some(request) = steppable_mcts.step(
                    &mut rng,
                    settings.search_fpu_root,
                    settings.search_fpu_child,
                    settings.search_virtual_loss_weight,
                    settings.weights,
                    &mapper,
                    0.0,
                ) {
                    _ = &mapper
                        .encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
                    requests.push(request);
                }

                if requests.len() >= batcher.get_batch_size() {
                    let (policy_data, value_data) = prediction(&mut onnx_session, &batcher);

                    for (i, request) in requests.iter().enumerate() {
                        let policy_data = policy_data
                            [i * mapper.policy_len()..(i + 1) * mapper.policy_len()]
                            .as_ref();

                        let value_data = value_data[i * 3..(i + 1) * 3].as_ref();

                        steppable_mcts
                            .consume(
                                &mut rng,
                                &settings,
                                request.node_id,
                                ConsumeValues::ConsumeWithOptionalCallback {
                                    values_net: value_data,
                                    policy_net_f16: policy_data,
                                    callback: None,
                                },
                                false,
                            )
                            .expect("Failed to consume values");
                    }

                    requests.clear();

                    if steppable_mcts.is_enough_rollouts() {
                        break;
                    }
                }
            }

            let gradient = colorgrad::preset::bu_pu();

            // Print root eval
            let root_values = steppable_mcts.tree.root_q_abs_values();
            let root_values = root_values.value_abs;

            println!(
                "Network {} root eval: P1 {:.3} P2 {:.3} P3 {:.3}",
                demo.name, root_values[0], root_values[1], root_values[2]
            );

            let max_visits = steppable_mcts
                .tree
                .root_nodes()
                .map(|mv| mv.complete_visits)
                .max()
                .unwrap_or(1) as f32;

            (
                ValuesAbs::<3> {
                    value_abs: root_values,
                    moves_left: 0.0,
                },
                steppable_mcts
                    .tree
                    .root_nodes()
                    .map(|mv| {
                        let prob = mv.complete_visits as f32 / max_visits;
                        let color = gradient.at(prob).to_css_hex();

                        VisHexHighlight {
                            q: mv.last_move.unwrap().q,
                            r: mv.last_move.unwrap().r,
                            color,
                            opacity: 1.0,
                        }
                    })
                    .collect(),
            )
        } else {
            let rng = &mut rng();
            let random_abs = ValuesAbs::<3> {
                value_abs: [
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                ],
                moves_left: 0.0,
            };

            (random_abs, create_random_highlights(demo.board.clone()))
        };

        let document =
            render_hex_board(&demo.board, &vis_board, Some(highlights), Some(eval), false);

        let svg_filename = format!("{dir}/{name}.svg", dir = dir, name = demo.name);

        svg::save(svg_filename.as_str(), &document).unwrap();

        if let Err(e) = convert_svg_to_pdf(
            svg_filename.as_str(),
            format!("{dir}/{name}.pdf", dir = dir, name = demo.name).as_str(),
        ) {
            eprintln!("Failed to convert SVG to PDF: {}", e);
        }
    }
}
