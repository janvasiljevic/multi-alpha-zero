use analysis_util::{convert_svg_to_pdf, VisArrow};
use chess_visualize::vis_chess_render::{render_board, VisHighlightedHex};
use colorgrad::Gradient;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use game_tri_chess::pos::MemoryPos;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::{Board, InputMapper};
use maz_util::network::{auto_non_cpu_model, batcher_from, cpu_model, prediction_special, prediction_with_post_processing};
use std::fs;
use ort::session::builder::GraphOptimizationLevel;
use chess_visualize::vis_chess_util::get_move_color;

struct ToVis {
    name: String,
    fen: String,
}

fn main() {
    let fens_content = std::fs::read_to_string("analysis/chess-visualize/fens.txt")
        .expect("Failed to read fens.txt");
    let mut demos = Vec::new();

    for (i, line) in fens_content.lines().enumerate() {
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        demos.push(ToVis {
            name: format!("{}_pos", i + 1),
            fen: fen.to_string(),
        });
    }

    let dir =
        std::env::var("CHESS_VIS_OPENING_DIR").unwrap_or("pdfs/chess/policy_decomp".to_string());
    let dir = dir.as_str();

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create directory {}: {}", dir, e);
        return;
    }

    let mut model =
        cpu_model("testing/debug/ChessDomain32/ChessDomain_850.onnx", GraphOptimizationLevel::Disable).unwrap();
    let mapper = ChessDomainMapper;
    let mut batcher = batcher_from(&model, &mapper);

    for demo in demos {
        let mut board = TriHexChess::new_with_fen(demo.fen.as_ref(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        board.fill_move_store(&mut move_store);

        let mut input = batcher.get_mut_item(0);
        mapper.encode_input(&mut input, &board);

        let (policy, _, material, from, to) = prediction_special(&mut model, &mut batcher);

        let from = from[0..96].to_vec();
        let to = to[0..96].to_vec();

        println!("From logits: {:?}", from);
        println!("To logits: {:?}", to);

        let name = demo.name;

        let max_from = from.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_from = from.iter().cloned().fold(f32::INFINITY, f32::min);

        let magma_gradient = colorgrad::preset::viridis();

        let offset = board.get_turn().unwrap().get_offset();

        let from_highlights = from
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let pos = MemoryPos(i as u8).to_global(offset);
                let qrs = pos.to_qrs_global();
                let color = magma_gradient
                    .at(((*v - min_from) / (max_from - min_from)).into())
                    .to_css_hex();

                let text = if (*v - max_from).abs() < 0.0001 || (*v - min_from).abs() < 0.0001 {
                    Some(format!("{:.2}", *v))
                } else {
                    None
                };

                VisHighlightedHex {
                    q: qrs.q,
                    r: qrs.r,
                    color,
                    opacity: 1.0,
                    text,
                }
            })
            .collect::<Vec<VisHighlightedHex>>();

        let max_to = to.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_to = to.iter().cloned().fold(f32::INFINITY, f32::min);

        let to_highlights = to
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let pos = MemoryPos(i as u8).to_global(offset);
                let qrs = pos.to_qrs_global();
                let color = magma_gradient
                    .at(((*v - min_to) / (max_to - min_to)).into())
                    .to_css_hex();

                let text = if (*v - max_to).abs() < 0.0001 || (*v - min_to).abs() < 0.0001 {
                    Some(format!("{:.2}", *v))
                } else {
                    None
                };

                VisHighlightedHex {
                    q: qrs.q,
                    r: qrs.r,
                    color,
                    opacity: 1.0,
                    text,
                }
            })
            .collect::<Vec<VisHighlightedHex>>();

        let from_document = render_board(&board, &vec![], Some(from_highlights), None, None, false);

        let svg_filename_from = format!("{dir}/{name}_from.svg");

        svg::save(svg_filename_from.as_str(), &from_document).unwrap();

        convert_svg_to_pdf(
            svg_filename_from.as_str(),
            format!("{dir}/{name}_from.pdf").as_str(),
        )
        .unwrap();

        let to_document = render_board(&board, &vec![], Some(to_highlights), None, None, false);
        let svg_filename = format!("{dir}/{name}_to.svg");
        svg::save(svg_filename.as_str(), &to_document).unwrap();

        convert_svg_to_pdf(
            svg_filename.as_str(),
            format!("{dir}/{name}_to.pdf").as_str(),
        )
        .unwrap();

        // delete from and to svg files
        fs::remove_file(svg_filename_from.as_str()).unwrap();
        fs::remove_file(svg_filename.as_str()).unwrap();

        // Finally, render a board without highlights
        let plain_document = render_board(&board, &vec![], None, None, None, false);
        let plain_svg_filename = format!("{dir}/{name}_plain.svg");

        svg::save(plain_svg_filename.as_str(), &plain_document).unwrap();
        convert_svg_to_pdf(
            plain_svg_filename.as_str(),
            format!("{dir}/{name}_plain.pdf").as_str(),
        )
        .unwrap();
        fs::remove_file(plain_svg_filename.as_str()).unwrap();

        let (mut moves_with_policy, _) =
            prediction_with_post_processing(&mut model, &mut batcher, &mut board, &mapper);

        // filter out moves with probability less than 0.1
        moves_with_policy.retain(|_mv, prob| *prob >= 0.1);


        let arrows = moves_with_policy
            .iter()
            .map(|(mv, _prob)| {
                let mv_color = get_move_color(*mv, &board);

                VisArrow {
                    from_q: mv.from.to_qrs_global().q,
                    from_r: mv.from.to_qrs_global().r,
                    from_i: mv.from,
                    to_i: mv.to,
                    to_q: mv.to.to_qrs_global().q,
                    to_r: mv.to.to_qrs_global().r,
                    opacity: 1.0,
                    color: mv_color,
                    text: Some(format!("{:.0}%", _prob * 100.0)),
                }
            })
            .collect::<Vec<_>>();


        let policy_document = render_board(&board, &arrows, None, None, None, false);
        let policy_svg_filename = format!("{dir}/{name}_policy.svg");
        svg::save(policy_svg_filename.as_str(), &policy_document).unwrap();
        convert_svg_to_pdf(
            policy_svg_filename.as_str(),
            format!("{dir}/{name}_policy.pdf").as_str(),
        )
        .unwrap();
        fs::remove_file(policy_svg_filename.as_str()).unwrap();
    }
}
