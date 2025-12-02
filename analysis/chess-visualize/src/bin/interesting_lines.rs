use serde::{Deserialize, Serialize};
use analysis_util::{convert_svg_to_pdf, VisArrow};
use chess_visualize::vis_chess_render::render_board;
use chess_visualize::vis_chess_util::get_move_color;
use game_tri_chess::chess_game::TriHexChess;
use maz_core::mapping::Board;

#[derive(Debug, Serialize, Deserialize)]
pub struct PoseSet {
    pub poses: Vec<Pose>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pose {
    pub name: String,
    pub fen: String,
    #[serde(default)]
    pub moves: Vec<String>,
}

fn save_to_svg_then_pdf_delete_svg(document: &svg::Document, filename_base: &str) {
    let svg_filename = format!("{}.svg", filename_base);
    let pdf_filename = format!("{}.pdf", filename_base);

    svg::save(svg_filename.as_str(), document).unwrap();

    convert_svg_to_pdf(svg_filename.as_str(), pdf_filename.as_str()).unwrap();

    std::fs::remove_file(svg_filename).expect("Unable to delete temporary SVG file");
}

fn main() {
    let file_path = "./analysis/chess-visualize/interesting-lines.yaml";
    let out_dir = "pdfs/chess/interesting_lines";

    std::fs::create_dir_all(out_dir).expect("Unable to create output directory");

    for entry in std::fs::read_dir(out_dir).expect("Unable to read output directory") {
        let entry = entry.expect("Unable to get directory entry");
        let path = entry.path();
        if path.is_file() && (path.extension().and_then(|s| s.to_str()) == Some("pdf")) {
            std::fs::remove_file(path).expect("Unable to delete file");
        }
    }

    let yaml_content = std::fs::read_to_string(file_path).expect("Unable to read file");
    let pose_set: PoseSet =
        serde_yaml::from_str(&yaml_content).expect("Unable to parse YAML content");

    for pose in pose_set.poses {
        println!("Pose Name: {}", pose.name);
        println!("FEN: {}", pose.fen);
        println!("Moves: {:?}", pose.moves);
        println!("---------------------------");

        let mut board = TriHexChess::new_with_fen(pose.fen.as_ref(), false).unwrap();
        let mut move_store = game_tri_chess::moves::ChessMoveStore::default();

        board.fill_move_store(&mut move_store);

        let document = render_board(&board, vec![].as_ref(), None, None, None, false);

        let filename_base = format!("{}/{}_0", out_dir, pose.name);
        save_to_svg_then_pdf_delete_svg(&document, filename_base.as_str());

        // Then play each move and save the board state
        let mut current_board = board;

        for (i, mv_str) in pose.moves.iter().enumerate() {

            let mut vis_arrows = vec![];

            for chess_move in move_store.into_iter() {
                if chess_move.notation_lan(&board.state.buffer) == *mv_str {
                    let mv_color = get_move_color(chess_move, &board);

                    vis_arrows.push(VisArrow {
                        from_q: chess_move.from.to_qrs_global().q,
                        from_r: chess_move.from.to_qrs_global().r,
                        from_i: chess_move.from,
                        to_i: chess_move.to,
                        to_q: chess_move.to.to_qrs_global().q,
                        to_r: chess_move.to.to_qrs_global().r,
                        opacity: 1.0,
                        color: mv_color,
                        text: Some(format!("{}", chess_move.notation_lan(&current_board.state.buffer))),
                    });

                    current_board.play_move_mut_with_store(&chess_move, &mut move_store, None);
                    let document = render_board(&current_board, &vis_arrows, None, None, None, false);

                    let filename_base = format!("{}/{}_move_{}", out_dir, pose.name, i + 1);
                    
                    println!("Preety debug: {}", current_board.fancy_debug());

                    save_to_svg_then_pdf_delete_svg(&document, filename_base.as_str());
                    break;
                }
            }
        }

    }
}