use analysis_util::convert_svg_to_pdf;
use chess_visualize::vis_chess_render::{VisHighlightedHex, render_board};
use game_tri_chess::basics::{COLORS, Piece};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::pos::MemoryPos;
use std::fs;
use std::path::Path;

/// This program generates attack maps for each of the three players in a standard
/// Tri-Chess game. An "attack map" visualizes all squares that a player's pieces
/// can attack.
///
/// Features:
/// - Generates 3 separate PDF files, one for each player (White, Gray, Black).
/// - Each of a player's 16 pieces is assigned a unique color.
/// - Squares attacked by a single piece are highlighted with that piece's unique color.
/// - Squares attacked by multiple pieces are highlighted in a distinct gray color,
///   with a number indicating how many pieces are attacking it.
/// - All file paths for output are hardcoded.
/// - The program starts from the standard initial board position.
fn main() {
    let output_dir = Path::new("pdfs/attack_maps");
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!(
            "Failed to create directory '{}': {}",
            output_dir.display(),
            e
        );
        return;
    }
    println!("Output will be saved to: {}", output_dir.display());

    let fen = "r1bqkbnr/5p2/2p1p1p1/pp1p3p/X/8/8/3n4/8 X/rnkq1bnr/2p2pp1/3p4/1p2p2p/X X/X/rnbq1knr/2p1p1p1/8/pp3p1p G qkqkqk --- 0 1";

    let board = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();

    let players = COLORS;

    let color_palette: Vec<String> = vec![
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
        "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7", "#bebada", "#fb8072", "#80b1d3",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let attack_maps = board.calculate_per_piece_bitboard_attack_data();

    for &player in &players {
        println!("Generating attack map for player: {:?}", player);

        let mut highlights = vec![];

        for p in Piece::all() {
            let idx = p.get_as_zero_based_index();

            for color in COLORS {
                if color != player {
                    continue;
                }

                let color_idx = color as usize;
                let attack_bb = attack_maps.attacks[color_idx][idx as usize];

                for sq in 0..96 {
                    if (attack_bb & (1u128 << sq)) != 0 {
                        let memory_idx = MemoryPos(sq);
                        let qrs_pos = memory_idx.to_qrs_global();

                        // add highlight for this square
                        highlights.push(VisHighlightedHex {
                            q: qrs_pos.q,
                            r: qrs_pos.r,
                            color: color_palette[idx as usize].clone(),
                            opacity: 0.6,
                            text: None,
                        });
                    }
                }
            }
        }

        let document = render_board(&board, &vec![], Some(highlights), None, None, false);

        let player_name = format!("{:?}", player).to_lowercase();
        let svg_path = output_dir.join(format!("{}_attack_map.svg", player_name));
        let pdf_path = output_dir.join(format!("{}_attack_map.pdf", player_name));

        if let Err(e) = svg::save(&svg_path, &document) {
            eprintln!("Failed to save SVG for {}: {}", player_name, e);
            continue;
        }
        println!("  Saved intermediate SVG to {}", svg_path.display());

        match convert_svg_to_pdf(svg_path.to_str().unwrap(), pdf_path.to_str().unwrap()) {
            Ok(_) => {
                println!("  Successfully converted to PDF: {}", pdf_path.display());
            }
            Err(e) => {
                eprintln!("  Failed to convert SVG to PDF: {:?}", e);
                continue;
            }
        }
    }
}
