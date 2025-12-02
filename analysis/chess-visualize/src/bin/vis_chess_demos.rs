use analysis_util::convert_svg_to_pdf;
use chess_visualize::vis_chess_demo_move::chess_moves_to_arrows;
use chess_visualize::vis_chess_render::{VisHighlightedHex, render_board};
use game_tri_chess::basics::{Color, MemoryBuffer, NormalState};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use game_tri_chess::phase::Phase::Normal;
use game_tri_chess::pos::{MemoryPos, QRSPos};
use maz_core::mapping::Board;
use maz_core::values_const::ValuesAbs;
use std::fs;

struct Demo {
    name: String,
    fen: String,
    show_moves: bool,
    highlights: Option<Vec<VisHighlightedHex>>,
    values: Option<Vec<f32>>,
    demo: bool,
    show_internal: bool,
}

fn generate_promotion_highlights(color: Color) -> Vec<VisHighlightedHex> {
    let mut highlights = Vec::new();
    let red_squares = vec![
        QRSPos { q: -7, r: 3, s: 3 },
        QRSPos { q: 6, r: -3, s: -4 },
        QRSPos { q: 7, r: -4, s: -4 },
    ];

    for i in 0..96 {
        let pos = MemoryPos(i as u8);

        let qrs = pos.to_qrs_local(color);

        let color = if red_squares.contains(&qrs) {
            "#d94a0d"
        } else {
            "#3b2c9e"
        };

        if qrs.is_promotion() {
            let qrs = pos.to_qrs_global();

            highlights.push(VisHighlightedHex {
                q: qrs.q,
                r: qrs.r,
                color: color.parse().unwrap(),
                opacity: 0.675,
                text: None,
            });
        }
    }
    highlights
}

fn main() {
    let demos = vec![
        Demo {
            name: "starting_position".to_string(),
            fen: TriHexChess::default().to_fen().to_string(),
            show_moves: false,
            values: None,
            highlights: None,
            demo: false,
            show_internal: false,
        },
        Demo {
            name: "white_pawns".to_string(),
            fen: "8/pppppppp/8/8/X/X X/X/X X/X/X W qkqkqk --- 0 1".to_string(),
            show_moves: true,
            values: None,
            highlights: Some(generate_promotion_highlights(Color::White)),
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "gray_pawn".to_string(),
            fen: "X/X/X X/8/pppppppp/8/8/X X/X/X G qkqkqk --- 0 1".to_string(),
            show_moves: true,
            values: None,
            highlights: Some(generate_promotion_highlights(Color::Gray)),
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "queen".to_string(),
            fen: "X/X/8/8/8/4q3 X/X/X X/X/X W qkqkqk --- 1 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "rook".to_string(),
            fen: "X/X/8/8/8/4r3 X/X/X X/X/X W qkqkqk --- 2 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "bishop".to_string(),
            fen: "5b2/8/8/8/X/8/8/8/43 X/X/X X/X/X W qkqkqk --- 2 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "king".to_string(),
            fen: "r3k2r/8/8/8/X/X X/X/X X/X/X W qkqkqk --- 0 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "king_center".to_string(),
            fen: "X/X/8/8/8/4k3 X/X/X X/X/X W qkqkqk --- 2 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "knight".to_string(),
            fen: "X/X/8/8/8/4n3 X/X/X X/X/X W qkqkqk --- 2 1".to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "en_passant".to_string(),
            fen: "8/p4pp1/1pp4p/3p4/8/8/8/6p1/X X/8/ppppppp1/8/7p/X X/X/8/pppp1ppp/8/4p3 W qkqkqk -85 0 2"
                .to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: false,
        },
        Demo {
            name: "internal_representation".to_string(),
            fen: TriHexChess::default().to_fen().to_string(),
            show_moves: true,
            values: None,
            highlights: None,
            demo: true,
            show_internal: true,
        }
    ];

    let dir = std::env::var("CHESS_VIS_DEMO_DIR").unwrap_or("pdfs/chess/demos".to_string());
    let dir = dir.as_str();

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create directory {}: {}", dir, e);
        return;
    }

    for demo in demos {
        let mut board = TriHexChess::new_with_fen(demo.fen.as_ref(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        if demo.show_moves {
            board.fill_move_store(&mut move_store);
        }

        if demo.show_internal {
            // Delete the buffer to show the internal representation
            board.state.buffer = MemoryBuffer::default();
            move_store.clear();
        }

        let arrows = if demo.show_moves && !demo.demo {
            chess_moves_to_arrows(&move_store, &board, false)
        } else {
            vec![]
        };

        let document = render_board(
            &board,
            &arrows,
            demo.highlights,
            match demo.values {
                Some(v) => Some(ValuesAbs::from_slice(&v.as_slice(), 0.0)),
                None => None,
            },
            if demo.demo { Some(move_store) } else { None },
            demo.show_internal,
        );

        let name = demo.name;

        let svg_filename = format!("{dir}/{name}.svg");

        svg::save(svg_filename.as_str(), &document).unwrap();

        if let Err(e) =
            convert_svg_to_pdf(svg_filename.as_str(), format!("{dir}/{name}.pdf").as_str())
        {
            eprintln!("Failed to convert SVG to PDF: {}", e);
        }
    }
}
