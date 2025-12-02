use analysis_util::convert_svg_to_pdf;
use chess_visualize::vis_chess_demo_move::chess_moves_to_arrows;
use chess_visualize::vis_chess_render::render_board;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use game_tri_chess::pos::MemoryPos;
use maz_core::mapping::Board;
use std::fs;

struct OpeningDemo {
    name: String,
    fen: String,
    move_from_i: u8,
    move_to_i: u8,
}

fn main() {
    // rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1
    // arrow from 2 to 78
    // rn1qkbnr/pppppppp/8/8/X/8/6b1/8/8 X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppp1p/8/8 G qkqkqk --- 0 1
    // arrow from 37 to 74
    // rn1qkbnr/pppppppp/8/8/X/8/6b1/8/8 X/rnbqk1nr/pppppppp/8/8/8/2b5/8/8 X/X/X W qkqk-- --- 0 2
    // arrow from 78 to 74

    let demos = vec![
        OpeningDemo {
            name: "pos_1".to_string(),
            fen: "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1".to_string(),
            move_from_i: 2,
            move_to_i: 78,
        },
        OpeningDemo {
            name: "pos_2".to_string(),
            fen: "rn1qkbnr/pppppppp/8/8/X/8/6b1/8/8 X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppp1p/8/8 G qkqkqk --- 0 1".to_string(),
            move_from_i: 37,
            move_to_i: 74,
        },
        OpeningDemo {
            name: "pos_3".to_string(),
            fen: "rn1qkbnr/pppppppp/8/8/X/8/6b1/8/8 X/rnbqk1nr/pppppppp/8/8/8/2b5/8/8 X/X/X W qkqk-- --- 0 2".to_string(),
            move_from_i: 78,
            move_to_i: 74,
        },
    ];

    let dir = std::env::var("CHESS_VIS_OPENING_DIR").unwrap_or("pdfs/chess/opening".to_string());
    let dir = dir.as_str();

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create directory {}: {}", dir, e);
        return;
    }

    for demo in demos {
        let mut board = TriHexChess::new_with_fen(demo.fen.as_ref(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        board.fill_move_store(&mut move_store);

        let arrow = chess_moves_to_arrows(&move_store, &board, false);

        // filter arrows to only keep the one matching move_from_i and move_to_i
        let arrows: Vec<chess_visualize::vis_chess_render::VisArrow> = arrow
            .into_iter()
            .filter(|a| {
                a.from_i == MemoryPos(demo.move_from_i) && a.to_i == MemoryPos(demo.move_to_i)
            })
            .collect();

        let document = render_board(&board, &arrows, None, None, None, false);

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
