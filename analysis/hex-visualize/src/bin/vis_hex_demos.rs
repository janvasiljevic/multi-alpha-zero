use analysis_util::convert_svg_to_pdf;
use game_hex::coords::AxialCoord;
use game_hex::game_hex::HexPlayer::{P1, P2, P3};
use game_hex::game_hex::{CellState, HexGame};
use hex_visualize::vis_hex::render_hex_board;
use hex_visualize::vis_hex_board::VisHexBoard;
use rand::SeedableRng;
use std::fs;

fn random_connected_example() -> HexGame {
    let size = 5;

    let mut board = HexGame::new(size).unwrap();

    // seed the random number generator for reproducibility
    let mut rng = rand::rngs::StdRng::seed_from_u64(12951234);

    for _ in 0..61 {
        board.play_random_move(&mut rng);
    }

    board.set_state(AxialCoord::new(-2, 1), CellState::Occupied(P1));
    board.set_state(AxialCoord::new(-3, 3), CellState::Occupied(P3));
    board.set_state(AxialCoord::new(-1, 3), CellState::Occupied(P2));
    board.set_state(AxialCoord::new(1, 3), CellState::Occupied(P2));
    board.set_state(AxialCoord::new(2, -4), CellState::Occupied(P1));
    board.set_state(AxialCoord::new(2, -3), CellState::Occupied(P1));

    board.set_state(AxialCoord::new(-2, -1), CellState::Empty);
    board.set_state(AxialCoord::new(-2, 0), CellState::Empty);
    board.set_state(AxialCoord::new(-1, 0), CellState::Occupied(P3));
    board.set_state(AxialCoord::new(-1, 1), CellState::Occupied(P3));

    board
}

fn random_example_for_5_moves() -> HexGame {
    let size = 5;

    let mut board = HexGame::new(size).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(123456);

    for _ in 0..5 {
        board.play_random_move(&mut rng);
    }

    board
}

fn random_example_for_finish_without_connection() -> HexGame {
    let size = 5;

    let mut board = HexGame::new(size).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(9654);

    for _ in 0..61 {
        board.play_random_move(&mut rng);
    }

    board
}

struct HexDemo {
    name: String,
    board: HexGame,
}

fn main() {
    let dir = std::env::var("HEX_VIS_DIR").unwrap_or("pdfs/hex/demos".to_string());
    let dir = dir.as_str();

    println!("Saving hex demo files to directory: {}", dir);

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create directory {}: {}", dir, e);
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

    let demos = vec![
        HexDemo {
            name: "empty_board".to_string(),
            board: HexGame::new(5).unwrap(),
        },
        HexDemo {
            name: "random_5_moves".to_string(),
            board: random_example_for_5_moves(),
        },
        HexDemo {
            name: "random_connected".to_string(),
            board: random_connected_example(),
        },
        HexDemo {
            name: "random_finish_no_connection".to_string(),
            board: random_example_for_finish_without_connection(),
        },
    ];

    let vis_board = VisHexBoard::new(4);

    for demo in demos {
        let document = render_hex_board(&demo.board, &vis_board, None, None, false);

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
