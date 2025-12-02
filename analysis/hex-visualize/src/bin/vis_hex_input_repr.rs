use analysis_util::convert_svg_to_pdf;
use game_hex::game_hex::{HexGame, HexPlayer};
use hex_visualize::vis_hex::{render_hex_board, PLAYER_COLORS};
use hex_visualize::vis_hex_board::VisHexBoard;
use maz_core::mapping::hex_absolute_mapper::HexAbsoluteMapper;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_core::mapping::InputMapper;
use maz_core::mapping::{Board, BoardPlayer};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use svg::node::element::{Circle, Group, Rectangle, Text};
use svg::node::Text as TextNode;
use svg::Document;

fn main() {
    let mut game = HexGame::new(2).unwrap();
    let mapper = HexAbsoluteMapper::new(&game);
    let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);

    let mut rng = StdRng::from_seed([22; 32]);
    for _ in 0..3 {
        game.play_random_move(&mut rng);
    }

    println!("{}", game.fancy_debug_visualize_board_indented(None));

    for player in [HexPlayer::P1, HexPlayer::P2, HexPlayer::P3] {
        game.current_turn = player;
        mapper.encode_input(&mut input_view.view_mut(), &game);
        println!("Generating SVG visualization...");
        let document = visualize_mapper_input(&input_view);

        let output_base_path = format!("mapper_visualization_absolute_{:?}", player);

        let svg_path = format!("output/{}.svg", output_base_path);
        svg::save(&svg_path, &document).expect("Failed to save SVG file");
        convert_svg_to_pdf(&svg_path
                           , &format!("output/{}.pdf", output_base_path))
            .expect("Failed to convert SVG to PDF");
        println!("  -> Saved to output/{}.pdf", output_base_path);
    }

    for player in [HexPlayer::P1, HexPlayer::P2, HexPlayer::P3] {
        println!("Generating visualization for {:?}...", player);

        let mapper = HexCanonicalMapper::new(&game);

        // Create a copy of the game and set the current turn to get the correct perspective
        let mut game_for_perspective = game.clone();
        game_for_perspective.current_turn = player;

        // Encode the board from this player's perspective
        let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);
        mapper.encode_input(&mut input_view.view_mut(), &game_for_perspective);

        // Generate the SVG using the adapted visualization function
        let document = visualize_canonical_mapper_input(&input_view, player);

        // Save the SVG and convert to PDF
        let output_base_path = format!("mapper_visualization_canonical_{:?}", player);
        let svg_path = format!("output/{}.svg", output_base_path);
        svg::save(&svg_path, &document).expect("Failed to save SVG file");

        convert_svg_to_pdf(&svg_path, &format!("output/{}.pdf", output_base_path))
            .expect("Failed to convert SVG to PDF");

        println!("  -> Saved to output/{}.pdf", output_base_path);
    }

    // save the board

    let vis_board = VisHexBoard::new(1);

    for player in [HexPlayer::P1, HexPlayer::P2, HexPlayer::P3] {
        game.current_turn = player;
        println!("Rendering board for {:?}...", player);
        let doc = render_hex_board(&game, &vis_board, None, None, false);

        let svg_path = format!("output/hex_board_{:?}.svg", player);
        svg::save(&svg_path, &doc).expect("Failed to save SVG file");
        convert_svg_to_pdf(&svg_path, &format!("output/hex_board_{:?}.pdf", player))
            .expect("Failed to convert SVG to PDF");
    }


}


struct CellStyle {
    fill: &'static str,
    symbol: Symbol,
}

enum Symbol {
    Player,
    None,
}

fn get_style_for_channel(channel_index: usize) -> CellStyle {
    match channel_index {
        0 => CellStyle {
            fill: PLAYER_COLORS[0],
            symbol: Symbol::Player,
        }, // P1 Pieces (Red)
        1 => CellStyle {
            fill: PLAYER_COLORS[1],
            symbol: Symbol::Player,
        }, // P2 Pieces (Blue)
        2 => CellStyle {
            fill: PLAYER_COLORS[2],
            symbol: Symbol::Player,
        }, // P3 Pieces (Green)
        3 => CellStyle {
            fill: "#cccccc",
            symbol: Symbol::None,
        }, // Empty (Light Gray)
        4 => CellStyle {
            fill: PLAYER_COLORS[0],
            symbol: Symbol::None,
        }, // P1 Turn (Red, bold border)
        5 => CellStyle {
            fill: PLAYER_COLORS[1],
            symbol: Symbol::None,
        }, // P2 Turn (Blue, bold border)
        6 => CellStyle {
            fill: PLAYER_COLORS[2],
            symbol: Symbol::None,
        }, // P3 Turn (Green, bold border)
        _ => panic!("Unknown channel index: {}", channel_index),
    }
}

/// Generates an SVG document visualizing the mapper's input tensor.
fn visualize_mapper_input(input: &Array2<bool>) -> Document {
    const SQUARE_SIZE: f64 = 30.0;
    const PADDING: f64 = 20.0;
    const HEADER_V_SIZE: f64 = 150.0;
    const HEADER_H_SIZE: f64 = 40.0;
    const FONT_SIZE: f64 = 12.0;

    let (num_cols, num_rows) = input.dim();

    println!("Input shape: {} cols x {} rows", num_cols, num_rows);

    let total_width = HEADER_V_SIZE + (num_cols as f64 * SQUARE_SIZE) + PADDING;
    let total_height = HEADER_H_SIZE + (num_rows as f64 * SQUARE_SIZE) + PADDING;

    let mut document = Document::new()
        .set("viewBox", (0, 0, total_width, total_height))
        .set("font-family", "sans-serif");

    // --- Draw Row and Column Headers ---
    let row_labels = [
        "P1 Pieces",
        "P2 Pieces",
        "P3 Pieces",
        "Empty",
        "P1 Turn",
        "P2 Turn",
        "P3 Turn",
    ];

    for i in 0..num_rows {
        let y = HEADER_H_SIZE + (i as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0) + (FONT_SIZE / 3.0);
        let label = row_labels.get(i).unwrap_or(&"Unknown");
        let text = Text::new("")
            .set("x", HEADER_V_SIZE - PADDING / 2.0)
            .set("y", y)
            .set("text-anchor", "end")
            .set("font-family", "'Inter 18pt'")
            .set("font-size", FONT_SIZE)
            .add(TextNode::new(*label));
        document = document.add(text);
    }

    for i in 0..num_cols {
        let x = HEADER_V_SIZE + (i as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0);
        let text = Text::new("")
            .set("x", x)
            .set("y", HEADER_H_SIZE - PADDING / 2.0)
            .set("text-anchor", "middle")
            .set("font-family", "'Inter 18pt'")
            .set("font-size", FONT_SIZE)
            .add(TextNode::new(i.to_string()));
        document = document.add(text);
    }

    // --- Draw the grid cells for active inputs ---
    for ((col, row), &present) in input.indexed_iter() {
        let style = get_style_for_channel(row);

        let mut group = svg::node::element::Group::new();

        let mut rect = Rectangle::new()
            .set("x", HEADER_V_SIZE + (col as f64 * SQUARE_SIZE))
            .set("y", HEADER_H_SIZE + (row as f64 * SQUARE_SIZE))
            .set("width", SQUARE_SIZE)
            .set("height", SQUARE_SIZE)
            .set("stroke", "black")
            .set("stroke-width", "1");

        if present && matches!(style.symbol, Symbol::None) {
            rect = rect.set("fill", style.fill);
        } else {
            rect = rect.set("fill", "white");
        }

        group = group.add(rect);

        if present {
            match style.symbol {
                Symbol::Player => {
                    // add an svg cirle in the center of the rect
                    let circle = svg::node::element::Circle::new()
                        .set(
                            "cx",
                            HEADER_V_SIZE + (col as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0),
                        )
                        .set(
                            "cy",
                            HEADER_H_SIZE + (row as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0),
                        )
                        .set("r", SQUARE_SIZE * 0.3)
                        .set("fill", style.fill)
                        .set("stroke", "black")
                        .set("stroke-width", "1");
                    group = group.add(circle);
                }
                Symbol::None => {}
            }
        }

        document = document.add(group);
    }

    document
}

/// Gets the appropriate color for a channel in the canonical mapper's input.
fn get_style_for_canonical_channel(channel_index: usize, current_player: HexPlayer) -> CellStyle {
    let me_idx = current_player as usize;
    let opp1_idx = current_player.next() as usize;
    let opp2_idx = current_player.next().next() as usize;

    match channel_index {
        0 => CellStyle {
            fill: PLAYER_COLORS[me_idx],
            symbol: Symbol::Player,
        }, // Me
        1 => CellStyle {
            fill: PLAYER_COLORS[opp1_idx],
            symbol: Symbol::Player,
        }, // Opponent 1
        2 => CellStyle {
            fill: PLAYER_COLORS[opp2_idx],
            symbol: Symbol::Player,
        }, // Opponent 2
        3 => CellStyle {
            fill: "#cccccc",
            symbol: Symbol::None,
        }, // Empty
        _ => panic!("Unknown channel index: {}", channel_index),
    }
}

/// Generates an SVG document visualizing the canonical mapper's input tensor.
fn visualize_canonical_mapper_input(input: &Array2<bool>, current_player: HexPlayer) -> Document {
    const SQUARE_SIZE: f64 = 30.0;
    const PADDING: f64 = 20.0;
    const HEADER_V_SIZE: f64 = 150.0;
    const HEADER_H_SIZE: f64 = 40.0;
    const FONT_SIZE: f64 = 12.0;

    // The tensor shape is [positions, channels].
    // `ndarray.dim()` returns (rows, cols), so `num_positions` is rows and `num_channels` is cols.
    let (num_positions, num_channels) = input.dim();

    // The visual layout has positions on the X-axis and channels on the Y-axis.
    let total_width = HEADER_V_SIZE + (num_positions as f64 * SQUARE_SIZE) + PADDING;
    let total_height = HEADER_H_SIZE + (num_channels as f64 * SQUARE_SIZE) + PADDING;

    let mut document = Document::new()
        .set("viewBox", (0, 0, total_width, total_height))
        .set("font-family", "sans-serif");

    // --- Draw Row and Column Headers ---
    let me_p = current_player as usize + 1;
    let opp1_p = current_player.next() as usize + 1;
    let opp2_p = current_player.next().next() as usize + 1;

    let row_labels = [
        format!("My (P{}) Pieces", me_p),
        format!("Next (P{}) Pieces", opp1_p),
        format!("2nd Next (P{}) Pieces", opp2_p),
        "Empty".to_string(),
    ];

    // Row labels (channels)
    for i in 0..4 {
        let y = HEADER_H_SIZE + (i as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0) + (FONT_SIZE / 3.0);
        let label = &row_labels[i];
        let text = Text::new("")
            .set("x", HEADER_V_SIZE - PADDING / 2.0)
            .set("y", y)
            .set("text-anchor", "end")
            .set("font-size", FONT_SIZE)
            .set("font-family", "'Inter 18pt'")
            .add(TextNode::new(label.as_str()));
        document = document.add(text);
    }

    // Column labels (canonical position indices)
    for i in 0..num_positions {
        let x = HEADER_V_SIZE + (i as f64 * SQUARE_SIZE) + (SQUARE_SIZE / 2.0);
        let text = Text::new("")
            .set("x", x)
            .set("y", HEADER_H_SIZE - PADDING / 2.0)
            .set("text-anchor", "middle")
            .set("font-family", "'Inter 18pt'")
            .set("font-size", FONT_SIZE)
            .add(TextNode::new(i.to_string()));
        document = document.add(text);
    }

    // --- Draw the grid cells for active inputs ---
    // `indexed_iter()` yields ((row, col), &value).
    // Here, row corresponds to `pos_idx` and col corresponds to `chan_idx`.
    for ((pos_idx, chan_idx), &present) in input.indexed_iter() {
        let style = get_style_for_canonical_channel(chan_idx, current_player);

        let mut group = Group::new();

        let base_x = HEADER_V_SIZE + (pos_idx as f64 * SQUARE_SIZE);
        let base_y = HEADER_H_SIZE + (chan_idx as f64 * SQUARE_SIZE);

        let mut rect = Rectangle::new()
            .set("x", base_x)
            .set("y", base_y)
            .set("width", SQUARE_SIZE)
            .set("height", SQUARE_SIZE)
            .set("stroke", "black")
            .set("stroke-width", "1");

        // Set fill based on symbol type
        if present && matches!(style.symbol, Symbol::None) {
            rect = rect.set("fill", style.fill);
        } else {
            rect = rect.set("fill", "white");
        }

        group = group.add(rect);

        // If the symbol is a player piece, draw a circle on top
        if present {
            if let Symbol::Player = style.symbol {
                let circle = Circle::new()
                    .set("cx", base_x + (SQUARE_SIZE / 2.0))
                    .set("cy", base_y + (SQUARE_SIZE / 2.0))
                    .set("r", SQUARE_SIZE * 0.35) // Made slightly larger to fill the cell better
                    .set("fill", style.fill)
                    .set("stroke", "black")
                    .set("stroke-width", "1");
                group = group.add(circle);
            }
        }

        document = document.add(group);
    }

    document
}
