use game_tri_chess::chess_game::TriHexChess;
use maz_core::mapping::InputMapper;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use ndarray::Array2;
use svg::Document;
use svg::node::element::{Rectangle, Text};

// --- CONSTANTS COPIED FROM YOUR ChessDomainMapper ---
// These are necessary for the visualization function to understand the tensor's structure.
const NUM_OF_FIELDS: usize = 96;
const TOTAL_CHANNELS: usize = 53;
const ME_PAWN_CH: usize = 0;
const NEXT_PAWN_CH: usize = 1;
const PREV_PAWN_CH: usize = 2;
const SYMMETRIC_PIECE_START_CH: usize = 3;
const KING_CH: usize = 7;
const ME_OWNER_CH: usize = 8;
const NEXT_OWNER_CH: usize = 9;
const PREV_OWNER_CH: usize = 10;
const ME_EP_CH: usize = 11;
const NEXT_EP_CH: usize = 12;
const PREV_EP_CH: usize = 13;
const ME_CASTLE_Q_CH: usize = 14;
const ME_CASTLE_K_CH: usize = 15;
const NEXT_CASTLE_Q_CH: usize = 16;
const NEXT_CASTLE_K_CH: usize = 17;
const PREV_CASTLE_Q_CH: usize = 18;
const PREV_CASTLE_K_CH: usize = 19;
const ME_PLAYER_PRESENT_CH: usize = 20;
const NEXT_PLAYER_PRESENT_CH: usize = 21;
const PREV_PLAYER_PRESENT_CH: usize = 22;
const IS_GRACE_PERIOD_CH: usize = 23;
const ONE_REP: usize = 24;
const TWO_REP: usize = 25;
const THIRD_MOVE_COUNTER: usize = 26;
const ATTACKS_START_CH: usize = 32;
const IS_CHECK_START_CH: usize = 50;
const ME_IS_IN_CHECK_CH: usize = 50;
const NEXT_IS_IN_CHECK_CH: usize = 51;
const PREV_IS_IN_CHECK_CH: usize = 52;

struct Zone<'a> {
    /// The starting channel index (inclusive).
    start_channel: usize,
    /// The ending channel index (exclusive).
    end_channel: usize,
    /// The highlight color (e.g., "#RRGGBB").
    color: &'a str,
    /// The opacity of the highlight (0.0 to 1.0).
    opacity: f64,
}

const ZONES: &[Zone] = &[
    // Pieces (Pawns, N, B, R, Q, K)
    Zone {
        start_channel: ME_PAWN_CH,
        end_channel: KING_CH + 1,
        color: "#2a9d8f", // Teal
        opacity: 0.35,
    },
    // Ownership (Which player owns the piece on a square)
    Zone {
        start_channel: ME_OWNER_CH,
        end_channel: PREV_OWNER_CH + 1,
        color: "#e9c46a", // Yellow
        opacity: 0.35,
    },
    // En Passant availability
    Zone {
        start_channel: ME_EP_CH,
        end_channel: PREV_EP_CH + 1,
        color: "#f4a261", // Orange
        opacity: 0.4,
    },
    // Castling rights for all players
    Zone {
        start_channel: ME_CASTLE_Q_CH,
        end_channel: PREV_CASTLE_K_CH + 1,
        color: "#e76f51", // Burnt Sienna
        opacity: 0.35,
    },
    // Player Presence (Indicates if a player is active in the game)
    Zone {
        start_channel: ME_PLAYER_PRESENT_CH,
        end_channel: PREV_PLAYER_PRESENT_CH + 1,
        color: "#a933be", // Magenta
        opacity: 0.35,
    },
    Zone {
        start_channel: IS_GRACE_PERIOD_CH,
        end_channel: IS_GRACE_PERIOD_CH + 1,
        color: "#8d99ae", // Gray
        opacity: 0.2,
    },
    // Repetition counters (1 or 2 repetitions of the position)
    Zone {
        start_channel: ONE_REP,
        end_channel: TWO_REP + 1,
        color: "#00b4d8", // Cyan
        opacity: 0.35,
    },
    // 50-move rule counter (represented as 6 bits)
    Zone {
        start_channel: THIRD_MOVE_COUNTER,
        end_channel: ATTACKS_START_CH,
        color: "#52b788", // Green
        opacity: 0.35,
    },
    // Attack Maps (Squares attacked by each player's pieces)
    Zone {
        start_channel: ATTACKS_START_CH,
        end_channel: IS_CHECK_START_CH,
        color: "#d00000", // Red
        opacity: 0.4,
    },
    // In Check status for each player
    Zone {
        start_channel: IS_CHECK_START_CH,
        end_channel: TOTAL_CHANNELS,
        color: "#457b9d", // Blue
        opacity: 0.3,
    },
];

/// A struct to hold styling information for an SVG cell.
struct SvgCellStyle {
    fill: String,
    symbol: String,
}

/// Determines the color and symbol for a specific cell in the tensor grid.
fn get_style_for_chess_channel(
    chan_idx: usize,
    pos_idx: usize,
    input: &Array2<bool>,
    text_mode: bool,
) -> SvgCellStyle {
    let symbol = if text_mode {
        match chan_idx {
            ME_PAWN_CH | NEXT_PAWN_CH | PREV_PAWN_CH => "P",
            c if (SYMMETRIC_PIECE_START_CH..=KING_CH).contains(&c) => {
                ["N", "B", "R", "Q", "K"][c - SYMMETRIC_PIECE_START_CH]
            }
            ME_EP_CH | NEXT_EP_CH | PREV_EP_CH => "e",
            c if (ATTACKS_START_CH..IS_CHECK_START_CH).contains(&c) => "x",
            c if (IS_CHECK_START_CH..TOTAL_CHANNELS).contains(&c) => "!",
            _ => "✓", // Global flags like castling, presence
        }
    } else {
        "" // In block mode, we don't need a symbol
    }
    .to_string();

    let fill = match chan_idx {
        ME_PAWN_CH
        | ME_OWNER_CH
        | ME_EP_CH
        | ME_CASTLE_Q_CH..=ME_CASTLE_K_CH
        | ME_PLAYER_PRESENT_CH
        | ME_IS_IN_CHECK_CH => "#e63946", // Red
        NEXT_PAWN_CH
        | NEXT_OWNER_CH
        | NEXT_EP_CH
        | NEXT_CASTLE_Q_CH..=NEXT_CASTLE_K_CH
        | NEXT_PLAYER_PRESENT_CH
        | NEXT_IS_IN_CHECK_CH => "#457b9d", // Blue
        PREV_PAWN_CH
        | PREV_OWNER_CH
        | PREV_EP_CH
        | PREV_CASTLE_Q_CH..=PREV_CASTLE_K_CH
        | PREV_PLAYER_PRESENT_CH
        | PREV_IS_IN_CHECK_CH => "#52b788", // Green

        // Symmetric pieces - color depends on the owner channel at the same position
        c if (SYMMETRIC_PIECE_START_CH..=KING_CH).contains(&c) => {
            if input[[pos_idx, ME_OWNER_CH]] {
                "#e63946"
            }
            // Red
            else if input[[pos_idx, NEXT_OWNER_CH]] {
                "#457b9d"
            }
            // Blue
            else if input[[pos_idx, PREV_OWNER_CH]] {
                "#2a9d8f"
            }
            // Green
            else {
                "#8d99ae"
            } // Gray (should not happen)
        }

        // Attack maps
        c if (ATTACKS_START_CH..ATTACKS_START_CH + 6).contains(&c) => "#e63946",
        c if (ATTACKS_START_CH + 6..ATTACKS_START_CH + 12).contains(&c) => "#457b9d",
        c if (ATTACKS_START_CH + 12..ATTACKS_START_CH + 18).contains(&c) => "#52b788",

        // Other global features
        IS_GRACE_PERIOD_CH => "#00b4d8", // Cyan
        ONE_REP | TWO_REP => "#ffc300",  // Yellow
        c if (THIRD_MOVE_COUNTER..ATTACKS_START_CH).contains(&c) => "#a933be", // Magenta

        _ => "#adb5bd", // Default
    }
    .to_string();

    SvgCellStyle { fill, symbol }
}

/// Generates an SVG document visualizing the chess tensor.
pub fn visualize_chess_tensor(input: &Array2<bool>, text_mode: bool) -> Document {
    const CELL_SIZE: f64 = 22.0;
    const PADDING: f64 = 20.0;
    const HEADER_V_SIZE: f64 = 4.0; // Space for channel labels
    const HEADER_H_SIZE: f64 = 4.0; // Space for position indices
    const FONT_SIZE: f64 = 12.0;

    let (num_positions, num_channels) = (NUM_OF_FIELDS, TOTAL_CHANNELS);

    let total_width = HEADER_V_SIZE + (num_positions as f64 * CELL_SIZE) + PADDING;
    let total_height = HEADER_H_SIZE + (num_channels as f64 * CELL_SIZE) + PADDING;

    let mut document = Document::new()
        .set("viewBox", (0, 0, total_width, total_height))
        .set("font-family", "monospace");

    // draw one big square with border around the grid
    let border_rect = Rectangle::new()
        .set("x", HEADER_V_SIZE)
        .set("y", HEADER_H_SIZE)
        .set("width", num_positions as f64 * CELL_SIZE)
        .set("height", num_channels as f64 * CELL_SIZE)
        .set("fill", "none")
        .set("stroke", "#000000")
        .set("stroke-width", 4.0);

    document = document.add(border_rect);

    // --- Draw the Grid Cells ---
    for ((pos_idx, chan_idx), &present) in input.indexed_iter() {
        let x = HEADER_V_SIZE + (pos_idx as f64 * CELL_SIZE);
        let y = HEADER_H_SIZE + (chan_idx as f64 * CELL_SIZE);

        let background_rect = Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", CELL_SIZE)
            .set("height", CELL_SIZE)
            .set("stroke", "#ced4da")
            .set("stroke-width", 1.0)
            .set("fill", "#f8f9fa");

        document = document.add(background_rect);

        if present {
            let style = get_style_for_chess_channel(chan_idx, pos_idx, input, text_mode);

            if text_mode {
                let symbol_text = Text::new(style.symbol)
                    .set("x", x + CELL_SIZE / 2.0)
                    .set("y", y + CELL_SIZE / 2.0 + FONT_SIZE * 0.35)
                    .set("text-anchor", "middle")
                    .set("font-size", FONT_SIZE + 2.0)
                    .set("font-weight", "bold")
                    .set("fill", style.fill);
                document = document.add(symbol_text);
            } else {
                // Block mode
                let block_rect = Rectangle::new()
                    .set("x", x)
                    .set("y", y)
                    .set("width", CELL_SIZE)
                    .set("height", CELL_SIZE)
                    .set("fill", style.fill);
                document = document.add(block_rect);
            }
        }
    }

    for zone in ZONES.iter() {
        if zone.start_channel >= num_channels || zone.start_channel >= zone.end_channel {
            continue;
        }
        let end_channel = zone.end_channel.min(num_channels);

        let x = HEADER_V_SIZE;
        let y = HEADER_H_SIZE + (zone.start_channel as f64 * CELL_SIZE);
        let width = num_positions as f64 * CELL_SIZE;
        let height = ((end_channel - zone.start_channel) as f64 * CELL_SIZE);

        let zone_rect = Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", width)
            .set("height", height)
            .set("fill", zone.color)
            .set("fill-opacity", zone.opacity);

        document = document.add(zone_rect);
    }

    document
}

fn main() {
    let board = TriHexChess::new_with_fen("r1bqkbnr/5p2/2p1p1p1/pp1p3p/X/8/8/3n4/8 X/rnkq1bnr/2p2pp1/3p4/1p2p2p/X X/X/rnbq1knr/2p1p1p1/8/pp3p1p G qk---- --- 0 5".as_ref(), false).unwrap();
    let mapper = ChessDomainMapper;
    let mut input = Array2::from_elem(mapper.input_board_shape(), false);
    mapper.encode_input(&mut input.view_mut(), &board);

    let document_block = visualize_chess_tensor(&input, false);
    let path_block = "chess_tensor_block_mode.svg";
    svg::save(path_block, &document_block).expect("Failed to save block mode SVG");

    println!("Successfully generated SVG with highlights: {}", path_block);
}
