use crate::vis_hex_board::VisHexBoard;
pub use analysis_util::vis_arrow::{ArrowProps, Point, VisArrow, create_arrow_component};
use game_hex::game_hex::{CellState, HexGame, HexGameOutcome};
use maz_core::values_const::ValuesAbs;
use std::collections::HashMap;
use svg::{Document, node};

use analysis_util::vis_turn::{VisTurnOptions, vis_turn};
use maz_core::mapping::Board;
use svg::Node;
use svg::node::element::{Definitions, Group, Polygon, Polyline, Text};

// Colorblind-friendly colors for the 3 players
pub const PLAYER_COLORS: [&str; 3] = ["#FFFFE4", "#E8351B", "#000000"]; // White, Red, Black
const BORDER_WIDTH: f64 = 20.0; // The thickness of the border
const SIZE: usize = 50;

fn render_player_borders(radius: i32, size: f64, border_group: &mut Group) {
    struct Point {
        x: f64,
        y: f64,
    }
    let stroke_width = 50;

    println!("radius: {}", radius);

    let size = if radius == 1 {
        125.0
    } else {
        2.0 * size * radius as f64 - BORDER_WIDTH
    };

    let hex_points: Vec<Point> = (0..6)
        .map(|i| {
            let angle_deg = 60.0 * i as f64;
            let angle_rad = std::f64::consts::PI / 180.0 * angle_deg;
            Point {
                x: size * angle_rad.cos(),
                y: size * angle_rad.sin(),
            }
        })
        .collect();

    let permuted_colors = [PLAYER_COLORS[1], PLAYER_COLORS[0], PLAYER_COLORS[2]];

    for i in 0..3 {
        let p1 = &hex_points[i];
        let p2 = &hex_points[(i + 1) % 6];
        let p3 = &hex_points[(i + 3) % 6];
        let p4 = &hex_points[(i + 4) % 6];

        // outline
        border_group.append(
            Polyline::new()
                .set("points", format!("{},{} {},{}", p1.x, p1.y, p2.x, p2.y))
                .set("fill", "none")
                .set("stroke", "black")
                .set("stroke-linecap", "round")
                .set("stroke-width", stroke_width + 2),
        );

        // outline
        border_group.append(
            Polyline::new()
                .set("points", format!("{},{} {},{}", p3.x, p3.y, p4.x, p4.y))
                .set("fill", "none")
                .set("stroke", "black")
                .set("stroke-linecap", "round")
                .set("stroke-width", stroke_width + 2),
        );

        // Side i
        border_group.append(
            Polyline::new()
                .set("points", format!("{},{} {},{}", p1.x, p1.y, p2.x, p2.y))
                .set("fill", "none")
                .set("stroke", permuted_colors[i])
                .set("opacity", 0.9)
                .set("stroke-linecap", "round")
                .set("stroke-width", stroke_width),
        );

        // Opposite side
        border_group.append(
            Polyline::new()
                .set("points", format!("{},{} {},{}", p3.x, p3.y, p4.x, p4.y))
                .set("fill", "none")
                .set("stroke", permuted_colors[i])
                .set("opacity", 0.9)
                .set("stroke-linecap", "round")
                .set("stroke-width", stroke_width),
        );
    }
}

pub struct VisHexHighlight {
    pub q: i32,
    pub r: i32,
    pub color: String,
    pub opacity: f64,
}

pub fn render_hex_board(
    hex_board: &HexGame,
    vis_hex_board: &VisHexBoard,
    highlights: Option<Vec<VisHexHighlight>>,
    evaluation: Option<ValuesAbs<3>>,
    show_internal_ids: bool,
) -> Document {
    let offset = if evaluation.is_some() { 80.0 } else { 0.0 };

    let mut document = Document::new().set(
        "viewBox",
        (
            vis_hex_board.view_box.0,
            vis_hex_board.view_box.1 - offset,
            vis_hex_board.view_box.2,
            vis_hex_board.view_box.3 + offset,
        ),
    );

    // Create a single <defs> block to hold definitions.

    // Hex coordinates -> Hex reference map for quick lookup (for pixel values)
    let hex_map: HashMap<(i32, i32), &_> = vis_hex_board
        .hexagons
        .iter()
        .map(|h| ((h.q, h.r), h))
        .collect();

    let defs = Definitions::new();
    let mut border_group = Group::new().set("id", "player-borders");

    render_player_borders(hex_board.radius, SIZE as f64, &mut border_group);

    document = document.add(defs);
    document = document.add(border_group);

    let hex_points = "50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5";

    // Background hexes
    for hex in &vis_hex_board.hexagons {
        let mut hex_group = Group::new()
            .set("transform", format!("translate({}, {})", hex.x, hex.y))
            .add(
                Polygon::new()
                    .set("points", hex_points)
                    // rotate for 90
                    .set(
                        "transform",
                        format!("rotate({})", 30.0), // Rotate by 30 degrees
                    )
                    .set("fill", "#FFFFFF")
                    .set("stroke", "black")
                    .set("stroke-width", 1),
            );

        if show_internal_ids {
            hex_group = hex_group.add(
                Text::new(format!("({}, {})", hex.q, hex.r))
                    .set("text-anchor", "middle")
                    .set("dominant-baseline", "central")
                    .set("font-size", "10")
                    .set("fill", "black")
                    .set("font-family", "'Roboto Mono'") // The family name is the same for all weights
                    .set("font-weight", "400")
                    .set("transform", "translate(0, 0)"),
            );
        }
        document = document.add(hex_group);
    }

    if let Some(outcome) = hex_board.outcome {
        match outcome {
            HexGameOutcome::Win { winner } => {
                let (_, maybe_path) = hex_board.check_connectivity(winner, false);

                if let Some(path) = maybe_path {
                    for axial_coord in path {
                        if let Some(hex) =
                            hex_map.get(&(axial_coord.q as i32, axial_coord.r as i32))
                        {
                            let highlight_group = Group::new()
                                .set("transform", format!("translate({}, {})", hex.x, hex.y))
                                .add(
                                    Polygon::new()
                                        .set("points", hex_points)
                                        .set("fill", "#FFA500")
                                        .set("transform", format!("rotate({})", 30.0))
                                        .set("fill-opacity", 0.6)
                                        .set("stroke", "black")
                                        .set("stroke-width", 1),
                                );
                            document = document.add(highlight_group);
                        }
                    }
                }
            }
        }
    }

    // Render highlights if any
    if let Some(highlights) = highlights {
        for highlight in highlights {
            if let Some(hex) = hex_map.get(&(highlight.q, highlight.r)) {
                let highlight_group = Group::new()
                    .set("transform", format!("translate({}, {})", hex.x, hex.y))
                    .add(
                        Polygon::new()
                            .set("points", hex_points)
                            .set("fill", highlight.color)
                            .set("transform", format!("rotate({})", 30.0))
                            .set("fill-opacity", highlight.opacity)
                            .set("stroke", "black")
                            .set("stroke-width", 1),
                    );
                document = document.add(highlight_group);
            }
        }
    }

    for (coords, cell_state) in hex_board.iter() {
        match cell_state {
            CellState::Occupied(p) => {
                let p_index = p as usize;

                let color = PLAYER_COLORS[p_index];

                let is_eliminated = hex_board.eliminated_players.contains(&p);

                let radius = SIZE as f64 * 0.6;

                // draw a circle with the player color
                let circle = node::element::Circle::new()
                    .set("cx", hex_map[&(coords.q as i32, coords.r as i32)].x)
                    .set("cy", hex_map[&(coords.q as i32, coords.r as i32)].y)
                    .set("r", radius)
                    .set("fill", color)
                    .set("fill-opacity", 0.8)
                    .set("stroke", "black")
                    .set("stroke-width", 1);
                document = document.add(circle);

                if (is_eliminated) {
                    // add a cross over the circle
                    let radius = radius / 2.0;

                    let cross_group = Group::new()
                        .add(
                            node::element::Line::new()
                                .set(
                                    "x1",
                                    hex_map[&(coords.q as i32, coords.r as i32)].x - radius,
                                )
                                .set(
                                    "y1",
                                    hex_map[&(coords.q as i32, coords.r as i32)].y - radius,
                                )
                                .set(
                                    "x2",
                                    hex_map[&(coords.q as i32, coords.r as i32)].x + radius,
                                )
                                .set(
                                    "y2",
                                    hex_map[&(coords.q as i32, coords.r as i32)].y + radius,
                                )
                                .set("stroke", "black")
                                .set("stroke-linecap", "round")
                                .set("stroke-width", 2),
                        )
                        .add(
                            node::element::Line::new()
                                .set(
                                    "x1",
                                    hex_map[&(coords.q as i32, coords.r as i32)].x + radius,
                                )
                                .set(
                                    "y1",
                                    hex_map[&(coords.q as i32, coords.r as i32)].y - radius,
                                )
                                .set(
                                    "x2",
                                    hex_map[&(coords.q as i32, coords.r as i32)].x - radius,
                                )
                                .set(
                                    "y2",
                                    hex_map[&(coords.q as i32, coords.r as i32)].y + radius,
                                )
                                .set("stroke", "black")
                                .set("stroke-linecap", "round")
                                .set("stroke-width", 2),
                        );

                    document = document.add(cross_group);
                }
            }
            _ => continue,
        }
    }

    // Turn indicator
    let turn_gorup = vis_turn(
        hex_board.current_turn as usize,
        PLAYER_COLORS.to_vec(),
        if hex_board.radius == 1 {
            175.0
        } else {
            2.0 * SIZE as f64 * hex_board.radius as f64
        },
        VisTurnOptions::default(),
    );

    document = document.add(turn_gorup);

    if let Some(eval) = evaluation {
        let dot_radius = 8.0;
        let dot_stroke_width = 2.0;
        let spacing = 120.0; // Wider spacing to accommodate the dot

        for (i, &val) in eval.value_abs.iter().enumerate() {
            let mut eval_item_group = Group::new().set(
                "transform",
                format!(
                    "translate({}, {})",
                    i as f64 * spacing - spacing,
                    vis_hex_board.view_box.1 - 20.0,
                ),
            );

            // Player color dot
            let player_dot = node::element::Circle::new()
                .set("cx", -55) // Position dot to the left of the text
                .set("cy", 0) // Vertically centered with the text
                .set("r", dot_radius)
                .set("fill", PLAYER_COLORS[i])
                .set("stroke", "black")
                .set("stroke-width", dot_stroke_width);

            // Evaluation text
            let score_text = Text::new(format!("{}{:.2}", if val >= 0.0 { "+" } else { "" }, val))
                .set("text-anchor", "middle") // Centered in its own space
                .set("dominant-baseline", "central")
                .set("font-size", "32px")
                .set("fill", if val >= 0.0 { "green" } else { "red" })
                .set("font-family", "'Inter 18pt'");

            eval_item_group = eval_item_group.add(player_dot).add(score_text);

            document = document.add(eval_item_group);
        }
    }

    document
}
