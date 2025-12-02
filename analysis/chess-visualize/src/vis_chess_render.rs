pub use crate::vis_chess_board::{BOARD_VIS, VisChessTile, VisHighlightedHex, is_border};
use crate::vis_chess_check::create_check;
use crate::vis_chess_demo_move::draw_demo_moves;
pub use crate::vis_chess_util::COLORS;
pub use crate::vis_chess_util::SIZE;
use analysis_util::convert_svg_to_pdf;
pub use analysis_util::vis_arrow::{ArrowProps, Point, VisArrow, create_arrow_component};
use analysis_util::vis_turn::{VisTurnOptions, vis_turn};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use maz_core::values_const::ValuesAbs;
use roxmltree::Document as XmlDocument;
use std::collections::{HashMap, HashSet};
use std::fs;
use svg::node::Text as TextNode;
use svg::node::element::{
    Definitions, Group, Marker, Path, Polygon, RadialGradient, Rectangle, Stop, TSpan, Text, Use,
};
use svg::{Document, node};

fn get_symbol_content(piece_name: &str) -> (String, String) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let path = format!("{}/pieces/{}.svg", manifest_dir, piece_name);
    let svg_text =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read piece SVG: {}", path));

    let doc = XmlDocument::parse(&svg_text).expect("Failed to parse piece SVG XML");
    let root_element = doc.root_element();

    let view_box = root_element.attribute("viewBox").unwrap_or("0 0 70 70");

    let mut inner_content = String::new();
    for node in root_element.children() {
        if node.is_element() {
            let range = node.range();
            inner_content.push_str(&svg_text[range.start..range.end]);
            inner_content.push('\n');
        }
    }

    (view_box.to_string(), inner_content)
}

pub fn render_board(
    chess_state: &TriHexChess,
    arrows: &[VisArrow],
    highlights: Option<Vec<VisHighlightedHex>>,
    evaluation: Option<ValuesAbs<3>>,
    demo_moves: Option<ChessMoveStore>,
    show_internal: bool,
) -> Document {
    let mut document = Document::new().set(
        "viewBox",
        (
            BOARD_VIS.view_box.0,
            BOARD_VIS.view_box.1,
            BOARD_VIS.view_box.2,
            BOARD_VIS.view_box.3,
        ),
    );

    let mut defs = Definitions::new();

    let gradient = RadialGradient::new()
        .set("id", "checkGradient")
        .set("cx", "50%")
        .set("cy", "50%")
        .set("r", "50%")
        .set("fx", "50%")
        .set("fy", "50%")
        .add(
            Stop::new()
                .set("offset", "50%")
                .set("stop-color", "transparent")
                .set("stop-opacity", 1.0),
        )
        .add(
            Stop::new()
                .set("offset", "100%")
                .set("stop-color", "red")
                .set("stop-opacity", 0.8),
        );

    defs = defs.add(gradient);

    let mut symbol_cache: HashMap<String, String> = HashMap::new();
    let mut marker_cache: HashMap<String, String> = HashMap::new();

    for (_, slot) in chess_state.state.buffer.non_empty_iter() {
        let (color, piece) = slot.get().unwrap();

        let piece_name = format!(
            "{}{}",
            color.to_string().to_lowercase(),
            piece.to_char().to_uppercase()
        );
        if !symbol_cache.contains_key(&piece_name) {
            let symbol_id = format!("piece-{}", piece_name);
            let (view_box, content) = get_symbol_content(&piece_name);
            let symbol = node::element::Symbol::new()
                .set("id", symbol_id.clone())
                .set("viewBox", view_box)
                .add(node::Blob::new(content));
            defs = defs.add(symbol);
            symbol_cache.insert(piece_name, symbol_id);
        }
    }

    //  Populate <defs> with all unique arrow markers
    let unique_colors: HashSet<_> = arrows.iter().map(|a| a.color.clone()).collect();
    for color in unique_colors {
        let safe_color_id = color.replace("#", "");
        let marker_id = format!("arrowhead-{}", safe_color_id);
        let arrowhead = Marker::new()
            .set("id", marker_id.clone())
            .set("viewBox", "0 0 10 10")
            .set("refX", 8)
            .set("refY", 5)
            .set("markerWidth", 4)
            .set("markerHeight", 4)
            .set("orient", "auto-start-reverse")
            .set("fill", color.clone())
            .add(Path::new().set("d", "M 0 0 L 10 5 L 0 10 z"));
        defs = defs.add(arrowhead);
        marker_cache.insert(color, marker_id);
    }

    document = document.add(defs);

    let hex_map: HashMap<(i8, i8), &VisChessTile> =
        BOARD_VIS.hexagons.iter().map(|h| ((h.q, h.r), h)).collect();

    let hex_points = "50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5";

    for hex in &BOARD_VIS.hexagons {
        if is_border(hex) {
            // First, draw the hexagon itself inside a group.
            let hex_group = Group::new()
                .set("transform", format!("translate({}, {})", hex.x, hex.y))
                .add(
                    Polygon::new()
                        .set("points", hex_points)
                        .set("fill", "none")
                        .set("stroke", "black")
                        .set("stroke-width", 10),
                );
            document = document.add(hex_group);
        }
    }

    for hex in &BOARD_VIS.hexagons {
        // First, draw the hexagon itself inside a group.
        let mut hex_group =
            Group::new().set("transform", format!("translate({}, {})", hex.x, hex.y));

        if !show_internal {
            hex_group = hex_group.add(
                Polygon::new()
                    .set("points", hex_points)
                    .set("fill", hex.color)
                    .set("stroke", "black")
                    .set("stroke-width", 0),
            );
        } else {
            hex_group = hex_group.add(
                Polygon::new()
                    .set("points", hex_points)
                    .set("fill", "white")
                    .set("stroke", "black")
                    .set("stroke-width", 1)
                    .set("stroke-dasharray", "10,5"),
            );
        }

        if show_internal {
            // The rotation of the board. Set to 0.0 for a static, non-rotated view.
            let rotation = 0.0;
            let debug_transform = format!("rotate({}) scale(0.5)", -rotation);

            // This group holds all the debug elements and applies the rotation and scaling.
            let debug_elements_group = Group::new()
                .set("transform", debug_transform)
                .add(
                    Text::new(hex.notation.as_str().to_uppercase())
                        .set("text-anchor", "middle")
                        .set("font-family", "'Inter 18pt'")
                        .set("dominant-baseline", "central")
                        .set("font-weight", 700)
                        .set("font-size", "22")
                        .set("fill", "orange")
                        .set("y", -50),
                )
                .add(
                    Text::new("")
                        .set("text-anchor", "middle")
                        .set("dominant-baseline", "central")
                        .set("font-weight", 700)
                        .set("font-size", "30")
                        .set("font-family", "'Inter 18pt'")
                        .add(TSpan::new(hex.q.to_string()).set("fill", "green"))
                        .add(TextNode::new(","))
                        .add(TSpan::new(hex.r.to_string()).set("fill", "blue"))
                        .add(TextNode::new(","))
                        .add(TSpan::new(hex.s.to_string()).set("fill", "red")),
                )
                .add(
                    Rectangle::new()
                        .set("x", -35)
                        .set("y", 35)
                        .set("width", 60)
                        .set("height", 30)
                        .set("fill", "black")
                        .set("rx", 10)
                        .set("ry", 10),
                )
                .add(
                    Text::new("")
                        .set("text-anchor", "middle")
                        .set("dominant-baseline", "central")
                        .set("font-weight", 700)
                        .set("font-size", "20")
                        .set("font-family", "'Inter 18pt'")
                        .set("y", 50)
                        .add(TSpan::new(hex.i.to_string()).set("fill", "white"))
                        .add(TextNode::new(",")),
                );

            hex_group = hex_group.add(debug_elements_group);
        }

        if let Some(letter) = &hex.rank {
            let text = Text::new(letter.clone())
                .set("text-anchor", "middle")
                .set("dominant-baseline", "central")
                .set("font-size", "18")
                .set("fill", "black")
                .set("font-weight", "900")
                .set("font-family", "'Inter 18pt'")
                .set(
                    "transform",
                    format!("translate({}, {})", SIZE + 3.0, -SIZE / 2.0 - 10.0),
                );
            hex_group = hex_group.add(text);
        }

        if let Some(number) = &hex.file {
            let text = Text::new(number.clone())
                .set("text-anchor", "middle")
                .set("dominant-baseline", "central")
                .set("font-size", "18")
                .set("fill", "black")
                .set("font-weight", "400")
                .set("font-family", "'Inter 18pt'")
                .set("transform", format!("translate({}, {})", 0, SIZE + 14.0,));
            hex_group = hex_group.add(text);
        }

        document = document.add(hex_group);
    }

    // Render Highlighted Squares (under pieces, over board)
    if let Some(ref highlights_slice) = highlights {
        for highlight in highlights_slice {
            if let Some(hex) = hex_map.get(&(highlight.q, highlight.r)) {
                let highlight_group = Group::new()
                    .set("transform", format!("translate({}, {})", hex.x, hex.y))
                    .add(
                        Polygon::new()
                            .set("points", hex_points)
                            .set("fill", highlight.color.clone())
                            .set("fill-opacity", highlight.opacity)
                            .set("stroke", highlight.color.clone())
                            .set("stroke-width", 1),
                    );
                document = document.add(highlight_group);
            }
        }
    }

    if let Some(move_store) = demo_moves {
        document = document.add(draw_demo_moves(&move_store, &chess_state));
    }

    if !show_internal {
        document = document.add(vis_turn(
            chess_state.get_turn().unwrap().into(),
            COLORS.to_vec(),
            480.0,
            VisTurnOptions {
                angle_offset: -90.0,
                stroke_width: 3.0,
                both_sided: true,
                angle_stride: 60.0,
                x_offsets: vec![20.0, -20.0, 20.0],
            },
        ));
    }

    let (check_highlight_group, check_arrow_group) = create_check(chess_state);

    document = document.add(check_highlight_group);

    // Render Pieces
    for (pos, slot) in chess_state.state.buffer.non_empty_iter() {
        let (color, piece) = slot.get().unwrap();
        let qrs = pos.to_qrs_global();

        if let Some(hex) = hex_map.get(&(qrs.q, qrs.r)) {
            let piece_name = format!(
                "{}{}",
                color.to_string().to_lowercase(),
                piece.to_char().to_uppercase()
            );
            if let Some(symbol_id) = symbol_cache.get(&piece_name) {
                let use_tag = Use::new()
                    .set("href", format!("#{}", symbol_id))
                    .set("width", 70)
                    .set("height", 70);
                let transform = format!("translate({}, {}) translate(-35, -35)", hex.x, hex.y);
                let group = Group::new().set("transform", transform).add(use_tag);
                document = document.add(group);
            }
        }
    }

    // Render Highlighted Squares (under pieces, over board)
    if let Some(highlights_slice) = highlights {
        for highlight in highlights_slice {
            if let Some(hex) = hex_map.get(&(highlight.q, highlight.r)) {
                if let Some(text) = highlight.text {
                    let text_element = Text::new(text)
                        .set("x", hex.x)
                        .set("y", hex.y)
                        .set("text-anchor", "middle")
                        .set("dominant-baseline", "central")
                        .set("font-size", "24px")
                        .set("fill", "black")
                        .set("font-family", "'Roboto Mono'") // The family name is the same for all weights
                        .set("font-weight", "bold")
                        .set("stroke", "white")
                        .set("stroke-width", "4.5")
                        .set("paint-order", "stroke");
                    document = document.add(text_element);
                }
            }
        }
    }

    if let Some(eval) = evaluation {
        for (i, &val) in eval.value_abs.iter().enumerate() {
            let circle_group = Group::new()
                .set(
                    "transform",
                    format!(
                        "translate({}, {})",
                        i as f64 * 80.0 - 80.0,
                        BOARD_VIS.view_box.1 + 40.0
                    ),
                )
                .add(
                    node::element::Circle::new()
                        .set("cx", 0)
                        .set("cy", 0)
                        .set("r", 35)
                        .set("fill", COLORS[i])
                        .set("fill-opacity", 0.4)
                        .set("stroke", "black")
                        .set("stroke-opacity", 0.6)
                        .set("stroke-width", 1),
                )
                .add(
                    Text::new(format!("{:.2}", val))
                        // .set("x", i as f64 * 80.0 - 80.0)
                        // .set("y", BOARD_VIS.view_box.1 + 40.0)
                        .set("text-anchor", "middle")
                        .set("dominant-baseline", "central")
                        .set("font-size", "20px")
                        .set("fill", if val >= 0.0 { "green" } else { "red" })
                        .set("font-family", "'Roboto Mono'") // The family name is the same for all weights
                        .set("font-weight", "bolder")
                        .set("stroke", "white")
                        .set("stroke-width", "3.5")
                        .set("paint-order", "stroke"),
                );
            document = document.add(circle_group);
        }
    }

    let mut text_groups = vec![];

    let mut arrows = arrows.to_vec();

    arrows.sort_by(|a, b| {
        let da = ((a.to_q as i32 - a.from_q as i32).pow(2)
            + (a.to_r as i32 - a.from_r as i32).pow(2)) as f64;
        let db = ((b.to_q as i32 - b.from_q as i32).pow(2)
            + (b.to_r as i32 - b.from_r as i32).pow(2)) as f64;
        db.partial_cmp(&da).unwrap()
    });

    for arrow_data in arrows {
        if let (Some(from_hex), Some(to_hex)) = (
            hex_map.get(&(arrow_data.from_q, arrow_data.from_r)),
            hex_map.get(&(arrow_data.to_q, arrow_data.to_r)),
        ) {
            let arrow_props = ArrowProps {
                from: Point {
                    x: from_hex.x,
                    y: from_hex.y,
                },
                to: Point {
                    x: to_hex.x,
                    y: to_hex.y,
                },
                color: &arrow_data.color,
                stroke_width: 12.0,
                opacity: arrow_data.opacity,
                text: arrow_data.text.as_deref(),
                trim_start: Some(35.0),
                trim_end: Some(10.0),
            };

            let (arrow_group, text_group) = create_arrow_component(arrow_props);

            document = document.add(arrow_group);
            text_groups.push(text_group);
        }
    }

    // So they appear on top of arrows
    for text_group in text_groups {
        document = document.add(text_group);
    }

    document = document.add(check_arrow_group);

    document
}

pub fn save_to_file(
    game: &TriHexChess,
    arrows: Vec<VisArrow>,
    highlights: Vec<VisHighlightedHex>,
    evaluation: Option<ValuesAbs<3>>,
    dir: &str,
    file_name: &str,
) -> String {
    let document = render_board(&game, &arrows, Some(highlights), evaluation, None, false);

    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create directory {}: {}", dir, e);
        panic!("Failed to create directory");
    }

    let svg_filename = format!("{dir}/{file_name}.svg");

    svg::save(svg_filename.as_str(), &document).unwrap();

    let written_pdf_file_path = convert_svg_to_pdf(
        svg_filename.as_str(),
        format!("{dir}/{file_name}.pdf").as_str(),
    )
    .unwrap_or_else(|e| {
        panic!("Failed to convert SVG to PDF: {}", e);
    });

    if let Err(e) = fs::remove_file(svg_filename.as_str()) {
        eprintln!("Failed to remove temporary SVG file: {}", e);
    }

    written_pdf_file_path
}
