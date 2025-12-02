use crate::vis_chess_util::{get_x, get_y};
use game_tri_chess::chess_game::TriHexChess;
use svg::node::element::{Group, Line, Polygon};

pub fn create_check(board: &TriHexChess) -> (Group, Group) {
    let mut check_group = Group::new().set("id", "arrows");

    let mut is_color_checked = vec![false, false, false];

    for check in board.get_check_metadata() {
        if !is_color_checked[check.player_attacked as usize] {
            let x = get_x(check.king.q);
            let y = get_y(check.king.q, check.king.r);

            let hexagon_highlight = Polygon::new()
                .set("points", "50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5")
                .set("fill", "url(#checkGradient)") // Uses the gradient from <defs>
                .set("stroke", "black")
                .set("stroke-width", 2);

            let positioned_highlight = Group::new()
                .set("transform", format!("translate({:.2}, {:.2})", x, y))
                .add(hexagon_highlight);

            check_group = check_group.add(positioned_highlight);

            is_color_checked[check.player_attacked as usize] = true;
        }
    }

    let mut arrow_group = Group::new().set("id", "check-arrows");

    for check in board.get_check_metadata() {
        const STROKE_WIDTH: f64 = 3.0;
        const TRIM_AMOUNT: f64 = 35.0; // Distance to shorten the arrow from both ends
        const ARROWHEAD_LENGTH: f64 = 15.0;
        const ARROWHEAD_WIDTH: f64 = 12.0;

        let start_x = get_x(check.attack.q);
        let start_y = get_y(check.attack.q, check.attack.r);
        let end_x = get_x(check.king.q);
        let end_y = get_y(check.king.q, check.king.r);

        let dx = end_x - start_x;
        let dy = end_y - start_y;
        let len = (dx.powi(2) + dy.powi(2)).sqrt();

        if len < 1e-6 {
            continue;
        }

        let unit_dx = dx / len;
        let unit_dy = dy / len;

        // Trimming
        let trimmed_start_x = start_x + unit_dx * TRIM_AMOUNT;
        let trimmed_start_y = start_y + unit_dy * TRIM_AMOUNT;

        let trimmed_end_x = end_x - unit_dx * TRIM_AMOUNT;
        let trimmed_end_y = end_y - unit_dy * TRIM_AMOUNT;

        // Shape
        let line_end_x = trimmed_end_x - unit_dx * ARROWHEAD_LENGTH;
        let line_end_y = trimmed_end_y - unit_dy * ARROWHEAD_LENGTH;

        // Arrowhead polygon points
        let perp_dx = -unit_dy;
        let perp_dy = unit_dx;

        let p1_x = trimmed_end_x; // Tip of the arrow
        let p1_y = trimmed_end_y;

        let p2_x = line_end_x + perp_dx * (ARROWHEAD_WIDTH / 2.0);
        let p2_y = line_end_y + perp_dy * (ARROWHEAD_WIDTH / 2.0);

        let p3_x = line_end_x - perp_dx * (ARROWHEAD_WIDTH / 2.0);
        let p3_y = line_end_y - perp_dy * (ARROWHEAD_WIDTH / 2.0);

        let line = Line::new()
            .set("x1", trimmed_start_x)
            .set("y1", trimmed_start_y)
            .set("x2", line_end_x)
            .set("y2", line_end_y)
            .set("stroke", "red")
            .set("stroke-width", STROKE_WIDTH)
            .set("stroke-linecap", "round");

        let arrowhead = Polygon::new()
            .set(
                "points",
                format!(
                    "{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}",
                    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
                ),
            )
            .set("fill", "red");

        let arrow_group_inner = Group::new().add(line).add(arrowhead);
        arrow_group = arrow_group.add(arrow_group_inner);
    }

    (check_group, arrow_group)
}
