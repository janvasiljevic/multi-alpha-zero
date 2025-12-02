use game_tri_chess::pos::MemoryPos;
use svg::node::element::{Group, Line, Polygon, TSpan, Text};

#[derive(Clone, Debug)]
pub struct VisArrow {
    pub from_i: MemoryPos,
    pub from_q: i8,
    pub from_r: i8,
    pub to_i: MemoryPos,
    pub to_q: i8,
    pub to_r: i8,
    pub opacity: f64,
    pub text: Option<String>,
    pub color: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub struct ArrowProps<'a> {
    pub from: Point,
    pub to: Point,
    pub color: &'a str,
    pub stroke_width: f64,
    pub opacity: f64,
    pub text: Option<&'a str>,
    pub trim_start: Option<f64>,
    pub trim_end: Option<f64>,
}

const ARROWHEAD_LENGTH: f64 = 25.0;
const ARROWHEAD_WIDTH: f64 = 18.0;

pub fn create_arrow_component(props: ArrowProps) -> (Group, Group) {
    let dx = props.to.x - props.from.x;
    let dy = props.to.y - props.from.y;
    let len = (dx.powi(2) + dy.powi(2)).sqrt();

    if len < 1e-6 {
        return (Group::new(), Group::new());
    }

    let unit_dx = dx / len;
    let unit_dy = dy / len;

    let trim_start_val = props.trim_start.unwrap_or(0.0);
    let trim_end_val = props.trim_end.unwrap_or(0.0);

    let start_point = Point {
        x: props.from.x + unit_dx * trim_start_val,
        y: props.from.y + unit_dy * trim_start_val,
    };

    let end_point = Point {
        x: props.to.x - unit_dx * trim_end_val,
        y: props.to.y - unit_dy * trim_end_val,
    };

    let line_end = Point {
        x: end_point.x - unit_dx * ARROWHEAD_LENGTH,
        y: end_point.y - unit_dy * ARROWHEAD_LENGTH,
    };

    let perp_dx = -unit_dy;
    let perp_dy = unit_dx;

    let point1 = end_point;
    let point2 = Point {
        x: line_end.x + perp_dx * (ARROWHEAD_WIDTH / 2.0),
        y: line_end.y + perp_dy * (ARROWHEAD_WIDTH / 2.0),
    };
    let point3 = Point {
        x: line_end.x - perp_dx * (ARROWHEAD_WIDTH / 2.0),
        y: line_end.y - perp_dy * (ARROWHEAD_WIDTH / 2.0),
    };
    let polygon_points = format!(
        "{},{} {},{} {},{}",
        point1.x, point1.y, point2.x, point2.y, point3.x, point3.y
    );

    let mut arrow_group = Group::new().set("opacity", props.opacity);

    if props.opacity == 1.0 {
        let line = Line::new()
            .set("x1", start_point.x)
            .set("y1", start_point.y)
            .set("x2", line_end.x)
            .set("y2", line_end.y)
            .set("stroke", "white")
            .set("stroke-width", props.stroke_width + 4.0)
            .set("stroke-linecap", "round");

        let arrowhead = Polygon::new()
            .set("points", polygon_points.clone())
            .set("fill", props.color)
            .set("stroke", "white")
            .set("stroke-width", 4);

        arrow_group = arrow_group.add(line).add(arrowhead);
    }

    let line = Line::new()
        .set("x1", start_point.x)
        .set("y1", start_point.y)
        .set("x2", line_end.x)
        .set("y2", line_end.y)
        .set("stroke", props.color)
        .set("stroke-width", props.stroke_width)
        .set("stroke-linecap", "round");

    let arrowhead = Polygon::new()
        .set("points", polygon_points.clone())
        .set("fill", props.color);

    arrow_group = arrow_group.add(line).add(arrowhead);

    let mut text_group = Group::new();

    // Optional Text
    if let Some(text_content) = props.text {
        let mid_x = (start_point.x + line_end.x) / 2.0;
        let mid_y = (start_point.y + line_end.y) / 2.0;

        let mut angle_deg = dy.atan2(dx).to_degrees();
        if angle_deg > 90.0 {
            angle_deg -= 180.0;
        } else if angle_deg < -90.0 {
            angle_deg += 180.0;
        }

        let mut text_element = Text::new("")
            .set("x", mid_x)
            .set("y", mid_y)
            .set(
                "transform",
                format!("rotate({} {} {})", angle_deg, mid_x, mid_y),
            )
            .set("fill", "black")
            .set("font-family", "'Inter 18pt'")
            .set("font-size", "14px")
            .set("font-weight", "400")
            .set("text-anchor", "middle")
            .set("dominant-baseline", "middle");

        for (i, line) in text_content.split('\n').enumerate() {
            let dy = if i == 0 { "0" } else { "1.2em" };
            let tspan = TSpan::new(line).set("x", mid_x).set("dy", dy);
            text_element = text_element.add(tspan);
        }

        text_group = text_group.add(text_element);
    }

    (arrow_group, text_group)
}
