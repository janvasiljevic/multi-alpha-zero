use svg::node::element::{Group, Polygon};

#[derive(Clone, Debug)]
pub struct VisTurnOptions {
    pub both_sided: bool,
    pub stroke_width: f64,
    pub angle_offset: f64,
    pub angle_stride: f64,
    pub x_offsets: Vec<f64>,
}

impl Default for VisTurnOptions {
    fn default() -> Self {
        Self {
            both_sided: true,
            stroke_width: 1.0,
            angle_offset: -90.0,
            angle_stride: 120.0,
            x_offsets: vec![0.0, 0.0, 0.0], // No offset by default
        }
    }
}

pub fn vis_turn(
    current_turn: usize, // Which color to use
    colors: Vec<&str>,   // Colors for each player
    radius: f64,
    options: VisTurnOptions,
) -> Group {
    assert_eq!(colors.len(), 3);

    let mut group = Group::new();

    if current_turn >= colors.len() {
        return group;
    }

    let i = current_turn;

    let angle_offset = options.angle_offset;
    let base_angle_deg = angle_offset + i as f64 * options.angle_stride;

    let base_angle_rad = base_angle_deg.to_radians();
    let perp_dir_x = -base_angle_rad.sin();
    let perp_dir_y = base_angle_rad.cos();

    let offset_x = perp_dir_x * options.x_offsets[i];
    let offset_y = perp_dir_y * options.x_offsets[i];

    let opacity = 1.0; // Since we only draw the current turn, this is always 1.0

    let mut angles_deg = vec![base_angle_deg];
    if options.both_sided {
        angles_deg.push(base_angle_deg + 180.0);
    }

    for angle_deg in angles_deg {
        let angle_rad = angle_deg.to_radians();

        let dir_x = angle_rad.cos();
        let dir_y = angle_rad.sin();

        let arrow_perp_x = -dir_y;
        let arrow_perp_y = dir_x;

        let inner_dist = radius - 28.0;
        let outer_dist = radius - 4.0;
        let width = 18.0;

        let p1_x_base = dir_x * inner_dist;
        let p1_y_base = dir_y * inner_dist;

        let p2_x_base = dir_x * outer_dist + arrow_perp_x * width;
        let p2_y_base = dir_y * outer_dist + arrow_perp_y * width;

        let p3_x_base = dir_x * outer_dist - arrow_perp_x * width;
        let p3_y_base = dir_y * outer_dist - arrow_perp_y * width;

        let arrow = Polygon::new()
            .set(
                "points",
                format!(
                    // Add the calculated offset to each point
                    "{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}",
                    p1_x_base + offset_x,
                    p1_y_base + offset_y,
                    p2_x_base + offset_x,
                    p2_y_base + offset_y,
                    p3_x_base + offset_x,
                    p3_y_base + offset_y
                ),
            )
            .set("fill", colors[i])
            .set("opacity", opacity)
            .set("stroke", "black")
            .set("stroke-width", options.stroke_width);

        group = group.add(arrow);
    }

    group
}
