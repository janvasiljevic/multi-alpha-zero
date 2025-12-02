use std::collections::HashMap;

const SIZE: usize = 50;
const SQRT_3: f64 = 1.7320508075688772;

pub fn get_x(q: i32, r: i32, size: f64) -> f64 {
    size * SQRT_3 * q as f64 + (SQRT_3 / 2.0) * r as f64 * size
}

pub fn get_y(q: i32, r: i32, size: f64) -> f64 {
    (3.0 / 2.0) * r as f64 * size
}

#[derive(Clone, Debug)]
pub struct VisHexTile {
    pub q: i32,
    pub r: i32,
    pub s: i32,
    pub x: f64,
    pub y: f64,
}

impl VisHexTile {
    pub fn new(q: i32, r: i32, i: u8) -> Self {
        Self {
            q,
            r,
            s: -q - r, // In hex coordinates, q + r + s = 0
            x: get_x(q, r, SIZE as f64),
            y: get_y(q, r, SIZE as f64),
        }
    }
}

pub struct VisHexBoard {
    pub hexagons: Vec<VisHexTile>,
    pub view_box: (f64, f64, f64, f64),
}

impl VisHexBoard {
    pub fn new(radius: i8) -> Self {
        let mut el_map = HashMap::new();

        for r in -(radius as i32)..=(radius as i32) {
            let q_start = (-radius as i32).max(-r - (radius as i32));
            let q_end = (radius as i32).min(-r + (radius as i32));

            for q in q_start..=q_end {
                let key = (q, r);
                let tile = VisHexTile::new(q, r, 0);
                el_map.insert(key, tile);
            }
        }

        let hexagons: Vec<VisHexTile> = el_map.values().cloned().collect();

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for hex in &hexagons {
            min_x = min_x.min(hex.x);
            min_y = min_y.min(hex.y);
            max_x = max_x.max(hex.x);
            max_y = max_y.max(hex.y);
        }

        let mut view_box = (min_x, min_y, max_x, max_y);
        let pad = SIZE as f64 * 2.0;
        view_box.0 -= pad;
        view_box.1 -= pad;
        let width = view_box.2 - view_box.0 + pad;
        let height = view_box.3 - view_box.1 + pad * 1.15; // Extra space
        view_box.2 = width;
        view_box.3 = height;

        Self { hexagons, view_box }
    }
}
