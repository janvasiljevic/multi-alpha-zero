use crate::vis_chess_util;
use crate::vis_chess_util::{COLORS, SIZE};
use game_tri_chess::pos::MemoryPos;
use std::collections::HashMap;

pub struct VisChessTile {
    pub q: i8,
    pub r: i8,
    pub s: i8,
    pub x: f64,
    pub y: f64,
    pub i: u8,
    pub notation: String,

    pub rank: Option<String>,
    pub file: Option<String>,

    pub color: &'static str,
}

pub fn is_border(hex: &VisChessTile) -> bool {
    hex.q == 3 || hex.r == 3 || hex.s == 3 || hex.q == -4 || hex.r == -4 || hex.s == -4
}

fn qrs_to_algebraic(g_q: i8, g_r: i8) -> String {
    format!(
        "{}{}",
        A_FILES[(g_q + 7) as usize],
        A_RANKS[(g_r + 7) as usize]
    )
}



impl VisChessTile {
    pub fn new(q: i8, r: i8, i: u8) -> Self {
        let col = q;
        let row = r + (q - (q & 1)) / 2;

        let rank: Option<String> = if RANKS_IDX.contains(&(i as usize)) {
            let rank_idx = RANKS_IDX.iter().position(|&x| x == i as usize).unwrap();
            Some(A_RANKS[rank_idx].to_string())
        } else {
            None
        };

        let file: Option<String> = if FILES_IDX.contains(&(i as usize)) {
            let file_idx = FILES_IDX.iter().position(|&x| x == i as usize).unwrap();
            Some(A_FILES[file_idx].to_string())
        } else {
            None
        };

        let color = match (col.rem_euclid(2), row.rem_euclid(3)) {
            (0, 0) => COLORS[0], // W
            (0, 1) => COLORS[1], // G
            (0, 2) => COLORS[2], // B
            (1, 0) => COLORS[2], // B
            (1, 1) => COLORS[0], // W
            (1, 2) => COLORS[1], // G
            _ => unreachable!(),
        };

        Self {
            q,
            i,
            s: -1 - q - r,
            r,
            x: vis_chess_util::get_x(q),
            y: vis_chess_util::get_y(q, r),
            notation: qrs_to_algebraic(q, r),
            color,
            rank,
            file,
        }
    }
}

pub struct VisChessBoard {
    pub hexagons: Vec<VisChessTile>,
    pub view_box: (f64, f64, f64, f64),
}

const A_FILES: [&str; 15] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
];
const A_RANKS: [&str; 15] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
];

const FILES_IDX: [usize; 15] = [71, 79, 87, 0, 1, 2, 3, 4, 5, 6, 7, 56, 48, 40, 32];
const RANKS_IDX: [usize; 15] = [7, 15, 23, 32, 33, 34, 35, 36, 37, 38, 39, 88, 80, 72, 64];


impl VisChessBoard {
    pub fn new() -> Self {
        let mut el_map = HashMap::new();

        for i in 0..96 {
            let mem_pos = MemoryPos(i);
            let qrs = mem_pos.to_qrs_global();
            el_map.insert((qrs.q, qrs.r, i), VisChessTile::new(qrs.q, qrs.r, i));
        }

        let hexagons: Vec<VisChessTile> = el_map.into_values().collect();

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
        let pad = SIZE * 1.3;
        view_box.0 -= pad;
        view_box.1 -= pad;
        let width = view_box.2 - view_box.0 + pad;
        let height = view_box.3 - view_box.1 + pad * 1.15; // Extra space
        view_box.2 = width;
        view_box.3 = height;

        Self { hexagons, view_box }
    }
}

pub struct VisHighlightedHex {
    pub q: i8,
    pub r: i8,
    pub color: String,
    pub opacity: f64,
    pub text: Option<String>,
}

lazy_static::lazy_static! {
    pub static ref BOARD_VIS: VisChessBoard = VisChessBoard::new();
}
