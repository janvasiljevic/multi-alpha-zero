use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{MoveType, PseudoLegalMove};
use maz_core::mapping::Board;

pub const SIZE: f64 = 50.0;
// pub const COLORS: [&str; 3] = ["#F7CB9A", "#DCA66B", "#C18545"]; // W, G, B
pub const COLORS: [&str; 3] = ["#FFFFFF", "#C5C5C5", "#8E8E8E"];
// W, G, B
const SQRT_3: f64 = 1.73205081;

pub fn get_x(q: i8) -> f64 {
    SIZE * 1.5 * (q as f64) + SIZE * 0.5
}

pub fn get_y(q: i8, r: i8) -> f64 {
    let q_f = q as f64;
    let r_f = r as f64;

    let term1 = (-SQRT_3 / 2.0) * q_f;
    let term2 = -SQRT_3 * r_f;
    let offset = (SQRT_3 * SIZE) / 2.0;

    SIZE * (term1 + term2) - offset
}

pub fn get_move_color(mv: PseudoLegalMove, board: &TriHexChess) -> String {
    let is_capture = board.state.buffer[mv.get_capture_slot()].get().is_some();

    match mv.move_type {
        MoveType::Move => {
            if is_capture {
                "red".to_string()
            } else {
                "#FFDE21".to_string()
            }
        }
        MoveType::DoublePawnPush => "orange".to_string(),
        MoveType::EnPassant(_) => "purple".to_string(),
        MoveType::Promotion(_) => "green".to_string(),
        MoveType::EnPassantPromotion(_) => "green".to_string(),
        MoveType::CastleKingSide => "blue".to_string(),
        MoveType::CastleQueenSide => "blue".to_string(),
    }
}
