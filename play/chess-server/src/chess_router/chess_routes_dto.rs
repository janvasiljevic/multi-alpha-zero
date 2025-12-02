use crate::chess_router::dto_traits::{ApiBoard, ApiMove};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::PseudoLegalMove;
use game_tri_chess::pos::FullCoordinates;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// #[derive(Deserialize, Serialize, JsonSchema, Debug)]
// pub enum ChessPlayerDto {
//     W,
//     G,
//     B,
// }
//
// impl ChessPlayerDto {
//     // fn new_from_internal(player: &Color) -> Self {
//     //     match player {
//     //         Color::White => ChessPlayerDto::W,
//     //         Color::Gray => ChessPlayerDto::G,
//     //         Color::Black => ChessPlayerDto::B,
//     //     }
//     // }
//
//     fn to_internal(&self) -> Color {
//         match self {
//             ChessPlayerDto::W => Color::White,
//             ChessPlayerDto::G => Color::Gray,
//             ChessPlayerDto::B => Color::Black,
//         }
//     }
// }

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub struct ChessMoveDto {
    pub from: FullCoordinates,
    pub to: FullCoordinates,
    pub promotion: Option<String>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub struct ChessMoveWrapperDto {
    pub inner: ChessMoveDto,
    pub prior: f32,
    pub confidence: f32,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub struct ChessBoardDto {
    pub board_state: String,
}

impl ApiMove<TriHexChess> for ChessMoveWrapperDto {
    fn new_from_internal(mv: PseudoLegalMove, score: f32, prior: f32) -> Self {
        ChessMoveWrapperDto {
            inner: ChessMoveDto {
                from: FullCoordinates::from_memory_pos(mv.from),
                to: FullCoordinates::from_memory_pos(mv.to),
                promotion: mv
                    .get_promotion_piece()
                    .map(|p| p.to_char().to_ascii_lowercase().to_string()),
            },
            prior,
            confidence: score,
        }
    }
}

impl ApiBoard<TriHexChess> for ChessBoardDto {
    fn new_from_internal(board: &TriHexChess) -> Self {
        ChessBoardDto {
            board_state: board.to_fen(),
        }
    }
}
