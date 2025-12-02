use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub enum GameResult {
    Win,
    Loss,
    Draw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct TablebaseValue {
    pub result: GameResult,
    pub distance: u8,
    pub best_move_from: u8,
    pub best_move_to: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct CanonicalKeyKqk {
    pub strong_king_pos: u8,
    pub strong_queen_pos: u8,
    pub weak_king_pos: u8,
    pub is_strong_side_to_move: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct CanonicalKeyKrk {
    pub strong_king_pos: u8,
    pub strong_rook_pos: u8,
    pub weak_king_pos: u8,
    pub is_strong_side_to_move: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub enum CanonicalKey {
    Kqk(CanonicalKeyKqk),
    Krk(CanonicalKeyKrk),
}

pub type Tablebase = HashMap<CanonicalKey, TablebaseValue>;

#[derive(Debug, Clone, Copy)]
pub enum Endgame {
    Kqk,
    Krk,
}
