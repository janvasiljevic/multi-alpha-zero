use crate::basics::Piece;
use crate::pos::MEMORY_TO_Q;

type MoveVector = (i8, i8, i8);

/// Possible promotion pieces
pub const PROM_P: [Piece; 4] = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];

/// 12 knight jumps
pub const KNIGHT_V: [MoveVector; 12] = [
    (1, -3, 2),
    (-1, -2, 3),
    (2, -3, 1),
    (-2, -1, 3),
    (3, -2, -1),
    (3, -1, -2),
    (-3, 1, 2),
    (-3, 2, 1),
    (2, 1, -3),
    (1, 2, -3),
    (-1, 3, -2),
    (-2, 3, -1),
];

/// Pawn attack left (used for en passant)
pub const L_P_ATTACK_V: MoveVector = (-1, 2, -1);

/// Pawn attack right (used for en passant)
pub const R_P_ATTACK_V: MoveVector = (1, 1, -2);

/// Pawn attack vectors
pub const P_ATTACK: [MoveVector; 2] = [L_P_ATTACK_V, R_P_ATTACK_V];

/// Reverse lookup for P_ATTACK_V (to determine checks)
pub const R_KING_P_ATTACK_V: [MoveVector; 2] = [(1, 1, -2), (2, -1, -1)];

/// Reverse lookup for R_P_ATTACK (to determine checks)
pub const L_KING_P_ATTACK_V: [MoveVector; 2] = [(-1, 2, -1), (-2, 1, 1)];

/// 6 diagonal directions (for bishops)
pub const DIAGONAL_V: [MoveVector; 6] = [
    (2, -1, -1),
    (1, -2, 1),
    (-1, -1, 2),
    (-2, 1, 1),
    (-1, 2, -1),
    (1, 1, -2),
];

/// 6 line directions
pub const LINE_V: [MoveVector; 6] = [
    (1, 0, -1),
    (1, -1, 0),
    (0, -1, 1),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 1, -1),
];

/// 12 directions (for queens and kings)
pub const ALL_DIR_V: [MoveVector; 12] = [
    (1, 0, -1),
    (1, -1, 0),
    (0, -1, 1),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 1, -1),
    (2, -1, -1),
    (1, -2, 1),
    (-1, -1, 2),
    (-2, 1, 1),
    (-1, 2, -1),
    (1, 1, -2),
];

const A_FILE: [char; 15] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
];

const A_RANK: [u8; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

pub const fn qrs_to_buffer_indexes(q: i8, r: i8) -> (usize, usize) {
    ((q + 7) as usize, (r + 7) as usize)
}

pub const MEM_POS_TO_FILE: [char; 96] = {
    let mut lut = [' '; 96];
    let mut i = 0;
    while i < 96 {
        lut[i] = A_FILE[qrs_to_buffer_indexes(MEMORY_TO_Q[0][i], MEMORY_TO_Q[1][i]).0];
        i += 1;
    }
    lut
};

pub const MEM_POS_TO_RANK: [u8; 96] = {
    let mut lut = [0; 96];
    let mut i = 0;
    while i < 96 {
        lut[i] = A_RANK[qrs_to_buffer_indexes(MEMORY_TO_Q[0][i], MEMORY_TO_Q[1][i]).1];
        i += 1;
    }
    lut
};
