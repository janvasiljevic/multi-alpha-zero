use crate::basics::{Color, Piece, COLORS};
use crate::chess_game::TriHexChess;
use crate::constants::{
    ALL_DIR_V, DIAGONAL_V, KNIGHT_V, LINE_V, L_KING_P_ATTACK_V, R_KING_P_ATTACK_V,
};
use crate::pos::{FullCoordinates, MemoryPos, QRSPos};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Represents an attack on the king by a piece
#[cfg_attr(feature = "wasm", wasm_bindgen(inspectable, js_name = AttackInfo))]
#[derive(Clone, Copy, Debug)]
pub struct CheckInformation {
    /// Position of the attacking piece
    pub attack: FullCoordinates,
    /// Position of the king who is being attacked
    pub king: FullCoordinates,
    /// The player who is being attacked
    pub player_attacked: Color,
}

impl CheckInformation {
    pub fn new(attack_i: MemoryPos, king_pos: MemoryPos, player_attacked: Color) -> Self {
        CheckInformation {
            attack: FullCoordinates::from_memory_pos(attack_i),
            king: FullCoordinates::from_memory_pos(king_pos),
            player_attacked,
        }
    }
}

impl TriHexChess {
    // Helper function for locating checking pieces
    fn locate_check_sliding(
        &self,
        local_qrs: QRSPos,
        offset: u8,
        array: &[(i8, i8, i8)],
        other_piece: Piece,
        turn: Color,
        add_attack: &mut impl FnMut(MemoryPos),
    ) {
        for (q, r, s) in array {
            let mut qrs = local_qrs.add(*q, *r, *s);

            'inner: while qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_empty() {
                    qrs.set(qrs.q + *q, qrs.r + *r, qrs.s + *s);
                    continue 'inner;
                }

                if self.state.buffer[to_i].is_player(turn) {
                    break 'inner;
                }

                let piece = self.state.buffer[to_i].piece().unwrap();

                if piece == other_piece || piece == Piece::Queen {
                    add_attack(to_i);
                }

                break 'inner;
            }
        }
    }

    pub fn get_check_metadata(&self) -> Vec<CheckInformation> {
        let mut all_check_info = Vec::new();

        for color in COLORS {
            if let Some(king_pos) = self.state.buffer.king_mem_pos(color) {
                all_check_info.extend(self.locate_checking_pieces(king_pos, color));
            }
        }

        all_check_info
    }

    // Helper functions for locating checking pieces
    pub fn locate_checking_pieces(
        &self,
        king_pos: MemoryPos,
        turn: Color,
    ) -> Vec<CheckInformation> {
        let mut acc = Vec::new();
        let offset = turn.get_offset();
        // Local coordinates of the king
        let king_l_qrs = king_pos.to_qrs_local(turn);

        let mut add_attack = |attack_i: MemoryPos| {
            acc.push(CheckInformation::new(attack_i, king_pos, turn));
        };

        // Bishop and queen
        self.locate_check_sliding(
            king_l_qrs,
            offset,
            &DIAGONAL_V,
            Piece::Bishop,
            turn,
            &mut add_attack,
        );

        // Rook and queen
        self.locate_check_sliding(
            king_l_qrs,
            offset,
            &LINE_V,
            Piece::Rook,
            turn,
            &mut add_attack,
        );

        let mut qrs = QRSPos::default();

        // Knight
        for (q, r, s) in KNIGHT_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_enemy(turn)
                    && self.state.buffer[to_i].piece().unwrap() == Piece::Knight
                {
                    acc.push(CheckInformation::new(to_i, king_pos, turn));
                }
            }
        }

        // King -> This shouldn't be possible, but just in case
        for (q, r, s) in ALL_DIR_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_enemy(turn)
                    && self.state.buffer[to_i].piece().unwrap() == Piece::King
                {
                    acc.push(CheckInformation::new(to_i, king_pos, turn));
                }
            }
        }

        let right_player = turn.right_player();
        let left_player = turn.left_player();

        for (q, r, s) in R_KING_P_ATTACK_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].matches(right_player, Piece::Pawn) {
                    acc.push(CheckInformation::new(to_i, king_pos, turn));
                }
            }
        }

        for (q, r, s) in L_KING_P_ATTACK_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].matches(left_player, Piece::Pawn) {
                    acc.push(CheckInformation::new(to_i, king_pos, turn));
                }
            }
        }

        acc
    }
}
