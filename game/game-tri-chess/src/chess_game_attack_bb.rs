use crate::basics::{Piece, QRSPos, COLORS};
use crate::chess_game::TriHexChess;
use crate::constants::{ALL_DIR_V, DIAGONAL_V, KNIGHT_V, LINE_V, P_ATTACK};
use crate::pos::MemoryPos;

const fn init_pawn_attacks() -> [[u128; 96]; 3] {
    let mut attacks = [[0u128; 96]; 3];
    let mut color_idx = 0;
    while color_idx < 3 {
        let color = COLORS[color_idx];
        let offset = color.get_offset();
        let mut from_sq = 0;
        while from_sq < 96 {
            let from_pos = MemoryPos(from_sq as u8);
            let local_qrs = from_pos.to_qrs_local(color);
            let mut bb = 0u128;

            let mut attack_idx = 0;
            while attack_idx < P_ATTACK.len() {
                let (q, r, s) = P_ATTACK[attack_idx];
                let attack_qrs = local_qrs.add(q, r, s);
                if attack_qrs.is_in() {
                    let attack_pos = attack_qrs.to_pos().to_global(offset);
                    bb |= 1u128 << attack_pos.0;
                }
                attack_idx += 1;
            }
            attacks[color_idx][from_sq] = bb;
            from_sq += 1;
        }
        color_idx += 1;
    }
    attacks
}

/// Generic function to generate attack bitboards for "leaper" pieces (Knight, King).
/// Their moves are absolute, so the pattern is the same regardless of player color.
const fn init_leaper_attacks(deltas: &[(i8, i8, i8)]) -> [u128; 96] {
    let mut attacks = [0u128; 96];
    let mut from_sq = 0;
    while from_sq < 96 {
        let from_pos = MemoryPos(from_sq as u8);
        let qrs = from_pos.to_qrs_global();
        let mut bb = 0u128;

        let mut delta_idx = 0;
        while delta_idx < deltas.len() {
            let (q, r, s) = deltas[delta_idx];
            let attack_qrs = qrs.add(q, r, s);
            if attack_qrs.is_in() {
                let attack_pos = attack_qrs.to_pos();
                bb |= 1u128 << attack_pos.0;
            }
            delta_idx += 1;
        }
        attacks[from_sq] = bb;
        from_sq += 1;
    }
    attacks
}

static PAWN_ATTACKS_BB: [[u128; 96]; 3] = init_pawn_attacks();
static KNIGHT_ATTACKS_BB: [u128; 96] = init_leaper_attacks(&KNIGHT_V);
static KING_ATTACKS_BB: [u128; 96] = init_leaper_attacks(&ALL_DIR_V);

#[inline(always)]
fn piece_to_index(piece: Piece) -> usize {
    (piece as u8 - 1) as usize
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PerPieceBitboardAttackData {
    /// Attack bitboards. Indexed by: [Color as usize][piece_to_index(Piece)].
    /// The inner array has 6 slots for Pawn, Knight, Bishop, Rook, Queen, King.
    pub attacks: [[u128; 6]; 3],
    /// Is king in check for each player. Indexed by Color as usize.
    pub is_in_check: [bool; 3],
}

impl TriHexChess {
    pub fn calculate_per_piece_bitboard_attack_data(&self) -> PerPieceBitboardAttackData {
        let mut data = PerPieceBitboardAttackData::default();

        let occupied_bb = self.state.buffer.get_occupied_bitboard();
        let mut kings_pos: [Option<MemoryPos>; 3] = [None, None, None];

        for (pos, slot) in self.state.buffer.non_empty_iter() {
            let (player, piece) = slot.get().unwrap();
            let player_idx = player as usize;

            if piece == Piece::King {
                kings_pos[player_idx] = Some(pos);
            }

            let attacks = match piece {
                // Luts
                Piece::Pawn => PAWN_ATTACKS_BB[player_idx][pos.0 as usize],
                Piece::Knight => KNIGHT_ATTACKS_BB[pos.0 as usize],
                Piece::King => KING_ATTACKS_BB[pos.0 as usize],

                // Sliding
                Piece::Bishop => {
                    let offset = player.get_offset();
                    let local_qrs = pos.to_qrs_local(player);
                    self.get_sliding_attacks_bb(occupied_bb, local_qrs, offset, &DIAGONAL_V)
                }
                Piece::Rook => {
                    let offset = player.get_offset();
                    let local_qrs = pos.to_qrs_local(player);
                    self.get_sliding_attacks_bb(occupied_bb, local_qrs, offset, &LINE_V)
                }
                Piece::Queen => {
                    let offset = player.get_offset();
                    let local_qrs = pos.to_qrs_local(player);
                    self.get_sliding_attacks_bb(occupied_bb, local_qrs, offset, &ALL_DIR_V)
                }
            };

            let piece_idx = piece_to_index(piece);
            data.attacks[player_idx][piece_idx] |= attacks;
        }

        // Check status
        for king_owner in COLORS {
            if let Some(king_pos) = kings_pos[king_owner as usize] {
                let king_bit = 1u128 << king_pos.0;

                let opponent1 = king_owner.right_player();
                let opponent2 = king_owner.left_player();

                let opponent1_total_attacks = data.attacks[opponent1 as usize]
                    .iter()
                    .fold(0u128, |acc, &bb| acc | bb);

                let opponent2_total_attacks = data.attacks[opponent2 as usize]
                    .iter()
                    .fold(0u128, |acc, &bb| acc | bb);

                if (opponent1_total_attacks & king_bit != 0)
                    || (opponent2_total_attacks & king_bit != 0)
                {
                    data.is_in_check[king_owner as usize] = true;
                }
            }
        }

        data
    }

    #[inline]
    fn get_sliding_attacks_bb(
        &self,
        occupied: u128,
        local_qrs: QRSPos,
        offset: u8,
        directions: &[(i8, i8, i8)],
    ) -> u128 {
        let mut attacks = 0u128;
        for (q, r, s) in directions {
            let mut qrs = local_qrs.add(*q, *r, *s);

            while qrs.is_in() {
                let to_pos = qrs.to_pos().to_global(offset);
                let to_bit = 1u128 << to_pos.0;

                attacks |= to_bit;

                // If we hit an occupied square (friend/enemy), stop sliding.
                if occupied & to_bit != 0 {
                    break;
                }

                qrs = qrs.add(*q, *r, *s);
            }
        }
        attacks
    }
}
