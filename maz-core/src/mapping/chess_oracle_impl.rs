use crate::mapping::{Board, Oracle, OracleAnalysis, OracleError, Outcome};
use game_tri_chess::basics::Piece;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::PseudoLegalMove;
use game_tri_chess::pos::MemoryPos;
use oracle_tri_chess::oracle::{
    detect_endgame, to_canonical_key, DetectedEndgame, TriHexEndgameOracle,
};
use oracle_tri_chess::tablebase::{CanonicalKey, GameResult};

impl Oracle<TriHexChess> for TriHexEndgameOracle {
    fn probe(
        &self,
        board: &TriHexChess,
    ) -> Result<OracleAnalysis<<TriHexChess as Board>::Move>, OracleError> {
        // 1. Detect which endgame we are in. If it's not one we know,
        // return the specific `NotApplicable` error.
        let detected_endgame = detect_endgame(board).map_err(|_| OracleError::NotApplicable)?;

        // 2. Create the key and probe the correct table.
        let (key, table) = match detected_endgame {
            DetectedEndgame::Kqk { strong, weak } => (
                to_canonical_key(board, strong, weak, Piece::Queen).unwrap(),
                &self.kqk_table,
            ),
            DetectedEndgame::Krk { strong, weak } => (
                to_canonical_key(board, strong, weak, Piece::Rook).unwrap(),
                &self.krk_table,
            ),
        };

        let value = table.get(&key).ok_or(OracleError::NotApplicable)?;

        // 3. Convert the internal `TablebaseValue` to the public `OracleAnalysis`.

        let mut outcome_abs = vec![-1.0; board.player_num()];

        // Determine which player is strong and which is weak from the detected endgame.
        let (strong_side, weak_side) = match detected_endgame {
            DetectedEndgame::Kqk { strong, weak } => (strong, weak),
            DetectedEndgame::Krk { strong, weak } => (strong, weak),
        };

        // Get the `is_strong_side_to_move` flag FROM THE KEY.
        let is_strong_turn = match key {
            CanonicalKey::Kqk(k) => k.is_strong_side_to_move,
            CanonicalKey::Krk(k) => k.is_strong_side_to_move,
        };

        let outcome = match value.result {
            GameResult::Win => {
                if is_strong_turn {
                    Outcome::WonBy(strong_side as u8)
                } else {
                    Outcome::WonBy(weak_side as u8)
                }
            }
            GameResult::Loss => {
                if is_strong_turn {
                    Outcome::WonBy(weak_side as u8)
                } else {
                    Outcome::WonBy(strong_side as u8)
                }
            }
            GameResult::Draw => {
                let mut partial_draw_mask = 0u8;

                partial_draw_mask |= 1 << (strong_side as u8);
                partial_draw_mask |= 1 << (weak_side as u8);

                Outcome::PartialDraw(partial_draw_mask)
            }
        };

        match value.result {
            GameResult::Win => {
                // It's a win for the player whose turn it is.
                let winner_idx: usize = if is_strong_turn {
                    strong_side
                } else {
                    weak_side
                }
                .into();
                outcome_abs[winner_idx] = 1.0;
            }
            GameResult::Loss => {
                // It's a loss for the player whose turn it is, meaning the OTHER player wins.
                let winner_idx: usize = if is_strong_turn {
                    weak_side
                } else {
                    strong_side
                }
                .into();
                outcome_abs[winner_idx] = 1.0;
            }
            GameResult::Draw => {
                // It's a draw between the two active players.
                let p1_idx: usize = strong_side.into();
                let p2_idx: usize = weak_side.into();
                outcome_abs[p1_idx] = 0.0;
                outcome_abs[p2_idx] = 0.0;
            }
        }

        let best_move = PseudoLegalMove {
            from: MemoryPos(value.best_move_from),
            to: MemoryPos(value.best_move_to),
            move_type: game_tri_chess::moves::MoveType::Move,
        };
        // let mut policy = vec![0.0; board_mapper.policy_len()];
        //
        // let move_idx = board_mapper.move_to_index(board.player_current(), best_move);
        // policy[move_idx] = 1.0;

        Ok(OracleAnalysis {
            outcome_abs,
            best_move,
            outcome,
        })
    }
}
