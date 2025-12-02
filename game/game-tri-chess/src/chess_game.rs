use crate::basics::{
    CastleFlags, Color, EnPassantState, MemoryBuffer, MemorySlot, NormalState, Piece, COLORS,
};
use crate::chess_zobrist::PRECOMPUTED_ZOBRIST;
use crate::constants::{
    ALL_DIR_V, DIAGONAL_V, KNIGHT_V, LINE_V, L_KING_P_ATTACK_V, L_P_ATTACK_V, PROM_P, P_ATTACK,
    R_KING_P_ATTACK_V, R_P_ATTACK_V,
};
use crate::fen::to_fen;
use crate::moves::{ChessMoveStore, MoveType, PassantWithPromotion, PseudoLegalMove, TurnCache};
use crate::pos::{MemoryPos, QRSPos};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::phase::Phase;

use crate::repetition_history::RepetitionHistory;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct State {
    pub buffer: MemoryBuffer,
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    pub phase: Phase,
    pub castle: CastleFlags,
    pub en_passant: EnPassantState,
    pub third_move: u16,
    pub turn_counter: u16,
}

impl State {
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    pub fn empty(turn: Color) -> Self {
        let normal_state = NormalState::new_with_turn(turn);

        State {
            buffer: MemoryBuffer::default(),
            phase: Phase::Normal(normal_state),
            castle: CastleFlags::default(),
            en_passant: EnPassantState::default(),
            third_move: 0,
            turn_counter: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct TriHexChess {
    pub state: State,
    pub is_using_grace_period: bool,
    pub zobrist_hash: u64,
    pub repetition_history: RepetitionHistory,
}

impl Hash for TriHexChess {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.zobrist_hash.hash(state);
    }
}

impl PartialEq for TriHexChess {
    fn eq(&self, other: &Self) -> bool {
        // TODO: Perform a test for this! (But currently looks good)
        if self.zobrist_hash == other.zobrist_hash {
            if self.state.buffer != other.state.buffer
                || self.state.phase != other.state.phase
                || self.state.castle != other.state.castle
                || self.state.en_passant != other.state.en_passant
            {
                println!("Zobrist hash is the same, but states are different!");
                println!("Self: {:?}. Other: {:?}", self.to_fen(), other.to_fen());
            }
        }

        self.zobrist_hash == other.zobrist_hash
            && self.state.buffer == other.state.buffer
            && self.state.phase == other.state.phase
            && self.state.castle == other.state.castle
            && self.state.en_passant == other.state.en_passant
    }
}

impl Eq for TriHexChess {}

impl Default for TriHexChess {
    fn default() -> Self {
        let default_fen = "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1";

        TriHexChess::new_with_fen(default_fen.as_bytes(), false).unwrap()
    }
}

impl TriHexChess {
    pub fn get_turn(&self) -> Option<Color> {
        if let Phase::Normal(player) = self.state.phase {
            Some(player.get_turn())
        } else {
            None
        }
    }

    pub fn is_over(&self) -> bool {
        matches!(self.state.phase, Phase::Won(_) | Phase::Draw(_))
    }

    pub fn get_phase(&self) -> Phase {
        self.state.phase
    }

    pub fn set_phase(&mut self, phase: Phase) {
        self.state.phase = phase;
    }

    pub fn default_with_grace_period() -> Self {
        let default_fen = "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1";

        TriHexChess::new_with_fen(default_fen.as_bytes(), true).unwrap()
    }

    pub fn new_with_fen(fen: &[u8], is_using_grace_period: bool) -> Result<Self, String> {
        let mut game = TriHexChess {
            state: State::default(),
            zobrist_hash: 0,
            is_using_grace_period,
            repetition_history: RepetitionHistory::default(),
        };

        game.set_fen(fen)?;
        game.zobrist_hash = game.calculate_full_hash();

        game.repetition_history
            .record(game.state.third_move + 1, game.zobrist_hash);

        Ok(game)
    }

    /// Creates a Zobrist hash for the current game state, by:
    /// - XORing the turn hash
    /// - XORing the piece hashes for each piece on the board
    /// - XORing the castling rights for each player
    /// - XORing the en passant square for each player
    pub fn calculate_full_hash(&self) -> u64 {
        let mut hash = 0u64;

        hash ^= PRECOMPUTED_ZOBRIST.turn(self.state.phase.get_turn());

        for i in 0..96 {
            if let Some((color, piece)) = self.state.buffer[i].get() {
                hash ^= PRECOMPUTED_ZOBRIST.piece(MemoryPos(i as u8), color, piece);
            }
        }

        for color in COLORS {
            if self.state.castle.can_king_side(color) {
                hash ^= PRECOMPUTED_ZOBRIST.castling_right(color);
            }
            if self.state.castle.can_queen_side(color) {
                hash ^= PRECOMPUTED_ZOBRIST.castling_left(color);
            }
        }

        for color in COLORS {
            if let Some(ep_square) = self.state.en_passant.get(color) {
                hash ^= PRECOMPUTED_ZOBRIST.en_passant(ep_square);
            }
        }

        hash
    }

    pub fn to_fen(&self) -> String {
        to_fen(&self.state)
    }

    pub fn set_fen(&mut self, fen: &[u8]) -> Result<(), String> {
        self.state.update_from_fen(fen)?;

        Ok(())
    }

    pub fn set_turn(&mut self, turn: Color) {
        if let Phase::Normal(player) = &mut self.state.phase {
            player.set_turn(turn);
        } else {
            panic!("Cannot set turn when phase is not normal");
        }

        self.zobrist_hash = self.calculate_full_hash();
    }

    /// Updates the pseudo moves for the current turn.
    /// Returns the number of pseudo moves generated.
    pub fn update_pseudo_moves(
        &mut self,
        move_store: &mut ChessMoveStore,
        all_promotions: bool,
    ) -> u8 {
        // Should only be called on 'normal' phases
        debug_assert!(
            matches!(self.state.phase, Phase::Normal(_)),
            "Phase should be normal when updating pseudo moves"
        );

        move_store.clear();

        let turn = match self.state.phase {
            Phase::Normal(player) => player.get_turn(),
            _ => return 0,
        };

        let offset = turn.get_offset();

        // Update current turn state
        let king_pos = self.state.buffer.king_qrs_local(turn);
        let is_check = king_pos.is_some_and(|king_pos| self.is_check(king_pos, offset, turn));

        move_store.turn_cache = TurnCache {
            turn,
            king_pos,
            is_check,
            grace_period: if self.is_using_grace_period {
                self.state.turn_counter == 1 && (turn == Color::White || turn == Color::Gray)
                // Only use grace period on the first turn of the game (we start count at 1)
                // and only for white and gray so they can't abuse their extra turn to attack black
            } else {
                false
            },
        };

        // For each piece of the current player generate moves
        for i in 0u8..96 {
            let slot = self.state.buffer[i as usize];

            if slot.is_empty() {
                continue;
            }

            if slot.is_enemy(move_store.turn_cache.turn) {
                continue;
            }

            let pos = MemoryPos(i);
            let local_qrs = pos.to_qrs_local(move_store.turn_cache.turn);

            match slot.piece().unwrap() {
                Piece::Pawn => {
                    self.generate_pawn_moves(pos, local_qrs, offset, move_store, all_promotions)
                }
                Piece::Knight => self.generate_knight_moves(pos, local_qrs, offset, move_store),
                Piece::Bishop => {
                    self.generate_sliding_moves(pos, local_qrs, offset, &DIAGONAL_V, move_store)
                }
                Piece::Rook => {
                    self.generate_sliding_moves(pos, local_qrs, offset, &LINE_V, move_store)
                }
                Piece::Queen => {
                    self.generate_sliding_moves(pos, local_qrs, offset, &ALL_DIR_V, move_store)
                }
                Piece::King => self.generate_king_moves(pos, local_qrs, offset, move_store),
            }
        }

        move_store.len() as u8
    }

    fn generate_pawn_moves(
        &mut self,
        from_i: MemoryPos,
        local_qrs: QRSPos,
        offset: u8,
        move_store: &mut ChessMoveStore,
        all_promotions: bool,
    ) {
        // Move forward
        let qrs = local_qrs.add(0, 1, -1);

        // If we can't even move forward, return
        if !qrs.is_in() {
            return;
        }

        let mut to_i = qrs.to_pos().to_global(offset);

        if self.state.buffer[to_i].is_empty() {
            self.add_pawn_move(from_i, to_i, qrs, offset, None, move_store, all_promotions);

            // Move two squares forward
            if from_i.is_original_pawn(move_store.turn_cache.turn) {
                to_i.0 += 8;

                if self.state.buffer[to_i].is_empty() {
                    self.test_and_add_move(
                        PseudoLegalMove {
                            from: from_i,
                            to: to_i,
                            move_type: MoveType::DoublePawnPush,
                        },
                        offset,
                        false,
                        move_store,
                    );
                }
            }
        }
        // If the piece in front of the pawn is an enemy pawn,
        // check if we can capture it (en passant)
        else if self.state.buffer[to_i].is_enemy(move_store.turn_cache.turn)
            && self.state.buffer[to_i].piece() == Some(Piece::Pawn)
        {
            let left_player = move_store.turn_cache.turn.left_player();

            if let Some(behind_enemy_pawn_pos) = self.state.en_passant.get(left_player) {
                if self.state.buffer[behind_enemy_pawn_pos].is_empty() {
                    let qrs_attack = local_qrs.add(L_P_ATTACK_V.0, L_P_ATTACK_V.1, L_P_ATTACK_V.2);

                    if qrs_attack.is_in() {
                        let att_i = qrs_attack.to_pos().to_global(offset);

                        if behind_enemy_pawn_pos == att_i {
                            self.add_pawn_move(
                                from_i,
                                att_i,
                                qrs_attack,
                                offset,
                                Some(to_i),
                                move_store,
                                all_promotions,
                            );
                        }
                    }
                }
            }

            let right_player = move_store.turn_cache.turn.right_player();

            if let Some(behind_enemy_pawn_pos) = self.state.en_passant.get(right_player) {
                if self.state.buffer[behind_enemy_pawn_pos].is_empty() {
                    let qrs_attack = local_qrs.add(R_P_ATTACK_V.0, R_P_ATTACK_V.1, R_P_ATTACK_V.2);

                    if qrs_attack.is_in() {
                        let att_i = qrs_attack.to_pos().to_global(offset);

                        if behind_enemy_pawn_pos == att_i {
                            self.add_pawn_move(
                                from_i,
                                att_i,
                                qrs_attack,
                                offset,
                                Some(to_i),
                                move_store,
                                all_promotions,
                            );
                        }
                    }
                }
            }
        }

        // Normal attacks
        for (q, r, s) in P_ATTACK {
            let qrs = local_qrs.add(q, r, s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_enemy(move_store.turn_cache.turn) {
                    self.add_pawn_move(from_i, to_i, qrs, offset, None, move_store, all_promotions);
                }
            }
        }
    }

    fn add_pawn_move(
        &mut self,
        from: MemoryPos,
        to: MemoryPos,
        to_qrs: QRSPos,
        offset: u8,
        en_passant: Option<MemoryPos>,
        move_store: &mut ChessMoveStore,
        all_promotions: bool,
    ) {
        if to_qrs.is_promotion() {
            if !self.test_move(
                PseudoLegalMove {
                    from,
                    to,
                    move_type: match en_passant {
                        Some(pos) => MoveType::EnPassant(pos),
                        None => MoveType::Move,
                    },
                },
                offset,
                move_store,
            ) {
                return;
            }

            if all_promotions {
                for piece in PROM_P {
                    move_store.push(PseudoLegalMove {
                        from,
                        to,
                        move_type: match en_passant {
                            Some(pos) => {
                                MoveType::EnPassantPromotion(PassantWithPromotion::new(pos, piece))
                            }
                            None => MoveType::Promotion(piece),
                        },
                    });
                }
            } else {
                move_store.push(PseudoLegalMove {
                    from,
                    to,
                    move_type: match en_passant {
                        Some(pos) => MoveType::EnPassantPromotion(PassantWithPromotion::new(
                            pos,
                            Piece::Queen,
                        )),
                        None => MoveType::Promotion(Piece::Queen),
                    },
                });
            }
        } else {
            self.test_and_add_move(
                PseudoLegalMove {
                    from,
                    to,
                    move_type: match en_passant {
                        Some(pos) => MoveType::EnPassant(pos),
                        None => MoveType::Move,
                    },
                },
                offset,
                false,
                move_store,
            );
        }
    }

    fn generate_knight_moves(
        &mut self,
        from: MemoryPos,
        local_qrs: QRSPos,
        offset: u8,
        move_store: &mut ChessMoveStore,
    ) {
        // todo: optimizations :)
        // let mut can_move_freely = false;

        for (q, r, s) in KNIGHT_V {
            let qrs = local_qrs.add(q, r, s);

            if qrs.is_in() {
                let to = qrs.to_pos().to_global(offset);

                // If the square is empty or an enemy, we can move there
                if !self.state.buffer[to].is_player(move_store.turn_cache.turn) {
                    self.test_and_add_move(
                        PseudoLegalMove {
                            from,
                            to,
                            move_type: MoveType::Move,
                        },
                        offset,
                        false,
                        move_store,
                    );
                    // todo: optimizations :)
                    // We know, that if we moved the knight anywhere, and it was legal,
                    // then it must mean it wasn't pinned, so we don't need to check
                    // if other moves are legal.
                    // However, we can't just early return, since we need to check if the knight
                    // can capture another piece that is attacking the king (if in check).
                    // can_move_freely = move_was_legal;
                }
            }
        }
    }

    /// For bishops, rooks and queens
    #[inline]
    fn generate_sliding_moves(
        &mut self,
        from_i: MemoryPos,
        local_qrs: QRSPos,
        offset: u8,
        directions: &[(i8, i8, i8)],
        move_store: &mut ChessMoveStore,
    ) {
        for (q, r, s) in directions {
            let mut qrs = local_qrs.add(*q, *r, *s);
            // let is_line_pinned = false;

            'inner: while qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_player(move_store.turn_cache.turn) {
                    break 'inner;
                }

                self.test_and_add_move(
                    PseudoLegalMove {
                        from: from_i,
                        to: to_i,
                        move_type: MoveType::Move,
                    },
                    offset,
                    false,
                    move_store,
                );

                if self.state.buffer[to_i].is_enemy(move_store.turn_cache.turn) {
                    break 'inner;
                }

                qrs = qrs.add(*q, *r, *s);
            }
        }
    }

    /// The kings can either move in all directions or castle
    fn generate_king_moves(
        &mut self,
        from_i: MemoryPos,
        local_qrs: QRSPos,
        offset: u8,
        move_store: &mut ChessMoveStore,
    ) {
        for (q, r, s) in ALL_DIR_V {
            let qrs = local_qrs.add(q, r, s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                // If it's either empty or an enemy piece, we can move there
                if !self.state.buffer[to_i].is_player(move_store.turn_cache.turn) {
                    self.test_and_add_move(
                        PseudoLegalMove {
                            from: from_i,
                            to: to_i,
                            move_type: MoveType::Move,
                        },
                        offset,
                        false,
                        move_store,
                    );
                }
            }
        }

        // Can't castle if the king is in check
        if move_store.turn_cache.is_check {
            return;
        }

        let can_o_o = self.state.castle.can_king_side(move_store.turn_cache.turn);
        let can_o_o_o = self.state.castle.can_queen_side(move_store.turn_cache.turn);

        // If the player can't castle (either because he or the rooks moved, return)
        if !can_o_o && !can_o_o_o {
            return;
        }

        // The king always starts at positions 4, 36, or 68
        let og_king_slot = self.state.buffer[offset as usize + 4];

        // If the king is not in the original position, return
        if !og_king_slot.matches(move_store.turn_cache.turn, Piece::King) {
            return;
        }

        if can_o_o
            && self.state.buffer[offset as usize + 5].is_empty()
            && self.state.buffer[offset as usize + 6].is_empty()
            && self.state.buffer[offset as usize + 7]
                .matches(move_store.turn_cache.turn, Piece::Rook)
        {
            self.test_and_add_move(
                PseudoLegalMove {
                    from: from_i,
                    to: from_i.add(2),
                    move_type: MoveType::CastleKingSide,
                },
                offset,
                false,
                move_store,
            );
        }

        if can_o_o_o
            && self.state.buffer[offset as usize + 3].is_empty()
            && self.state.buffer[offset as usize + 2].is_empty()
            && self.state.buffer[offset as usize + 1].is_empty()
            && self.state.buffer[offset as usize].matches(move_store.turn_cache.turn, Piece::Rook)
        {
            self.test_and_add_move(
                PseudoLegalMove {
                    from: from_i,
                    to: from_i.add(-2),
                    move_type: MoveType::CastleQueenSide,
                },
                offset,
                false,
                move_store,
            );
        }
    }

    /// If the move is valid, add it to the move store and return true.
    /// todo: docs
    fn test_and_add_move(
        &mut self,
        pseudo_move: PseudoLegalMove,
        offset: u8,
        can_move_freely: bool,
        move_store: &mut ChessMoveStore,
    ) -> bool {
        if move_store.turn_cache.grace_period {
            if self.state.buffer[pseudo_move.get_capture_slot()]
                .is_enemy(move_store.turn_cache.turn)
            {
                return false;
            }
        }

        if (can_move_freely && !move_store.turn_cache.is_check)
            || self.test_move(pseudo_move, offset, move_store)
        {
            move_store.push(pseudo_move);
            return true;
        }

        false
    }

    /// Checks if the move can be made (the king is not in check)
    /// True if the move is valid, false otherwise
    fn test_move(
        &mut self,
        pseudo_move: PseudoLegalMove,
        offset: u8,
        move_store: &ChessMoveStore,
    ) -> bool {
        if move_store.turn_cache.king_pos.is_none() {
            return true; // Early return if king_pos is None
        }

        // If the move is made by the king, don't use the cached king position in the turn state
        let king_pos = match self.state.buffer[pseudo_move.from].piece() {
            Some(Piece::King) => pseudo_move.to.to_qrs_local(move_store.turn_cache.turn),
            _ => move_store.turn_cache.king_pos.unwrap(),
        };

        let capture_slot = self.state.buffer[pseudo_move.get_capture_slot()];

        #[cfg(debug_assertions)]
        let debug_buffer = self.state.buffer.clone();

        // Make the move
        self.make_test_move(pseudo_move);

        // Check if the king is in check
        let is_legal = !self.is_check(king_pos, offset, move_store.turn_cache.turn);

        // Restore the board
        self.un_make_test_move(pseudo_move, capture_slot);

        #[cfg(debug_assertions)]
        {
            debug_assert!(
                self.state.buffer == debug_buffer,
                "Buffer was corrupted after un-making test move"
            );
        }

        // If the move is a castle, we need to check if the king is in check in the middle of the castle
        if is_legal
            && matches!(
                pseudo_move.move_type,
                MoveType::CastleKingSide | MoveType::CastleQueenSide
            )
        {
            let middle_mem_pos = match pseudo_move.move_type {
                MoveType::CastleKingSide => pseudo_move.from.add(1),
                MoveType::CastleQueenSide => pseudo_move.from.add(-1),
                _ => unreachable!(),
            };

            // Set the king in the middle of the castle
            self.state.buffer[middle_mem_pos] = self.state.buffer[pseudo_move.from];
            self.state.buffer[pseudo_move.from] = MemorySlot::empty();

            // Check if the king is in check
            let king_qrs = middle_mem_pos.to_qrs_local(move_store.turn_cache.turn);
            let is_legal = !self.is_check(king_qrs, offset, move_store.turn_cache.turn);

            // Restore the king
            self.state.buffer[pseudo_move.from] = self.state.buffer[middle_mem_pos];
            self.state.buffer[middle_mem_pos] = MemorySlot::empty();

            if !is_legal {
                return false;
            }
        }

        is_legal
    }

    /// Makes an 'optimized' version of the make move function. This function is meant only
    /// for checking if king is in check after a move.
    /// - For promotions we don't care which piece is promoted to (just leave it as a pawn)
    /// - For castles we don't care about moving the rooks, since they wouldn't change
    ///   if the king is in check
    fn make_test_move(&mut self, pseudo_legal: PseudoLegalMove) {
        if let Some(passant_mem_pos) = match pseudo_legal.move_type {
            MoveType::EnPassant(pos) => Some(pos),
            MoveType::EnPassantPromotion(promotion) => Some(promotion.get().0),
            _ => None,
        } {
            debug_assert!(
                self.state.buffer[passant_mem_pos].piece() == Some(Piece::Pawn),
                "En passant slot is not a pawn"
            );
            debug_assert!(
                self.state.buffer[pseudo_legal.to].is_empty(),
                "Move slot for en passant is not empty"
            );

            self.state.buffer[passant_mem_pos] = MemorySlot::empty();
        }

        self.state.buffer[pseudo_legal.to] = self.state.buffer[pseudo_legal.from];
        self.state.buffer[pseudo_legal.from] = MemorySlot::empty();
    }

    /// Unmakes the [`TriHexChess::make_test_move`] function
    fn un_make_test_move(&mut self, pseudo_legal: PseudoLegalMove, captured_slot: MemorySlot) {
        match pseudo_legal.move_type {
            MoveType::EnPassant(passant_mem_pos) => {
                self.state.buffer[pseudo_legal.from] = self.state.buffer[pseudo_legal.to];
                self.state.buffer[pseudo_legal.to] = MemorySlot::empty();
                self.state.buffer[passant_mem_pos] = captured_slot;
            }
            MoveType::EnPassantPromotion(passant_promotion) => {
                let (passant_mem_pos, _) = passant_promotion.get();
                self.state.buffer[pseudo_legal.from] = self.state.buffer[pseudo_legal.to];
                self.state.buffer[pseudo_legal.to] = MemorySlot::empty();
                self.state.buffer[passant_mem_pos] = captured_slot;
            }
            _ => {
                self.state.buffer[pseudo_legal.from] = self.state.buffer[pseudo_legal.to];
                self.state.buffer[pseudo_legal.to] = captured_slot;
            }
        }
    }

    /// Returns the number of times the current position has occurred in the game history
    /// since the last irreversible move.
    pub fn get_repetition_count(&self) -> u8 {
        self.repetition_history
            .count_occurrences(self.state.third_move + 1, self.zobrist_hash)
    }

    pub fn commit_move(&mut self, m: &PseudoLegalMove, move_store: &ChessMoveStore) {
        // debug_assert!(!move_store.is_empty(), "Move store should not be empty");
        // debug_assert!(move_store.contains(*m), "Move not in move store");

        let turn = self.state.phase.get_turn();
        let to = m.to;
        let from = m.from;

        // debug_assert!(
        //     self.test_move(*m, turn.get_offset(), move_store),
        //     "Move is not legal"
        // );

        // If a piece is captured, remove it from the hash.
        if let Some((captured_player, captured_piece)) = self.state.buffer[to].get() {
            self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, captured_player, captured_piece);
        }

        // 'third move counter' resets to zero after a capture or a pawn move and incremented otherwise
        let mut increment_third_move = self.state.buffer[to].piece().is_none();

        // Captured pieces side effects
        match self.state.buffer[to].piece() {
            // Check if the move takes a pawn that can be captured en passant
            // If so, remove the information about the en passant square from the state
            Some(Piece::Pawn) => {
                for color in COLORS {
                    if self.state.en_passant.get(color) == Some(to) {
                        // Remove the en passant square from the zobrist hash
                        self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.en_passant(to);

                        self.state.en_passant.remove(color);
                        break; // A square can only be an en passant target for one player at a time.
                    }
                }
            }
            // Remove the ability to castle with the rook that was captured
            Some(Piece::Rook) => {
                let player = self.state.buffer[to].player().unwrap();
                // Only XOR out castling rights if they were previously available.
                if to.is_queen_rook_og() && self.state.castle.can_queen_side(player) {
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_left(player);
                    self.state.castle.set_queen_side(player, false);
                } else if to.is_king_rook_og() && self.state.castle.can_king_side(player) {
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_right(player);
                    self.state.castle.set_king_side(player, false);
                }
            }
            // If the king is captured, the game is over for that player
            Some(Piece::King) => {
                let color = self.state.buffer[to].player().unwrap();

                self.state.buffer.remove_player_pieces(color);
                self.state.en_passant.remove(color);
                self.state.castle.remove_all(color);

                if let Phase::Normal(state) = &mut self.state.phase {
                    state.remove_player(color);
                }

                // It's easier to just recalculate the zobrist hash than to keep track of all the changes
                self.zobrist_hash = self.calculate_full_hash();
            }
            _ => {}
        }

        // Moved pieces side effects
        match self.state.buffer[from].piece() {
            // If the king is moved, the player can't castle anymore
            Some(Piece::King) => {
                // Only XOR out castling rights if they were previously available.
                if self.state.castle.can_queen_side(turn) {
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_left(turn);
                }
                if self.state.castle.can_king_side(turn) {
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_right(turn);
                }

                self.state.castle.remove_all(turn);
            }
            // If a rook is moved, the player can't castle with that rook anymore
            Some(Piece::Rook) => {
                if from.is_queen_rook_og() && self.state.castle.can_queen_side(turn) {
                    self.state.castle.set_queen_side(turn, false);
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_left(turn);
                } else if from.is_king_rook_og() && self.state.castle.can_king_side(turn) {
                    self.state.castle.set_king_side(turn, false);
                    self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.castling_right(turn);
                }
            }
            Some(Piece::Pawn) => increment_third_move = false,
            _ => {}
        }

        // Remove the 'from' piece from the zobrist hash since it is no longer there
        let (from_player, from_piece) = self.state.buffer[from].get().unwrap_or_else(|| {
            panic!(
                "Move from empty square: {:?} (buffer={:?}). Move played: {:?}. Move store: {:?}. Fen: {}",
                from, self.state.buffer, m, move_store,
                self.to_fen()
            )
        });

        self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(from, from_player, from_piece);

        // Move the piece and clear the old position
        self.state.buffer[to] = self.state.buffer[from];
        self.state.buffer[from] = MemorySlot::empty();

        // Update the zobrist hash with the new piece
        let (to_player, to_piece) = self.state.buffer[to].get().unwrap();
        self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, to_player, to_piece);

        // Remove the en passant square from the zobrist hash if it exists
        if let Some(ep_square) = self.state.en_passant.get(turn) {
            self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.en_passant(ep_square);
        }

        // Clear current en passant square
        self.state.en_passant.remove(turn);

        if increment_third_move {
            self.state.third_move += 1;
        } else {
            self.state.third_move = 0;
        }

        // Variant based side effects
        match m.move_type {
            // Set the en passant square
            MoveType::DoublePawnPush => {
                let ep_square = to.add(-8);
                self.state.en_passant.set_pos(turn, ep_square);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.en_passant(ep_square);
            }
            // Overwrite the pawn with the promotion piece
            // Also need to correct the zobrist hash
            MoveType::Promotion(promotion) => {
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, to_player, Piece::Pawn);
                self.state.buffer[to].set_piece(promotion);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, to_player, promotion);
            }
            // 'Eat' the pawn that was captured en passant
            MoveType::EnPassant(en_passant) => {
                let en_passant_player = self.state.buffer[en_passant].player().unwrap();
                self.zobrist_hash ^=
                    PRECOMPUTED_ZOBRIST.piece(en_passant, en_passant_player, Piece::Pawn);

                self.state.buffer[en_passant] = MemorySlot::empty();
            }
            // Combines `MoveType::EnPassant` and `MoveType::Promotion` side effects
            // Rare move, happens only on few specific squares
            MoveType::EnPassantPromotion(wrapper) => {
                let (en_passant, promotion) = wrapper.get();

                let en_passant_player = self.state.buffer[en_passant].player().unwrap();

                self.state.buffer[en_passant] = MemorySlot::empty();
                self.state.buffer[to].set_piece(promotion);

                // Remove the en passant pawn from the zobrist hash
                self.zobrist_hash ^=
                    PRECOMPUTED_ZOBRIST.piece(en_passant, en_passant_player, Piece::Pawn);
                // Remove the pawn that was promoted from the zobrist hash and add the promotion piece
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, to_player, Piece::Pawn);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(to, to_player, promotion);
            }
            // We don't need to update the can castle flags here,
            // since we already did that before when checking if the king moved
            // We do however need to update the zobrist hash, specifically for the Rooks,
            // since king is already handled in 'from-to' zobrist hash
            MoveType::CastleKingSide | MoveType::CastleQueenSide => {
                let (rook_from, rook_to) = match m.move_type {
                    MoveType::CastleKingSide => (to.add(1), to.add(-1)),
                    MoveType::CastleQueenSide => (to.add(-2), to.add(1)),
                    _ => unreachable!(),
                };

                debug_assert!(self.state.buffer[rook_from].piece() == Some(Piece::Rook));
                debug_assert!(self.state.buffer[rook_to].is_empty());

                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(rook_from, turn, Piece::Rook);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.piece(rook_to, turn, Piece::Rook);

                self.state.buffer[rook_to] = self.state.buffer[rook_from];
                self.state.buffer[rook_from] = MemorySlot::empty();
            }
            // Only ::Move is left, but that doesn't have any side effects
            _ => {}
        }
    }

    pub fn next_turn_unsafe(&mut self, next_turn: Color) {
        self.state.phase = match &self.state.phase {
            Phase::Normal(player_state) => {
                let current_turn = player_state.get_turn();
                let mut new_player_state = *player_state;

                let (should_advance, next_turn_actual) = new_player_state.get_next_turn();

                debug_assert!(
                    next_turn == next_turn_actual,
                    "Next turn does not match the actual next turn"
                );

                new_player_state.clear_all_stale();
                new_player_state.set_turn(next_turn);

                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.turn(current_turn);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.turn(next_turn);

                if should_advance {
                    self.state.turn_counter += 1;
                }

                Phase::Normal(new_player_state)
            }
            _ => panic!("Game is over, can't advance turn"),
        };
    }

    /// Advances the turn to the next player and updates the pseudo moves.
    /// If `advance_turn` is false, it only updates the pseudo moves without advancing the turn.
    /// Some old note:
    // 8/8/ppp5/3p4/8/8/3rkr2/8/X X/4k3/8/8/8/X X/X/4k3/8/8/8 G ------ --- 0 1
    pub fn next_turn(
        &mut self,
        advance_turn: bool,
        move_store: &mut ChessMoveStore,
        all_promotions: bool,
    ) {
        debug_assert!(
            !self.is_over(),
            "Game is over, but we are trying to advance the turn"
        );

        if !advance_turn {
            self.update_pseudo_moves(move_store, all_promotions);

            return;
        }

        debug_assert!(
            {
                // We check if the fen and game state before and after the move are the same
                // This is nice to have, because sometimes `update_pseudo_moves` can 'corrupt'
                // the game state when un-making test moves
                let mut cloned_game = self.clone();
                let mut cloned_move_store = move_store.clone();

                let prev_fen = cloned_game.to_fen();

                cloned_game.update_pseudo_moves(&mut cloned_move_store, all_promotions);

                let next_fen = cloned_game.to_fen();

                prev_fen == next_fen
            },
            "Next move generation at fen: {} failed. Prev. and next fens are NOT the same",
            self.to_fen()
        );

        if let Phase::Normal(player_state) = self.state.phase {
            // ZOBRIST: Get the turn before any changes are made.
            let prev_turn = player_state.get_turn();
            let mut new_player_state = player_state;

            new_player_state.clear_all_stale();

            loop {
                let (should_advance, next_turn) = new_player_state.get_next_turn();

                if should_advance {
                    self.state.turn_counter += 1;
                }

                if next_turn == prev_turn {
                    if new_player_state.get_player_count() == 1 {
                        self.state.phase = Phase::Won(next_turn);
                        return;
                    } else if new_player_state.are_other_players_stale(next_turn) {
                        self.state.phase = Phase::Draw(player_state.to_stalemate(next_turn));
                        return;
                    }
                }

                new_player_state.set_turn(next_turn);

                if new_player_state.are_other_players_stale(next_turn) {
                    self.state.phase = Phase::Draw(player_state.to_stalemate(next_turn));
                    return;
                }

                self.state.phase = Phase::Normal(new_player_state);

                self.update_pseudo_moves(move_store, all_promotions);

                // If at least one move is possible, break the loop
                if !move_store.is_empty() {
                    break;
                }

                // If the player can't move, but is in check, he lost
                // Don't break the loop, since it's next player's turn after this
                if move_store.turn_cache.is_check {
                    self.state.buffer.remove_player_pieces(next_turn);
                    new_player_state.remove_player(next_turn);
                    self.state.en_passant.remove(next_turn);
                    self.state.castle.remove_all(next_turn);
                    // Removing all pieces might drastically change the board,
                    // so we need to clear all stale flags and re-do the loop
                    // Weird examples such as:
                    // 8/8/4p3/k6p/8/8/4p3/8/4r3/8/5p2/4p3 X/8/8/8/p7/7k/8/8/8 X/3p4/8/8/8/5k2/8/8/8 W ------ --- 0 1
                    new_player_state.clear_all_stale();

                    // Same goes for the zobrist hash
                    self.zobrist_hash = self.calculate_full_hash();
                }
                // If the player can't move, but is not in check, he
                // gets marked as stale
                else {
                    // If there are only two players left, the game is a stalemate
                    if new_player_state.get_player_count() <= 2 {
                        self.state.phase = Phase::Draw(player_state.to_stalemate(next_turn));
                        return;
                    }

                    new_player_state.set_stale(next_turn);
                }
            }

            self.state.phase = Phase::Normal(new_player_state);

            // Zobrist update
            let final_turn = new_player_state.get_turn();
            if prev_turn != final_turn {
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.turn(prev_turn);
                self.zobrist_hash ^= PRECOMPUTED_ZOBRIST.turn(final_turn);
            }

            self.repetition_history
                .record(self.state.third_move + 1, self.zobrist_hash);

            if self.state.third_move >= 100 {
                self.state.phase = Phase::Draw(new_player_state.to_stalemate(final_turn));
                move_store.clear(); // Game is over, no more moves.
                return;
            }

            if self.get_repetition_count() >= 3 {
                self.state.phase = Phase::Draw(new_player_state.to_stalemate(final_turn));
                move_store.clear(); // Game is over.
                return;
            }
        }
    }

    fn is_check_sliding(
        &self,
        king_local_qrs: QRSPos,
        offset: u8,
        array: &[(i8, i8, i8)],
        other_piece: Piece,
        turn: Color,
    ) -> bool {
        for (q, r, s) in array {
            let mut qrs = king_local_qrs.add(*q, *r, *s);
            'inner: while qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_empty() {
                    qrs.set(qrs.q + q, qrs.r + r, qrs.s + s);
                    continue;
                }

                if self.state.buffer[to_i].is_player(turn) {
                    break 'inner;
                }

                // The piece can only be an enemy piece
                let piece = self.state.buffer[to_i].piece().unwrap();

                if piece == other_piece || piece == Piece::Queen {
                    return true;
                }

                break 'inner;
            }
        }

        false
    }

    fn is_check(&self, king_l_qrs: QRSPos, offset: u8, turn: Color) -> bool {
        debug_assert!(king_l_qrs.is_in(), "King is not in bounds");
        debug_assert!(
            self.state.buffer.king_mem_pos(turn) == Some(king_l_qrs.to_pos().to_global(offset)),
            "King is not in the correct position. {:?} != {:?}",
            self.state.buffer.king_mem_pos(turn),
            king_l_qrs.to_pos().to_global(offset)
        );

        // Bishop and queen
        if self.is_check_sliding(king_l_qrs, offset, &DIAGONAL_V, Piece::Bishop, turn) {
            return true;
        }

        // Rook and queen
        if self.is_check_sliding(king_l_qrs, offset, &LINE_V, Piece::Rook, turn) {
            return true;
        }

        let mut qrs = QRSPos::default();

        // Knight
        for (q, r, s) in KNIGHT_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_enemy(turn)
                    && self.state.buffer[to_i].piece().unwrap() == Piece::Knight
                {
                    return true;
                }
            }
        }

        // King
        // I think we only need to check this if the king moved? (some faster version?)
        for (q, r, s) in ALL_DIR_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].is_enemy(turn)
                    && self.state.buffer[to_i].piece().unwrap() == Piece::King
                {
                    return true;
                }
            }
        }

        // I think we only need to check this if the king moved?
        let left_player = turn.left_player();
        let right_player = turn.right_player();

        for (q, r, s) in L_KING_P_ATTACK_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].matches(left_player, Piece::Pawn) {
                    return true;
                }
            }
        }

        for (q, r, s) in R_KING_P_ATTACK_V {
            qrs.set(king_l_qrs.q + q, king_l_qrs.r + r, king_l_qrs.s + s);

            if qrs.is_in() {
                let to_i = qrs.to_pos().to_global(offset);

                if self.state.buffer[to_i].matches(right_player, Piece::Pawn) {
                    return true;
                }
            }
        }

        false
    }

    pub fn material_count(&self) -> [u8; 3] {
        let mut counts = [0u8; 3];

        for (_, slot) in self.state.buffer.non_empty_iter() {
            if let Some((player, piece)) = slot.get() {
                counts[player as usize] += piece.material();
            }
        }

        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rayon::prelude::*;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_queen_disappear() {
        // Test a weird position where the queen disappears after pseudo move generation
        // Diagnosis: It's fixed! The problem was that en passant was able to take the queen
        // (or any other piece)
        let fen = "rnb2bnr/p3pp2/1pkp4/7p/X/8/8/4q3/8 X/rnb1kbnr/1pp4p/p1q2pp1/3pp3/X 8/6b1/8/8/X/rnbqk1nr/p1p2p2/3p2pp/1p2p3 G --qkqk --5 0 1";
        let mut game = TriHexChess::new_with_fen(fen.as_bytes(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        game.update_pseudo_moves(&mut move_store, true);

        assert_eq!(game.to_fen(), fen);
    }

    #[test]
    fn test_move_gen() {
        let mut game = TriHexChess::default();

        let mut move_store = ChessMoveStore::default();

        game.update_pseudo_moves(&mut move_store, true);

        assert!(game.update_pseudo_moves(&mut move_store, true) > 0);

        println!(
            "Number of moves: {}",
            game.update_pseudo_moves(&mut move_store, true)
        );

        for m in move_store.iter() {
            println!("{}", m.notation_lan(&game.state.buffer));
        }
    }

    #[test]
    fn test_with_self_play() {
        let total_moves_made = Arc::new(AtomicUsize::new(0));
        let total_pseudo_moves_generated = Arc::new(AtomicUsize::new(0));
        let total_games_completed = Arc::new(AtomicUsize::new(0));

        let num_threads = rayon::current_num_threads() - 2;
        // When testing with 100k games it takes around 25 minutes.
        // Around 1.3 billion moves get generated
        let games = 10_000;

        let games_per_thread = (games + num_threads - 1) / num_threads;

        println!(
            "Starting test with {} threads, {} games per thread",
            num_threads, games_per_thread
        );

        let game_length_to_number_of_moves = Arc::new(Mutex::new(vec![0.0; 501]));

        // Spawn `num_threads` threads to process the games
        (0..num_threads).into_par_iter().for_each(|thread_id| {
            // let seed_str = format!("82345_{}", thread_id);
            let seed_str = format!("182345_{}", thread_id);
            let hash = Sha256::digest(seed_str.as_bytes());
            let seed: [u8; 32] = hash.into();
            let game_length_to_number_of_moves = vec![(0u64, 0u64); 501];

            let mut rng = StdRng::from_seed(seed);

            let mut local_moves_made = 0;
            let mut local_pseudo_moves_generated = 0;
            let mut local_games_completed = 0;

            // Each thread processes its assigned batch of games
            for game_id in 0..games_per_thread {
                let global_game_id = thread_id * games_per_thread + game_id;

                if global_game_id >= games {
                    break; // Don't exceed total desired games
                }

                let mut game = TriHexChess::default();
                let mut move_store = ChessMoveStore::default();

                let mut moves_in_game = 0;
                let mut pseudo_moves_in_game = 0;

                game.update_pseudo_moves(&mut move_store, true);

                'game_loop: loop {
                    // Count pseudo legal moves for this position
                    let length = move_store.len();
                    pseudo_moves_in_game += length;

                    if game.is_over() {
                        break 'game_loop;
                    }

                    let index = rng.random_range(0..length);
                    let pseudo_move = move_store.get(index).unwrap();

                    let pre_commit_fen = game.to_fen();

                    game.commit_move(&pseudo_move, &move_store);

                    let post_commit_number_of_players = match game.state.phase {
                        Phase::Normal(state) => state.get_player_count(),
                        _ => unreachable!()
                    };

                    let post_commit_fen = game.to_fen();
                    let post_commit_turn = game.get_turn().unwrap();

                    assert_ne!(pre_commit_fen, post_commit_fen, "Fen strings are the same");

                    game.next_turn(true, &mut move_store, true);
                    let post_next_turn_fen = game.to_fen();

                    assert!(
                        {
                            // If after eating the king the game is over, the fen strings should be the same,
                            // since advancing the turn should not change the fen string
                            if post_commit_number_of_players == 1 {
                                post_commit_fen == post_next_turn_fen
                            }
                            // If a stalemate is reached, the fen strings should be the same if the turn is the same
                            else if let Phase::Draw(stalemate_state) = game.state.phase {
                                if stalemate_state.get_turn() == post_commit_turn {
                                    post_commit_fen == post_next_turn_fen
                                } else {
                                    post_commit_fen != post_next_turn_fen
                                }
                            }
                            // Else the fen strings should be different, since we are advancing the turn
                            else {
                                post_commit_fen != post_next_turn_fen
                            }
                        },
                        "Failed fen: {} == {}. Pre move fen: {}. Phase: {:?}",
                        post_commit_fen, post_next_turn_fen, pre_commit_fen, game.state.phase
                    );

                    moves_in_game += 1;

                    if moves_in_game >= 500 {
                        // Cap at 1000 moves per game to prevent infinite games
                        break 'game_loop;
                    }
                }

                local_moves_made += moves_in_game;
                local_pseudo_moves_generated += pseudo_moves_in_game;
                local_games_completed += 1;

                // Periodically log progress
                if local_games_completed % 100 == 0 {
                    println!(
                        "Thread {}: Completed {} games, current totals - Moves: {}, Pseudo moves: {}",
                        thread_id, local_games_completed, local_moves_made, local_pseudo_moves_generated
                    );
                }
            }

            // Log thread completion
            println!(
                "Thread {} completed {} games: Total moves: {}, Total pseudo moves: {}, Avg moves per game: {:.2}, Avg branching factor: {:.2}",
                thread_id,
                local_games_completed,
                local_moves_made,
                local_pseudo_moves_generated,
                local_moves_made as f64 / local_games_completed as f64,
                local_pseudo_moves_generated as f64 / local_moves_made.max(1) as f64
            );

            // Update global counters with thread totals
            total_moves_made.fetch_add(local_moves_made, Ordering::Relaxed);
            total_pseudo_moves_generated.fetch_add(local_pseudo_moves_generated, Ordering::Relaxed);
            total_games_completed.fetch_add(local_games_completed, Ordering::Relaxed);
        });

        let games = total_games_completed.load(Ordering::Relaxed);
        let moves = total_moves_made.load(Ordering::Relaxed);
        let pseudo_moves = total_pseudo_moves_generated.load(Ordering::Relaxed);

        println!(
            "Games completed: {}. Moves made: {}. Pseudo moves generated: {}. Branching factor: {:.2}",
            games,
            moves,
            pseudo_moves,
            pseudo_moves as f64 / moves as f64
        );
    }

    #[test]
    fn test_en_passant() {
        let fen = "rnbqkbnr/pppppp1p/8/6p1/X/X X/rnbqkbnr/pp1ppppp/8/2p5/X X/X/rnbqkbnr/ppp1pppp/3p4/8 W qkqkqk 73- 0 2";

        let mut game = TriHexChess::new_with_fen(fen.as_bytes(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        game.update_pseudo_moves(&mut move_store, true);

        for m in move_store.iter() {
            println!("Testing move: {:?}", m);

            let mut cloned_game = game.clone();
            let mut cloned_move_store = move_store.clone();

            cloned_game.commit_move(m, &cloned_move_store);
            cloned_game.next_turn(true, &mut cloned_move_store, true);
        }
    }

    // pre move: 8/8/4p3/k6p/8/8/4p3/8/2r5/8/5p2/4p3 X/8/8/8/p7/7k/8/8/8 X/3p4/8/8/8/5k2/8/8/8 W ------ --- 0 1
    // post move: 8/8/4p3/k6p/8/8/4p3/8/4r3/8/5p2/4p3 X/8/8/8/p7/7k/8/8/8 X/3p4/8/8/8/5k2/8/8/8 W ------ --- 0 1
    // post turn: 8/8/4p3/k6p/8/8/4p3/8/4r3/8/5p2/4p3 X/8/8/8/p7/7k/8/8/8 X/X/X W ------ --- 0 1
    // should be Normal(White to play), is Stalemate(White) - FIXED. Still a super weird position
    #[test]
    fn test_weird_1() {
        let fen = "8/8/4p3/k6p/8/8/4p3/8/4r3/8/5p2/4p3 X/8/8/8/p7/7k/8/8/8 X/3p4/8/8/8/5k2/8/8/8 W ------ --- 0 1";
        let mut game = TriHexChess::new_with_fen(fen.as_bytes(), false).unwrap();

        let mut move_store = ChessMoveStore::default();

        game.next_turn(true, &mut move_store, true);

        assert!(matches!(game.state.phase, Phase::Normal(_)));
    }

    #[test]
    fn test_weird_2() {
        let fen = "rnbq1bnr/pp1ppppp/3k4/2p5/X/X X/rnbq1bnr/pppppp1p/5k2/6p1/X X/X/rnbqkbnr/p2ppppp/8/1pp5 W ----qk -73 0 3";
        let mut game = TriHexChess::new_with_fen(fen.as_bytes(), false).unwrap();
        let mut move_store = ChessMoveStore::default();

        game.update_pseudo_moves(&mut move_store, true);
        println!("Move store: {:?}", move_store);

        for m in move_store.iter() {
            println!("{}", m.notation_lan(&game.state.buffer));
        }
    }

    #[test]
    fn test_3_fold() {
        struct TestCase {
            fen: &'static str,
            moves: Vec<&'static str>,
            expected_count: u8,
        }

        let cases = vec![TestCase {
            fen: "X/X/X 8/4r3/3k2p1/8/8/8/8/5n2/X 3k1p2/8/8/8/X/8/8/8/2p5 G ------ --- 0 5",
            moves: vec!["Rh5-i5", "Kg4-f4", "Ri5-h5", "Kf4-g4", "Rh5-i5", "Kg4-f4", "Ri5-h5", "Kf4-g4"],

            expected_count: 3,
        }];

        for case in cases {
            let mut game = TriHexChess::new_with_fen(case.fen.as_bytes(), false).unwrap();
            let mut move_store = ChessMoveStore::default();

            game.update_pseudo_moves(&mut move_store, true);

            for &mv in &case.moves {
                let pseudo_move = move_store
                    .iter()
                    .find(|m| m.notation_lan(&game.state.buffer) == mv)
                    .expect(&format!("Move {} not found in move store", mv));

                game.commit_move(pseudo_move, &move_store);
                game.next_turn(true, &mut move_store, true);
            }

            let count = game.get_repetition_count();
            println!("Debug history: {:?}", game.repetition_history);

            assert_eq!(
                count, case.expected_count,
                "Failed on case with fen: {}",
                case.fen
            );
        }
    }
}
