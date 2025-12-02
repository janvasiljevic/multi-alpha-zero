use crate::mapping::{
    Board, BoardPlayer, InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper,
};
use colored::Colorize;
use game_tri_chess::basics::{
    CastleFlags, Color, EnPassantState, MemorySlot, Piece, PlayerState,
};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::pos::MemoryPos;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use crate::mapping::chess_canonical_mapper::ChessCanonicalMapper;

/// Features a mapping of:
///
/// [Local features - Per tile diff.]
/// - Is Me/Next/Prev Pawn (because they act like different pieces) (3 channels)
/// - Is Knight
/// - Is Bishop
/// - Is Rook
/// - Is Queen
/// - Is King
/// - Is Me/Next/Prev Owner of the associated Piece (3 channels)
/// - Is En Passant target square for Me/Next/Prev (3 channels)
///
/// [Global features - Same for all tiles]
/// - Can Castle King-side + Queen-side for Me/Next/Prev (2 * 3 = 6 channels)
/// - Is Player present (3 channels) - the first channel is always true.
/// - Is Grace Period (1 channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessHybridCanonicalMapper;

const NUM_OF_FIELDS: usize = 96;

// Local Features (14 channels)
const ME_PAWN_CH: usize = 0;
const NEXT_PAWN_CH: usize = 1;
const PREV_PAWN_CH: usize = 2;

const SYMMETRIC_PIECE_START_CH: usize = 3; // Knight=3, Bishop=4, Rook=5, Queen=6, King=7

const KING_CH: usize = 7;

const ME_OWNER_CH: usize = 8;
const NEXT_OWNER_CH: usize = 9;
const PREV_OWNER_CH: usize = 10;

// const EP_START_CH: usize = 11;
const ME_EP_CH: usize = 11;
const NEXT_EP_CH: usize = 12;
const PREV_EP_CH: usize = 13;

// Global Features
const CASTLING_START_CH: usize = 14;
const ME_CASTLE_Q_CH: usize = 14;
const ME_CASTLE_K_CH: usize = 15;
const NEXT_CASTLE_Q_CH: usize = 16;
const NEXT_CASTLE_K_CH: usize = 17;
const PREV_CASTLE_Q_CH: usize = 18;
const PREV_CASTLE_K_CH: usize = 19;

const PLAYER_PRESENCE_START_CH: usize = 20;
const ME_PLAYER_PRESENT_CH: usize = 20;
const NEXT_PLAYER_PRESENT_CH: usize = 21;
const PREV_PLAYER_PRESENT_CH: usize = 22;

const IS_GRACE_PERIOD_CH: usize = 23;

const TOTAL_CHANNELS: usize = 24;

impl PolicyMapper<TriHexChess> for ChessHybridCanonicalMapper {
    fn policy_len(&self) -> usize {
        NUM_OF_FIELDS * NUM_OF_FIELDS
    }

    fn move_to_index(&self, player: Color, mv: PseudoLegalMove) -> usize {
        ChessCanonicalMapper.move_to_index(player, mv)
    }

    fn index_to_move(
        &self,
        board: &TriHexChess,
        move_store: &ChessMoveStore,
        index: usize,
    ) -> Option<PseudoLegalMove> {
        ChessCanonicalMapper.index_to_move(board, move_store, index)
    }
}

impl InputMapper<TriHexChess> for ChessHybridCanonicalMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        [NUM_OF_FIELDS, TOTAL_CHANNELS]
    }

    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &TriHexChess) {
        input_view.fill(false);

        let me = board.player_current();
        let next = me.next();
        let prev = me.next().next();
        let me_offset = me.get_offset();

        for (i, field) in board.state.buffer.non_empty_iter() {
            let piece = field.piece().unwrap();
            let player = field.player().unwrap();
            let local_pos = i.to_local(me_offset).0 as usize;

            match piece {
                Piece::Pawn => {
                    // Pawn type is tied to the player.
                    if player == me {
                        input_view[[local_pos, ME_PAWN_CH]] = true;
                    } else if player == next {
                        input_view[[local_pos, NEXT_PAWN_CH]] = true;
                    } else {
                        input_view[[local_pos, PREV_PAWN_CH]] = true;
                    }
                }
                _ => {
                    // Symmetric Pieces: N, B, R, Q, K
                    let piece_type_channel = SYMMETRIC_PIECE_START_CH + (piece as u8 - 2) as usize;
                    input_view[[local_pos, piece_type_channel]] = true;

                    if player == me {
                        input_view[[local_pos, ME_OWNER_CH]] = true;
                    } else if player == next {
                        input_view[[local_pos, NEXT_OWNER_CH]] = true;
                    } else {
                        input_view[[local_pos, PREV_OWNER_CH]] = true;
                    }
                }
            }
        }

        let en_passant_state = board.state.en_passant;
        if let Some(pos) = en_passant_state.get(me) {
            input_view[[pos.to_local(me_offset).0 as usize, ME_EP_CH]] = true;
        }
        if let Some(pos) = en_passant_state.get(next) {
            input_view[[pos.to_local(me_offset).0 as usize, NEXT_EP_CH]] = true;
        }
        if let Some(pos) = en_passant_state.get(prev) {
            input_view[[pos.to_local(me_offset).0 as usize, PREV_EP_CH]] = true;
        }

        let player_state = board.get_phase();
        input_view
            .slice_mut(s![.., ME_PLAYER_PRESENT_CH])
            .fill(true);

        match player_state {
            Phase::Normal(state) => {
                input_view
                    .slice_mut(s![.., NEXT_PLAYER_PRESENT_CH])
                    .fill(state.is_present(next));
                input_view
                    .slice_mut(s![.., PREV_PLAYER_PRESENT_CH])
                    .fill(state.is_present(prev));
            }
            Phase::Won(_) => {}
            Phase::Draw(_) => {}
        }

        // Castling Rights
        let castle_flags = board.state.castle;
        input_view
            .slice_mut(s![.., ME_CASTLE_Q_CH])
            .fill(castle_flags.can_queen_side(me));
        input_view
            .slice_mut(s![.., ME_CASTLE_K_CH])
            .fill(castle_flags.can_king_side(me));
        input_view
            .slice_mut(s![.., NEXT_CASTLE_Q_CH])
            .fill(castle_flags.can_queen_side(next));
        input_view
            .slice_mut(s![.., NEXT_CASTLE_K_CH])
            .fill(castle_flags.can_king_side(next));
        input_view
            .slice_mut(s![.., PREV_CASTLE_Q_CH])
            .fill(castle_flags.can_queen_side(prev));
        input_view
            .slice_mut(s![.., PREV_CASTLE_K_CH])
            .fill(castle_flags.can_king_side(prev));

        // Grace Period
        let is_grace = board.is_using_grace_period && board.state.turn_counter < 3; // Or however you define it
        input_view
            .slice_mut(s![.., IS_GRACE_PERIOD_CH])
            .fill(is_grace);
    }

    fn is_absolute(&self) -> bool {
        false
    }
}

impl ReverseInputMapper<TriHexChess> for ChessHybridCanonicalMapper {
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, scalars: &Vec<f32>) -> TriHexChess {
        debug_assert_eq!(
            input_view.shape(),
            &[NUM_OF_FIELDS, TOTAL_CHANNELS],
            "Input view has incorrect dimensions."
        );
        debug_assert!(
            !scalars.is_empty(),
            "Player turn must be provided as a scalar."
        );

        let original_player =
            Color::from_u8(scalars[0] as u8).expect("Invalid player color provided in scalars.");

        let me_player = original_player;
        let next_player = me_player.next();
        let prev_player = me_player.next().next();

        let mut game = TriHexChess::default();
        game.state.buffer.clear();

        for canonical_pos in 0..NUM_OF_FIELDS {
            let absolute_pos =
                MemoryPos(canonical_pos as u8).to_global(original_player.get_offset());
            let mut piece_placed = false;

            // Check for asymmetric pieces (Pawns) first
            if input_view[[canonical_pos, ME_PAWN_CH]] {
                game.state.buffer[absolute_pos.0 as usize] =
                    MemorySlot::new(me_player, Piece::Pawn);
                piece_placed = true;
            } else if input_view[[canonical_pos, NEXT_PAWN_CH]] {
                game.state.buffer[absolute_pos.0 as usize] =
                    MemorySlot::new(next_player, Piece::Pawn);
                piece_placed = true;
            } else if input_view[[canonical_pos, PREV_PAWN_CH]] {
                game.state.buffer[absolute_pos.0 as usize] =
                    MemorySlot::new(prev_player, Piece::Pawn);
                piece_placed = true;
            }

            if piece_placed {
                continue;
            }

            // If no pawn, check for symmetric pieces
            let mut piece_type: Option<Piece> = None;
            for i in 0..5 {
                if input_view[[canonical_pos, SYMMETRIC_PIECE_START_CH + i]] {
                    // Assumes Piece enum starts with Pawn=1, Knight=2, ...
                    piece_type = Some(Piece::from_u8((i + 2) as u8).unwrap());
                    break;
                }
            }

            if let Some(piece) = piece_type {
                let mut owner: Option<Color> = None;
                if input_view[[canonical_pos, ME_OWNER_CH]] {
                    owner = Some(me_player);
                } else if input_view[[canonical_pos, NEXT_OWNER_CH]] {
                    owner = Some(next_player);
                } else if input_view[[canonical_pos, PREV_OWNER_CH]] {
                    owner = Some(prev_player);
                }

                if let Some(player) = owner {
                    game.state.buffer[absolute_pos.0 as usize] = MemorySlot::new(player, piece);
                }
            }
        }

        // Decode Castling Rights (read from the first field, as it's global)
        let mut castle_flags = CastleFlags::new(0);
        castle_flags.set_queen_side(me_player, input_view[[0, ME_CASTLE_Q_CH]]);
        castle_flags.set_king_side(me_player, input_view[[0, ME_CASTLE_K_CH]]);
        castle_flags.set_queen_side(next_player, input_view[[0, NEXT_CASTLE_Q_CH]]);
        castle_flags.set_king_side(next_player, input_view[[0, NEXT_CASTLE_K_CH]]);
        castle_flags.set_queen_side(prev_player, input_view[[0, PREV_CASTLE_Q_CH]]);
        castle_flags.set_king_side(prev_player, input_view[[0, PREV_CASTLE_K_CH]]);
        game.state.castle = castle_flags;

        // Decode En Passant
        let mut en_passant_state = EnPassantState::default();
        for canonical_pos in 0..NUM_OF_FIELDS {
            let absolute_pos =
                MemoryPos(canonical_pos as u8).to_global(original_player.get_offset());
            if input_view[[canonical_pos, ME_EP_CH]] {
                en_passant_state.set_pos(me_player, absolute_pos);
            }
            if input_view[[canonical_pos, NEXT_EP_CH]] {
                en_passant_state.set_pos(next_player, absolute_pos);
            }
            if input_view[[canonical_pos, PREV_EP_CH]] {
                en_passant_state.set_pos(prev_player, absolute_pos);
            }
        }
        game.state.en_passant = en_passant_state;


        game.is_using_grace_period = input_view[[0, IS_GRACE_PERIOD_CH]];


        game.state.third_move = 0;
        // Turn counter cannot be perfectly reconstructed, so we reset it.
        // If grace period is on, we can assume it's an early turn.
        game.state.turn_counter = if game.is_using_grace_period { 1 } else { 5 };

        let mut player_state = PlayerState::default();
        player_state.set_turn(original_player);

        if input_view[[0, ME_PLAYER_PRESENT_CH]] {
            player_state.set_player(me_player);
        }
        if input_view[[0, NEXT_PLAYER_PRESENT_CH]] {
            player_state.set_player(next_player);
        }
        if input_view[[0, PREV_PLAYER_PRESENT_CH]] {
            player_state.set_player(prev_player);
        }
        game.set_phase(Phase::Normal(player_state));

        // Finalize the game state
        game.zobrist_hash = game.calculate_full_hash();
        game
    }
}

impl MetaPerformanceMapper<TriHexChess> for ChessHybridCanonicalMapper {
    fn average_number_of_moves(&self) -> usize {
        37
    }
}

pub fn pretty_print_hybrid_chess_tensor(input: &Array2<bool>) -> String {
    let mut output = String::new();

    let channel_labels = vec![
        "mP ", "nP ", "pP ", // Asymmetric Pawns
        " N ", " B ", " R ", " Q ", " K ", // Symmetric Pieces
        "mO ", "nO ", "pO ", // Owners
        "mEP", "nEP", "pEP", // En Passant
        "mQc", "mKc", "nQc", "nKc", "pQc", "pKc", // Castling
        "mPr", "nPr", "pPr", // Player Presence
        " GP", // Grace Period
    ];

    assert_eq!(
        channel_labels.len(),
        TOTAL_CHANNELS,
        "Mismatch between labels and total channels"
    );

    for i in 0..TOTAL_CHANNELS {
        output.push_str(&format!("{: >3} ", channel_labels[i]));

        for j in 0..NUM_OF_FIELDS {
            if input[[j, i]] {
                let char = "■";

                let colored_string = match i {
                    // --- Asymmetric Pawns ---
                    ME_PAWN_CH => char.red(),
                    NEXT_PAWN_CH => char.blue(),
                    PREV_PAWN_CH => char.green(),

                    // For a symmetric piece, we check its corresponding owner channel on the same field to determine color.
                    c if c >= SYMMETRIC_PIECE_START_CH && c <= KING_CH => {
                        if input[[j, ME_OWNER_CH]] { char.red() }
                        else if input[[j, NEXT_OWNER_CH]] { char.blue() }
                        else if input[[j, PREV_OWNER_CH]] { char.green() }
                        else { char.white() } // Should not happen for a piece, but a safe fallback
                    },

                    ME_OWNER_CH | ME_EP_CH => char.red(),
                    NEXT_OWNER_CH | NEXT_EP_CH => char.blue(),
                    PREV_OWNER_CH | PREV_EP_CH => char.green(),
                    c if c >= CASTLING_START_CH && c <= PREV_CASTLE_K_CH => {
                        match (c - CASTLING_START_CH) / 2 { 0 => char.red(), 1 => char.blue(), 2 => char.green(), _ => char.normal() }
                    },
                    c if c >= PLAYER_PRESENCE_START_CH && c <= PREV_PLAYER_PRESENT_CH => {
                        match c - PLAYER_PRESENCE_START_CH { 0 => char.red(), 1 => char.blue(), 2 => char.green(), _ => char.normal() }
                    },

                    IS_GRACE_PERIOD_CH => char.cyan(),
                    _ => char.normal(),
                };
                output.push_str(&format!("{}", colored_string));
            } else {
                output.push_str("_");
            }
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_tri_chess::basics::{Color, COLORS};
    use game_tri_chess::chess_game::TriHexChess;
    use game_tri_chess::phase::Phase;
    use ndarray::Array2;
    use std::collections::HashSet;
    use game_tri_chess::moves::MoveType;

    #[test]
    fn policy_mapping_roundtrip_initial_pos() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessHybridCanonicalMapper;
        let mut move_store = <TriHexChess as Board>::MoveStore::default();

        for player in COLORS {
            board.set_phase(Phase::Normal(PlayerState::new_with_turn(player)));
            board.update_pseudo_moves(&mut move_store, false);

            let mut seen_indices = HashSet::new();
            let legal_moves = move_store.iter();

            for mv in legal_moves {
                let index = mapper.move_to_index(player, *mv);
                assert!(index < mapper.policy_len(), "Index out of bounds");
                assert!(
                    seen_indices.insert(index),
                    "Policy index collision for player {:?} on move {:?} -> index {}",
                    player,
                    mv,
                    index
                );

                let roundtrip_move = mapper
                    .index_to_move(&board, &move_store, index)
                    .expect("Failed to map index back to move");

                // Our index_to_move should find the exact move or a promotion to queen.
                assert_eq!(mv.from, roundtrip_move.from, "from square mismatch");
                assert_eq!(mv.to, roundtrip_move.to, "to square mismatch");
            }
        }
    }

    /// Tests a specific scenario with an en passant move.
    #[test]
    fn policy_mapping_en_passant() {
        let mapper = ChessHybridCanonicalMapper;
        let mut board = TriHexChess::default_with_grace_period();
        let mut move_store = <TriHexChess as Board>::MoveStore::default();

        let move1 = PseudoLegalMove {
            from: MemoryPos(10),
            to: MemoryPos(26),
            move_type: MoveType::DoublePawnPush,
        };
        board.commit_move(&move1, &move_store);
        board.next_turn(true, &mut move_store, false);

        let move2 = PseudoLegalMove {
            from: MemoryPos(47),
            to: MemoryPos(63),
            move_type: MoveType::DoublePawnPush,
        };
        board.commit_move(&move2, &move_store);
        board.next_turn(true, &mut move_store, false);

        // Black: f7-f5
        let move3 = PseudoLegalMove {
            from: MemoryPos(77),
            to: MemoryPos(93),
            move_type: MoveType::DoublePawnPush,
        };
        board.commit_move(&move3, &move_store);
        board.next_turn(true, &mut move_store, false);

        board.update_pseudo_moves(&mut move_store, false);

        let ep_move = move_store
            .iter()
            .find(|m| matches!(m.move_type, MoveType::EnPassant(_)))
            .expect("En passant move not found");

        let index = mapper.move_to_index(Color::White, *ep_move);
        let roundtrip_move = mapper
            .index_to_move(&board, &move_store, index)
            .expect("Failed to map EP index back to move");

        assert_eq!(ep_move, &roundtrip_move);
    }

    #[test]
    fn policy_out_of_bounds() {
        let mut board = TriHexChess::default_with_grace_period();
        let mut move_store = <TriHexChess as Board>::MoveStore::default();
        let mapper = ChessHybridCanonicalMapper;
        let policy_len = mapper.policy_len();
        board.update_pseudo_moves(&mut move_store, false);
        assert!(
            mapper
                .index_to_move(&board, &move_store, policy_len)
                .is_none()
        );
        assert!(
            mapper
                .index_to_move(&board, &move_store, policy_len + 1)
                .is_none()
        );
    }


    #[test]
    fn rotational_symmetry_move_to_index() {
        let mapper = ChessHybridCanonicalMapper;
        let white_move = PseudoLegalMove {
            from: MemoryPos(12),
            to: MemoryPos(28),
            move_type: MoveType::Move,
        };
        let gray_move = PseudoLegalMove {
            from: MemoryPos(44),
            to: MemoryPos(60),
            move_type: MoveType::Move,
        };
        let black_move = PseudoLegalMove {
            from: MemoryPos(76),
            to: MemoryPos(92),
            move_type: MoveType::Move,
        };
        let white_index = mapper.move_to_index(Color::White, white_move);
        let gray_index = mapper.move_to_index(Color::Gray, gray_move);
        let black_index = mapper.move_to_index(Color::Black, black_move);
        assert_eq!(
            white_index, gray_index,
            "Symmetry broken between White and Gray"
        );
        assert_eq!(
            white_index, black_index,
            "Symmetry broken between White and Black"
        );
    }


    #[test]
    fn rotational_symmetry_index_to_move() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessHybridCanonicalMapper;
        let mut move_store = <TriHexChess as Board>::MoveStore::default();
        let canonical_move_local = PseudoLegalMove {
            from: MemoryPos(12),
            to: MemoryPos(28),
            move_type: MoveType::DoublePawnPush,
        };
        let index = mapper.move_to_index(Color::White, canonical_move_local);
        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::White)));
        board.update_pseudo_moves(&mut move_store, false);
        let white_move = mapper.index_to_move(&board, &move_store, index).unwrap();
        assert_eq!(white_move.from.0, 12);
        assert_eq!(white_move.to.0, 28);
        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Gray)));
        board.update_pseudo_moves(&mut move_store, false);
        let gray_move = mapper.index_to_move(&board, &move_store, index).unwrap();
        assert_eq!(gray_move.from.0, 44);
        assert_eq!(gray_move.to.0, 60);
        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Black)));
        board.update_pseudo_moves(&mut move_store, false);
        let black_move = mapper.index_to_move(&board, &move_store, index).unwrap();
        assert_eq!(black_move.from.0, 76);
        assert_eq!(black_move.to.0, 92);
    }


    #[test]
    fn rotational_symmetry_encode_input() {
        let mapper = ChessHybridCanonicalMapper;

        let mut board1 = TriHexChess::default_with_grace_period();
        board1.state.buffer.clear();
        board1.state.buffer[MemoryPos(12).0 as usize] = MemorySlot::new(Color::White, Piece::Pawn);
        board1.state.buffer[MemoryPos(51).0 as usize] = MemorySlot::new(Color::Gray, Piece::Knight);
        board1.state.buffer[MemoryPos(91).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Bishop);
        board1.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::White)));

        board1.state.castle.set_queen_side(Color::White, true);
        board1.state.castle.set_king_side(Color::Gray, true);

        board1.state.en_passant.set_pos(Color::White, MemoryPos(16));

        let mut board2 = TriHexChess::default_with_grace_period();
        board2.state.buffer.clear();
        let gray_offset = Color::Gray.get_offset();
        board2.state.buffer[MemoryPos(12).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::Gray, Piece::Pawn);
        board2.state.buffer[MemoryPos(51).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Knight);
        board2.state.buffer[MemoryPos(91).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::White, Piece::Bishop);
        board2.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Gray)));

        board2.state.castle.set_queen_side(Color::Gray, true);
        board2.state.castle.set_king_side(Color::Black, true);

        board2.state.en_passant.set_pos(Color::Gray, MemoryPos(48));

        let mut board3 = TriHexChess::default_with_grace_period();
        board3.state.buffer.clear();
        let black_offset = Color::Black.get_offset();
        board3.state.buffer[MemoryPos(12).to_global(black_offset).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Pawn);
        board3.state.buffer[MemoryPos(51).to_global(black_offset).0 as usize] =
            MemorySlot::new(Color::White, Piece::Knight);
        board3.state.buffer[MemoryPos(91).to_global(black_offset).0 as usize] =
            MemorySlot::new(Color::Gray, Piece::Bishop);
        board3.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Black)));

        board3.state.castle.set_queen_side(Color::Black, true);
        board3.state.castle.set_king_side(Color::White, true);

        board3.state.en_passant.set_pos(Color::Black, MemoryPos(80));

        let mut input1 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input2 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input3 = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input1.view_mut(), &board1);
        mapper.encode_input(&mut input2.view_mut(), &board2);
        mapper.encode_input(&mut input3.view_mut(), &board3);

        if input1 != input2 || input1 != input3 {
            println!("--- Board 1 (White's Turn) ---");
            println!("{}", pretty_print_hybrid_chess_tensor(&input1));
            println!("--- Board 2 (Gray's Turn) ---");
            println!("{}", pretty_print_hybrid_chess_tensor(&input2));
            println!("--- Board 3 (Black's Turn) ---");
            println!("{}", pretty_print_hybrid_chess_tensor(&input3));
        }

        assert_eq!(
            input1, input2,
            "Input for White and Gray rotated boards should be identical"
        );
        assert_eq!(
            input1, input3,
            "Input for White and Black rotated boards should be identical"
        );
    }

    #[test]
    fn test() {
        // FEN with castling (qkqkqk) and en passant (targets on file 1 for each player)
        let fen = "rnbqkbnr/1pppppp1/8/p6p/X/X X/rnbqkbnr/1pppppp1/8/p6p/X X/X/rnbqkbnr/1pppppp1/8/p6p W qkqkqk 111 0 1";
        let mut board = TriHexChess::new_with_fen(fen.as_ref(), true).unwrap();
        let mapper = ChessHybridCanonicalMapper;
        let mut input_w = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input_g = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input_b = Array2::from_elem(mapper.input_board_shape(), false);

        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::White)));
        mapper.encode_input(&mut input_w.view_mut(), &board);

        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Gray)));
        mapper.encode_input(&mut input_g.view_mut(), &board);

        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Black)));
        mapper.encode_input(&mut input_b.view_mut(), &board);

        assert_eq!(input_w, input_g);
        assert_eq!(input_w, input_b);

        colored::control::set_override(true);
        println!("Symmetrically Encoded Tensor (White's Perspective)");
        println!("{}", pretty_print_hybrid_chess_tensor(&input_w));
    }

    #[test]
    fn test_vis_fen() {
        colored::control::set_override(true);

        let fen =
            "rnbq1bnr/ppppp2p/3k1p2/6p1/X/X X/r1bqkbnr/p1pppppp/8/np6/X X/X/X W --qk-- 7-- 0 1";

        let board = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();
        let mapper = ChessHybridCanonicalMapper;
        let mut input_w = Array2::from_elem(mapper.input_board_shape(), false);
        mapper.encode_input(&mut input_w.view_mut(), &board);

        println!("Symmetrically Encoded Tensor (White's Perspective)");
        println!("{}", pretty_print_hybrid_chess_tensor(&input_w));

        let board = TriHexChess::default_with_grace_period();
        mapper.encode_input(&mut input_w.view_mut(), &board);

        println!("Symmetrically Encoded Tensor (Default Position)");
        println!("{}", pretty_print_hybrid_chess_tensor(&input_w));
    }
}
