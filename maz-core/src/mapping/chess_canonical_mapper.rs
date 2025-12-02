use crate::mapping::{
    Board, BoardPlayer, InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper,
};
use game_tri_chess::basics::{Color, MemorySlot, Piece, PlayerState};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, MoveType, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::pos::MemoryPos;
use ndarray::{ArrayView2, ArrayViewMut2};

/// Represents the most basic mapping for TriHexChess,
/// without promotions -> **All pieces are always promoted to Queen.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessCanonicalMapper;

const NUM_OF_FIELDS: usize = 96; // 96 fields in TriHexChess
const CHANNELS_PER_PLAYER: usize = 6; // 6 channels per player (6 piece types)
const TOTAL_CHANNELS: usize = 3 * CHANNELS_PER_PLAYER; // 3 players

// const ME_CH: usize = 0;
const OPP_NEXT_CH: usize = 1 * CHANNELS_PER_PLAYER;
const OPP_NEXT_NEXT_CH: usize = 2 * CHANNELS_PER_PLAYER;

impl PolicyMapper<TriHexChess> for ChessCanonicalMapper {
    fn policy_len(&self) -> usize {
        NUM_OF_FIELDS * NUM_OF_FIELDS // From - to - mapping
    }

    /// `mv` is give in absolute coordinates (from/to),
    /// we just need to convert it to the canonical local coordinates.
    fn move_to_index(&self, player: Color, mv: PseudoLegalMove) -> usize {
        #[cfg(debug_assertions)]
        {
            // check that the move isn't a promotion to non-queen piece
            match mv.move_type {
                MoveType::Promotion(promoted_piece) => {
                    debug_assert_eq!(
                        promoted_piece,
                        Piece::Queen,
                        "Only promotion to Queen is supported in CanonicalChessMapping"
                    );
                }
                MoveType::EnPassantPromotion(wrapper) => {
                    let (_, promoted_piece) = wrapper.get();
                    debug_assert_eq!(
                        promoted_piece,
                        Piece::Queen,
                        "Only promotion to Queen is supported in CanonicalChessMapping"
                    );
                }
                _ => {}
            }
        }

        let offset = player.get_offset();

        let canonical_from = mv.from.to_local(offset).0 as usize;
        let canonical_to = mv.to.to_local(offset).0 as usize;

        debug_assert!(canonical_from < NUM_OF_FIELDS);
        debug_assert!(canonical_to < NUM_OF_FIELDS);

        canonical_from * NUM_OF_FIELDS + canonical_to
    }

    /// Converts a flat index into a PseudoLegalMove.
    /// The policy is given in canonical coordinates, so we just need to convert it back to absolute coordinates.
    /// Then we filter the moves in the board's move store to find the matching move.
    fn index_to_move(
        &self,
        board: &TriHexChess,
        move_store: &ChessMoveStore,
        index: usize,
    ) -> Option<PseudoLegalMove> {
        if index >= self.policy_len() {
            return None;
        }

        let player = board.player_current();
        let offset = player.get_offset();

        let canonical_from = index / NUM_OF_FIELDS;
        let canonical_to = index % NUM_OF_FIELDS;

        let from_pos = MemoryPos(canonical_from as u8).to_global(offset);
        let to_pos = MemoryPos(canonical_to as u8).to_global(offset);

        for mv in move_store {
            if mv.from == from_pos && mv.to == to_pos {
                match mv.move_type {
                    MoveType::Promotion(Piece::Queen) => {
                        return Some(*mv);
                    }
                    // Handle en passant promotion to Queen
                    MoveType::EnPassantPromotion(p) if p.get().1 == Piece::Queen => {
                        return Some(*mv); // Found preferred move, return immediately.
                    }
                    // For any other move type that matches from/to
                    _ => {
                        if !matches!(
                            mv.move_type,
                            MoveType::Promotion(_) | MoveType::EnPassantPromotion(_)
                        ) {
                            return Some(*mv); // Return the first matching move.
                        }
                    }
                }
            }
        }

        None
    }
}

impl InputMapper<TriHexChess> for ChessCanonicalMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        [NUM_OF_FIELDS, TOTAL_CHANNELS]
    }

    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &TriHexChess) {
        input_view.fill(false);

        let me = board.player_current();
        let next = me.next();
        let next_next = next.next();

        for (i, field) in board.state.buffer.non_empty_iter() {
            let piece = field.piece().unwrap();
            let player = field.player().unwrap();

            // Pieces start from 1, so we subtract 1 to get the index
            // 0 = Pawn, 1 = Knight, 2 = Bishop, 3 = Rook, 4 = Queen, 5 = King
            let piece_index = (piece as u8 - 1) as usize;

            let local_pos = i.to_local(me.get_offset()).0 as usize;

            if player == me {
                // Encode my pieces
                input_view[[local_pos, piece_index]] = true;
            } else if player == next {
                // Encode opponent's next pieces
                input_view[[local_pos, OPP_NEXT_CH + piece_index]] = true;
            } else if player == next_next {
                // Encode opponent's next-next pieces
                input_view[[local_pos, OPP_NEXT_NEXT_CH + piece_index]] = true;
            }
        }
    }

    fn is_absolute(&self) -> bool {
        false
    }
}

impl ReverseInputMapper<TriHexChess> for ChessCanonicalMapper {
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
        let next_next_player = next_player.next();

        let mut game = TriHexChess::default_with_grace_period();
        game.state.buffer.clear(); // Clears all pieces from the board

        for canonical_pos in 0..NUM_OF_FIELDS {
            for channel_index in 0..TOTAL_CHANNELS {
                if input_view[[canonical_pos, channel_index]] {
                    let (player, piece) = if channel_index < CHANNELS_PER_PLAYER {
                        let piece = Piece::from_u8((channel_index + 1) as u8).unwrap();
                        (me_player, piece)
                    } else if channel_index < 2 * CHANNELS_PER_PLAYER {
                        let piece_index = channel_index - CHANNELS_PER_PLAYER;
                        let piece = Piece::from_u8((piece_index + 1) as u8).unwrap();
                        (next_player, piece)
                    } else {
                        let piece_index = channel_index - 2 * CHANNELS_PER_PLAYER;
                        let piece = Piece::from_u8((piece_index + 1) as u8).unwrap();
                        (next_next_player, piece)
                    };

                    let offset = original_player.get_offset();
                    let absolute_pos = MemoryPos(canonical_pos as u8).to_global(offset);

                    game.state.buffer[absolute_pos.0 as usize] = MemorySlot::new(player, piece);

                    break;
                }
            }
        }

        game.state.castle = Default::default();
        game.state.en_passant = Default::default();
        game.state.third_move = 0;
        game.state.turn_counter = 1;

        // Reconstruct the PlayerState based on which kings are on the board.
        let mut player_state = PlayerState::default();
        player_state.set_turn(original_player);
        if game.state.buffer.king_mem_pos(Color::White).is_none() {
            player_state.remove_player(Color::White);
        }
        if game.state.buffer.king_mem_pos(Color::Gray).is_none() {
            player_state.remove_player(Color::Gray);
        }
        if game.state.buffer.king_mem_pos(Color::Black).is_none() {
            player_state.remove_player(Color::Black);
        }
        game.set_phase(Phase::Normal(player_state));

        // The board state is now fully constructed. We must update the internal caches.
        game.zobrist_hash = game.calculate_full_hash();

        game
    }
}

impl MetaPerformanceMapper<TriHexChess> for ChessCanonicalMapper {
    fn average_number_of_moves(&self) -> usize {
        // TODO: Experimentally determined value, not critical ATM
        27
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_tri_chess::basics::{Color, NormalState, COLORS};
    use game_tri_chess::chess_game::TriHexChess;
    use game_tri_chess::phase::Phase;
    use ndarray::Array2;
    use std::collections::HashSet;
    use colored::Colorize;

    fn pretty_print_tensor_format(input: &Array2<bool>) -> String {
        let mut output = String::new();

        // go through each row and then field
        for i in 0..TOTAL_CHANNELS {
            // print the first character of the piece if true, otherwise print "_"
            let piece_char = match i % CHANNELS_PER_PLAYER {
                0 => "P", // Pawn
                1 => "N", // Knight
                2 => "B", // Bishop
                3 => "R", // Rook
                4 => "Q", // Queen
                5 => "K", // King
                _ => "?",
            };
            output.push_str(format!("{piece_char} ").as_str());

            for j in 0..NUM_OF_FIELDS {
                if input[[j, i]] {
                    let char = "□";

                    let colored_string = match i / CHANNELS_PER_PLAYER {
                        0 => char.red(), // White pieces
                        1 => char.blue(),    // Gray pieces
                        2 => char.green(),    // Black pieces
                        _ => char.normal(),
                    };

                    output.push_str(&format!("{colored_string}"));
                } else {
                    output.push_str("_");
                }
            }
            output.push('\n');
        }

        output
    }

    #[test]
    fn test_canonical_chess_mapping() {
        let mapping = ChessCanonicalMapper;

        let mut board = TriHexChess::default_with_grace_period();

        let mut processed_inputs = vec![];

        for color in COLORS {
            let mut player_state = NormalState::default();
            player_state.set_turn(color);
            board.set_phase(Phase::Normal(player_state));

            let mut input = Array2::from_elem(mapping.input_board_shape(), false);
            let mut input_view = input.view_mut();

            mapping.encode_input(&mut input_view, &board);

            assert_eq!(input_view.shape(), &[NUM_OF_FIELDS, TOTAL_CHANNELS]);

            processed_inputs.push(input_view.to_owned());
        }
        let white_input = &processed_inputs[0];

        for color in &[Color::Gray, Color::Black] {
            let input = &processed_inputs[*color as u8 as usize];
            for i in 0..NUM_OF_FIELDS {
                for j in 0..TOTAL_CHANNELS {
                    assert_eq!(
                        input[[i, j]],
                        white_input[[i, j]],
                        "Mismatch at field {}, channel {}",
                        i,
                        j
                    );
                }
            }
        }

        let mut output = String::new();

        for i in 0..NUM_OF_FIELDS {
            for j in 0..TOTAL_CHANNELS {
                if white_input[[i, j]] {
                    output.push_str("X");
                } else {
                    output.push_str("_");
                }
            }
            output.push('\n');
        }

        println!("Processed input:\n{}", output);
    }

    /// Verifies that for a given player, all their legal moves can be converted
    /// to a unique index and back to the original move.
    #[test]
    fn policy_mapping_roundtrip_initial_pos() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessCanonicalMapper;
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
                    "Policy index collision for player {:?}",
                    player
                );

                let roundtrip_move = mapper
                    .index_to_move(&board, &move_store, index)
                    .expect("Failed to map index back to move");

                // Due to promotions, multiple moves can map to the same index.
                // We just need to ensure the from/to squares are correct.
                // Our index_to_move is designed to always pick the Queen promotion if available.
                assert_eq!(mv.from, roundtrip_move.from, "from square mismatch");
                assert_eq!(mv.to, roundtrip_move.to, "to square mismatch");
            }
        }
    }

    /// Verifies that an index outside the valid policy range returns None.
    #[test]
    fn policy_out_of_bounds() {
        let mut board = TriHexChess::default_with_grace_period();

        let mut move_store = <TriHexChess as Board>::MoveStore::default();
        let mapper = ChessCanonicalMapper;
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

    /// The core test for canonical policy mapping.
    /// A move and its rotationally symmetric equivalents for other players
    /// MUST map to the exact same policy index.
    #[test]
    fn rotational_symmetry_move_to_index() {
        let mapper = ChessCanonicalMapper;

        let white_move = PseudoLegalMove {
            from: MemoryPos(12),
            to: MemoryPos(28),
            move_type: MoveType::Move,
        };

        let green_move = PseudoLegalMove {
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
        let green_index = mapper.move_to_index(Color::Gray, green_move);
        let black_index = mapper.move_to_index(Color::Black, black_move);

        assert_eq!(
            white_index, green_index,
            "Symmetry broken between White and Green"
        );
        assert_eq!(
            white_index, black_index,
            "Symmetry broken between White and Black"
        );
    }

    /// The inverse of the symmetry test above. A single index must map to the correct
    /// absolute move depending on the current player's perspective.
    #[test]
    fn rotational_symmetry_index_to_move() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessCanonicalMapper;
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
        let green_move = mapper.index_to_move(&board, &move_store, index).unwrap();
        assert_eq!(green_move.from.0, 44);
        assert_eq!(green_move.to.0, 60);

        board.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Black)));
        board.update_pseudo_moves(&mut move_store, false);
        let black_move = mapper.index_to_move(&board, &move_store, index).unwrap();
        assert_eq!(black_move.from.0, 76);
        assert_eq!(black_move.to.0, 92);
    }

    /// The core test for the canonical input representation.
    /// Creates three boards that are rotationally identical and asserts
    /// that their encoded input tensors are exactly the same.
    #[test]
    fn rotational_symmetry_encode_input() {
        let mapper = ChessCanonicalMapper;

        // Create a custom, asymmetric board state from White's perspective.
        let mut board1 = TriHexChess::default_with_grace_period();
        board1.state.buffer.clear();
        board1.state.buffer[MemoryPos(12).0 as usize] = MemorySlot::new(Color::White, Piece::Pawn);
        board1.state.buffer[MemoryPos(51).0 as usize] = MemorySlot::new(Color::Gray, Piece::Knight);
        board1.state.buffer[MemoryPos(91).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Bishop);
        board1.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::White)));

        // Create board 2, which is board 1 rotated for Gray's perspective.
        let mut board2 = TriHexChess::default_with_grace_period();
        board2.state.buffer.clear();
        board2.state.buffer[MemoryPos(12).to_global(Color::Gray.get_offset()).0 as usize] =
            MemorySlot::new(Color::Gray, Piece::Pawn);
        board2.state.buffer[MemoryPos(51).to_global(Color::Gray.get_offset()).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Knight);
        board2.state.buffer[MemoryPos(91).to_global(Color::Gray.get_offset()).0 as usize] =
            MemorySlot::new(Color::White, Piece::Bishop);
        board2.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Gray)));

        // Create board 3, rotated for Black's perspective.
        let mut board3 = TriHexChess::default_with_grace_period();
        board3.state.buffer.clear();
        board3.state.buffer[MemoryPos(12).to_global(Color::Black.get_offset()).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Pawn);
        board3.state.buffer[MemoryPos(51).to_global(Color::Black.get_offset()).0 as usize] =
            MemorySlot::new(Color::White, Piece::Knight);
        board3.state.buffer[MemoryPos(91).to_global(Color::Black.get_offset()).0 as usize] =
            MemorySlot::new(Color::Gray, Piece::Bishop);
        board3.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Black)));

        let mut input1 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input2 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input3 = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input1.view_mut(), &board1);
        mapper.encode_input(&mut input2.view_mut(), &board2);
        mapper.encode_input(&mut input3.view_mut(), &board3);

        assert_eq!(
            input1, input2,
            "Input for White and Green rotated boards should be identical"
        );
        assert_eq!(
            input1, input3,
            "Input for White and Black rotated boards should be identical"
        );
    }

    /// Verifies that a board state can be encoded and then decoded
    /// back to the same set of piece positions.
    #[test]
    fn input_mapper_roundtrip() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessCanonicalMapper;
        let mut move_store = <TriHexChess as Board>::MoveStore::default();
        let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);

        board.update_pseudo_moves(&mut move_store, false);

        board.commit_move(&move_store.get(0).unwrap(), &move_store);
        board.next_turn(true, &mut move_store, false);

        board.commit_move(&move_store.get(1).unwrap(), &move_store);
        board.next_turn(true, &mut move_store, false);

        mapper.encode_input(&mut input_view.view_mut(), &board);

        let player_scalar = vec![board.player_current() as u8 as f32];
        let decoded_board = mapper.decode_input(&input_view.view(), &player_scalar);

        assert_eq!(
            board.state.buffer, decoded_board.state.buffer,
            "Decoded piece positions do not match original"
        );

        assert_eq!(
            board.player_current(),
            decoded_board.player_current(),
            "Decoded player turn does not match original"
        );
    }

    #[test]
    fn test() {
        let fen = "rnbqkbnr/1pppppp1/8/p6p/X/X X/rnbqkbnr/1pppppp1/8/p6p/X X/X/rnbqkbnr/1pppppp1/8/p6p W qkqkqk 111 0 3";
        let mut board = TriHexChess::new_with_fen(fen.as_ref(), true).unwrap();

        let mapper = ChessCanonicalMapper;

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

        println!("{}", pretty_print_tensor_format(&input_w));
        println!("{}", pretty_print_tensor_format(&input_g));
    }
}
