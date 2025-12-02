use colored::Colorize;
use crate::mapping::{
    Board, BoardPlayer, InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper,
};
use game_tri_chess::basics::{Color, MemorySlot, Piece, PlayerState, COLORS};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::pos::MemoryPos;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use crate::mapping::chess_canonical_mapper::ChessCanonicalMapper;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessExtendedCanonicalMapper;

const NUM_OF_FIELDS: usize = 96;
const CHANNELS_PER_PLAYER: usize = 6;
const PIECE_CHANNELS: usize = 3 * CHANNELS_PER_PLAYER; // 18 channels for pieces


const CASTLING_CHANNELS: usize = 3 * 2;
const EN_PASSANT_CHANNELS: usize = 3;
const TOTAL_CHANNELS: usize = PIECE_CHANNELS + CASTLING_CHANNELS + EN_PASSANT_CHANNELS + 1; // 18 + 6 + 3 + 1 = 28

const OPP_NEXT_CH: usize = 1 * CHANNELS_PER_PLAYER;
const OPP_NEXT_NEXT_CH: usize = 2 * CHANNELS_PER_PLAYER;

const CASTLING_START_CH: usize = PIECE_CHANNELS;
const ME_CASTLE_CH: usize = CASTLING_START_CH; // Channels 18, 19
const OPP_NEXT_CASTLE_CH: usize = CASTLING_START_CH + 2; // Channels 20, 21
const OPP_NEXT_NEXT_CASTLE_CH: usize = CASTLING_START_CH + 4; // Channels 22, 23

// En Passant channels (24-26)
const EN_PASSANT_START_CH: usize = PIECE_CHANNELS + CASTLING_CHANNELS;
const ME_EN_PASSANT_CH: usize = EN_PASSANT_START_CH; // Channel 24 (EP created by me)
const OPP_NEXT_EN_PASSANT_CH: usize = EN_PASSANT_START_CH + 1; // Channel 25 (EP created by next opp)
const OPP_NEXT_NEXT_EN_PASSANT_CH: usize = EN_PASSANT_START_CH + 2; // Channel 26 (EP created by prev opp)

const IS_GRACE_PERIOD: usize = TOTAL_CHANNELS - 1;

impl PolicyMapper<TriHexChess> for ChessExtendedCanonicalMapper {
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

impl InputMapper<TriHexChess> for ChessExtendedCanonicalMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        [NUM_OF_FIELDS, TOTAL_CHANNELS]
    }

    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &TriHexChess) {
        input_view.fill(false);

        let me = board.player_current();
        let next = me.next();
        let next_next = next.next();
        let me_offset = me.get_offset();

        // --- 1. Encode Piece Positions (as before) ---
        for (i, field) in board.state.buffer.non_empty_iter() {
            let piece = field.piece().unwrap();
            let player = field.player().unwrap();

            let piece_index = (piece as u8 - 1) as usize;
            let local_pos = i.to_local(me_offset).0 as usize;

            if player == me {
                input_view[[local_pos, piece_index]] = true;
            } else if player == next {
                input_view[[local_pos, OPP_NEXT_CH + piece_index]] = true;
            } else if player == next_next {
                input_view[[local_pos, OPP_NEXT_NEXT_CH + piece_index]] = true;
            }
        }

        // --- 2. Encode Castling Rights ---
        let castle_flags = board.state.castle;
        // My rights
        input_view
            .slice_mut(s![.., ME_CASTLE_CH])
            .fill(castle_flags.can_king_side(me));
        input_view
            .slice_mut(s![.., ME_CASTLE_CH + 1])
            .fill(castle_flags.can_queen_side(me));
        // Next opponent's rights
        input_view
            .slice_mut(s![.., OPP_NEXT_CASTLE_CH])
            .fill(castle_flags.can_king_side(next));
        input_view
            .slice_mut(s![.., OPP_NEXT_CASTLE_CH + 1])
            .fill(castle_flags.can_queen_side(next));
        // Next-next opponent's rights
        input_view
            .slice_mut(s![.., OPP_NEXT_NEXT_CASTLE_CH])
            .fill(castle_flags.can_king_side(next_next));
        input_view
            .slice_mut(s![.., OPP_NEXT_NEXT_CASTLE_CH + 1])
            .fill(castle_flags.can_queen_side(next_next));

        input_view
            .slice_mut(s![.., IS_GRACE_PERIOD])
            .fill(if board.is_using_grace_period {
                board.state.turn_counter == 1
            } else {
                false
            });

        // --- 3. Encode En Passant ---
        let en_passant_state = board.state.en_passant;
        // EP square created by my move (vulnerable to others)
        if let Some(pos) = en_passant_state.get(me) {
            let local_pos = pos.to_local(me_offset).0 as usize;
            input_view[[local_pos, ME_EN_PASSANT_CH]] = true;
        }
        // EP square created by next opponent's move
        if let Some(pos) = en_passant_state.get(next) {
            let local_pos = pos.to_local(me_offset).0 as usize;
            input_view[[local_pos, OPP_NEXT_EN_PASSANT_CH]] = true;
        }
        // EP square created by previous opponent's move (vulnerable to me)
        if let Some(pos) = en_passant_state.get(next_next) {
            let local_pos = pos.to_local(me_offset).0 as usize;
            input_view[[local_pos, OPP_NEXT_NEXT_EN_PASSANT_CH]] = true;
        }
    }

    fn is_absolute(&self) -> bool {
        false
    }
}

impl ReverseInputMapper<TriHexChess> for ChessExtendedCanonicalMapper {
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

        // When decoding create WITH NO GRACE PERIOD.
        let mut game = TriHexChess::default();
        game.state.buffer.clear(); // Clears all pieces from the board

        // --- 1. Decode Piece Positions ---
        for canonical_pos in 0..NUM_OF_FIELDS {
            for channel_index in 0..PIECE_CHANNELS {
                if input_view[[canonical_pos, channel_index]] {
                    let (player, piece) = if channel_index < CHANNELS_PER_PLAYER {
                        (
                            me_player,
                            Piece::from_u8((channel_index + 1) as u8).unwrap(),
                        )
                    } else if channel_index < 2 * CHANNELS_PER_PLAYER {
                        let piece_index = channel_index - CHANNELS_PER_PLAYER;
                        (
                            next_player,
                            Piece::from_u8((piece_index + 1) as u8).unwrap(),
                        )
                    } else {
                        let piece_index = channel_index - 2 * CHANNELS_PER_PLAYER;
                        (
                            next_next_player,
                            Piece::from_u8((piece_index + 1) as u8).unwrap(),
                        )
                    };

                    let absolute_pos =
                        MemoryPos(canonical_pos as u8).to_global(original_player.get_offset());
                    game.state.buffer[absolute_pos.0 as usize] = MemorySlot::new(player, piece);
                    break;
                }
            }
        }

        // --- 2. Decode Castling Rights ---
        let mut castle_flags = game_tri_chess::basics::CastleFlags::new(0);
        // My rights
        castle_flags.set_king_side(me_player, input_view[[0, ME_CASTLE_CH]]);
        castle_flags.set_queen_side(me_player, input_view[[0, ME_CASTLE_CH + 1]]);
        // Next opponent's rights
        castle_flags.set_king_side(next_player, input_view[[0, OPP_NEXT_CASTLE_CH]]);
        castle_flags.set_queen_side(next_player, input_view[[0, OPP_NEXT_CASTLE_CH + 1]]);
        // Next-next opponent's rights
        castle_flags.set_king_side(next_next_player, input_view[[0, OPP_NEXT_NEXT_CASTLE_CH]]);
        castle_flags.set_queen_side(
            next_next_player,
            input_view[[0, OPP_NEXT_NEXT_CASTLE_CH + 1]],
        );
        game.state.castle = castle_flags;

        // --- 3. Decode En Passant ---
        let mut en_passant_state = game_tri_chess::basics::EnPassantState::default();
        let players = [me_player, next_player, next_next_player];
        let channels = [
            ME_EN_PASSANT_CH,
            OPP_NEXT_EN_PASSANT_CH,
            OPP_NEXT_NEXT_EN_PASSANT_CH,
        ];

        for i in 0..3 {
            for canonical_pos in 0..NUM_OF_FIELDS {
                if input_view[[canonical_pos, channels[i]]] {
                    let absolute_pos =
                        MemoryPos(canonical_pos as u8).to_global(original_player.get_offset());
                    en_passant_state.set_pos(players[i], absolute_pos);
                    break; // Only one EP square per player
                }
            }
        }
        game.state.en_passant = en_passant_state;

        // --- Restore rest of the state ---
        // These might need to be passed in via `scalars` for perfect reconstruction,
        // but for now we reset them.
        game.state.third_move = 0;
        game.state.turn_counter = 1;

        // Reconstruct the PlayerState based on which kings are on the board.
        let mut player_state = PlayerState::default();
        player_state.set_turn(original_player);
        for &player_color in COLORS.iter() {
            if game.state.buffer.king_mem_pos(player_color).is_none() {
                player_state.remove_player(player_color);
            }
        }
        game.set_phase(Phase::Normal(player_state));

        // The board state is now fully constructed. We must update the internal caches.
        game.zobrist_hash = game.calculate_full_hash();
        game
    }
}

impl MetaPerformanceMapper<TriHexChess> for ChessExtendedCanonicalMapper {
    fn average_number_of_moves(&self) -> usize {
        37
    }
}

pub fn pretty_print_extended_chess_tensor(input: &Array2<bool>) -> String {
    let mut output = String::new();

    let mut channel_labels = vec![];
    // Piece labels
    for color_prefix in ["W", "G", "B"] {
        for piece_char in ["P", "N", "B", "R", "Q", "K"] {
            channel_labels.push(format!("{color_prefix}{piece_char}"));
        }
    }
    // Castling labels
    for color_prefix in ["W", "G", "B"] {
        channel_labels.push(format!("{color_prefix}K")); // Kingside
        channel_labels.push(format!("{color_prefix}Q")); // Queenside
    }
    // En Passant labels
    for color_prefix in ["W", "G", "B"] {
        channel_labels.push(format!("{color_prefix}E"));
    }

    channel_labels.push("GP".to_string()); // Grace Period

    // go through each row and then field
    for i in 0..TEST_TOTAL_CHANNELS {
        output.push_str(format!("{: >3} ", channel_labels[i]).as_str());

        for j in 0..NUM_OF_FIELDS {
            if input[[j, i]] {
                let char = "■"; // Use a more visible character
                let colored_string = if i < PIECE_CHANNELS {
                    match i / CHANNELS_PER_PLAYER {
                        0 => char.red(),
                        1 => char.blue(),
                        2 => char.green(),
                        _ => char.normal(),
                    }
                } else if i < CASTLING_START_CH + CASTLING_CHANNELS {
                    char.yellow() // Castling rights in yellow
                } else if i == IS_GRACE_PERIOD {
                    char.cyan() // Grace period in cyan
                } else {
                    char.magenta() // En passant in magenta
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

// The total number of channels has changed. This constant must be updated.
const TEST_TOTAL_CHANNELS: usize = 28;


#[cfg(test)]
mod tests {
    use super::*;
    use game_tri_chess::basics::{Color, COLORS};
    use game_tri_chess::chess_game::TriHexChess;
    use game_tri_chess::phase::Phase;
    use ndarray::Array2;
    use std::collections::HashSet;
    use game_tri_chess::moves::MoveType;

    /// Verifies that for a given player, all their legal moves can be converted
    /// to a unique index and back to the original move.
    #[test]
    fn policy_mapping_roundtrip_initial_pos() {
        let mut board = TriHexChess::default_with_grace_period();
        let mapper = ChessExtendedCanonicalMapper;
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
        let mapper = ChessExtendedCanonicalMapper;
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
        let mapper = ChessExtendedCanonicalMapper;
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
        let mapper = ChessExtendedCanonicalMapper;
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
        let mapper = ChessExtendedCanonicalMapper;
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

    /// The core test for the canonical input representation.
    /// Creates three boards that are rotationally identical and asserts
    /// that their encoded input tensors are exactly the same.
    /// **This test is now updated to include asymmetric castling and en passant states.**
    #[test]
    fn rotational_symmetry_encode_input() {
        let mapper = ChessExtendedCanonicalMapper;

        // --- Board 1: White's Perspective ---
        let mut board1 = TriHexChess::default_with_grace_period();
        board1.state.buffer.clear();
        board1.state.buffer[MemoryPos(12).0 as usize] = MemorySlot::new(Color::White, Piece::Pawn);
        board1.state.buffer[MemoryPos(51).0 as usize] = MemorySlot::new(Color::Gray, Piece::Knight);
        board1.state.buffer[MemoryPos(91).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Bishop);
        board1.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::White)));
        // Asymmetric castling: W=Q, G=K, B=-
        board1.state.castle.set_queen_side(Color::White, true);
        board1.state.castle.set_king_side(Color::Gray, true);
        // Asymmetric en passant: Black just moved a pawn, creating an EP target for White/Gray
        board1.state.en_passant.set_pos(Color::White, MemoryPos(16)); // Black pawn on e7->e5, EP on e6

        // --- Board 2: Rotated for Gray's Perspective ---
        let mut board2 = TriHexChess::default_with_grace_period();
        board2.state.buffer.clear();
        let gray_offset = Color::Gray.get_offset();
        board2.state.buffer[MemoryPos(12).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::Gray, Piece::Pawn); // White's pawn becomes Gray's
        board2.state.buffer[MemoryPos(51).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::Black, Piece::Knight); // Gray's knight becomes Black's
        board2.state.buffer[MemoryPos(91).to_global(gray_offset).0 as usize] =
            MemorySlot::new(Color::White, Piece::Bishop); // Black's bishop becomes White's
        board2.set_phase(Phase::Normal(PlayerState::new_with_turn(Color::Gray)));
        // Rotated castling: G=Q, B=K, W=-
        board2.state.castle.set_queen_side(Color::Gray, true);
        board2.state.castle.set_king_side(Color::Black, true);
        // Rotated en passant: White just moved pawn, EP square is rotated version of b1's
        board2.state.en_passant.set_pos(Color::Gray, MemoryPos(48)); // Black's EP becomes White's

        // --- Board 3: Rotated for Black's Perspective ---
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
        // Rotated castling: B=Q, W=K, G=-
        board3.state.castle.set_queen_side(Color::Black, true);
        board3.state.castle.set_king_side(Color::White, true);
        // Rotated en passant: Gray just moved pawn
        board3.state.en_passant.set_pos(Color::Black, MemoryPos(80));

        let mut input1 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input2 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input3 = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input1.view_mut(), &board1);
        mapper.encode_input(&mut input2.view_mut(), &board2);
        mapper.encode_input(&mut input3.view_mut(), &board3);

        if input1 != input2 || input1 != input3 {
            println!("--- Board 1 (White's Turn) ---");
            println!("{}", pretty_print_extended_chess_tensor(&input1));
            println!("--- Board 2 (Gray's Turn) ---");
            println!("{}", pretty_print_extended_chess_tensor(&input2));
            println!("--- Board 3 (Black's Turn) ---");
            println!("{}", pretty_print_extended_chess_tensor(&input3));
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
        let mapper = ChessExtendedCanonicalMapper;
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
        println!("--- Symmetrically Encoded Tensor (White's Perspective) ---");
        println!("{}", pretty_print_extended_chess_tensor(&input_w));
    }

    #[test]
    fn test_vis_fen() {
        let fen = "rnbq1bnr/ppppp2p/3k1p2/6p1/X/X X/r1bqkbnr/p1pppppp/8/np6/X X/X/X W --qk-- 7-- 0 1";

        let board = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();
        let mapper = ChessExtendedCanonicalMapper;
        let mut input_w = Array2::from_elem(mapper.input_board_shape(), false);
        mapper.encode_input(&mut input_w.view_mut(), &board);
        colored::control::set_override(true);
        println!("--- Symmetrically Encoded Tensor (White's Perspective) ---");
        println!("{}", pretty_print_extended_chess_tensor(&input_w));

    }
}
