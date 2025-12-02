use crate::mapping::chess_canonical_mapper::ChessCanonicalMapper;
use crate::mapping::chess_hybrid_canonical_mapper::ChessHybridCanonicalMapper;
use crate::mapping::{
    Board, BoardPlayer, InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper,
};
use colored::Colorize;
use game_tri_chess::basics::{Color, Piece};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::pos::MemoryPos;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, s};

/// Features a rich mapping of domain-specific knowledge:
///
/// --- [Base Features - Same as Hybrid Mapper] ---
/// [Local features]
/// - Is Me/Next/Prev Pawn (3 channels)
/// - Is Knight, Bishop, Rook, Queen, King (5 channels)
/// - Is Me/Next/Prev Owner of the associated Piece (3 channels)
/// - Is En Passant target square for Me/Next/Prev (3 channels)
///
/// [Global features]
/// - Can Castle King-side + Queen-side for Me/Next/Prev (6 channels)
/// - Is Player present (3 channels)
/// - Is Grace Period (1 channel)
///
/// [Domain-Specific Features]
/// [Local features]
/// - Per-piece attack maps for Me/Next/Prev (6 pieces * 3 players = 18 channels)
///
/// [Global features]
/// - Is Me/Next/Prev in check (3 channels)
/// - Repetition counter for Me/Next/Prev (3 placeholder channels)
/// - Turn counter progression (6 placeholder channels)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessDomainMapper;

const NUM_OF_FIELDS: usize = 96;

// --- Base piece representation (14 channels) ---
const ME_PAWN_CH: usize = 0;
const NEXT_PAWN_CH: usize = 1;
const PREV_PAWN_CH: usize = 2;

const SYMMETRIC_PIECE_START_CH: usize = 3; // N, B, R, Q, K (5 channels)

const KING_CH: usize = SYMMETRIC_PIECE_START_CH + 4;

const ME_OWNER_CH: usize = 8;
const NEXT_OWNER_CH: usize = 9;
const PREV_OWNER_CH: usize = 10;

const ME_EP_CH: usize = 11;
const NEXT_EP_CH: usize = 12;
const PREV_EP_CH: usize = 13;

// --- Base global features (10 channels) ---
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

// --- Other fundamental global features (8 channels) ---
// const REPETITION_START_CH: usize = 24;
const ONE_REP: usize = 24;
const TWO_REP: usize = 25;

const THIRD_MOVE_COUNTER: usize = 26; // 6 channels for third move counter (50 move rule)

// This constant is the single most important part of the new design.
pub const NUM_HASHABLE_CHANNELS: usize = 32;

// --- Per-piece attack maps (18 channels) ---
const ATTACKS_START_CH: usize = NUM_HASHABLE_CHANNELS; // Starts at 32
// Me's attacks (P, N, B, R, Q, K) = 32-37
// Next's attacks (P, N, B, R, Q, K) = 38-43
// Prev's attacks (P, N, B, R, Q, K) = 44-49

// --- Is in check status (3 channels) ---
const IS_CHECK_START_CH: usize = ATTACKS_START_CH + 18; // Starts at 50
const ME_IS_IN_CHECK_CH: usize = 50;
const NEXT_IS_IN_CHECK_CH: usize = 51;
const PREV_IS_IN_CHECK_CH: usize = 52;

const TOTAL_CHANNELS: usize = 53;
/// Helper to get the base channel index for a player's attack maps.

#[inline(always)]
fn player_attack_start_ch(player_id: usize) -> usize {
    ATTACKS_START_CH + player_id * 6
}

/// Helper to get the index for a piece's bitboard.
#[inline(always)]
fn piece_to_index(piece: Piece) -> usize {
    (piece as u8 - 1) as usize
}

impl PolicyMapper<TriHexChess> for ChessDomainMapper {
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

const fn init_coordinate_maps() -> [[usize; 96]; 3] {
    let mut maps = [[0usize; 96]; 3];
    let offsets = [
        Color::White.get_offset(), // 0
        Color::Gray.get_offset(),  // 32
        Color::Black.get_offset(), // 64
    ];

    let mut player_idx = 0;
    while player_idx < 3 {
        let offset = offsets[player_idx];
        let mut global_sq_idx = 0;
        while global_sq_idx < 96 {
            let global_pos = MemoryPos::new(global_sq_idx as u8);
            let local_pos = global_pos.to_local(offset);
            maps[player_idx][global_sq_idx] = local_pos.0 as usize;
            global_sq_idx += 1;
        }
        player_idx += 1;
    }
    maps
}

static COORDINATE_MAPS: [[usize; 96]; 3] = init_coordinate_maps();

impl InputMapper<TriHexChess> for ChessDomainMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        [NUM_OF_FIELDS, TOTAL_CHANNELS]
    }

    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &TriHexChess) {
        input_view.fill(false);

        let me = board.player_current();
        let next = me.next();
        let prev = me.next().next();

        let global_to_local_map = &COORDINATE_MAPS[me as usize];

        let input_slice = input_view.as_slice_mut().unwrap();

        let attack_data = board.calculate_per_piece_bitboard_attack_data();

        // Encode piece positions
        for (global_pos, field) in board.state.buffer.non_empty_iter() {
            let piece = field.piece().unwrap();
            let player = field.player().unwrap();
            let local_pos = global_to_local_map[global_pos.0 as usize]; // Instant lookup

            match piece {
                Piece::Pawn => {
                    let channel = if player == me {
                        ME_PAWN_CH
                    } else if player == next {
                        NEXT_PAWN_CH
                    } else {
                        PREV_PAWN_CH
                    };
                    input_slice[local_pos * TOTAL_CHANNELS + channel] = true;
                }
                _ => {
                    let piece_type_channel = SYMMETRIC_PIECE_START_CH + (piece as u8 - 2) as usize;
                    input_slice[local_pos * TOTAL_CHANNELS + piece_type_channel] = true;

                    let owner_channel = if player == me {
                        ME_OWNER_CH
                    } else if player == next {
                        NEXT_OWNER_CH
                    } else {
                        PREV_OWNER_CH
                    };
                    input_slice[local_pos * TOTAL_CHANNELS + owner_channel] = true;
                }
            }
        }

        // Encode en passant squares
        if let Some(pos) = board.state.en_passant.get(me) {
            let local_pos = global_to_local_map[pos.0 as usize]; // Instant lookup
            input_slice[local_pos * TOTAL_CHANNELS + ME_EP_CH] = true;
        }
        if let Some(pos) = board.state.en_passant.get(next) {
            let local_pos = global_to_local_map[pos.0 as usize]; // Instant lookup
            input_slice[local_pos * TOTAL_CHANNELS + NEXT_EP_CH] = true;
        }
        if let Some(pos) = board.state.en_passant.get(prev) {
            let local_pos = global_to_local_map[pos.0 as usize]; // Instant lookup
            input_slice[local_pos * TOTAL_CHANNELS + PREV_EP_CH] = true;
        }

        // Encode per-piece attack maps
        let players = [me, next, prev];
        for (player_id, &player) in players.iter().enumerate() {
            let base_ch = player_attack_start_ch(player_id);
            for piece in Piece::all() {
                let piece_idx = piece_to_index(piece);
                let mut attack_bb = attack_data.attacks[player as usize][piece_idx];
                while attack_bb != 0 {
                    let global_idx = attack_bb.trailing_zeros() as usize;
                    let local_pos = global_to_local_map[global_idx]; // Instant lookup
                    input_slice[local_pos * TOTAL_CHANNELS + base_ch + piece_idx] = true;
                    attack_bb &= attack_bb - 1;
                }
            }
        }

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

        // Player Presence
        input_view
            .slice_mut(s![.., ME_PLAYER_PRESENT_CH])
            .fill(true);
        if let Phase::Normal(state) = board.get_phase() {
            input_view
                .slice_mut(s![.., NEXT_PLAYER_PRESENT_CH])
                .fill(state.is_present(next));
            input_view
                .slice_mut(s![.., PREV_PLAYER_PRESENT_CH])
                .fill(state.is_present(prev));
        }

        // Grace Period
        let is_grace = board.is_using_grace_period && board.state.turn_counter < 3;
        input_view
            .slice_mut(s![.., IS_GRACE_PERIOD_CH])
            .fill(is_grace);

        // Is In Check status
        input_view
            .slice_mut(s![.., ME_IS_IN_CHECK_CH])
            .fill(attack_data.is_in_check[me as usize]);
        input_view
            .slice_mut(s![.., NEXT_IS_IN_CHECK_CH])
            .fill(attack_data.is_in_check[next as usize]);
        input_view
            .slice_mut(s![.., PREV_IS_IN_CHECK_CH])
            .fill(attack_data.is_in_check[prev as usize]);

        // Repetition Counter
        let rep_count = board.get_repetition_count();

        if rep_count == 1 {
            input_view.slice_mut(s![.., ONE_REP]).fill(true);
        }
        if rep_count >= 2 {
            input_view.slice_mut(s![.., TWO_REP]).fill(true);
        }

        const TURN_COUNTER_CHANNELS: usize = 6;

        let turn_value = board.state.third_move.min(120) as usize;

        for i in 0..TURN_COUNTER_CHANNELS {
            // Check if the i-th bit of turn_value is 1
            if (turn_value >> i) & 1 == 1 {
                // This is a global feature, so set it for all 96 fields
                input_view
                    .slice_mut(s![.., THIRD_MOVE_COUNTER + i])
                    .fill(true);
            }
        }
    }

    fn is_absolute(&self) -> bool {
        false
    }

    fn num_hashable_channels(&self) -> Option<usize> {
        Some(NUM_HASHABLE_CHANNELS)
    }
}

impl ReverseInputMapper<TriHexChess> for ChessDomainMapper {
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, scalars: &Vec<f32>) -> TriHexChess {
        // The first 24 channels are identical to the Hybrid Canonical Mapper.
        let base_feature_view = input_view.slice(s![.., 0..IS_GRACE_PERIOD_CH + 1]);

        let mut board = ChessHybridCanonicalMapper.decode_input(&base_feature_view, scalars);

        let mut third_move_counter = 0;
        for i in 0..6 {
            if input_view[[0, THIRD_MOVE_COUNTER + i]] {
                third_move_counter |= 1 << i;
            }
        }

        board.state.third_move = third_move_counter.min(120) as u8 as u16;

        board
    }
}

// MetaPerformanceMapper can also be reused directly.
impl MetaPerformanceMapper<TriHexChess> for ChessDomainMapper {
    fn average_number_of_moves(&self) -> usize {
        ChessHybridCanonicalMapper.average_number_of_moves()
    }
}

pub fn pretty_print_domain_chess_tensor(input: &Array2<bool>) -> String {
    let mut output = String::new();

    let mut channel_labels = Vec::new();

    // Base piece features (0-13)
    channel_labels.extend(
        [
            "mP ", "nP ", "pP ", " N ", " B ", " R ", " Q ", " K ", "mO ", "nO ", "pO ", "mEP",
            "nEP", "pEP",
        ]
        .iter()
        .map(|&s| s.to_string()),
    );

    // Base global features (14-23)
    channel_labels.extend(
        [
            "mQc", "mKc", "nQc", "nKc", "pQc", "pKc", "mPr", "nPr", "pPr", " GP",
        ]
        .iter()
        .map(|&s| s.to_string()),
    );

    // Repetition features (24-25)
    channel_labels.extend(["1Rp", "2Rp"].iter().map(|&s| s.to_string()));

    // Turn counter features (26-31)
    for i in 0..6 {
        channel_labels.push(format!("T_b{}", i));
    }

    // Attack features (32-49)
    let players = ["m", "n", "p"];
    let pieces = ["P", "N", "B", "R", "Q", "K"];
    for p_label in players {
        for pc_label in pieces {
            channel_labels.push(format!("{}{}>", p_label, pc_label));
        }
    }

    // Check status features (50-52)
    channel_labels.extend(["mCk", "nCk", "pCk"].iter().map(|&s| s.to_string()));

    assert_eq!(
        channel_labels.len(),
        TOTAL_CHANNELS,
        "Mismatch between labels ({}) and total channels ({})",
        channel_labels.len(),
        TOTAL_CHANNELS
    );

    let is_global = |ch: usize| -> bool {
        (ch >= CASTLING_START_CH && ch < ATTACKS_START_CH) || (ch >= IS_CHECK_START_CH)
    };

    for i in 0..TOTAL_CHANNELS {
        output.push_str(&format!("{: >4} ", channel_labels[i]));

        let mut has_data = false;
        for j in 0..NUM_OF_FIELDS {
            if input[[j, i]] {
                has_data = true;
                let char = "■";

                let colored_string = match i {
                    // --- Base Piece Features ---
                    ME_PAWN_CH | ME_OWNER_CH | ME_EP_CH => char.red(),
                    NEXT_PAWN_CH | NEXT_OWNER_CH | NEXT_EP_CH => char.blue(),
                    PREV_PAWN_CH | PREV_OWNER_CH | PREV_EP_CH => char.green(),
                    c if (SYMMETRIC_PIECE_START_CH..=KING_CH).contains(&c) => {
                        if input[[j, ME_OWNER_CH]] {
                            char.red()
                        } else if input[[j, NEXT_OWNER_CH]] {
                            char.blue()
                        } else if input[[j, PREV_OWNER_CH]] {
                            char.green()
                        } else {
                            char.white()
                        }
                    }

                    // --- Base Global Features ---
                    c if (ME_CASTLE_Q_CH..=ME_CASTLE_K_CH).contains(&c) => char.red(),
                    c if (NEXT_CASTLE_Q_CH..=NEXT_CASTLE_K_CH).contains(&c) => char.blue(),
                    c if (PREV_CASTLE_Q_CH..=PREV_CASTLE_K_CH).contains(&c) => char.green(),
                    ME_PLAYER_PRESENT_CH => char.red(),
                    NEXT_PLAYER_PRESENT_CH => char.blue(),
                    PREV_PLAYER_PRESENT_CH => char.green(),
                    IS_GRACE_PERIOD_CH => char.cyan(),
                    ONE_REP | TWO_REP => char.yellow(),
                    c if (THIRD_MOVE_COUNTER..THIRD_MOVE_COUNTER + 6).contains(&c) => {
                        char.magenta()
                    }

                    // --- Derived Attack Features ---
                    c if (ATTACKS_START_CH..ATTACKS_START_CH + 6).contains(&c) => char.red(),
                    c if (ATTACKS_START_CH + 6..ATTACKS_START_CH + 12).contains(&c) => char.blue(),
                    c if (ATTACKS_START_CH + 12..ATTACKS_START_CH + 18).contains(&c) => {
                        char.green()
                    }

                    // --- Derived Check Features ---
                    ME_IS_IN_CHECK_CH => char.red(),
                    NEXT_IS_IN_CHECK_CH => char.blue(),
                    PREV_IS_IN_CHECK_CH => char.green(),

                    _ => char.normal(),
                };
                output.push_str(&format!("{}", colored_string));
            } else {
                output.push_str("_");
            }
        }

        // For global features, only print if there's data to avoid clutter.
        if is_global(i) && !has_data {
            output.push_str(" (empty)");
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_mapper_smoke_test() {
        let board = TriHexChess::default_with_grace_period();
        let mapper = ChessDomainMapper;
        let mut input = Array2::from_elem(mapper.input_board_shape(), false);

        colored::control::set_override(true);

        mapper.encode_input(&mut input.view_mut(), &board);

        // A simple check: the "me player present" channel should be all true.
        let me_present_slice = input.slice(s![.., ME_PLAYER_PRESENT_CH]);
        assert!(
            me_present_slice.iter().all(|&v| v),
            "Me player should always be present"
        );

        // Another check: White's pawns should light up the pawn attack channels.
        let white_pawn_attack_ch = ATTACKS_START_CH + piece_to_index(Piece::Pawn);
        let white_pawn_attack_slice = input.slice(s![.., white_pawn_attack_ch]);
        assert!(
            white_pawn_attack_slice.iter().any(|&v| v),
            "White's pawn attacks should be present"
        );

        println!("{}", pretty_print_domain_chess_tensor(&input));

        let board = mapper.decode_input(&input.view(), &vec![0.0]);

        assert_eq!(board.player_current(), Color::White);
    }
}
