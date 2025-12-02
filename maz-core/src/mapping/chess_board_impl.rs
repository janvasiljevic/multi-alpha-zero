use crate::mapping::aux_values::get_num_of_aux_features;
use crate::mapping::{Board, BoardPlayer, MoveStore, Outcome};
use crate::values_const::ValuesAbs;
use game_tri_chess::basics::Color::White;
use game_tri_chess::basics::{CastleFlags, Color, EnPassantState, MemoryPos, MemorySlot, NormalState, Piece, COLORS};
use game_tri_chess::chess_game::{State, TriHexChess};
use game_tri_chess::moves::{ChessMoveStore, MoveType, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::repetition_history::RepetitionHistory;
use rand::{rng, Rng};
use rand::prelude::IndexedRandom;
use rand_distr::{Normal, Distribution};

impl BoardPlayer for Color {
    fn next(self) -> Self {
        self.right_player()
    }
}

impl MoveStore<PseudoLegalMove> for ChessMoveStore {
    fn clear(&mut self) {
        self.clear();
    }

    fn push(&mut self, mv: PseudoLegalMove) {
        if matches!(
            mv.move_type,
            MoveType::Promotion(_) | MoveType::EnPassantPromotion(_)
        ) {
            match mv.move_type {
                MoveType::Promotion(promoted_piece) => {
                    if promoted_piece != Piece::Queen {
                        return;
                    }
                }
                MoveType::EnPassantPromotion(wrapper) => {
                    if wrapper.get().1 != Piece::Queen {
                        return;
                    }
                }
                _ => unreachable!(),
            }
        }

        self.push(mv);
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn iter(&self) -> impl Iterator<Item=PseudoLegalMove> {
        self.iter().copied()
    }
}

fn fen_to_url(board: &TriHexChess) -> String {
    let cloned_board = board.clone();
    let fen = cloned_board.to_fen();

    format!(
        "Turn: {} - {} (http://localhost:5173?fen={})",
        board.state.turn_counter,
        fen,
        urlencoding::encode(&*cloned_board.to_fen())
    )
}

fn create_random_pawn_game(mut rng: &mut impl Rng) -> TriHexChess {
    let mut board = TriHexChess {
        state: State {
            buffer: Default::default(),
            phase: Phase::Normal(NormalState::new(White, true, true, true)),
            castle: CastleFlags::new(0),
            en_passant: EnPassantState::default(),
            third_move: 0,
            turn_counter: 1,
        },
        is_using_grace_period: false,
        zobrist_hash: 0,
        repetition_history: RepetitionHistory::default(),
    };

    // place kings at their starting positions
    board.state.buffer[4] = MemorySlot::new(White, Piece::King);
    board.state.buffer[36] = MemorySlot::new(Color::Gray, Piece::King);
    board.state.buffer[68] = MemorySlot::new(Color::Black, Piece::King);

    let pawn_starts: [Vec<MemoryPos>; 3] = [
        (8..16).map(|i| MemoryPos(i)).collect(), // White
        (40..48).map(|i| MemoryPos(i)).collect(), // Gray
        (72..80).map(|i| MemoryPos(i)).collect(), // Black
    ];

    let dist = Normal::new(5.5, 1.0).unwrap();

    for color in COLORS {
        for pos in &pawn_starts[color as usize] {
            board.state.buffer[*pos] = MemorySlot::empty();
        }

        let num_pawns_float: f32 = dist.sample(&mut rng); // Specify the type here
        let num_pawns = num_pawns_float.round().clamp(3.0, 8.0) as usize;

        let chosen_positions = pawn_starts[color as usize]
            .choose_multiple(&mut rng, num_pawns);

        for pos in chosen_positions {
            board.state.buffer[*pos] = MemorySlot::new(color, Piece::Pawn);
        }
    }

    let turn = rng.random_range(0..3);
    let normal_state = NormalState::new(Color::from_u8(turn).unwrap(), true, true, true);
    board.state.phase = Phase::Normal(normal_state);

    board.zobrist_hash = board.calculate_full_hash();

    board
}

impl Board for TriHexChess {
    type Move = PseudoLegalMove;
    type Player = Color;
    type MoveStore = ChessMoveStore;
    type HashKey = u64;

    fn new(&self) -> Self {
        TriHexChess::default_with_grace_period()
    }

    fn new_varied(&self) -> Self {
        // TODO: Seems to be same?
        let mut rng = rng();

        // White vs Green (6 percent)
        let w_vs_g =
            "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/X W qkqk-- --- 0 1";

        // White vs Black (6 percent)
        let w_vs_b =
            "rnbqkbnr/pppppppp/8/8/X/X X/X/X X/X/rnbqkbnr/pppppppp/8/8 W qk--qk --- 0 1";

        // Green vs Black (6 percent)
        let g_vs_b =
            "X/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 G --qkqk --- 0 1";


        match rng.random_range(0..100) {
            0..=6 => TriHexChess::new_with_fen(w_vs_g.as_ref(), false).unwrap(),
            7..=12 => TriHexChess::new_with_fen(w_vs_b.as_ref(), false).unwrap(),
            13..=18 => TriHexChess::new_with_fen(g_vs_b.as_ref(), false).unwrap(),
            19..=28 => create_random_pawn_game(&mut rng),

            _ => TriHexChess::default_with_grace_period(),
        }
    }

    fn player_current(&self) -> Self::Player {
        match self.get_phase() {
            Phase::Normal(player_state) => player_state.get_turn(),
            Phase::Draw(stalemate_state) => stalemate_state.get_turn(),
            Phase::Won(color) => color,
        }
    }

    fn player_num(&self) -> usize {
        3
    }

    fn player_num_of_active(&self) -> usize {
        match self.get_phase() {
            Phase::Normal(normal) => normal.get_player_count() as usize,
            Phase::Draw(stalemate_state) => stalemate_state.get_player_count() as usize,
            Phase::Won(_) => 1
        }
    }

    fn fancy_debug(&self) -> String {
        fen_to_url(self)
    }

    fn fill_move_store(&mut self, store: &mut Self::MoveStore) {
        self.update_pseudo_moves(store, false);
    }

    fn play_move_mut_with_store(
        &mut self,
        mv: &Self::Move,
        move_store: &mut Self::MoveStore,
        next_player_hint: Option<Self::Player>,
    ) {
        self.commit_move(mv, &move_store);

        // If the next player hint is provided and matches the expected next player, it means:
        // - no player was skipped (e.g., due to checkmate or stalemate).
        // Thus, we can directly set the next turn to the hinted player.
        // get_turn() return the next LOGICAL player (so if white -> gray).
        // It doesn't take into account that a player may be missing.
        if let Some(next_player) = next_player_hint
            && next_player == self.get_turn().unwrap().next()
        {
            self.next_turn_unsafe(next_player);
            move_store.clear();
        } else {
            self.next_turn(true, move_store, false);
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_over()
    }

    fn outcome(&self, forced_end: bool) -> Option<Outcome> {
        if forced_end && !self.is_over() {
            match self.get_phase() {
                Phase::Normal(player_state) => {
                    let mut partial_draw_mask = 0u8;

                    for color in COLORS {
                        if player_state.is_present(color) {
                            partial_draw_mask |= 1 << color as u8;
                        }
                    }

                    return Some(Outcome::PartialDraw(partial_draw_mask));
                }
                // Others should then be handled below
                _ => {}
            }
        }

        match self.get_phase() {
            Phase::Normal(_) => None,
            Phase::Draw(stalemate_state) => {
                let mut partial_draw_mask = 0u8;

                for color in COLORS {
                    if stalemate_state.is_present(color) {
                        partial_draw_mask |= 1 << color as u8;
                    }
                }

                Some(Outcome::PartialDraw(partial_draw_mask))
            }
            Phase::Won(color) => Some(Outcome::WonBy(color as u8)),
        }
    }

    fn hash_key(&self) -> Self::HashKey {
        self.zobrist_hash
    }

    fn get_heuristic_vector(&self) -> Option<Vec<f32>> {
        match self.get_phase() {
            Phase::Won(_) | Phase::Draw(_) => None,
            Phase::Normal(players) => {
                let mut material_values = [0u16; 3]; // Use u16 to be safe

                for i in 0u8..96 {
                    let slot = self.state.buffer[i as usize].get();
                    if let Some((color, piece)) = slot {
                        material_values[color as usize] += piece.material() as u16;
                    }
                }

                let numbers_of_players = players.get_player_count();
                if numbers_of_players < 2 {
                    return None;
                }

                let mut outcome_vector = vec![0.0f32; 3];

                // A decisive advantage is roughly a queen (9) or a rook and a piece.
                // We use this to normalize the material difference into the [-1, 1] range.
                // A difference of 10 should strongly push the evaluation towards +1 or -1.
                const NORMALIZATION_FACTOR: f32 = 10.0;

                for p1_color in COLORS {
                    if !players.is_present(p1_color) {
                        // An eliminated player has the worst possible outcome.
                        outcome_vector[p1_color as usize] = -1.0;
                        continue;
                    }

                    let mut opponent_material_sum = 0.0;
                    let mut opponent_count = 0.0;

                    for p2_color in COLORS {
                        if p1_color == p2_color || !players.is_present(p2_color) {
                            continue;
                        }
                        opponent_material_sum += material_values[p2_color as usize] as f32;
                        opponent_count += 1.0;
                    }

                    if opponent_count > 0.0 {
                        let avg_opponent_material = opponent_material_sum / opponent_count;
                        let material_advantage =
                            material_values[p1_color as usize] as f32 - avg_opponent_material;

                        // Use tanh to squash the value into a range of (-1, 1).
                        // A material advantage of NORMALIZATION_FACTOR will result in an evaluation of ~0.76.

                        let heuristic = (material_advantage / NORMALIZATION_FACTOR).tanh();

                        if heuristic > 0.0 {
                            // Scale down so that amassing material is not too rewarding.
                            const POSITIVE_SCALE_FACTOR: f32 = 0.06;
                            outcome_vector[p1_color as usize] = heuristic * POSITIVE_SCALE_FACTOR;
                        } else {
                            // Penalize negative scores more strongly to encourage avoiding losing material.
                            const NEGATIVE_SCALE_FACTOR: f32 = 0.3;
                            outcome_vector[p1_color as usize] = heuristic * NEGATIVE_SCALE_FACTOR;
                        }
                    }
                }

                Some(outcome_vector)
            }
        }
    }

    fn can_game_end_early(&self) -> bool {
        match self.get_phase() {
            Phase::Normal(normal) => {
                // TODO: Maybe add a check to see if any player is in check?

                // If any of the players is stale, the game can not end early, because they might
                // be checkmated even with low material (Ultra rare, but better safe than sorry).
                for color in COLORS {
                    if normal.is_stale(color) {
                        return false;
                    }
                }
            }

            Phase::Won(_) => {
                return false;
            }

            Phase::Draw(_) => {
                return false;
            }
        }

        let mut player_materials = [PieceCounter::default(); 3];

        for i in 0u8..96 {
            let slot = self.state.buffer[i as usize].get();

            if let Some((color, piece)) = slot {
                player_materials[color as usize].pawns += (piece == Piece::Pawn) as u8;
                player_materials[color as usize].knights += (piece == Piece::Knight) as u8;
                player_materials[color as usize].bishops += (piece == Piece::Bishop) as u8;
                player_materials[color as usize].rooks += (piece == Piece::Rook) as u8;
                player_materials[color as usize].queens += (piece == Piece::Queen) as u8;
            }
        }

        let player_count = match self.get_phase() {
            Phase::Normal(normal) => normal.get_player_count(),
            _ => unreachable!(),
        };

        for color in COLORS {
            if player_materials[color as usize].has_mating_potential(player_count) {
                return false;
            }
        }

        true
    }


    /// Returns auxiliary values that may be useful for the neural network.
    /// If 4 aux values are requested, the 4th value is the turn counter normalized to [0, 1].
    fn get_aux_values(&self, is_absolute: bool) -> Option<Vec<f32>> {
        let mut values = vec![0u8; 3];

        const NORMALIZATION_FACTOR: f32 = 70.0;

        for i in 0..96 {
            let slot = self.state.buffer[i as usize].get();
            if let Some((color, piece)) = slot {
                let piece_value = piece.material();
                values[color as usize] += piece_value;
            }
        }

        let values_normed = values
            .iter()
            .map(|v| ((v as f32) / NORMALIZATION_FACTOR).min(0.95))
            .collect::<Vec<f32>>();

        let aux_values = ValuesAbs::<3> {
            value_abs: [
                values_normed[0],
                values_normed[1],
                values_normed[2],
            ],
            moves_left: f32::NAN, // Doesn't matter
        };

        let perspective_values = if is_absolute {
            aux_values.value_abs.as_slice().to_vec()
        } else {
            aux_values
                .pov(self.player_current() as usize)
                .value_pov
                .as_slice()
                .to_vec()
        };

        let number_of_aux_features = get_num_of_aux_features();

        match number_of_aux_features {
            4 => {
                let mut extended = perspective_values;
                let third_move = self.state.third_move.min(120) as usize;
                extended.push((third_move as f32) / 128.0);
                Some(extended)
            }
            _ => Some(perspective_values),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct PieceCounter {
    pub pawns: u8,
    pub knights: u8,
    pub bishops: u8,
    pub rooks: u8,
    pub queens: u8,
}

impl PieceCounter {
    // TODO: Think more about this function.
    /// It's super hard to say how to design this function properly,
    /// because the NN most certainly can't play checkmate with just a knight and bishop,
    /// but it's not impossible to checkmate with just those pieces...
    pub fn has_mating_potential(&self, player_count: u8) -> bool {
        match player_count {
            2 => {
                self.pawns > 0
                    || self.rooks > 0
                    || self.queens > 0
                    || self.knights >= 2 // Not sure about this one...
                    || self.bishops >= 3
            }
            _ => {
                // In 3-player chess, it's even harder to checkmate with minimal material,
                // so we require at least two rooks or queen or a pawn.
                // We can assume when 3 player are on board, that with a knight and bishop,
                // it's almost impossible to checkmate the third player, because the other
                // player would interfere.
                self.pawns > 0
                    || self.rooks >= 1
                    || self.queens > 0
                    || (self.rooks > 1 && self.knights > 0)
                    || (self.rooks > 1 && self.bishops > 0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rng;
    use crate::mapping::Board;
    use game_tri_chess::chess_game::TriHexChess;
    use crate::mapping::chess_board_impl::create_random_pawn_game;

    #[test]
    fn test_heuristic_vector() {
        let game = TriHexChess::default_with_grace_period();
        let vec = game.get_heuristic_vector().unwrap();
        assert_eq!(vec.len(), 3);
        assert!(vec.iter().all(|&x| x.abs() <= 1.0));

        let default_fen = TriHexChess::default().to_fen();

        let fens = [
            "8/8/1k6/8/X/X 8/6r1/8/8/3k4/5b2/1p6/8/X X/X/8/8/8/4k3 B ------ --- 16 1",
            "8/8/8/2k4p/8/8/8/2p5/8/4p3/8/r7 k7/8/8/8/X/X X/X/X W ------ --- 0 1",
            "X/X/8/1k6/8/8 1n6/1kb5/8/8/X/X X/8/1k6/8/8/X G ------ --- 0 1",
            "X/X/8/k7/8/8 k7/8/8/8/8/1r6/8/8/X X/X/X G ------ --- 0 1",
            "X/X/X X/8/k7/8/8/8/8/8/q7 8/1k6/1n6/8/X/X B ------ --- 0 1",
            "6k1/8/8/8/X/X 3p4/1q6/8/8/8/8/6k1/5p2/8/4p3/8/8 X/X/X G ------ --- 0 1",
            "X/X/X kp6/8/8/8/X/8/8/3q4/8 8/4k3/8/8/2r5/8/8/8/X B ------ --- 0 1",
            "8/k6n/8/1p6/8/8/8/5n2/X X/X/X X/X/kr6/8/8/7p B ------ --- 0 1",
            "8/3k4/8/2q5/X/8/2p5/8/8 8/8/k7/8/X/X X/8/8/k7/8/X B ------ --- 0 1",
            "8/3k4/8/8/X/X n7/8/8/8/3k4/8/8/8/4p3/8/8/3p4 X/8/8/8/2k5/X W ------ --- 0 1",
            "X/X/8/3k4/8/8 8/8/k7/8/X/X X/8/1k6/8/8/X G ------ --- 0 1",
            default_fen.as_str(),
        ];

        for fen in fens {
            // Set grace period to false, since these don't have turn counters set
            let game = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();
            let vec = game.get_heuristic_vector().unwrap();
            assert_eq!(vec.len(), 3);
            assert!(
                vec.iter().all(|&x| x.abs() <= 1.0),
                "Failed on fen: {} with vec: {:?}",
                fen,
                vec
            );

            println!("{} -> {:?}", game.fancy_debug(), vec);
            println!("Aux values: {:?}", game.get_aux_values(true));
        }
    }

    #[test]
    fn test_end_early() {
        let ok_to_end_fens = [
            "X/X/1k6/8/8/8 1k6/8/8/8/X/X 8/4b3/8/8/1k6/8/8/8/X W ------ --- 0 1",
            "X/X/8/3k4/8/8 8/8/k7/8/X/X X/8/nk6/8/8/X G ------ --- 0 1",
            "X/X/8/3k4/8/8 8/8/k7/8/X/X X/8/1k6/8/8/X G ------ --- 0 1",
        ];

        let default_fen = TriHexChess::default().to_fen();

        let not_ok_to_end_fens = [
            "8/8/8/2k4p/8/8/8/2p5/8/4p3/8/r7 k7/8/8/8/X/X X/X/X W ------ --- 0 1",
            "X/X/8/k7/8/8 k7/8/8/8/8/1r6/8/8/X X/X/X G ------ --- 0 1",
            "X/X/X X/8/k7/8/8/8/8/8/q7 8/1k6/1n6/8/X/X B ------ --- 0 1",
            "6k1/8/8/8/X/X 3p4/1q6/8/8/8/8/6k1/5p2/8/4p3/8/8 X/X/X G ------ --- 0 1",
            "X/X/X kp6/8/8/8/X/8/8/3q4/8 8/4k3/8/8/2r5/8/8/8/X B ------ --- 0 1",
            "8/3k4/8/8/X/X n7/8/8/8/3k4/8/8/8/4p3/8/8/3p4 X/8/8/8/2k5/X W ------ --- 0 1",
            default_fen.as_str(),
        ];

        for fen in ok_to_end_fens {
            let game = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();
            assert!(game.can_game_end_early(), "Failed on fen: {}", fen);
        }

        for fen in not_ok_to_end_fens {
            let game = TriHexChess::new_with_fen(fen.as_ref(), false).unwrap();
            assert!(!game.can_game_end_early(), "Failed on fen: {}", fen);
        }
    }

    #[test]
    fn random() {
        let only_left_player =
            "rnbqkbnr/pppppppp/8/8/X/X X/X/X X/X/rnbqkbnr/pppppppp/8/8 W qk--qk --- 0 1";

        let game = TriHexChess::new_with_fen(only_left_player.as_ref(), false).unwrap();
        println!("{:?}", game.state.phase);
    }

    #[test]
    fn test_and_print_100_random_pawn_games() {
        let mut rng = rng();

        for _ in 0..100 {
            let game = create_random_pawn_game(&mut rng);

            println!("FEN: {}", game.fancy_debug());
        }
    }
}
