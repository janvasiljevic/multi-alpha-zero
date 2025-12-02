use crate::mapping::{Board, BoardPlayer, Outcome};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::{HexGame, HexGameOutcome, HexPlayer, ALL_HEX_PLAYERS};

impl BoardPlayer for HexPlayer {
    fn next(self) -> Self {
        self.next_hex_player()
    }
}

impl Board for HexGame {
    type Move = AxialCoord;
    type Player = HexPlayer;
    type MoveStore = Vec<Self::Move>;
    type HashKey = u64;

    fn new(&self) -> Self {
        HexGame::new(self.radius + 1).unwrap()
    }

    fn player_current(&self) -> Self::Player {
        self.current_turn
    }

    fn player_num(&self) -> usize {
        3
    }

    fn player_num_of_active(&self) -> usize {
        3 - self.eliminated_players.len()
    }

    fn fancy_debug(&self) -> String {
        self.fancy_debug_visualize_board_indented(None)
    }

    fn fill_move_store(&mut self, store: &mut Self::MoveStore) {
        self.fill_vector_with_moves(store);
    }

    fn play_move_mut_with_store(
        &mut self,
        mv: &Self::Move,
        move_store: &mut Self::MoveStore,
        _next_player_hint: Option<Self::Player>,
    ) {
        let res = self.make_move_mut(*mv);

        if res.is_err() {
            panic!("Invalid move attempted: {:?}", mv);
        }

        if self.is_terminal() {
            move_store.clear();
        } else {
            self.fill_vector_with_moves(move_store);
        }
    }

    fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }

    fn outcome(&self, forced_end: bool) -> Option<Outcome> {
        
        if forced_end && self.outcome.is_none() {
            let mut partial_draw_mask = 0u8;
            
            for player in ALL_HEX_PLAYERS {
                if !self.eliminated_players.contains(&player) {
                    partial_draw_mask |= 1 << (player as u8);
                }
            }
            
            return Some(Outcome::PartialDraw(partial_draw_mask));
        }
        
        if let Some(outcome) = &self.outcome {
            // If hex has an outcome, it can only be a win by one of the players.
            match outcome {
                HexGameOutcome::Win { winner } => Some(Outcome::WonBy(*winner as u8)),
            }
        } else {
            None
        }
    }

    fn hash_key(&self) -> Self::HashKey {
        self.zobrist_hash
    }
}
