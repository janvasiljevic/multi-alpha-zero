use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::{HexGame, HexGameOutcome};

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub(crate) struct BoardTile {
    pub q: i32,
    pub r: i32,
    pub state: Option<HexPlayer>,
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub(crate) enum HexPlayer {
    P1,
    P2,
    P3,
}

impl HexPlayer {
    fn new_from_internal(player: game_hex::game_hex::HexPlayer) -> Self {
        match player {
            game_hex::game_hex::HexPlayer::P1 => HexPlayer::P1,
            game_hex::game_hex::HexPlayer::P2 => HexPlayer::P2,
            game_hex::game_hex::HexPlayer::P3 => HexPlayer::P3,
        }
    }

    fn to_internal(&self) -> game_hex::game_hex::HexPlayer {
        match self {
            HexPlayer::P1 => game_hex::game_hex::HexPlayer::P1,
            HexPlayer::P2 => game_hex::game_hex::HexPlayer::P2,
            HexPlayer::P3 => game_hex::game_hex::HexPlayer::P3,
        }
    }
}

#[derive(Deserialize, Serialize, JsonSchema, Debug)]
pub(crate) struct BoardState {
    radius: u32,
    board: Vec<BoardTile>,
    current_turn: HexPlayer,
    eliminated_players: Vec<HexPlayer>,
    winner: Option<HexPlayer>,
}

impl BoardState {
    pub(crate) fn new_from_internal(internal: HexGame) -> Self {
        let state = internal.get_board_state();

        let mut board_state = Vec::with_capacity(state.len());

        for (coord, state) in state.iter() {
            let hex_coord = BoardTile {
                q: coord.q,
                r: coord.r,
                state: match state {
                    game_hex::game_hex::CellState::Empty => None,
                    game_hex::game_hex::CellState::Occupied(player) => {
                        Some(HexPlayer::new_from_internal(*player))
                    }
                },
            };

            board_state.push(hex_coord);
        }

        BoardState {
            radius: internal.radius as u32,
            board: board_state,
            current_turn: HexPlayer::new_from_internal(internal.current_turn),
            eliminated_players: internal
                .eliminated_players
                .iter()
                .map(|p| HexPlayer::new_from_internal(*p))
                .collect(),
            winner: internal.outcome.map(|p| match p {
                HexGameOutcome::Win { winner } => HexPlayer::new_from_internal(winner),
            }),
        }
    }

    pub(crate) fn to_internal(&self) -> HexGame {
        let mut game = HexGame::new((self.radius as i32) + 1).unwrap();

        for tile in &self.board {
            if let Some(player) = &tile.state {
                game.set_state(
                    AxialCoord {
                        q: tile.q,
                        r: tile.r,
                    },
                    game_hex::game_hex::CellState::Occupied(player.to_internal()),
                );
            }
        }

        game.rebuild_internal_state(self.current_turn.to_internal());

        game
    }
}
