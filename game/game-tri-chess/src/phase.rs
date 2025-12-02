use std::fmt::{Debug, Display, Formatter};
use crate::basics::{Color, NormalState, PlayerState, DrawState};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Normal(NormalState),       // Normal state (2 or 3 players are still in the game)
    Won(Color),                // A player has won the game
    Draw(DrawState),
}

impl Default for Phase {
    fn default() -> Self {
        Phase::Normal(PlayerState::default())
    }
}

impl Phase {
    pub fn get_turn(&self) -> Color {
        match self {
            Phase::Normal(player_state) => player_state.get_turn(),
            Phase::Won(color) => *color,
            Phase::Draw(stalemate_state) => stalemate_state.get_turn(),
        }
    }
}

impl Display for Phase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Normal(player) => write!(f, "{}", player.get_turn().to_fen()),
            Phase::Won(color) => write!(f, "{}", color.to_fen()),
            Phase::Draw(color) => write!(f, "{}", color.get_turn().to_fen()),
        }
    }
}

impl Debug for Phase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Normal(player) => write!(f, "Normal({:?})", player),
            Phase::Won(color) => write!(f, "Won({})", color.to_fen()),
            Phase::Draw(stalemate_state) => write!(f, "Stalemate({:?})", stalemate_state),
        }
    }
}