use crate::search_settings::SearchSettings;
use crate::self_play_game::{ProduceOutput, SPGStats, SelfPlayGame, Simulation};
use crate::steppable_mcts::{CacheShapeMissmatch, ConsumeValues};
use rand::Rng;
use tokio::sync::mpsc::UnboundedSender;
use maz_core::mapping::{Board, MetaBoardMapper, OptionalSharedOracle};
use crate::learning_target_modifier::LearningModifier;

#[derive(Debug, Clone)]
pub enum DynamicSelfPlayGame<B: Board + 'static> {
    Two(SelfPlayGame<B, 2>),
    Three(SelfPlayGame<B, 3>),
    Four(SelfPlayGame<B, 4>),
}

// A macro to dispatch calls to the inner enum variant.
macro_rules! dispatch_game {
    ($self:ident, $name:ident($($arg:expr),*)) => {
        match $self {
            DynamicSelfPlayGame::Two(g) => g.$name($($arg),*),
            DynamicSelfPlayGame::Three(g) => g.$name($($arg),*),
            DynamicSelfPlayGame::Four(g) => g.$name($($arg),*),
        }
    };
    // A mutable version for methods that take &mut self
     ($self:ident, mut $name:ident($($arg:expr),*)) => {
        match $self {
            DynamicSelfPlayGame::Two(g) => g.$name($($arg),*),
            DynamicSelfPlayGame::Three(g) => g.$name($($arg),*),
            DynamicSelfPlayGame::Four(g) => g.$name($($arg),*),
        }
    };
}

impl<B: Board> DynamicSelfPlayGame<B> {
    pub fn new(
        game: &B,
        mapper: &impl MetaBoardMapper<B>,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        oracle: OptionalSharedOracle<B>,
        learning_modifier: LearningModifier
    ) -> Self {
        match game.player_num() {
            2 => DynamicSelfPlayGame::Two(SelfPlayGame::<B, 2>::new(game, mapper, rng, settings, oracle, learning_modifier)),
            3 => DynamicSelfPlayGame::Three(SelfPlayGame::<B, 3>::new(game, mapper, rng, settings, oracle, learning_modifier)),
            4 => DynamicSelfPlayGame::Four(SelfPlayGame::<B, 4>::new(game, mapper, rng, settings, oracle, learning_modifier)),
            n => panic!("Unsupported number of players: {n}"),
        }
    }

    pub fn advance(
        &mut self,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        samples_tx: &UnboundedSender<Simulation<B>>,
        board_mapper: &impl MetaBoardMapper<B>,
    ) -> Option<ProduceOutput<B>> {
        dispatch_game!(self, mut advance(rng, settings, samples_tx, board_mapper))
    }

    pub fn receive(
        &mut self,
        rng: &mut impl Rng,
        settings: &SearchSettings,
        node_id: usize,
        values: ConsumeValues,
    ) -> Result<(), CacheShapeMissmatch> {
        dispatch_game!(self, mut receive(rng, settings, node_id, values))
    }

    pub fn stats(&self) -> &SPGStats {
        match self {
            DynamicSelfPlayGame::Two(g) => &g.stats,
            DynamicSelfPlayGame::Three(g) => &g.stats,
            DynamicSelfPlayGame::Four(g) => &g.stats,
        }
    }

    pub fn get_position_count(&self) -> usize {
        dispatch_game!(self, get_position_count())
    }
}
