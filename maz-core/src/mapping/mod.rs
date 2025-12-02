use ndarray::{ArrayView2, ArrayViewMut2};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

pub mod hex_board_impl;
pub mod hex_absolute_mapper;

pub mod chess_board_impl;
pub mod chess_canonical_mapper;
pub mod hex_canonical_mapper;
pub mod chess_extended_canonical_mapper;
pub mod hex_wrapper_mapper;
pub mod chess_hybrid_canonical_mapper;
mod chess_oracle_impl;
pub mod chess_domain_canonical_mapper;
pub mod aux_values;
pub mod chess_wrapper_mapper;

/// An intermediate representation of the outcome of a game.
/// Supports all possible outcomes - up to 8 players, since there are 8 bits in an u8.
/// To see how the values are interpreted, look at [`ValuesAbs::from_outcome`].
/// WonBy represent the index of the player that won (0-indexed).
/// PartialDraw is a bitmask of players that drew, where the bits are set for players that drew and unset for players that lost.
/// - it follows the format that the first bit is for player 0, the second for player 1, etc. e.g:
/// - `0b00000101` means that player 0 and player 2 drew, while player 1 lost.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Outcome {
    WonBy(u8), // Player number that won (0-indexed)
    AllDraw,
    PartialDraw(u8), // Bitmask of players that drew - others lost
}

impl Display for Outcome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::WonBy(player) => write!(f, "Won by Player {}", player),
            Outcome::AllDraw => write!(f, "All Draw"),
            Outcome::PartialDraw(mask) => {
                let mut players = vec![];
                for i in 0..8 {
                    if (mask >> i) & 1 == 1 {
                        players.push(format!("Player {}", i));
                    }
                }
                write!(f, "Partial Draw between {}", players.join(", "))
            }
        }
    }
}

pub trait BoardPlayer: Into<usize> + From<usize> + Copy + Debug + PartialEq + Eq {
    /// Returns the next player in the game.
    fn next(self) -> Self;

    fn to_char(self) -> char {
        // Default implementation, can be overridden by specific players
        match self.into() {
            0 => '0',
            1 => '1',
            2 => '2',
            3 => '3',
            _ => '?', // Fallback for unexpected player numbers
        }
    }
}

pub trait MoveStore<M: Copy>: Default + Debug + Clone + IntoIterator<Item = M> {
    /// Clears the store of all moves.
    fn clear(&mut self);

    /// Adds a move to the store.
    fn push(&mut self, mv: M);

    /// Returns the number of moves in the store.
    fn len(&self) -> usize;

    /// Returns true if the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the moves.
    fn iter(&self) -> impl Iterator<Item = M>;
}

impl<M: Copy + Debug> MoveStore<M> for Vec<M> {
    fn clear(&mut self) {
        Vec::clear(self);
    }

    fn push(&mut self, mv: M) {
        Vec::push(self, mv);
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn iter(&self) -> impl Iterator<Item = M> {
        self.as_slice().iter().copied()
    }
}

pub trait Board: Clone + Debug + Send + Sync + Hash + Eq + PartialEq {
    type Move: Eq + Hash + Debug + Copy + Display;
    type Player: BoardPlayer;
    type MoveStore: MoveStore<Self::Move> + Send + Sync;
    type HashKey: Eq + Hash + Debug + Copy + Send + Sync;

    fn new(&self) -> Self;

    /// Experimental: Create different starting positions
    fn new_varied(&self) -> Self {
        self.new()
    }

    fn player_current(&self) -> Self::Player;

    /// Player_next return the next LOGICAL player, not the actual next player to move.
    fn player_next(&self) -> Self::Player {
        self.player_current().next()
    }

    fn player_num(&self) -> usize;

    fn player_num_of_active(&self) -> usize;

    /// Returns a fancy debug string for the board. By default, it's just the debug representation,
    /// however it's more intended to use colours and other fancy formatting.
    fn fancy_debug(&self) -> String {
        format!("{:?}", self)
    }

    /// Generates all legal moves for the current position and adds them to the
    /// provided `store`. The store is cleared before new moves are added.
    fn fill_move_store(&mut self, store: &mut Self::MoveStore);

    /// A convenience method that allocates a new move store and returns it.
    /// This is less efficient for performance-critical loops but easier to use.
    fn legal_moves(&mut self) -> Self::MoveStore {
        debug_assert!(
            !self.is_terminal(),
            "Cannot get legal moves on a terminal state."
        );

        let mut store = Self::MoveStore::default();
        self.fill_move_store(&mut store);
        store
    }

    fn play_move_clone(&self, mv: &Self::Move, move_store: &mut Self::MoveStore) -> Self {
        let mut new_board = self.clone();
        new_board.play_move_mut_with_store(mv, move_store, None);
        new_board
    }

    /// Plays a move on the board, modifying it in place.
    /// Since internally boards may need to generate legal moves for various reasons,
    /// we provide a move store to avoid repeated allocations and to avoid calling [`Board::fill_move_store`].
    /// The `next_player_hint` is an optional hint which may be used to skip some internal calculations
    /// if the next player is known in advance. Note that this will then not fill up the move store.
    /// For correctness, it should empty it.
    fn play_move_mut_with_store(
        &mut self,
        mv: &Self::Move,
        move_store: &mut Self::MoveStore,
        next_player_hint: Option<Self::Player>,
    );

    fn is_terminal(&self) -> bool;

    fn outcome(&self, forced_end: bool) -> Option<Outcome>;

    fn hash_key(&self) -> Self::HashKey;

    fn key_eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }

    /// E.g. material balance in chess.
    fn get_heuristic_vector(&self) -> Option<Vec<f32>> {
        None
    }

    /// E.g. in chess it doesn't make sense to continue the game, when there is no checkmate possible anymore.
    fn can_game_end_early(&self) -> bool {
        false
    }

    /// Auxiliary values that may be useful for the neural network.
    /// E.g. in chess Material balance.
    fn get_aux_values(&self, _is_absolute: bool) -> Option<Vec<f32>> {
        None
    }
}

pub trait InputMapper<B: Board>: Debug + Send + Sync {
    /// Shape of the input tensor for the board.
    /// Example for 3 player TTT: `[15, 4]` (15 positions, 1 hot encoding for each player and empty).
    fn input_board_shape(&self) -> [usize; 2];

    /// Encode this board.
    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &B);

    fn is_absolute(&self) -> bool;

    /// Defines how many of the initial input channels are fundamental for hashing.
    ///
    /// - Returning `None` (the default) indicates all channels should be hashed.
    /// - Returning `Some(n)` indicates that only the first `n` channels (from index 0 to n-1)
    ///   are fundamental to the state and should be hashed. This requires the mapper
    ///   to organize its channels accordingly.
    ///
    /// This is a significant optimization for mappers with derived features.
    fn num_hashable_channels(&self) -> Option<usize> {
        None
    }
}

pub trait ReverseInputMapper<B: Board>: Debug + Send + Sync {
    /// Decode the input tensor into a board.
    /// The input tensor is expected to be of shape [batch_size, board_size, field_size].
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, scalars: &Vec<f32>) -> B;
}

/// A way to encode and decode moves on a board into a tensor.
/// We could also bound the trait to Copy, however then we give up Vec and Hash...
pub trait PolicyMapper<B: Board>: Debug + Send + Sync + Clone {
    fn policy_len(&self) -> usize;

    /// Get the index in the policy tensor corresponding to the given move.
    fn move_to_index(&self, player: B::Player, mv: B::Move) -> usize;

    /// Get the move corresponding to the given index in the policy tensor.
    /// A return of `None` means that this index does not correspond to any move (on this board or otherwise).
    fn index_to_move(&self, board: &B, move_store: &B::MoveStore, index: usize) -> Option<B::Move>;
}

/// A trait for mapping the performance of a board game, such as average number of moves per game.
pub trait MetaPerformanceMapper<B: Board>: Debug + Send + Sync {
    fn average_number_of_moves(&self) -> usize;
}

/// Utility trait automatically implemented for anything that implements both [InputMapper] and [PolicyMapper].
pub trait BoardMapper<B: Board>: InputMapper<B> + PolicyMapper<B> {}

impl<B: Board, M: InputMapper<B> + PolicyMapper<B>> BoardMapper<B> for M {}

/// Utility trait automatically implemented for anything that implements both [BoardMapper] and [MetaPerformanceMapper].
pub trait MetaBoardMapper<B: Board>: BoardMapper<B> + MetaPerformanceMapper<B> {}

impl<B: Board, M: BoardMapper<B> + MetaPerformanceMapper<B>> MetaBoardMapper<B> for M {}

#[derive(Debug, Clone)]
pub struct OracleAnalysis<M: Copy> {
    /// The final outcome of the game, as an absolute value vector.
    pub outcome_abs: Vec<f32>,

    /// The single best move to make.
    pub best_move: M,

    pub outcome: Outcome,
}

#[derive(Debug)]
pub enum OracleError {
    NotApplicable, // Used when a position is not an endgame the oracle knows about.
}

/// This trait is designed to be used as a singleton (`Arc<dyn Oracle<B>>`) and shared
/// across all self-play threads.
pub trait Oracle<B: Board>: Send + Sync + Debug {
    /// Probes the tablebase for a given board position.
    ///
    /// If the position is recognized as a known endgame, it returns `Ok(OracleAnalysis)`.
    /// If the position is not a known endgame, it should return `Err(OracleError::NotApplicable)`.
    fn probe(&self, board: &B) -> Result<OracleAnalysis<B::Move>, OracleError>;
}

pub type OptionalSharedOracle<B> = Option<Arc<dyn Oracle<B>>>;
