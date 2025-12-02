use crate::basics::{Color, MemoryBuffer, Piece};
use crate::pos::{MemoryPos, QRSPos};
use std::fmt::{Debug, Display, Formatter};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveType {
    Move,
    DoublePawnPush,
    EnPassant(MemoryPos),
    Promotion(Piece),
    EnPassantPromotion(PassantWithPromotion),
    CastleKingSide,
    CastleQueenSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassantWithPromotion(u8);

const HEXES: [u8; 6] = [24, 31, 56, 63, 88, 95];

/// A struct that stores a passant with promotion move in a single byte
/// Important so it doesn't cause [`MoveType`] to be 1 byte larger
impl PassantWithPromotion {
    pub fn new(pos: MemoryPos, piece: Piece) -> Self {
        // passant with promotion can happen only on few specific squares
        debug_assert!(
            HEXES.contains(&pos.0),
            "Invalid en passant promotion square {:?}",
            pos
        );

        let index = HEXES.iter().position(|&x| x == pos.0).unwrap() as u8;

        // 0,1,2 bits = index, 3,4,5 bits = piece
        Self(index | ((piece as u8) << 3))
    }

    pub fn get(&self) -> (MemoryPos, Piece) {
        let index = self.0 & 0b111; // Get first 3 bits
        let piece = self.0 >> 3; // Get the 2nd 3 bits

        (
            MemoryPos::new(HEXES[index as usize]),
            Piece::from_u8(piece).unwrap(),
        )
    }
}

/// In total takes 4 bytes of memory with no alignment
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PseudoLegalMove {
    pub from: MemoryPos,
    pub to: MemoryPos,
    pub move_type: MoveType,
}

/// For notations refer to https://www.chessprogramming.org/Algebraic_Chess_Notation
impl PseudoLegalMove {
    /// Returns the move in Universal Chess Interface (UCI) notation.
    pub fn notation_uci(&self) -> String {
        self.notation_simple("")
    }

    /// Same as UCI but with '-' delimiter between from and to hexes.
    /// Similar to 'pure' notation on the Chess wiki.
    pub fn notation_hyphen(&self) -> String {
        self.notation_simple("-")
    }

    /// Returns the move in long algebraic notation (LAN).
    pub fn notation_lan(&self, board_buffer: &MemoryBuffer) -> String {
        let is_capture = !board_buffer[self.get_capture_slot()].is_empty();

        let piece_char = match board_buffer[self.from].get() {
            Some((_, piece)) => {
                if piece == Piece::Pawn {
                    None
                } else {
                    Some(piece.to_char().to_ascii_uppercase())
                }
            }
            None => None, // This should never happen
        };

        let capture_str = if is_capture { "x" } else { "-" };
        let piece_str = piece_char.map(|c| c.to_string()).unwrap_or_default();

        match self.move_type {
            MoveType::Promotion(piece) => format!(
                "{}{}{}{}{}",
                piece_str,
                self.from.get_uci_notation(),
                capture_str,
                self.to.get_uci_notation(),
                piece.to_char().to_ascii_uppercase()
            ),
            MoveType::EnPassantPromotion(wrapper) => {
                let (pos, piece) = wrapper.get();
                format!(
                    "{}{}{}{}{}",
                    piece_str,
                    self.from.get_uci_notation(),
                    capture_str,
                    pos.get_uci_notation(),
                    piece.to_char().to_ascii_uppercase()
                )
            }
            MoveType::CastleKingSide => "O-O".to_string(),
            MoveType::CastleQueenSide => "O-O-O".to_string(),
            _ => format!(
                "{}{}{}{}",
                piece_str,
                self.from.get_uci_notation(),
                capture_str,
                self.to.get_uci_notation()
            ),
        }
    }

    pub fn notation_simple(&self, delimiter: &str) -> String {
        match self.move_type {
            MoveType::Promotion(piece) => format!(
                "{}{}{}{}",
                self.from.get_uci_notation(),
                delimiter,
                self.to.get_uci_notation(),
                piece.to_char().to_ascii_lowercase()
            ),
            MoveType::EnPassantPromotion(wrapper) => {
                let (pos, piece) = wrapper.get();
                format!(
                    "{}{}{}{}",
                    self.from.get_uci_notation(),
                    delimiter,
                    pos.get_uci_notation(),
                    piece.to_char().to_ascii_lowercase()
                )
            }
            _ => format!(
                "{}{}{}",
                self.from.get_uci_notation(),
                delimiter,
                self.to.get_uci_notation()
            ),
        }
    }

    /// Returns potential capture position for the move
    pub fn get_capture_slot(&self) -> MemoryPos {
        match self.move_type {
            MoveType::EnPassant(pos) => pos,
            MoveType::EnPassantPromotion(wrapper) => wrapper.get().0,
            _ => self.to,
        }
    }

    pub fn get_promotion_piece(&self) -> Option<Piece> {
        match self.move_type {
            MoveType::Promotion(piece) => Some(piece),
            MoveType::EnPassantPromotion(wrapper) => Some(wrapper.get().1),
            _ => None,
        }
    }
}

impl Debug for PseudoLegalMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Move")
            .field("from", &self.from.0)
            .field("to", &self.to.0)
            .field("move_type", &self.move_type)
            .field("pure", &self.notation_hyphen())
            .finish()
    }
}

impl Display for PseudoLegalMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.notation_hyphen())
    }
}

const MAX_PSEUDO_MOVES: usize = 384; // Arbitrary limit, should be enough for any position

/// Information about the current turn, that needs to be re-calculated every turn
/// Used in pseudo move generation and should only be used in that context!
#[derive(Default, Clone, Copy, Debug)]
pub struct TurnCache {
    pub turn: Color,
    pub king_pos: Option<QRSPos>,
    pub is_check: bool,
    pub grace_period: bool,
}

/// An efficient store for pseudo-legal moves
/// Instead of using a Vec, which would require heap allocation, we use a fixed-size array
/// This is a trade-off, as we can only store a limited number of moves, but it is much faster
/// 384 moves should be enough (?)
#[derive(Clone, Copy)]
#[cfg_attr(feature = "wasm", wasm_bindgen(js_name = ChessMoveStore))]
pub struct ChessMoveStore {
    list: [Option<PseudoLegalMove>; MAX_PSEUDO_MOVES],
    count: u16,
    capacity: u16,
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    pub turn_cache: TurnCache,
}

impl Default for ChessMoveStore {
    fn default() -> Self {
        Self {
            list: [const { None }; MAX_PSEUDO_MOVES],
            count: 0,
            capacity: MAX_PSEUDO_MOVES as u16,
            turn_cache: TurnCache::default(),
        }
    }
}

impl ChessMoveStore {
    pub fn clear(&mut self) {
        self.count = 0;
    }

    pub fn push(&mut self, pseudo_move: PseudoLegalMove) {
        if self.count >= self.capacity {
            panic!("PseudoMoveStore is full");
        }

        self.list[self.count as usize] = Some(pseudo_move);
        self.count += 1;
    }

    pub fn get(&self, index: usize) -> Option<PseudoLegalMove> {
        debug_assert!(index < self.count as usize, "Index out of bounds");

        self.list[index]
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn iter(&self) -> impl Iterator<Item = &PseudoLegalMove> {
        self.list
            .iter()
            .take(self.count as usize) // Only iterate over the valid items
            .filter_map(Option::as_ref) // Convert `&Option<T>` to `Option<&T>` and flatten
    }

    pub fn contains(&self, pseudo_move: PseudoLegalMove) -> bool {
        self.iter().any(|m| *m == pseudo_move)
    }
}

impl Debug for ChessMoveStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChessMoveStore")
            .field("count", &self.count)
            .field("capacity", &self.capacity)
            .field("moves", &self.iter().collect::<Vec<_>>())
            .finish()
    }
}

// A struct for our immutable iterator.
// The lifetime 'a ensures it doesn't outlive the ChessMoveStore it borrows from.
pub struct MoveStoreIter<'a> {
    store: &'a ChessMoveStore,
    index: usize,
}

impl<'a> Iterator for MoveStoreIter<'a> {
    type Item = &'a PseudoLegalMove;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.store.count as usize {
            // Get the item, unwrap it (we know it's Some), and advance the index
            let item = self.store.list[self.index].as_ref();
            self.index += 1;
            item
        } else {
            None
        }
    }
}

// Implement the trait for &ChessMoveStore
impl<'a> IntoIterator for &'a ChessMoveStore {
    type Item = &'a PseudoLegalMove;
    type IntoIter = MoveStoreIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        MoveStoreIter {
            store: self,
            index: 0,
        }
    }
}

pub struct MoveStoreIntoIter {
    store: ChessMoveStore,
    index: usize,
}

impl Iterator for MoveStoreIntoIter {
    type Item = PseudoLegalMove;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.store.count as usize {
            // Take the value out of the Option, leaving None behind
            let item = self.store.list[self.index].take();
            self.index += 1;
            item
        } else {
            None
        }
    }
}

// Implement the trait for ChessMoveStore itself
impl IntoIterator for ChessMoveStore {
    type Item = PseudoLegalMove;
    type IntoIter = MoveStoreIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        MoveStoreIntoIter {
            store: self,
            index: 0,
        }
    }
}
