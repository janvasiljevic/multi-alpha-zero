pub use crate::pos::{MemoryPos, QRSPos};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Index, IndexMut};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

/// Color of the Player. Called 'Color' instead of 'Player' to
/// clearly differentiate from the 'PieceType' enum (same first letter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub enum Color {
    #[default]
    White = 0b00,
    Gray = 0b01,
    Black = 0b10,
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Color::White => 'W',
                Color::Gray => 'G',
                Color::Black => 'B',
            }
        )
    }
}

impl From<Color> for usize {
    fn from(color: Color) -> Self {
        match color {
            Color::White => 0,
            Color::Gray => 1,
            Color::Black => 2,
        }
    }
}

impl From<usize> for Color {
    fn from(value: usize) -> Self {
        match value {
            0 => Color::White,
            1 => Color::Gray,
            2 => Color::Black,
            _ => panic!("Invalid player index: {}", value),
        }
    }
}

/// Array of all possible colors. Iterating over this array yield 0, 1, 2
pub const COLORS: [Color; 3] = [Color::White, Color::Gray, Color::Black];

/// Chess piece representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub enum Piece {
    Pawn = 0b001, // Start with 1 to avoid empty slots
    Knight = 0b010,
    Bishop = 0b011,
    Rook = 0b100,
    Queen = 0b101,
    King = 0b110,
}

/// A Memory slot is an 8 bit value that represents a player and a piece: CC00 0PPP,
/// where CC is the [`Color`] (player) and PPP is the [`Piece`] type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MemorySlot(u8);

/// MemoryBuffer is a 96 slot array of [`MemorySlot`]s. In total of 768 bits of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct MemoryBuffer([MemorySlot; 96]);

impl MemorySlot {
    /// Create an empty slot
    pub fn empty() -> Self {
        Self(0)
    }

    /// Check if the slot is occupied by a player and piece
    pub fn is_player(&self, player: Color) -> bool {
        !self.is_empty() && self.player() == Some(player)
    }

    /// Check if the slot is occupied by an enemy player and piece
    pub fn is_enemy(&self, player: Color) -> bool {
        !self.is_empty() && self.player() != Some(player)
    }

    pub fn new(player: Color, piece: Piece) -> Self {
        Self(((player as u8) << 3) | (piece as u8))
    }

    /// Set the player and piece
    pub fn set(&mut self, player: Color, piece: Piece) {
        self.0 = ((player as u8) << 3) | (piece as u8);
    }

    pub fn get(&self) -> Option<(Color, Piece)> {
        if self.is_empty() {
            None
        } else {
            Some((self.player().unwrap(), self.piece().unwrap()))
        }
    }

    /// Check if the slot is empty
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Get the player (if not empty)
    pub fn player(&self) -> Option<Color> {
        if self.is_empty() {
            None
        } else {
            Color::from_u8(self.0 >> 3)
        }
    }

    /// Get the piece type (if not empty)
    pub fn piece(&self) -> Option<Piece> {
        if self.is_empty() {
            None
        } else {
            Piece::from_u8(self.0 & 0b111)
        }
    }

    pub fn set_piece(&mut self, piece: Piece) {
        self.0 = (self.0 & 0b11000) | (piece as u8);
    }

    /// Checks if memory slots matches the player and piece
    pub fn matches(&self, player: Color, piece: Piece) -> bool {
        self.0 == ((player as u8) << 3) | (piece as u8)
    }
}

impl Debug for MemorySlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "Empty")
        } else {
            write!(
                f,
                "{:?} {:?}",
                self.player().unwrap(),
                self.piece().unwrap()
            )
        }
    }
}

impl Default for MemoryBuffer {
    fn default() -> Self {
        Self([MemorySlot::empty(); 96])
    }
}

impl Index<usize> for MemoryBuffer {
    type Output = MemorySlot;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<MemoryPos> for MemoryBuffer {
    type Output = MemorySlot;

    fn index(&self, index: MemoryPos) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl IndexMut<MemoryPos> for MemoryBuffer {
    fn index_mut(&mut self, index: MemoryPos) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

impl IndexMut<usize> for MemoryBuffer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl MemoryBuffer {
    pub fn king_qrs_local(&self, color: Color) -> Option<QRSPos> {
        self.king_mem_pos(color).map(|pos| pos.to_qrs_local(color))
    }

    pub fn clear(&mut self) {
        for slot in self.0.iter_mut() {
            *slot = MemorySlot::empty();
        }
    }

    pub fn remove_player_pieces(&mut self, player: Color) {
        for slot in self.0.iter_mut() {
            if slot.player() == Some(player) {
                *slot = MemorySlot::empty();
            }
        }
    }

    pub fn king_mem_pos(&self, color: Color) -> Option<MemoryPos> {
        for i in 0..96 {
            if self[i].matches(color, Piece::King) {
                return Some(MemoryPos(i as u8));
            }
        }

        None
    }

    pub fn non_empty_iter(&self) -> impl Iterator<Item = (MemoryPos, MemorySlot)> + '_ {
        self.0.iter().enumerate().filter_map(|(i, &slot)| {
            if slot.is_empty() {
                None
            } else {
                Some((MemoryPos(i as u8), slot))
            }
        })
    }

    #[inline]
    pub fn get_occupied_bitboard(&self) -> u128 {
        let mut bitboard = 0u128;
        for i in 0..96 {
            if !self.0[i].is_empty() {
                // Set the i-th bit to 1.
                bitboard |= 1u128 << i;
            }
        }
        bitboard
    }
}

impl Color {
    pub fn left_player(&self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Gray => Color::White,
            Color::Black => Color::Gray,
        }
    }

    pub fn right_player(&self) -> Color {
        match self {
            Color::White => Color::Gray,
            Color::Gray => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0b00 => Some(Color::White),
            0b01 => Some(Color::Gray),
            0b10 => Some(Color::Black),
            _ => None,
        }
    }

    pub fn to_fen(self) -> char {
        match self {
            Color::White => 'W',
            Color::Gray => 'G',
            Color::Black => 'B',
        }
    }

    pub const fn get_offset(&self) -> u8 {
        match self {
            Color::White => 0,
            Color::Gray => 32,
            Color::Black => 64,
        }
    }
}

impl Piece {
    pub fn get_as_zero_based_index(self) -> u8 {
        self as u8 - 1
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0b001 => Some(Piece::Pawn),
            0b010 => Some(Piece::Knight),
            0b011 => Some(Piece::Bishop),
            0b100 => Some(Piece::Rook),
            0b101 => Some(Piece::Queen),
            0b110 => Some(Piece::King),
            _ => None,
        }
    }
    pub const fn material(&self) -> u8 {
        match self {
            Piece::Pawn => 1,
            Piece::Knight => 3,
            Piece::Bishop => 3,
            Piece::Rook => 5,
            Piece::Queen => 9,
            Piece::King => 0,
        }
    }

    pub const fn to_char(self) -> char {
        match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        }
    }

    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'p' => Some(Self::Pawn),
            'n' => Some(Self::Knight),
            'b' => Some(Self::Bishop),
            'r' => Some(Self::Rook),
            'q' => Some(Self::Queen),
            'k' => Some(Self::King),
            _ => None,
        }
    }

    pub const fn all() -> [Piece; 6] {
        [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ]
    }
}

/// State of the players in the game. Who's turn is it and which players are still in the game.
///
/// Structured as 0bCC 'WGB' 'WGB':
/// - CC is the current player [`Color`] (so 00 for White, 01 for Gray, 10 for Black)
/// - Two sets of 3 bits for each player: WGB, where W is White, G is Gray and B is Black
///     - The first set is dedicated to "stale" players (players with 0 legal moves)
///     - The second set is dedicated to "active" players (players still in the game)
///
/// The `STALEMATE` const parameter is used to differentiate between normal game state and stalemate state.
/// Stalemate state has the same structure, but doesn't contain the STALE bits, since the type
/// information already informs that the game is in stalemate.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct PlayerState<const STALEMATE: bool>(pub u8);

pub const ACTIVE_BITS: u8 = 0b0000111; // Bits 0-2 for active players
pub const STALE_BITS: u8 = 0b00111000; // Bits 3-5 for stale players
pub const TURN_BITS: u8 = 0b11000000; // Bits 6-7 for turn
pub const NON_TURN_BITS: u8 = !TURN_BITS; // Equal to 0b00111111 (or ACTIVE_BITS | STALE_BITS)

pub type NormalState = PlayerState<false>;
pub type DrawState = PlayerState<true>;


impl DrawState {
   pub fn get_drawn_players(&self) -> Vec<Color> {
        let mut drawn_players = Vec::new();
        for color in COLORS {
            if self.is_present(color) {
                drawn_players.push(color);
            }
        }
        drawn_players
    }
    
    pub fn get_eliminated_players(&self) -> Vec<Color> {
        let mut eliminated_players = Vec::new();
        for color in COLORS {
            if !self.is_present(color) {
                eliminated_players.push(color);
            }
        }
        eliminated_players
    }
}

impl NormalState {
    /// Creates a new player state with the given turn and all players present.
    pub fn new_with_turn(turn: Color) -> Self {
        PlayerState::<false>::new(turn, true, true, true)
    }

    /// Creates a stalemate structure, that contains turn information and which players are still in the game.
    pub fn to_stalemate(self, turn: Color) -> DrawState {
        PlayerState::<true>(self.0 & ACTIVE_BITS | (turn as u8) << 6)
    }

    /// Gets the next player that is still in the game. Returns a tuple:
    /// .0 = should the turn counter be advanced (if the [Color::White] player was reached)
    /// .1 = the next player
    pub fn get_next_turn(&self) -> (bool, Color) {
        debug_assert!(
            self.is_present(Color::White) as u8
                + self.is_present(Color::Gray) as u8
                + self.is_present(Color::Black) as u8
                >= 1,
            "At least one player must be in the game"
        );

        let mut color = self.get_turn();

        let mut advance_turn_counter = false;
        loop {
            color = color.right_player();
            advance_turn_counter |= color == Color::White;
            if self.is_present(color) {
                return (advance_turn_counter, color);
            }
        }
    }

    /// Returns true if all other present players are stale
    pub fn are_other_players_stale(&self, color: Color) -> bool {
        for other_color in COLORS {
            if other_color != color && self.is_present(other_color) && !self.is_stale(other_color) {
                return false; // Found a present player who is NOT stale
            }
        }
        true // All present players (except `color`) are stale
    }

    pub fn set_stale(&mut self, color: Color) {
        self.0 |= 1 << (color as u8 + 3);
    }

    pub fn clear_all_stale(&mut self) {
        self.0 &= !STALE_BITS;
    }

    pub fn is_stale(&self, color: Color) -> bool {
        self.0 & (1 << (color as u8 + 3)) != 0
    }

    pub fn remove_stale(&mut self, color: Color) {
        self.0 &= !(1 << (color as u8 + 3));
    }
}

impl<const STALEMATE: bool> PlayerState<STALEMATE> {
    pub fn get_turn(&self) -> Color {
        Color::from_u8(self.0 >> 6).unwrap()
    }

    pub fn set_turn(&mut self, color: Color) {
        self.0 = (self.0 & (NON_TURN_BITS)) | ((color as u8) << 6);
    }

    pub fn is_present(&self, color: Color) -> bool {
        self.0 & (1 << (color as u8)) != 0
    }

    pub fn set_player(&mut self, color: Color) {
        self.0 |= 1 << (color as u8);
    }

    pub fn remove_player(&mut self, color: Color) {
        self.0 &= !(1 << (color as u8));
    }

    pub fn get_player_count(&self) -> u8 {
        self.is_present(Color::White) as u8
            + self.is_present(Color::Gray) as u8
            + self.is_present(Color::Black) as u8
    }

    pub fn new(turn: Color, white: bool, gray: bool, black: bool) -> Self {
        let mut state = PlayerState(0);

        state.set_turn(turn);
        if white {
            state.set_player(Color::White);
        }
        if gray {
            state.set_player(Color::Gray);
        }
        if black {
            state.set_player(Color::Black);
        }

        state
    }
}

impl<const STALEMATE: bool> Display for PlayerState<STALEMATE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self.get_turn() {
                Color::White => 'W',
                Color::Gray => 'G',
                Color::Black => 'B',
            }
        )
    }
}

impl Debug for NormalState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Turn: {:?} Stalemate[W: {} | G: {} | B: {}], Present[W: {} | G: {} | B: {}]",
            self.get_turn(),
            self.is_stale(Color::White),
            self.is_stale(Color::Gray),
            self.is_stale(Color::Black),
            self.is_present(Color::White),
            self.is_present(Color::Gray),
            self.is_present(Color::Black),
        )
    }
}

impl Debug for DrawState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Last turn was {:?}, Players present: [W: {} | G: {} | B: {}]",
            self.get_turn(),
            self.is_present(Color::White),
            self.is_present(Color::Gray),
            self.is_present(Color::Black),
        )
    }
}

/// Represents the flags for castling
/// The u8 is split into 4 2 bit sections: <Empty> <Black> <Gray> <White>
/// Each section represents bit flags as: <QueenSide> <KingSide>
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct CastleFlags(u8);

impl CastleFlags {
    pub const fn new(u8: u8) -> Self {
        CastleFlags(u8)
    }

    pub fn can_queen_side(&self, player: Color) -> bool {
        self.0 & (1 << ((player as u8) * 2 + 1)) != 0
    }

    pub fn can_king_side(&self, player: Color) -> bool {
        self.0 & (1 << ((player as u8) * 2)) != 0
    }

    pub fn remove_all(&mut self, player: Color) {
        self.0 &= !(0b11 << ((player as u8) * 2));
    }

    pub fn set_queen_side(&mut self, player: Color, can_castle: bool) {
        if can_castle {
            self.0 |= 1 << ((player as u8) * 2 + 1);
        } else {
            self.0 &= !(1 << ((player as u8) * 2 + 1));
        }
    }

    pub fn set_king_side(&mut self, player: Color, can_castle: bool) {
        if can_castle {
            self.0 |= 1 << ((player as u8) * 2);
        } else {
            self.0 &= !(1 << ((player as u8) * 2));
        }
    }
}

/// Memory posses for en passant are stored in the EnPassantState
/// Presented in string format: Range from 1 to 8 (inclusive), or None if no en passant is available
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct EnPassantState([Option<MemoryPos>; 3]);

impl Display for CastleFlags {
    /// Triple of 2 characters for each player
    /// The first character is 'q' if the player can castle queen side, '-' otherwise
    /// The second character is 'k' if the player can castle king side, '-' otherwise
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let colors = [Color::White, Color::Gray, Color::Black];
        let mut chars = ['-'; 6];

        for (i, &color) in colors.iter().enumerate() {
            chars[i * 2] = if self.can_queen_side(color) { 'q' } else { '-' };
            chars[i * 2 + 1] = if self.can_king_side(color) { 'k' } else { '-' };
        }

        write!(f, "{}", chars.iter().collect::<String>())
    }
}
impl EnPassantState {
    /// Used in testing. 0 = None, 1-8 = Local rank
    #[cfg(test)]
    pub fn new_from_test(files: [u8; 3]) -> Self {
        let mut en_passant = EnPassantState([None; 3]);

        for color in COLORS {
            if files[color as usize] != 0 {
                let pos = MemoryPos(files[color as usize] - 1 + color.get_offset() + 16);
                en_passant.set_pos(color, pos);
            }
        }

        en_passant
    }

    /// Set the en passant position for a player.
    /// Must be a valid rank ('second pawn row' for the corresponding player)
    pub fn set_pos(&mut self, color: Color, pos: MemoryPos) {
        debug_assert!(
            match color {
                Color::White => 16 <= pos.0 && pos.0 < 24,
                Color::Gray => 48 <= pos.0 && pos.0 < 56,
                Color::Black => 80 <= pos.0 && pos.0 < 88,
            },
            "Invalid en passant position: {} for {:?}",
            pos.0,
            color
        );

        self.0[color as usize] = Some(pos)
    }

    pub fn remove(&mut self, color: Color) {
        self.0[color as usize] = None
    }

    pub fn get(&self, color: Color) -> Option<MemoryPos> {
        self.0[color as usize]
    }

    pub fn get_char(&self, color: Color) -> char {
        self.0[color as usize].map_or('-', |x| ((x.0 - color.get_offset() - 16) + b'1') as char)
    }
}

impl Display for EnPassantState {
    // --- if no en passant is available
    // 3-- if en passant is available on rank 3 for W
    // 888 if all en passant is available for all players on rank 8
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}",
            self.get_char(Color::White),
            self.get_char(Color::Gray),
            self.get_char(Color::Black)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_passant_test() {
        let mut en_passant = EnPassantState::default();
        en_passant.set_pos(Color::White, MemoryPos(16));
        en_passant.set_pos(Color::Gray, MemoryPos(48));
        en_passant.set_pos(Color::Black, MemoryPos(80));

        assert_eq!(en_passant.get(Color::White), Some(MemoryPos(16)));
        assert_eq!(en_passant.get(Color::Gray), Some(MemoryPos(48)));
        assert_eq!(en_passant.get(Color::Black), Some(MemoryPos(80)));

        let fen_string = en_passant.to_string();

        assert_eq!(fen_string, "111");

        en_passant.remove(Color::White);

        let fen_string = en_passant.to_string();

        assert_eq!(fen_string, "-11");

        en_passant.set_pos(Color::White, MemoryPos(23));
        en_passant.remove(Color::Gray);
        en_passant.set_pos(Color::Black, MemoryPos(87));

        let fen_string = en_passant.to_string();

        assert_eq!(fen_string, "8-8");
    }

    #[test]
    fn test_memory_slot() {
        let mut slot = MemorySlot::empty();
        slot.set(Color::White, Piece::Pawn);

        assert_eq!(slot.player(), Some(Color::White));
        assert_eq!(slot.piece(), Some(Piece::Pawn));

        let slot = MemorySlot::empty();
        assert_eq!(slot.player(), None);
        assert_eq!(slot.piece(), None);
    }

    #[test]
    fn castle_test() {
        let mut flags = CastleFlags::default();

        for color in COLORS {
            assert!(!flags.can_king_side(color));
            assert!(!flags.can_queen_side(color));
        }

        // Gray and White can castle queen side
        flags = CastleFlags(0b00001010);

        // White
        assert!(!flags.can_king_side(Color::White));
        assert!(flags.can_queen_side(Color::White));
        // Gray
        assert!(!flags.can_king_side(Color::Gray));
        assert!(flags.can_queen_side(Color::Gray));
        // Black
        assert!(!flags.can_king_side(Color::Black));
        assert!(!flags.can_queen_side(Color::Black));

        // Everybody can castle
        flags = CastleFlags(0b00111111);

        for color in COLORS {
            assert!(flags.can_king_side(color));
            assert!(flags.can_queen_side(color));
        }

        flags = CastleFlags(0b00000000);

        flags.set_king_side(Color::White, true);
        flags.set_queen_side(Color::Gray, true);
        flags.set_king_side(Color::Black, true);
    }

    #[test]
    fn test_player_state() {
        let mut state = PlayerState::new(Color::White, true, true, true);

        assert_eq!(state.get_turn(), Color::White);
        assert!(state.is_present(Color::White));
        assert!(state.is_present(Color::Gray));
        assert!(state.is_present(Color::Black));

        state.set_turn(Color::Gray);
        assert_eq!(state.get_turn(), Color::Gray);

        state.remove_player(Color::White);
        assert!(!state.is_present(Color::White));
        assert!(state.is_present(Color::Gray));
        assert!(state.is_present(Color::Black));

        let (advance_next_turn, next_color) = state.get_next_turn();
        assert_eq!(next_color, Color::Black);
        assert_eq!(advance_next_turn, false);

        state.0 = 0b00111111;

        for color in COLORS {
            assert!(state.is_present(color));
            assert!(state.is_stale(color));
        }

        assert_eq!(state.get_turn(), Color::White);

        state.clear_all_stale();

        for color in COLORS {
            assert!(state.is_present(color));
            assert!(!state.is_stale(color));
        }
    }
}
