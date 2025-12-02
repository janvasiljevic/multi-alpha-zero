use crate::basics::{CastleFlags, Color, MemoryBuffer, MemoryPos, MemorySlot, NormalState, Piece, COLORS};
use crate::check_information::CheckInformation;
use crate::chess_game::{State, TriHexChess};
use crate::fen::to_fen;
use crate::moves::{ChessMoveStore, MoveType, PassantWithPromotion, PseudoLegalMove};
use crate::phase::Phase;
use crate::phase::Phase::Normal;
use crate::pos::FullCoordinates;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[cfg(debug_assertions)]
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[allow(dead_code)]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[allow(unused_variables)]
fn debug_log(s: &str) {
    #[cfg(debug_assertions)]
    {
        log(s);
    }
}

/// Lightweight move info for UI hover/highlighting
#[wasm_bindgen(js_name = PatternHighlight)]
pub struct PatternHighlight {
    pub to: FullCoordinates,
    pub move_type: WebMoveType,
}

/// All possible move types
#[wasm_bindgen(js_name = MoveType)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebMoveType {
    Move,
    DoublePawnPush,
    Capture,
    EnPassant,
    EnPassantPromotion,
    Promotion,
    CapturePromotion,
    CastleKingSide,
    CastleQueenSide,
}

#[wasm_bindgen(inspectable, js_name = Castling)]
#[derive(Clone, Debug)]
pub struct Castling {
    pub can_castle_king_side: bool,
    pub can_castle_queen_side: bool,
}

#[wasm_bindgen]
impl Castling {
    #[wasm_bindgen(constructor)]
    pub fn new(can_castle_king_side: bool, can_castle_queen_side: bool) -> Self {
        Castling {
            can_castle_king_side,
            can_castle_queen_side,
        }
    }
}

#[wasm_bindgen(js_name = UnvalidatedBoard)]
#[derive(Clone, Debug, Default)]
pub struct UnvalidatedBoard {
    pieces: Vec<ChessPiece>,
    turn: Color,
    en_passant_white: Option<u8>,
    en_passant_gray: Option<u8>,
    en_passant_black: Option<u8>,
    castlings: Vec<Castling>,
    turn_counter: u32,
    third_move_count: u32,
}

#[wasm_bindgen]
impl UnvalidatedBoard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_pieces(&mut self, pieces: Vec<ChessPiece>) {
        self.pieces = pieces;
    }

    pub fn set_turn(&mut self, turn: Color) {
        self.turn = turn;
    }

    pub fn set_all_en_passant(&mut self, white: Option<u8>, gray: Option<u8>, black: Option<u8>) {
        self.en_passant_white = white;
        self.en_passant_gray = gray;
        self.en_passant_black = black;
    }

    pub fn set_turn_counter(&mut self, turn_counter: u32) {
        self.turn_counter = turn_counter;
    }

    pub fn set_third_move_count(&mut self, third_move_count: u32) {
        self.third_move_count = third_move_count;
    }

    pub fn set_castlings(&mut self, castlings: Vec<Castling>) {
        self.castlings = castlings;
    }

    /// Converts the unvalidated board into a FEN string
    pub fn to_fen(&self) -> Result<String, String> {
        if self.castlings.len() != 3 {
            return Err("Castlings must have exactly 3 entries, one for each player".to_string());
        }

        if self.pieces.len() > 96 {
            return Err("Too many pieces on the board".to_string());
        }

        let mut castle_flags = CastleFlags::default();

        for (i, castling) in self.castlings.iter().enumerate() {
            let color = COLORS[i];

            castle_flags.set_king_side(color, castling.can_castle_king_side);
            castle_flags.set_queen_side(color, castling.can_castle_queen_side);
        }

        let buffer = {
            let mut buffer = MemoryBuffer::default();

            for piece in &self.pieces {
                let pos = piece.coordinates.to_memory_pos();

                if !buffer[pos].is_empty() {
                    return Err(format!(
                        "Multiple pieces on the same square: {:?}",
                        piece.coordinates
                    ));
                }

                buffer[pos] = MemorySlot::new(piece.player, piece.piece);
            }

            buffer
        };

        let mut player_presence = [false; 3];

        for color in COLORS {
            let mut has_king = false;
            let mut has_other_piece = false;

            for (_, slot) in buffer.non_empty_iter() {
                if slot.player().unwrap() == color {
                    if slot.piece().unwrap() == Piece::King {
                        has_king = true;
                    } else {
                        has_other_piece = true;
                    }
                }
            }

            if !has_king && has_other_piece {
                return Err(format!("Player {:?} has pieces but no king", color));
            }

            player_presence[color as usize] = has_king;
        }

        if !player_presence[self.turn as usize] {
            return Err(format!(
                "It's {:?}'s turn but they have no king on the board",
                self.turn
            ));
        }

        let mut normal_state = NormalState::default();

        for color in COLORS {
            if player_presence[color as usize] {
                normal_state.set_player(color);
            }
        }

        normal_state.set_turn(self.turn);

        let mut board = TriHexChess {
            state: State {
                buffer,
                phase: Normal(normal_state),
                castle: castle_flags,
                en_passant: Default::default(),
                third_move: self.third_move_count as u16,
                turn_counter: self.turn_counter as u16,
            },
            is_using_grace_period: false,
            zobrist_hash: 0, // Set as last
            repetition_history: Default::default(),
        };

        board.zobrist_hash = board.calculate_full_hash();

        Ok(to_fen(&board.state))
    }
}

/// Represents a piece on the board
#[wasm_bindgen(inspectable)]
#[derive(Clone, Copy, Debug)]
pub struct ChessPiece {
    pub piece: Piece,
    pub player: Color,
    pub coordinates: FullCoordinates,
}

#[wasm_bindgen]
impl ChessPiece {
    #[wasm_bindgen(constructor)]
    pub fn new_piece(piece: Piece, player: Color, memory_index: u8) -> Self {
        ChessPiece {
            piece,
            player,
            coordinates: FullCoordinates::from_raw_index(memory_index),
        }
    }
}

#[wasm_bindgen(js_name = MoveWrapper)]
#[derive(Clone, Debug)]
pub struct MoveWrapper {
    pub from: FullCoordinates,
    pub to: FullCoordinates,
    pub move_type: WebMoveType,
    pub color: Color,
    pub piece: Piece,

    notation_lan: String,

    /// Specifies the coordinates of the pawn that can be captured by en passant
    #[wasm_bindgen(readonly)]
    en_passant: Option<FullCoordinates>,
}

/// Represents the material value for each player
/// Queen: 9, Rook: 5, Bishop: 3, Knight: 3, Pawn: 1, King has no value
/// Starting material value for each player: 39
#[wasm_bindgen(js_name = MaterialCounter)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WebMaterialCounter {
    pub white: u8,
    pub gray: u8,
    pub black: u8,
}

#[wasm_bindgen]
impl MoveWrapper {
    fn from_move(m: &PseudoLegalMove, buffer: &MemoryBuffer) -> Self {
        let is_capture = !buffer[m.to].is_empty();

        MoveWrapper {
            from: FullCoordinates::from_memory_pos(m.from),
            to: FullCoordinates::from_memory_pos(m.to),
            color: buffer[m.from].player().unwrap(),
            piece: buffer[m.from].piece().unwrap(),
            move_type: match m.move_type {
                MoveType::Move => {
                    if is_capture {
                        WebMoveType::Capture
                    } else {
                        WebMoveType::Move
                    }
                }
                MoveType::Promotion(_) => {
                    if is_capture {
                        WebMoveType::CapturePromotion
                    } else {
                        WebMoveType::Promotion
                    }
                }
                MoveType::EnPassantPromotion(_) => WebMoveType::EnPassantPromotion,
                MoveType::EnPassant(_) => WebMoveType::EnPassant,
                MoveType::CastleKingSide => WebMoveType::CastleKingSide,
                MoveType::CastleQueenSide => WebMoveType::CastleQueenSide,
                MoveType::DoublePawnPush => WebMoveType::DoublePawnPush,
            },
            notation_lan: m.notation_lan(&buffer),
            en_passant: match m.move_type {
                MoveType::EnPassant(pos) => Some(FullCoordinates::from_memory_pos(pos)),
                MoveType::EnPassantPromotion(passant_promotion) => {
                    let (pos, _) = passant_promotion.get();
                    Some(FullCoordinates::from_memory_pos(pos))
                }
                _ => None,
            },
        }
    }

    fn to_pseudo_move(&self, promotion: Option<Piece>) -> PseudoLegalMove {
        PseudoLegalMove {
            from: self.from.to_memory_pos(),
            to: self.to.to_memory_pos(),
            move_type: match self.move_type {
                WebMoveType::Move => MoveType::Move,
                WebMoveType::DoublePawnPush => MoveType::DoublePawnPush,
                WebMoveType::Capture => MoveType::Move,
                WebMoveType::EnPassant => {
                    MoveType::EnPassant(self.en_passant.unwrap().to_memory_pos())
                }
                WebMoveType::Promotion => MoveType::Promotion(promotion.unwrap()),
                WebMoveType::CapturePromotion => MoveType::Promotion(promotion.unwrap()),
                WebMoveType::CastleKingSide => MoveType::CastleKingSide,
                WebMoveType::CastleQueenSide => MoveType::CastleQueenSide,
                WebMoveType::EnPassantPromotion => {
                    MoveType::EnPassantPromotion(PassantWithPromotion::new(
                        self.en_passant.unwrap().to_memory_pos(),
                        promotion.unwrap(),
                    ))
                }
            },
        }
    }

    fn is_promotion(&self) -> bool {
        matches!(
            self.move_type,
            WebMoveType::Promotion
                | WebMoveType::CapturePromotion
                | WebMoveType::EnPassantPromotion
        )
    }

    #[wasm_bindgen(getter, js_name = getNotationLAN)]
    pub fn notation_lan(&self) -> String {
        self.notation_lan.clone()
    }
}

#[wasm_bindgen(js_name = GameState)]
pub struct GameStateWeb {
    /// Who's turn it is
    pub turn: Color,
    /// Who won the game, undefined if the game is ongoing
    pub won: Option<Color>,
    /// If the game is a stalemate
    pub is_stalemate: bool,
    /// The number of turns that have passed
    pub turn_counter: u32,
    /// The number of 'third moves' since the last capture or pawn move
    pub third_move_count: u32,
}

#[wasm_bindgen(js_name = TriHexChessWrapper)]
pub struct TriHexChessWrapper {
    #[wasm_bindgen(skip)]
    pub inner: TriHexChess,

    #[wasm_bindgen(skip)]
    pub move_store: ChessMoveStore,

    #[wasm_bindgen(skip)]
    pub cached_stores: [Option<ChessMoveStore>; 3],
}

#[wasm_bindgen]
impl TriHexChessWrapper {
    /// Create a new game from a FEN string
    #[wasm_bindgen(js_name = new)]
    pub fn from_fen_web(fen: &str) -> Result<TriHexChessWrapper, String> {
        set_panic_hook();

        if !fen.is_ascii() {
            return Err("FEN must be ASCII".to_string());
        }

        let bytes = fen.as_bytes();

        let mut game = TriHexChess::new_with_fen(bytes, false)?;

        let mut move_store = ChessMoveStore::default();

        game.update_pseudo_moves(&mut move_store, true);

        Ok(TriHexChessWrapper {
            inner: game,
            move_store,
            cached_stores: [None, None, None],
        })
    }

    #[wasm_bindgen(js_name = newDefault)]
    pub fn new_default_web() -> TriHexChessWrapper {
        set_panic_hook();

        let mut game = TriHexChess::default();
        let mut move_store = ChessMoveStore::default();
        game.update_pseudo_moves(&mut move_store, true);

        TriHexChessWrapper {
            inner: game,
            move_store,
            cached_stores: [None, None, None],
        }

    }

    /// Update the game state from a FEN string
    #[wasm_bindgen(js_name = setFen)]
    pub fn set_fen_web(&mut self, fen: &str) -> Result<(), String> {
        if !fen.is_ascii() {
            return Err("FEN must be ASCII".to_string());
        }

        let bytes = fen.as_bytes();

        self.inner.set_fen(bytes)?;
        self.inner.update_pseudo_moves(&mut self.move_store, true);
        self.cached_stores = [None, None, None];

        Ok(())
    }

    #[wasm_bindgen(js_name = getDebugState)]
    pub fn get_debug_state(&self) -> String {
        format!("{:?}", self.inner.state.phase)
    }

    /// Get the FEN string of the current game state
    #[wasm_bindgen(js_name = getFen)]
    pub fn get_fen_web(&self) -> String {
        to_fen(&self.inner.state)
    }

    /// Get information about the current state of the game (who's turn it is, who won, etc.)
    #[wasm_bindgen(js_name = getGameState)]
    pub fn get_game_state_web(&self) -> GameStateWeb {
        GameStateWeb {
            turn: match self.inner.state.phase {
                Phase::Normal(normal_state) => normal_state.get_turn(),
                Phase::Won(won_player) => won_player,
                Phase::Draw(stalemate_state) => stalemate_state.get_turn(),
            },
            won: match self.inner.state.phase {
                Phase::Won(won_player) => Some(won_player),
                _ => None,
            },
            is_stalemate: matches!(self.inner.state.phase, Phase::Draw(_)),
            turn_counter: self.inner.state.turn_counter as u32,
            third_move_count: self.inner.state.third_move as u32,
        }
    }

    #[wasm_bindgen(js_name = skipToNextPlayer)]
    pub fn skip_to_next_player(&mut self) -> Result<(), String> {
        match self.inner.state.phase {
            Phase::Normal(ref mut player_state) => {
                let (_, next_player) = player_state.get_next_turn();

                if next_player == player_state.get_turn() {
                    return Err("Cannot skip to the same player".to_string());
                }

                self.inner.next_turn(true, &mut self.move_store, true);

                self.cached_stores = [None, None, None];

                Ok(())
            }
            Phase::Won(_) | Phase::Draw(_) => Err("Game is already over".to_string()),
        }
    }

    /// Returns all the pieces present on the board
    #[wasm_bindgen(js_name = getPieces)]
    pub fn get_all_pieces_web(&self) -> Vec<ChessPiece> {
        let mut vec = Vec::with_capacity(64);
        for (pos, slot) in self.inner.state.buffer.non_empty_iter() {
            vec.push(ChessPiece {
                piece: slot.piece().unwrap(),
                player: slot.player().unwrap(),
                coordinates: FullCoordinates::from_memory_pos(pos),
            });
        }

        debug_log(&format!("Memory buffer: {:?}", self.inner.state.buffer));

        vec
    }

    /// Deletes the piece at the given coordinates
    /// This is a debugging function, not meant to be used in a real game
    #[wasm_bindgen(js_name = deletePiece)]
    pub fn delete_piece_web(&mut self, coordinates: FullCoordinates) {
        let pos = coordinates.to_memory_pos();
        self.inner.state.buffer[pos] = MemorySlot::empty();
    }

    /// Calculates the material value for each player
    #[wasm_bindgen(js_name = getMaterial)]
    pub fn get_material_web(&self) -> WebMaterialCounter {
        let mut material = WebMaterialCounter::default();

        for (_, slot) in self.inner.state.buffer.non_empty_iter() {
            let value = slot.piece().unwrap().material();
            match slot.player().unwrap() {
                Color::White => material.white += value,
                Color::Gray => material.gray += value,
                Color::Black => material.black += value,
            }
        }

        material
    }

    /// A debug function that returns all the legal moves available in the current position and turn
    #[wasm_bindgen(js_name = queryAllMoves)]
    pub fn query_all_moves_web(&mut self) -> Vec<MoveWrapper> {
        if self.inner.is_over() {
            return Vec::new();
        }

        let mut move_store = ChessMoveStore::default();

        self.inner.update_pseudo_moves(&mut move_store, true);

        move_store
            .into_iter()
            .map(|m| MoveWrapper::from_move(&m, &self.inner.state.buffer))
            .collect()
    }

    /// Get information about which pieces are attacking the king
    #[wasm_bindgen(js_name = getCheckMetadata)]
    pub fn get_check_metadata_web(&self) -> Vec<CheckInformation> {
        let mut all_check_info = Vec::new();

        for color in COLORS {
            if let Some(king_pos) = self.inner.state.buffer.king_mem_pos(color) {
                all_check_info.extend(self.inner.locate_checking_pieces(king_pos, color));
            }
        }

        debug_log(&format!("Check metadata: {:?}", all_check_info));

        all_check_info
    }

    /// Gets the legal moves available for the piece at the given coordinates
    #[wasm_bindgen(js_name = queryMoves)]
    pub fn query_moves_web(&mut self, from: FullCoordinates) -> Vec<MoveWrapper> {
        if self.inner.is_over() {
            return Vec::new();
        }

        let from_pos = from.to_memory_pos();
        let mut vec = Vec::new();

        for m in self.move_store.iter() {
            if m.from == from_pos {
                match m.move_type {
                    MoveType::Promotion(piece) => {
                        // Only add the queen promotion (reduce 4 moves into 1, and let the UI pick the piece)
                        if piece == Piece::Queen {
                            vec.push(MoveWrapper::from_move(m, &self.inner.state.buffer));
                        }
                    }
                    _ => {
                        vec.push(MoveWrapper::from_move(m, &self.inner.state.buffer));
                    }
                }
            }
        }

        vec
    }

    #[wasm_bindgen(js_name = getCastlingRights)]
    pub fn get_castling_rights_web(&self) -> Vec<Castling> {
        let mut castlings = Vec::with_capacity(3);

        for color in COLORS {
            let can_castle_king_side = self.inner.state.castle.can_king_side(color);
            let can_castle_queen_side = self.inner.state.castle.can_queen_side(color);

            castlings.push(Castling {
                can_castle_king_side,
                can_castle_queen_side,
            });
        }

        castlings
    }

    /// Commits the move to the game state
    /// If the move is a promotion it requires a promotion piece
    /// * `advance_turn` - If true, the turn will be advanced. Should be set to `true`
    #[wasm_bindgen(js_name = commitMove)]
    pub fn commit_move_web(
        &mut self,
        m: &MoveWrapper,
        promotion: Option<Piece>,
        advance_turn: bool,
    ) -> Result<(), String> {
        debug_assert!(if m.move_type == WebMoveType::EnPassant
            || m.move_type == WebMoveType::EnPassantPromotion
        {
            m.en_passant.is_some()
        } else {
            m.en_passant.is_none()
        });

        if matches!(self.inner.state.phase, Phase::Won(_) | Phase::Draw(_)) {
            return Err("Game is already over".to_string());
        }

        if m.is_promotion() && promotion.is_none() {
            return Err("Promotion move requires a promotion piece".to_string());
        }

        self.inner
            .commit_move(&m.to_pseudo_move(promotion), &mut self.move_store);
        self.inner
            .next_turn(advance_turn, &mut self.move_store, true);

        debug_log(&format!("Turn: {:?}", self.inner.state.phase));
        debug_log(&format!("Legal moves: {}", self.move_store.len()));
        debug_log(&format!("Moves: {:?}", self.move_store));

        self.cached_stores = [None, None, None];

        Ok(())
    }

    /// Returns available moves for a piece.
    /// Lazily calculates and caches moves for enemy pieces.
    #[wasm_bindgen(js_name = getPieceMovementPatterns)]
    pub fn get_piece_movement_patterns_web(&mut self, coordinates: FullCoordinates) -> Vec<PatternHighlight> {
        let pos = coordinates.to_memory_pos();
        let slot = self.inner.state.buffer[pos];

        if slot.is_empty() {
            return Vec::new();
        }

        let piece_player = slot.player().unwrap();

        // Check if it's the current player's turn (use primary move_store)
        let current_turn = match self.inner.state.phase {
            Phase::Normal(state) => Some(state.get_turn()),
            _ => None,
        };

        if Some(piece_player) == current_turn {
            return TriHexChessWrapper::collect_highlights_static(&self.move_store, pos, &self.inner.state.buffer);
        }

        // Check Cache for Enemy Player
        let player_idx = piece_player as usize;

        // If cache is empty, generate it
        if self.cached_stores[player_idx].is_none() {
            let mut temp_game = self.inner.clone();

            // Force phase to normal for that player to run move gen
            temp_game.state.phase = Phase::Normal(NormalState::new_with_turn(piece_player));

            let mut new_store = ChessMoveStore::default();
            temp_game.update_pseudo_moves(&mut new_store, true);

            self.cached_stores[player_idx] = Some(new_store);
        }

        // Return from Cache
        if let Some(store) = &self.cached_stores[player_idx] {
            TriHexChessWrapper::collect_highlights_static(store, pos, &self.inner.state.buffer)
        } else {
            Vec::new() // Should be unreachable
        }
    }

    /// Static helper to avoid `&self` borrowing conflicts when using cached stores
    fn collect_highlights_static(
        store: &ChessMoveStore,
        from_pos: MemoryPos,
        buffer: &MemoryBuffer
    ) -> Vec<PatternHighlight> {
        let mut highlights = Vec::new();

        for m in store.iter() {
            if m.from == from_pos {
                match m.move_type {
                    MoveType::Promotion(p) if p != Piece::Queen => continue,
                    MoveType::EnPassantPromotion(ep) if ep.get().1 != Piece::Queen => continue,
                    _ => {}
                }

                let is_capture = !buffer[m.to].is_empty();

                let web_move_type = match m.move_type {
                    MoveType::Move => {
                        if is_capture { WebMoveType::Capture } else { WebMoveType::Move }
                    }
                    MoveType::Promotion(_) => {
                        if is_capture { WebMoveType::CapturePromotion } else { WebMoveType::Promotion }
                    }
                    MoveType::EnPassantPromotion(_) => WebMoveType::EnPassantPromotion,
                    MoveType::EnPassant(_) => WebMoveType::EnPassant,
                    MoveType::CastleKingSide => WebMoveType::CastleKingSide,
                    MoveType::CastleQueenSide => WebMoveType::CastleQueenSide,
                    MoveType::DoublePawnPush => WebMoveType::DoublePawnPush,
                };

                highlights.push(PatternHighlight {
                    to: FullCoordinates::from_memory_pos(m.to),
                    move_type: web_move_type,
                });
            }
        }
        highlights
    }
}
