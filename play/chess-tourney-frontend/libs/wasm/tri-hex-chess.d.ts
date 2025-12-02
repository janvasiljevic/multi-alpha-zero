/* tslint:disable */
/* eslint-disable */
/**
 * Color of the Player. Called 'Color' instead of 'Player' to
 * clearly differentiate from the 'PieceType' enum (same first letter)
 */
export enum Color {
  White = 0,
  Gray = 1,
  Black = 2,
}
/**
 * All possible move types
 */
export enum MoveType {
  Move = 0,
  DoublePawnPush = 1,
  Capture = 2,
  EnPassant = 3,
  EnPassantPromotion = 4,
  Promotion = 5,
  CapturePromotion = 6,
  CastleKingSide = 7,
  CastleQueenSide = 8,
}
/**
 * Chess piece representation
 */
export enum Piece {
  Pawn = 1,
  Knight = 2,
  Bishop = 3,
  Rook = 4,
  Queen = 5,
  King = 6,
}
/**
 * Represents an attack on the king by a piece
 */
export class AttackInfo {
  private constructor();
/**
** Return copy of self without private attributes.
*/
  toJSON(): Object;
/**
* Return stringified version of self.
*/
  toString(): string;
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Position of the attacking piece
   */
  attack: Coordinates;
  /**
   * Position of the king who is being attacked
   */
  king: Coordinates;
  /**
   * The player who is being attacked
   */
  player_attacked: Color;
}
/**
 * Represents the flags for castling
 * The u8 is split into 4 2 bit sections: <Empty> <Black> <Gray> <White>
 * Each section represents bit flags as: <QueenSide> <KingSide>
 */
export class CastleFlags {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
export class Castling {
/**
** Return copy of self without private attributes.
*/
  toJSON(): Object;
/**
* Return stringified version of self.
*/
  toString(): string;
  free(): void;
  [Symbol.dispose](): void;
  constructor(can_castle_king_side: boolean, can_castle_queen_side: boolean);
  can_castle_king_side: boolean;
  can_castle_queen_side: boolean;
}
/**
 * An efficient store for pseudo-legal moves
 * Instead of using a Vec, which would require heap allocation, we use a fixed-size array
 * This is a trade-off, as we can only store a limited number of moves, but it is much faster
 * 384 moves should be enough (?)
 */
export class ChessMoveStore {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
/**
 * Represents a piece on the board
 */
export class ChessPiece {
/**
** Return copy of self without private attributes.
*/
  toJSON(): Object;
/**
* Return stringified version of self.
*/
  toString(): string;
  free(): void;
  [Symbol.dispose](): void;
  constructor(piece: Piece, player: Color, memory_index: number);
  piece: Piece;
  player: Color;
  coordinates: Coordinates;
}
export class Coordinates {
  private constructor();
/**
** Return copy of self without private attributes.
*/
  toJSON(): Object;
/**
* Return stringified version of self.
*/
  toString(): string;
  free(): void;
  [Symbol.dispose](): void;
  i: number;
  q: number;
  r: number;
  s: number;
}
/**
 * Memory posses for en passant are stored in the EnPassantState
 * Presented in string format: Range from 1 to 8 (inclusive), or None if no en passant is available
 */
export class EnPassantState {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
export class GameState {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Who's turn it is
   */
  turn: Color;
  /**
   * Who won the game, undefined if the game is ongoing
   */
  get won(): Color | undefined;
  /**
   * Who won the game, undefined if the game is ongoing
   */
  set won(value: Color | null | undefined);
  /**
   * If the game is a stalemate
   */
  is_stalemate: boolean;
  /**
   * The number of turns that have passed
   */
  turn_counter: number;
  /**
   * The number of 'third moves' since the last capture or pawn move
   */
  third_move_count: number;
}
/**
 * Represents the material value for each player
 * Queen: 9, Rook: 5, Bishop: 3, Knight: 3, Pawn: 1, King has no value
 * Starting material value for each player: 39
 */
export class MaterialCounter {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  white: number;
  gray: number;
  black: number;
}
/**
 * MemoryBuffer is a 96 slot array of [`MemorySlot`]s. In total of 768 bits of memory
 */
export class MemoryBuffer {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
export class MoveWrapper {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  from: Coordinates;
  to: Coordinates;
  move_type: MoveType;
  color: Color;
  piece: Piece;
  readonly getNotationLAN: string;
}
/**
 * Lightweight move info for UI hover/highlighting
 */
export class PatternHighlight {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  to: Coordinates;
  move_type: MoveType;
}
export class RepetitionHistory {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}
export class State {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  buffer: MemoryBuffer;
  castle: CastleFlags;
  en_passant: EnPassantState;
  third_move: number;
  turn_counter: number;
}
export class TriHexChess {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  state: State;
  is_using_grace_period: boolean;
  zobrist_hash: bigint;
  repetition_history: RepetitionHistory;
}
export class TriHexChessWrapper {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new game from a FEN string
   */
  static new(fen: string): TriHexChessWrapper;
  static newDefault(): TriHexChessWrapper;
  /**
   * Update the game state from a FEN string
   */
  setFen(fen: string): void;
  getDebugState(): string;
  /**
   * Get the FEN string of the current game state
   */
  getFen(): string;
  /**
   * Get information about the current state of the game (who's turn it is, who won, etc.)
   */
  getGameState(): GameState;
  skipToNextPlayer(): void;
  /**
   * Returns all the pieces present on the board
   */
  getPieces(): ChessPiece[];
  /**
   * Deletes the piece at the given coordinates
   * This is a debugging function, not meant to be used in a real game
   */
  deletePiece(coordinates: Coordinates): void;
  /**
   * Calculates the material value for each player
   */
  getMaterial(): MaterialCounter;
  /**
   * A debug function that returns all the legal moves available in the current position and turn
   */
  queryAllMoves(): MoveWrapper[];
  /**
   * Get information about which pieces are attacking the king
   */
  getCheckMetadata(): AttackInfo[];
  /**
   * Gets the legal moves available for the piece at the given coordinates
   */
  queryMoves(from: Coordinates): MoveWrapper[];
  getCastlingRights(): Castling[];
  /**
   * Commits the move to the game state
   * If the move is a promotion it requires a promotion piece
   * * `advance_turn` - If true, the turn will be advanced. Should be set to `true`
   */
  commitMove(m: MoveWrapper, promotion: Piece | null | undefined, advance_turn: boolean): void;
  /**
   * Returns available moves for a piece.
   * Lazily calculates and caches moves for enemy pieces.
   */
  getPieceMovementPatterns(coordinates: Coordinates): PatternHighlight[];
}
export class UnvalidatedBoard {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  set_pieces(pieces: ChessPiece[]): void;
  set_turn(turn: Color): void;
  set_all_en_passant(white?: number | null, gray?: number | null, black?: number | null): void;
  set_turn_counter(turn_counter: number): void;
  set_third_move_count(third_move_count: number): void;
  set_castlings(castlings: Castling[]): void;
  /**
   * Converts the unvalidated board into a FEN string
   */
  to_fen(): string;
}
