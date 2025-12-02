let wasm;
export function __wbg_set_wasm(val) {
    wasm = val;
}


let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}
/**
 * Color of the Player. Called 'Color' instead of 'Player' to
 * clearly differentiate from the 'PieceType' enum (same first letter)
 * @enum {0 | 1 | 2}
 */
export const Color = Object.freeze({
    White: 0, "0": "White",
    Gray: 1, "1": "Gray",
    Black: 2, "2": "Black",
});
/**
 * All possible move types
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8}
 */
export const MoveType = Object.freeze({
    Move: 0, "0": "Move",
    DoublePawnPush: 1, "1": "DoublePawnPush",
    Capture: 2, "2": "Capture",
    EnPassant: 3, "3": "EnPassant",
    EnPassantPromotion: 4, "4": "EnPassantPromotion",
    Promotion: 5, "5": "Promotion",
    CapturePromotion: 6, "6": "CapturePromotion",
    CastleKingSide: 7, "7": "CastleKingSide",
    CastleQueenSide: 8, "8": "CastleQueenSide",
});
/**
 * Chess piece representation
 * @enum {1 | 2 | 3 | 4 | 5 | 6}
 */
export const Piece = Object.freeze({
    Pawn: 1, "1": "Pawn",
    Knight: 2, "2": "Knight",
    Bishop: 3, "3": "Bishop",
    Rook: 4, "4": "Rook",
    Queen: 5, "5": "Queen",
    King: 6, "6": "King",
});

const AttackInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_attackinfo_free(ptr >>> 0, 1));
/**
 * Represents an attack on the king by a piece
 */
export class AttackInfo {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(AttackInfo.prototype);
        obj.__wbg_ptr = ptr;
        AttackInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    toJSON() {
        return {
            attack: this.attack,
            king: this.king,
            player_attacked: this.player_attacked,
        };
    }

    toString() {
        return JSON.stringify(this);
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AttackInfoFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_attackinfo_free(ptr, 0);
    }
    /**
     * Position of the attacking piece
     * @returns {Coordinates}
     */
    get attack() {
        const ret = wasm.__wbg_get_attackinfo_attack(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * Position of the attacking piece
     * @param {Coordinates} arg0
     */
    set attack(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_attackinfo_attack(this.__wbg_ptr, ptr0);
    }
    /**
     * Position of the king who is being attacked
     * @returns {Coordinates}
     */
    get king() {
        const ret = wasm.__wbg_get_attackinfo_king(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * Position of the king who is being attacked
     * @param {Coordinates} arg0
     */
    set king(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_attackinfo_king(this.__wbg_ptr, ptr0);
    }
    /**
     * The player who is being attacked
     * @returns {Color}
     */
    get player_attacked() {
        const ret = wasm.__wbg_get_attackinfo_player_attacked(this.__wbg_ptr);
        return ret;
    }
    /**
     * The player who is being attacked
     * @param {Color} arg0
     */
    set player_attacked(arg0) {
        wasm.__wbg_set_attackinfo_player_attacked(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) AttackInfo.prototype[Symbol.dispose] = AttackInfo.prototype.free;

const CastleFlagsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_castleflags_free(ptr >>> 0, 1));
/**
 * Represents the flags for castling
 * The u8 is split into 4 2 bit sections: <Empty> <Black> <Gray> <White>
 * Each section represents bit flags as: <QueenSide> <KingSide>
 */
export class CastleFlags {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(CastleFlags.prototype);
        obj.__wbg_ptr = ptr;
        CastleFlagsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CastleFlagsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_castleflags_free(ptr, 0);
    }
}
if (Symbol.dispose) CastleFlags.prototype[Symbol.dispose] = CastleFlags.prototype.free;

const CastlingFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_castling_free(ptr >>> 0, 1));

export class Castling {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Castling.prototype);
        obj.__wbg_ptr = ptr;
        CastlingFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    static __unwrap(jsValue) {
        if (!(jsValue instanceof Castling)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    toJSON() {
        return {
            can_castle_king_side: this.can_castle_king_side,
            can_castle_queen_side: this.can_castle_queen_side,
        };
    }

    toString() {
        return JSON.stringify(this);
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CastlingFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_castling_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get can_castle_king_side() {
        const ret = wasm.__wbg_get_castling_can_castle_king_side(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set can_castle_king_side(arg0) {
        wasm.__wbg_set_castling_can_castle_king_side(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get can_castle_queen_side() {
        const ret = wasm.__wbg_get_castling_can_castle_queen_side(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set can_castle_queen_side(arg0) {
        wasm.__wbg_set_castling_can_castle_queen_side(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} can_castle_king_side
     * @param {boolean} can_castle_queen_side
     */
    constructor(can_castle_king_side, can_castle_queen_side) {
        const ret = wasm.castling_new(can_castle_king_side, can_castle_queen_side);
        this.__wbg_ptr = ret >>> 0;
        CastlingFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) Castling.prototype[Symbol.dispose] = Castling.prototype.free;

const ChessMoveStoreFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chessmovestore_free(ptr >>> 0, 1));
/**
 * An efficient store for pseudo-legal moves
 * Instead of using a Vec, which would require heap allocation, we use a fixed-size array
 * This is a trade-off, as we can only store a limited number of moves, but it is much faster
 * 384 moves should be enough (?)
 */
export class ChessMoveStore {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChessMoveStoreFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chessmovestore_free(ptr, 0);
    }
}
if (Symbol.dispose) ChessMoveStore.prototype[Symbol.dispose] = ChessMoveStore.prototype.free;

const ChessPieceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chesspiece_free(ptr >>> 0, 1));
/**
 * Represents a piece on the board
 */
export class ChessPiece {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ChessPiece.prototype);
        obj.__wbg_ptr = ptr;
        ChessPieceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    static __unwrap(jsValue) {
        if (!(jsValue instanceof ChessPiece)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    toJSON() {
        return {
            piece: this.piece,
            player: this.player,
            coordinates: this.coordinates,
        };
    }

    toString() {
        return JSON.stringify(this);
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChessPieceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chesspiece_free(ptr, 0);
    }
    /**
     * @returns {Piece}
     */
    get piece() {
        const ret = wasm.__wbg_get_chesspiece_piece(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Piece} arg0
     */
    set piece(arg0) {
        wasm.__wbg_set_chesspiece_piece(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Color}
     */
    get player() {
        const ret = wasm.__wbg_get_chesspiece_player(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Color} arg0
     */
    set player(arg0) {
        wasm.__wbg_set_chesspiece_player(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Coordinates}
     */
    get coordinates() {
        const ret = wasm.__wbg_get_chesspiece_coordinates(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * @param {Coordinates} arg0
     */
    set coordinates(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_chesspiece_coordinates(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {Piece} piece
     * @param {Color} player
     * @param {number} memory_index
     */
    constructor(piece, player, memory_index) {
        const ret = wasm.chesspiece_new_piece(piece, player, memory_index);
        this.__wbg_ptr = ret >>> 0;
        ChessPieceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChessPiece.prototype[Symbol.dispose] = ChessPiece.prototype.free;

const CoordinatesFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_coordinates_free(ptr >>> 0, 1));

export class Coordinates {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Coordinates.prototype);
        obj.__wbg_ptr = ptr;
        CoordinatesFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    toJSON() {
        return {
            i: this.i,
            q: this.q,
            r: this.r,
            s: this.s,
        };
    }

    toString() {
        return JSON.stringify(this);
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CoordinatesFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_coordinates_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get i() {
        const ret = wasm.__wbg_get_coordinates_i(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set i(arg0) {
        wasm.__wbg_set_coordinates_i(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get q() {
        const ret = wasm.__wbg_get_coordinates_q(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set q(arg0) {
        wasm.__wbg_set_coordinates_q(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get r() {
        const ret = wasm.__wbg_get_coordinates_r(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set r(arg0) {
        wasm.__wbg_set_coordinates_r(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get s() {
        const ret = wasm.__wbg_get_coordinates_s(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set s(arg0) {
        wasm.__wbg_set_coordinates_s(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Coordinates.prototype[Symbol.dispose] = Coordinates.prototype.free;

const EnPassantStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_enpassantstate_free(ptr >>> 0, 1));
/**
 * Memory posses for en passant are stored in the EnPassantState
 * Presented in string format: Range from 1 to 8 (inclusive), or None if no en passant is available
 */
export class EnPassantState {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(EnPassantState.prototype);
        obj.__wbg_ptr = ptr;
        EnPassantStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EnPassantStateFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_enpassantstate_free(ptr, 0);
    }
}
if (Symbol.dispose) EnPassantState.prototype[Symbol.dispose] = EnPassantState.prototype.free;

const GameStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_gamestate_free(ptr >>> 0, 1));

export class GameState {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(GameState.prototype);
        obj.__wbg_ptr = ptr;
        GameStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GameStateFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_gamestate_free(ptr, 0);
    }
    /**
     * Who's turn it is
     * @returns {Color}
     */
    get turn() {
        const ret = wasm.__wbg_get_gamestate_turn(this.__wbg_ptr);
        return ret;
    }
    /**
     * Who's turn it is
     * @param {Color} arg0
     */
    set turn(arg0) {
        wasm.__wbg_set_gamestate_turn(this.__wbg_ptr, arg0);
    }
    /**
     * Who won the game, undefined if the game is ongoing
     * @returns {Color | undefined}
     */
    get won() {
        const ret = wasm.__wbg_get_gamestate_won(this.__wbg_ptr);
        return ret === 3 ? undefined : ret;
    }
    /**
     * Who won the game, undefined if the game is ongoing
     * @param {Color | null} [arg0]
     */
    set won(arg0) {
        wasm.__wbg_set_gamestate_won(this.__wbg_ptr, isLikeNone(arg0) ? 3 : arg0);
    }
    /**
     * If the game is a stalemate
     * @returns {boolean}
     */
    get is_stalemate() {
        const ret = wasm.__wbg_get_gamestate_is_stalemate(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * If the game is a stalemate
     * @param {boolean} arg0
     */
    set is_stalemate(arg0) {
        wasm.__wbg_set_gamestate_is_stalemate(this.__wbg_ptr, arg0);
    }
    /**
     * The number of turns that have passed
     * @returns {number}
     */
    get turn_counter() {
        const ret = wasm.__wbg_get_gamestate_turn_counter(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The number of turns that have passed
     * @param {number} arg0
     */
    set turn_counter(arg0) {
        wasm.__wbg_set_gamestate_turn_counter(this.__wbg_ptr, arg0);
    }
    /**
     * The number of 'third moves' since the last capture or pawn move
     * @returns {number}
     */
    get third_move_count() {
        const ret = wasm.__wbg_get_gamestate_third_move_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The number of 'third moves' since the last capture or pawn move
     * @param {number} arg0
     */
    set third_move_count(arg0) {
        wasm.__wbg_set_gamestate_third_move_count(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) GameState.prototype[Symbol.dispose] = GameState.prototype.free;

const MaterialCounterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_materialcounter_free(ptr >>> 0, 1));
/**
 * Represents the material value for each player
 * Queen: 9, Rook: 5, Bishop: 3, Knight: 3, Pawn: 1, King has no value
 * Starting material value for each player: 39
 */
export class MaterialCounter {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MaterialCounter.prototype);
        obj.__wbg_ptr = ptr;
        MaterialCounterFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MaterialCounterFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_materialcounter_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get white() {
        const ret = wasm.__wbg_get_materialcounter_white(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set white(arg0) {
        wasm.__wbg_set_materialcounter_white(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get gray() {
        const ret = wasm.__wbg_get_materialcounter_gray(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set gray(arg0) {
        wasm.__wbg_set_materialcounter_gray(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get black() {
        const ret = wasm.__wbg_get_materialcounter_black(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set black(arg0) {
        wasm.__wbg_set_materialcounter_black(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) MaterialCounter.prototype[Symbol.dispose] = MaterialCounter.prototype.free;

const MemoryBufferFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_memorybuffer_free(ptr >>> 0, 1));
/**
 * MemoryBuffer is a 96 slot array of [`MemorySlot`]s. In total of 768 bits of memory
 */
export class MemoryBuffer {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MemoryBuffer.prototype);
        obj.__wbg_ptr = ptr;
        MemoryBufferFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MemoryBufferFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_memorybuffer_free(ptr, 0);
    }
}
if (Symbol.dispose) MemoryBuffer.prototype[Symbol.dispose] = MemoryBuffer.prototype.free;

const MoveWrapperFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_movewrapper_free(ptr >>> 0, 1));

export class MoveWrapper {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MoveWrapper.prototype);
        obj.__wbg_ptr = ptr;
        MoveWrapperFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MoveWrapperFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_movewrapper_free(ptr, 0);
    }
    /**
     * @returns {Coordinates}
     */
    get from() {
        const ret = wasm.__wbg_get_movewrapper_from(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * @param {Coordinates} arg0
     */
    set from(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_movewrapper_from(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Coordinates}
     */
    get to() {
        const ret = wasm.__wbg_get_movewrapper_to(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * @param {Coordinates} arg0
     */
    set to(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_movewrapper_to(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {MoveType}
     */
    get move_type() {
        const ret = wasm.__wbg_get_movewrapper_move_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {MoveType} arg0
     */
    set move_type(arg0) {
        wasm.__wbg_set_movewrapper_move_type(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Color}
     */
    get color() {
        const ret = wasm.__wbg_get_movewrapper_color(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Color} arg0
     */
    set color(arg0) {
        wasm.__wbg_set_movewrapper_color(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Piece}
     */
    get piece() {
        const ret = wasm.__wbg_get_movewrapper_piece(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Piece} arg0
     */
    set piece(arg0) {
        wasm.__wbg_set_movewrapper_piece(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {string}
     */
    get getNotationLAN() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.movewrapper_getNotationLAN(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) MoveWrapper.prototype[Symbol.dispose] = MoveWrapper.prototype.free;

const PatternHighlightFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_patternhighlight_free(ptr >>> 0, 1));
/**
 * Lightweight move info for UI hover/highlighting
 */
export class PatternHighlight {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(PatternHighlight.prototype);
        obj.__wbg_ptr = ptr;
        PatternHighlightFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PatternHighlightFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_patternhighlight_free(ptr, 0);
    }
    /**
     * @returns {Coordinates}
     */
    get to() {
        const ret = wasm.__wbg_get_patternhighlight_to(this.__wbg_ptr);
        return Coordinates.__wrap(ret);
    }
    /**
     * @param {Coordinates} arg0
     */
    set to(arg0) {
        _assertClass(arg0, Coordinates);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_patternhighlight_to(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {MoveType}
     */
    get move_type() {
        const ret = wasm.__wbg_get_patternhighlight_move_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {MoveType} arg0
     */
    set move_type(arg0) {
        wasm.__wbg_set_patternhighlight_move_type(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) PatternHighlight.prototype[Symbol.dispose] = PatternHighlight.prototype.free;

const RepetitionHistoryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_repetitionhistory_free(ptr >>> 0, 1));

export class RepetitionHistory {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(RepetitionHistory.prototype);
        obj.__wbg_ptr = ptr;
        RepetitionHistoryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RepetitionHistoryFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_repetitionhistory_free(ptr, 0);
    }
}
if (Symbol.dispose) RepetitionHistory.prototype[Symbol.dispose] = RepetitionHistory.prototype.free;

const StateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_state_free(ptr >>> 0, 1));

export class State {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(State.prototype);
        obj.__wbg_ptr = ptr;
        StateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StateFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_state_free(ptr, 0);
    }
    /**
     * @returns {MemoryBuffer}
     */
    get buffer() {
        const ret = wasm.__wbg_get_state_buffer(this.__wbg_ptr);
        return MemoryBuffer.__wrap(ret);
    }
    /**
     * @param {MemoryBuffer} arg0
     */
    set buffer(arg0) {
        _assertClass(arg0, MemoryBuffer);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_state_buffer(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {CastleFlags}
     */
    get castle() {
        const ret = wasm.__wbg_get_state_castle(this.__wbg_ptr);
        return CastleFlags.__wrap(ret);
    }
    /**
     * @param {CastleFlags} arg0
     */
    set castle(arg0) {
        _assertClass(arg0, CastleFlags);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_state_castle(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {EnPassantState}
     */
    get en_passant() {
        const ret = wasm.__wbg_get_state_en_passant(this.__wbg_ptr);
        return EnPassantState.__wrap(ret);
    }
    /**
     * @param {EnPassantState} arg0
     */
    set en_passant(arg0) {
        _assertClass(arg0, EnPassantState);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_state_en_passant(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {number}
     */
    get third_move() {
        const ret = wasm.__wbg_get_state_third_move(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set third_move(arg0) {
        wasm.__wbg_set_state_third_move(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get turn_counter() {
        const ret = wasm.__wbg_get_state_turn_counter(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set turn_counter(arg0) {
        wasm.__wbg_set_state_turn_counter(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) State.prototype[Symbol.dispose] = State.prototype.free;

const TriHexChessFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trihexchess_free(ptr >>> 0, 1));

export class TriHexChess {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TriHexChessFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trihexchess_free(ptr, 0);
    }
    /**
     * @returns {State}
     */
    get state() {
        const ret = wasm.__wbg_get_trihexchess_state(this.__wbg_ptr);
        return State.__wrap(ret);
    }
    /**
     * @param {State} arg0
     */
    set state(arg0) {
        _assertClass(arg0, State);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_trihexchess_state(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {boolean}
     */
    get is_using_grace_period() {
        const ret = wasm.__wbg_get_trihexchess_is_using_grace_period(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set is_using_grace_period(arg0) {
        wasm.__wbg_set_trihexchess_is_using_grace_period(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {bigint}
     */
    get zobrist_hash() {
        const ret = wasm.__wbg_get_trihexchess_zobrist_hash(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {bigint} arg0
     */
    set zobrist_hash(arg0) {
        wasm.__wbg_set_trihexchess_zobrist_hash(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {RepetitionHistory}
     */
    get repetition_history() {
        const ret = wasm.__wbg_get_trihexchess_repetition_history(this.__wbg_ptr);
        return RepetitionHistory.__wrap(ret);
    }
    /**
     * @param {RepetitionHistory} arg0
     */
    set repetition_history(arg0) {
        _assertClass(arg0, RepetitionHistory);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_trihexchess_repetition_history(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) TriHexChess.prototype[Symbol.dispose] = TriHexChess.prototype.free;

const TriHexChessWrapperFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trihexchesswrapper_free(ptr >>> 0, 1));

export class TriHexChessWrapper {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(TriHexChessWrapper.prototype);
        obj.__wbg_ptr = ptr;
        TriHexChessWrapperFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TriHexChessWrapperFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trihexchesswrapper_free(ptr, 0);
    }
    /**
     * Create a new game from a FEN string
     * @param {string} fen
     * @returns {TriHexChessWrapper}
     */
    static new(fen) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trihexchesswrapper_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return TriHexChessWrapper.__wrap(ret[0]);
    }
    /**
     * @returns {TriHexChessWrapper}
     */
    static newDefault() {
        const ret = wasm.trihexchesswrapper_newDefault();
        return TriHexChessWrapper.__wrap(ret);
    }
    /**
     * Update the game state from a FEN string
     * @param {string} fen
     */
    setFen(fen) {
        const ptr0 = passStringToWasm0(fen, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trihexchesswrapper_setFen(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {string}
     */
    getDebugState() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trihexchesswrapper_getDebugState(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get the FEN string of the current game state
     * @returns {string}
     */
    getFen() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trihexchesswrapper_getFen(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get information about the current state of the game (who's turn it is, who won, etc.)
     * @returns {GameState}
     */
    getGameState() {
        const ret = wasm.trihexchesswrapper_getGameState(this.__wbg_ptr);
        return GameState.__wrap(ret);
    }
    skipToNextPlayer() {
        const ret = wasm.trihexchesswrapper_skipToNextPlayer(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Returns all the pieces present on the board
     * @returns {ChessPiece[]}
     */
    getPieces() {
        const ret = wasm.trihexchesswrapper_getPieces(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Deletes the piece at the given coordinates
     * This is a debugging function, not meant to be used in a real game
     * @param {Coordinates} coordinates
     */
    deletePiece(coordinates) {
        _assertClass(coordinates, Coordinates);
        var ptr0 = coordinates.__destroy_into_raw();
        wasm.trihexchesswrapper_deletePiece(this.__wbg_ptr, ptr0);
    }
    /**
     * Calculates the material value for each player
     * @returns {MaterialCounter}
     */
    getMaterial() {
        const ret = wasm.trihexchesswrapper_getMaterial(this.__wbg_ptr);
        return MaterialCounter.__wrap(ret);
    }
    /**
     * A debug function that returns all the legal moves available in the current position and turn
     * @returns {MoveWrapper[]}
     */
    queryAllMoves() {
        const ret = wasm.trihexchesswrapper_queryAllMoves(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get information about which pieces are attacking the king
     * @returns {AttackInfo[]}
     */
    getCheckMetadata() {
        const ret = wasm.trihexchesswrapper_getCheckMetadata(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Gets the legal moves available for the piece at the given coordinates
     * @param {Coordinates} from
     * @returns {MoveWrapper[]}
     */
    queryMoves(from) {
        _assertClass(from, Coordinates);
        var ptr0 = from.__destroy_into_raw();
        const ret = wasm.trihexchesswrapper_queryMoves(this.__wbg_ptr, ptr0);
        var v2 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * @returns {Castling[]}
     */
    getCastlingRights() {
        const ret = wasm.trihexchesswrapper_getCastlingRights(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Commits the move to the game state
     * If the move is a promotion it requires a promotion piece
     * * `advance_turn` - If true, the turn will be advanced. Should be set to `true`
     * @param {MoveWrapper} m
     * @param {Piece | null | undefined} promotion
     * @param {boolean} advance_turn
     */
    commitMove(m, promotion, advance_turn) {
        _assertClass(m, MoveWrapper);
        const ret = wasm.trihexchesswrapper_commitMove(this.__wbg_ptr, m.__wbg_ptr, isLikeNone(promotion) ? 0 : promotion, advance_turn);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Returns available moves for a piece.
     * Lazily calculates and caches moves for enemy pieces.
     * @param {Coordinates} coordinates
     * @returns {PatternHighlight[]}
     */
    getPieceMovementPatterns(coordinates) {
        _assertClass(coordinates, Coordinates);
        var ptr0 = coordinates.__destroy_into_raw();
        const ret = wasm.trihexchesswrapper_getPieceMovementPatterns(this.__wbg_ptr, ptr0);
        var v2 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}
if (Symbol.dispose) TriHexChessWrapper.prototype[Symbol.dispose] = TriHexChessWrapper.prototype.free;

const UnvalidatedBoardFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_unvalidatedboard_free(ptr >>> 0, 1));

export class UnvalidatedBoard {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UnvalidatedBoardFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_unvalidatedboard_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.unvalidatedboard_new();
        this.__wbg_ptr = ret >>> 0;
        UnvalidatedBoardFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {ChessPiece[]} pieces
     */
    set_pieces(pieces) {
        const ptr0 = passArrayJsValueToWasm0(pieces, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.unvalidatedboard_set_pieces(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Color} turn
     */
    set_turn(turn) {
        wasm.unvalidatedboard_set_turn(this.__wbg_ptr, turn);
    }
    /**
     * @param {number | null} [white]
     * @param {number | null} [gray]
     * @param {number | null} [black]
     */
    set_all_en_passant(white, gray, black) {
        wasm.unvalidatedboard_set_all_en_passant(this.__wbg_ptr, isLikeNone(white) ? 0xFFFFFF : white, isLikeNone(gray) ? 0xFFFFFF : gray, isLikeNone(black) ? 0xFFFFFF : black);
    }
    /**
     * @param {number} turn_counter
     */
    set_turn_counter(turn_counter) {
        wasm.unvalidatedboard_set_turn_counter(this.__wbg_ptr, turn_counter);
    }
    /**
     * @param {number} third_move_count
     */
    set_third_move_count(third_move_count) {
        wasm.unvalidatedboard_set_third_move_count(this.__wbg_ptr, third_move_count);
    }
    /**
     * @param {Castling[]} castlings
     */
    set_castlings(castlings) {
        const ptr0 = passArrayJsValueToWasm0(castlings, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.unvalidatedboard_set_castlings(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Converts the unvalidated board into a FEN string
     * @returns {string}
     */
    to_fen() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.unvalidatedboard_to_fen(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
}
if (Symbol.dispose) UnvalidatedBoard.prototype[Symbol.dispose] = UnvalidatedBoard.prototype.free;

export function __wbg___wbindgen_throw_b855445ff6a94295(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

export function __wbg_attackinfo_new(arg0) {
    const ret = AttackInfo.__wrap(arg0);
    return ret;
};

export function __wbg_castling_new(arg0) {
    const ret = Castling.__wrap(arg0);
    return ret;
};

export function __wbg_castling_unwrap(arg0) {
    const ret = Castling.__unwrap(arg0);
    return ret;
};

export function __wbg_chesspiece_new(arg0) {
    const ret = ChessPiece.__wrap(arg0);
    return ret;
};

export function __wbg_chesspiece_unwrap(arg0) {
    const ret = ChessPiece.__unwrap(arg0);
    return ret;
};

export function __wbg_error_7534b8e9a36f1ab4(arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
        deferred0_0 = arg0;
        deferred0_1 = arg1;
        console.error(getStringFromWasm0(arg0, arg1));
    } finally {
        wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
    }
};

export function __wbg_movewrapper_new(arg0) {
    const ret = MoveWrapper.__wrap(arg0);
    return ret;
};

export function __wbg_new_8a6f238a6ece86ea() {
    const ret = new Error();
    return ret;
};

export function __wbg_patternhighlight_new(arg0) {
    const ret = PatternHighlight.__wrap(arg0);
    return ret;
};

export function __wbg_stack_0ed75d68575b0f3c(arg0, arg1) {
    const ret = arg1.stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

export function __wbindgen_cast_2241b6af4c4b2941(arg0, arg1) {
    // Cast intrinsic for `Ref(String) -> Externref`.
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
};

export function __wbindgen_init_externref_table() {
    const table = wasm.__wbindgen_externrefs;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
    ;
};

