use crate::basics::Color;
use crate::constants::{MEM_POS_TO_FILE, MEM_POS_TO_RANK};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

const fn mod_neg(n: i8, m: i8) -> i8 {
    ((n % m) + m) % m
}

/// Compile-time coordinate generation
/// Generated i -> (q, r, s) coordinates mappings used in [`MemoryPos`] and [`QRSPos`]
pub const fn generate_i_to_qrs_mapping() -> [[i8; 96]; 3] {
    let mut q = [0i8; 96];
    let mut r = [0i8; 96];
    let mut s = [0i8; 96];

    let mut mem_idx = 0;
    while mem_idx < 96 {
        let player_idx = (mem_idx >> 5) as i8; // Which player's perspective (0, 1, or 2)
        let local_mem_idx = (mem_idx & 31) as i8; // Index within the player's memory

        let col = local_mem_idx & 7i8;
        let row = local_mem_idx >> 3i8;

        // Get local coordinates first
        let secondary = if col < 4 { row - 4 } else { row - col };

        // arr[0] = col - 4
        // arr[1] = secondary
        // arr[2] = 3 - col - secondary
        let arr = [col - 4, secondary, 3 - col - secondary];

        // Then rotate based on player
        q[mem_idx] = arr[mod_neg(-player_idx, 3) as usize];
        r[mem_idx] = arr[mod_neg(1 - player_idx, 3) as usize];
        s[mem_idx] = arr[mod_neg(2 - player_idx, 3) as usize];

        mem_idx += 1;
    }

    [q, r, s]
}

/// Compile-time [`QRSPos`] to [`MemoryPos`] mapping generation
const fn generate_qrs_to_i_mapping() -> [[u8; 15]; 15] {
    let mut lut = [[0u8; 15]; 15];

    const fn prim_sec_to_mem_idx(prim: i8, s1st: i8, s2nd: i8) -> i8 {
        let col = prim + 4;
        let row = if col < 4 { s1st + 4 } else { -s2nd + 3 };

        row * 8 + col
    }

    const fn to_pos(q: i8, r: i8, s: i8) -> i8 {
        if r < 0 && 0 <= s {
            return prim_sec_to_mem_idx(q, r, s);
        }
        if s < 0 && 0 <= q {
            return 32 + prim_sec_to_mem_idx(r, s, q);
        }

        64 + prim_sec_to_mem_idx(s, q, r)
    }

    let mut q = -7i8;

    'incr_q: loop {
        let mut r = -7i8;

        'incr_r: loop {
            let s = -1 - q - r;

            let is_in = (s <= 3 && q <= 3 && r <= 3) || (q >= -4 && r >= -4 && s >= -4);

            if is_in {
                let pos = to_pos(q, r, s);
                lut[(q + 7) as usize][(r + 7) as usize] = pos as u8;
            }

            r += 1;

            if r > 7 {
                break 'incr_r;
            }
        }

        q += 1;
        if q > 7 {
            break 'incr_q;
        }
    }

    lut
}

const QR_TO_MEM: [[u8; 15]; 15] = generate_qrs_to_i_mapping();

/// Maps the Q and R components of a [`QRSPos`] to a [`MemoryPos`]
/// In a flattened array to avoid unnecessary indirection if the compiler doesn't optimize it
/// Index is calculated as `(q + 7) * 15 + (r + 7)`
pub const QR_TO_MEM_FLAT: [u8; 225] = {
    let mut arr = [0u8; 225];
    let mut i = 0;
    while i < 225 {
        arr[i] = QR_TO_MEM[i / 15][i % 15];
        i += 1;
    }
    arr
};

const COORDS: [[i8; 96]; 3] = generate_i_to_qrs_mapping();
const Q: &[i8; 96] = &COORDS[0];
const R: &[i8; 96] = &COORDS[1];
const S: &[i8; 96] = &COORDS[2];

/// Maps from [`MemoryPos`] u8 to the Q component of the corresponding [`QRSPos`]
pub const MEMORY_TO_Q: [&[i8]; 3] = [Q, R, S];

/// Maps from [`MemoryPos`] u8 to the R component of the corresponding [`QRSPos`]
pub const MEMORY_TO_R: [&[i8]; 3] = [R, S, Q];

/// Maps from [`MemoryPos`] u8 to the S component of the corresponding [`QRSPos`]
pub const MEMORY_TO_S: [&[i8]; 3] = [S, Q, R];

/// Memory position on the board: 0..95, represented as an u8
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPos(pub u8);

pub fn prim_sec_to_mem_idx(prim: i8, s1st: i8, s2nd: i8) -> i8 {
    let col = prim + 4;
    let row = if col < 4 { s1st + 4 } else { -s2nd + 3 };

    row * 8 + col
}

impl MemoryPos {
    #[inline(always)]
    pub const fn new(pos: u8) -> MemoryPos {
        // debug_assert!(pos < 96, "MemoryPos out of bounds: {}", pos);
        MemoryPos(pos)
    }

    /// Converts a memory position to local coordinates
    /// * `player` - The player's perspective to convert to
    pub const fn to_qrs_local(&self, player: Color) -> QRSPos {
        QRSPos {
            q: MEMORY_TO_Q[player as usize][self.0 as usize],
            r: MEMORY_TO_R[player as usize][self.0 as usize],
            s: MEMORY_TO_S[player as usize][self.0 as usize],
        }
    }

    /// Converts a memory position to global coordinates (in W view)
    pub const fn to_qrs_global(&self) -> QRSPos {
        QRSPos {
            q: MEMORY_TO_Q[0][self.0 as usize],
            r: MEMORY_TO_R[0][self.0 as usize],
            s: MEMORY_TO_S[0][self.0 as usize],
        }
    }

    /// Returns true if the memory position is the queen's rook original position
    pub fn is_queen_rook_og(&self) -> bool {
        self.0 == 0 || self.0 == 32 || self.0 == 64
    }

    /// Returns true if the memory position is the king's rook original position
    pub fn is_king_rook_og(&self) -> bool {
        self.0 == 7 || self.0 == 39 || self.0 == 71
    }

    /// Takes a memory location in local coordinates (W/B/G)
    /// converts to "global" coordinates (W view)
    pub const fn to_global(&self, offset: u8) -> MemoryPos {
        // debug_assert!(
        //     offset == 0 || offset == 32 || offset == 64,
        //     "Invalid offset: {}",
        //     offset
        // );
        MemoryPos::new((self.0 + offset) % 96)
    }

    pub const fn to_local(&self, offset: u8) -> MemoryPos {
        // debug_assert!(
        //     offset == 0 || offset == 32 || offset == 64,
        //     "Invalid offset: {}",
        //     offset
        // );
        MemoryPos::new((self.0 + 96 - offset) % 96)
    }

    #[inline(always)]
    pub fn is_original_pawn(&self, player: Color) -> bool {
        match player {
            Color::White => self.0 < 16,
            Color::Gray => self.0 >= 40 && self.0 < 48,
            Color::Black => self.0 >= 72 && self.0 < 80,
        }
    }

    pub fn add(&self, offset: i8) -> MemoryPos {
        MemoryPos::new((self.0 as i8 + offset) as u8)
    }

    pub fn get_uci_notation(&self) -> String {
        format!(
            "{}{}",
            MEM_POS_TO_FILE[self.0 as usize], MEM_POS_TO_RANK[self.0 as usize]
        )
    }
}

impl From<MemoryPos> for usize {
    fn from(pos: MemoryPos) -> usize {
        pos.0 as usize
    }
}

#[derive(Clone, Default, Debug, Copy, PartialEq, Eq)]
pub struct QRSPos {
    pub q: i8,
    pub r: i8,
    pub s: i8,
}

impl QRSPos {
    pub fn distance(&self, other: &QRSPos) -> i8 {
        ((self.q - other.q).abs() + (self.r - other.r).abs() + (self.s - other.s).abs()) / 2
    }

    pub fn is_promotion(&self) -> bool {
        if (-7 <= self.q && self.q <= -5) || (0 <= self.q && self.q <= 3) {
            return self.r == 3;
        }
        self.s == -4
    }

    #[deprecated(
        since = "0.0.0",
        note = "Use to_memory_pos instead. Here only for benchmarking and testing!"
    )]
    pub fn to_memory_pos_slow(&self) -> MemoryPos {
        if self.r < 0 && 0 <= self.s {
            return MemoryPos(prim_sec_to_mem_idx(self.q, self.r, self.s) as u8);
        }
        if self.s < 0 && 0 <= self.q {
            return MemoryPos(32 + prim_sec_to_mem_idx(self.r, self.s, self.q) as u8);
        }

        MemoryPos(64 + prim_sec_to_mem_idx(self.s, self.q, self.r) as u8)
    }

    /// Maps to [`MemoryPos`] using a lookup table
    #[inline(always)]
    pub const fn to_pos(&self) -> MemoryPos {
        // debug_assert!(self.is_in(), "QRSPos is out of bounds: {:?}", self);
        MemoryPos(QR_TO_MEM_FLAT[((self.q + 7) as usize) * 15 + ((self.r + 7) as usize)])
    }

    pub const fn add(&self, q: i8, r: i8, s: i8) -> QRSPos {
        QRSPos {
            q: self.q + q,
            r: self.r + r,
            s: self.s + s,
        }
    }

    pub fn set(&mut self, q: i8, r: i8, s: i8) {
        debug_assert!(
            q + r + s == -1,
            "Invalid QRS [not sum to -1]: {:?}",
            (q, r, s)
        );
        self.q = q;
        self.r = r;
        self.s = s;
    }

    pub const fn is_in(&self) -> bool {
        (self.s <= 3 && self.q <= 3 && self.r <= 3)
            || (self.q >= -4 && self.r >= -4 && self.s >= -4)
    }
}

// #[wasm_bindgen(inspectable, js_name = Coordinates)]
#[cfg_attr(feature = "wasm", wasm_bindgen(inspectable, js_name = Coordinates))]
#[cfg_attr(
    feature = "serde_support",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[derive(Clone, Copy, Debug)]
pub struct FullCoordinates {
    pub i: u8,
    pub q: i8,
    pub r: i8,
    pub s: i8,
}

impl FullCoordinates {
    pub fn from_memory_pos(pos: MemoryPos) -> Self {
        let qrs = pos.to_qrs_global();

        FullCoordinates {
            i: pos.0,
            q: qrs.q,
            r: qrs.r,
            s: qrs.s,
        }
    }


    pub fn from_raw_index(i: u8) -> Self {
        let pos = MemoryPos(i);
        let qrs = pos.to_qrs_global();

        FullCoordinates {
            i,
            q: qrs.q,
            r: qrs.r,
            s: qrs.s,
        }
    }

    pub fn to_memory_pos(&self) -> MemoryPos {
        MemoryPos(self.i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basics::COLORS;

    #[test]
    fn test_all_qrs() {
        for q in -4..8 {
            for r in -4..4 - q {
                let s = -1 - q - r;
                let qrs = QRSPos { q, r, s };
                #[allow(deprecated)]
                let mem = qrs.to_memory_pos_slow();

                assert_eq!(
                    qrs,
                    mem.to_qrs_global(),
                    "qrs: {:?} gave mem: {:?}",
                    qrs,
                    mem
                );

                let mem_fast = qrs.to_pos();

                assert_eq!(mem, mem_fast, "qrs: {:?} gave mem: {:?}", qrs, mem);
            }
        }

        for q in -7..4 {
            for r in -4 - q..4 {
                let s = -1 - q - r;
                let qrs = QRSPos { q, r, s };
                #[allow(deprecated)]
                let mem = qrs.to_memory_pos_slow();
                assert_eq!(
                    qrs,
                    mem.to_qrs_global(),
                    "qrs: {:?} gave mem: {:?}",
                    qrs,
                    mem
                );

                let mem_fast = qrs.to_pos();

                assert_eq!(mem, mem_fast, "qrs: {:?} gave mem: {:?}", qrs, mem);
            }
        }
    }

    #[test]
    fn verify_coordinates() {
        // Verify all sums equal -1
        for i in 0..96 {
            assert_eq!(Q[i] + R[i] + S[i], -1, "Sum error at index {}", i);
        }

        println!("q: {:?}", Q);
        println!("r: {:?}", R);
        println!("s: {:?}", S);
    }

    #[test]
    fn all() {
        for color in COLORS {
            let offset = color.get_offset();
            for i in 0..96 {
                let mem = MemoryPos(i as u8);
                let qrs = mem.to_qrs_local(color);
                let mem2 = qrs.to_pos().to_global(offset);
                assert_eq!(mem, mem2, "Color: {:?}, i: {}", color, i);
            }
        }
    }

    #[test]
    fn print_qr_to_mem() {
        println!("QR_TO_MEM: {:?}", QR_TO_MEM);
    }
}
