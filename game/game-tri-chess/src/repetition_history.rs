#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

const REPETITION_HISTORY_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct RepetitionHistory {
    hashes: [u64; REPETITION_HISTORY_SIZE],
}

impl Default for RepetitionHistory {
    fn default() -> Self {
        Self {
            hashes: [0; REPETITION_HISTORY_SIZE],
        }
    }
}

impl RepetitionHistory {
    /// Records a new Zobrist hash in the history.
    ///
    /// * `history_len` - third move counter
    /// * `hash` - zobrist hash of the new board state.
    pub fn record(&mut self, history_len: u16, hash: u64) {
        // We only record if at least one reversible move has been made.
        // `history_len` is 1-based after being incremented.
        if history_len > 0 {
            // Use modulo for a circular buffer effect, ensuring we don't go out of bounds.
            let index = (history_len as usize - 1) % REPETITION_HISTORY_SIZE;
            self.hashes[index] = hash;
        }
    }

    /// Counts the number of times a given hash appears in the relevant history.
    ///
    /// # Arguments
    /// * `history_len` - The number of reversible half-moves to check against (your `third_move` counter).
    /// * `hash` - The Zobrist hash to search for.
    ///
    /// # Returns
    /// The total number of occurrences (1 means it's seen for the first time in the current sequence,
    /// 2 means it's a repeat, 3 means it's a threefold repetition draw).
    pub fn count_occurrences(&self, history_len: u16, hash: u64) -> u8 {
        // If no reversible moves have been made, no repetitions are possible.
        if history_len == 1 {
            return 0;
        }

        let mut count = 0;
        let history_len_usize = history_len as usize;

        // Determine how many past positions are relevant and stored.
        let num_to_check = std::cmp::min(history_len_usize, REPETITION_HISTORY_SIZE);

        // Iterate through the last `num_to_check` positions.
        for i in 0..num_to_check {
            // Calculate the index in the circular buffer.
            // `history_len_usize - 1` is the index of the newest item.
            // `history_len_usize - 1 - i` walks backwards through time.
            let index = (history_len_usize - 1 - i) % REPETITION_HISTORY_SIZE;
            if self.hashes[index] == hash {
                count += 1;
            }
        }
        count
    }
}