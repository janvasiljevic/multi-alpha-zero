use crate::basics::{CastleFlags, Color, EnPassantState, MemoryBuffer, Piece, PlayerState, COLORS};
use crate::chess_game::State;
use crate::phase::Phase;
use crate::pos::MemoryPos;

impl State {
    /// 90ns per call (10_000_000 calls). With alloc it's around 1.5 to 2 times slower
    pub fn update_from_fen(&mut self, fen: &[u8]) -> Result<(), String> {
        let parts: Vec<&[u8]> = fen.split(|&b| b == b' ').collect();

        if parts.len() != 8 {
            return Err(format!(
                "Invalid FEN: Expected 8 parts, got {}",
                parts.len()
            ));
        }

        self.buffer.clear();
        self.buffer.parse(&parts[0..3])?;
        self.phase.parse(parts[3])?;

        match &mut self.phase {
            Phase::Normal(player_state) => {
                for (_, mem) in self.buffer.non_empty_iter() {
                    player_state.set_player(mem.player().unwrap());
                }

                if !player_state.is_present(player_state.get_turn()) {
                    return Err("No pieces for the current player - Invalid FEN".to_string());
                }
            }
            _ => {
                // Since `self.phase.parse` explicitly sets ::Normal, this should be unreachable
                unreachable!()
            }
        }

        self.castle.parse(parts[4])?;
        self.en_passant.parse(parts[5])?;
        
        // todo: implement the actual logic for this!
        self.third_move.parse(parts[6])?;


        self.turn_counter.parse(parts[7])?;

        Ok(())
    }
}

pub trait FenParser<T: ?Sized> {
    fn parse(&mut self, sub_fen: T) -> Result<&mut Self, String>;
}

impl FenParser<&[&[u8]]> for MemoryBuffer {
    fn parse(&mut self, sub_fen: &[&[u8]]) -> Result<&mut Self, String> {
        for color in COLORS {
            let mut piece_index = 0u8;

            let component = sub_fen[color as usize];

            for char in component.iter() {
                match char {
                    b'/' => {}
                    b'X' => {
                        piece_index += 32;
                    }
                    b'1'..=b'8' => {
                        piece_index += *char - b'0';
                    }
                    _ => match Piece::from_char(*char as char) {
                        Some(piece) => {
                            debug_assert!(
                                self[piece_index as usize].is_empty(),
                                "The memory slot is not empty"
                            );
                            self[piece_index as usize].set(color, piece);
                            piece_index += 1;
                        }
                        None => {
                            return Err("Unexpected character in memory slot".to_string());
                        }
                    },
                }
            }
        }

        Ok(self)
    }
}

impl FenParser<&[u8]> for u16 {
    // fn parse(&mut self, value: &[u8]) -> Result<&mut Self, String> {
    //     match value.len() {
    //         1 => *self = value[0] as u16 - b'0' as u16,
    //         2 => *self = (value[0] as u16 - b'0' as u16) * 10 + (value[1] as u16 - b'0' as u16),
    //         _ => return Err(format!("Invalid number (u16): {:?}", value)),
    //     }
    //
    //     Ok(self)
    // }
    //
    fn parse(&mut self, value: &[u8]) -> Result<&mut Self, String> {
        let s = std::str::from_utf8(value)
            .map_err(|_| format!("Invalid UTF-8: {:?}", value))?;
        *self = s.parse::<u16>()
            .map_err(|_| format!("Invalid number (u16): {:?}", value))?;
        Ok(self)
    }
}

impl FenParser<&[u8]> for Phase {
    fn parse(&mut self, value: &[u8]) -> Result<&mut Self, String> {
        let mut player_state = PlayerState::default();
        match value[0] {
            b'W' => player_state.set_turn(Color::White),
            b'G' => player_state.set_turn(Color::Gray),
            b'B' => player_state.set_turn(Color::Black),
            _ => return Err("Invalid Color".to_string()),
        };

        *self = Phase::Normal(player_state);

        Ok(self)
    }
}

impl FenParser<&[u8]> for CastleFlags {
    fn parse(&mut self, value: &[u8]) -> Result<&mut Self, String> {
        if value.len() != 6 {
            return Err("Invalid castle flags. Should be len of 6".to_string());
        }

        for color in COLORS {
            self.set_queen_side(color, value[(color as usize) * 2] == b'q');
            self.set_king_side(color, value[(color as usize) * 2 + 1] == b'k');
        }

        Ok(self)
    }
}

impl FenParser<&[u8]> for EnPassantState {
    fn parse(&mut self, value: &[u8]) -> Result<&mut Self, String> {
        if value.len() != 3 {
            return Err("Invalid en passant".to_string());
        }

        for color in COLORS {
            match value[color as usize] {
                b'-' => {
                    self.remove(color);
                }
                b'1'..=b'8' => {
                    let i = value[color as usize] - b'1' + color.get_offset() + 16;

                    self.set_pos(color, MemoryPos(i));
                }
                _ => {
                    return Err("Invalid en passant".to_string());
                }
            }
        }

        Ok(self)
    }
}

// TODO: This should be an 'impl' for State
// Also instead of using String::with_capacity(200) I think we can use
// something more lik &mut [u8; X] and then convert it to a string,
// or at least a Vec<u8> and then convert it to a string, since we are only dealing with ASCII
pub fn to_fen(state: &State) -> String {
    let mut builder_string = String::with_capacity(200);

    for current_color in COLORS {
        'tri_loop: for tri in 0..3 {
            // Check if the following 32 memory slots are empty. If they are, add an 'X' to FEN.
            'x_32_loop: for j in 0..32 {
                if state.buffer[tri * 32 + j].player() == Some(current_color) {
                    break 'x_32_loop;
                }
                if j == 31 {
                    builder_string.push('X');

                    if tri == 2 {
                        break 'tri_loop;
                    }

                    builder_string.push('/');
                    continue 'tri_loop;
                }
            }

            for row in 0..4 {
                let mut empty_count = 0;

                for col in 0..8 {
                    let mem = state.buffer[tri * 32 + row * 8 + col];

                    match mem.player() {
                        Some(color) => {
                            if color == current_color {
                                if empty_count > 0 {
                                    builder_string.push((empty_count + b'0') as char);
                                    empty_count = 0;
                                }

                                builder_string.push(mem.piece().unwrap().to_char());
                            } else {
                                empty_count += 1;
                            }
                        }
                        None => {
                            empty_count += 1;
                        }
                    }
                }

                if empty_count > 0 {
                    builder_string.push((empty_count + b'0') as char);
                }

                // Always add a slash except for the last rank
                if !(tri == 2 && row == 3) {
                    builder_string.push('/');
                }
            }
        }

        builder_string.push(' ');
    }

    builder_string.push_str(&state.phase.to_string());
    builder_string.push(' ');
    builder_string.push_str(&state.castle.to_string());
    builder_string.push(' ');
    builder_string.push_str(&state.en_passant.to_string());
    builder_string.push(' ');
    builder_string.push_str(&state.third_move.to_string());
    builder_string.push(' ');
    builder_string.push_str(&state.turn_counter.to_string());

    builder_string
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_en_passant_parsing() {
        let en_passant_cases: &[(&str, EnPassantState)] = &[
            ("1-1", EnPassantState::new_from_test([1, 0, 1])),
            ("---", EnPassantState::new_from_test([0, 0, 0])),
            ("1-2", EnPassantState::new_from_test([1, 0, 2])),
            ("777", EnPassantState::new_from_test([7, 7, 7])),
            ("888", EnPassantState::new_from_test([8, 8, 8])),
        ];

        for (label, expected_en_passant) in en_passant_cases {
            println!("Testing en passant: {} - {}", label, expected_en_passant); // Helpful for debugging

            let mut en_passant = EnPassantState::default();
            let result = en_passant.parse(label.as_bytes());

            match result {
                Ok(parsed_en_passant) => {
                    assert_eq!(
                        *parsed_en_passant, *expected_en_passant,
                        "Incorrect en passant for {}",
                        label
                    );
                }
                Err(e) => {
                    panic!("Failed to parse en passant for {}: {}", label, e);
                }
            }
        }
    }

    #[test]
    fn test_flags_parsing() {
        const FLAGS: &[(&str, CastleFlags)] = &[("qkqkqk", CastleFlags::new(0b00111111))];

        for (label, expected_flags) in FLAGS {
            println!("Testing flags: {} - {}", label, expected_flags); // Helpful for debugging

            let mut flags = CastleFlags::default();
            let result = flags.parse(label.as_bytes());

            match result {
                Ok(parsed_flags) => {
                    assert_eq!(
                        *parsed_flags, *expected_flags,
                        "Incorrect flags for {}",
                        label
                    );
                }
                Err(e) => {
                    panic!("Failed to parse flags for {}: {}", label, e);
                }
            }
        }
    }

    #[test]
    fn test_from_fen_all_cases() {
        const FENS: &[(&str, &str)] = &[
            (
                "Normal",
                "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1",
            ),
            (
                "Only pawns",
                "8/pppppppp/8/8/X/X X/8/pppppppp/8/8/X X/X/8/pppppppp/8/8 W qkqkqk --- 0 1",
            ),
            (
                "Castle test",
                "r3k2r/8/pppp4/8/X/X X/r3k2r/8/pppp4/8/X X/X/r3k2r/8/pppp4/8 W qkqkqk --- 0 1",
            ),
            (
                "Checking pawn checks",
                "8/pppp1ppp/4p3/2k5/X/X X/8/pppp1ppp/3kp3/8/X X/X/8/pppp1p1p/3kp1p1/8 W qkqkqk --- 0 4",
            ),
            (
                "Promotion test",
                "X/4p3/5p2/5p2/6p1/2p5/2p5/2p5/1p6 X/X/X X/X/X W qkqkqk --- 0 1",
            ),
            (
                "Pawns check",
                "rnbq1bnr/pppppppp/3k4/8/X/X X/rnbqkbnr/pppp1ppp/8/4p3/X X/X/rnbq1bnr/pppppppp/5k2/8 W qkqkqk --- 0 2",
            ),
            (
                "Check",
                "rnbq1bnr/pppppppp/8/8/X/8/8/2k5/8 X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1",
            ),
            (
                "Checks 2",
                "rnbq1bnr/pppp1ppp/3k4/4p3/X/X 8/8/8/6q1/rnb1kbnr/pppp1ppp/8/4p3/X X/X/rnb1kbnr/pppp1ppp/8/4pq2 G qkqkqk --- 0 1",
            ),
            (
                "Testing",
                "rnbqkbnr/8/pppp4/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 W qkqkqk --- 0 1",
            ),
            (
                "Pin fun",
                "rnbqkbnr/pppppppp/8/8/X/X X/rnb1kbnr/pppppppp/8/5q2/X 8/8/q7/8/X/rnb1kbnr/pppppppp/8/8 W qkqkqk --- 0 1",
            ),
        ];

        for (label, fen) in FENS {
            println!("Testing FEN: {} - {}", label, fen); // Helpful for debugging

            let mut state = State::default();

            let result = state.update_from_fen(fen.as_bytes());

            match result {
                Ok(()) => {
                    println!("Parsed FEN: {}", to_fen(&state)); // Helpful for debugging
                    assert_eq!(
                        to_fen(&state),
                        fen.to_string(),
                        "Incorrect FEN for {}",
                        label
                    );
                }
                Err(e) => {
                    panic!("Failed to parse FEN for {}: {}", label, e);
                }
            }
        }
    }
}
