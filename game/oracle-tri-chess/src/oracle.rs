use crate::tablebase::{CanonicalKey, CanonicalKeyKqk, GameResult, Tablebase};
use crate::tablebase::{CanonicalKeyKrk};
use bincode::config::standard;
use bincode::decode_from_std_read;
use game_tri_chess::basics::{Color, MemoryPos, Piece};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{MoveType, PseudoLegalMove};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

#[derive(Debug)]
pub struct TriHexEndgameOracle {
    pub kqk_table: Tablebase,
    pub krk_table: Tablebase,
}

#[derive(Debug)]
pub struct OracleResult {
    pub outcome: GameResult,
    pub distance_to_mate: u8,
    pub best_move_notation: Option<String>,
}

#[derive(Debug)]
pub enum OracleErrorInner {
    Io(std::io::Error),
    Deserialization(bincode::error::DecodeError),
    Fen(String),
    UnsupportedMaterial(String),
    PositionNotInTable,
}

/// Holds information about the detected endgame.
pub enum DetectedEndgame {
    Kqk { strong: Color, weak: Color },
    Krk { strong: Color, weak: Color },
}

impl TriHexEndgameOracle {
    /// Loads all available tablebases from disk.
    pub fn new(kqk_path: &str, krk_path: &str) -> Result<Self, OracleErrorInner> {
        println!("Loading endgame tablebases...");
        let kqk_table = load_table(kqk_path)?;
        let krk_table = load_table(krk_path)?;
        Ok(TriHexEndgameOracle {
            kqk_table,
            krk_table,
        })
    }

    /// Queries the oracle with a FEN string.
    pub fn query_fen(&self, fen: &str) -> Result<OracleResult, OracleErrorInner> {
        let pos = TriHexChess::new_with_fen(fen.as_bytes(), false).map_err(OracleErrorInner::Fen)?;
        self.query_position(&pos)
    }

    /// Main query logic: detects endgame, creates key, and probes the correct table.
    pub fn query_position(&self, pos: &TriHexChess) -> Result<OracleResult, OracleErrorInner> {
        // 1. Detect which endgame this position belongs to.
        let detected_endgame = detect_endgame(pos)?;

        // 2. Create the appropriate canonical key and probe the correct table.
        let (key, table) = match detected_endgame {
            DetectedEndgame::Kqk { strong, weak } => (
                to_canonical_key(pos, strong, weak, Piece::Queen).unwrap(),
                &self.kqk_table,
            ),
            DetectedEndgame::Krk { strong, weak } => (
                to_canonical_key(pos, strong, weak, Piece::Rook).unwrap(),
                &self.krk_table,
            ),
        };

        let value = table.get(&key).ok_or(OracleErrorInner::PositionNotInTable)?;

        // 3. Format the result for the user. The logic is now simpler because the
        // generator has already calculated the best move for both wins and losses.
        let mut best_move_notation = None;
        if value.result != GameResult::Draw {
            let best_move = PseudoLegalMove {
                from: MemoryPos(value.best_move_from),
                to: MemoryPos(value.best_move_to),
                move_type: MoveType::Move,
            };
            best_move_notation = Some(best_move.notation_lan(&pos.state.buffer));
        }

        Ok(OracleResult {
            outcome: value.result,
            distance_to_mate: value.distance,
            best_move_notation,
        })
    }
}

// --- Helper Functions ---

/// Helper to load a single tablebase file.
fn load_table(path: &str) -> Result<Tablebase, OracleErrorInner> {
    let start = Instant::now();
    println!(" -> Loading '{}'...", path);
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            println!("    ...warning: could not open file: {}", e);
            return Ok(HashMap::new()); // Return an empty table if file doesn't exist
        }
    };

    let mut reader = BufReader::new(file);
    let config = standard();
    let table: Tablebase =
        decode_from_std_read(&mut reader, config).map_err(OracleErrorInner::Deserialization)?;
    println!(
        "    ...loaded {} entries in {:?}.",
        table.len(),
        start.elapsed()
    );
    Ok(table)
}

/// Dynamically determines which endgame a position belongs to from its material.
pub fn detect_endgame(pos: &TriHexChess) -> Result<DetectedEndgame, OracleErrorInner> {
    let mut pieces_by_color: HashMap<Color, Vec<Piece>> = HashMap::new();
    for i in 0..96 {
        if let Some((color, piece)) = pos.state.buffer[i as usize].get() {
            pieces_by_color.entry(color).or_default().push(piece);
        }
    }

    if pieces_by_color.len() != 2 {
        return Err(OracleErrorInner::UnsupportedMaterial(format!(
            "Expected 2 players, found {}",
            pieces_by_color.len()
        )));
    }

    let mut strong_side = None;
    let mut weak_side = None;
    let mut strong_piece = None;

    for (color, pieces) in &pieces_by_color {
        if pieces.len() == 2 {
            strong_side = Some(*color);
            if pieces.contains(&Piece::King) {
                strong_piece = pieces.iter().find(|&&p| p != Piece::King).copied();
            } else {
                return Err(OracleErrorInner::UnsupportedMaterial(
                    "Strong side has no king.".into(),
                ));
            }
        } else if pieces.len() == 1 {
            weak_side = Some(*color);
            if pieces[0] != Piece::King {
                return Err(OracleErrorInner::UnsupportedMaterial(
                    "Weak side has a non-king piece.".into(),
                ));
            }
        }
    }

    let strong = strong_side.ok_or_else(|| {
        OracleErrorInner::UnsupportedMaterial(
            "Could not identify strong side (side with 2 pieces).".into(),
        )
    })?;
    let weak = weak_side.ok_or_else(|| {
        OracleErrorInner::UnsupportedMaterial("Could not identify weak side (side with 1 piece).".into())
    })?;

    match strong_piece {
        Some(Piece::Queen) => Ok(DetectedEndgame::Kqk { strong, weak }),
        Some(Piece::Rook) => Ok(DetectedEndgame::Krk { strong, weak }),
        Some(p) => Err(OracleErrorInner::UnsupportedMaterial(format!(
            "Unsupported strong piece: {:?}",
            p
        ))),
        None => Err(OracleErrorInner::UnsupportedMaterial(
            "Strong side has two kings.".into(),
        )),
    }
}

/// The canonical key conversion function, which must be identical to the generator's.
pub fn to_canonical_key(
    pos: &TriHexChess,
    s_color: Color,
    w_color: Color,
    s_piece: Piece,
) -> Option<CanonicalKey> {
    let mut skp = None;
    let mut wkp = None;
    let mut spp = None;
    let turn = pos.get_turn()?;
    for i in 0u8..96 {
        if let Some((c, p)) = pos.state.buffer[i as usize].get() {
            if c == s_color {
                if p == Piece::King {
                    skp = Some(i);
                } else if p == s_piece {
                    spp = Some(i);
                }
            } else if c == w_color && p == Piece::King {
                wkp = Some(i);
            }
        }
    }
    let is_strong_turn = turn == s_color;
    match s_piece {
        Piece::Queen => Some(CanonicalKey::Kqk(CanonicalKeyKqk {
            strong_king_pos: skp?,
            strong_queen_pos: spp?,
            weak_king_pos: wkp?,
            is_strong_side_to_move: is_strong_turn,
        })),
        Piece::Rook => Some(CanonicalKey::Krk(CanonicalKeyKrk {
            strong_king_pos: skp?,
            strong_rook_pos: spp?,
            weak_king_pos: wkp?,
            is_strong_side_to_move: is_strong_turn,
        })),
        _ => None,
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn load_tables() {
        let oracle = super::TriHexEndgameOracle::new("./../../tablebases/kqk_tablebase.bin", "./../../tablebases/krk_tablebase.bin").unwrap();
        assert!(oracle.kqk_table.len() > 0);
        assert!(oracle.krk_table.len() > 0);

        let fen_1 = "r3k3/8/8/8/X/X X/4k3/8/8/8/X X/X/X W qkqkqk --- 0 1";
        let fen_2 = "r7/8/3k4/8/X/X X/5k2/8/8/8/X X/X/X W ----qk --- 2 2";
        let fen_3 = "r7/8/8/8/8/8/8/4k3/X X/8/8/2k5/8/X X/X/X W ----qk --- 6 4";
        let fen_4 = "8/8/6r1/8/8/8/8/4k3/X X/8/8/8/1k6/X X/X/X W ----qk --- 8 5";

        let result_1 = oracle.query_fen(fen_1).unwrap();
        println!("Result for position 1: {:?}", result_1);

        let result_2 = oracle.query_fen(fen_2).unwrap();
        println!("Result for position 2: {:?}", result_2);

        let result_3 = oracle.query_fen(fen_3).unwrap();
        println!("Result for position 3: {:?}", result_3);

        let result_4 = oracle.query_fen(fen_4).unwrap();
        println!("Result for position 4: {:?}", result_4);
    }
}
