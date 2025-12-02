use bincode::{config::standard, encode_into_std_write};
use game_tri_chess::basics::{Color, MemoryPos, MemorySlot, Piece};
use game_tri_chess::chess_game::{State, TriHexChess};
use game_tri_chess::constants::{ALL_DIR_V, LINE_V};
use game_tri_chess::moves::{ChessMoveStore, MoveType, PseudoLegalMove};
use game_tri_chess::repetition_history::RepetitionHistory;
use oracle_tri_chess::tablebase::{
    CanonicalKey, CanonicalKeyKqk, CanonicalKeyKrk, Endgame, GameResult, Tablebase, TablebaseValue,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

fn main() {
    let endgame_to_generate = Endgame::Kqk;

    let (strong_piece, output_file) = match endgame_to_generate {
        Endgame::Kqk => (Piece::Queen, "kqk_tablebase.bin"),
        Endgame::Krk => (Piece::Rook, "krk_tablebase.bin"),
    };

    println!("Starting {:?} tablebase generation...", endgame_to_generate);
    let start_time = Instant::now();

    let strong_side = Color::White;
    let weak_side = Color::Gray; // Using Gray as the second color

    println!("\nStep 1: Generating all legal positions...");
    let all_positions = generate_all_positions(strong_side, weak_side, strong_piece);
    println!("-> Generated {} unique positions.", all_positions.len());

    println!("\nStep 2: Finding all checkmate and stalemate positions...");
    let (checkmates, stalemates) = find_terminal_positions(&all_positions);
    println!(
        "-> Found {} checkmates and {} stalemates.",
        checkmates.len(),
        stalemates.len()
    );

    println!("\nStep 3: Running retrograde analysis...");
    let tablebase = build_tablebase(
        &all_positions,
        checkmates,
        stalemates,
        strong_side,
        weak_side,
        strong_piece,
    );
    let duration = start_time.elapsed();
    println!("\nTablebase generation complete in {:?}.", duration);

    println!("\nStep 4: Saving tablebase to file '{}'...", output_file);
    let file = File::create(output_file).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    let config = standard();
    encode_into_std_write(&tablebase, &mut writer, config).expect("Failed to serialize tablebase");
    println!("-> Save complete.");

    print_tablebase_summary(&tablebase);
}

fn generate_all_positions(
    strong_side: Color,
    weak_side: Color,
    strong_piece: Piece,
) -> HashSet<TriHexChess> {
    let mut positions = HashSet::new();

    for sk_idx in 0..96 {
        for sp_idx in 0..96 {
            if sk_idx == sp_idx {
                continue;
            }
            for wk_idx in 0..96 {
                if wk_idx == sk_idx || wk_idx == sp_idx {
                    continue;
                }

                let sk_pos = MemoryPos(sk_idx);
                let wk_pos = MemoryPos(wk_idx);

                if sk_pos
                    .to_qrs_local(Color::White)
                    .distance(&wk_pos.to_qrs_local(Color::White))
                    <= 1
                {
                    continue;
                }

                let mut base_pos = TriHexChess {
                    state: State::empty(Color::White),
                    zobrist_hash: 0,
                    is_using_grace_period: false,
                    repetition_history: RepetitionHistory::default(),
                };

                base_pos.state.buffer[sk_idx as usize] = MemorySlot::new(strong_side, Piece::King);
                base_pos.state.buffer[sp_idx as usize] = MemorySlot::new(strong_side, strong_piece);
                base_pos.state.buffer[wk_idx as usize] = MemorySlot::new(weak_side, Piece::King);

                let mut strong_to_move = base_pos.clone();
                strong_to_move.set_turn(strong_side);
                positions.insert(strong_to_move);

                let mut weak_to_move = base_pos;
                weak_to_move.set_turn(weak_side);
                positions.insert(weak_to_move);
            }
        }
    }
    positions
}

/// The main retrograde analysis function.
pub fn build_tablebase(
    all_positions: &HashSet<TriHexChess>,
    checkmates: Vec<TriHexChess>,
    stalemates: Vec<TriHexChess>,
    strong_side: Color,
    weak_side: Color,
    strong_piece: Piece,
) -> Tablebase {
    let mut table: Tablebase = HashMap::with_capacity(all_positions.len());
    let mut queue: VecDeque<TriHexChess> = VecDeque::new();
    let to_key = |pos: &TriHexChess| to_canonical_key(pos, strong_side, weak_side, strong_piece);

    for pos in checkmates {
        if let Some(key) = to_key(&pos) {
            let value = TablebaseValue {
                result: GameResult::Loss,
                distance: 0,
                best_move_from: 0,
                best_move_to: 0,
            };
            if table.insert(key, value).is_none() {
                queue.push_back(pos);
            }
        }
    }
    for pos in stalemates {
        if let Some(key) = to_key(&pos) {
            let value = TablebaseValue {
                result: GameResult::Draw,
                distance: 0,
                best_move_from: 0,
                best_move_to: 0,
            };
            if table.insert(key, value).is_none() {
                queue.push_back(pos);
            }
        }
    }
    println!("Seeded tablebase with {} terminal positions.", queue.len());

    let mut positions_processed = 0;
    while let Some(pos) = queue.pop_front() {
        positions_processed += 1;
        if positions_processed % 100_000 == 0 {
            println!(
                "Processed {} positions... Table size: {}, Queue size: {}",
                positions_processed,
                table.len(),
                queue.len()
            );
        }

        let current_value = *table.get(&to_key(&pos).unwrap()).unwrap();
        let predecessors = generate_predecessors(&pos, strong_side, weak_side, strong_piece);

        for (pred, forward_move) in predecessors {
            let pred_key = match to_key(&pred) {
                Some(k) => k,
                None => continue,
            };
            if table.contains_key(&pred_key) {
                continue;
            }

            if current_value.result == GameResult::Loss {
                let new_value = TablebaseValue {
                    result: GameResult::Win,
                    distance: current_value.distance + 1,
                    best_move_from: forward_move.from.0,
                    best_move_to: forward_move.to.0,
                };
                table.insert(pred_key, new_value);
                queue.push_back(pred);
                continue;
            }

            //  Check if all successors of the predecessor are now classified.
            if all_successors_are_classified(&pred, &table, strong_side, weak_side, strong_piece) {
                if are_all_moves_losing(&pred, &table, strong_side, weak_side, strong_piece) {
                    // This position is a guaranteed loss. Now, find the best move to prolong the game.
                    let (best_move, max_dist) = find_best_defensive_move(
                        &pred,
                        &table,
                        strong_side,
                        weak_side,
                        strong_piece,
                    );

                    let new_value = TablebaseValue {
                        result: GameResult::Loss,
                        distance: max_dist + 1,
                        best_move_from: best_move.from.0,
                        best_move_to: best_move.to.0,
                    };
                    table.insert(pred_key, new_value);
                    queue.push_back(pred);
                } else {
                    // Not a forced loss -> Draw. (No change here)
                    let new_value = TablebaseValue {
                        result: GameResult::Draw,
                        distance: 0,
                        best_move_from: 0,
                        best_move_to: 0,
                    };
                    table.insert(pred_key, new_value);
                    queue.push_back(pred);
                }
            }
        }
    }
    table
}

fn find_best_defensive_move(
    pos: &TriHexChess,
    table: &Tablebase,
    strong_side: Color,
    weak_side: Color,
    strong_piece: Piece,
) -> (PseudoLegalMove, u8) {
    let mut game = pos.clone();
    let mut move_store = ChessMoveStore::default();
    game.update_pseudo_moves(&mut move_store, false);

    let mut best_move = move_store
        .get(0)
        .expect("Losing position must have legal moves")
        .clone();
    let mut longest_dtm = 0;

    for m in move_store.iter() {
        let mut next_pos = game.clone();
        next_pos.commit_move(m, &move_store);
        let opponent_color = if pos.get_turn().unwrap() == strong_side {
            weak_side
        } else {
            strong_side
        };
        next_pos.set_turn(opponent_color);

        let key = to_canonical_key(&next_pos, strong_side, weak_side, strong_piece).unwrap();
        let successor_value = table.get(&key).unwrap();

        // We are looking for the successor position (which is a win for the opponent)
        // that has the largest distance.
        if successor_value.distance > longest_dtm {
            longest_dtm = successor_value.distance;
            best_move = *m;
        }
    }

    (best_move, longest_dtm)
}

fn to_canonical_key(
    pos: &TriHexChess,
    strong_side_color: Color,
    weak_side_color: Color,
    strong_piece: Piece,
) -> Option<CanonicalKey> {
    let mut strong_king_pos = None;
    let mut weak_king_pos = None;
    let mut strong_piece_pos = None;
    let current_turn = pos.get_turn()?;

    for i in 0u8..96 {
        if let Some((color, piece)) = pos.state.buffer[i as usize].get() {
            if color == strong_side_color {
                if piece == Piece::King {
                    strong_king_pos = Some(i);
                } else if piece == strong_piece {
                    strong_piece_pos = Some(i);
                } else {
                    return None;
                }
            } else if color == weak_side_color {
                if piece == Piece::King {
                    weak_king_pos = Some(i);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
    }

    let skp = strong_king_pos?;
    let spp = strong_piece_pos?;
    let wkp = weak_king_pos?;
    let is_strong_turn = current_turn == strong_side_color;

    match strong_piece {
        Piece::Queen => Some(CanonicalKey::Kqk(CanonicalKeyKqk {
            strong_king_pos: skp,
            strong_queen_pos: spp,
            weak_king_pos: wkp,
            is_strong_side_to_move: is_strong_turn,
        })),
        Piece::Rook => Some(CanonicalKey::Krk(CanonicalKeyKrk {
            strong_king_pos: skp,
            strong_rook_pos: spp,
            weak_king_pos: wkp,
            is_strong_side_to_move: is_strong_turn,
        })),
        _ => None,
    }
}

fn generate_predecessors(
    pos: &TriHexChess,
    strong_side: Color,
    weak_side: Color,
    strong_piece: Piece,
) -> Vec<(TriHexChess, PseudoLegalMove)> {
    let mut predecessors = Vec::new();
    if pos.is_over() {
        return predecessors;
    }

    let previous_player = if pos.get_turn().unwrap() == strong_side {
        weak_side
    } else {
        strong_side
    };

    let mut moved_piece_candidates = Vec::new();
    for i in 0..96 {
        let mem_pos = MemoryPos(i as u8);
        if let Some((player, piece)) = pos.state.buffer[mem_pos.0 as usize].get() {
            if player == previous_player && (piece == Piece::King || piece == strong_piece) {
                moved_piece_candidates.push((piece, mem_pos));
            }
        }
    }

    for (piece, to_pos) in moved_piece_candidates {
        let from_squares = get_potential_from_squares(pos, piece, to_pos);
        for from_pos in from_squares {
            if !pos.state.buffer[from_pos.0 as usize].is_empty() {
                continue;
            }

            let mut pred = pos.clone();
            pred.state.buffer[from_pos.0 as usize] = pred.state.buffer[to_pos.0 as usize];
            pred.state.buffer[to_pos.0 as usize] = MemorySlot::empty();
            pred.set_turn(previous_player);

            let mut move_store = ChessMoveStore::default();
            let mut temp_pred = pred.clone();
            temp_pred.update_pseudo_moves(&mut move_store, false);

            let forward_move = PseudoLegalMove {
                from: from_pos,
                to: to_pos,
                move_type: MoveType::Move,
            };
            if move_store.contains(forward_move) {
                predecessors.push((pred, forward_move));
            }
        }
    }
    predecessors
}

fn get_potential_from_squares(
    pos: &TriHexChess,
    piece: Piece,
    to_pos: MemoryPos,
) -> Vec<MemoryPos> {
    let mut from_squares = Vec::new();
    let color = pos.state.buffer[to_pos.0 as usize].player().unwrap();
    let local_qrs_to = to_pos.to_qrs_local(color);
    let offset = color.get_offset();

    let directions = match piece {
        Piece::King | Piece::Queen => &ALL_DIR_V[..],
        Piece::Rook => &LINE_V[..],
        _ => return from_squares,
    };

    for &(q, r, s) in directions {
        let mut current_qrs = local_qrs_to.add(-q, -r, -s);
        'ray_trace: while current_qrs.is_in() {
            let potential_from_pos = current_qrs.to_pos().to_global(offset);
            if !pos.state.buffer[potential_from_pos.0 as usize].is_empty() {
                break 'ray_trace;
            }
            from_squares.push(potential_from_pos);
            if piece == Piece::King {
                break 'ray_trace;
            }
            current_qrs = current_qrs.add(-q, -r, -s);
        }
    }
    from_squares
}

fn find_terminal_positions(
    all_positions: &HashSet<TriHexChess>,
) -> (Vec<TriHexChess>, Vec<TriHexChess>) {
    let mut checkmates = Vec::new();
    let mut stalemates = Vec::new();
    let mut move_store = ChessMoveStore::default();
    for pos in all_positions {
        let mut game = pos.clone();
        game.update_pseudo_moves(&mut move_store, false);
        if move_store.is_empty() {
            if move_store.turn_cache.is_check {
                checkmates.push(pos.clone());
            } else {
                stalemates.push(pos.clone());
            }
        }
    }
    (checkmates, stalemates)
}

fn all_successors_are_classified(
    pos: &TriHexChess,
    table: &Tablebase,
    s_color: Color,
    w_color: Color,
    s_piece: Piece,
) -> bool {
    let mut game = pos.clone();
    let mut move_store = ChessMoveStore::default();
    game.update_pseudo_moves(&mut move_store, false);
    for m in move_store.iter() {
        let mut next_pos = game.clone();
        next_pos.commit_move(m, &move_store);
        next_pos.set_turn(if pos.get_turn().unwrap() == s_color {
            w_color
        } else {
            s_color
        });
        if let Some(key) = to_canonical_key(&next_pos, s_color, w_color, s_piece) {
            if !table.contains_key(&key) {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

fn are_all_moves_losing(
    pos: &TriHexChess,
    table: &Tablebase,
    s_color: Color,
    w_color: Color,
    s_piece: Piece,
) -> bool {
    let mut game = pos.clone();
    let mut move_store = ChessMoveStore::default();
    game.update_pseudo_moves(&mut move_store, false);
    if move_store.is_empty() {
        return false;
    }
    for m in move_store.iter() {
        let mut next_pos = game.clone();
        next_pos.commit_move(m, &move_store);
        next_pos.set_turn(if pos.get_turn().unwrap() == s_color {
            w_color
        } else {
            s_color
        });
        let key = match to_canonical_key(&next_pos, s_color, w_color, s_piece) {
            Some(k) => k,
            None => return false,
        };
        match table.get(&key) {
            Some(entry) if entry.result == GameResult::Win => continue,
            _ => return false,
        }
    }
    true
}

fn print_tablebase_summary(table: &Tablebase) {
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut max_dtm = 0;

    for value in table.values() {
        match value.result {
            GameResult::Win => {
                wins += 1;
                if value.distance > max_dtm {
                    max_dtm = value.distance;
                }
            }
            GameResult::Loss => losses += 1,
            GameResult::Draw => draws += 1,
        }
    }
    println!("\n--- Tablebase Generation Summary ---");
    println!("Total canonical positions: {}", table.len());
    println!("  Wins (for side to move):   {}", wins);
    println!("  Losses (for side to move): {}", losses);
    println!("  Draws (for side to move):  {}", draws);
    println!("------------------------------------");
    println!("Longest mate found: {} plies (half-moves)", max_dtm);
}
