use crate::arena_config::{ArenaConfig, PlayerConfig};
use crate::arena_csv_writer::{write_2_player_arena_results_to_csv, write_arena_results_to_csv};
use crate::chess_arena::{AlphaZeroPlayer, ChessArenaPlayer, ChessInnerPlayer};
use crate::util::chess_mapper_from_network_shapes;
use anyhow::Result;
use game_tri_chess::chess_game::TriHexChess;
use itertools::Itertools;
use maz_core::mapping::{Board, Oracle, Outcome};
use maz_util::network::{alpha_zero_shapes, auto_non_cpu_model};
use oracle_tri_chess::oracle::TriHexEndgameOracle;
use rand::prelude::SliceRandom;
use rand::rng;
use rayon::prelude::*;
use skillratings::elo::{elo, EloConfig, EloRating};
use skillratings::Outcomes;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{info, warn};

/// Plays a single game of 2-player chess on a 3-player board.
fn play_game_2_player(
    players_map: &HashMap<usize, &ChessInnerPlayer>,
    config: &ArenaConfig,
    mut board: TriHexChess,
    oracle: &TriHexEndgameOracle,
) -> Outcome {
    let mut move_store = <TriHexChess as Board>::MoveStore::default();

    // Loop for a maximum number of moves to prevent infinite games.
    for _ in 0..750 {
        if board.is_terminal() {
            break;
        }

        if board.can_game_end_early() {
            return board
                .outcome(true)
                .expect("Forced early end must have an outcome");
        }

        let current_player_idx: usize = board.player_current().into();

        match oracle.probe(&board) {
            Ok(analysis) => {
                let outcome = analysis.outcome;
                info!(
                    "Finished by oracle with outcome {}. {}",
                    outcome,
                    board.fancy_debug()
                );

                return outcome;
            }
            Err(_) => {}
        }

        if let Some(current_player) = players_map.get(&current_player_idx) {
            let best_move = current_player.get_best_move(&board, config);
            board.fill_move_store(&mut move_store);
            board.play_move_mut_with_store(&best_move, &mut move_store, None);
        } else {
            panic!(
                "Game logic error: A non-participating player's turn was reached. FEN: {}",
                board.to_fen()
            );
        }
    }

    if !board.is_terminal() {
        info!(
            "Game reached move limit, forcing an end. FEN: {}",
            board.to_fen()
        );
        return board
            .outcome(true)
            .expect("Forced end must have an outcome");
    }

    board
        .outcome(false)
        .expect("Terminal board must have an outcome")
}

#[derive(Clone)]
struct TwoPlayerGameSetup {
    fen: &'static str,
    player1_idx: usize,
    player2_idx: usize,
    color1_board_idx: usize,
    color2_board_idx: usize,
}

fn generate_ascii_arena_table_2p_string(results: Vec<(String, EloRating)>) -> String {
    let mut output = String::new();
    output.push_str("\n--- Arena Finished ---\n");
    output.push_str(&format!(
        "{:<30} | {:>10}\n",
        "Player (Model)", "Elo Rating"
    ));
    output.push_str(&format!("{:-<31}|{:-<12}\n", "", ""));
    for (name, rating) in results {
        output.push_str(&format!("{:<30} | {:>10.2}\n", name, rating.rating));
    }
    output.push_str("----------------------------------------------\n");
    output
}

pub fn run_arena_chess_2_player(
    config: &ArenaConfig,
    maybe_device_id: Option<usize>,
    _sessions_per_gpu: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Starting 2-Player Chess Arena for config: '{}'",
        config.config_unique_name
    );
    let start_time = Instant::now();

    let players: Vec<ChessInnerPlayer> = config
        .players
        .iter()
        .map(
            |player_config| -> Result<ChessInnerPlayer, Box<dyn std::error::Error>> {
                match player_config {
                    PlayerConfig::AlphaZeroPlayer { onnx_path: path } => {
                        let name = path.split('/').last().expect("Model path must have a name");
                        info!(" > Loading '{}' for GPU {:?}", name, maybe_device_id);
                        let model = auto_non_cpu_model(path, maybe_device_id)?;
                        let model_shapes = alpha_zero_shapes(&model)?;

                        Ok(ChessInnerPlayer::AlphaZero(AlphaZeroPlayer {
                            name: name.to_string(),
                            model: Mutex::new(model),
                            mapper: chess_mapper_from_network_shapes(model_shapes)?,
                        }))
                    }
                    _ => Err(format!(
                        "Unsupported player type for Chess Arena: {:?}",
                        player_config
                    )
                    .into()),
                }
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    if players.len() < 2 {
        return Err(format!(
            "2-Player Chess arena requires at least 2 players. Found {}.",
            players.len()
        )
        .into());
    }

    let oracle = match TriHexEndgameOracle::new(
        "./tablebases/kqk_tablebase.bin",
        "./tablebases/krk_tablebase.bin",
    ) {
        Ok(oracle) => Some(Arc::new(oracle)),
        Err(e) => {
            // This is a fatal error if we expect an oracle for chess.
            panic!("Failed to initialize TriHexEndgameOracle: {:?}", e);
        }
    };

    // Standard ELO, not Weng-Lin
    let ratings = Mutex::new(
        players
            .iter()
            .map(|p| (p.name().to_string(), EloRating::new()))
            .collect::<HashMap<_, _>>(),
    );

    let elo_config = EloConfig::default();

    let w_vs_g_fen = "rnbqkbnr/pppppppp/8/8/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/X W qkqk-- --- 0 1";
    let w_vs_b_fen = "rnbqkbnr/pppppppp/8/8/X/X X/X/X X/X/rnbqkbnr/pppppppp/8/8 W qk--qk --- 0 1";
    let g_vs_b_fen = "X/X/X X/rnbqkbnr/pppppppp/8/8/X X/X/rnbqkbnr/pppppppp/8/8 G --qkqk --- 0 1";

    let game_definitions = vec![
        (w_vs_g_fen, 0, 1), // White vs Green
        (w_vs_b_fen, 0, 2), // White vs Black
        (g_vs_b_fen, 1, 2), // Green vs Black
    ];

    let matchups = (0..players.len()).combinations(2).collect::<Vec<_>>();

    let all_games_to_play: Vec<TwoPlayerGameSetup> = matchups
        .clone()
        .into_iter()
        .flat_map(|matchup| {
            let p1_idx = matchup[0];
            let p2_idx = matchup[1];

            game_definitions
                .iter()
                .flat_map(move |(fen, color1, color2)| {
                    vec![
                        TwoPlayerGameSetup {
                            fen,
                            player1_idx: p1_idx,
                            player2_idx: p2_idx,
                            color1_board_idx: *color1,
                            color2_board_idx: *color2,
                        },
                        TwoPlayerGameSetup {
                            fen,
                            player1_idx: p2_idx,
                            player2_idx: p1_idx,
                            color1_board_idx: *color1,
                            color2_board_idx: *color2,
                        },
                    ]
                })
        })
        .collect();

    let repeated_games: Vec<_> = (0..config.games_per_matchup)
        .flat_map(|_| all_games_to_play.clone())
        .collect();

    // reshuffle to avoid any ordering bias
    let mut repeated_games = repeated_games;
    let mut rng = rng();
    repeated_games.shuffle(&mut rng);

    let total_games = repeated_games.len();

    info!(
        "Total players: {}. Total matchups: {}. Total games: {} ({} per matchup)",
        players.len(),
        matchups.len(),
        total_games,
        config.games_per_matchup
    );

    let games_finished = Mutex::new(0);

    repeated_games.par_iter().for_each(|game_setup| {
        let player1 = &players[game_setup.player1_idx];
        let player2 = &players[game_setup.player2_idx];

        let mut players_map = HashMap::new();
        players_map.insert(game_setup.color1_board_idx, player1);
        players_map.insert(game_setup.color2_board_idx, player2);

        let board = TriHexChess::new_with_fen(game_setup.fen.as_ref(), false).unwrap();
        let outcome = play_game_2_player(&players_map, config, board, oracle.as_deref().unwrap());

        let p1_color = game_setup.color1_board_idx;

        let elo_outcome = match outcome {
            Outcome::WonBy(winner) => {
                if (winner as usize) == p1_color {
                    Outcomes::WIN
                } else {
                    Outcomes::LOSS
                }
            }
            Outcome::PartialDraw(_) => Outcomes::DRAW,
            Outcome::AllDraw => Outcomes::DRAW,
        };

        let mut ratings_guard = ratings.lock().unwrap();
        let p1_name = player1.name();
        let p2_name = player2.name();

        let rating1 = ratings_guard[p1_name];
        let rating2 = ratings_guard[p2_name];

        let (new_rating1, new_rating2) = elo(&rating1, &rating2, &elo_outcome, &elo_config);

        ratings_guard.insert(p1_name.to_string(), new_rating1);
        ratings_guard.insert(p2_name.to_string(), new_rating2);
        drop(ratings_guard);

        let mut games_finished_guard = games_finished.lock().unwrap();
        *games_finished_guard += 1;
        if *games_finished_guard % 10 == 0 || *games_finished_guard == total_games {
            let current_ratings = ratings.lock().unwrap();
            let current_ratings = current_ratings.clone();
            let mut current_results: Vec<_> = current_ratings.iter().collect();
            current_results.sort_by(|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap());

            info!(
                "Progress: {}/{} games completed. Current standings:\n{}",
                *games_finished_guard,
                total_games,
                generate_ascii_arena_table_2p_string(
                    current_results
                        .into_iter()
                        .map(|(n, r)| (n.clone(), r.clone()))
                        .collect()
                )
            );
        }
    });

    let final_ratings_map = ratings.into_inner().unwrap();
    let mut final_results: Vec<_> = final_ratings_map.into_iter().collect();

    final_results.sort_by(|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap());

    if !config.out_file.is_empty() {
        write_2_player_arena_results_to_csv(
            &config.config_unique_name,
            Path::new(&config.out_file),
            &final_results,
        )?;
    } else {
        warn!("No output file specified in config, skipping CSV write.");
    }

    info!("{}", generate_ascii_arena_table_2p_string(final_results));
    info!("Arena finished in {:?}", start_time.elapsed());

    Ok(())
}
