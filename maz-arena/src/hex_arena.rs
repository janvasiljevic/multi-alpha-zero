use crate::arena_config::{ArenaConfig, PlayerConfig};
use crate::arena_csv_writer::write_arena_results_to_csv;
use crate::arena_util::generate_ascii_arena_table;
use crate::hex_player::{
    AlphaZeroPlayer, HexArenaPlayer, HexInnerPlayer, MctsPlayer, RandomPlayer,
};
use crate::util::outcome_to_ranks;
use game_hex::game_hex::HexGame;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use maz_core::mapping::hex_absolute_mapper::HexAbsoluteMapper;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_core::mapping::hex_wrapper_mapper::HexWrapperMapper;
use maz_core::mapping::{Board, Outcome};
use maz_util::network::{alpha_zero_shapes, auto_non_cpu_model};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use skillratings::weng_lin::{weng_lin_multi_team, WengLinConfig, WengLinRating};
use skillratings::MultiTeamOutcome;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tracing::{info, warn};

fn play_game(players: [&HexInnerPlayer; 3], config: &ArenaConfig, og_board: &HexGame) -> Outcome {
    let mut board = og_board.clone().new();
    let mut move_store = <HexGame as Board>::MoveStore::default();

    loop {
        if board.is_terminal() {
            break;
        }

        let current_player_idx: usize = board.player_current().into();
        let current_player = players[current_player_idx];
        let best_move = current_player.get_best_move(&board, config);

        board.fill_move_store(&mut move_store);
        board.play_move_mut_with_store(&best_move, &mut move_store, None);
    }
    board
        .outcome(false)
        .expect("Terminal board must have an outcome")
}

pub fn run_arena_hex(
    config: &ArenaConfig,
    maybe_device_id: Option<usize>,
    sessions_per_gpu: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Hex Arena");

    let start_of_load = Instant::now();
    let mut number_of_fields: Option<usize> = None;
    let mut maybe_game: Option<HexGame> = None;

    let players: Vec<HexInnerPlayer> = config
        .players
        .iter()
        .map(|player_config| -> Result<HexInnerPlayer, Box<dyn std::error::Error>> {
            match player_config {
                PlayerConfig::AlphaZeroPlayer { onnx_path: path } => {
                    let name = path.split('/').last().expect("Model path must have a name");
                    let model = auto_non_cpu_model(path, maybe_device_id)?;
                    let sizes = alpha_zero_shapes(&model)?;

                    match number_of_fields {
                        Some(current_number_of_fields) => {
                            if current_number_of_fields != sizes.input_shape[1] {
                                return Err(format!("All models must operate on same board size. Model '{}' has input shape {:?}, expected number of fields: {}", name, sizes.input_shape, current_number_of_fields).into());
                            }
                        }
                        None => {
                            number_of_fields = Some(sizes.input_shape[1]);
                            match number_of_fields.unwrap() {
                                61 => maybe_game = Some(HexGame::new(5).unwrap()),
                                37 => maybe_game = Some(HexGame::new(4).unwrap()),
                                19 => maybe_game = Some(HexGame::new(3).unwrap()),
                                7 => maybe_game = Some(HexGame::new(2).unwrap()),
                                _ => return Err(format!(
                                    "Unsupported board size with {} fields.",
                                    number_of_fields.unwrap()
                                ).into()),


                            }
                        }
                    }

                    let game = maybe_game.as_ref().unwrap();
                    let is_absolute = sizes.input_shape[2] == 7;

                    Ok(HexInnerPlayer::MctsModel(AlphaZeroPlayer {
                        name: name.to_string(),
                        model: Mutex::new(model),
                        mapper: if is_absolute {
                            HexWrapperMapper::Absolute(HexAbsoluteMapper::new(game))
                        } else {
                            HexWrapperMapper::Canonical(HexCanonicalMapper::new(game))
                        },
                    }))
                }
                PlayerConfig::Random { unique_name: name } => Ok(HexInnerPlayer::Random(RandomPlayer::new(name.clone()))),
                PlayerConfig::Mcts { unique_name: name, num_of_playouts: num_of_simulations } => Ok(HexInnerPlayer::Mcts(MctsPlayer::new(name.clone(), *num_of_simulations))),
            }
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    if maybe_game.is_none() {
        return Err("No valid models loaded to determine game size.")?;
    }

    let game = maybe_game.expect("Game should have been determined");

    info!(
        "All players initialized successfully in {:?}.",
        start_of_load.elapsed()
    );

    let ratings = Mutex::new(
        players
            .iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    WengLinRating {
                        rating: config.initial_elo,
                        uncertainty: config.get_elo_uncertainty(),
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    );

    let num_of_board_players = game.player_num();
    let weng_lin_config = WengLinConfig::new();

    let matchups = (0..players.len())
        .combinations(num_of_board_players)
        .collect::<Vec<_>>();

    let all_games_to_play: Vec<Vec<usize>> = matchups
        .into_iter()
        .flat_map(|matchup| {
            let permutations = matchup.into_iter().permutations(num_of_board_players);
            permutations.flat_map(move |p| std::iter::repeat(p).take(config.games_per_matchup))
        })
        .collect();

    let total_games = all_games_to_play.len();
    info!("Total games to be played: {}", total_games);

    let progress_bar = ProgressBar::new(total_games as u64).with_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    let pool = ThreadPoolBuilder::new()
        .num_threads(sessions_per_gpu)
        .build()?;

    pool.install(|| {
        all_games_to_play.par_iter().for_each(|seating| {
            // Seating now works with the generic `Player` enum
            let seated_players: [&HexInnerPlayer; 3] = [
                &players[seating[0]],
                &players[seating[1]],
                &players[seating[2]],
            ];

            let outcome = play_game(seated_players, config, &game);
            let ranks = outcome_to_ranks(outcome);

            let mut ratings_guard = ratings.lock().unwrap();

            let p_names = [
                seated_players[0].name(),
                seated_players[1].name(),
                seated_players[2].name(),
            ];
            let p_ratings = [
                ratings_guard[p_names[0]],
                ratings_guard[p_names[1]],
                ratings_guard[p_names[2]],
            ];

            let binding_0 = [p_ratings[0]];
            let binding_1 = [p_ratings[1]];
            let binding_2 = [p_ratings[2]];

            let rating_groups = vec![
                (&binding_0[..], MultiTeamOutcome::new(ranks[0] as usize)),
                (&binding_1[..], MultiTeamOutcome::new(ranks[1] as usize)),
                (&binding_2[..], MultiTeamOutcome::new(ranks[2] as usize)),
            ];

            let new_teams_ratings = weng_lin_multi_team(&rating_groups, &weng_lin_config);

            ratings_guard.insert(p_names[0].to_string(), new_teams_ratings[0][0]);
            ratings_guard.insert(p_names[1].to_string(), new_teams_ratings[1][0]);
            ratings_guard.insert(p_names[2].to_string(), new_teams_ratings[2][0]);

            progress_bar.inc(1);
        });
    });
    progress_bar.finish_with_message("All games completed!");

    let final_ratings_map = ratings.into_inner().unwrap();
    let mut final_results: Vec<_> = final_ratings_map.into_iter().collect();

    final_results.sort_by(|a, b| {
        let rating_a = a.1.rating - 3.0 * a.1.uncertainty;
        let rating_b = b.1.rating - 3.0 * b.1.uncertainty;
        rating_b.partial_cmp(&rating_a).unwrap()
    });

    if !config.out_file.is_empty() {
        write_arena_results_to_csv(
            &config.config_unique_name,
            Path::new(&config.out_file),
            &final_results,
        )?;
    } else {
        warn!("No output file specified in config, skipping CSV write.");
    }

    generate_ascii_arena_table(final_results);

    Ok(())
}
