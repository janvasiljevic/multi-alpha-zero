use crate::arena_config::{ArenaConfig, PlayerConfig};
use crate::arena_csv_writer::write_arena_results_to_csv;
use crate::arena_util::generate_ascii_arena_table;
use crate::util::{chess_mapper_from_network_shapes, outcome_to_ranks};
use anyhow::Result;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::PseudoLegalMove;
use itertools::Itertools;
use maz_core::mapping::chess_wrapper_mapper::ChessWrapperMapper;
use maz_core::mapping::{Board, InputMapper, Oracle, Outcome, PolicyMapper};
use maz_trainer::search_settings::SearchSettings;
use maz_trainer::self_play_game::ProduceOutput;
use maz_trainer::steppable_mcts::{ConsumeValues, SteppableMCTS};
use maz_util::network::{alpha_zero_shapes, auto_non_cpu_model, batcher_from, prediction};
use oracle_tri_chess::oracle::TriHexEndgameOracle;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{SeedableRng, rng};
use rayon::prelude::*;
use skillratings::MultiTeamOutcome;
use skillratings::weng_lin::{WengLinConfig, WengLinRating, weng_lin_multi_team};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// A trait representing a player in the chess arena.
pub trait ChessArenaPlayer {
    /// Returns the name of the player.
    fn name(&self) -> &str;
    /// Determines the best move for the given board state using the provided configuration.
    fn get_best_move(&self, board: &TriHexChess, config: &ArenaConfig) -> PseudoLegalMove;
}

/// An AlphaZero-style player that uses an ONNX model and MCTS.
pub struct AlphaZeroPlayer {
    pub name: String,
    pub model: Mutex<ort::session::Session>,
    pub mapper: ChessWrapperMapper,
}

/// An enum to dispatch to different concrete player implementations.
pub enum ChessInnerPlayer {
    AlphaZero(AlphaZeroPlayer),
}

impl ChessArenaPlayer for ChessInnerPlayer {
    fn name(&self) -> &str {
        match self {
            ChessInnerPlayer::AlphaZero(p) => p.name(),
        }
    }

    fn get_best_move(&self, board: &TriHexChess, config: &ArenaConfig) -> PseudoLegalMove {
        match self {
            ChessInnerPlayer::AlphaZero(p) => p.get_best_move(board, config),
        }
    }
}

impl ChessArenaPlayer for AlphaZeroPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_best_move(&self, board: &TriHexChess, config: &ArenaConfig) -> PseudoLegalMove {
        let settings = SearchSettings::default();
        let mut rng = StdRng::from_os_rng();

        let mut steppable_mcts = SteppableMCTS::<_, 3>::new_with_capacity(
            board,
            &self.mapper,
            config.mcts_rollouts_per_move as u64,
            true,
        );

        let model_guard = self.model.lock().unwrap();
        let mut batcher = batcher_from(&*model_guard, &self.mapper);
        drop(model_guard);

        let mut requests: Vec<ProduceOutput<TriHexChess>> =
            Vec::with_capacity(batcher.get_batch_size());

        let mut i = 0;
        let virtual_size_batch = 4;

        loop {
            if let Some(request) = steppable_mcts.step(
                &mut rng,
                settings.search_fpu_root,
                settings.search_fpu_child,
                settings.search_virtual_loss_weight,
                settings.weights,
                &self.mapper,
                settings.contempt,
            ) {
                self.mapper
                    .encode_input(&mut batcher.get_mut_item(requests.len()), &request.board);
                requests.push(request);
            }

            i += 1;

            if requests.len() >= batcher.get_batch_size() || i >= virtual_size_batch {
                let mut model_guard = self.model.lock().unwrap();
                let (policy_data, value_data) = prediction(&mut *model_guard, &batcher);

                for (i, request) in requests.iter().enumerate() {
                    let policy_net_f16 = &policy_data
                        [i * self.mapper.policy_len()..(i + 1) * self.mapper.policy_len()];
                    let values_net = &value_data[i * 3..(i + 1) * 3];

                    steppable_mcts
                        .consume(
                            &mut rng,
                            &settings,
                            request.node_id,
                            ConsumeValues::ConsumeWithOptionalCallback {
                                policy_net_f16,
                                values_net,
                                callback: None,
                            },
                            false,
                        )
                        .expect("Impossible to error out here, we aren't taking in Cache results.");
                }
                requests.clear();
                i = 0;

                if steppable_mcts.is_enough_rollouts() {
                    break;
                }
            }
        }

        let best_child_id = steppable_mcts
            .tree
            .best_child(0)
            .expect("MCTS root should have at least one child after rollouts");

        steppable_mcts.tree[best_child_id]
            .last_move
            .expect("Best child node must have a move associated with it")
    }
}

/// Plays a single game of 3-player chess between the given players.
fn play_game(
    players: [&ChessInnerPlayer; 3],
    config: &ArenaConfig,
    mut board: TriHexChess,
    oracle: &TriHexEndgameOracle,
) -> Outcome {
    let mut move_store = <TriHexChess as Board>::MoveStore::default();

    for _ in 0..750 {
        if board.is_terminal() {
            break;
        }

        if board.can_game_end_early() {
            return board
                .outcome(true)
                .expect("Forced early end must have an outcome");
        }

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

        let current_player_idx: usize = board.player_current().into();
        let current_player = players[current_player_idx];
        let best_move = current_player.get_best_move(&board, config);
        board.fill_move_store(&mut move_store);
        board.play_move_mut_with_store(&best_move, &mut move_store, None);
    }

    if !board.is_terminal() {
        info!(
            "Game somehow reached move limit, without being ended early - forcing an end. Pretty print: \n{}",
            board.fancy_debug()
        );
        return board
            .outcome(true)
            .expect("Forced end must have an outcome");
    }

    board
        .outcome(false)
        .expect("Terminal board must have an outcome")
}

/// Main function to run a chess arena tournament based on a given configuration.
pub fn run_arena_chess(
    config: &ArenaConfig,
    maybe_device_id: Option<usize>,
    _sessions_per_gpu: usize, // Not used in this implementation yet, but kept for API consistency
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Starting Chess Arena for config: '{}'",
        config.config_unique_name
    );

    let oracle = match TriHexEndgameOracle::new(
        "./game/oracle-tri-chess/tablebases/kqk_tablebase.bin",
        "./game/oracle-tri-chess/tablebases/krk_tablebase.bin",
    ) {
        Ok(oracle) => Some(Arc::new(oracle)),
        Err(e) => {
            // This is a fatal error if we expect an oracle for chess.
            panic!("Failed to initialize TriHexEndgameOracle: {:?}", e);
        }
    };

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
                        let sizes = alpha_zero_shapes(&model)?;

                        Ok(ChessInnerPlayer::AlphaZero(AlphaZeroPlayer {
                            name: name.to_string(),
                            model: Mutex::new(model),
                            mapper: chess_mapper_from_network_shapes(sizes)?,
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

    if players.len() < 3 {
        return Err(format!(
            "Chess arena requires at least 3 players. Found {}.",
            players.len()
        )
        .into());
    }

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

    let weng_lin_config = WengLinConfig::new();
    let matchups = (0..players.len()).combinations(3).collect::<Vec<_>>();

    let mut all_games_to_play: Vec<Vec<usize>> = matchups
        .into_iter()
        .flat_map(|matchup| {
            let permutations = matchup.into_iter().permutations(3);
            permutations.flat_map(move |p| std::iter::repeat(p).take(config.games_per_matchup))
        })
        .collect();

    let mut rng = rng();
    all_games_to_play.shuffle(&mut rng);

    let total_games = all_games_to_play.len();
    info!("Total games to be played for this config: {}", total_games);

    let games_finished = Mutex::new(0);

    all_games_to_play.par_iter().for_each(|seating| {
        let seated_players: [&ChessInnerPlayer; 3] = [
            &players[seating[0]],
            &players[seating[1]],
            &players[seating[2]],
        ];

        let board = TriHexChess::default_with_grace_period();
        let outcome = play_game(seated_players, config, board, oracle.as_deref().unwrap());
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

        let team1_ratings_arr = [p_ratings[0]];
        let team2_ratings_arr = [p_ratings[1]];
        let team3_ratings_arr = [p_ratings[2]];

        let rating_groups = vec![
            (
                &team1_ratings_arr[..],
                MultiTeamOutcome::new(ranks[0] as usize),
            ),
            (
                &team2_ratings_arr[..],
                MultiTeamOutcome::new(ranks[1] as usize),
            ),
            (
                &team3_ratings_arr[..],
                MultiTeamOutcome::new(ranks[2] as usize),
            ),
        ];

        let new_teams_ratings = weng_lin_multi_team(&rating_groups, &weng_lin_config);

        ratings_guard.insert(p_names[0].to_string(), new_teams_ratings[0][0]);
        ratings_guard.insert(p_names[1].to_string(), new_teams_ratings[1][0]);
        ratings_guard.insert(p_names[2].to_string(), new_teams_ratings[2][0]);

        let mut games_finished_guard = games_finished.lock().unwrap();
        *games_finished_guard += 1;
        info!(
            "Progress: {}/{} games completed.",
            *games_finished_guard, total_games
        );

        if *games_finished_guard % 10 == 0 {
            info!(
                "Intermediate ratings after {} games:",
                *games_finished_guard
            );
            let intermediate_ratings: Vec<_> = ratings_guard.iter().collect();
            let mut sorted_intermediate: Vec<_> = intermediate_ratings.into_iter().collect();
            sorted_intermediate.sort_by(|a, b| {
                let rating_a = a.1.rating - 3.0 * a.1.uncertainty;
                let rating_b = b.1.rating - 3.0 * b.1.uncertainty;
                rating_b.partial_cmp(&rating_a).unwrap()
            });
            let str_builder = sorted_intermediate
                .iter()
                .map(|(name, rating)| {
                    format!(
                        "{}: {:.2} (σ={:.2})",
                        name, rating.rating, rating.uncertainty
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");

            info!("Current Ratings: {}", str_builder);
        }

        drop(games_finished_guard);
    });

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
