use analysis_util::parquet_reader::{AllSamples, ParquetDataReader};
use chess_visualize::vis_chess_render::{save_to_file, VisArrow};
use chess_visualize::vis_chess_util::get_move_color;
use chess_visualize::DatasetViewerConfig;
use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
};
use game_tri_chess::chess_game::TriHexChess;
use maz_core::mapping::chess_wrapper_mapper::{auto_detect_chess_mapper, ChessWrapperMapper};
use maz_core::mapping::{Board, InputMapper, PolicyMapper, ReverseInputMapper};
use maz_core::values_const::ValuesPov;
use ndarray::Array2;
use rayon::prelude::*;
use std::io::{stdout, Write};
use std::sync::Mutex;

fn run_simulation_visualization(
    samples: &AllSamples,
    sim_id: &str,
    mapper: ChessWrapperMapper,
) -> Result<(), Box<dyn std::error::Error>> {
    let top_n_moves = 5;
    let input_shape = mapper.input_board_shape();

    // Get all samples for the selected sim_id
    let game_samples = samples.get_all_samples_for_sim_id(sim_id);
    if game_samples.is_empty() {
        println!("No samples found for sim_id: {}", sim_id);
        return Ok(());
    }

    // Clear all files in the output directory
    let dir = "pdfs/chess_vis";
    std::fs::create_dir_all(dir)?;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() || path.file_name().unwrap().to_str().unwrap().starts_with('.') {
            continue;
        }
        if path.is_file() {
            if let Some(ext) = path.extension() {
                // Clear both to be safe
                if ext == "pdf" || ext == "svg" {
                    std::fs::remove_file(path)?;
                }
            }
        }
    }

    println!(
        "Processing sim_id: {}. Found {} moves.",
        sim_id,
        game_samples.len()
    );
    println!("Generating SVGs in {} using 4 threads...", dir);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;

    let mut to_print = Mutex::new(Vec::new());

    pool.install(|| {
        game_samples.par_iter().for_each(|sample| {
            let input_data = match Array2::from_shape_vec(
                (input_shape[0], input_shape[1]),
                sample.inner.encoded_board.clone(),
            ) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!(
                        "Error creating ndarray for move {}: {}. Skipping.",
                        sample.inner.current_move_count, e
                    );
                    return;
                }
            };

            let mut board =
                mapper.decode_input(&input_data.view(), &vec![sample.inner.player_index as f32]);
            let mut moves = <TriHexChess as Board>::MoveStore::default();
            board.fill_move_store(&mut moves);

            let mut to_print_unlocked = to_print.lock().unwrap();

            to_print_unlocked.push((
                sample.inner.current_move_count,
                format!(
                    "  Generating SVG for Move: {}/{}. {}. \n Q: {:?}. Z: {:?}\n",
                    sample.inner.current_move_count,
                    sample.moves_left + sample.inner.current_move_count,
                    board.fancy_debug(),
                    sample.inner.q_values,
                    sample.z_values
                ),
            ));

            drop(to_print_unlocked);

            // Get top N moves from MCTS policy
            let mut mv_with_probs: Vec<(<TriHexChess as Board>::Move, f32)> = moves
                .iter()
                .map(|mv| {
                    let policy_index = mapper.move_to_index(board.player_current(), *mv);
                    let prob = sample.inner.mcts_policy[policy_index];
                    (*mv, prob)
                })
                .collect();

            mv_with_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            mv_with_probs.truncate(top_n_moves);

            let arrows = mv_with_probs
                .iter()
                .map(|(mv, _prob)| {
                    let mv_color = get_move_color(*mv, &board);

                    VisArrow {
                        from_q: mv.from.to_qrs_global().q,
                        from_r: mv.from.to_qrs_global().r,
                        from_i: mv.from,
                        to_i: mv.to,
                        to_q: mv.to.to_qrs_global().q,
                        to_r: mv.to.to_qrs_global().r,
                        opacity: 1.0,
                        color: mv_color,
                        text: Some(format!("{:.0}", _prob * 100.0)),
                    }
                })
                .collect::<Vec<_>>();

            save_to_file(
                &board,
                arrows,
                vec![],
                Some(
                    ValuesPov::<3>::from_slice(
                        &sample.z_values.as_slice(),
                        0.0,
                        sample.inner.player_index,
                    )
                    .abs(),
                ),
                dir,
                format!(
                    "board_{}_move_{}.svg",
                    sample.sim_id, sample.inner.current_move_count
                )
                .as_str(),
            );
        });
    });

    let mut to_print = to_print.lock().unwrap();
    to_print.sort_by_key(|(move_count, _)| *move_count);
    for (_move_count, message) in to_print.iter() {
        println!("{}", message);
    }

    println!("\nDone! SVGs generated. Press any key to return to the menu.");

    loop {
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(_) = event::read()? {
                break;
            }
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(default_value = "analysis/chess-visualize/dataset-vis-config.yaml")]
    pub(crate) config_path: String,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config = DatasetViewerConfig::load(&cli.config_path)?;
    let file_path = config.parquet_path;

    let samples = ParquetDataReader::read_file(std::path::Path::new(file_path.as_str()))?;
    let all_sim_ids = samples.get_unique_sim_ids();

    // get the first available sample
    let sample = samples.get_all_samples_for_sim_id(&all_sim_ids[0]);
    let picked_sample = sample.get(0).ok_or("No samples found in the dataset")?;
    let input = &picked_sample.inner.encoded_board;

    let wrapper_mapper = auto_detect_chess_mapper(input.as_ref());

    println!(
        "Data loaded. Found {} unique simulations. Detected input mapper: {}.",
        all_sim_ids.len(),
        wrapper_mapper.get_name()
    );

    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, cursor::Hide)?;

    let mut selected_index: usize = 0;

    loop {
        execute!(stdout, cursor::MoveTo(0, 0), Clear(ClearType::All))?;
        writeln!(
            stdout,
            "Select a Simulation ID (Use UP/DOWN arrows, ENTER to select, Q to quit)\r\n"
        )?;

        // Display a scrollable window of IDs to avoid filling the screen
        let window_size = 10;
        let start = selected_index.saturating_sub(window_size / 2);
        let end = (start + window_size).min(all_sim_ids.len());
        let start = end.saturating_sub(window_size).max(0);

        if start > 0 {
            writeln!(stdout, "  ...\r")?;
        }

        fn get_sim_id_formatted(idx: usize, samples: &AllSamples) -> String {
            let sim_id = &samples.get_unique_sim_ids()[idx];
            let all_samples = samples.get_all_samples_for_sim_id(sim_id);
            let count = all_samples.len();
            let outcome = ValuesPov::<3>::from_slice(
                all_samples.last().unwrap().z_values.as_slice(),
                0.0,
                all_samples.last().unwrap().inner.player_index,
            );
            format!("{} ({} samples). O: {:?}", sim_id, count, outcome)
        }

        for i in start..end {
            if i == selected_index {
                writeln!(stdout, "> {}\r", get_sim_id_formatted(i, &samples))?;
            } else {
                writeln!(stdout, "  {}\r", get_sim_id_formatted(i, &samples))?;
            }
        }
        if end < all_sim_ids.len() {
            writeln!(stdout, "  ...\r")?;
        }
        stdout.flush()?;

        if event::poll(std::time::Duration::from_millis(500))? {
            if let Event::Key(key_event) = event::read()? {
                // We only care about key presses, not releases
                if key_event.kind == KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Up => selected_index = selected_index.saturating_sub(1),
                        KeyCode::Down => {
                            if selected_index < all_sim_ids.len() - 1 {
                                selected_index += 1;
                            }
                        }
                        KeyCode::Enter => {
                            execute!(
                                stdout,
                                Clear(ClearType::Purge),
                                cursor::MoveTo(0, 0),
                                cursor::Show
                            )?;
                            disable_raw_mode()?;

                            if let Err(e) = run_simulation_visualization(
                                &samples,
                                &all_sim_ids[selected_index],
                                wrapper_mapper.clone(),
                            ) {
                                eprintln!("\nAn error occurred: {}", e);
                                println!("Press Enter to continue...");
                                std::io::stdin().read_line(&mut String::new())?;
                            }

                            // Re-enter TUI mode for the menu
                            enable_raw_mode()?;
                            execute!(stdout, cursor::Hide)?;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    execute!(stdout, cursor::Show)?;
    disable_raw_mode()?;
    Ok(())
}
