use analysis_util::{VisArrow, convert_svg_to_pdf};
use chess_visualize::vis_chess_render::render_board;
use chess_visualize::vis_chess_util::get_move_color;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use maz_core::mapping::Board;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct OpeningRecord {
    opening_sequence: String,
    times_played: usize,
}

fn read_openings_from_csv(file_path: &str) -> Result<Vec<OpeningRecord>, Box<dyn Error>> {
    let file = File::open(file_path)?;

    let mut rdr = csv::Reader::from_reader(file);

    let records = rdr
        .deserialize()
        .collect::<Result<Vec<OpeningRecord>, csv::Error>>()?;

    Ok(records)
}

fn main() {
    let file_path = "./analysis/chess-visualize/opening_new_6.csv";

    println!("Reading openings from: {}", file_path);

    match read_openings_from_csv(&file_path) {
        Ok(openings) => {
            for (i, record) in openings.iter().take(100).enumerate() {
                let mut board = TriHexChess::default();
                let moves = record
                    .opening_sequence
                    .split_whitespace()
                    .collect::<Vec<&str>>();

                let mut move_store = ChessMoveStore::default();

                board.fill_move_store(&mut move_store);

                // find the move in the move store

                let mut vis_arrows = vec![];

                for (i, mv) in moves.iter().enumerate() {
                    for chess_move in move_store.into_iter() {
                        if chess_move.notation_hyphen() == *mv {
                            let mv_color = get_move_color(chess_move, &board);

                            board.play_move_mut_with_store(&chess_move, &mut move_store, None);

                            vis_arrows.push(VisArrow {
                                from_q: chess_move.from.to_qrs_global().q,
                                from_r: chess_move.from.to_qrs_global().r,
                                from_i: chess_move.from,
                                to_i: chess_move.to,
                                to_q: chess_move.to.to_qrs_global().q,
                                to_r: chess_move.to.to_qrs_global().r,
                                opacity: 1.0,
                                color: mv_color,
                                text: Some(format!("{}", i + 1)),
                            });

                            break;
                        }
                    }
                }

                let document = render_board(&board, &vis_arrows, None, None, None, false);

                let svg_filename = format!(
                    "pdfs/chess/openings/opening_{}_{}.svg",
                    i + 1,
                    record.times_played
                );
                let pdf_filename = format!(
                    "pdfs/chess/openings/opening_{}_{}.pdf",
                    i + 1,
                    record.times_played
                );

                svg::save(svg_filename.as_str(), &document).unwrap();

                convert_svg_to_pdf(svg_filename.as_str(), pdf_filename.as_str()).unwrap();

                std::fs::remove_file(svg_filename.as_str()).unwrap();
            }
        }
        Err(e) => {
            eprintln!("Error reading the CSV file: {}", e);
        }
    }
}
