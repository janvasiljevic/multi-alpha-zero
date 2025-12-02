// src/arena_csv_writer.rs

use csv::Writer;
use serde::Serialize;
use skillratings::weng_lin::WengLinRating;
use std::path::Path;
use log::warn;
use skillratings::elo::EloRating;
use tracing::info;

#[derive(Serialize)]
struct ArenaResultRow<'a> {
    config_name: &'a str,
    rank: u32,
    model_name: &'a str,
    rating: f64,
    uncertainty: f64,
    conservative_rating: f64,
}

pub fn write_arena_results_to_csv(
    config_name: &str,
    output_path: &Path,
    sorted_results: &[(String, WengLinRating)],
) -> Result<(), Box<dyn std::error::Error>> {
    
    if output_path.exists() {
        std::fs::remove_file(output_path)?;
        warn!("Output file {} already exists, overwriting.", output_path.display());
    }

    info!("Writing arena results to {}", output_path.display());

    let mut writer = Writer::from_path(output_path)?;

    for (i, (name, rating)) in sorted_results.iter().enumerate() {
        let rank = (i + 1) as u32;
        let conservative_rating = rating.rating - 3.0 * rating.uncertainty;

        let row = ArenaResultRow {
            config_name,
            rank,
            model_name: name,
            rating: rating.rating,
            uncertainty: rating.uncertainty,
            conservative_rating,
        };

        writer.serialize(row)?;
    }

    writer.flush()?;

    info!("Successfully wrote results to {}", output_path.display());
    Ok(())
}


pub fn write_2_player_arena_results_to_csv(
    config_name: &str,
    output_path: &Path,
    sorted_results: &[(String, EloRating)],
) -> Result<(), Box<dyn std::error::Error>> {

    if output_path.exists() {
        std::fs::remove_file(output_path)?;
        warn!("Output file {} already exists, overwriting.", output_path.display());
    }

    info!("Writing 2-Player arena results to {}", output_path.display());

    let mut writer = Writer::from_path(output_path)?;

    for (i, (name, rating)) in sorted_results.iter().enumerate() {
        let rank = (i + 1) as u32;

        let row = ArenaResultRow {
            config_name,
            rank,
            model_name: name,
            rating: rating.rating,
            uncertainty: 0.0, // Elo does not have uncertainty
            conservative_rating: rating.rating, // Elo does not have conservative rating
        };

        writer.serialize(row)?;
    }

    writer.flush()?;

    info!("Successfully wrote 2-Player results to {}", output_path.display());
    Ok(())
}