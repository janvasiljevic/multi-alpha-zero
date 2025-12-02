use maz_util::math::safe_div;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{info, Instrument};
use maz_config::config::CycleCollectionSettings;

/// Estimates the remaining time in seconds for the cycle to complete,
fn estimate_remaining_time(
    settings: &CycleCollectionSettings,
    samples: usize,
    games: usize,
    elapsed_seconds: f64,
    samples_per_sec: f64,
    games_per_sec: f64,
) -> f64 {
    let mut potential_end_times: Vec<f64> = Vec::new();

    // Time to meet all required thresholds
    let mut threshold_times: Vec<f64> = Vec::new();

    let avoid_zero = |per_sec: f64, current: u64, min: u64| -> f64 {
        if per_sec > 1e-6 {
            let to_go = min.saturating_sub(current);
            to_go as f64 / per_sec
        } else if current < min {
            f64::INFINITY // If we can't make progress, it will take "forever"
        } else {
            0.0 // If we already met the target, no time needed
        }
    };

    if let Some(min_samples) = settings.min_samples_per_cycle {
        threshold_times.push(avoid_zero(samples_per_sec, samples as u64, min_samples));
    }

    if let Some(min_games) = settings.min_games_per_cycle {
        threshold_times.push(avoid_zero(games_per_sec, games as u64, min_games));
    }

    // If there are any thresholds, the time to meet them is the MAX of the individual times.
    if !threshold_times.is_empty() {
        let time_to_meet_all_thresholds = threshold_times
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        potential_end_times.push(time_to_meet_all_thresholds);
    }

    // Time until time limit is reached
    if let Some(max_time) = settings.max_time_per_cycle_s {
        let time_left_for_timeout = (max_time as f64) - elapsed_seconds;
        if time_left_for_timeout > 0.0 {
            potential_end_times.push(time_left_for_timeout);
        }
    }

    // The actual end time will be the MINIMUM of the potential end times
    potential_end_times
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0) // If no conditions are set, remaining time is 0.
}

pub fn create_reporter(
    samples_clone: Arc<AtomicUsize>,
    total_games_clone: Arc<AtomicUsize>,
    cycle_settings: CycleCollectionSettings,
    stop_signal_clone: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let reporting_interval = Duration::from_secs(2);
        let mut interval = tokio::time::interval(reporting_interval);
        let start_time = Instant::now();

        let mut last_reported_samples = 0;
        let mut last_reported_games = 0;

        let conditions_counter = cycle_settings.min_games_per_cycle.map_or(0, |_| 1)
            + cycle_settings.min_samples_per_cycle.map_or(0, |_| 1);

        let mut conditions_descriptions: Vec<String> = Vec::with_capacity(conditions_counter);

        loop {
            if stop_signal_clone.load(Ordering::Relaxed) {
                break;
            }

            interval.tick().await;

            let current_samples = samples_clone.load(Ordering::Relaxed);
            let current_games = total_games_clone.load(Ordering::Relaxed);
            let elapsed_seconds = start_time.elapsed().as_secs();

            if let Some(max_time) = cycle_settings.max_time_per_cycle_s {
                if elapsed_seconds >= max_time && !stop_signal_clone.swap(true, Ordering::Relaxed) {
                    info!(
                        "Self-play cycle complete. Reason: time limit ({}s >= {}s). Triggering shutdown.",
                        elapsed_seconds, max_time
                    );
                    stop_signal_clone.store(true, Ordering::Relaxed);

                    break;
                }
            }

            if conditions_counter > 0 {
                let mut conditions_met = 0;
                conditions_descriptions.clear();

                if let Some(min_games) = cycle_settings.min_games_per_cycle {
                    if current_games >= min_games as usize {
                        conditions_met += 1;
                        conditions_descriptions.push(format!(
                            "min_games_per_cycle met: {current_games} >= {min_games}"
                        ));
                    }
                }

                if let Some(min_samples) = cycle_settings.min_samples_per_cycle {
                    if (current_samples as u64) >= min_samples {
                        conditions_met += 1;
                        conditions_descriptions.push(format!(
                            "min_samples_per_cycle met: {current_samples} >= {min_samples}"
                        ));
                    }
                }

                if conditions_met >= conditions_counter {
                    info!(
                        "Threshold ({}/{}) conditions met. Conditions: {})",
                        conditions_met,
                        conditions_counter,
                        conditions_descriptions.join(", ")
                    );

                    stop_signal_clone.swap(true, Ordering::Relaxed);
                }
            }

            let current_count = samples_clone.load(Ordering::Relaxed);
            let current_games = total_games_clone.load(Ordering::Relaxed);

            if current_count == last_reported_samples {
                info!(
                    "No new samples collected, Still at {} samples and {} games.",
                    current_count, current_games
                );

                continue;
            }

            let elapsed_seconds = start_time.elapsed().as_secs_f64();

            let samples_increase = current_count - last_reported_samples;
            let games_increase = current_games - last_reported_games;

            let samples_per_sec = safe_div(current_count as f64, elapsed_seconds);
            let games_per_sec = safe_div(current_games as f64, elapsed_seconds);

            let display_with_percent = |value: usize, min: Option<u64>| {
                if let Some(min) = min {
                    format!(
                        "{}/{} ({:.1}%)",
                        value,
                        min,
                        (value as f64 / min as f64) * 100.0
                    )
                } else {
                    value.to_string()
                }
            };

            let estimate_time = estimate_remaining_time(
                &cycle_settings,
                current_count,
                current_games,
                elapsed_seconds,
                samples_per_sec,
                games_per_sec,
            );

            info!(
                "[+{} samp./ +{} G.] Collected {} samples and {} games in {}. Speed of {:.2} samp./s and {:.2} G/s. Estimated time left: {:.2}s",
                samples_increase,
                games_increase,
                display_with_percent(current_count, cycle_settings.min_samples_per_cycle),
                display_with_percent(current_games, cycle_settings.min_games_per_cycle),
                cycle_settings.max_time_per_cycle_s.map_or(
                    format!("{elapsed_seconds:.0}s"),
                    |t| format!(
                        "{:.0}/{:.0}s ({:.1}%)",
                        elapsed_seconds as u64,
                        t,
                        (elapsed_seconds / t as f64) * 100.0
                    )
                ),
                samples_per_sec,
                games_per_sec,
                estimate_time
            );

            last_reported_samples = current_count;
            last_reported_games = current_games;
        }
    }.instrument(tracing::info_span!("Progress")))
}
