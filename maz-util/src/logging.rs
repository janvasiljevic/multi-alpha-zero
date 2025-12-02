use atty::Stream;
use chrono::Timelike;
use once_cell::sync::Lazy;
use std::env;
use std::time::Instant;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

static START_TIME: Lazy<Instant> = Lazy::new(Instant::now);

struct WallClockAndUptime;

impl FormatTime for WallClockAndUptime {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        // Wall clock time
        let now = chrono::Local::now();
        let time_str = format!("{:02}:{:02}:{:02}", now.hour(), now.minute(), now.second());

        // Uptime
        let elapsed = START_TIME.elapsed();
        let secs = elapsed.as_secs();
        let millis = elapsed.subsec_millis();

        write!(w, "{} +{}.{}s", time_str, secs, format!("{:03}", millis))
    }
}

pub fn setup_logging(disable_ort: bool) {
    let ansi = match env::var("RUST_LOG_STYLE").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        _ => atty::is(Stream::Stdout), // Fallback
    };

    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"))
        .add_directive(if disable_ort {
            "ort=off".parse().unwrap()
        } else {
            "ort=warn".parse().unwrap()
        });

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_timer(WallClockAndUptime)
        .with_ansi(ansi)
        .with_target(false)
        .with_level(true);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}
