use clap::Parser;

/// A self-play data generator for training neural networks for board games.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the configuration file.
    #[arg(default_value = "config.yaml")]
    pub(crate) config_path: String,
}
