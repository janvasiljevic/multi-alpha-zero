use anyhow::Context;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Clone, Default)]
pub struct CollectorStats {
    pub total_games_played: u64,
    pub total_samples_generated: u64,
}

impl std::ops::AddAssign for CollectorStats {
    fn add_assign(&mut self, other: Self) {
        self.total_games_played += other.total_games_played;
        self.total_samples_generated += other.total_samples_generated;
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Record {
    pub cycle_number: u64,
    pub total_games_played: u64,
    pub total_samples_generated: u64,

    pub self_play_time_seconds: f64,
    pub model_training_time_seconds: f64,
    pub total_time_seconds: f64,

    pub nodes_backed_up: u64,
    pub inferences_per_second: f64,

    pub cache_hits: u64,
    pub nodes_processed: u64,
    pub nodes_per_second: f64,

    pub training_steps: u64,
}

/// Keeps track of global statistics and logs them to a CSV file.
/// Needed for analysis and for resuming training from the last cycle if it was interrupted.
#[derive(Debug)]
pub struct CsvLogger {
    path: PathBuf,
    records: Vec<Record>,
}

impl CsvLogger {
    /// Loads records from the CSV file at the given path, or creates a new,
    /// empty file with a header if it doesn't exist.
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();

        let records = if path.exists() {
            let file = File::open(path)
                .with_context(|| format!("Failed to open log file at {:?}", path))?;
            let mut rdr = csv::Reader::from_reader(file);
            let records = rdr
                .deserialize()
                .collect::<Result<Vec<Record>, _>>()
                .with_context(|| format!("Failed to parse CSV records from {:?}", path))?;
            info!("Loaded {} existing records.", records.len());
            records
        } else {
            info!("No log file found at {:?}, creating a new one.", path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "Failed to create parent directory for log file at {:?}",
                        parent
                    )
                })?;
            }

            // Write the header by serializing a default record, but then re-open the file
            // and delete everything except the header line.
            {
                let file = File::create(path)
                    .with_context(|| format!("Failed to create new log file at {:?}", path))?;
                let mut wtr = csv::Writer::from_writer(file);
                wtr.serialize(Record::default())?;
                wtr.flush()?;
            }

            let file_content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read temporary log file at {:?}", path))?;

            let header_line = match file_content.find('\n') {
                Some(index) => &file_content[..=index],
                None => &file_content, // In case there's no newline
            };

            std::fs::write(path, header_line)
                .with_context(|| format!("Failed to write header to log file at {:?}", path))?;

            Vec::new()
        };

        Ok(Self {
            path: path.to_path_buf(),
            records,
        })
    }

    /// Appends a new record to the in-memory list and writes it to the CSV file.
    pub fn append(&mut self, record: Record) -> anyhow::Result<()> {
        // Open the file in append mode.
        let file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("Failed to open log file for appending at {:?}", self.path))?;

        // Create a writer that does NOT write a new header.
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(file);

        // Serialize the new record and flush it to disk.
        wtr.serialize(&record).with_context(|| {
            format!(
                "Failed to serialize record for cycle {}",
                record.cycle_number
            )
        })?;
        wtr.flush().context("Failed to flush CSV writer")?;

        // Also add it to our in-memory list.
        self.records.push(record);

        Ok(())
    }

    /// Returns the cycle number of the last record, or 0 if there are no records.
    pub fn last_cycle_number(&self) -> Option<u64> {
        self.records.last().map(|r| r.cycle_number)
    }

    pub fn last_training_steps(&self) -> u64 {
        self.records.last().map_or(0, |r| r.training_steps)
    }

    pub fn generate_collector_stats(&self) -> CollectorStats {
        let mut stats = CollectorStats::default();
        for record in &self.records {
            stats.total_games_played += record.total_games_played;
            stats.total_samples_generated += record.total_samples_generated;
        }
        stats
    }
}
