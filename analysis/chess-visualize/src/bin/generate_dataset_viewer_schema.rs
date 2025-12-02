use schemars::_private::serde_json::to_string_pretty;
use schemars::schema_for;
use chess_visualize::DatasetViewerConfig;

fn main() {
    let schema = schema_for!(DatasetViewerConfig);
    std::fs::write(
        "analysis/chess-visualize/dataset-viewer.schema.json",
        to_string_pretty(&schema).unwrap(),
    )
    .expect("Unable to write file");
}
