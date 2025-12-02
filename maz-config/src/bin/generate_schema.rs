use maz_config::config::AppConfig;
use schemars::_private::serde_json::to_string_pretty;
use schemars::schema_for;

fn main() {
    let schema = schema_for!(AppConfig);
    std::fs::write(
        "config.schema.json",
        to_string_pretty(&schema).unwrap(),
    )
    .expect("Unable to write file");
}
