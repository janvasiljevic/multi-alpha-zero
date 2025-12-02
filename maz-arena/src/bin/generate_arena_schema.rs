use maz_arena::arena_config::ArenaConfigList;
use schemars::_private::serde_json::to_string_pretty;
use schemars::schema_for;

fn main() {
    let schema = schema_for!(ArenaConfigList);
    std::fs::write(
        "maz-arena/arena-config.schema.json",
        to_string_pretty(&schema).unwrap(),
    )
    .expect("Unable to write file");
}
