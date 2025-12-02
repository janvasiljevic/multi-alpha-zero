use analysis_util::parquet_reader::ParquetDataReader;
use game_hex::game_hex::HexGame;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use maz_core::mapping::{InputMapper, ReverseInputMapper};
use maz_core::values_const::ValuesPov;
use maz_util::network::policy_to_hashmap;
use ndarray::Array2;

pub fn main() {
    let file_path = "./testing/final_samples_canonical_test_new.parquet";
    let samples = ParquetDataReader::read_file(std::path::Path::new(file_path)).unwrap();

    let policy_len = samples.samples[0].inner.legal_moves_mask.len();

    let hex_radius = match policy_len {
        61 => 5,
        37 => 4,
        19 => 3,
        7 => 2,
        _ => panic!("Unexpected policy length: {}", policy_len),
    };

    println!(
        "Hex radius deduced from policy length {}: {}",
        policy_len, hex_radius
    );

    let sim_index = 1;

    let hex_game = HexGame::new(hex_radius).unwrap();

    let mapper = HexCanonicalMapper::new(&hex_game);
    let input_shape = mapper.input_board_shape();
    let all_sim_ids = samples.get_unique_sim_ids();
    let game_1 = samples.get_all_samples_for_sim_id(
        all_sim_ids
            .get(sim_index)
            .expect("Sim ID not found in the samples"),
    );

    for sample in game_1 {
        let input_data = Array2::from_shape_vec(
            (input_shape[0], input_shape[1]),
            sample.inner.encoded_board.clone(),
        )
        .unwrap();

        let board =
            mapper.decode_input(&input_data.view(), &vec![sample.inner.player_index as f32]);

        let mut moves = hex_game.get_valid_empty_cells();

        let recorded_policy_map =
            policy_to_hashmap(&sample.inner.mcts_policy, &board, &mut moves, &mapper);

        let q_values =
            ValuesPov::<3>::from_slice(&sample.inner.q_values, 0.0, sample.inner.player_index)
                .abs()
                .value_abs
                .to_vec();

        let outcome = ValuesPov::<3>::from_slice(&sample.z_values, 0.0, sample.inner.player_index)
            .abs()
            .value_abs
            .to_vec();

        println!(
            "\n\nSim ID: {}. Move: {}/{}. Player {:?} (offset of: {}). Raw Q-values: {} Raw Z-values: {}. Abs Q-values: {}. Abs Z-values: {}.\n{}",
            sample.sim_id,
            sample.inner.current_move_count,
            sample.moves_left + sample.inner.current_move_count,
            sample.inner.player_index + 1,
            sample.inner.player_index,
            sample
                .inner
                .q_values
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<String>>()
                .join(", "),
            sample
                .z_values
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<String>>()
                .join(", "),
            q_values
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<String>>()
                .join(", "),
            outcome
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<String>>()
                .join(", "),
            board.fancy_debug_visualize_board_indented(Some((recorded_policy_map, outcome)))
        );
    }
}
