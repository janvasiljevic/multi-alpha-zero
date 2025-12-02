use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use rand::prelude::*;

// Generates a set of realistic board positions for benchmarking.
fn generate_benchmark_positions(num_positions: usize) -> Vec<TriHexChess> {
    let mut positions = Vec::with_capacity(num_positions);
    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "Generating {} board positions for benchmarking...",
        num_positions
    );

    while positions.len() < num_positions {
        let mut game = TriHexChess::default();
        let mut move_store = ChessMoveStore::default();

        // Play out a single random game
        for _ in 0..500 { // Max 500 moves per game
            if game.is_over() {
                break;
            }

            game.update_pseudo_moves(&mut move_store, true);

            if move_store.is_empty() {
                break;
            }

            // Before making a move, save the current state
            if positions.len() < num_positions {
                positions.push(game);
            } else {
                return positions;
            }

            let move_index = rng.random_range(0..move_store.len());
            let random_move = move_store.get(move_index).unwrap();

            game.commit_move(&random_move, &move_store);
            game.next_turn(true, &mut move_store, true);
        }
    }

    println!("Finished generating positions.");
    positions
}

fn bench_attack_generation(c: &mut Criterion) {
    // Generate a diverse set of game states.
    let positions = generate_benchmark_positions(2000);

    let mut group = c.benchmark_group("Bitboard Attack Generation");

    // Configure the benchmark to report throughput in terms of positions per second
    group.throughput(Throughput::Elements(positions.len() as u64));

    group.bench_function("calculate_per_piece_bitboard_attack_data", |b| {
        b.iter(|| {
            // We iterate through all our test positions in each run.
            for game_state in &positions {
                std::hint::black_box(game_state.calculate_per_piece_bitboard_attack_data());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_attack_generation);
criterion_main!(benches);