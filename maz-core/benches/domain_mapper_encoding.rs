use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::InputMapper;
use ndarray::Array2;
use rand::prelude::*;
use std::hint::black_box;
use maz_core::mapping::chess_hybrid_canonical_mapper::ChessHybridCanonicalMapper;

fn generate_benchmark_positions(num_positions: usize) -> Vec<TriHexChess> {
    let mut positions = Vec::with_capacity(num_positions);
    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "Generating {} board positions for benchmarking. This may take a moment...",
        num_positions
    );

    while positions.len() < num_positions {
        let mut game = TriHexChess::default();
        let mut move_store = ChessMoveStore::default();

        for _ in 0..500 {
            // Max 500 moves per game
            if game.is_over() {
                break;
            }

            game.update_pseudo_moves(&mut move_store, true);

            if move_store.is_empty() {
                break;
            }

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

fn bench_mapper_encoding(c: &mut Criterion) {
    let positions = generate_benchmark_positions(1000);

    let mapper = ChessDomainMapper;

    let mut input_buffer = Array2::from_elem(mapper.input_board_shape(), false);

    let mut group = c.benchmark_group("Mapper Encoding");

    group.throughput(Throughput::Elements(positions.len() as u64));

    group.bench_function("ChessDomainMapper::encode_input", |b| {
        b.iter(|| {
            for game_state in &positions {
                mapper.encode_input(&mut input_buffer.view_mut(), black_box(game_state));
            }
        })
    });


    let mapper = ChessHybridCanonicalMapper;

    let mut input_buffer = Array2::from_elem(mapper.input_board_shape(), false);

    group.throughput(Throughput::Elements(positions.len() as u64));

    group.bench_function("ChessHybridCanonicalMapper::encode_input", |b| {
        b.iter(|| {
            for game_state in &positions {
                mapper.encode_input(&mut input_buffer.view_mut(), black_box(game_state));
            }
        })
    });


    group.finish();
}

criterion_group!(benches, bench_mapper_encoding);
criterion_main!(benches);
