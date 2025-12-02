use ahash::RandomState;
use criterion::{Criterion, criterion_group, criterion_main};
use game_hex::game_hex::HexGame;
use game_tri_chess::chess_game::TriHexChess;
use maz_core::mapping::InputMapper;
use maz_core::mapping::chess_canonical_mapper::ChessCanonicalMapper;
use maz_core::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use maz_core::mapping::hex_canonical_mapper::HexCanonicalMapper;
use ndarray::{Array2, s};
use std::hint::black_box;

fn hash_mappable_input(
    hash_builder: &RandomState,
    array: &Array2<bool>,
    num_channels_to_hash: Option<usize>,
) -> u64 {
    match num_channels_to_hash {
        None => hash_builder.hash_one(array),

        Some(n) => hash_builder.hash_one(&array.slice(s![.., 0..n])),
    }
}

fn bench_ahash_on_games(c: &mut Criterion) {
    let game = TriHexChess::default_with_grace_period();
    let hash_builder = RandomState::with_seed(42);

    let mut group = c.benchmark_group("Ahash");
    group.throughput(criterion::Throughput::Elements(1));

    // --- Chess Canonical (baseline) ---
    let mapper_canonical = ChessCanonicalMapper;
    let mut array_canonical = Array2::default(mapper_canonical.input_board_shape());
    mapper_canonical.encode_input(&mut array_canonical.view_mut(), &game);
    // It will return None, causing the full (and correct) hash.
    let num_ch_canonical = mapper_canonical.num_hashable_channels();

    group.bench_function("chess::canonical", |b| {
        b.iter(|| {
            black_box(hash_mappable_input(
                &hash_builder,
                &array_canonical,
                num_ch_canonical,
            ));
        });
    });

    let mapper_domain = ChessDomainMapper;
    let mut array_domain = Array2::default(mapper_domain.input_board_shape());
    mapper_domain.encode_input(&mut array_domain.view_mut(), &game);
    let num_ch_domain = mapper_domain.num_hashable_channels();

    // Benchmark the OLD way (hashing everything) for comparison
    group.bench_function("chess::domain (full hash)", |b| {
        b.iter(|| {
            black_box(hash_builder.hash_one(&array_domain));
        });
    });

    // Benchmark the NEW, efficient subset hash
    group.bench_function("chess::domain (subset hash)", |b| {
        b.iter(|| {
            black_box(hash_mappable_input(
                &hash_builder,
                &array_domain,
                num_ch_domain,
            ));
        });
    });

    // --- Hex (baseline) ---
    let game_hex = HexGame::new(5).unwrap();
    let mapper_hex = HexCanonicalMapper::new(&game_hex);
    let mut array_hex = Array2::default(mapper_hex.input_board_shape());
    mapper_hex.encode_input(&mut array_hex.view_mut(), &game_hex);
    let num_ch_hex = mapper_hex.num_hashable_channels();

    group.bench_function("hex", |b| {
        b.iter(|| {
            black_box(hash_mappable_input(&hash_builder, &array_hex, num_ch_hex));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_ahash_on_games);
criterion_main!(benches);
