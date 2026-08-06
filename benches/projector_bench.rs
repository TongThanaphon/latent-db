//! `Projector::project_into` benchmark -- Issue #11 (SIMD vector math
//! primitives).
//!
//! Baseline: scalar `row.iter().zip(v).map(|(m,x)| m*x).sum()` per output
//! row. Reference point for this ticket's SIMD dispatch change -- see
//! `benches/BASELINE.md`'s Issue #11 section for the recorded before/after
//! numbers.
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench projector_bench`

use latent_db::Projector;

#[path = "support/mod.rs"]
mod support;
use support::{assert_monotonic, bench_us, Row};

/// Projects every vector in `vectors` through `p`, reusing one scratch
/// output buffer (`Projector::project_into`, not `project`) so the timing
/// isolates the matrix-vector product rather than `Vec` allocation.
/// Accumulates a checksum so the compiler can't hoist the loop away.
fn bench_project(p: &Projector, vectors: &[Vec<f32>]) -> f32 {
    let mut out = vec![0.0f32; p.out_dim()];
    let mut checksum = 0.0f32;
    for v in vectors {
        p.project_into(v, &mut out);
        checksum += out[0];
    }
    checksum
}

const N: usize = 500;

fn main() {
    // Small pair: matches `search_bench`'s real DIM=64 / sketch_dim=16 config.
    let small = Projector::new(64, 16, 42);
    let small_vectors: Vec<Vec<f32>> = support::synthetic_corpus(N, 64, 1);

    // Large pair: 8x the multiply-adds per row (512*64 vs 64*16), large
    // enough to make a per-row SIMD win clearly visible over per-row
    // loop/function-call overhead.
    let large = Projector::new(512, 64, 42);
    let large_vectors: Vec<Vec<f32>> = support::synthetic_corpus(N, 512, 2);

    let us_small = bench_us(3, 20, || {
        std::hint::black_box(bench_project(
            std::hint::black_box(&small),
            std::hint::black_box(&small_vectors),
        ));
    });
    let us_large = bench_us(3, 20, || {
        std::hint::black_box(bench_project(
            std::hint::black_box(&large),
            std::hint::black_box(&large_vectors),
        ));
    });

    assert_monotonic(
        us_small,
        us_large,
        "project 64->16 vs 512->64, n=500 (8x the work per row)",
    );

    // Budgets are ~5x this machine's current (post-SIMD) measured time --
    // not the pre-SIMD scalar numbers in benches/BASELINE.md, which this
    // code no longer runs and so can't be checked against as a regression
    // gate.
    let rows = [
        Row {
            name: "project(64->16, n=500)",
            us: us_small,
            budget_us: 300.0,
        },
        Row {
            name: "project(512->64, n=500)",
            us: us_large,
            budget_us: 15_000.0,
        },
    ];

    support::finish(&format!("projector_bench (n={N})"), &rows);
}
