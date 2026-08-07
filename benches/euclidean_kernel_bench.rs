//! Isolated `simd_squared_euclidean_f32` kernel benchmark -- Issue #11 (SIMD
//! vector math primitives).
//!
//! `viable_graph_bench`'s `build_viable_graph` end-to-end numbers only show
//! a modest speedup from this ticket's SIMD change (see
//! `benches/BASELINE.md`'s Issue #11 section): `build_viable_graph` also
//! pays an O(n log n) `sort_unstable_by` per node and adjacency-list
//! bookkeeping, both untouched by this ticket, which dilute the O(n^2)
//! distance pass's own speedup in that bench's total. This bench isolates
//! just the arithmetic kernel `manifold::euclidean` calls
//! (`simd_squared_euclidean_f32`, public via `crate::simd`) at the same
//! `dim=16` and roughly the same O(n^2) call count as `viable_graph_bench`'s
//! kNN pass, so the kernel-level speedup is reproducible on its own rather
//! than asserted as one-off prose.
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench euclidean_kernel_bench`

use latent_db::simd::simd_squared_euclidean_f32;

#[path = "support/mod.rs"]
mod support;
use support::{assert_monotonic, bench_us, Row};

const DIM: usize = 16;
// Same order as `viable_graph_bench`'s `build_viable_graph(n=300/1200)` kNN
// pass: N vectors, each compared against every other (N-1) of them.
const N_SMALL: usize = 300;
const N_LARGE: usize = 1200;

/// Runs `n * (n-1)` calls to `simd_squared_euclidean_f32` over `dim`-sized
/// vectors drawn from `vectors` (cycling through pairs), the same shape as
/// one full O(n^2) kNN distance pass. Accumulates a checksum so the
/// compiler can't hoist the loop away.
fn bench_all_pairs(vectors: &[Vec<f32>]) -> f32 {
    let n = vectors.len();
    let mut checksum = 0.0f32;
    for a in 0..n {
        for b in 0..n {
            if a != b {
                checksum += simd_squared_euclidean_f32(&vectors[a], &vectors[b]);
            }
        }
    }
    checksum
}

fn main() {
    let small_vectors = support::synthetic_corpus(N_SMALL, DIM, 1);
    let large_vectors = support::synthetic_corpus(N_LARGE, DIM, 2);

    let us_small = bench_us(2, 10, || {
        std::hint::black_box(bench_all_pairs(std::hint::black_box(&small_vectors)));
    });
    let us_large = bench_us(2, 10, || {
        std::hint::black_box(bench_all_pairs(std::hint::black_box(&large_vectors)));
    });

    // N_LARGE/N_SMALL = 4x -> ~16x the pair count, a large enough gap to
    // prove the timer distinguishes a genuine cost difference from noise.
    assert_monotonic(
        us_small,
        us_large,
        "all-pairs squared-euclidean n=300 vs n=1200",
    );

    // Budgets are ~4-6x this machine's measured post-SIMD baseline (see
    // benches/BASELINE.md).
    let rows = [
        Row {
            name: "all_pairs_euclidean(n=300)",
            us: us_small,
            budget_us: 1200.0,
        },
        Row {
            name: "all_pairs_euclidean(n=1200)",
            us: us_large,
            budget_us: 15_000.0,
        },
    ];

    support::finish(&format!("euclidean_kernel_bench (dim={DIM})"), &rows);
}
