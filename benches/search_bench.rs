//! `LatentDb::search` / `search_steered` benchmark -- Issue 1 hot path 1/3.
//!
//! Baseline: scalar cosine similarity over PQ-decoded candidates, HashMap-
//! backed record storage. Reference point for later PQ/SIMD optimization
//! tickets (Issue 2, Issue 11).
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench search_bench`

use latent_db::SteeringVector;

#[path = "support/mod.rs"]
mod support;
use support::{assert_monotonic, bench_us, self_check_regression_gate_fires, DbConfig, Row};

const N: usize = 2000;
const DIM: usize = 64;
const K: usize = 10;

const DB_CFG: DbConfig = DbConfig {
    n_subspaces: 8,
    n_pq_centroids: 16,
    n_index_centroids: 32,
    sketch_dim: 16,
    seed: 100,
};

fn main() {
    let corpus = support::synthetic_corpus(N, DIM, DB_CFG.seed);
    let db = support::build_db(&corpus, &DB_CFG);
    let query = support::synthetic_corpus(1, DIM, DB_CFG.seed.wrapping_add(999)).remove(0);

    // Two `nprobe` scenarios of deliberately different cost: probing a
    // single centroid bucket vs. every bucket (exhaustive-equivalent). This
    // demonstrates the harness distinguishes a genuinely slower call from a
    // faster one, not just noise -- the same mechanism a later ticket's
    // "still under budget" claim relies on.
    let us_low_nprobe = bench_us(5, 50, || {
        std::hint::black_box(db.search(std::hint::black_box(&query), K, 1));
    });
    let us_high_nprobe = bench_us(5, 50, || {
        std::hint::black_box(db.search(std::hint::black_box(&query), K, DB_CFG.n_index_centroids));
    });
    assert_monotonic(
        us_low_nprobe,
        us_high_nprobe,
        "search nprobe=1 vs nprobe=all",
    );

    let steering = SteeringVector::new(vec![1.0; DIM], 0.5, 10.0).unwrap();
    let us_steered = bench_us(5, 50, || {
        std::hint::black_box(db.search_steered(
            std::hint::black_box(&query),
            K,
            DB_CFG.n_index_centroids,
            &steering,
        ));
    });

    self_check_regression_gate_fires();

    // Budgets are ~4-6x this machine's measured baseline (see benches/BASELINE.md)
    // -- enough slack to stay green on slower CI hardware while still catching
    // a genuine multi-x regression.
    let rows = [
        Row {
            name: "search(nprobe=1)",
            us: us_low_nprobe,
            budget_us: 100.0,
        },
        Row {
            name: "search(nprobe=all)",
            us: us_high_nprobe,
            budget_us: 1800.0,
        },
        Row {
            name: "search_steered(nprobe=all)",
            us: us_steered,
            budget_us: 1800.0,
        },
    ];

    support::finish(&format!("search_bench (N={N} dim={DIM} k={K})"), &rows);
}
