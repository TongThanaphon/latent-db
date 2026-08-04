//! `LatentDb::build_viable_graph` + `ViableGraph::geodesic` / `random_walk`
//! benchmark -- Issue 1 hot path 3/3.
//!
//! Baseline: `build_viable_graph` is O(n^2) pairwise distance for kNN (see
//! `manifold.rs` module docs) -- rebuilt from scratch on every call, no
//! incremental maintenance. Reference point for any future indexed/ANN graph
//! build (e.g. Issue 8's bandit-guided region exploration builds on this).
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench viable_graph_bench`

use latent_db::{LatentDb, ViableGraph};

#[path = "support/mod.rs"]
mod support;
use support::{assert_monotonic, bench_us, DbConfig, Row};

const DIM: usize = 16;
const K_NEAREST: usize = 6;

const DB_CFG: DbConfig = DbConfig {
    n_subspaces: 4,
    n_pq_centroids: 16,
    n_index_centroids: 8,
    sketch_dim: 8,
    seed: 300,
};

const N_SMALL: usize = 300;
const N_LARGE: usize = 1200;

fn build_graph(db: &LatentDb) -> ViableGraph {
    db.build_viable_graph(|_| true, K_NEAREST, false)
}

fn main() {
    let db_small = support::build_db(
        &support::synthetic_corpus(N_SMALL, DIM, DB_CFG.seed),
        &DB_CFG,
    );
    let db_large = support::build_db(
        &support::synthetic_corpus(N_LARGE, DIM, DB_CFG.seed),
        &DB_CFG,
    );

    // O(n^2) build cost: N_LARGE/N_SMALL = 4x the records should cost
    // roughly 16x, a large enough gap to prove the timer distinguishes a
    // genuine regression from noise.
    let us_build_small = bench_us(2, 10, || {
        std::hint::black_box(build_graph(std::hint::black_box(&db_small)));
    });
    let us_build_large = bench_us(2, 10, || {
        std::hint::black_box(build_graph(std::hint::black_box(&db_large)));
    });
    assert_monotonic(
        us_build_small,
        us_build_large,
        "build_viable_graph n=300 vs n=1200",
    );

    let graph = build_graph(&db_large);
    // Predicate is `|_| true`, so every inserted record survives -- ids are
    // the sequential `0..N_LARGE` `LatentDb::insert` assigns them (see
    // `support::build_db` doc comment).
    let src: u64 = 0;
    let dst: u64 = (N_LARGE - 1) as u64;

    let us_geodesic = bench_us(5, 30, || {
        std::hint::black_box(graph.geodesic(std::hint::black_box(src), std::hint::black_box(dst)));
    });
    let us_random_walk = bench_us(5, 30, || {
        std::hint::black_box(graph.random_walk(std::hint::black_box(src), 20, 0xC0FFEE));
    });

    // Budgets are ~4-6x this machine's measured baseline (see benches/BASELINE.md).
    let rows = [
        Row {
            name: "build_viable_graph(n=300)",
            us: us_build_small,
            budget_us: 8000.0,
        },
        Row {
            name: "build_viable_graph(n=1200)",
            us: us_build_large,
            budget_us: 90000.0,
        },
        Row {
            name: "geodesic(n=1200)",
            us: us_geodesic,
            budget_us: 150.0,
        },
        Row {
            name: "random_walk(n=1200, 20 steps)",
            us: us_random_walk,
            budget_us: 20.0,
        },
    ];

    support::finish(
        &format!("viable_graph_bench (k_nearest={K_NEAREST})"),
        &rows,
    );
}
