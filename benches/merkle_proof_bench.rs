//! `LatentDb::merkle_proof` benchmark -- Issue 1 hot path 2/3.
//!
//! Baseline: full-rebuild Merkle tree. `merkle_proof(id)` rehashes every
//! stored record's leaf and rebuilds the whole tree from scratch on every
//! call (see `db.rs`'s `merkle_tree` doc comment) -- no incremental / cached
//! levels yet. Reference point for the cached-levels optimization (Issue 4).
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench merkle_proof_bench`

#[path = "support/mod.rs"]
mod support;
use support::{assert_monotonic, bench_us, DbConfig, Row};

const DIM: usize = 32;

const DB_CFG: DbConfig = DbConfig {
    n_subspaces: 4,
    n_pq_centroids: 16,
    n_index_centroids: 8,
    sketch_dim: 8,
    seed: 200,
};

const N_SMALL: usize = 200;
const N_LARGE: usize = 2000;

fn main() {
    // Two record-count scenarios of deliberately different cost: a
    // full-rebuild Merkle tree is O(n), so a 10x larger record set should
    // measure meaningfully slower -- proof the timer distinguishes a
    // genuine regression from noise, not just printing a number.
    let db_small = support::build_db(
        &support::synthetic_corpus(N_SMALL, DIM, DB_CFG.seed),
        &DB_CFG,
    );
    let db_large = support::build_db(
        &support::synthetic_corpus(N_LARGE, DIM, DB_CFG.seed),
        &DB_CFG,
    );
    // Ids are sequential from 0 (see `support::build_db` doc comment); id 0
    // exists in both since both corpora have >= 1 record.
    let id: u64 = 0;

    let us_small = bench_us(5, 30, || {
        std::hint::black_box(db_small.merkle_proof(std::hint::black_box(id)));
    });
    let us_large = bench_us(5, 30, || {
        std::hint::black_box(db_large.merkle_proof(std::hint::black_box(id)));
    });
    assert_monotonic(us_small, us_large, "merkle_proof n=200 vs n=2000");

    // Budgets are ~4-5x this machine's measured baseline (see benches/BASELINE.md).
    let rows = [
        Row {
            name: "merkle_proof(n=200)",
            us: us_small,
            budget_us: 600.0,
        },
        Row {
            name: "merkle_proof(n=2000)",
            us: us_large,
            budget_us: 3500.0,
        },
    ];

    support::finish("merkle_proof_bench (full-rebuild baseline)", &rows);
}
