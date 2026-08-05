//! `LatentDb::merkle_proof` benchmark -- Issue 4 (cached-levels Merkle
//! tree).
//!
//! Two scenarios, both at the same two record counts as the original
//! full-rebuild baseline (see `benches/BASELINE.md`):
//! - "cold": the cache-invalidating `insert` happens *outside* the timed
//!   window (see `bench_cold_merkle_proof` below), so only the `merkle_proof`
//!   call itself -- the full O(n log n) `merkle_leaves()` rehash +
//!   `MerkleTree::build()` -- is measured, same shape as the pre-Issue-4
//!   baseline and the same "exclude setup from the timed window" convention
//!   `tests/search_alloc_check.rs` uses for its one-time LUT build.
//! - "warm": the cache is primed once, then every timed `merkle_proof` call
//!   has no mutation before it -- the O(log n) cached sibling-path lookup
//!   `ensure_merkle_cache` (see `db.rs`) is meant to provide.
//!
//! N `merkle_proof` calls between two mutations now cost roughly one cold
//! call plus (N-1) warm calls -- O(n log n + N log n) total -- instead of N
//! cold calls -- O(N * n log n) -- which is the drop Issue 4's acceptance
//! criteria asks this bench to demonstrate.
//!
//! `std::time::Instant`, `harness = false` (no criterion dev-dependency).
//!
//! Run: `cargo bench --bench merkle_proof_bench`

#[path = "support/mod.rs"]
mod support;
use latent_db::LatentDb;
use std::time::{Duration, Instant};
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
const WARMUP: usize = 5;
const ITERS: usize = 30;

/// Best-of-`iters` microseconds for a `merkle_proof(id)` call, where every
/// iteration -- warmup and timed alike -- is preceded by an `insert` from
/// `extra` that invalidates the cache first. The insert itself runs
/// *outside* the timer, so this isolates the cold-rebuild cost of
/// `merkle_proof` alone, the same "exclude setup from the timed window"
/// convention `bench_us`'s own callers use for one-time build steps
/// (mirrors `tests/search_alloc_check.rs`'s LUT-build exclusion). `db`'s
/// record count grows by one per iteration (`warmup + iters` inserts total)
/// as a side effect of invalidating the cache this way; `extra` is sized
/// exactly to that, and the resulting record-count drift is small relative
/// to `db`'s starting size, so it doesn't blur the n=200-vs-n=2000 gap
/// `assert_monotonic` checks below.
fn bench_cold_merkle_proof(
    db: &mut LatentDb,
    extra: &mut impl Iterator<Item = Vec<f32>>,
    id: u64,
    warmup: usize,
    iters: usize,
) -> f64 {
    let mut timed_call = || {
        db.insert(&extra.next().unwrap(), "extra").unwrap();
        let t0 = Instant::now();
        std::hint::black_box(db.merkle_proof(std::hint::black_box(id)));
        Instant::now() - t0
    };
    for _ in 0..warmup {
        timed_call();
    }
    let mut best = Duration::MAX;
    for _ in 0..iters {
        let dt = timed_call();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

fn main() {
    // Two record-count scenarios of deliberately different cost: a
    // full-rebuild Merkle tree is O(n), so a 10x larger record set should
    // measure meaningfully slower when cold -- proof the timer distinguishes
    // a genuine regression from noise, not just printing a number.
    let mut db_small = support::build_db(
        &support::synthetic_corpus(N_SMALL, DIM, DB_CFG.seed),
        &DB_CFG,
    );
    let mut db_large = support::build_db(
        &support::synthetic_corpus(N_LARGE, DIM, DB_CFG.seed),
        &DB_CFG,
    );
    // Ids are sequential from 0 (see `support::build_db` doc comment); id 0
    // exists in both since both corpora have >= 1 record, and is never
    // removed below, so it stays provable throughout.
    let id: u64 = 0;

    // "warm": prime the cache once, then repeatedly call `merkle_proof`
    // with no mutation in between.
    let _ = db_small.merkle_proof(id);
    let _ = db_large.merkle_proof(id);
    let warm_small = bench_us(WARMUP, ITERS, || {
        std::hint::black_box(db_small.merkle_proof(std::hint::black_box(id)));
    });
    let warm_large = bench_us(WARMUP, ITERS, || {
        std::hint::black_box(db_large.merkle_proof(std::hint::black_box(id)));
    });

    // "cold": a fresh, never-before-inserted vector goes in before every
    // call (outside the timed window -- see `bench_cold_merkle_proof`),
    // invalidating the cache each time so the measured `merkle_proof` always
    // pays a full rebuild.
    let mut extra_small =
        support::synthetic_corpus(WARMUP + ITERS, DIM, DB_CFG.seed + 101).into_iter();
    let mut extra_large =
        support::synthetic_corpus(WARMUP + ITERS, DIM, DB_CFG.seed + 102).into_iter();
    let cold_small = bench_cold_merkle_proof(&mut db_small, &mut extra_small, id, WARMUP, ITERS);
    let cold_large = bench_cold_merkle_proof(&mut db_large, &mut extra_large, id, WARMUP, ITERS);
    assert_monotonic(cold_small, cold_large, "merkle_proof cold n=200 vs n=2000");

    // The whole point of the cache: a warm call must measure far below a
    // cold one at the same record count, or `ensure_merkle_cache` isn't
    // actually being reused.
    assert!(
        warm_large * 5.0 < cold_large,
        "warm merkle_proof(n={N_LARGE}) measured {warm_large} us, not far below a cold \
         one ({cold_large} us) -- the cache doesn't look like it's being reused"
    );

    // Cold budgets carry over the original full-rebuild baseline's margin
    // (see `benches/BASELINE.md`); warm budgets are generous relative to
    // this machine's measured ~0.1us but still tight enough to catch a
    // regression back to a full rebuild on every call.
    let rows = [
        Row {
            name: "merkle_proof warm (n=200)",
            us: warm_small,
            budget_us: 20.0,
        },
        Row {
            name: "merkle_proof warm (n=2000)",
            us: warm_large,
            budget_us: 20.0,
        },
        Row {
            name: "merkle_proof cold (n=200)",
            us: cold_small,
            budget_us: 600.0,
        },
        Row {
            name: "merkle_proof cold (n=2000)",
            us: cold_large,
            budget_us: 3500.0,
        },
    ];

    support::finish("merkle_proof_bench (cached-levels)", &rows);
}
