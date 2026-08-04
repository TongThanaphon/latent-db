//! Allocation-audit harness demonstration -- Issue 1 acceptance criterion.
//!
//! Proves the harness *works*, not that `search` is already allocation-free:
//! `LatentDb::search` decodes a PQ-compressed candidate into a fresh `Vec<f32>`
//! per candidate and collects results into a fresh `Vec<SearchHit>`, so it
//! should -- and does -- report a non-zero allocation count today. Later
//! optimization tickets (e.g. Issue 2's PQ asymmetric distance, which would
//! let `search` score candidates without ever materializing a decoded
//! vector) get to point at this same test and watch the count drop.
//!
//! `#![cfg(debug_assertions)]`: `latent_db::alloc`'s counters only exist in
//! debug builds (see `src/alloc.rs` docs) -- this whole file compiles to
//! nothing under `cargo test --release`, matching how `src/lib.rs` only
//! installs `TrackingAllocator` as the `#[global_allocator]` when
//! `debug_assertions` is on.

#![cfg(debug_assertions)]

use latent_db::alloc::{get_alloc_stats, reset_alloc_stats};

#[path = "../benches/support/mod.rs"]
mod support;
use support::DbConfig;

const DB_CFG: DbConfig = DbConfig {
    n_subspaces: 2,
    n_pq_centroids: 8,
    n_index_centroids: 4,
    sketch_dim: 8,
    seed: 901,
};

#[test]
fn search_allocates_today_proving_the_harness_detects_it() {
    let corpus = support::synthetic_corpus(200, 16, 900);
    let db = support::build_db(&corpus, &DB_CFG);

    // Warmup: settle any one-time lazy allocations unrelated to `search`
    // itself (e.g. this thread's TLS/allocator first-touch bookkeeping).
    for _ in 0..3 {
        let _ = db.search(&corpus[0], 5, db.n_index_centroids());
    }

    reset_alloc_stats();
    let hits = db.search(&corpus[0], 5, db.n_index_centroids());
    std::hint::black_box(&hits);
    let (count, bytes) = get_alloc_stats();

    assert!(
        count > 0,
        "expected search() to allocate today (it decodes each PQ candidate \
         into a fresh Vec and collects hits into a fresh Vec) -- the harness \
         reported zero, which means either search() got allocation-free (great! \
         update this test to lock that in) or the tracking allocator isn't wired up"
    );
    assert!(
        bytes > 0,
        "non-zero alloc count with zero bytes is inconsistent"
    );
}
