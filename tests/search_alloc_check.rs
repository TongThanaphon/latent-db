//! Allocation-audit harness demonstration -- Issue 1 acceptance criterion --
//! plus Issue 2's zero-allocation acceptance criterion for the PQ
//! asymmetric-distance candidate-scoring loop.
//!
//! The first test below proves the harness *works*, not that `search` is
//! allocation-free overall: `LatentDb::search` still clones each hit's
//! metadata `String` and collects results into a fresh `Vec<SearchHit>`, so
//! it should -- and does -- report a non-zero allocation count. What Issue
//! 2 actually removed is the per-candidate PQ decode inside the *scoring*
//! step; the second test isolates exactly that step (via
//! `PqCodec::build_query_lut` + `QueryLut::cosine_score`, excluding the
//! one-time per-query LUT build) and asserts it allocates nothing.
//!
//! `#![cfg(debug_assertions)]`: `latent_db::alloc`'s counters only exist in
//! debug builds (see `src/alloc.rs` docs) -- this whole file compiles to
//! nothing under `cargo test --release`, matching how `src/lib.rs` only
//! installs `TrackingAllocator` as the `#[global_allocator]` when
//! `debug_assertions` is on.

#![cfg(debug_assertions)]

use latent_db::alloc::{get_alloc_stats, reset_alloc_stats};
use latent_db::PqCodec;

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
        "expected search() to allocate today (it clones each hit's metadata \
         String and collects hits into a fresh Vec<SearchHit>) -- the harness \
         reported zero, which means either search() got allocation-free (great! \
         update this test to lock that in) or the tracking allocator isn't wired up"
    );
    assert!(
        bytes > 0,
        "non-zero alloc count with zero bytes is inconsistent"
    );
}

/// Issue 2 acceptance criterion: the candidate-scoring loop (summing a
/// candidate's PQ codes against a precomputed per-query LUT) performs zero
/// heap allocations, excluding the one-time per-query LUT build.
#[test]
fn candidate_scoring_loop_is_alloc_free_excluding_lut_build() {
    let corpus = support::synthetic_corpus(300, 32, 950);
    let codec = PqCodec::train(&corpus, 4, 16, 15, 951);
    let codes: Vec<Vec<u8>> = corpus.iter().map(|v| codec.encode(v)).collect();
    let query = &corpus[0];

    // Warmup, and build (but don't yet measure) the per-query LUT -- that
    // one-time build is explicitly excluded from this test's assertion.
    let lut = codec.build_query_lut(query);
    for c in &codes {
        std::hint::black_box(lut.cosine_score(c));
    }

    reset_alloc_stats();
    let mut total = 0.0f32;
    for c in &codes {
        total += lut.cosine_score(c);
    }
    std::hint::black_box(total);
    let (count, bytes) = get_alloc_stats();

    assert_eq!(
        count, 0,
        "candidate-scoring loop should be alloc-free (excluding the one-time \
         per-query LUT build), but observed {count} allocations ({bytes} bytes)"
    );
}
