//! Allocation-audit harness demonstration -- Issue 1 acceptance criterion --
//! plus Issue 2's zero-allocation acceptance criterion for the PQ
//! asymmetric-distance candidate-scoring loop, and Issue 3's zero-allocation
//! acceptance criterion for `LatentDb::insert`'s flat record-storage arena.
//!
//! The first test below proves the harness *works*, not that `search` is
//! allocation-free overall: `LatentDb::search` still clones each hit's
//! metadata `String` and collects results into a fresh `Vec<SearchHit>`, so
//! it should -- and does -- report a non-zero allocation count. What Issue
//! 2 actually removed is the per-candidate PQ decode inside the *scoring*
//! step; the second test isolates exactly that step (via
//! `PqCodec::build_query_lut` + `QueryLut::cosine_score`, excluding the
//! one-time per-query LUT build) and asserts it allocates nothing. The third
//! test isolates `insert`'s record-storage write (codes + metadata into
//! `RecordArena`) once the arena's own geometric-doubling growth has
//! stabilized past the ids under test.
//!
//! `#![cfg(debug_assertions)]`: `latent_db::alloc`'s counters only exist in
//! debug builds (see `src/alloc.rs` docs) -- this whole file compiles to
//! nothing under `cargo test --release`, matching how `src/lib.rs` only
//! installs `TrackingAllocator` as the `#[global_allocator]` when
//! `debug_assertions` is on.

#![cfg(debug_assertions)]

use latent_db::alloc::{get_alloc_stats, reset_alloc_stats};
use latent_db::{EvictionPolicy, LatentDb, PqCodec};

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

/// Issue 3 acceptance criterion: steady-state `insert()` -- once the
/// `RecordArena`'s geometric-doubling growth (16, 32, 64, ..., 1024, ...)
/// for *every* arena buffer (the four parallel `codes`/`hash`/`meta_span`/
/// `live` arrays, and `meta_bytes` on its own independent amortized
/// schedule) has already stabilized past the ids/metadata volume being
/// inserted -- performs zero heap allocations. 1000 warmup inserts push all
/// five buffers' capacity well past what the 24 ids measured below
/// (1000..1024) need, so none of them trigger a grow.
///
/// This only holds when the caller already owns the metadata `String`, as
/// it does here (every metadata `String` is built *before* the measured
/// window and moved, not cloned, into `insert`): `impl Into<String>` is the
/// identity for an already-owned `String`, so no conversion allocation
/// happens inside `insert` itself. A caller passing a borrowed `&str` (or a
/// fresh `format!(..)`) still allocates once per call for that conversion
/// -- see `insert_with_borrowed_str_metadata_allocates_once_for_the_conversion`
/// below -- since `insert`'s `impl Into<String>` signature is frozen by
/// Issue 3's own acceptance criteria and can't be changed to avoid it.
/// With an owned `String`, the only remaining allocation `insert` could
/// possibly do is one of the arena's own growth points (or, if the arena
/// isn't the bottleneck it's meant to be, `hash_to_id`'s `HashMap` growth
/// or `CentroidIndex`'s per-bucket `Vec` growth -- both pre-existing,
/// amortized data structures untouched by this ticket's arena change).
#[test]
fn insert_within_arena_capacity_is_alloc_free() {
    let corpus = support::synthetic_corpus(1024, 16, 970);
    let mut db = LatentDb::build(
        &corpus,
        DB_CFG.n_subspaces,
        DB_CFG.n_pq_centroids,
        DB_CFG.n_index_centroids,
        DB_CFG.sketch_dim,
        DB_CFG.seed,
    );

    let warmup_meta: Vec<String> = (0..1000).map(|i| format!("record-{i}")).collect();
    for (i, m) in warmup_meta.into_iter().enumerate() {
        db.insert(&corpus[i], m).unwrap();
    }

    let timed_meta: Vec<String> = (1000..1024).map(|i| format!("record-{i}")).collect();
    reset_alloc_stats();
    for (i, m) in timed_meta.into_iter().enumerate() {
        db.insert(&corpus[1000 + i], m).unwrap();
    }
    let (count, bytes) = get_alloc_stats();

    assert_eq!(
        count, 0,
        "steady-state insert() within arena capacity should be alloc-free, \
         but observed {count} allocations ({bytes} bytes) over 24 inserts"
    );
}

/// Documents the boundary the test above depends on: a caller that doesn't
/// already own a `String` still pays at least one allocation per `insert()`
/// call, for the `impl Into<String>` conversion, even within arena
/// capacity. (Not pinned to exactly one: `hash_to_id`'s `HashMap` and
/// `CentroidIndex`'s per-bucket `Vec` are also amortized-growth structures
/// that can happen to cross their own growth point on any given call,
/// independent of this test; only the arena's own five buffers are what
/// the ticket promises steady-state growth-free access to.) This isn't a
/// bug to fix -- `insert`'s signature is frozen by Issue 3's own acceptance
/// criteria -- just the honest edge of "steady-state insert within arena
/// capacity is alloc-free".
#[test]
fn insert_with_borrowed_str_metadata_allocates_once_for_the_conversion() {
    let corpus = support::synthetic_corpus(32, 16, 971);
    let mut db = LatentDb::build(
        &corpus,
        DB_CFG.n_subspaces,
        DB_CFG.n_pq_centroids,
        DB_CFG.n_index_centroids,
        DB_CFG.sketch_dim,
        DB_CFG.seed,
    );
    // Warm up past the arena's first growth boundary (starts at capacity
    // 16, doubles to 32 once a 17th id needs a slot) so the timed insert
    // below can't be conflated with a capacity grow -- it should measure
    // only the `impl Into<String>` conversion.
    for v in &corpus[..20] {
        db.insert(v, "warmup").unwrap();
    }

    reset_alloc_stats();
    db.insert(&corpus[20], "borrowed-str-metadata").unwrap();
    let (count, bytes) = get_alloc_stats();

    assert!(
        count >= 1,
        "expected at least one allocation (the `impl Into<String>` conversion) for a \
         borrowed &str metadata argument, got {count} allocations ({bytes} bytes)"
    );
}

/// Issue 3's other acceptance criterion: `LowestEnergy` eviction evaluates
/// each candidate's `energy()` exactly once (a precomputed list, then
/// sorted) rather than recomputing it inside the sort comparator. `energy()`
/// itself isn't optimized to be alloc-free (`pq.decode` + `projector.project`
/// -- 2 allocations per call, by design; see the fix's own doc comment), so
/// its call count is directly observable through the alloc harness: a
/// once-per-candidate precompute over n=100 candidates costs on the order of
/// 2n = 200 allocations, while recomputing inside an n log n sort comparator
/// would cost on the order of 2*n*log2(n) ~= 1330.
#[test]
fn lowest_energy_eviction_precomputes_energy_once_per_candidate() {
    let corpus = support::synthetic_corpus(100, 16, 990);
    let mut db = LatentDb::build(
        &corpus,
        DB_CFG.n_subspaces,
        DB_CFG.n_pq_centroids,
        DB_CFG.n_index_centroids,
        DB_CFG.sketch_dim,
        DB_CFG.seed,
    );
    db.set_eviction_policy(EvictionPolicy::LowestEnergy);
    for v in &corpus {
        db.insert(v, "x").unwrap();
    }
    assert_eq!(db.len(), 100);

    // A single `set_record_budget` call below `len()` triggers exactly one
    // `enforce_budget()` pass that scores all 100 currently-live candidates
    // in one shot (unlike per-insert eviction, which would spread the same
    // total work across many smaller passes and blur the O(n) vs. O(n log n)
    // signal this test is isolating).
    reset_alloc_stats();
    db.set_record_budget(50);
    let (count, bytes) = get_alloc_stats();
    assert_eq!(db.len(), 50);

    assert!(
        count < 400,
        "LowestEnergy eviction over 100 candidates allocated {count} times ({bytes} bytes) -- \
         expected roughly 2*n ~= 200 for a once-per-candidate energy() precompute; a count this \
         high looks like energy() is being recomputed inside the sort comparator again \
         (~2*n*log2(n) ~= 1330 for n=100)"
    );
}
