//! `LatentDb::merkle_proof` / `merkle_root` allocation-audit -- Issue 4
//! acceptance criterion: after the first call following a mutation (or
//! construction) pays the full O(n) `merkle_leaves()` rehash + `MerkleTree`
//! rebuild, every further call before the next `insert`/`remove` should
//! reuse the cached tree and levels rather than rebuilding, which shows up
//! as a sharp drop in allocation count -- a rebuild allocates roughly once
//! per stored record (`record_leaf_bytes` inside `merkle_leaves`) plus once
//! per tree level, while a cached lookup only allocates the returned
//! `MerkleProof`'s `siblings` `Vec`.
//!
//! `#![cfg(debug_assertions)]`: see `tests/search_alloc_check.rs` for why
//! this whole file only compiles under `cargo test` (not `--release`).

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
    seed: 801,
};

/// A cold `merkle_proof` call (cache empty -- construction, or the most
/// recent `insert`/`remove` invalidated it) allocates roughly once per
/// stored record; a warm call (no mutation since the last `merkle_proof`/
/// `merkle_root`) should allocate only the handful of bytes the returned
/// `MerkleProof` itself needs, regardless of record count.
#[test]
fn warm_merkle_proof_allocates_far_less_than_a_cold_one() {
    let corpus = support::synthetic_corpus(500, 16, 800);
    let db = support::build_db(&corpus, &DB_CFG);
    let id: u64 = 0;

    // Warmup: settle any one-time lazy allocations unrelated to
    // `merkle_proof` itself (e.g. this thread's TLS/allocator first-touch
    // bookkeeping). Deliberately *not* `merkle_proof`/`merkle_root`, which
    // would prime the very cache this test needs to measure cold.
    for _ in 0..3 {
        std::hint::black_box(db.get_metadata(0));
    }

    reset_alloc_stats();
    let cold_proof = db.merkle_proof(id);
    let (cold_count, cold_bytes) = get_alloc_stats();
    assert!(cold_proof.is_some());

    reset_alloc_stats();
    let warm_proof = db.merkle_proof(id);
    let (warm_count, warm_bytes) = get_alloc_stats();
    assert!(warm_proof.is_some());

    assert!(
        warm_count * 10 < cold_count,
        "expected a warm merkle_proof() call (no mutation since the previous one) to \
         allocate far less than a cold one over 500 records, but cold={cold_count} \
         allocations ({cold_bytes} bytes), warm={warm_count} allocations \
         ({warm_bytes} bytes) -- looks like the tree is still being rebuilt from \
         scratch on every call"
    );
}

/// `insert`/`remove` must invalidate the cache: a `merkle_proof` call right
/// after a mutation should cost a cold (full-rebuild) allocation count
/// again, not silently reuse a now-stale cached tree.
#[test]
fn merkle_proof_pays_a_cold_rebuild_again_right_after_a_remove() {
    let corpus = support::synthetic_corpus(500, 16, 810);
    let mut db = support::build_db(&corpus, &DB_CFG);
    let id: u64 = 0;
    let last_id = (corpus.len() - 1) as u64;

    // Warm the cache, then measure its (cheap) warm cost.
    let _ = db.merkle_proof(id);
    reset_alloc_stats();
    let _ = db.merkle_proof(id);
    let (warm_count, warm_bytes) = get_alloc_stats();

    // A mutation should invalidate the cache -- the next call pays a cold
    // rebuild again, not the warm cost measured above. Removing the last
    // inserted id (rather than `id` itself, which is what's being proved)
    // keeps the proof target live.
    db.remove(last_id);
    reset_alloc_stats();
    let _ = db.merkle_proof(id);
    let (post_mutation_count, post_mutation_bytes) = get_alloc_stats();

    assert!(
        post_mutation_count > warm_count * 10,
        "expected merkle_proof() right after a remove() to pay a cold rebuild \
         (allocation count well above the warm {warm_count} allocations / \
         {warm_bytes} bytes), but got {post_mutation_count} allocations \
         ({post_mutation_bytes} bytes) -- the cache doesn't look invalidated by mutation"
    );
}
