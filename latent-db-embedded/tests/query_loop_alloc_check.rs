//! 1,000,000-iteration zero-allocation query-loop guard (Issue #24): wires
//! `storage::ZeroAllocLatent` + `topology::ViableNode` + `steering::{
//! steer_next, EdgeWeights}` into one representative query/trajectory-search
//! step, runs it 1,000,000 times, and asserts the whole run allocates
//! exactly 0 bytes on the heap -- reusing this repo's `latent_db::alloc`
//! tracking-allocator harness (`reset_alloc_stats`/`get_alloc_stats`) rather
//! than building a new one, the same harness `tests/search_alloc_check.rs`
//! (repo root) uses.
//!
//! `#![cfg(debug_assertions)]`: same gate as every other `*_alloc_check`
//! test in this repo -- `latent_db::alloc`'s counters only exist in debug
//! builds (see `src/alloc.rs`'s docs), and `latent_db`'s `#[global_allocator]`
//! install is gated the same way. [`allocator_is_wired_into_this_crates_test_binary`]
//! below is this file's own positive control proving that install is
//! actually reachable from *this* crate's test binary, not just the root
//! crate's -- mirroring `tests/search_alloc_check.rs`'s
//! `search_allocates_today_proving_the_harness_detects_it`, which exists for
//! the same reason: `get_alloc_stats()` returns `(0, 0)` both when a run is
//! genuinely alloc-free *and* when the allocator isn't wired up at all, so
//! this file's headline `(0, 0)` assertion is only meaningful once something
//! in the file also proves the harness can report a nonzero count.
//!
//! **Runtime, and why this test is slow on purpose:** `DIM = 768` and no
//! release-mode optimization (this runs under plain `cargo test`, not
//! `--release`, so LLVM does none of its usual auto-vectorization/inlining)
//! means each of the 1,000,000 iterations costs several microseconds even
//! in the cheapest wiring that still does real work -- measured ~7.2s total
//! for this one test on this machine. That is an
//! acceptance-criteria tradeoff, not an oversight: #24 explicitly asks for
//! exactly 1,000,000 iterations, gated the same way as this repo's other
//! `*_alloc_check` tests (i.e. under `cargo test`, not release-only), so
//! there's no smaller-N or release-only escape hatch available here.
//! `steering::steer_next`'s commit/revert loop is already 8-way
//! loop-unrolled (see its own doc comment) specifically because this test's
//! sizing spike showed the naive `Iterator::zip` version costing ~2.4x more
//! per call under `-O0` -- that optimization alone cut this test's runtime
//! roughly in half.

#![cfg(debug_assertions)]

use latent_db::alloc::{get_alloc_stats, reset_alloc_stats};
use latent_db_embedded::steering::{steer_next, EdgeWeights};
use latent_db_embedded::storage::{dot_product, ZeroAllocLatent};
use latent_db_embedded::topology::{Boundary, ViableNode};
use latent_db_embedded::DIM;

fn sample_vector(seed: f32) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = ((i as f32 + seed) * 0.037).sin() * 0.05;
    }
    v
}

/// One representative query/trajectory-search step: score `node`'s live
/// neighbors against the current trajectory position `state` (a MIPS-style
/// raw dot product, not a normalized cosine -- cheaper, and a real
/// zero-alloc embedded caller is exactly the kind that would precompute or
/// skip normalization rather than pay for it every step), reinforce the
/// winning edge's weight, then advance `state` one step toward a fixed
/// steering direction, gated by the topology boundary. Returns what
/// `steer_next` returns: whether the step committed.
fn query_step(
    state: &mut [f32; DIM],
    node: &ViableNode,
    latents: &[ZeroAllocLatent],
    weights: &mut EdgeWeights,
    direction: &[f32; DIM],
    alpha: f32,
    boundary: &Boundary,
) -> bool {
    let mut best_edge = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (edge_idx, &neighbor_id) in node.live_neighbors().iter().enumerate() {
        let score = dot_product(state, &latents[neighbor_id as usize].data);
        if score > best_score {
            best_score = score;
            best_edge = edge_idx;
        }
    }
    weights.update(best_edge, best_score * 0.01);
    steer_next(state, direction, alpha, boundary)
}

/// Every iteration steers by a small `alpha` that stays well inside
/// `boundary`, except every `PROBE_PERIOD`-th iteration, which uses a
/// deliberately oversized `alpha` guaranteed to push at least one coordinate
/// outside `boundary`. That keeps the trajectory itself well-behaved (no
/// dimension permanently saturates the boundary and wedges every later step
/// into the reject branch -- an earlier version of this test tuned the
/// normal step size against the boundary directly and found exactly that
/// failure mode: once *any* dimension pins against an edge, steer_next
/// rejects essentially every subsequent step regardless of alpha's sign,
/// which is not representative of anything) while still regularly forcing
/// `steer_next`'s boundary-rejection/revert branch, so this test's `(0, 0)`
/// alloc assertion actually covers the whole function, not just its
/// always-succeeds path.
const PROBE_PERIOD: usize = 1_000;
const NORMAL_ALPHA: f32 = 0.001;
/// `direction`'s largest coordinate magnitude is `0.05` (see `sample_vector`);
/// `boundary` below is `[-1.0, 1.0]`. `50.0 * 0.05 = 2.5`, comfortably past
/// either edge from anywhere `state` can reach under `NORMAL_ALPHA` steps.
const PROBE_ALPHA: f32 = 50.0;

fn alpha_for(i: usize) -> f32 {
    if i % PROBE_PERIOD == PROBE_PERIOD - 1 {
        PROBE_ALPHA
    } else if i.is_multiple_of(2) {
        NORMAL_ALPHA
    } else {
        -NORMAL_ALPHA
    }
}

#[test]
fn query_trajectory_loop_million_iterations_is_alloc_free() {
    let latents = [
        ZeroAllocLatent::new(sample_vector(0.0), 0),
        ZeroAllocLatent::new(sample_vector(1.0), 1),
        ZeroAllocLatent::new(sample_vector(2.0), 2),
    ];
    let mut node = ViableNode::new(0);
    node.push_neighbor(1);
    node.push_neighbor(2);

    let mut weights = EdgeWeights::zeroed();
    let boundary = Boundary {
        min: -1.0,
        max: 1.0,
    };
    let direction = sample_vector(3.0);
    let mut state = [0.0f32; DIM];

    const ITERATIONS: usize = 1_000_000;

    // Warmup: settle first-touch TLS/lazy init before measuring, same
    // convention `tests/search_alloc_check.rs` (repo root) uses.
    for i in 0..1_000 {
        std::hint::black_box(query_step(
            &mut state,
            &node,
            &latents,
            &mut weights,
            &direction,
            alpha_for(i),
            &boundary,
        ));
    }
    state = [0.0f32; DIM];
    weights = EdgeWeights::zeroed();

    let mut committed = 0usize;
    let mut rejected = 0usize;

    reset_alloc_stats();
    for i in 0..ITERATIONS {
        if query_step(
            &mut state,
            &node,
            &latents,
            &mut weights,
            &direction,
            alpha_for(i),
            &boundary,
        ) {
            committed += 1;
        } else {
            rejected += 1;
        }
    }
    let (count, bytes) = get_alloc_stats();

    assert_eq!(
        (count, bytes),
        (0, 0),
        "query/trajectory loop over {ITERATIONS} iterations (storage + topology + \
         steering wiring) should be fully alloc-free, but observed {count} allocations \
         ({bytes} bytes)"
    );
    // Proves the run above actually exercised both of `steer_next`'s
    // branches, not just its always-succeeds path -- exact count, since the
    // probe schedule is deterministic.
    assert_eq!(
        rejected,
        ITERATIONS / PROBE_PERIOD,
        "expected exactly one rejected (boundary-violating probe) step per \
         {PROBE_PERIOD}-iteration block"
    );
    assert_eq!(committed, ITERATIONS - rejected);
}

/// Positive control: proves `latent_db`'s `#[global_allocator]` install is
/// actually reachable from *this* crate's own test binary before trusting
/// the `(0, 0)` assertion above -- see this file's module doc for why that
/// matters. Mirrors `tests/search_alloc_check.rs`'s (repo root)
/// `search_allocates_today_proving_the_harness_detects_it`.
#[test]
fn allocator_is_wired_into_this_crates_test_binary() {
    for _ in 0..3 {
        let v: Vec<u8> = vec![0u8; 8];
        std::hint::black_box(&v);
    }

    reset_alloc_stats();
    let v: Vec<u8> = vec![0u8; 1024];
    std::hint::black_box(&v);
    let (count, bytes) = get_alloc_stats();

    assert!(
        count > 0,
        "expected the tracking allocator to see this allocation from within \
         latent-db-embedded's own test binary -- got 0, which would mean the \
         million-iteration test's (0, 0) assertion is checking nothing"
    );
    assert!(bytes >= 1024);
}
