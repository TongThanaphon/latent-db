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
//! install is gated the same way. Verified directly before writing this
//! test that the allocator is actually reachable from *this* crate's test
//! binary (not just the root crate's own tests): `#[global_allocator]` is
//! process-wide, and `latent-db-embedded`'s `[dev-dependencies]` on
//! `latent-db` pulls that static into this binary's link too.
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
        // Small magnitude: keeps every `steer_next` step below well inside
        // `BOUNDARY` for the whole run, without needing a dependency-free
        // RNG to pick a dynamically safe step size.
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
/// steering direction, gated by the topology boundary.
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
        let alpha = if i % 2 == 0 { 0.001 } else { -0.001 };
        std::hint::black_box(query_step(
            &mut state,
            &node,
            &latents,
            &mut weights,
            &direction,
            alpha,
            &boundary,
        ));
    }
    state = [0.0f32; DIM];
    weights = EdgeWeights::zeroed();

    reset_alloc_stats();
    for i in 0..ITERATIONS {
        let alpha = if i % 2 == 0 { 0.001 } else { -0.001 };
        std::hint::black_box(query_step(
            &mut state,
            &node,
            &latents,
            &mut weights,
            &direction,
            alpha,
            &boundary,
        ));
    }
    let (count, bytes) = get_alloc_stats();

    assert_eq!(
        (count, bytes),
        (0, 0),
        "query/trajectory loop over {ITERATIONS} iterations (storage + topology + \
         steering wiring) should be fully alloc-free, but observed {count} allocations \
         ({bytes} bytes)"
    );
}
