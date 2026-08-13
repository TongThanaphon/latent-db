//! Micro-benchmark for the query/trajectory-search step wired together in
//! `tests/query_loop_alloc_check.rs` -- `storage::{ZeroAllocLatent,
//! dot_product}` + `topology::ViableNode` + `steering::{steer_next,
//! EdgeWeights}` (Issue #24).
//!
//! `std::time::Instant`, `harness = false`, a PASS/REGRESSION budget table,
//! and an `assert_monotonic` harness self-check -- same shape as
//! `benches/storage_bench.rs` (this crate) and `benches/support/mod.rs`
//! (root crate). The per-iteration `query_step` logic is duplicated from
//! `tests/query_loop_alloc_check.rs` rather than shared, matching
//! `storage_bench.rs`'s own precedent (see its header comment) for why a
//! handful of small, target-specific helpers get copied instead of
//! restructured into a shared module.
//!
//! Run: `cargo bench -p latent-db-embedded --bench query_loop_bench`

use latent_db_embedded::steering::{steer_next, EdgeWeights};
use latent_db_embedded::storage::{dot_product, ZeroAllocLatent};
use latent_db_embedded::topology::{Boundary, ViableNode};
use latent_db_embedded::DIM;
use std::time::{Duration, Instant};

fn sample_vector(seed: f32) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = ((i as f32 + seed) * 0.037).sin() * 0.05;
    }
    v
}

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

/// Best-of-`iters` wall-clock microseconds for a closure, after `warmup`
/// untimed calls. Mirrors `benches/support/mod.rs::bench_us` at the repo
/// root.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = Instant::now() - t0;
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

fn fmt_us(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.2} us")
    } else {
        format!("{:.3} ms", us / 1000.0)
    }
}

/// One row of a bench report: a named scenario measured against a latency
/// budget. Mirrors `benches/support/mod.rs::Row`.
struct Row {
    name: &'static str,
    us: f64,
    budget_us: f64,
}

impl Row {
    fn pass(&self) -> bool {
        self.us <= self.budget_us
    }
}

/// Print a report table and return whether every row passed its budget.
/// Mirrors `benches/support/mod.rs::report`.
fn report(title: &str, rows: &[Row]) -> bool {
    println!("=== {title} ===\n");
    println!(
        "{:<28} {:>14} {:>14}  verdict",
        "scenario", "measured", "budget"
    );
    println!("{:-<28} {:-<14} {:-<14}  -------", "", "", "");
    let mut all_pass = true;
    for row in rows {
        let pass = row.pass();
        if !pass {
            all_pass = false;
        }
        println!(
            "{:<28} {:>14} {:>14}  {}",
            row.name,
            fmt_us(row.us),
            fmt_us(row.budget_us),
            if pass { "PASS" } else { "REGRESSION" }
        );
    }
    println!();
    all_pass
}

/// Two scenarios of deliberately different cost should measure in the
/// expected order -- proves the timer distinguishes a genuinely slower call
/// from a faster one, not just noise. Mirrors
/// `benches/support/mod.rs::assert_monotonic`.
fn assert_monotonic(smaller_us: f64, larger_us: f64, context: &str) {
    assert!(
        larger_us >= smaller_us,
        "harness sanity ({context}): the larger workload should never measure faster \
         than the smaller one (got {larger_us} us vs {smaller_us} us) -- timer or \
         scenario is broken"
    );
}

fn bench_query_steps(calls: usize) -> f64 {
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

    bench_us(2, 10, || {
        for i in 0..calls {
            let alpha = if i % 2 == 0 { 0.001 } else { -0.001 };
            let committed = query_step(
                std::hint::black_box(&mut state),
                std::hint::black_box(&node),
                std::hint::black_box(&latents),
                std::hint::black_box(&mut weights),
                std::hint::black_box(&direction),
                alpha,
                std::hint::black_box(&boundary),
            );
            std::hint::black_box(committed);
        }
    })
}

fn main() {
    const CALLS_SMALL: usize = 2_500;
    const CALLS_LARGE: usize = 10_000;

    let small_us = bench_query_steps(CALLS_SMALL);
    let large_us = bench_query_steps(CALLS_LARGE);
    let per_iter_us = large_us / CALLS_LARGE as f64;

    // 4x the call count should never measure faster -- proves the timer
    // actually distinguishes cost here before trusting the budget table
    // below.
    assert_monotonic(
        small_us,
        large_us,
        &format!("query_step n={CALLS_SMALL} vs n={CALLS_LARGE}"),
    );

    // Budget is a generous margin over this machine's measured
    // post-implementation baseline (see benches/BASELINE.md's
    // query_loop_bench section), same convention every other bench in this
    // workspace uses. This is a release-profile (`cargo bench`) figure, not
    // the debug-mode per-iteration cost `tests/query_loop_alloc_check.rs`
    // pays under plain `cargo test` -- LLVM optimizes this build.
    let rows = [Row {
        name: "query_step (per-iter)",
        us: per_iter_us,
        budget_us: 3.0,
    }];

    let all_pass = report(
        &format!("query_loop_bench (DIM={DIM}, {CALLS_LARGE} calls)"),
        &rows,
    );
    if all_pass {
        std::process::exit(0);
    } else {
        eprintln!("query_loop_bench: one or more scenarios exceeded its latency budget");
        std::process::exit(1);
    }
}
