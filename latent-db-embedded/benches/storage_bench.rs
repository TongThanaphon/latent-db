//! Micro-benchmark for `latent_db_embedded::storage::{dot_product,
//! cosine_similarity}` at `DIM = 768` (Issue #21).
//!
//! `std::time::Instant`, `harness = false`, a PASS/REGRESSION budget table,
//! and an `assert_monotonic` harness self-check -- the same shape
//! `benches/support/mod.rs` (root crate) shares across every `benches/*.rs`
//! binary there. Not reused directly here since `support::build_db`/
//! `synthetic_corpus` pull in `rand`, a dependency this crate's bench has no
//! other reason to add -- the handful of helpers actually needed
//! (`bench_us`, `Row`, `report`, `assert_monotonic`) are small enough to
//! duplicate rather than restructure the shared module around.
//!
//! Run: `cargo bench -p latent-db-embedded --bench storage_bench`

use latent_db_embedded::storage::{cosine_similarity, dot_product};
use latent_db_embedded::DIM;
use std::time::{Duration, Instant};

fn sample_vector(seed: f32) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = ((i as f32 + seed) * 0.037).sin();
    }
    v
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

fn bench_calls(
    f: impl Fn(&[f32; DIM], &[f32; DIM]) -> f32,
    a: &[f32; DIM],
    b: &[f32; DIM],
    calls: usize,
) -> f64 {
    bench_us(2, 10, || {
        let mut acc = 0.0f32;
        for _ in 0..calls {
            acc += f(std::hint::black_box(a), std::hint::black_box(b));
        }
        std::hint::black_box(acc);
    })
}

fn main() {
    let a = sample_vector(0.0);
    let b = sample_vector(1.0);

    const CALLS_SMALL: usize = 2_500;
    const CALLS_LARGE: usize = 10_000;

    let dot_small_us = bench_calls(dot_product, &a, &b, CALLS_SMALL);
    let dot_large_us = bench_calls(dot_product, &a, &b, CALLS_LARGE);
    let cosine_small_us = bench_calls(cosine_similarity, &a, &b, CALLS_SMALL);
    let cosine_large_us = bench_calls(cosine_similarity, &a, &b, CALLS_LARGE);

    // 4x the call count should never measure faster -- proves the timer
    // actually distinguishes cost here before trusting the budget table
    // below.
    assert_monotonic(
        dot_small_us,
        dot_large_us,
        &format!("dot_product n={CALLS_SMALL} vs n={CALLS_LARGE}"),
    );
    assert_monotonic(
        cosine_small_us,
        cosine_large_us,
        &format!("cosine_similarity n={CALLS_SMALL} vs n={CALLS_LARGE}"),
    );

    // Budgets are ~4-6x this machine's measured post-implementation
    // baseline (see benches/BASELINE.md's storage_bench section), same
    // generous-margin convention every other bench in this workspace uses.
    let rows = [
        Row {
            name: "dot_product",
            us: dot_large_us,
            budget_us: 6_000.0,
        },
        Row {
            name: "cosine_similarity",
            us: cosine_large_us,
            budget_us: 10_000.0,
        },
    ];

    let all_pass = report(
        &format!("storage_bench (DIM={DIM}, {CALLS_LARGE} calls)"),
        &rows,
    );
    if all_pass {
        std::process::exit(0);
    } else {
        eprintln!("storage_bench: one or more scenarios exceeded its latency budget");
        std::process::exit(1);
    }
}
