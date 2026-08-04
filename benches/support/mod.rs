//! Shared bench infrastructure: `std::time::Instant`-based timing (no
//! criterion -- not a dev-dependency of this crate, matching katgpt-rs's own
//! bench convention of hand-rolled timers in `harness = false` binaries), a
//! deterministic synthetic corpus + `LatentDb` builder, and a PASS/REGRESSION
//! gate printer shared by every `benches/*.rs` binary via
//! `#[path = "support/mod.rs"]`.
//!
//! Lives at `support/mod.rs` (not `support.rs`) so cargo's bench
//! auto-discovery (`benches/*.rs`, `benches/*/main.rs`) never treats it as a
//! bench target of its own -- no `autobenches = false` needed in Cargo.toml.

#![allow(dead_code)] // each bench binary only uses a subset of these helpers.

use latent_db::LatentDb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

/// Deterministic synthetic corpus: `n` vectors of `dim` floats in `[-1, 1)`.
pub fn synthetic_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// The five `LatentDb::build` params that always travel together across
/// every bench's setup.
pub struct DbConfig {
    pub n_subspaces: usize,
    pub n_pq_centroids: usize,
    pub n_index_centroids: usize,
    pub sketch_dim: usize,
    pub seed: u64,
}

/// Train a `LatentDb` on `corpus` per `cfg`, then insert every vector in it.
/// Ids come back sequential (`0..corpus.len()`, per `LatentDb::insert`'s
/// `next_id` counter) as long as `corpus` has no exact-duplicate vectors --
/// true here since `synthetic_corpus`'s float draws collide with
/// probability ~0.
pub fn build_db(corpus: &[Vec<f32>], cfg: &DbConfig) -> LatentDb {
    let mut db = LatentDb::build(
        corpus,
        cfg.n_subspaces,
        cfg.n_pq_centroids,
        cfg.n_index_centroids,
        cfg.sketch_dim,
        cfg.seed,
    );
    for (i, v) in corpus.iter().enumerate() {
        db.insert(v, format!("record-{i}")).unwrap();
    }
    db
}

/// Best-of-`iters` wall-clock microseconds for a closure, after `warmup`
/// untimed calls. Best-of-N (not mean) filters out scheduler jitter/page
/// faults on the first few calls while still reporting genuine hot-path cost.
pub fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
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

pub fn fmt_us(us: f64) -> String {
    if us < 1.0 {
        format!("{:.2} ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{:.2} us", us)
    } else {
        format!("{:.3} ms", us / 1000.0)
    }
}

/// One row of a bench report: a named scenario measured against a latency
/// budget. `budget_us` should be set generously above this machine's actual
/// baseline (a few-x safety margin) so `cargo bench` stays green across
/// hardware while still catching a genuine multi-x regression.
pub struct Row {
    pub name: &'static str,
    pub us: f64,
    pub budget_us: f64,
}

impl Row {
    pub fn pass(&self) -> bool {
        self.us <= self.budget_us
    }
}

/// Print a report table and return whether every row passed its budget.
/// Mirrors the PASS/FAIL table + `all_pass` idiom used throughout
/// katgpt-rs's own `harness = false` benches (e.g. `procrustes_bench`).
pub fn report(title: &str, rows: &[Row]) -> bool {
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
/// expected order. Every bench uses this to prove its timer distinguishes a
/// genuinely slower call from a faster one, not just noise.
pub fn assert_monotonic(smaller_us: f64, larger_us: f64, context: &str) {
    assert!(
        larger_us >= smaller_us,
        "harness sanity ({context}): the larger workload should never measure faster \
         than the smaller one (got {larger_us} us vs {smaller_us} us) -- timer or \
         scenario is broken"
    );
}

/// Print the report table, then exit the process: 0 if every row passed its
/// budget, 1 (with a diagnostic on stderr) otherwise. Matches the
/// `all_pass` + `std::process::exit` idiom katgpt-rs's own benches use
/// (e.g. `procrustes_bench`).
pub fn finish(title: &str, rows: &[Row]) -> ! {
    let all_pass = report(title, rows);
    if all_pass {
        std::process::exit(0);
    } else {
        eprintln!("{title}: one or more scenarios exceeded its latency budget");
        std::process::exit(1);
    }
}

/// Harness self-check: prove the PASS/REGRESSION gate can actually *fire*,
/// not just pass. Times a deliberately slow synthetic operation (a 2ms
/// sleep -- reliable across hardware, unlike trying to force a real hot
/// path to regress) against a budget it cannot possibly meet, and asserts
/// the gate reports it as a regression. Printed separately from -- and does
/// not affect the exit code of -- the real hot-path scenarios: this is
/// infrastructure self-test, not a claim about `LatentDb`'s performance.
pub fn self_check_regression_gate_fires() {
    const BUDGET_US: f64 = 200.0; // far below a 2ms sleep, on any machine.
    let us = bench_us(0, 1, || std::thread::sleep(Duration::from_millis(2)));
    let row = Row {
        name: "self-check (2ms sleep)",
        us,
        budget_us: BUDGET_US,
    };
    assert!(
        !row.pass(),
        "harness self-check FAILED: a 2ms sleep measured {us} us, under its {BUDGET_US} us \
         budget -- the regression gate did not fire for a scenario that is deliberately, \
         unmissably too slow. The timer or the PASS/REGRESSION comparator is broken."
    );
    println!(
        "harness self-check: regression gate correctly reported REGRESSION for a synthetic \
         {} sleep vs {} budget (proves the comparator fires; does not affect this binary's \
         exit code)\n",
        fmt_us(us),
        fmt_us(BUDGET_US)
    );
}
