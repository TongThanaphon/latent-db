# Bench baseline (Issue 1)

Reference numbers for today's implementation -- scalar cosine similarity,
`HashMap`-backed record storage, full-rebuild Merkle tree on every
`merkle_proof` call, O(n^2) pairwise-distance `build_viable_graph`. Every
later optimization ticket in this batch (PQ asymmetric distance, SIMD vector
math, cached-levels Merkle tree, ...) should be measured against these
numbers, not against a fresh from-scratch run.

**Date:** 2026-08-04
**Host:** Apple M2, macOS (Darwin 25.5.0, arm64), unloaded
**Toolchain:** `rustc 1.95.0 (59807616e 2026-04-14)`, stable, no criterion
dev-dependency (hand-rolled `std::time::Instant`, `harness = false`)
**Run:** `cargo bench` (each number below is best-of-N after warmup; see
`benches/support/mod.rs::bench_us` and each bench file's warmup/iters counts)

Each bench also runs `support::self_check_regression_gate_fires()` (or an
equivalent `assert_monotonic` between two differently-sized scenarios),
proving the PASS/REGRESSION comparator actually fires on a deliberately-too-
slow synthetic operation and not just that everything happens to pass.

Run-to-run variance on a laptop is real (±30% is unremarkable) -- treat these
as order-of-magnitude reference points, not exact SLAs. The `budget_us` gate
baked into each bench (visible in its PASS/REGRESSION table) is set several
times above the measured baseline for this reason; it exists to catch a
genuine multi-x regression, not to assert an exact number.

## `search_bench` (N=2000 records, dim=64, k=10)

| scenario | measured |
|---|---|
| `search(nprobe=1)` | ~11-16 us |
| `search(nprobe=all)` (32 buckets) | ~300-385 us |
| `search_steered(nprobe=all)` | ~300-330 us |

## `merkle_proof_bench` (dim=32, full-rebuild-per-call baseline)

| scenario | measured |
|---|---|
| `merkle_proof(n=200)` | ~57-118 us |
| `merkle_proof(n=2000)` | ~550-722 us |

10x more records costs roughly 6-10x more time, consistent with the O(n)
`merkle_leaves()` rehash + `MerkleTree::build()` on every call (see
`db.rs::merkle_tree` doc comment) -- the thing Issue 4 (cached-levels Merkle
tree) is meant to fix.

## `viable_graph_bench` (dim=16, k_nearest=6)

| scenario | measured |
|---|---|
| `build_viable_graph(n=300)` | ~1.2-1.9 ms |
| `build_viable_graph(n=1200)` | ~19.3-19.7 ms |
| `geodesic(n=1200)` | ~21-27 us |
| `random_walk(n=1200, 20 steps)` | ~0.6 us |

4x more records costs ~10-16x more build time, consistent with the O(n^2)
pairwise-distance kNN build (`manifold.rs` module docs).
