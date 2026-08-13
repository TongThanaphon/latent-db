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

## Issue #11 (SIMD vector math primitives) -- before/after

Same host/toolchain as above (Apple M2, NEON backend). Before = this file's
scalar baseline above (`viable_graph_bench`) and a fresh scalar run recorded
for the newly-added `projector_bench` just before the SIMD change landed;
after = the same benches run once `Projector::project`, `manifold::euclidean`,
and `superpose::cosine_sim`/`circular_convolve`/`circular_correlate` all route
through `crate::simd`'s dispatched kernels (reusing Issue #2's dispatch
scaffold, plus one new kernel, `simd_squared_euclidean_f32`, added for
`euclidean`'s subtract-then-square accumulation shape, which
`simd_dot_f32`/`simd_lut_row_sum_f32` don't cover).

### `projector_bench` (new bench, n=500 vectors per row)

| scenario | before (scalar) | after (SIMD) | speedup |
|---|---|---|---|
| `project(64->16, n=500)` | ~296 us | ~46-57 us | ~5.2-6.5x |
| `project(512->64, n=500)` | ~10.7 ms | ~2.83-2.91 ms | ~3.7-3.8x |

### `viable_graph_bench` (dim=16, k_nearest=6)

| scenario | before (scalar) | after (SIMD) | speedup |
|---|---|---|---|
| `build_viable_graph(n=300)` | ~1.2-1.9 ms | ~1.0-1.7 ms | modest, noisy at this n |
| `build_viable_graph(n=1200)` | ~19.3-19.8 ms | ~18.2-18.3 ms | ~1.08x, consistent across 3 runs |
| `geodesic(n=1200)` | ~19-27 us | ~19-25 us | within run-to-run noise |
| `random_walk(n=1200, 20 steps)` | ~0.6 us | ~0.6 us | unaffected (no distance calls) |

The end-to-end `build_viable_graph(n=1200)` win is real but modest (~8%),
not the multi-x jump `projector_bench` shows, because `build_viable_graph`'s
per-node `sort_unstable_by` over ~1199 candidates and its other O(n^2)
bookkeeping (adjacency lists, `id_to_node` map) account for a comparable
share of that bench's total time at `dim=16` -- the O(n^2) `euclidean` calls
are only part of the picture there.

### `euclidean_kernel_bench` (new bench, dim=16, isolates just the O(n^2)
distance-pass kernel `manifold::euclidean` calls, decoupled from
`build_viable_graph`'s sort/bookkeeping)

`simd_squared_euclidean_f32` is public (`crate::simd`), but the pre-SIMD
scalar loop it replaced in `manifold::euclidean` is gone from the tree, so
this bench -- added specifically so the kernel-level claim below is
reproducible from a committed file rather than one-off prose -- can only
report the current (post-SIMD) number, not a re-runnable scalar comparison:

| scenario (n * (n-1) calls) | measured (after SIMD) |
|---|---|
| `all_pairs_euclidean(n=300)` | ~180-440 us |
| `all_pairs_euclidean(n=1200)` | ~2.9-3.5 ms |

For context: an ad hoc (uncommitted, not reproducible) scalar-vs-SIMD
comparison run once during this ticket's development, same dim=16 and the
same ~1.44M-call order as `n=1200` above, measured the scalar loop at
~17.9 ms and the SIMD kernel at ~6.6 ms for that call count -- a ~2.7x
kernel-level speedup, consistent with `all_pairs_euclidean(n=1200)`'s
measured range above being well under half of that scalar figure. This
confirms the kernel itself is genuinely multi-x faster; `build_viable_graph`'s
diluted ~8% end-to-end number reflects the surrounding O(n log n) sort cost,
not a weak kernel.

## Issue #21 (`latent-db-embedded` storage: `ZeroAllocLatent` + unrolled dot/cosine)

Same host as above (Apple M2, NEON `hw_sqrt_aarch64` path). New crate,
new bench (`latent-db-embedded/benches/storage_bench.rs`): plain 8-way
loop-unrolled scalar Rust (no `core::arch` SIMD intrinsics -- `DIM = 768` is
a compile-time constant LLVM auto-vectorizes on its own), so these numbers
aren't directly comparable to `latent_db::simd`'s hand-dispatched AVX2/NEON
kernels above -- they're a baseline for *this* crate's own future
optimization tickets to measure against.

**Run:** `cargo bench -p latent-db-embedded --bench storage_bench`

### `storage_bench` (dim=768, 10,000 calls per measured scenario)

| scenario | measured | budget |
|---|---|---|
| `dot_product` | ~0.66-1.4 ms | 6.0 ms |
| `cosine_similarity` | ~1.8-2.2 ms | 10.0 ms |

Also asserts `dot_product`/`cosine_similarity` at 2,500 calls never measure
slower than at 10,000 calls (this bench's `assert_monotonic` harness
self-check, in place of `support::self_check_regression_gate_fires()` --
see this file's header note above for why `storage_bench.rs` doesn't import
`benches/support/mod.rs` directly).

## Issues #22-#24 (`latent-db-embedded` topology + steering + zero-alloc query loop)

Same host as above (Apple M2). #22 (`topology::{ViableNode, is_viable}`) and
#23 (`steering::{steer_next, EdgeWeights}`) are unit-tested only, no bench of
their own -- #24 is the one with a runtime-sensitive acceptance criterion
(see below).

**Run:** `cargo bench -p latent-db-embedded --bench query_loop_bench`

### `query_loop_bench` (dim=768, 10,000 calls, release profile)

Reports **per-iteration** latency directly (`total / calls`), not a raw
batch total like `storage_bench` above -- #24's acceptance criteria asks
specifically for per-iteration latency. Each call uses a periodic-probe
alpha schedule (a small in-bounds step every iteration, plus a deliberately
oversized one every 1,000th) so the measured cost reflects both of
`steer_next`'s branches (commit and boundary-reject/revert), not just its
always-succeeds path.

| scenario | measured | budget |
|---|---|---|
| `query_step` (per-iter) | ~0.63-0.64 us | 3.0 us |

Also asserts 2,500 calls never measure slower than 10,000 (same
`assert_monotonic` self-check as `storage_bench`).

### `tests/query_loop_alloc_check.rs`: 1,000,000-iteration debug-mode run

This is `cargo test`'s (unoptimized, `-O0`) cost, not `cargo bench`'s
(optimized) -- the two numbers above and below aren't comparable to each
other. Measured **~7.2s** for the full 1,000,000-iteration loop under plain
`cargo test`, asserting `get_alloc_stats()` reports `(0, 0)` allocations
across the whole run -- exactly 999,000 committed and 1,000 rejected steps
per the same probe schedule (verified by an exact-count assertion in the
test itself), and a separate positive-control test proving the tracking
allocator is actually reachable from this crate's test binary before that
`(0, 0)` is trusted. That runtime is a deliberate acceptance-criteria
tradeoff (#24 asks for exactly 1,000,000 iterations, gated the same way as
this repo's other `*_alloc_check` tests -- i.e. debug-mode, not
release-only), not an oversight; see the test file's own header comment for
the full reasoning, including the `steer_next` loop-unrolling change (~2x
runtime cut) this sizing spike motivated.
