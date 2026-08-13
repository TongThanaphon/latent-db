# ADR-0001: `latent-db-embedded` as a coexisting crate, not a `no_std` conversion

## Status

Accepted (Issue #20).

## Context

`latent-db-embedded` needs to host fixed-size, zero-allocation vector/graph
primitives for embedded targets: a `#[repr(C)] ZeroAllocLatent` (#21),
`ViableNode` + boundary checks (#22), in-place steering (#23), and a
1,000,000-iteration zero-alloc query loop (#24). Three decisions had to be
made before any of that code could be written.

## Decision 1: separate crate, not a `no_std` conversion of `latent-db`

`latent-db` (the root package) is built around `HashMap`-based indexing,
`String` metadata, `Vec`-backed dynamic-dimension vectors, `serde`,
`wasm-bindgen`, and an `axum`/`tokio` CLI server — every one of those is
either unavailable under `no_std` or fundamentally heap-shaped. Converting it
in place would mean ripping out and re-justifying its entire std-dependent
surface for the benefit of a use case (bare-metal, fixed `DIM = 768`,
zero-allocation) that most of its existing callers don't have and don't want.

Instead, `latent-db-embedded` is a new workspace member (root `Cargo.toml`
gained a `[workspace]` section; `latent-db` is unchanged apart from that) with
its own `Cargo.toml`, zero `[dependencies]`, and a fixed, compile-time
`DIM`/`MAX_NEIGHBORS` rather than `latent-db`'s runtime-dynamic dimension. It
shares no code with `latent-db` today, and doesn't depend on it even as a dev
dependency yet — #21, when it adds tests that cross-check numeric output
against `latent_db::simd::simd_dot_f32`, is the ticket that should add that
`[dev-dependencies]` entry (dev-dependencies don't link into the built
`no_std` artifact, only into `cargo test`/bench binaries, so that won't
compromise the zero-dependency build). Adding it here, before any test that
needs it exists, would just be an unused dependency pulling in `latent-db`'s
own dependency tree for nothing.

## Decision 2: `sqrt` on stable `no_std`, without `libm`

`cosine_similarity` (#21) needs `sqrt`. `f32::sqrt` is a `std` method — it is
**not** available on `core::f32` (verified directly on this toolchain,
`rustc 1.95.0`: `x.sqrt()` in a `#![no_std]` crate with no dependencies fails
with `E0599: no method named sqrt found for type f32`). Three ways to get it
back were considered:

1. **The `libm` crate.** Pure Rust, `no_std`, no transitive deps of its own —
   but it's still an entry under `[dependencies]`, which fails #20's own
   acceptance criterion (zero dependencies) and every downstream ticket that
   inherits it.
2. **`core::intrinsics::sqrtf32`.** Dependency-free and in `core`, but gated
   behind `#![feature(core_intrinsics)]` — nightly-only. Rejected; this crate
   targets stable.
3. **Hardware sqrt via `core::arch`, with a software fallback.** `core::arch`
   is stable and lives in `core` — no `std`, no dependency. This dev
   machine's host target is `aarch64-apple-darwin`, so verifying the x86_64
   path required cross-compiling: with `rustup target add x86_64-apple-darwin`,
   `core::arch::x86_64::_mm_sqrt_ss` (used unconditionally, not behind a
   `cfg(target_arch = "x86_64")` gate that would silently elide it on this
   host) compiles and links cleanly under `#![no_std]` with an empty
   `[dependencies]` table when built with `--target x86_64-apple-darwin`.
   The NEON path was checked natively: `core::arch::aarch64::vsqrtq_f32`
   compiles cleanly under the same conditions on this host directly, no
   cross-compilation needed.

**Decision:** mirror `src/simd.rs`'s existing per-architecture dispatch shape
(x86_64 via `core::arch::x86_64`, aarch64 via `core::arch::aarch64`, both
stable and dependency-free) for the hardware path, and fall back to a
software approximation — bit-hack seed (fast inverse square root) plus
Newton-Raphson refinement iterations, pure integer/float arithmetic, no
dependency — on any target without a covered intrinsic (e.g. plain `wasm32`
without `simd128`, or other architectures).

**Tradeoff:** the hardware path is bit-accurate and fast, identical in
behavior to what `latent-db`'s own `std`-backed `.sqrt()` produces. The
software fallback is approximate, and converges more slowly than "a couple
iterations" might suggest: the classic bit-hack seed starts around 3–4%
relative error; each Newton-Raphson iteration roughly squares the error, so
one iteration lands near `1.7e-3` relative error and two near `1e-5`–`5e-6` —
still well short of `f32`'s own `~1.2e-7` relative ULP, which needs a third
iteration to approach. #21's tolerance-matching tests against
`latent_db::simd::simd_dot_f32`-derived cosine values need to pick an
iteration count (and tolerance) with this convergence rate in mind on
fallback-only targets, not assume near-ULP accuracy from one or two rounds.
Also worth flagging for #21: computing `sqrt(x)` as `x * rsqrt(x)` is
undefined at `x == 0` (the seed step divides by zero), a real input for
`cosine_similarity` whenever either operand is the zero vector — the
fallback path needs an explicit zero-guard, not just accuracy tuning.

## Decision 3: `#[repr(align(4))]` wrapper-const for `include_bytes!`

`include_bytes!("...")` yields `&'static [u8; N]` with no alignment
guarantee beyond 1 — the bytes are placed wherever the linker's data section
happens to put them. Reinterpreting that buffer as `&[f32]` via a raw pointer
cast requires 4-byte alignment (`f32`'s natural alignment); doing so without
verifying it first is UB the moment the buffer isn't actually aligned, which
`include_bytes!` never promises.

**Idiom** (implemented in #21, recorded here since it's a scaffold-level
constraint on the storage module's public shape): wrap the included bytes in
a `#[repr(align(4))]` newtype —

```rust
#[repr(align(4))]
struct Aligned<const N: usize>([u8; N]);

static VECTOR_BYTES: Aligned<{ DIM * 4 }> = Aligned(*include_bytes!("../data/example.bin"));
```

— so the *static's* address, not the raw `include_bytes!` output, carries
the alignment guarantee the compiler actually enforces. The zero-copy
`&[u8] -> &[f32]` conversion function still runtime-checks alignment (per
#21's acceptance criteria: reject misaligned input rather than trust the
caller) — the wrapper makes that check pass for `include_bytes!`-sourced
data instead of silently falling through to a copying fallback or, worse,
skipping the check.
