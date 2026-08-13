//! Fixed-size, zero-allocation vector storage (Issue #21): `ZeroAllocLatent`,
//! loop-unrolled `dot_product`/`cosine_similarity` sized for LLVM
//! auto-vectorization (no SIMD intrinsics crate), and the zero-copy
//! `&[u8] -> &[f32; DIM]` view + `include_bytes!` example described in
//! `docs/adr/0001-latent-db-embedded-coexistence.md`.

use crate::DIM;

/// Number of bytes a single `DIM`-length `f32` vector occupies.
pub const VECTOR_BYTES: usize = DIM * 4;

/// A single embedding vector plus its id. Entirely stack/static-resident --
/// no heap-backed field -- and `#[repr(C)]` so its layout is stable for
/// zero-copy reinterpretation from raw bytes (e.g. via [`view_bytes_as_vector`]
/// plus a trailing id, or `include_bytes!`-sourced tables laid out by an
/// external tool). Deliberately not `Copy`: at `VECTOR_BYTES + 8` (3080)
/// bytes, an implicit by-value copy is exactly the kind of hidden cost this
/// crate exists to make callers spell out (`.clone()`) rather than trigger
/// silently.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct ZeroAllocLatent {
    pub data: [f32; DIM],
    pub id: u64,
}

impl ZeroAllocLatent {
    /// Builds a latent from its vector and id, no allocation.
    pub const fn new(data: [f32; DIM], id: u64) -> Self {
        Self { data, id }
    }
}

/// Dot product `Σ a[i] * b[i]`, 8-way loop-unrolled with independent
/// accumulators so LLVM can auto-vectorize it at `-O` without any
/// `core::arch` SIMD intrinsics -- this module stays free of the runtime
/// `cpuid`/arch dispatch `latent_db::simd` uses, since `DIM` is a
/// compile-time constant LLVM can already unroll and vectorize on its own.
#[inline(always)]
pub fn dot_product(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = DIM / 8;
    let mut i = 0;
    for _ in 0..chunks {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        acc[4] += a[i + 4] * b[i + 4];
        acc[5] += a[i + 5] * b[i + 5];
        acc[6] += a[i + 6] * b[i + 6];
        acc[7] += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut sum = acc.iter().sum::<f32>();
    // Dead when `DIM % 8 == 0` (true for `DIM = 768` today), kept so this
    // function stays correct if `DIM` ever changes.
    while i < DIM {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Cosine similarity in `[-1.0, 1.0]`; `0.0` if either operand is the zero
/// vector (matches `latent_db::cosine_sim`'s zero-guard threshold).
#[inline(always)]
pub fn cosine_similarity(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let dot = dot_product(a, b);
    let denom = sqrt_f32(dot_product(a, a)) * sqrt_f32(dot_product(b, b));
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

// ── sqrt: hardware intrinsic where available, software fallback elsewhere ──
// (ADR-0001 decision 2 -- no `libm`, no nightly `core::intrinsics`.)

#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        hw_sqrt_x86_64(x)
    }
    #[cfg(target_arch = "aarch64")]
    {
        hw_sqrt_aarch64(x)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        software_sqrt_f32(x)
    }
}

/// SSE2 `sqrtss` -- baseline-guaranteed on every x86_64 target (part of the
/// ABI), so unlike this crate's would-be AVX2 paths this needs no runtime
/// `cpuid` feature check.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn hw_sqrt_x86_64(x: f32) -> f32 {
    use core::arch::x86_64::{_mm_cvtss_f32, _mm_set_ss, _mm_sqrt_ss};
    // Safety: `_mm_set_ss`/`_mm_sqrt_ss`/`_mm_cvtss_f32` operate on a
    // 128-bit register value, not memory -- no pointer, no alignment or
    // bounds requirement. SSE2 is mandatory on x86_64.
    unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

/// NEON `fsqrt` -- mandatory on ARMv8-A (baseline aarch64), matching
/// `latent_db::simd`'s own "NEON always available on aarch64" dispatch
/// shape.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn hw_sqrt_aarch64(x: f32) -> f32 {
    use core::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32, vsqrtq_f32};
    // Safety: operates entirely on vector-register values (a broadcast of
    // `x`, then lane 0 of the sqrt'd result) -- no pointer, no memory
    // access. NEON is mandatory on aarch64.
    unsafe { vgetq_lane_f32(vsqrtq_f32(vdupq_n_f32(x)), 0) }
}

/// Fast inverse square root (bit-hack seed + Newton-Raphson refinement),
/// for targets with no covered hardware sqrt intrinsic above (e.g. plain
/// `wasm32` without `simd128`). Not bit-accurate like the hardware paths --
/// see ADR-0001 decision 2 for the convergence analysis this iteration
/// count is picked from.
#[inline(always)]
#[allow(dead_code)]
fn software_sqrt_f32(x: f32) -> f32 {
    // `1 / sqrt(x)`'s bit-hack seed divides conceptually by `x` at `x == 0`;
    // `cosine_similarity`'s own `denom < 1e-9` guard never reaches this for
    // its call sites, but this function is `pub(crate)`-callable in
    // isolation (and directly unit-tested below), so it guards itself too.
    if x <= 0.0 {
        return 0.0;
    }
    let half_x = 0.5 * x;
    let mut y = f32::from_bits(0x5f37_59df - (x.to_bits() >> 1));
    // 3 refinement iterations: the bit-hack seed starts ~3-4% relative
    // error; each iteration roughly squares it (iter 1 ~1.7e-3, iter 2
    // ~1e-5-5e-6, iter 3 approaches f32's ~1.2e-7 ULP without quite
    // reaching it) -- see ADR-0001 decision 2.
    for _ in 0..3 {
        y *= 1.5 - half_x * y * y;
    }
    x * y
}

// ── Zero-copy `&[u8] -> &[f32; DIM]` view (ADR-0001 decision 3) ───────────

/// Reinterprets `bytes` as a `&[f32; DIM]` with no copy, provided `bytes` is
/// exactly [`VECTOR_BYTES`] long and 4-byte aligned. Returns `None` instead
/// of triggering UB when either precondition fails -- callers that only
/// have `include_bytes!`-sourced data (no alignment guarantee beyond 1)
/// must wrap it in an [`Aligned`] newtype first, as [`example_vector`] does.
pub fn view_bytes_as_vector(bytes: &[u8]) -> Option<&[f32; DIM]> {
    if bytes.len() != VECTOR_BYTES {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
        return None;
    }
    // Safety: `bytes` is exactly `DIM * 4` bytes (checked above) and its
    // address is a multiple of `align_of::<f32>()` (checked above), so a
    // `DIM`-length `f32` read starting at that address is in-bounds and
    // aligned. Every `u8` bit pattern is a valid `f32` bit pattern (NaN and
    // subnormal payloads included), so there is no value-validity UB
    // either.
    let floats: &[f32] = unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f32, DIM) };
    floats.try_into().ok()
}

/// Forces 4-byte (`f32`) alignment onto a wrapped byte array. `include_bytes!`
/// only guarantees 1-byte alignment for its output on its own; wrapping it
/// in a `static` of this type puts the *static's* alignment -- which the
/// compiler does enforce -- under the bytes instead.
#[repr(align(4))]
struct Aligned<const N: usize>([u8; N]);

static EXAMPLE_VECTOR_BYTES: Aligned<VECTOR_BYTES> =
    Aligned(*include_bytes!("../data/example.bin"));

/// A worked `include_bytes!` example: a `DIM`-length vector baked into the
/// compiled artifact at compile time and read back with zero copies via
/// [`view_bytes_as_vector`], proving the [`Aligned`] wrapper idiom actually
/// makes that check pass for `include_bytes!`-sourced data.
pub fn example_vector() -> &'static [f32; DIM] {
    view_bytes_as_vector(&EXAMPLE_VECTOR_BYTES.0)
        .expect("data/example.bin is VECTOR_BYTES long; Aligned<N> guarantees 4-byte alignment")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vector(seed: f32) -> [f32; DIM] {
        let mut v = [0.0f32; DIM];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = ((i as f32 + seed) * 0.037).sin();
        }
        v
    }

    fn reference_dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_product_matches_reference() {
        let a = sample_vector(0.0);
        let b = sample_vector(1.0);
        let got = dot_product(&a, &b);
        let want = reference_dot(&a, &b);
        assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
    }

    #[test]
    fn dot_product_matches_latent_db_simd_dot() {
        let a = sample_vector(0.0);
        let b = sample_vector(1.0);
        let got = dot_product(&a, &b);
        let want = latent_db::simd::simd_dot_f32(&a, &b);
        assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
    }

    #[test]
    fn dot_product_self_is_nonnegative() {
        // cosine_similarity's sqrt_f32 calls rely on this holding for any
        // input, not just the sample vectors above.
        for seed in [0.0, 1.0, -3.5, 42.0] {
            let v = sample_vector(seed);
            assert!(dot_product(&v, &v) >= 0.0);
        }
    }

    // Issue #21 asks this to match "the existing squared-euclidean-derived
    // cosine path" -- no such function exists in `latent_db` (its one
    // squared-euclidean-derived function, `manifold::euclidean`, is a
    // distance, not a cosine). `superpose::cosine_sim` (re-exported as
    // `latent_db::cosine_sim`) is the crate's actual cosine similarity --
    // dot-product/norm-derived, same `simd_dot_f32` kernel this module's
    // own `dot_product` is cross-checked against above -- and the closest
    // real match to what the issue is asking to be numerically consistent
    // with.
    #[test]
    fn cosine_similarity_matches_latent_db_cosine_sim() {
        let a = sample_vector(0.0);
        let b = sample_vector(1.0);
        let got = cosine_similarity(&a, &b);
        let want = latent_db::cosine_sim(&a, &b);
        assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
    }

    #[test]
    fn cosine_similarity_self_is_one() {
        let a = sample_vector(0.0);
        let got = cosine_similarity(&a, &a);
        assert!((got - 1.0).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn cosine_similarity_zero_vector_is_zero() {
        let zero = [0.0f32; DIM];
        let a = sample_vector(0.0);
        assert_eq!(cosine_similarity(&zero, &a), 0.0);
        assert_eq!(cosine_similarity(&zero, &zero), 0.0);
    }

    // The dispatcher only ever exercises one sqrt backend per host (hardware
    // on this x86_64/aarch64 dev machine, software everywhere else), so the
    // software fallback -- the live path on e.g. plain wasm32 -- gets zero
    // coverage from the tests above on any single CI runner. Call it
    // directly, same convention `latent_db::simd`'s own tests use for its
    // scalar fallback.

    #[test]
    fn software_sqrt_matches_std_sqrt() {
        for x in [0.25f32, 1.0, 2.0, 9.0, 100.0, 0.001, 123456.0] {
            let got = software_sqrt_f32(x);
            let want = x.sqrt();
            let rel_err = (got - want).abs() / want;
            assert!(
                rel_err < 1e-4,
                "x={x}: got {got}, want {want}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn software_sqrt_zero_guard() {
        assert_eq!(software_sqrt_f32(0.0), 0.0);
        assert_eq!(software_sqrt_f32(-1.0), 0.0);
    }

    #[test]
    fn sqrt_dispatch_matches_std_sqrt() {
        for x in [0.25f32, 1.0, 2.0, 9.0, 100.0, 0.001, 123456.0] {
            let got = sqrt_f32(x);
            let want = x.sqrt();
            assert!((got - want).abs() < 1e-3, "x={x}: got {got}, want {want}");
        }
    }

    #[test]
    fn zero_alloc_latent_stores_fields() {
        let data = sample_vector(2.0);
        let latent = ZeroAllocLatent::new(data, 42);
        assert_eq!(latent.data, data);
        assert_eq!(latent.id, 42);
    }

    #[test]
    fn zero_alloc_latent_is_repr_c_no_heap() {
        // #[repr(C)] with only fixed-size fields -> size is exactly
        // `data` + `id` (both align to a multiple of 8 already, so no
        // padding either). An exact match, not just a lower bound -- a
        // smuggled-in heap-backed field (e.g. `Vec<f32>`, +24 bytes on this
        // target) would inflate this size and fail the assertion.
        assert_eq!(core::mem::size_of::<ZeroAllocLatent>(), VECTOR_BYTES + 8);
    }

    fn aligned_buf() -> Aligned<{ VECTOR_BYTES + 4 }> {
        let mut bytes = [0u8; VECTOR_BYTES + 4];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        Aligned(bytes)
    }

    #[test]
    fn view_bytes_as_vector_accepts_aligned_correctly_sized_input() {
        let buf = aligned_buf();
        let view = view_bytes_as_vector(&buf.0[..VECTOR_BYTES]);
        assert!(view.is_some());
    }

    #[test]
    fn view_bytes_as_vector_rejects_wrong_length() {
        let buf = aligned_buf();
        assert!(view_bytes_as_vector(&buf.0[..VECTOR_BYTES - 4]).is_none());
        assert!(view_bytes_as_vector(&buf.0[..]).is_none());
    }

    #[test]
    fn view_bytes_as_vector_rejects_misaligned_input() {
        let buf = aligned_buf();
        // `buf.0`'s address is 4-aligned (guaranteed by `Aligned`'s
        // `#[repr(align(4))]`); offsetting by 1 byte deterministically
        // breaks that alignment, unlike slicing a bare `[u8; N]` (align 1)
        // whose base address alignment isn't guaranteed either way.
        let misaligned = &buf.0[1..1 + VECTOR_BYTES];
        assert!(view_bytes_as_vector(misaligned).is_none());
    }

    #[test]
    fn example_vector_loads_via_zero_copy_view() {
        let v = example_vector();
        // Matches `data/example.bin`'s generator: `sin(i * 0.037)`.
        assert!((v[0] - 0.0f32).abs() < 1e-6);
        let expected_1 = (1.0f32 * 0.037).sin();
        assert!((v[1] - expected_1).abs() < 1e-6);
    }
}
