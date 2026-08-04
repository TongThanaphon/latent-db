//! SIMD dispatch for PQ asymmetric distance computation (Issue #2).
//!
//! Two kernels, each dispatched over AVX2+FMA (x86_64, runtime `cpuid`
//! detection), NEON (aarch64, mandatory on ARMv8+), WASM SIMD128 (`wasm32`,
//! compile-time `target_feature = "simd128"` gate), and a scalar fallback
//! everywhere else:
//!
//! - [`simd_dot_f32`] -- plain dot product, used to build the per-subspace
//!   query-to-centroid lookup table (see `pq::PqCodec::build_query_lut`) and
//!   to compute a query's squared norm.
//! - [`simd_lut_row_sum_f32`] -- gather-sum over a flat LUT (`Σ_i
//!   table[row_base[i] + codes[i]]`), used by `pq::QueryLut::score_parts` to
//!   score a candidate's PQ codes against the LUT without ever decoding them
//!   back to a full-precision vector.
//!
//! This is the same stable-Rust, no-nightly `core::arch` dispatch shape
//! katgpt-rs's own SIMD kernels use (`katgpt-types::simd`) -- runtime `cpuid`
//! caching on x86_64, mandatory NEON on aarch64, compile-time-gated WASM
//! SIMD128, and an always-correct scalar fallback -- reimplemented
//! independently here (this crate has no dependency on katgpt-rs) so later
//! math tickets in this batch can share this module rather than each
//! duplicating their own dispatch.

/// SIMD-accelerated dot product: `Σ a[i] * b[i]`. `a` and `b` must be the
/// same length; returns `0.0` for empty inputs.
///
/// `len` is `a.len().min(b.len())`, not `a.len()` outright: every backend
/// below does raw pointer arithmetic up to `len` with no further bounds
/// check (the `debug_assert_eq!` disappears in release builds), so a
/// mismatched pair must shrink `len` rather than trust the longer slice --
/// otherwise a release-mode caller bug becomes an out-of-bounds read instead
/// of a panic.
#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot product operands must match in length"
    );
    let len = a.len().min(b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f32(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_dot_f32(a, b, len) }
        } else {
            scalar_dot_f32(a, b, len)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_dot_f32(a, b, len) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_dot_f32(a, b, len)
    }
}

/// SIMD-accelerated LUT gather-sum: `Σ_i table[row_base[i] + codes[i] as u32]`.
///
/// `row_base` and `codes` must be the same length (one entry per PQ
/// subspace); `row_base[i]` is that subspace's flat offset into `table`
/// (`subspace_index * n_centroids`) and `codes[i]` is the centroid index
/// within that subspace. Zero heap allocations -- the whole point of this
/// kernel is scoring a candidate's PQ codes against a precomputed LUT
/// without ever materializing a decoded vector.
#[inline(always)]
pub fn simd_lut_row_sum_f32(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
    debug_assert_eq!(
        row_base.len(),
        codes.len(),
        "row_base/codes length mismatch"
    );
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_lut_row_sum_f32(table, row_base, codes) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_lut_row_sum_f32(table, row_base, codes) }
        } else {
            scalar_lut_row_sum_f32(table, row_base, codes)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_lut_row_sum_f32(table, row_base, codes) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_lut_row_sum_f32(table, row_base, codes)
    }
}

// ── Scalar fallback ──────────────────────────────────────────────────────

#[inline(always)]
#[allow(dead_code)]
fn scalar_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    // 4 independent accumulators keep the FMA pipeline full on targets
    // without hardware f32 SIMD (matches katgpt-rs's `scalar_dot_f32`
    // convention). `mul_add` preserves single-rounding FMA semantics on
    // hardware that has it, numerically matching the SIMD paths below.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        acc[0] = a[i].mul_add(b[i], acc[0]);
        acc[1] = a[i + 1].mul_add(b[i + 1], acc[1]);
        acc[2] = a[i + 2].mul_add(b[i + 2], acc[2]);
        acc[3] = a[i + 3].mul_add(b[i + 3], acc[3]);
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_lut_row_sum_f32(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
    let n = codes.len();
    let mut acc = [0.0f32; 4];
    let chunks = n / 4;
    let mut i = 0;
    for _ in 0..chunks {
        acc[0] += table[(row_base[i] + codes[i] as u32) as usize];
        acc[1] += table[(row_base[i + 1] + codes[i + 1] as u32) as usize];
        acc[2] += table[(row_base[i + 2] + codes[i + 2] as u32) as usize];
        acc[3] += table[(row_base[i + 3] + codes[i + 3] as u32) as usize];
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < n {
        sum += table[(row_base[i] + codes[i] as u32) as usize];
        i += 1;
    }
    sum
}

// ── x86_64 runtime detection ─────────────────────────────────────────────

/// Detect AVX2+FMA support on x86_64, cached after the first call. Mirrors
/// katgpt-rs's `is_avx2_fma_available` (`__cpuid`-based, `AtomicBool` +
/// `Once` cache).
#[cfg(target_arch = "x86_64")]
fn is_avx2_fma_available() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Once;
        static CACHED: AtomicBool = AtomicBool::new(false);
        static INIT: Once = Once::new();
        // `__cpuid` became safe in Rust 1.93 (returns a Copy struct, no
        // memory dereference); the `unsafe` wrappers stay for older Rust,
        // and this block-level allow silences the resulting
        // unnecessary-unsafe-block lint under `-D warnings` on 1.93+
        // without affecting the inner statements. Matches katgpt-rs's own
        // `is_avx2_fma_available`.
        #[allow(unused_unsafe)]
        INIT.call_once(|| {
            let cpuid1 = unsafe { core::arch::x86_64::__cpuid(1) };
            let has_avx = (cpuid1.ecx & (1 << 28)) != 0;
            let has_fma = (cpuid1.ecx & (1 << 12)) != 0;
            let cpuid7 = unsafe { core::arch::x86_64::__cpuid(7) };
            let has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;
            CACHED.store(has_avx && has_fma && has_avx2, Ordering::Relaxed);
        });
        CACHED.load(Ordering::Relaxed)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn horizontal_sum_256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
        _mm_movehdup_ps, _mm_movehl_ps,
    };
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let total = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(total)
    }
}

// ── AVX2 backend (x86_64) ────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks8 = len / 8;
        for _ in 0..chunks8 {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        let mut sum = horizontal_sum_256(acc);
        while i < len {
            sum = a[i].mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_lut_row_sum_f32(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm256_add_epi32, _mm256_add_ps, _mm256_cvtepu8_epi32, _mm256_i32gather_ps,
        _mm256_loadu_si256, _mm256_setzero_ps, _mm_loadl_epi64,
    };

    unsafe {
        let n = codes.len();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks8 = n / 8;
        for _ in 0..chunks8 {
            let code_bytes = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
            let code_i32 = _mm256_cvtepu8_epi32(code_bytes);
            let base_i32 = _mm256_loadu_si256(row_base.as_ptr().add(i) as *const _);
            let idx = _mm256_add_epi32(base_i32, code_i32);
            let gathered = _mm256_i32gather_ps::<4>(table.as_ptr(), idx);
            acc = _mm256_add_ps(acc, gathered);
            i += 8;
        }
        let mut sum = horizontal_sum_256(acc);
        while i < n {
            sum += table[(row_base[i] + codes[i] as u32) as usize];
            i += 1;
        }
        sum
    }
}

// ── NEON backend (aarch64) ───────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks4 = len / 4;
        for _ in 0..chunks4 {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < len {
            sum = a[i].mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

/// Gathers from a LUT for four codes at once and vector-accumulates the
/// result. NEON has no gather instruction, so the index lookup itself is
/// scalar -- same honest tradeoff as katgpt-rs's `dequant_via_lut_neon`: the
/// win comes from the precomputed LUT (no per-candidate PQ decode), not from
/// a vectorized gather that doesn't exist on this ISA.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_lut_row_sum_f32(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};

    unsafe {
        let n = codes.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks4 = n / 4;
        for _ in 0..chunks4 {
            let gathered = [
                table[(row_base[i] + codes[i] as u32) as usize],
                table[(row_base[i + 1] + codes[i + 1] as u32) as usize],
                table[(row_base[i + 2] + codes[i + 2] as u32) as usize],
                table[(row_base[i + 3] + codes[i + 3] as u32) as usize],
            ];
            let v = vld1q_f32(gathered.as_ptr());
            acc = vaddq_f32(acc, v);
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < n {
            sum += table[(row_base[i] + codes[i] as u32) as usize];
            i += 1;
        }
        sum
    }
}

// ── WASM SIMD128 backend ─────────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};

    unsafe {
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        let chunks4 = len / 4;
        for _ in 0..chunks4 {
            let va = v128_load(a.as_ptr().add(i).cast());
            let vb = v128_load(b.as_ptr().add(i).cast());
            acc = f32x4_add(acc, f32x4_mul(va, vb));
            i += 4;
        }
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while i < len {
            sum = a[i].mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_lut_row_sum_f32(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_splat, v128_load};

    unsafe {
        let n = codes.len();
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        let chunks4 = n / 4;
        for _ in 0..chunks4 {
            let gathered = [
                table[(row_base[i] + codes[i] as u32) as usize],
                table[(row_base[i + 1] + codes[i + 1] as u32) as usize],
                table[(row_base[i + 2] + codes[i + 2] as u32) as usize],
                table[(row_base[i + 3] + codes[i + 3] as u32) as usize],
            ];
            let v = v128_load(gathered.as_ptr().cast());
            acc = f32x4_add(acc, v);
            i += 4;
        }
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while i < n {
            sum += table[(row_base[i] + codes[i] as u32) as usize];
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn reference_lut_row_sum(table: &[f32], row_base: &[u32], codes: &[u8]) -> f32 {
        row_base
            .iter()
            .zip(codes.iter())
            .map(|(&rb, &c)| table[(rb + c as u32) as usize])
            .sum()
    }

    /// Runs `f` (either `simd_dot_f32` or the scalar fallback) against
    /// [`reference_dot`] across a length sweep that covers the empty case,
    /// sub-chunk tails, and multiple full SIMD-width chunks.
    fn check_dot(f: impl Fn(&[f32], &[f32]) -> f32) {
        for len in [0usize, 1, 3, 4, 5, 8, 15, 16, 17, 64, 130] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.7).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 1.3).cos()).collect();
            let got = f(&a, &b);
            let want = reference_dot(&a, &b);
            assert!(
                (got - want).abs() < 1e-3,
                "len={len}: got {got}, want {want}"
            );
        }
    }

    /// Runs `f` (either `simd_lut_row_sum_f32` or the scalar fallback)
    /// against [`reference_lut_row_sum`] across an `n_subspaces` sweep that
    /// covers the empty case, sub-chunk tails, and multiple full chunks.
    fn check_lut_row_sum(f: impl Fn(&[f32], &[u32], &[u8]) -> f32) {
        let n_centroids = 16u32;
        let max_n_subspaces = 16usize;
        let table: Vec<f32> = (0..(n_centroids as usize * max_n_subspaces))
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        for n_subspaces in [0usize, 1, 3, 4, 5, 8, 9, 16] {
            let row_base: Vec<u32> = (0..n_subspaces as u32).map(|s| s * n_centroids).collect();
            let codes: Vec<u8> = (0..n_subspaces).map(|i| (i * 3 % 16) as u8).collect();
            let got = f(&table, &row_base, &codes);
            let want = reference_lut_row_sum(&table, &row_base, &codes);
            assert!(
                (got - want).abs() < 1e-3,
                "n_subspaces={n_subspaces}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn dot_matches_reference_for_various_lengths() {
        check_dot(simd_dot_f32);
    }

    #[test]
    fn lut_row_sum_matches_reference_for_various_lengths() {
        check_lut_row_sum(simd_lut_row_sum_f32);
    }

    // The dispatcher only ever exercises one backend per target (NEON on
    // this machine's aarch64, scalar on a non-SIMD128 wasm32 build, etc.),
    // so the scalar fallback -- the actual live path on wasm32 without
    // `+simd128` and on x86_64 without AVX2 -- gets zero coverage from the
    // tests above on any single CI runner. Call it directly.

    #[test]
    fn scalar_dot_matches_reference_for_various_lengths() {
        check_dot(|a, b| scalar_dot_f32(a, b, a.len()));
    }

    #[test]
    fn scalar_lut_row_sum_matches_reference_for_various_lengths() {
        check_lut_row_sum(scalar_lut_row_sum_f32);
    }

    #[test]
    fn dot_dispatch_matches_scalar_fallback() {
        for len in [0usize, 1, 4, 17, 130] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.53).cos()).collect();
            let dispatched = simd_dot_f32(&a, &b);
            let scalar = scalar_dot_f32(&a, &b, len);
            assert!(
                (dispatched - scalar).abs() < 1e-3,
                "len={len}: dispatched={dispatched}, scalar={scalar}"
            );
        }
    }

    #[test]
    fn lut_row_sum_dispatch_matches_scalar_fallback() {
        let n_centroids = 16u32;
        let table: Vec<f32> = (0..(n_centroids as usize * 9))
            .map(|i| (i as f32 * 0.17).cos())
            .collect();
        for n_subspaces in [0usize, 1, 4, 9] {
            let row_base: Vec<u32> = (0..n_subspaces as u32).map(|s| s * n_centroids).collect();
            let codes: Vec<u8> = (0..n_subspaces).map(|i| (i * 5 % 16) as u8).collect();
            let dispatched = simd_lut_row_sum_f32(&table, &row_base, &codes);
            let scalar = scalar_lut_row_sum_f32(&table, &row_base, &codes);
            assert!(
                (dispatched - scalar).abs() < 1e-3,
                "n_subspaces={n_subspaces}: dispatched={dispatched}, scalar={scalar}"
            );
        }
    }
}
