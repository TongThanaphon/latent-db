//! Latent superposition: pack several key/value pairs into a *single*
//! latent vector, and later pull an individual value back out given its key.
//!
//! This module's name is borrowed from katgpt-rs's "MUX-Latent" context
//! compression, but the algorithm here is *not* what katgpt-rs actually
//! does. katgpt-rs's real MUX-Latent (`crates/katgpt-core/src/mux_latent/`)
//! builds decay-weighted one-hot span weights (`Σ decay^j × onehot(t_j)`)
//! purely to shrink the decoder's attention footprint, but it also stores
//! the original tokens verbatim alongside those weights — so its `EXPAND(i)`
//! is a literal lookup, not a reconstruction, and the whole scheme is
//! lossless by construction.
//!
//! What's implemented below is a genuinely different, harder technique: the
//! classic Holographic Reduced Representation (HRR) trick (Plate, 1995),
//! which katgpt-rs does not implement at all:
//!
//!   bind(key, value)   = circular_convolve(key, value)
//!   bundle(a, b, ...)  = normalize(a + b + ...)
//!   unbind(bundle,key) = circular_correlate(key, bundle) ~= value
//!
//! Binding is approximately invertible and bundling is linear, so a whole
//! batch of (key, value) pairs can be superposed into one vector of the
//! *same* dimensionality as a single value, at the cost of retrieval noise
//! that grows with how many pairs are packed together. Unlike katgpt-rs's
//! MUX-Latent, there is no retained original value to fall back on — this
//! is inherently lossy, and that lossy tradeoff is *not* something
//! katgpt-rs's README or source validates or mirrors.

/// Circular convolution: binds `a` and `b` into one vector of the same size.
pub fn circular_convolve(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "bind operands must match in length");
    let n = a.len();
    let mut out = vec![0.0f32; n];
    for (i, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &aj) in a.iter().enumerate() {
            // (i - j) mod n
            let idx = (i + n - j) % n;
            sum += aj * b[idx];
        }
        *out_val = sum;
    }
    out
}

/// Circular correlation: approximately inverts a circular convolution.
/// `unbind(convolve(key, value), key) ~= value` (up to noise).
pub fn circular_correlate(key: &[f32], bundle: &[f32]) -> Vec<f32> {
    assert_eq!(
        key.len(),
        bundle.len(),
        "unbind operands must match in length"
    );
    let n = key.len();
    let mut out = vec![0.0f32; n];
    for (i, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &keyj) in key.iter().enumerate() {
            let idx = (i + j) % n;
            sum += keyj * bundle[idx];
        }
        *out_val = sum;
    }
    out
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = norm(&v);
    if n > 1e-9 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

/// A single superposed "slot" holding many (key, value) pairs compressed,
/// lossily, into one vector via HRR bind/bundle — named after but not
/// algorithmically equivalent to katgpt-rs's (lossless) MUX-Latent span
/// compression.
pub struct SuperposedSlot {
    dim: usize,
    bundle: Vec<f32>,
    /// Number of pairs packed in, tracked purely for diagnostics
    /// (accuracy degrades as this grows).
    pub count: usize,
}

impl SuperposedSlot {
    pub fn new(dim: usize) -> Self {
        SuperposedSlot {
            dim,
            bundle: vec![0.0; dim],
            count: 0,
        }
    }

    /// Pack one more (key, value) pair into this slot. Cheap: O(dim^2) per
    /// insert, O(dim) storage regardless of how many pairs are packed.
    pub fn insert(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.dim);
        assert_eq!(value.len(), self.dim);
        let bound = circular_convolve(key, value);
        for (b, bound_val) in self.bundle.iter_mut().zip(bound.iter()) {
            *b += bound_val;
        }
        self.count += 1;
    }

    /// Recover an approximation of the value stored under `key`.
    /// Retrieval quality degrades gracefully as more pairs share the slot.
    pub fn expand(&self, key: &[f32]) -> Vec<f32> {
        circular_correlate(key, &self.bundle)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Cosine similarity, used to score how well an `expand()` matches the
/// original value (1.0 = perfect, 0.0 = orthogonal/no signal left).
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let denom = norm(a) * norm(b);
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

#[allow(dead_code)]
pub fn unit(v: Vec<f32>) -> Vec<f32> {
    normalize(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pair_recovers_almost_exactly() {
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.1, 0.5, -0.3, 0.2, 0.0, 0.4, -0.1, 0.6];
        let mut slot = SuperposedSlot::new(8);
        slot.insert(&key, &value);
        let recovered = slot.expand(&key);
        let sim = cosine_sim(&recovered, &value);
        assert!(sim > 0.99, "expected near-perfect recovery, got sim={sim}");
    }

    #[test]
    fn accuracy_degrades_as_more_pairs_are_bundled() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0);
        let dim = 64;

        let rand_vec =
            |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect() };

        let target_key = rand_vec(&mut rng);
        let target_value = rand_vec(&mut rng);

        let mut sims = Vec::new();
        for n_extra in [0usize, 4, 16, 64] {
            let mut slot = SuperposedSlot::new(dim);
            slot.insert(&target_key, &target_value);
            for _ in 0..n_extra {
                slot.insert(&rand_vec(&mut rng), &rand_vec(&mut rng));
            }
            let recovered = slot.expand(&target_key);
            sims.push(cosine_sim(&recovered, &target_value));
        }
        // More clutter in the slot should not improve retrieval quality.
        for w in sims.windows(2) {
            assert!(
                w[0] + 1e-3 >= w[1],
                "similarity should not increase with more clutter: {:?}",
                sims
            );
        }
    }
}
