//! Product Quantization (PQ) codec.
//!
//! Inspired by katgpt-rs's "Hybrid OCT+PQ" KV cache codec: split a vector
//! into sub-vectors, and quantize each sub-vector against a small learned
//! codebook. This is the compression layer that lets us store thousands of
//! embeddings in a fraction of the raw f32 footprint while still supporting
//! approximate reconstruction for scoring.
//!
//! Each sub-vector is replaced by a single `u8` codebook index, so a vector
//! of dimension `d` split into `m` subspaces with `k` centroids per subspace
//! compresses `d * 4` bytes down to `m` bytes (plus the shared codebooks).

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct PqCodec {
    dim: usize,
    n_subspaces: usize,
    sub_dim: usize,
    n_centroids: usize,
    /// codebooks[subspace][centroid] = `Vec<f32>` of length sub_dim
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl PqCodec {
    /// Train a PQ codec from a batch of training vectors using Lloyd's
    /// k-means algorithm, independently per subspace.
    pub fn train(
        training_vectors: &[Vec<f32>],
        n_subspaces: usize,
        n_centroids: usize,
        iterations: usize,
        seed: u64,
    ) -> Self {
        assert!(
            !training_vectors.is_empty(),
            "need at least one training vector"
        );
        let dim = training_vectors[0].len();
        assert!(
            dim.is_multiple_of(n_subspaces),
            "dim ({dim}) must be divisible by n_subspaces ({n_subspaces})"
        );
        let sub_dim = dim / n_subspaces;

        let mut codebooks = Vec::with_capacity(n_subspaces);
        for s in 0..n_subspaces {
            let sub_vectors: Vec<Vec<f32>> = training_vectors
                .iter()
                .map(|v| v[s * sub_dim..(s + 1) * sub_dim].to_vec())
                .collect();
            let centroids = kmeans(
                &sub_vectors,
                n_centroids,
                iterations,
                seed.wrapping_add(s as u64),
            );
            codebooks.push(centroids);
        }

        PqCodec {
            dim,
            n_subspaces,
            sub_dim,
            n_centroids,
            codebooks,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn code_len(&self) -> usize {
        self.n_subspaces
    }

    /// Encode a full-precision vector into PQ codes (one byte per subspace).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        let mut codes = Vec::with_capacity(self.n_subspaces);
        for s in 0..self.n_subspaces {
            let sub = &v[s * self.sub_dim..(s + 1) * self.sub_dim];
            let mut best_idx = 0usize;
            let mut best_dist = f32::MAX;
            for (c_idx, centroid) in self.codebooks[s].iter().enumerate() {
                let d = sq_dist(sub, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }
            codes.push(best_idx as u8);
        }
        codes
    }

    /// Reconstruct an approximate vector from PQ codes.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.n_subspaces, "code length mismatch");
        let mut out = Vec::with_capacity(self.dim);
        for (s, &code) in codes.iter().enumerate() {
            let c_idx = code as usize;
            out.extend_from_slice(&self.codebooks[s][c_idx]);
        }
        out
    }

    /// Bytes of raw storage per vector once quantized (codes only, not
    /// counting the shared codebooks which are amortized across the DB).
    pub fn compressed_bytes(&self) -> usize {
        self.n_subspaces
    }

    /// Bytes of storage a raw f32 vector of this dimension would need.
    pub fn raw_bytes(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }

    #[allow(dead_code)]
    fn n_centroids(&self) -> usize {
        self.n_centroids
    }
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Minimal Lloyd's-algorithm k-means. Re-seeds any centroid that loses all
/// of its points to a random data point, so clusters never go empty.
pub(crate) fn kmeans(data: &[Vec<f32>], k: usize, iterations: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let n = data.len();
    let k = k.min(n).max(1);
    let dim = data[0].len();

    // Random initial centroids drawn from the data itself (Forgy init).
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|_| data[rng.gen_range(0..n)].clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations.max(1) {
        // Assignment step.
        for (i, v) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f32::MAX;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = sq_dist(v, c);
                if d < best_dist {
                    best_dist = d;
                    best = c_idx;
                }
            }
            assignments[i] = best;
        }

        // Update step.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed dead centroid with a random data point.
                centroids[c] = data[rng.gen_range(0..n)].clone();
            } else {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_is_approximate() {
        let training: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                let x = i as f32;
                vec![x.sin(), x.cos(), (x * 0.5).sin(), (x * 0.3).cos()]
            })
            .collect();
        let codec = PqCodec::train(&training, 2, 16, 10, 1);
        let v = training[5].clone();
        let codes = codec.encode(&v);
        assert_eq!(codes.len(), codec.code_len());
        let decoded = codec.decode(&codes);
        assert_eq!(decoded.len(), v.len());
        let err: f32 = sq_dist(&v, &decoded);
        assert!(err < 1.0, "reconstruction error too high: {err}");
    }

    #[test]
    fn compression_ratio_is_meaningful() {
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32; 32]).collect();
        let codec = PqCodec::train(&training, 4, 16, 5, 2);
        assert!(codec.compressed_bytes() < codec.raw_bytes());
    }
}
