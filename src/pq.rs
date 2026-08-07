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
        let mut codes = vec![0u8; self.n_subspaces];
        self.encode_into(v, &mut codes);
        codes
    }

    /// Same as [`Self::encode`], but writes into a caller-provided buffer
    /// (`out.len() == self.code_len()`) instead of allocating a fresh `Vec`.
    /// Lets a hot path (e.g. `LatentDb::insert`) encode directly into a
    /// pre-allocated arena slot instead of paying a heap allocation every
    /// call just to immediately copy the result somewhere else.
    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        assert_eq!(out.len(), self.n_subspaces, "output buffer size mismatch");
        for (s, out_byte) in out.iter_mut().enumerate() {
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
            *out_byte = best_idx as u8;
        }
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

    /// Build a per-query lookup table for asymmetric distance computation:
    /// this query's dot product and squared-norm against *every* centroid in
    /// *every* subspace, computed once. Scoring a candidate's PQ codes
    /// against the returned [`QueryLut`] (`QueryLut::cosine_score`) then
    /// reproduces `cosine_sim(query, self.decode(codes))` within tight
    /// floating-point tolerance -- same dot product and norms, just summed
    /// via LUT lookups over `codes` (a different accumulation order than a
    /// linear scan over the decoded vector, so not bit-exact) instead of a
    /// fresh per-candidate decode -- without ever materializing the decoded
    /// vector.
    ///
    /// O(n_subspaces * n_centroids * sub_dim) to build (SIMD-accelerated,
    /// see `crate::simd`); call once per query, then reuse the result to
    /// score every candidate.
    pub fn build_query_lut(&self, query: &[f32]) -> QueryLut {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");

        let mut dot = Vec::with_capacity(self.n_subspaces * self.n_centroids);
        let mut norm_sq = Vec::with_capacity(self.n_subspaces * self.n_centroids);
        for s in 0..self.n_subspaces {
            let q_sub = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            for centroid in &self.codebooks[s] {
                dot.push(crate::simd::simd_dot_f32(q_sub, centroid));
                norm_sq.push(crate::simd::simd_dot_f32(centroid, centroid));
            }
        }
        let row_base: Vec<u32> = (0..self.n_subspaces as u32)
            .map(|s| s * self.n_centroids as u32)
            .collect();
        let query_norm = crate::simd::simd_dot_f32(query, query).sqrt();

        QueryLut {
            n_subspaces: self.n_subspaces,
            dot,
            norm_sq,
            row_base,
            query_norm,
        }
    }
}

/// Precomputed per-query lookup table for PQ asymmetric distance
/// computation (product-quantization ADC): this query's dot product and
/// squared norm against every centroid in every subspace, built once by
/// [`PqCodec::build_query_lut`] and then reused to score every search
/// candidate.
///
/// Scoring a candidate's codes (`QueryLut::cosine_score`) sums `n_subspaces`
/// LUT lookups instead of decoding the candidate back to a full-precision
/// vector -- no allocation, no per-candidate reconstruction.
pub struct QueryLut {
    n_subspaces: usize,
    /// dot[s * n_centroids + c] = dot(query_sub[s], codebooks[s][c])
    dot: Vec<f32>,
    /// norm_sq[s * n_centroids + c] = ||codebooks[s][c]||^2
    norm_sq: Vec<f32>,
    /// row_base[s] = s * n_centroids -- each subspace's flat offset into
    /// `dot`/`norm_sq`, precomputed once so the per-candidate scoring loop
    /// never repeats the multiply.
    row_base: Vec<u32>,
    query_norm: f32,
}

impl QueryLut {
    /// Sum, over `codes`, of this LUT's per-subspace dot products and
    /// squared norms: `(Σ_s dot[s][codes[s]], Σ_s norm_sq[s][codes[s]])`.
    /// These are exactly the two quantities `cosine_sim` needs -- the dot
    /// product of query and decoded candidate, and the decoded candidate's
    /// squared norm -- computed without ever decoding `codes`.
    #[inline]
    fn score_parts(&self, codes: &[u8]) -> (f32, f32) {
        debug_assert_eq!(codes.len(), self.n_subspaces, "code length mismatch");
        let dot_sum = crate::simd::simd_lut_row_sum_f32(&self.dot, &self.row_base, codes);
        let norm_sq_sum = crate::simd::simd_lut_row_sum_f32(&self.norm_sq, &self.row_base, codes);
        (dot_sum, norm_sq_sum)
    }

    /// Cosine similarity between the query this LUT was built for and the
    /// candidate's decoded (approximate) vector -- the same formula as
    /// `superpose::cosine_sim(query, pq.decode(codes))`, within tight
    /// floating-point tolerance (LUT lookups sum in a different order than a
    /// linear scan over the decoded vector, so not bit-exact -- see
    /// `query_lut_cosine_score_matches_decode_baseline` below).
    #[inline]
    pub fn cosine_score(&self, codes: &[u8]) -> f32 {
        let (dot, norm_sq) = self.score_parts(codes);
        let denom = self.query_norm * norm_sq.sqrt();
        if denom < 1e-9 {
            0.0
        } else {
            dot / denom
        }
    }
}

pub(crate) fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
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

    /// `QueryLut::cosine_score` must reproduce `cosine_sim(query,
    /// pq.decode(codes))` for every candidate -- the whole point of the LUT
    /// is to compute the same score the old decode-then-cosine-sim path
    /// would, without ever materializing the decoded vector.
    #[test]
    fn query_lut_cosine_score_matches_decode_baseline() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(100);
        let training: Vec<Vec<f32>> = (0..300)
            .map(|_| (0..32).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let codec = PqCodec::train(&training, 4, 16, 15, 101);

        let all_codes: Vec<Vec<u8>> = training.iter().map(|v| codec.encode(v)).collect();

        for q_idx in [0usize, 7, 42, 199] {
            let query = &training[q_idx];
            let lut = codec.build_query_lut(query);
            for codes in &all_codes {
                let decoded = codec.decode(codes);
                let want = crate::superpose::cosine_sim(query, &decoded);
                let got = lut.cosine_score(codes);
                assert!(
                    (got - want).abs() < 1e-4,
                    "q_idx={q_idx}: lut score {got} != decode baseline {want}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "query dimension mismatch")]
    fn query_lut_rejects_wrong_query_dimension() {
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32; 16]).collect();
        let codec = PqCodec::train(&training, 2, 8, 5, 3);
        codec.build_query_lut(&[0.0; 8]);
    }

    #[test]
    fn encode_into_matches_encode() {
        let training: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                vec![
                    (i as f32).sin(),
                    (i as f32).cos(),
                    i as f32 * 0.1,
                    -(i as f32),
                ]
            })
            .collect();
        let codec = PqCodec::train(&training, 2, 8, 10, 5);
        let v = &training[3];
        let expected = codec.encode(v);
        let mut out = vec![0u8; codec.code_len()];
        codec.encode_into(v, &mut out);
        assert_eq!(expected, out);
    }

    #[test]
    #[should_panic(expected = "output buffer size mismatch")]
    fn encode_into_rejects_wrong_output_length() {
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32; 16]).collect();
        let codec = PqCodec::train(&training, 2, 8, 5, 3);
        let mut out = vec![0u8; 1];
        codec.encode_into(&training[0], &mut out);
    }
}
