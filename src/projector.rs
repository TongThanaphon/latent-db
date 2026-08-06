//! Random projection for dimensionality reduction.
//!
//! Inspired by katgpt-rs's `ShardEmbedding` (JL random orthogonal-ish
//! projection `[f32;64] -> [f32;8]`). Used here to build a cheap, fixed
//! (deterministic, seedable) low-dimensional "sketch" of a full embedding
//! so the index can do approximate nearest-centroid search without ever
//! touching the full-size vector.
//!
//! By the Johnson-Lindenstrauss lemma, a random projection to O(log n / eps^2)
//! dimensions approximately preserves pairwise distances, which is exactly
//! the property we need for an approximate index.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Projector {
    /// Row-major out_dim x in_dim matrix.
    matrix: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    /// BLAKE3 commitment over `matrix`'s bytes (borrowed from katgpt-rs's
    /// `JlProjectionMatrix`), so a projector loaded from disk can be
    /// checked for corruption/tampering before it's trusted to reproduce
    /// the same sketch space every other stored vector was indexed under.
    commitment: [u8; 32],
}

impl Projector {
    /// Build a new random projector. Deterministic given the same seed,
    /// so the same projector can be reconstructed without persisting the
    /// full matrix if desired (here we persist it for simplicity).
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (out_dim as f32).sqrt();
        let mut matrix = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..(in_dim * out_dim) {
            // Box-Muller transform for approximately standard-normal entries.
            let u1: f32 = rng.gen_range(1e-9..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            matrix.push(z * scale);
        }
        let mut p = Projector {
            matrix,
            in_dim,
            out_dim,
            commitment: [0u8; 32],
        };
        p.commit();
        p
    }

    /// Recompute and store the BLAKE3 commitment over the current matrix.
    pub fn commit(&mut self) {
        self.commitment = *Self::hash_matrix(&self.matrix).as_bytes();
    }

    /// Check that the stored commitment still matches the current matrix
    /// contents — e.g. after deserializing a `Projector` from disk.
    pub fn verify(&self) -> bool {
        self.commitment == *Self::hash_matrix(&self.matrix).as_bytes()
    }

    fn hash_matrix(matrix: &[f32]) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        for x in matrix {
            hasher.update(&x.to_le_bytes());
        }
        hasher.finalize()
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Project a full-size vector down to the sketch space.
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        self.project_into(v, &mut out);
        out
    }

    /// Same as [`Self::project`], but writes into a caller-provided buffer
    /// (`out.len() == self.out_dim()`) instead of allocating a fresh `Vec`.
    /// Lets a hot path (e.g. `LatentDb::insert`) reuse one scratch buffer
    /// across calls instead of paying a heap allocation every time.
    pub fn project_into(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(v.len(), self.in_dim, "vector dimension mismatch");
        assert_eq!(out.len(), self.out_dim, "output buffer size mismatch");
        for (o, out_val) in out.iter_mut().enumerate() {
            let row_off = o * self.in_dim;
            let row = &self.matrix[row_off..row_off + self.in_dim];
            *out_val = crate::simd::simd_dot_f32(row, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_deterministic() {
        let p1 = Projector::new(64, 8, 42);
        let p2 = Projector::new(64, 8, 42);
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        assert_eq!(p1.project(&v), p2.project(&v));
    }

    #[test]
    fn projection_reduces_dimension() {
        let p = Projector::new(64, 8, 7);
        let v: Vec<f32> = vec![1.0; 64];
        assert_eq!(p.project(&v).len(), 8);
    }

    #[test]
    fn commitment_verifies_untampered_matrix() {
        let p = Projector::new(64, 8, 42);
        assert!(p.verify());
    }

    #[test]
    fn commitment_detects_tampering() {
        let mut p = Projector::new(64, 8, 42);
        p.matrix[0] += 1.0;
        assert!(!p.verify());
    }

    #[test]
    fn commitment_survives_serde_roundtrip() {
        let p = Projector::new(64, 8, 42);
        let bytes = bincode::serialize(&p).unwrap();
        let reloaded: Projector = bincode::deserialize(&bytes).unwrap();
        assert!(reloaded.verify());
    }

    #[test]
    fn project_into_matches_project() {
        let p = Projector::new(64, 8, 42);
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let expected = p.project(&v);
        let mut out = vec![0.0f32; 8];
        p.project_into(&v, &mut out);
        assert_eq!(expected, out);
    }

    #[test]
    #[should_panic(expected = "output buffer size mismatch")]
    fn project_into_rejects_wrong_output_length() {
        let p = Projector::new(64, 8, 42);
        let v: Vec<f32> = vec![1.0; 64];
        let mut out = vec![0.0f32; 4];
        p.project_into(&v, &mut out);
    }
}
