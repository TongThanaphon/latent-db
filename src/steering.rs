//! Latent-space steering: inject a frozen direction vector into a query
//! before search, biasing results toward (or away from) a concept axis
//! without retraining anything.
//!
//! Inspired by katgpt-rs's Latent Field Steering primitive
//! (`crates/katgpt-core/src/latent_steering.rs`, Plan 309): a unit-norm
//! direction `v` plus a strength `alpha in [0, 1]`, BLAKE3-committed so a
//! persisted steering vector can be checked for tampering/corruption before
//! use, with a freeze/thaw envelope for storing it as opaque bytes.
//!
//! Reimplemented independently for `LatentDb` rather than ported: the
//! katgpt-rs original mutates a live NPC's *stored* per-tick latent state
//! (`apply_latent_steering` writes into `&mut [f32]`), plus a whole
//! localized-field layer (`FieldSupport::{Global,Radius,Zone}` and
//! `apply_field_to_crowd`) for steering many game entities by world
//! position. `LatentDb` records have no position, and they live as PQ codes,
//! not raw floats -- steering a *stored* record would mean decode -> add ->
//! re-encode, which is lossy and would silently break that record's content
//! hash, its dedup entry, and its Merkle leaf. So steering here applies to
//! the *query* vector instead, before projection/search -- the same
//! additive op (`state[i] += alpha * direction[i]`), just on the read path
//! rather than mutating storage. See `LatentDb::search_steered`.

use blake3::Hasher;

/// Errors returned by [`SteeringVector::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteeringError {
    /// `direction`'s L2 norm deviates from 1.0 by more than the
    /// constructor's tolerance.
    NotUnitNorm,
    /// `alpha` is outside `[0.0, 1.0]`.
    AlphaOutOfRange,
}

impl std::fmt::Display for SteeringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SteeringError::NotUnitNorm => write!(f, "direction vector is not unit-norm"),
            SteeringError::AlphaOutOfRange => write!(f, "alpha must be in [0.0, 1.0]"),
        }
    }
}

impl std::error::Error for SteeringError {}

/// A unit-norm direction in embedding space plus a scalar strength,
/// BLAKE3-committed so tampering/corruption is detectable via [`Self::verify`].
#[derive(Debug, Clone)]
pub struct SteeringVector {
    direction: Vec<f32>,
    alpha: f32,
    commitment: [u8; 32],
}

impl SteeringVector {
    /// Construct a steering vector, validating unit-norm (within `norm_tol`)
    /// and `alpha in [0, 1]`.
    pub fn new(direction: Vec<f32>, alpha: f32, norm_tol: f32) -> Result<Self, SteeringError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(SteeringError::AlphaOutOfRange);
        }
        let norm = l2_norm(&direction);
        if (norm - 1.0).abs() > norm_tol {
            return Err(SteeringError::NotUnitNorm);
        }
        let commitment = compute_commitment(&direction, alpha);
        Ok(Self {
            direction,
            alpha,
            commitment,
        })
    }

    /// Construct without validation -- caller guarantees unit-norm + alpha
    /// range (e.g. a direction already verified via a [`SteeringEnvelope`]).
    pub fn new_unchecked(direction: Vec<f32>, alpha: f32) -> Self {
        let commitment = compute_commitment(&direction, alpha);
        Self {
            direction,
            alpha,
            commitment,
        }
    }

    /// Re-check unit-norm (within `tol`) and that the stored commitment
    /// still matches the current contents.
    pub fn verify(&self, tol: f32) -> bool {
        let norm = l2_norm(&self.direction);
        (norm - 1.0).abs() <= tol
            && compute_commitment(&self.direction, self.alpha) == self.commitment
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.direction
    }

    #[inline]
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Apply steering in place: `state[i] += alpha * direction[i]`.
    ///
    /// # Panics
    /// Panics if `state.len() != self.dim()`.
    #[inline]
    pub fn apply(&self, state: &mut [f32]) {
        self.apply_weighted(state, 1.0);
    }

    /// Apply steering with an explicit extra weight `w`: effective strength
    /// is `alpha * w`. `w <= 0.0` is a no-op (skips the write entirely).
    ///
    /// # Panics
    /// Panics if `state.len() != self.dim()`.
    pub fn apply_weighted(&self, state: &mut [f32], w: f32) {
        assert_eq!(
            state.len(),
            self.dim(),
            "state dim {} != steering dim {}",
            state.len(),
            self.dim()
        );
        if w <= 0.0 {
            return;
        }
        let scale = self.alpha * w;
        for (s, d) in state.iter_mut().zip(self.direction.iter()) {
            *s += scale * d;
        }
    }
}

/// Self-contained freeze/thaw envelope for a [`SteeringVector`] -- BLAKE3
/// commitment + serialized payload together, so a steering vector can be
/// persisted as opaque bytes and later thawed with tamper detection, the
/// same freeze/thaw pattern `Projector` uses for its matrix.
#[derive(Debug, Clone)]
pub struct SteeringEnvelope {
    commitment: [u8; 32],
    /// `dim` (u32 LE) || `direction` (f32 LE x dim) || `alpha` (f32 LE).
    payload: Vec<u8>,
}

impl SteeringEnvelope {
    /// Freeze a [`SteeringVector`] into a self-contained envelope.
    pub fn freeze(v: &SteeringVector) -> Self {
        let dim = v.dim() as u32;
        let mut payload = Vec::with_capacity(4 + v.dim() * 4 + 4);
        payload.extend_from_slice(&dim.to_le_bytes());
        for &f in v.as_slice() {
            payload.extend_from_slice(&f.to_le_bytes());
        }
        payload.extend_from_slice(&v.alpha.to_le_bytes());
        let commitment = *blake3::hash(&payload).as_bytes();
        Self { commitment, payload }
    }

    /// Whether the envelope's commitment still matches its payload.
    #[inline]
    pub fn verify(&self) -> bool {
        *blake3::hash(&self.payload).as_bytes() == self.commitment
    }

    /// Thaw the envelope back into a [`SteeringVector`]. Returns `None` if
    /// the commitment doesn't match the payload (tampered/corrupted), or the
    /// payload is malformed (truncated, wrong length, or contains NaN).
    pub fn thaw(&self) -> Option<SteeringVector> {
        if !self.verify() {
            return None;
        }
        Self::deserialize(&self.payload)
    }

    #[inline]
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Serialized payload bytes, for external persistence / transport.
    #[inline]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    fn deserialize(payload: &[u8]) -> Option<SteeringVector> {
        if payload.len() < 4 {
            return None;
        }
        let dim = u32::from_le_bytes(payload[..4].try_into().ok()?) as usize;
        let expected_len = 4 + dim * 4 + 4;
        if payload.len() != expected_len {
            return None;
        }
        let mut direction = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 4 + i * 4;
            let f = f32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?);
            if f.is_nan() {
                return None;
            }
            direction.push(f);
        }
        let alpha = f32::from_le_bytes(payload[4 + dim * 4..4 + dim * 4 + 4].try_into().ok()?);
        if alpha.is_nan() {
            return None;
        }
        Some(SteeringVector::new_unchecked(direction, alpha))
    }
}

fn compute_commitment(direction: &[f32], alpha: f32) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for &f in direction {
        hasher.update(&f.to_le_bytes());
    }
    hasher.update(&alpha.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_direction(d: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut v: Vec<f32> = (0..d)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect();
        let norm = l2_norm(&v);
        for x in &mut v {
            *x /= norm.max(1e-12);
        }
        v
    }

    #[test]
    fn new_rejects_non_unit_norm() {
        let dir = vec![1.0, 1.0, 0.0];
        let err = SteeringVector::new(dir, 0.5, 1e-4).unwrap_err();
        assert_eq!(err, SteeringError::NotUnitNorm);
    }

    #[test]
    fn new_rejects_alpha_out_of_range() {
        let dir = unit_direction(4, 1);
        let err = SteeringVector::new(dir, 1.5, 1e-4).unwrap_err();
        assert_eq!(err, SteeringError::AlphaOutOfRange);
    }

    #[test]
    fn new_accepts_a_valid_unit_vector() {
        let dir = unit_direction(8, 2);
        let v = SteeringVector::new(dir, 0.5, 1e-4).unwrap();
        assert!(v.verify(1e-4));
    }

    #[test]
    fn apply_shifts_state_by_alpha_times_direction() {
        let dir = unit_direction(4, 3);
        let v = SteeringVector::new(dir.clone(), 0.5, 1e-4).unwrap();
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let original = state.clone();
        v.apply(&mut state);
        for i in 0..4 {
            assert!((state[i] - (original[i] + 0.5 * dir[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_weighted_scales_alpha_by_w() {
        let dir = unit_direction(4, 4);
        let v = SteeringVector::new(dir.clone(), 0.5, 1e-4).unwrap();
        let mut state = vec![0.0; 4];
        v.apply_weighted(&mut state, 0.5);
        for i in 0..4 {
            assert!((state[i] - (0.5 * 0.5 * dir[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_weighted_with_nonpositive_w_is_a_noop() {
        let dir = unit_direction(4, 5);
        let v = SteeringVector::new(dir, 0.5, 1e-4).unwrap();
        let mut state = vec![9.0; 4];
        let original = state.clone();
        v.apply_weighted(&mut state, 0.0);
        assert_eq!(state, original);
        v.apply_weighted(&mut state, -1.0);
        assert_eq!(state, original);
    }

    #[test]
    #[should_panic(expected = "state dim")]
    fn apply_panics_on_dim_mismatch() {
        let dir = unit_direction(4, 6);
        let v = SteeringVector::new(dir, 0.5, 1e-4).unwrap();
        let mut state = vec![0.0; 3];
        v.apply(&mut state);
    }

    #[test]
    fn verify_detects_direct_tampering() {
        let dir = unit_direction(4, 7);
        let mut v = SteeringVector::new(dir, 0.5, 1e-4).unwrap();
        v.direction[0] += 1.0;
        assert!(!v.verify(1e-4));
    }

    #[test]
    fn envelope_freeze_thaw_roundtrips() {
        let dir = unit_direction(8, 8);
        let v = SteeringVector::new(dir, 0.7, 1e-4).unwrap();
        let envelope = SteeringEnvelope::freeze(&v);
        assert!(envelope.verify());
        let thawed = envelope.thaw().expect("valid envelope thaws");
        assert_eq!(thawed.as_slice(), v.as_slice());
        assert_eq!(thawed.alpha(), v.alpha());
    }

    #[test]
    fn envelope_thaw_rejects_a_tampered_payload() {
        let dir = unit_direction(8, 9);
        let v = SteeringVector::new(dir, 0.7, 1e-4).unwrap();
        let mut envelope = SteeringEnvelope::freeze(&v);
        envelope.payload[10] ^= 0xFF;
        assert!(!envelope.verify());
        assert!(envelope.thaw().is_none());
    }

    #[test]
    fn envelope_thaw_rejects_a_truncated_payload() {
        let dir = unit_direction(8, 10);
        let v = SteeringVector::new(dir, 0.7, 1e-4).unwrap();
        let mut envelope = SteeringEnvelope::freeze(&v);
        envelope.payload.truncate(envelope.payload.len() - 4);
        // Truncating without updating the commitment should fail verify.
        assert!(!envelope.verify());
    }

    #[test]
    fn envelope_payload_is_stable_and_reloadable() {
        let dir = unit_direction(8, 11);
        let v = SteeringVector::new(dir, 0.3, 1e-4).unwrap();
        let envelope = SteeringEnvelope::freeze(&v);
        let bytes = envelope.payload().to_vec();
        let commitment = envelope.commitment();
        let reloaded = SteeringEnvelope {
            commitment,
            payload: bytes,
        };
        let thawed = reloaded.thaw().expect("reloaded envelope thaws");
        assert_eq!(thawed.as_slice(), v.as_slice());
    }
}
