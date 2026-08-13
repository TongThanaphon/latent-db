//! In-place trajectory steering (Issue #23): `steer_next()` advances a
//! caller-owned position vector by a steering direction, gated by
//! [`crate::topology::is_viable`], plus an O(1) edge-weight update for a
//! [`crate::topology::ViableNode`]'s outgoing edges.
//!
//! **Where the edge weight lives:** #22 fixes `ViableNode`'s fields to
//! exactly `vector_idx`/`neighbors`/`neighbor_count` ("stores only" --
//! no room for a per-edge weight there), while #23 asks for an O(1) update
//! of "a `ViableNode` edge's ... weight". [`EdgeWeights`] resolves that by
//! living *beside* a `ViableNode` rather than inside it: a `[f32;
//! MAX_NEIGHBORS]` array index-aligned with `ViableNode::neighbors` (slot
//! `i` here is the weight for `neighbors[i]`), stack/static-resident like
//! everything else in this crate. "O(1), no scan over `neighbors`" then
//! falls out directly -- the caller already knows (or looked up once) which
//! edge index it means, so [`EdgeWeights::update`] is a single indexed
//! read-modify-write.

use crate::topology::{is_viable, Boundary};
use crate::{DIM, MAX_NEIGHBORS};

/// Per-edge memory/confidence weight for a [`crate::topology::ViableNode`]'s
/// outgoing edges. See the module doc for why this is a companion array
/// rather than a field on `ViableNode` itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeWeights {
    weights: [f32; MAX_NEIGHBORS],
}

impl EdgeWeights {
    /// All edge weights start at zero.
    pub const fn zeroed() -> Self {
        Self {
            weights: [0.0; MAX_NEIGHBORS],
        }
    }

    /// The current weight at `edge_index` (the position in
    /// `ViableNode::neighbors` the weight applies to).
    pub fn get(&self, edge_index: usize) -> f32 {
        self.weights[edge_index]
    }

    /// O(1) read-modify-write of the weight at `edge_index`: a single
    /// indexed add, no scan over `neighbors` to locate the edge -- the
    /// caller supplies the index directly. Returns the updated weight.
    pub fn update(&mut self, edge_index: usize, delta: f32) -> f32 {
        self.weights[edge_index] += delta;
        self.weights[edge_index]
    }
}

/// Advances `state` in place by `alpha * direction`, provided the resulting
/// position is still [`is_viable`] within `boundary`.
///
/// If the candidate step would leave the safe region, `state` is reverted to
/// its original value and this returns `false`. Otherwise `state` holds the
/// new position and this returns `true`.
///
/// No second `[f32; DIM]` buffer is allocated to hold the candidate before
/// checking it: `state` is mutated in place, speculatively, then reverted in
/// place if `is_viable` rejects the result -- the whole call touches only
/// the caller-owned slice plus scalar locals.
pub fn steer_next(
    state: &mut [f32; DIM],
    direction: &[f32; DIM],
    alpha: f32,
    boundary: &Boundary,
) -> bool {
    for (s, d) in state.iter_mut().zip(direction.iter()) {
        *s += alpha * d;
    }
    if is_viable(state, boundary) {
        true
    } else {
        for (s, d) in state.iter_mut().zip(direction.iter()) {
            *s -= alpha * d;
        }
        false
    }
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

    // ── steer_next ──────────────────────────────────────────────────────

    #[test]
    fn steer_next_moves_state_toward_target_when_viable() {
        let mut state = [0.0f32; DIM];
        let direction = sample_vector(1.0);
        let original = state;
        let boundary = Boundary {
            min: -10.0,
            max: 10.0,
        };

        let committed = steer_next(&mut state, &direction, 0.5, &boundary);

        assert!(committed);
        for (i, ((&s, &o), &d)) in state
            .iter()
            .zip(original.iter())
            .zip(direction.iter())
            .enumerate()
        {
            let want = o + 0.5 * d;
            assert!((s - want).abs() < 1e-6, "dim {i}: got {s}, want {want}");
        }
    }

    #[test]
    fn steer_next_rejects_a_step_that_leaves_the_safe_region() {
        // Boundary tight enough that any nonzero step from the origin
        // leaves it.
        let boundary = Boundary {
            min: -0.01,
            max: 0.01,
        };
        let mut state = [0.0f32; DIM];
        let direction = sample_vector(2.0);

        let committed = steer_next(&mut state, &direction, 1.0, &boundary);

        assert!(!committed);
        // Reverted, not left mid-step -- compare with a tolerance since
        // `(a + x) - x` isn't guaranteed bit-exact under IEEE 754 rounding.
        for (i, &s) in state.iter().enumerate() {
            assert!(s.abs() < 1e-5, "dim {i}: expected revert to ~0.0, got {s}");
        }
    }

    #[test]
    fn steer_next_rejected_step_keeps_state_viable_for_a_later_call() {
        let boundary = Boundary {
            min: -1.0,
            max: 1.0,
        };
        // Start already at the edge; a further positive step in the same
        // direction would leave the region and must be reverted.
        let mut state = [1.0f32; DIM];
        let direction = [1.0f32; DIM];

        assert!(!steer_next(&mut state, &direction, 0.1, &boundary));
        // Still viable after the reverted attempt -- a subsequent call
        // isn't corrupted by the rejected one.
        assert!(is_viable(&state, &boundary));
    }

    // ── EdgeWeights ─────────────────────────────────────────────────────

    #[test]
    fn edge_weights_start_at_zero() {
        let weights = EdgeWeights::zeroed();
        for i in 0..MAX_NEIGHBORS {
            assert_eq!(weights.get(i), 0.0);
        }
    }

    #[test]
    fn edge_weights_update_reports_before_and_after() {
        let mut weights = EdgeWeights::zeroed();

        assert_eq!(weights.get(3), 0.0);
        let after_first = weights.update(3, 0.5);
        assert_eq!(after_first, 0.5);
        assert_eq!(weights.get(3), 0.5);

        let after_second = weights.update(3, -0.2);
        assert!((after_second - 0.3).abs() < 1e-6);
        assert!((weights.get(3) - 0.3).abs() < 1e-6);

        // Untouched slots are unaffected -- the update is O(1) on exactly
        // the given edge index, not a scan/rewrite over all of them.
        assert_eq!(weights.get(0), 0.0);
    }
}
