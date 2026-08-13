//! Fixed-size graph topology (Issue #22): `ViableNode` -- a stack/static
//! -resident graph node with up to `MAX_NEIGHBORS` outgoing edges -- and
//! `is_viable()`, the zero-allocation boundary check that gates whether
//! steering (#23) is allowed to start from a given vector's coordinates.

use crate::{DIM, MAX_NEIGHBORS};

/// A single graph node: the id of the vector it represents, plus up to
/// `MAX_NEIGHBORS` outgoing edges stored as a fixed-size array with an
/// explicit live-count rather than a heap-backed collection (e.g. `Vec<u32>`).
///
/// `Copy`: at `4 + MAX_NEIGHBORS * 4 + 1` bytes (69, padded to 72 by
/// `repr(C)` alignment) it's small enough that an implicit by-value copy is
/// the same negligible cost as passing a reference -- unlike
/// [`crate::storage::ZeroAllocLatent`]'s deliberate `!Copy` at
/// `VECTOR_BYTES + 8` (3080) bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViableNode {
    pub vector_idx: u32,
    pub neighbors: [u32; MAX_NEIGHBORS],
    pub neighbor_count: u8,
}

impl ViableNode {
    /// A node with no outgoing edges yet.
    pub const fn new(vector_idx: u32) -> Self {
        Self {
            vector_idx,
            neighbors: [0; MAX_NEIGHBORS],
            neighbor_count: 0,
        }
    }

    /// The live neighbor ids -- `neighbors[..neighbor_count]`. Never reads
    /// the unset trailing slots past `neighbor_count`.
    pub fn live_neighbors(&self) -> &[u32] {
        &self.neighbors[..self.neighbor_count as usize]
    }

    /// Appends `neighbor_vector_idx` as an outgoing edge. Returns `false`
    /// (no-op, `self` unchanged) if `neighbor_count` is already at
    /// `MAX_NEIGHBORS`.
    pub fn push_neighbor(&mut self, neighbor_vector_idx: u32) -> bool {
        let i = self.neighbor_count as usize;
        if i >= MAX_NEIGHBORS {
            return false;
        }
        self.neighbors[i] = neighbor_vector_idx;
        self.neighbor_count += 1;
        true
    }
}

/// A uniform, per-dimension safe region: a vector is viable only if *every*
/// coordinate falls within `[min, max]`, inclusive on both ends.
///
/// Deliberately a single scalar pair rather than a `[f32; DIM]` pair of
/// per-dimension bounds: every coordinate shares the same region, so
/// `is_viable` naturally has both a per-dimension reading (does *this*
/// coordinate hold) and an aggregate reading (do *all* of them) without a
/// second, DIM-sized field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundary {
    pub min: f32,
    pub max: f32,
}

/// Whether every coordinate of `v` falls within `boundary`, inclusive on
/// both ends. Allocates nothing; short-circuits on the first violation.
pub fn is_viable(v: &[f32; DIM], boundary: &Boundary) -> bool {
    v.iter().all(|&x| x >= boundary.min && x <= boundary.max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vector(seed: f32) -> [f32; DIM] {
        let mut v = [0.0f32; DIM];
        for (i, slot) in v.iter_mut().enumerate() {
            // Bounded in [-1, 1] by construction (sin), handy for the
            // boundary tests below.
            *slot = ((i as f32 + seed) * 0.037).sin();
        }
        v
    }

    // ── ViableNode ──────────────────────────────────────────────────────

    #[test]
    fn viable_node_is_repr_c_no_heap() {
        // Exact match, not just a lower bound -- a smuggled-in heap-backed
        // field (e.g. `Vec<u32>`) would inflate this size and fail the
        // assertion. `repr(C)` layout: u32 (4) + [u32; MAX_NEIGHBORS]
        // (MAX_NEIGHBORS * 4) + u8 (1), padded up to u32's 4-byte alignment.
        let expected = (4 + MAX_NEIGHBORS * 4 + 1).next_multiple_of(4);
        assert_eq!(core::mem::size_of::<ViableNode>(), expected);
    }

    #[test]
    fn push_neighbor_updates_count_and_slot() {
        let mut node = ViableNode::new(7);
        assert_eq!(node.vector_idx, 7);
        assert_eq!(node.live_neighbors(), &[] as &[u32]);

        assert!(node.push_neighbor(11));
        assert!(node.push_neighbor(22));
        assert_eq!(node.neighbor_count, 2);
        assert_eq!(node.live_neighbors(), &[11, 22]);
    }

    #[test]
    fn push_neighbor_stops_at_max_neighbors_no_reads_of_unset_slots() {
        let mut node = ViableNode::new(0);
        for i in 0..MAX_NEIGHBORS as u32 {
            assert!(node.push_neighbor(i));
        }
        assert_eq!(node.neighbor_count as usize, MAX_NEIGHBORS);
        // One more push is a no-op -- count doesn't overflow past the array.
        assert!(!node.push_neighbor(999));
        assert_eq!(node.neighbor_count as usize, MAX_NEIGHBORS);

        // `live_neighbors()` reports exactly the pushed ids, in order --
        // proves iteration is bounded by `neighbor_count`, not the full
        // backing array (which would be indistinguishable here since it's
        // fully populated, but the length itself is the bound under test).
        let expected: Vec<u32> = (0..MAX_NEIGHBORS as u32).collect();
        assert_eq!(node.live_neighbors(), expected.as_slice());
    }

    #[test]
    fn live_neighbors_never_reads_past_neighbor_count() {
        let mut node = ViableNode::new(0);
        node.push_neighbor(42);
        // Directly poison a slot past `neighbor_count` with a sentinel that
        // would fail the assertion below if `live_neighbors()` leaked it.
        node.neighbors[1] = 0xDEAD_BEEF;
        assert_eq!(node.live_neighbors(), &[42]);
    }

    // ── is_viable ───────────────────────────────────────────────────────

    #[test]
    fn in_bounds_vector_is_viable() {
        let v = sample_vector(0.0);
        let boundary = Boundary {
            min: -1.0,
            max: 1.0,
        };
        assert!(is_viable(&v, &boundary));
    }

    #[test]
    fn single_dimension_out_of_bounds_is_not_viable() {
        let mut v = sample_vector(0.0);
        let boundary = Boundary {
            min: -1.0,
            max: 1.0,
        };
        assert!(is_viable(&v, &boundary));
        // Only one coordinate violates -- the per-dimension reading of the
        // acceptance criteria.
        v[400] = 1.5;
        assert!(!is_viable(&v, &boundary));
    }

    #[test]
    fn every_dimension_out_of_bounds_is_not_viable() {
        let v = [5.0f32; DIM];
        let boundary = Boundary {
            min: -1.0,
            max: 1.0,
        };
        // Every coordinate violates -- the aggregate reading of the
        // acceptance criteria.
        assert!(!is_viable(&v, &boundary));
    }

    #[test]
    fn boundary_exact_coordinates_are_viable() {
        let mut v = [0.0f32; DIM];
        v[0] = -2.0;
        v[1] = 2.0;
        let boundary = Boundary {
            min: -2.0,
            max: 2.0,
        };
        // Inclusive on both ends.
        assert!(is_viable(&v, &boundary));

        v[1] = 2.0 + f32::EPSILON.max(1e-6);
        assert!(!is_viable(&v, &boundary));
    }
}
