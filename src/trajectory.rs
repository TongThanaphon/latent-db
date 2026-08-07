//! Geometric diagnostics over [`ViableGraph`] paths.
//!
//! Ported from katgpt-rs's `latent_trajectory_geometry.rs` (distilled from
//! Pandey, Singh, Mahdid, *Trajectory Geometry of Transformer Representations
//! Across Layers*, arXiv:2606.09287) -- pure geometry over a sequence of
//! vectors with no game-specific assumptions, so the port carries over
//! directly. [`geodesic`](ViableGraph::geodesic) and
//! [`random_walk`](ViableGraph::random_walk) hand back a bare `Vec<u64>`;
//! this module turns that into something a caller can reason about: is the
//! path a direct, committed traversal, or is it oscillating between
//! clusters, and how much do two paths from the same start diverge.
//!
//! Purely additive -- [`ViableGraph::geodesic`] / [`ViableGraph::random_walk`]
//! are untouched; [`path_geometry`] and [`bifurcation_ratio`] only ever read
//! a graph via [`ViableGraph::coords_of`].
//!
//! Three deliberate departures from the katgpt-rs original:
//! - It operates on raw `&[&[f32]]` state sequences; this version resolves
//!   `&[u64]` record-id paths against the [`ViableGraph`] they came from,
//!   since that's the shape `geodesic()`/`random_walk()` actually produce.
//! - It uses a polynomial `fast_acos` approximation to keep curvature
//!   computation to ~2ns/call, justified there by a per-token router budget.
//!   No such budget exists here -- a diagnostic runs once per traversal
//!   result, not per token -- so this version calls stdlib `f32::acos`
//!   directly rather than porting the approximation.
//! - It allocates a small `Vec<f32>` per resolvable step for the running
//!   displacement, rather than porting the original's two preallocated
//!   ping-pong buffers -- same reasoning: not a per-token hot loop, so the
//!   extra allocations don't matter here.

use crate::manifold::{euclidean, ViableGraph};
use crate::superpose::cosine_sim;

/// Geometric diagnostic over a [`ViableGraph`] path (paper eq. 3, 4, 6).
///
/// Computed by [`path_geometry`]. All three measurements are raw geometry --
/// not probabilities or confidence scores.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PathGeometry {
    /// Total Euclidean displacement accumulated along the path (paper eq. 3,
    /// `L(τ)`). Larger values indicate more movement through the space.
    pub length: f32,

    /// Mean turning-angle (radians) between consecutive displacement vectors
    /// (paper eq. 4, `κ̄`).
    ///
    /// Range `[0, π]`: `0.0` is a straight-line, committed path; near `π` is
    /// a reversal -- ping-ponging between two regions without committing.
    ///
    /// `0.0` if the path has fewer than 2 resolvable steps (need >= 2
    /// displacements for one turning angle).
    pub mean_curvature: f32,

    /// Minimum adjacent-step cosine similarity (paper eq. 6, `min_l SIM(l)`).
    ///
    /// Range `[-1, 1]` (clamped to `0.0` when either endpoint is the zero
    /// vector). Sharp drops localize where the path's direction changes most.
    ///
    /// `0.0` if the path has no resolvable steps at all.
    pub min_adjacent_cosine: f32,

    /// Number of resolvable displacement steps. Ids not present in the graph
    /// don't count and break the curvature chain at that point.
    pub n_steps: usize,
}

/// Progressive separation between two [`ViableGraph`] paths (paper Finding
/// 3), e.g. two `random_walk()`s from the same start.
///
/// Computed by [`bifurcation_ratio`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BifurcationResult {
    /// `‖a_L − b_L‖₂ / max(‖a_0 − b_0‖₂, ε)`. Values `> 1.0` indicate
    /// progressive separation; `< 1.0` indicates convergence.
    pub separation_ratio: f32,

    /// First step index (0-based) where pairwise separation exceeds `1.5 ×`
    /// the initial separation. `None` if the paths never diverge past that
    /// threshold, or if the initial separation is already below `ε` (no
    /// baseline to grow from).
    pub onset_step: Option<usize>,

    /// Final-step pairwise Euclidean separation `‖a_L − b_L‖₂`.
    pub final_separation: f32,
}

/// Score a path's length, curvature, and minimum adjacent-step cosine
/// similarity over the [`ViableGraph`] it came from.
///
/// `path` is typically the output of [`ViableGraph::geodesic`] or
/// [`ViableGraph::random_walk`], but any id sequence works.
///
/// # Edge cases
///
/// - `path.len() < 2` -> `Default::default()`.
/// - An id not present in `graph` is skipped: it contributes no length or
///   cosine, and it breaks the curvature chain (the next resolvable step
///   starts a fresh displacement rather than turning relative to a gap).
pub fn path_geometry(graph: &ViableGraph, path: &[u64]) -> PathGeometry {
    if path.len() < 2 {
        return PathGeometry::default();
    }

    let mut length = 0.0f32;
    let mut min_adjacent_cosine = 1.0f32;
    let mut curvature_sum = 0.0f32;
    let mut curvature_count = 0u32;
    let mut n_steps = 0usize;
    let mut prev_disp: Option<Vec<f32>> = None;

    for pair in path.windows(2) {
        let (Some(prev), Some(curr)) = (graph.coords_of(pair[0]), graph.coords_of(pair[1])) else {
            prev_disp = None;
            continue;
        };

        let disp: Vec<f32> = prev.iter().zip(curr).map(|(p, c)| c - p).collect();
        let disp_norm = euclidean(prev, curr);
        length += disp_norm;
        n_steps += 1;

        let cos = cosine_sim(prev, curr);
        if cos < min_adjacent_cosine {
            min_adjacent_cosine = cos;
        }

        if let Some(pd) = &prev_disp {
            let pd_norm = pd.iter().map(|v| v * v).sum::<f32>().sqrt();
            if pd_norm > f32::EPSILON && disp_norm > f32::EPSILON {
                let cos_dd = cosine_sim(pd, &disp).clamp(-1.0, 1.0);
                curvature_sum += cos_dd.acos();
                curvature_count += 1;
            }
        }
        prev_disp = Some(disp);
    }

    if n_steps == 0 {
        return PathGeometry::default();
    }

    PathGeometry {
        length,
        mean_curvature: if curvature_count > 0 {
            curvature_sum / curvature_count as f32
        } else {
            0.0
        },
        min_adjacent_cosine,
        n_steps,
    }
}

/// Progressive separation between two paths sharing an index (e.g. two
/// [`ViableGraph::random_walk`]s from the same start): how much `a` and `b`
/// pull apart from their first step to their last.
///
/// # Edge cases
///
/// - `a.len() != b.len()`, either is empty, or any id is missing from
///   `graph` -> `Default::default()` (`onset_step: None`).
/// - Initial separation below `ε = 1e-8` (paths start at/near the same
///   point): `separation_ratio` is `f32::INFINITY` if the paths end up
///   separated, or `1.0` if they stay together; `onset_step` is `None`
///   either way -- there's no nonzero baseline to grow past.
pub fn bifurcation_ratio(graph: &ViableGraph, a: &[u64], b: &[u64]) -> BifurcationResult {
    if a.len() != b.len() || a.is_empty() {
        return BifurcationResult::default();
    }

    let mut coords_a = Vec::with_capacity(a.len());
    let mut coords_b = Vec::with_capacity(b.len());
    for (&ia, &ib) in a.iter().zip(b) {
        let (Some(ca), Some(cb)) = (graph.coords_of(ia), graph.coords_of(ib)) else {
            return BifurcationResult::default();
        };
        coords_a.push(ca);
        coords_b.push(cb);
    }

    let epsilon = 1e-8f32;
    let initial_sep = euclidean(coords_a[0], coords_b[0]);
    let last = coords_a.len() - 1;
    let final_separation = euclidean(coords_a[last], coords_b[last]);

    let separation_ratio = if initial_sep > epsilon {
        final_separation / initial_sep
    } else if final_separation > epsilon {
        f32::INFINITY
    } else {
        1.0
    };

    let mut onset_step = None;
    if initial_sep > epsilon {
        let threshold = 1.5 * initial_sep;
        for i in 1..coords_a.len() {
            if euclidean(coords_a[i], coords_b[i]) > threshold {
                onset_step = Some(i);
                break;
            }
        }
    }

    BifurcationResult {
        separation_ratio,
        onset_step,
        final_separation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::build_viable_graph;

    fn graph_of(points: &[(u64, Vec<f32>)]) -> ViableGraph {
        build_viable_graph(points.iter().cloned(), |_| true, 4, false)
    }

    // ── path_geometry: length ──────────────────────────────────────────────

    #[test]
    fn short_path_is_default() {
        let g = graph_of(&[(0, vec![0.0, 0.0])]);
        assert_eq!(path_geometry(&g, &[]), PathGeometry::default());
        assert_eq!(path_geometry(&g, &[0]), PathGeometry::default());
    }

    #[test]
    fn straight_collinear_path_has_low_curvature_and_matches_length() {
        // Nonzero origin: a zero-vector state clamps that step's cosine to
        // 0.0 (see `unknown_id_breaks_the_curvature_chain...` and the
        // `min_adjacent_cosine` doc comment), which would mask the "cosine
        // == 1.0 for a straight path" assertion below.
        let points = vec![
            (0u64, vec![1.0, 0.0]),
            (1u64, vec![2.0, 0.0]),
            (2u64, vec![3.0, 0.0]),
            (3u64, vec![4.0, 0.0]),
        ];
        let g = graph_of(&points);
        let geom = path_geometry(&g, &[0, 1, 2, 3]);
        assert!((geom.length - 3.0).abs() < 1e-5);
        assert!(
            geom.mean_curvature.abs() < 1e-4,
            "straight path should have ~0 curvature, got {}",
            geom.mean_curvature
        );
        assert!((geom.min_adjacent_cosine - 1.0).abs() < 1e-5);
        assert_eq!(geom.n_steps, 3);
    }

    #[test]
    fn zigzag_path_has_high_curvature() {
        // Ping-pong between (0,0) and (1,0): every turn is a full reversal.
        let points = vec![
            (0u64, vec![1.0, 0.0]),
            (1u64, vec![0.0, 0.0]),
            (2u64, vec![1.0, 0.0]),
            (3u64, vec![0.0, 0.0]),
        ];
        let g = graph_of(&points);
        let geom = path_geometry(&g, &[0, 1, 2, 3]);
        assert!(
            (geom.mean_curvature - std::f32::consts::PI).abs() < 1e-3,
            "zigzag path should have curvature near PI, got {}",
            geom.mean_curvature
        );
    }

    #[test]
    fn straight_path_curvature_is_far_below_zigzag() {
        let straight = graph_of(&[
            (0u64, vec![0.0, 0.0]),
            (1u64, vec![1.0, 0.0]),
            (2u64, vec![2.0, 0.0]),
        ]);
        let zigzag = graph_of(&[
            (0u64, vec![1.0, 0.0]),
            (1u64, vec![0.0, 0.0]),
            (2u64, vec![1.0, 0.0]),
        ]);
        let straight_geom = path_geometry(&straight, &[0, 1, 2]);
        let zigzag_geom = path_geometry(&zigzag, &[0, 1, 2]);
        assert!(zigzag_geom.mean_curvature - straight_geom.mean_curvature > 1.0);
    }

    #[test]
    fn unknown_id_breaks_the_curvature_chain_without_panicking() {
        let points = vec![
            (0u64, vec![0.0, 0.0]),
            (1u64, vec![1.0, 0.0]),
            (2u64, vec![2.0, 0.0]),
        ];
        let g = graph_of(&points);
        // id 999 isn't a node: that step contributes no length/cosine and
        // resets the curvature chain, but the call must not panic.
        let geom = path_geometry(&g, &[0, 999, 2]);
        assert_eq!(geom.length, 0.0);
        assert_eq!(geom.n_steps, 0);
        assert_eq!(geom.mean_curvature, 0.0);
    }

    // ── bifurcation_ratio ───────────────────────────────────────────────────

    #[test]
    fn identical_paths_report_zero_bifurcation() {
        let points = vec![
            (0u64, vec![0.0, 0.0]),
            (1u64, vec![1.0, 0.0]),
            (2u64, vec![2.0, 0.0]),
        ];
        let g = graph_of(&points);
        let path = [0u64, 1, 2];
        let r = bifurcation_ratio(&g, &path, &path);
        assert_eq!(r.final_separation, 0.0);
        assert_eq!(r.separation_ratio, 1.0);
        assert_eq!(r.onset_step, None);
    }

    #[test]
    fn immediately_diverging_paths_report_high_separation() {
        // Both paths start close together (separation 0.2) then pull apart
        // sharply -- a nonzero baseline so separation_ratio/onset_step are
        // determinate rather than the initial-sep~0 INFINITY edge case.
        let points = vec![
            (0u64, vec![0.0, 0.1]),
            (1u64, vec![1.0, 0.1]),
            (2u64, vec![2.0, 0.1]),
            (10u64, vec![0.0, -0.1]),
            (11u64, vec![1.0, 1.0]),
            (12u64, vec![2.0, 2.0]),
        ];
        let g = graph_of(&points);
        let a = [0u64, 1, 2];
        let b = [10u64, 11, 12];
        let r = bifurcation_ratio(&g, &a, &b);
        assert!(
            r.separation_ratio > 5.0,
            "expected strong divergence, got ratio {}",
            r.separation_ratio
        );
        assert_eq!(r.onset_step, Some(1));
        assert!(r.final_separation > 1.5);
    }

    #[test]
    fn parallel_paths_never_bifurcate() {
        let points = vec![
            (0u64, vec![0.0, 0.0]),
            (1u64, vec![1.0, 0.0]),
            (2u64, vec![2.0, 0.0]),
            (10u64, vec![0.0, 1.0]),
            (11u64, vec![1.0, 1.0]),
            (12u64, vec![2.0, 1.0]),
        ];
        let g = graph_of(&points);
        let r = bifurcation_ratio(&g, &[0, 1, 2], &[10, 11, 12]);
        assert!((r.separation_ratio - 1.0).abs() < 1e-5);
        assert_eq!(r.onset_step, None);
    }

    #[test]
    fn length_mismatch_and_unknown_ids_are_defensive() {
        let g = graph_of(&[(0u64, vec![0.0, 0.0]), (1u64, vec![1.0, 0.0])]);
        let mismatched = bifurcation_ratio(&g, &[0, 1], &[0]);
        assert_eq!(mismatched, BifurcationResult::default());

        let unknown = bifurcation_ratio(&g, &[0, 999], &[0, 1]);
        assert_eq!(unknown, BifurcationResult::default());
    }
}
