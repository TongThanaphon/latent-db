//! Viable-subset navigation graph over stored records.
//!
//! Inspired by katgpt-rs's Viable Manifold Graph primitive
//! (`crates/katgpt-core/src/viable_manifold_graph.rs`, itself a distillation
//! of arXiv:2206.00106, "Mario Plays on a Manifold"): sample latent points,
//! keep the ones a caller-supplied predicate calls "viable", connect each
//! kept point to its `k` nearest kept neighbors, then navigate that discrete
//! subgraph (A* geodesic / random walk) so every visited point satisfies the
//! predicate by construction.
//!
//! Reimplemented independently for `LatentDb`'s record space rather than
//! copied: the katgpt-rs original also gates node admission on a
//! differential-geometry "pullback volume" `log det(J_f^T J_f)` of a
//! caller-supplied smooth map `f`. That's a genuine tool when `f` is a
//! nonlinear decoder (e.g. a generator mapping latent code to a rendered
//! level), but `LatentDb`'s own dimensionality map (`Projector::project`) is
//! a fixed linear matrix, so its Jacobian -- and therefore that volume field
//! -- is constant everywhere: it couldn't discriminate viable from
//! non-viable points here even in principle. Dropped rather than ported; the
//! predicate alone does the filtering, same as katgpt-rs's own predicate
//! parameter.
//!
//! Graph build is O(n^2) (pairwise distances for kNN) and, like the
//! batch-trained PQ codebooks and the per-mutation-rebuilt `merkle_tree()`
//! (cached across calls, but still a full rebuild the next time it's
//! touched after an `insert`/`remove`), is meant to be rebuilt whenever the
//! underlying record set changes rather than maintained incrementally --
//! fine at this crate's prototype scale.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Discrete navigation graph over a filtered, kNN-connected subset of
/// records.
///
/// Nodes are `LatentDb` record ids; edges are undirected and connect each
/// kept record to its `k` nearest kept neighbors (Euclidean, in the vector
/// space the caller supplied -- typically the DB's approximate decoded
/// vectors). Built via [`build_viable_graph`]; [`Self::geodesic`] and
/// [`Self::random_walk`] can only ever visit ids that passed the predicate.
#[derive(Debug, Clone)]
pub struct ViableGraph {
    ids: Vec<u64>,
    /// Row-major `[n_nodes * dim]` coordinates, one row per `ids[i]`.
    coords: Vec<f32>,
    dim: usize,
    id_to_node: HashMap<u64, u32>,
    /// Sorted, deduplicated adjacency list per node index.
    adjacency: Vec<Vec<u32>>,
}

impl ViableGraph {
    pub fn n_nodes(&self) -> usize {
        self.ids.len()
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().map(|n| n.len()).sum::<usize>() / 2
    }

    /// Whether `id` survived the viability predicate and is a graph node.
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_node.contains_key(&id)
    }

    /// Ids of `id`'s graph neighbors, or an empty `Vec` if `id` isn't a node.
    pub fn neighbors(&self, id: u64) -> Vec<u64> {
        match self.id_to_node.get(&id) {
            Some(&node) => self.adjacency[node as usize]
                .iter()
                .map(|&n| self.ids[n as usize])
                .collect(),
            None => Vec::new(),
        }
    }

    /// `id`'s coordinates in this graph's vector space, or `None` if `id`
    /// isn't a node.
    pub fn coords_of(&self, id: u64) -> Option<&[f32]> {
        self.id_to_node.get(&id).map(|&node| self.node_coords(node))
    }

    fn node_coords(&self, node: u32) -> &[f32] {
        let start = node as usize * self.dim;
        &self.coords[start..start + self.dim]
    }

    /// Shortest path (A*, Euclidean edge weight) from `src` to `dst`.
    ///
    /// Returns `None` if either id isn't a graph node, or if `dst` is
    /// unreachable from `src`. Returns `Some(vec![src])` if `src == dst`.
    pub fn geodesic(&self, src: u64, dst: u64) -> Option<Vec<u64>> {
        let src_node = *self.id_to_node.get(&src)?;
        let dst_node = *self.id_to_node.get(&dst)?;
        if src_node == dst_node {
            return Some(vec![src]);
        }

        let n = self.n_nodes();
        let dst_coords = self.node_coords(dst_node).to_vec();
        let mut g_score = vec![f32::INFINITY; n];
        let mut came_from = vec![u32::MAX; n];
        let mut closed = vec![false; n];
        g_score[src_node as usize] = 0.0;

        let mut open: BinaryHeap<std::cmp::Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        open.push(std::cmp::Reverse((OrdF32(0.0), src_node)));

        while let Some(std::cmp::Reverse((_, cur))) = open.pop() {
            let cur_idx = cur as usize;
            if closed[cur_idx] {
                continue;
            }
            closed[cur_idx] = true;
            if cur == dst_node {
                break;
            }
            let g_cur = g_score[cur_idx];
            let cur_coords = self.node_coords(cur).to_vec();
            for &nxt in &self.adjacency[cur_idx] {
                let nxt_idx = nxt as usize;
                if closed[nxt_idx] {
                    continue;
                }
                let w = euclidean(&cur_coords, self.node_coords(nxt));
                let tentative = g_cur + w;
                if tentative < g_score[nxt_idx] {
                    g_score[nxt_idx] = tentative;
                    came_from[nxt_idx] = cur;
                    let h = euclidean(self.node_coords(nxt), &dst_coords);
                    open.push(std::cmp::Reverse((OrdF32(tentative + h), nxt)));
                }
            }
        }

        if !closed[dst_node as usize] {
            return None;
        }

        let mut path = vec![dst_node];
        let mut cur = dst_node;
        while cur != src_node {
            let pred = came_from[cur as usize];
            path.push(pred);
            cur = pred;
        }
        path.reverse();
        Some(path.into_iter().map(|n| self.ids[n as usize]).collect())
    }

    /// Random walk of `steps` hops starting at `start`, choosing uniformly
    /// among the current node's neighbors at each step. Deterministic given
    /// `seed`. Every visited id satisfies the predicate that built this
    /// graph -- the walk can never step off it.
    ///
    /// Returns a path of length `steps + 1` (including `start`), or an empty
    /// `Vec` if `start` isn't a node. Parks at the current node for any
    /// remaining steps if it has no neighbors (e.g. an isolated point).
    pub fn random_walk(&self, start: u64, steps: usize, seed: u64) -> Vec<u64> {
        let Some(&start_node) = self.id_to_node.get(&start) else {
            return Vec::new();
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let mut path: Vec<u32> = Vec::with_capacity(steps + 1);
        path.push(start_node);
        let mut cur = start_node;
        for _ in 0..steps {
            let neighbors = &self.adjacency[cur as usize];
            if neighbors.is_empty() {
                let last = *path.last().unwrap();
                path.resize(steps + 1, last);
                break;
            }
            cur = neighbors[rng.gen_range(0..neighbors.len())];
            path.push(cur);
        }
        path.into_iter().map(|n| self.ids[n as usize]).collect()
    }

    /// Partition this graph's nodes into `boundary classes`: equivalence
    /// classes under recursive kNN-neighborhood structure.
    ///
    /// A from-scratch reinterpretation of katgpt-rs's bisimulation-refinement
    /// idea (`crates/katgpt-core/src/bisimulation/refine.rs`,
    /// signature-based partition refinement over a labeled transition graph)
    /// for a kNN graph rather than a `(state, op, state')` transition system
    /// -- `ViableGraph` has no operators between records, so a node's
    /// "signature" here is just the sorted multiset of its neighbors'
    /// classes, with no operator-label component. Reimplemented rather than
    /// ported.
    ///
    /// Two nodes end up in the same class when their neighborhoods are
    /// structurally indistinguishable under this refinement: not merely
    /// "same literal neighbor id set", but recursively -- the sorted
    /// multiset of one's neighbors' classes matches the other's exactly
    /// (same classes, same counts), and that condition is in turn checked
    /// recursively on the neighbors (so, e.g., two leaves hanging off the
    /// same hub collapse together even though the hub itself is a distinct
    /// class). Computed by fixed-point signature refinement (1-WL-style
    /// color refinement): start every node in one class, then repeatedly
    /// reclassify each node by the sorted multiset of its neighbors'
    /// current classes (canonicalizing labels each iteration so the
    /// fixed-point check can't oscillate on a relabeling) until the
    /// partition stops changing. Like 1-WL, this is a sound but incomplete
    /// test: it never merges two nodes that are truly structurally
    /// distinct, but on some regular substructures (e.g. two non-isomorphic
    /// neighborhoods that still look alike degree-by-degree at every depth)
    /// it can under-separate them into one class. Purely additive: doesn't
    /// read or affect [`Self::geodesic`], [`Self::random_walk`], or
    /// [`Self::neighbors`].
    pub fn boundary_classes(&self) -> BoundaryClasses {
        let n = self.n_nodes();
        if n == 0 {
            return BoundaryClasses {
                id_to_class: HashMap::new(),
                n_classes: 0,
            };
        }

        let mut current_class: Vec<u32> = vec![0; n];
        let mut new_class: Vec<u32> = vec![0; n];
        let mut signatures: Vec<Vec<u32>> = vec![Vec::new(); n];

        loop {
            for (node, sig) in signatures.iter_mut().enumerate() {
                sig.clear();
                sig.extend(
                    self.adjacency[node]
                        .iter()
                        .map(|&nbr| current_class[nbr as usize]),
                );
                sig.sort_unstable();
            }

            let mut order: Vec<u32> = (0..n as u32).collect();
            order.sort_by(|&a, &b| {
                signatures[a as usize]
                    .cmp(&signatures[b as usize])
                    .then(a.cmp(&b))
            });

            let mut next_class = 0u32;
            for (pos, &node) in order.iter().enumerate() {
                if pos > 0 && signatures[node as usize] != signatures[order[pos - 1] as usize] {
                    next_class += 1;
                }
                new_class[node as usize] = next_class;
            }

            canonicalize_boundary_classes(&mut new_class);

            if new_class == current_class {
                break;
            }
            current_class.copy_from_slice(&new_class);
        }

        let n_classes = current_class
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        let id_to_class = self
            .ids
            .iter()
            .zip(current_class.iter())
            .map(|(&id, &class)| (id, class))
            .collect();

        BoundaryClasses {
            id_to_class,
            n_classes,
        }
    }
}

/// Class id assigned by [`ViableGraph::boundary_classes`].
pub type BoundaryClassId = u32;

/// Partition of a [`ViableGraph`]'s nodes into boundary classes, computed by
/// [`ViableGraph::boundary_classes`].
#[derive(Debug, Clone)]
pub struct BoundaryClasses {
    id_to_class: HashMap<u64, BoundaryClassId>,
    n_classes: usize,
}

impl BoundaryClasses {
    /// Number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// `id`'s class, or `None` if `id` wasn't a node of the graph this
    /// partition was computed from.
    pub fn class_of(&self, id: u64) -> Option<BoundaryClassId> {
        self.id_to_class.get(&id).copied()
    }

    /// Whether `a` and `b` are both graph nodes and share a class.
    pub fn same_class(&self, a: u64, b: u64) -> bool {
        matches!((self.class_of(a), self.class_of(b)), (Some(x), Some(y)) if x == y)
    }

    /// Ids grouped by class: ascending class id, ascending id within each
    /// class.
    pub fn classes(&self) -> Vec<Vec<u64>> {
        let mut groups: Vec<Vec<u64>> = vec![Vec::new(); self.n_classes];
        for (&id, &class) in &self.id_to_class {
            groups[class as usize].push(id);
        }
        for group in &mut groups {
            group.sort_unstable();
        }
        groups
    }
}

/// Renumber class labels so the class of node 0 becomes class 0, the next
/// new class encountered walking node index ascending becomes 1, etc. --
/// makes the label vector invariant under class-id permutation, so
/// [`ViableGraph::boundary_classes`]'s fixed-point check can tell a stable
/// partition from one that's merely had its labels shuffled between
/// iterations (which would otherwise oscillate forever).
fn canonicalize_boundary_classes(labels: &mut [u32]) {
    let mut old_to_new: HashMap<u32, u32> = HashMap::new();
    for &label in labels.iter() {
        let next = old_to_new.len() as u32;
        old_to_new.entry(label).or_insert(next);
    }
    for label in labels.iter_mut() {
        *label = old_to_new[label];
    }
}

pub(crate) fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Total-order `f32` wrapper so `BinaryHeap` can be used as a min-heap by
/// float value without an `ordered-float` dependency. NaN never occurs on
/// this crate's Euclidean distances (finite inputs -> finite sums of
/// squares), so `total_cmp` is a genuine total order here.
#[derive(Copy, Clone, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a [`ViableGraph`] from `(id, vector)` pairs, keeping only those for
/// which `predicate(vector)` is true, then connecting each kept record to its
/// `k_nearest` kept neighbors (Euclidean).
///
/// If `edge_midpoint_check` is set, an edge between `a` and `b` is only added
/// when the segment midpoint `(a + b) / 2` also satisfies `predicate` --
/// this rejects edges that would "shortcut" across a non-viable gap between
/// two viable regions, at the cost of one extra predicate call per candidate
/// edge.
///
/// Usually reached via `LatentDb::build_viable_graph`, which supplies the
/// DB's own (id, approximate decoded vector) pairs in a deterministic order.
pub fn build_viable_graph<F>(
    records: impl Iterator<Item = (u64, Vec<f32>)>,
    predicate: F,
    k_nearest: usize,
    edge_midpoint_check: bool,
) -> ViableGraph
where
    F: Fn(&[f32]) -> bool,
{
    let mut ids: Vec<u64> = Vec::new();
    let mut coords: Vec<f32> = Vec::new();
    let mut dim = 0usize;
    for (id, v) in records {
        if predicate(&v) {
            dim = v.len();
            ids.push(id);
            coords.extend_from_slice(&v);
        }
    }
    let n = ids.len();
    let mut id_to_node = HashMap::with_capacity(n);
    for (i, &id) in ids.iter().enumerate() {
        id_to_node.insert(id, i as u32);
    }

    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    let k = k_nearest.min(n.saturating_sub(1));
    let mut dist_buf: Vec<(f32, u32)> = Vec::with_capacity(n.saturating_sub(1));
    let mut mid_buf = vec![0.0f32; dim];
    for a in 0..n {
        let za = &coords[a * dim..(a + 1) * dim];
        dist_buf.clear();
        for b in 0..n {
            if a == b {
                continue;
            }
            let zb = &coords[b * dim..(b + 1) * dim];
            dist_buf.push((euclidean(za, zb), b as u32));
        }
        dist_buf.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        for &(_, b) in dist_buf.iter().take(k) {
            let zb = &coords[(b as usize) * dim..(b as usize + 1) * dim];
            if edge_midpoint_check {
                for j in 0..dim {
                    mid_buf[j] = 0.5 * (za[j] + zb[j]);
                }
                if !predicate(&mid_buf) {
                    continue;
                }
            }
            adjacency[a].push(b);
            adjacency[b as usize].push(a as u32);
        }
    }
    for adj in &mut adjacency {
        adj.sort_unstable();
        adj.dedup();
    }

    ViableGraph {
        ids,
        coords,
        dim,
        id_to_node,
        adjacency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two disks (radius 1.5 at (-2,0)/(+2,0)) joined by a thin corridor
    /// (|x|<2 AND |y|<0.4) -- the same toy viable set katgpt-rs's own tests
    /// use, reproduced here independently.
    fn two_disk_corridor(z: &[f32]) -> bool {
        let (x, y) = (z[0], z[1]);
        let dl = ((x + 2.0).powi(2) + y * y).sqrt();
        let dr = ((x - 2.0).powi(2) + y * y).sqrt();
        dl < 1.5 || dr < 1.5 || (x.abs() < 2.0 && y.abs() < 0.4)
    }

    fn grid_records(step: f32) -> Vec<(u64, Vec<f32>)> {
        let mut out = Vec::new();
        let mut id = 0u64;
        let mut i = -5.0f32;
        while i <= 5.0 {
            let mut j = -5.0f32;
            while j <= 5.0 {
                out.push((id, vec![i, j]));
                id += 1;
                j += step;
            }
            i += step;
        }
        out
    }

    fn build_corridor_graph() -> ViableGraph {
        build_viable_graph(grid_records(0.25).into_iter(), two_disk_corridor, 4, true)
    }

    #[test]
    fn keeps_only_records_passing_the_predicate() {
        let records = grid_records(0.25);
        let expected_kept = records.iter().filter(|(_, v)| two_disk_corridor(v)).count();
        let g = build_viable_graph(records.into_iter(), two_disk_corridor, 4, false);
        assert_eq!(g.n_nodes(), expected_kept);
        assert!(g.n_edges() > 0);
    }

    #[test]
    fn edge_midpoint_check_rejects_edges_that_shortcut_a_non_viable_gap() {
        // A=(0,0), B=(4,0); predicate keeps points near either but not the
        // gap between them (midpoint (2,0) fails it). With only two nodes
        // and k_nearest=1, each is trivially the other's nearest neighbor --
        // so whether the edge exists depends entirely on the midpoint check.
        let predicate = |z: &[f32]| z[0].abs() < 0.5 || (z[0] - 4.0).abs() < 0.5;
        let records = vec![(0u64, vec![0.0, 0.0]), (1u64, vec![4.0, 0.0])];

        let without_check = build_viable_graph(records.clone().into_iter(), predicate, 1, false);
        assert_eq!(
            without_check.geodesic(0, 1),
            Some(vec![0, 1]),
            "no midpoint check: the sole kNN candidate edge should connect them"
        );

        let with_check = build_viable_graph(records.into_iter(), predicate, 1, true);
        assert!(
            with_check.geodesic(0, 1).is_none(),
            "midpoint (2,0) fails the predicate, so the edge should be rejected"
        );
    }

    #[test]
    fn geodesic_crosses_the_corridor_and_stays_viable() {
        let records = grid_records(0.25);
        let g = build_viable_graph(records.iter().cloned(), two_disk_corridor, 4, true);

        let lookup: HashMap<u64, Vec<f32>> = records.into_iter().collect();
        let src = *lookup
            .iter()
            .filter(|(_, v)| two_disk_corridor(v))
            .min_by(|(_, a), (_, b)| {
                let da = (a[0] + 2.0).powi(2) + a[1].powi(2);
                let db = (b[0] + 2.0).powi(2) + b[1].powi(2);
                da.total_cmp(&db)
            })
            .unwrap()
            .0;
        let dst = *lookup
            .iter()
            .filter(|(_, v)| two_disk_corridor(v))
            .min_by(|(_, a), (_, b)| {
                let da = (a[0] - 2.0).powi(2) + a[1].powi(2);
                let db = (b[0] - 2.0).powi(2) + b[1].powi(2);
                da.total_cmp(&db)
            })
            .unwrap()
            .0;

        let path = g.geodesic(src, dst).expect("corridor connects the disks");
        assert_eq!(path.first(), Some(&src));
        assert_eq!(path.last(), Some(&dst));
        for id in &path {
            assert!(
                two_disk_corridor(&lookup[id]),
                "path visited non-viable id {id}"
            );
        }
        // A shortest path never revisits a node.
        let mut sorted = path.clone();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), before);
    }

    #[test]
    fn geodesic_is_none_for_disconnected_nodes() {
        // Two tight clusters, far apart, each bigger than k_nearest: with
        // k=1 every point's single nearest neighbor stays inside its own
        // cluster, so the two clusters end up as separate components. (Two
        // *lone* points would always end up connected to each other -- kNN
        // has no distance cutoff, just "k nearest", and with only one other
        // node it's trivially the nearest.)
        let records = vec![
            (0u64, vec![0.0, 0.0]),
            (1u64, vec![0.1, 0.0]),
            (2u64, vec![0.0, 0.1]),
            (10u64, vec![100.0, 100.0]),
            (11u64, vec![100.1, 100.0]),
            (12u64, vec![100.0, 100.1]),
        ];
        let g = build_viable_graph(records.into_iter(), |_| true, 1, false);
        assert!(g.geodesic(0, 10).is_none());
    }

    #[test]
    fn geodesic_is_none_for_unknown_ids() {
        let g = build_corridor_graph();
        assert!(g.geodesic(999_999, 0).is_none());
    }

    #[test]
    fn random_walk_never_leaves_the_viable_subset() {
        let records = grid_records(0.25);
        let g = build_viable_graph(records.iter().cloned(), two_disk_corridor, 4, true);
        let lookup: HashMap<u64, Vec<f32>> = records.into_iter().collect();
        let start = *lookup.iter().find(|(_, v)| two_disk_corridor(v)).unwrap().0;

        let walk = g.random_walk(start, 40, 0xC0FFEE);
        assert_eq!(walk.len(), 41);
        assert_eq!(walk.first(), Some(&start));
        for id in &walk {
            assert!(
                two_disk_corridor(&lookup[id]),
                "walk visited non-viable id {id}"
            );
        }
    }

    #[test]
    fn random_walk_is_deterministic_for_a_fixed_seed() {
        let g = build_corridor_graph();
        let start = g.ids[0];
        let walk_a = g.random_walk(start, 30, 42);
        let walk_b = g.random_walk(start, 30, 42);
        assert_eq!(walk_a, walk_b);
    }

    #[test]
    fn random_walk_parks_at_an_isolated_node() {
        let records = vec![(7u64, vec![0.0, 0.0])];
        let g = build_viable_graph(records.into_iter(), |_| true, 4, false);
        let walk = g.random_walk(7, 5, 1);
        assert_eq!(walk, vec![7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn random_walk_on_unknown_start_is_empty() {
        let g = build_corridor_graph();
        assert!(g.random_walk(999_999, 5, 1).is_empty());
    }

    #[test]
    fn boundary_classes_separates_a_hub_from_its_symmetric_leaves() {
        // hub(1) at the origin, two leaves (0, 2) equidistant on either
        // side -- colinear, so each leaf's own nearest-neighbor pass picks
        // the hub (dist 1) over the other leaf (dist 2), and the hub ends
        // up wired to both leaves regardless of which single leaf its own
        // pass happens to pick (edges are added unconditionally from
        // whichever side's pass selects them).
        let records = vec![
            (0u64, vec![-1.0, 0.0]),
            (1u64, vec![0.0, 0.0]),
            (2u64, vec![1.0, 0.0]),
        ];
        let g = build_viable_graph(records.into_iter(), |_| true, 1, false);
        assert_eq!(g.neighbors(1).len(), 2, "hub should connect to both leaves");

        let classes = g.boundary_classes();
        assert_eq!(classes.n_classes(), 2);
        assert!(
            classes.same_class(0, 2),
            "the two leaves have identical neighbor sets ({{hub}}) and must collapse"
        );
        assert!(
            !classes.same_class(0, 1),
            "the hub has a structurally distinct neighborhood (degree 2 vs 1)"
        );
    }

    #[test]
    fn boundary_classes_collapse_across_isomorphic_but_disconnected_stars() {
        // Two copies of the same hub-and-leaves shape, far enough apart
        // that each node's nearest neighbor stays inside its own copy. A
        // naive "same literal neighbor id set" partitioning would leave
        // every node in its own singleton class here -- no two nodes share
        // a neighbor id across components -- so this exercises the
        // fixed-point refinement recognizing that group A's hub and group
        // B's hub play the same structural role despite sharing no
        // neighbors at all.
        let records = vec![
            (10u64, vec![-1.0, 0.0]),
            (11u64, vec![0.0, 0.0]),
            (12u64, vec![1.0, 0.0]),
            (20u64, vec![999.0, 0.0]),
            (21u64, vec![1000.0, 0.0]),
            (22u64, vec![1001.0, 0.0]),
        ];
        let g = build_viable_graph(records.into_iter(), |_| true, 1, false);

        let classes = g.boundary_classes();
        assert_eq!(
            classes.n_classes(),
            2,
            "just {{hub, leaf}} structural roles"
        );
        assert!(classes.same_class(11, 21), "hub A ~ hub B");
        assert!(classes.same_class(10, 20), "leaf A ~ leaf B");
        assert!(classes.same_class(10, 12), "leaves within A collapse");
        assert!(classes.same_class(12, 22), "leaves across A/B collapse");
        assert!(!classes.same_class(11, 10), "hub role != leaf role");

        let groups = classes.classes();
        assert_eq!(groups.len(), 2);
        let hub_group = groups.iter().find(|group| group.contains(&11)).unwrap();
        assert_eq!(hub_group, &vec![11, 21]);
        let leaf_group = groups.iter().find(|group| group.contains(&10)).unwrap();
        assert_eq!(leaf_group, &vec![10, 12, 20, 22]);
    }

    #[test]
    fn boundary_classes_on_empty_graph_has_no_classes() {
        let g = build_viable_graph(
            std::iter::empty::<(u64, Vec<f32>)>(),
            |_: &[f32]| true,
            4,
            false,
        );
        let classes = g.boundary_classes();
        assert_eq!(classes.n_classes(), 0);
        assert_eq!(classes.class_of(0), None);
        assert!(classes.classes().is_empty());
    }

    #[test]
    fn boundary_classes_of_unknown_id_is_none() {
        let g = build_corridor_graph();
        assert_eq!(g.boundary_classes().class_of(999_999), None);
    }

    #[test]
    fn boundary_classes_is_deterministic() {
        let g = build_corridor_graph();
        let a = g.boundary_classes();
        let b = g.boundary_classes();
        assert_eq!(a.n_classes(), b.n_classes());
        for &id in &g.ids {
            assert_eq!(a.class_of(id), b.class_of(id));
        }
    }
}
