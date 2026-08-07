//! Bandit-guided region exploration over stored record embeddings.
//!
//! Ported from katgpt-rs's `manifold_bandit::LatentTaskTree`
//! (`crates/katgpt-core/src/manifold_bandit/mod.rs`, distilled from
//! McKenzie, Hansen, Wang, *Manifold Bandits: Bayesian Curriculum Learning
//! over the Latent Geometry of Large Language Models*, arXiv:2606.19750): a
//! frozen hierarchical clustering of an "arm" space -- here, currently-stored
//! record embeddings -- plus top-down Thompson-sampling descent and per-arm
//! Beta belief updates. [`RegionTree::sample`] descends root -> leaf, drawing
//! a Beta sample from each child's posterior at every branch and following
//! the largest draw; [`RegionTree::observe`] runs a predict-update Bayesian
//! filter on the sampled leaf's belief, then re-aggregates every ancestor's
//! belief bottom-up by Empirical Bayes evidence pooling. Repeated
//! sample/observe cycles bias future descents toward regions that have paid
//! off, without ever retraining an embedding or index. Reached via
//! `LatentDb::build_region_tree`.
//!
//! Three deliberate departures from the katgpt-rs original:
//! - **Construction.** katgpt-rs's real pipeline is PCA -> UMAP-substitute 2D
//!   embed -> Chart Test -> adaptive-epsilon DBSCAN, recursed per cluster.
//!   This port keeps the PCA step (power iteration + Hotelling deflation,
//!   same technique as the source) but clusters with this crate's own seeded
//!   k-means ([`crate::pq::kmeans`], already used to train the PQ codebooks
//!   and centroid index) instead of porting DBSCAN/Chart-Test/UMAP -- a fixed
//!   branching factor per level rather than density-discovered cluster
//!   counts. Simpler, and reuses infrastructure this crate already has and
//!   tests, at the cost of not discovering irregular cluster counts/shapes
//!   the way DBSCAN would -- a reasonable first version per the issue's own
//!   scoping note, with finer reward-shaping deferred to a follow-up.
//! - **No BLAKE3 commitment, no R279 phase gate.** katgpt-rs's tree is
//!   BLAKE3-committable (freeze/thaw integrity) and has an optional N>=d
//!   observation-count gate on Empirical Bayes aggregation. Neither is part
//!   of this issue's acceptance criteria; dropped to keep the first version
//!   scoped to build + sample + observe.
//! - **Zero-alloc is not a goal here.** katgpt-rs's `sample`/`observe` are
//!   allocation-free (a stack-allocated `ArmPath`). This port uses a plain
//!   `Vec<usize>` per-arm path lookup instead -- this module isn't on any of
//!   `LatentDb`'s allocation-tracked hot paths (see `alloc.rs`), so the
//!   simpler representation isn't a meaningful cost here.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for [`RegionTree`] construction.
#[derive(Clone, Copy, Debug)]
pub struct RegionTreeConfig {
    /// Reduce embeddings to this dimensionality (via PCA) before clustering
    /// at each recursion level, when the current subset's dimension exceeds
    /// it. Default: 8.
    pub pca_dim: usize,
    /// Number of clusters to split a subset into at each recursion level
    /// (k for the recursive k-means split). Default: 4.
    pub branching_factor: usize,
    /// Stop recursing at this depth, turning the remaining subset into a
    /// flat group of individual leaf arms. Default: 4.
    pub max_depth: usize,
    /// Stop recursing once a subset has this many records or fewer,
    /// regardless of depth. Default: 4.
    pub min_leaf_size: usize,
    /// Drift rate for each leaf's non-stationary belief filter (see
    /// [`RegionBelief::predict`]), clamped to `[0.0, 0.999]`; `0.0` (the
    /// default) means stationary -- no drift, a plain Beta-Bernoulli
    /// posterior.
    pub filter_drift_rate: f32,
    /// Lloyd's-algorithm iterations for the k-means split at each recursion
    /// level (passed straight through to [`crate::pq::kmeans`]). Default: 15,
    /// the same default this crate's other `kmeans` callers
    /// (`PqCodec::train`, `CentroidIndex::train`) use.
    pub kmeans_iterations: usize,
    /// Seed for the deterministic PCA/k-means construction pipeline.
    /// Default: 42.
    pub seed: u64,
}

impl Default for RegionTreeConfig {
    fn default() -> Self {
        RegionTreeConfig {
            pca_dim: 8,
            branching_factor: 4,
            max_depth: 4,
            min_leaf_size: 4,
            filter_drift_rate: 0.0,
            kmeans_iterations: 15,
            seed: 42,
        }
    }
}

/// Per-leaf non-stationary belief: a Beta(alpha, beta) posterior with an
/// optional predict-step drift toward uniform Beta(1, 1) between
/// observations (models belief decay in a non-stationary environment; a
/// no-op when `drift_rate == 0.0`).
#[derive(Clone, Copy, Debug)]
struct RegionBelief {
    alpha: f32,
    beta: f32,
    drift_rate: f32,
    last_step: u64,
}

impl RegionBelief {
    fn new(drift_rate: f32) -> Self {
        RegionBelief {
            alpha: 1.0,
            beta: 1.0,
            drift_rate: drift_rate.clamp(0.0, 0.999),
            last_step: 0,
        }
    }

    /// Decay belief toward uniform Beta(1, 1) proportional to elapsed steps
    /// since the last observation. No-op with `drift_rate == 0.0` or
    /// `elapsed == 0`.
    fn predict(&mut self, current_step: u64) {
        let elapsed = current_step.saturating_sub(self.last_step);
        if elapsed == 0 || self.drift_rate <= 0.0 {
            return;
        }
        let decay = (1.0 - self.drift_rate).powi(elapsed as i32);
        self.alpha = self.alpha * decay + (1.0 - decay);
        self.beta = self.beta * decay + (1.0 - decay);
    }

    /// Beta-Bernoulli conjugate update: `reward` is clamped to `[0, 1]`,
    /// `alpha += reward`, `beta += (1 - reward)`.
    fn update(&mut self, reward: f32, current_step: u64) {
        let r = reward.clamp(0.0, 1.0);
        self.alpha += r;
        self.beta += 1.0 - r;
        self.last_step = current_step;
    }
}

/// A node in the [`RegionTree`]. Internal nodes carry the Empirical Bayes
/// aggregate Beta(alpha, beta) of their subtree; leaves carry one record's
/// own [`RegionBelief`].
#[derive(Debug)]
enum TreeNode {
    Internal {
        children: Vec<TreeNode>,
        alpha: f32,
        beta: f32,
    },
    Leaf {
        record_id: u64,
        belief: RegionBelief,
    },
}

impl TreeNode {
    fn beta_params(&self) -> (f32, f32) {
        match self {
            TreeNode::Internal { alpha, beta, .. } => (*alpha, *beta),
            TreeNode::Leaf { belief, .. } => (belief.alpha, belief.beta),
        }
    }

    fn leaf(record_id: u64, drift_rate: f32) -> Self {
        TreeNode::Leaf {
            record_id,
            belief: RegionBelief::new(drift_rate),
        }
    }

    /// Build an internal node whose Empirical Bayes aggregate pools the
    /// **evidence** (observed successes/failures) from its children:
    /// `parent_alpha = 1 + sum(child_alpha - 1)`, `parent_beta = 1 +
    /// sum(child_beta - 1)`. Starts at Beta(1, 1) when every child is
    /// uniform (high variance -> explores), and concentrates as children
    /// accumulate evidence -- without the pseudo-count dilution a plain sum
    /// or mean would introduce.
    fn internal(children: Vec<TreeNode>) -> Self {
        let (alpha, beta) = aggregate(&children);
        TreeNode::Internal {
            children,
            alpha,
            beta,
        }
    }
}

/// Evidence-pooled Beta(alpha, beta) aggregate over `children` (see
/// [`TreeNode::internal`]'s doc comment for the formula).
fn aggregate(children: &[TreeNode]) -> (f32, f32) {
    let n = children.len() as f32;
    let (a, b) = children.iter().fold((0.0f32, 0.0f32), |(sa, sb), c| {
        let (ca, cb) = c.beta_params();
        (sa + ca, sb + cb)
    });
    ((a - n + 1.0).max(1.0), (b - n + 1.0).max(1.0))
}

/// A frozen hierarchical clustering of currently-stored record embeddings
/// (built by [`RegionTree::build`] / `LatentDb::build_region_tree`) plus its
/// mutable Thompson-sampling sampler state.
///
/// The tree topology is fixed at construction time; only the Beta posteriors
/// drift as [`Self::observe`] is called. Rebuild (don't mutate topology) when
/// the underlying record set changes meaningfully -- same "rebuild rather
/// than maintain incrementally" contract as `ViableGraph`/`build_viable_graph`.
#[derive(Debug)]
pub struct RegionTree {
    root: TreeNode,
    /// record id -> root-to-leaf child-index path, built once at
    /// construction so `sample`/`observe` don't need to search the tree.
    arm_paths: HashMap<u64, Vec<usize>>,
}

impl RegionTree {
    /// Build a region tree from `records` (id, embedding) pairs: PCA to
    /// `config.pca_dim` (skipped if the subset's dimension is already at or
    /// below that), k-means split into `config.branching_factor` clusters,
    /// recurse into each non-empty cluster, until a subset has
    /// `config.min_leaf_size` or fewer records or `config.max_depth` is
    /// reached -- at which point it becomes a flat group of individual leaf
    /// arms, one per record (a lone record becomes a bare leaf).
    ///
    /// # Panics
    /// Panics if `records` is empty -- there's nothing to build a tree over.
    pub fn build(records: &[(u64, Vec<f32>)], config: RegionTreeConfig) -> Self {
        assert!(
            !records.is_empty(),
            "RegionTree::build: need at least one record"
        );
        let root = build_recursive(records, &config, 0);
        let mut arm_paths = HashMap::new();
        let mut path = Vec::new();
        collect_arm_paths(&root, &mut path, &mut arm_paths);
        RegionTree { root, arm_paths }
    }

    /// Number of arms (leaves, one per record) in the tree.
    pub fn num_arms(&self) -> usize {
        self.arm_paths.len()
    }

    /// Thompson-sample a leaf record id by descending the tree: at each
    /// internal node, draw a Beta sample from every child's posterior and
    /// descend into the child with the largest draw. Deterministic given
    /// `seed` (a fresh `StdRng` is seeded per call, matching
    /// `ViableGraph::random_walk`'s own per-call seeding convention -- pass a
    /// different seed, e.g. an increasing step counter, across repeated
    /// sample/observe cycles).
    pub fn sample(&self, seed: u64) -> u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut node = &self.root;
        loop {
            match node {
                TreeNode::Leaf { record_id, .. } => return *record_id,
                TreeNode::Internal { children, .. } => {
                    debug_assert!(
                        !children.is_empty(),
                        "region_tree: internal node with no children"
                    );
                    let mut best_idx = 0usize;
                    let mut best_sample = f32::NEG_INFINITY;
                    for (i, child) in children.iter().enumerate() {
                        let (a, b) = child.beta_params();
                        let s = sample_beta(a, b, &mut rng);
                        if s > best_sample {
                            best_sample = s;
                            best_idx = i;
                        }
                    }
                    node = &children[best_idx];
                }
            }
        }
    }

    /// Observe a reward (clamped to `[0, 1]`) on `record_id`: runs the
    /// predict-update belief filter on that leaf, then recomputes every
    /// ancestor's Empirical Bayes aggregate on the way back up.
    ///
    /// No-op if `record_id` isn't an arm of this tree (e.g. a record removed
    /// since the tree was built) -- mirrors `LatentDb::remove`'s own
    /// silent-no-op-on-unknown-id convention rather than panicking.
    pub fn observe(&mut self, record_id: u64, reward: f32, step: u64) {
        let Some(path) = self.arm_paths.get(&record_id).cloned() else {
            return;
        };
        Self::observe_recursive(&mut self.root, &path, reward, step);
    }

    fn observe_recursive(node: &mut TreeNode, path: &[usize], reward: f32, step: u64) {
        match node {
            TreeNode::Leaf { belief, .. } => {
                belief.predict(step);
                belief.update(reward, step);
            }
            TreeNode::Internal {
                children,
                alpha,
                beta,
            } => {
                Self::observe_recursive(&mut children[path[0]], &path[1..], reward, step);
                let (a, b) = aggregate(children);
                *alpha = a;
                *beta = b;
            }
        }
    }

    /// Current Beta(alpha, beta) belief for `record_id`'s leaf, or `None` if
    /// it isn't an arm of this tree. Diagnostic -- not needed by
    /// `sample`/`observe`, which walk the tree directly.
    pub fn leaf_belief(&self, record_id: u64) -> Option<(f32, f32)> {
        Some(self.leaf_node(record_id)?.beta_params())
    }

    /// Every record id sharing `record_id`'s immediate parent region (the
    /// group of leaves `sample()` would have narrowed down to, one branch
    /// before landing on `record_id` itself) -- including `record_id`. A
    /// lone record with no siblings returns just itself. `None` if
    /// `record_id` isn't an arm of this tree.
    pub fn region_members(&self, record_id: u64) -> Option<Vec<u64>> {
        let path = self.arm_paths.get(&record_id)?;
        let Some(parent_path) = path.split_last().map(|(_, rest)| rest) else {
            return Some(vec![record_id]);
        };
        let mut node = &self.root;
        for &idx in parent_path {
            let TreeNode::Internal { children, .. } = node else {
                // Invariant: `arm_paths` is derived from this exact tree at
                // construction time, so every non-final path index must
                // resolve to an `Internal` node. Checked loudly in debug;
                // in release, degrade to "just this record" rather than
                // panic or index out of bounds.
                debug_assert!(false, "region_tree: path continues past a leaf");
                return Some(vec![record_id]);
            };
            node = &children[idx];
        }
        let mut members = Vec::new();
        collect_leaf_ids(node, &mut members);
        members.sort_unstable();
        Some(members)
    }

    /// Root-to-leaf walk shared by [`Self::leaf_belief`] -- `None` if
    /// `record_id` isn't an arm of this tree.
    fn leaf_node(&self, record_id: u64) -> Option<&TreeNode> {
        let path = self.arm_paths.get(&record_id)?;
        let mut node = &self.root;
        for &idx in path {
            let TreeNode::Internal { children, .. } = node else {
                debug_assert!(false, "region_tree: path continues past a leaf");
                return None;
            };
            node = &children[idx];
        }
        Some(node)
    }
}

fn collect_leaf_ids(node: &TreeNode, out: &mut Vec<u64>) {
    match node {
        TreeNode::Leaf { record_id, .. } => out.push(*record_id),
        TreeNode::Internal { children, .. } => {
            for child in children {
                collect_leaf_ids(child, out);
            }
        }
    }
}

fn collect_arm_paths(node: &TreeNode, path: &mut Vec<usize>, out: &mut HashMap<u64, Vec<usize>>) {
    match node {
        TreeNode::Leaf { record_id, .. } => {
            out.insert(*record_id, path.clone());
        }
        TreeNode::Internal { children, .. } => {
            for (i, child) in children.iter().enumerate() {
                path.push(i);
                collect_arm_paths(child, path, out);
                path.pop();
            }
        }
    }
}

// ── Build pipeline: PCA -> recursive k-means ────────────────────────────

fn build_recursive(
    records: &[(u64, Vec<f32>)],
    config: &RegionTreeConfig,
    depth: usize,
) -> TreeNode {
    let n = records.len();
    let drift = config.filter_drift_rate;

    if n <= config.min_leaf_size || depth >= config.max_depth {
        return make_leaf_or_group(records, drift);
    }

    let dim = records[0].1.len();
    let pca_target = config.pca_dim.min(dim).min(n.saturating_sub(1)).max(1);
    let projected: Vec<Vec<f32>> = if dim > pca_target {
        pca_reduce(records, pca_target, config.seed.wrapping_add(depth as u64))
    } else {
        records.iter().map(|(_, v)| v.clone()).collect()
    };

    let centroids = crate::pq::kmeans(
        &projected,
        config.branching_factor,
        config.kmeans_iterations,
        config
            .seed
            .wrapping_add(depth as u64)
            .wrapping_add(1_000_003),
    );

    let mut buckets: Vec<Vec<(u64, Vec<f32>)>> = vec![Vec::new(); centroids.len()];
    for (i, (id, v)) in records.iter().enumerate() {
        let p = &projected[i];
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (c_idx, c) in centroids.iter().enumerate() {
            let d = crate::pq::sq_dist(p, c);
            if d < best_dist {
                best_dist = d;
                best = c_idx;
            }
        }
        buckets[best].push((*id, v.clone()));
    }

    let children: Vec<TreeNode> = buckets
        .into_iter()
        .filter(|b| !b.is_empty())
        .map(|b| build_recursive(&b, config, depth + 1))
        .collect();

    if children.len() <= 1 {
        // k-means degenerated to (effectively) one cluster -- no meaningful
        // subdivision at this level, so stop here rather than recursing
        // forever on an identical subset.
        return make_leaf_or_group(records, drift);
    }

    TreeNode::internal(children)
}

/// Turn a subset into a leaf (single record) or a flat group of individual
/// leaf arms (multiple records that recursion stopped subdividing).
fn make_leaf_or_group(records: &[(u64, Vec<f32>)], drift_rate: f32) -> TreeNode {
    if records.len() == 1 {
        TreeNode::leaf(records[0].0, drift_rate)
    } else {
        TreeNode::internal(
            records
                .iter()
                .map(|(id, _)| TreeNode::leaf(*id, drift_rate))
                .collect(),
        )
    }
}

/// Maximum power-iteration steps before giving up on an eigenvector.
const PCA_MAX_ITERS: usize = 200;
/// Convergence threshold for power iteration (1 - cosine similarity).
const PCA_CONVERGENCE: f32 = 1e-6;

/// Reduce `records`' embeddings (N vectors of dimension D) to `n_components`
/// dimensions via PCA: center, form the D x D covariance matrix, extract the
/// top `n_components` eigenvectors by power iteration with Hotelling
/// deflation, and project. Deterministic given `seed` (seeds each
/// eigenvector's initial direction).
///
/// Takes `records` (rather than a pre-extracted `&[Vec<f32>]`) so the caller
/// doesn't need to clone every embedding just to hand this function a
/// same-shaped slice it would only read once.
fn pca_reduce(records: &[(u64, Vec<f32>)], n_components: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = records.len();
    let dim = records[0].1.len();

    let mut mean = vec![0.0f32; dim];
    for (_, v) in records {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in &mut mean {
        *m *= inv_n;
    }
    let centered: Vec<Vec<f32>> = records
        .iter()
        .map(|(_, v)| v.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();

    let denom = (n as f32 - 1.0).max(1.0);
    let mut cov = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut s = 0.0f32;
            for row in &centered {
                s += row[i] * row[j];
            }
            cov[i * dim + j] = s / denom;
            cov[j * dim + i] = cov[i * dim + j];
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(n_components);
    for _ in 0..n_components {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        normalize(&mut v);

        for _ in 0..PCA_MAX_ITERS {
            let mut new_v = vec![0.0f32; dim];
            for i in 0..dim {
                new_v[i] = crate::simd::simd_dot_f32(&cov[i * dim..(i + 1) * dim], &v);
            }
            normalize(&mut new_v);
            let dot = crate::simd::simd_dot_f32(&v, &new_v);
            let converged = (dot.abs() - 1.0).abs() < PCA_CONVERGENCE;
            v = new_v;
            if converged {
                break;
            }
        }

        let mut cv = vec![0.0f32; dim];
        for i in 0..dim {
            cv[i] = crate::simd::simd_dot_f32(&cov[i * dim..(i + 1) * dim], &v);
        }
        let lambda = crate::simd::simd_dot_f32(&v, &cv);

        for i in 0..dim {
            for j in 0..dim {
                cov[i * dim + j] -= lambda * v[i] * v[j];
            }
        }

        eigenvectors.push(v);
    }

    centered
        .iter()
        .map(|row| {
            eigenvectors
                .iter()
                .map(|ev| crate::simd::simd_dot_f32(row, ev))
                .collect()
        })
        .collect()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ── Beta sampling -- Gamma-ratio method (Marsaglia-Tsang + Box-Muller) ──
//
// Ported from katgpt-rs's `manifold_bandit::sample_beta`/`sample_gamma`
// (same module doc rationale: Jöhnk's algorithm, used elsewhere in this
// codebase's other Beta samplers, has acceptance rate that collapses for
// the large alpha/beta a heavily-observed posterior reaches; the Gamma-ratio
// identity `Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))` stays >90%
// per iteration regardless). Rebuilt against `rand::Rng`/`StdRng` (this
// crate's own RNG convention) instead of katgpt-rs's `fastrand`.

fn sample_beta(alpha: f32, beta: f32, rng: &mut StdRng) -> f32 {
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.gen::<f32>();
    }
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    x / (x + y)
}

fn sample_gamma(shape: f32, rng: &mut StdRng) -> f32 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u = rng.gen::<f32>().max(f32::EPSILON);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.gen::<f32>().max(f32::EPSILON);
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

fn standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().max(f32::EPSILON);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_records(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n as u64)
            .map(|id| {
                let v = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                (id, v)
            })
            .collect()
    }

    #[test]
    fn build_covers_every_record_as_an_arm() {
        let records = synthetic_records(40, 12, 1);
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        assert_eq!(tree.num_arms(), records.len());
        for (id, _) in &records {
            assert!(tree.leaf_belief(*id).is_some());
        }
    }

    #[test]
    #[should_panic(expected = "need at least one record")]
    fn build_panics_on_empty_input() {
        RegionTree::build(&[], RegionTreeConfig::default());
    }

    #[test]
    fn single_record_builds_a_bare_leaf() {
        let records = vec![(7u64, vec![1.0, 2.0, 3.0])];
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        assert_eq!(tree.num_arms(), 1);
        assert_eq!(tree.sample(0), 7);
        assert_eq!(tree.leaf_belief(7), Some((1.0, 1.0)));
    }

    #[test]
    fn sample_always_returns_a_known_arm() {
        let records = synthetic_records(50, 8, 2);
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        let ids: std::collections::HashSet<u64> = records.iter().map(|(id, _)| *id).collect();
        for seed in 0..200u64 {
            assert!(ids.contains(&tree.sample(seed)));
        }
    }

    #[test]
    fn sample_is_deterministic_for_a_fixed_seed() {
        let records = synthetic_records(50, 8, 3);
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        assert_eq!(tree.sample(42), tree.sample(42));
    }

    #[test]
    fn observe_on_unknown_arm_is_a_silent_no_op() {
        let records = synthetic_records(20, 8, 4);
        let mut tree = RegionTree::build(&records, RegionTreeConfig::default());
        let before: Vec<(f32, f32)> = records
            .iter()
            .map(|(id, _)| tree.leaf_belief(*id).unwrap())
            .collect();
        tree.observe(999_999, 1.0, 1);
        let after: Vec<(f32, f32)> = records
            .iter()
            .map(|(id, _)| tree.leaf_belief(*id).unwrap())
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn observe_raises_the_observed_leafs_alpha_and_its_ancestors() {
        let records = synthetic_records(30, 8, 5);
        let config = RegionTreeConfig {
            branching_factor: 3,
            max_depth: 2,
            min_leaf_size: 2,
            ..RegionTreeConfig::default()
        };
        let mut tree = RegionTree::build(&records, config);
        let arm = records[0].0;
        let (alpha_before, _) = tree.leaf_belief(arm).unwrap();
        let (root_alpha_before, _) = tree.root.beta_params();

        for step in 0..20u64 {
            tree.observe(arm, 1.0, step);
        }

        let (alpha_after, _) = tree.leaf_belief(arm).unwrap();
        let (root_alpha_after, _) = tree.root.beta_params();
        assert!(alpha_after > alpha_before + 15.0);
        assert!(
            root_alpha_after > root_alpha_before,
            "repeated reward on one leaf should raise the root's aggregate alpha too"
        );
    }

    /// Issue #8's acceptance criterion: a synthetic multi-cluster corpus with
    /// a simulated reward signal (one cluster consistently more rewarding
    /// than the others) should, after repeated sample/observe cycles, be
    /// sampled from that cluster more often than a *measured* no-learning
    /// baseline would -- a second tree, same topology, same seeds, but
    /// `observe()` never called. A measured baseline (rather than an assumed
    /// `1 / n_clusters`) holds even if the tree's own leaf groups end up
    /// unevenly sized, so this can't silently pass because of a lucky/unlucky
    /// prior split instead of genuine learning.
    ///
    /// Each cluster sits on its own dedicated axis (`v[cluster] += 10.0`,
    /// every other dimension independently jittered) so the three clusters
    /// are linearly separable along three different directions -- unlike a
    /// corpus that only varies one shared dimension while broadcasting the
    /// rest, which collapses PCA's covariance to near-rank-1 and can leave
    /// k-means unable to recover the intended clusters at all.
    #[test]
    fn repeated_sample_observe_converges_toward_the_more_rewarding_cluster() {
        let per_cluster = 12;
        let dim = 8;
        let clusters = [0usize, 1, 2];
        let mut jitter_rng = StdRng::seed_from_u64(777);
        let mut records: Vec<(u64, Vec<f32>)> = Vec::new();
        let mut cluster_of: HashMap<u64, usize> = HashMap::new();
        let mut next_id = 0u64;
        for &cluster in &clusters {
            for _ in 0..per_cluster {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| jitter_rng.gen_range(-0.05f32..0.05))
                    .collect();
                v[cluster] += 10.0;
                records.push((next_id, v));
                cluster_of.insert(next_id, cluster);
                next_id += 1;
            }
        }

        let config = RegionTreeConfig {
            pca_dim: 4,
            branching_factor: 3,
            max_depth: 1,
            min_leaf_size: 1,
            filter_drift_rate: 0.0,
            kmeans_iterations: 15,
            seed: 11,
        };

        let reward_prob = |cluster: usize| -> f32 {
            match cluster {
                0 => 0.9,
                1 => 0.5,
                _ => 0.1,
            }
        };
        let n_trials: u64 = 900;
        let warmup = n_trials / 2;

        let late_cluster0_rate = |tree: &mut RegionTree, with_learning: bool| -> f32 {
            let mut reward_rng = StdRng::seed_from_u64(999);
            let mut hits = 0u32;
            for step in 0..n_trials {
                let arm = tree.sample(step);
                let cluster = cluster_of[&arm];
                if with_learning {
                    let reward = if reward_rng.gen::<f32>() < reward_prob(cluster) {
                        1.0
                    } else {
                        0.0
                    };
                    tree.observe(arm, reward, step);
                }
                if step >= warmup && cluster == 0 {
                    hits += 1;
                }
            }
            hits as f32 / (n_trials - warmup) as f32
        };

        let mut baseline_tree = RegionTree::build(&records, config);
        let baseline_rate = late_cluster0_rate(&mut baseline_tree, false);

        let mut tree = RegionTree::build(&records, config);
        assert_eq!(tree.num_arms(), records.len());
        let learned_rate = late_cluster0_rate(&mut tree, true);

        assert!(
            learned_rate > baseline_rate + 0.2,
            "expected sample/observe learning to beat the tree's own measured no-learning \
             baseline by a solid margin: learned_rate={learned_rate}, baseline_rate={baseline_rate}"
        );
    }

    #[test]
    fn drift_decays_a_stale_belief_toward_uniform() {
        // reward=0.5 moves both alpha and beta symmetrically away from
        // Beta(1, 1) (alpha=beta=1.5), so predict's decay-toward-uniform is
        // visible on both sides -- reward=1.0/0.0 would leave one of the two
        // already sitting at 1.0 with nowhere further to decay.
        let mut belief = RegionBelief::new(0.5);
        belief.update(0.5, 0);
        let (stale_alpha, stale_beta) = (belief.alpha, belief.beta);
        belief.predict(20);
        assert!(belief.alpha < stale_alpha);
        assert!(belief.beta < stale_beta);
        assert!((belief.alpha - 1.0).abs() < 1e-3);
        assert!((belief.beta - 1.0).abs() < 1e-3);
    }

    #[test]
    fn zero_drift_rate_leaves_a_stale_belief_unchanged() {
        let mut belief = RegionBelief::new(0.0);
        belief.update(1.0, 0);
        let before = (belief.alpha, belief.beta);
        belief.predict(100);
        assert_eq!((belief.alpha, belief.beta), before);
    }

    #[test]
    fn region_members_includes_the_record_itself_and_its_group() {
        let records = synthetic_records(30, 8, 6);
        let config = RegionTreeConfig {
            branching_factor: 3,
            max_depth: 2,
            min_leaf_size: 2,
            ..RegionTreeConfig::default()
        };
        let tree = RegionTree::build(&records, config);
        let arm = records[0].0;
        let members = tree.region_members(arm).unwrap();
        assert!(members.contains(&arm));
        assert!(
            members.len() >= 2,
            "min_leaf_size=2 rules out a bare singleton region here"
        );
    }

    #[test]
    fn region_members_of_a_lone_record_is_just_itself() {
        let records = vec![(7u64, vec![1.0, 2.0, 3.0])];
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        assert_eq!(tree.region_members(7), Some(vec![7]));
    }

    #[test]
    fn region_members_of_unknown_id_is_none() {
        let records = synthetic_records(20, 8, 7);
        let tree = RegionTree::build(&records, RegionTreeConfig::default());
        assert_eq!(tree.region_members(999_999), None);
    }
}
