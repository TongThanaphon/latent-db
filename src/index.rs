//! Centroid-based approximate nearest-neighbour index.
//!
//! Inspired by katgpt-rs's "Schema Centroid" (per-class embedding centroids
//! for informed KG entity init). Here we reuse the same idea as an IVF-style
//! index: partition the (projected, low-dimensional) embedding space into
//! `k` centroids, bucket every record under its nearest centroid, and at
//! query time only score records whose bucket-centroid is close to the
//! query -- avoiding a full linear scan over every stored vector.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::projector::Projector;

#[derive(Serialize, Deserialize)]
pub struct CentroidIndex {
    centroids: Vec<Vec<f32>>,
    /// centroid index -> record ids assigned to it
    buckets: HashMap<usize, Vec<u64>>,
    /// record id -> centroid index (so we can move/remove records)
    assignment: HashMap<u64, usize>,
}

impl CentroidIndex {
    /// Build centroids from a batch of *projected* (low-dim) training
    /// vectors using the same simple k-means as the PQ codec.
    pub fn train(projected_training: &[Vec<f32>], n_centroids: usize, iterations: usize, seed: u64) -> Self {
        let centroids = crate::pq::kmeans(projected_training, n_centroids, iterations, seed);
        CentroidIndex { centroids, buckets: HashMap::new(), assignment: HashMap::new() }
    }

    fn nearest_centroid(&self, projected: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (idx, c) in self.centroids.iter().enumerate() {
            let d: f32 = c.iter().zip(projected.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
            if d < best_dist {
                best_dist = d;
                best = idx;
            }
        }
        best
    }

    /// Assign a record id to its nearest centroid bucket.
    pub fn insert(&mut self, id: u64, projected: &[f32]) {
        let c = self.nearest_centroid(projected);
        self.buckets.entry(c).or_default().push(id);
        self.assignment.insert(id, c);
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(c) = self.assignment.remove(&id) {
            if let Some(bucket) = self.buckets.get_mut(&c) {
                bucket.retain(|&x| x != id);
            }
        }
    }

    /// Return candidate record ids from the `nprobe` centroids closest to
    /// the query. This is the "approximate" part of ANN: exact search would
    /// require scanning all buckets.
    pub fn candidates(&self, projected_query: &[f32], nprobe: usize) -> Vec<u64> {
        let mut dists: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                let d: f32 = c.iter().zip(projected_query.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
                (d, idx)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut out = Vec::new();
        for (_, idx) in dists.into_iter().take(nprobe.max(1)) {
            if let Some(bucket) = self.buckets.get(&idx) {
                out.extend_from_slice(bucket);
            }
        }
        out
    }

    pub fn n_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// The centroid a given record id is currently bucketed under, if any.
    pub fn assigned_centroid(&self, id: u64) -> Option<usize> {
        self.assignment.get(&id).copied()
    }

    /// The centroid vector at `idx`, in the same (projected) space records
    /// are bucketed in.
    pub fn centroid_vector(&self, idx: usize) -> Option<&[f32]> {
        self.centroids.get(idx).map(|c| c.as_slice())
    }
}

/// Convenience helper: project then insert in one call.
pub fn project_and_insert(index: &mut CentroidIndex, projector: &Projector, id: u64, full_vec: &[f32]) {
    let projected = projector.project(full_vec);
    index.insert(id, &projected);
}
