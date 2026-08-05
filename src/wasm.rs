//! wasm-bindgen surface for `LatentDb`, so it can be loaded and searched
//! directly from JS in a browser (no server needed for this half -- see
//! `web/` for a demo). Gated behind the `wasm` feature; has zero effect on
//! default native builds.
//!
//! `LatentDb::save`/`load` use `std::fs`, which has no real backing on
//! `wasm32-unknown-unknown`, so this wrapper goes through bincode bytes
//! directly instead (produced offline, natively, by `examples/build_index.rs`).

use wasm_bindgen::prelude::*;

use crate::db::LatentDb;
use crate::steering::SteeringVector;

#[wasm_bindgen]
pub struct WasmLatentDb {
    inner: LatentDb,
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[wasm_bindgen]
impl WasmLatentDb {
    /// Deserialize a `LatentDb` previously produced by `bincode::serialize`
    /// (see `examples/build_index.rs`), e.g. from a `fetch()`'d byte buffer.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmLatentDb, JsValue> {
        let inner: LatentDb =
            bincode::deserialize(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Doesn't go through `LatentDb::load` (`std::fs` has no real
        // backing on wasm32-unknown-unknown), so it must run the same
        // post-deserialize check `load` does -- see
        // `LatentDb::validate_after_deserialize`'s doc comment.
        inner
            .validate_after_deserialize()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmLatentDb { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Approximate nearest-neighbour search. `query` is the caller's own
    /// pre-computed embedding (this crate never computes embeddings itself
    /// -- see README). Returns a JSON string: `[{"id":..,"score":..,"metadata":".."}, ...]`.
    pub fn search(&self, query: Vec<f32>, k: usize, nprobe: usize) -> Result<String, JsValue> {
        let hits = self.inner.search(&query, k, nprobe);
        let json: Vec<serde_json::Value> = hits
            .into_iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.id,
                    "score": h.score,
                    "metadata": h.metadata,
                })
            })
            .collect();
        serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Like [`Self::search`], but first shifts `query` by a steering
    /// direction before searching -- see `steering::SteeringVector`.
    /// `direction` must be unit-norm (within `norm_tol`) and the same
    /// dimension as the DB; `alpha` (steering strength) must be in `[0, 1]`.
    /// Returns the same JSON shape as [`Self::search`].
    #[wasm_bindgen(js_name = searchSteered)]
    #[allow(clippy::too_many_arguments)]
    pub fn search_steered(
        &self,
        query: Vec<f32>,
        k: usize,
        nprobe: usize,
        direction: Vec<f32>,
        alpha: f32,
        norm_tol: f32,
    ) -> Result<String, JsValue> {
        let steering = SteeringVector::new(direction, alpha, norm_tol)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let hits = self.inner.search_steered(&query, k, nprobe, &steering);
        let json: Vec<serde_json::Value> = hits
            .into_iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.id,
                    "score": h.score,
                    "metadata": h.metadata,
                })
            })
            .collect();
        serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Explore a Viable Manifold Graph over the `neighborhood_size` records
    /// closest (Euclidean, over the approximate decoded vector) to `anchor`
    /// -- typically the same embedding just passed to `search`. Reports
    /// graph size, a random walk of `steps` hops (seeded by `seed`) starting
    /// near `anchor`, and -- when a second in-graph record is found -- a
    /// geodesic between the two nearest in-graph records to `anchor`.
    ///
    /// The Euclidean radius that gates graph membership is derived from
    /// `neighborhood_size` (the distance to the `neighborhood_size`-th
    /// closest record) rather than taken as a raw parameter: a fixed radius
    /// would have to be picked in the caller's embedding-distance units,
    /// which vary by model and aren't knowable in advance, whereas "include
    /// my N nearest records" is scale-invariant and can never silently
    /// produce an empty graph the way a mis-scaled fixed radius can.
    ///
    /// Returns a JSON object:
    /// `{ nNodes, nEdges, radius, randomWalk: [{id,metadata}, ...],
    ///    geodesic: { from, to, hops, path: [{id,metadata}, ...] } | null }`.
    #[wasm_bindgen(js_name = exploreNeighborhood)]
    pub fn explore_neighborhood(
        &self,
        anchor: Vec<f32>,
        neighborhood_size: usize,
        k_nearest: usize,
        steps: usize,
        seed: u64,
    ) -> Result<String, JsValue> {
        if anchor.len() != self.inner.dim() {
            return Err(JsValue::from_str(&format!(
                "anchor dim {} != db dim {}",
                anchor.len(),
                self.inner.dim()
            )));
        }

        // Rank every record by actual Euclidean distance to `anchor` (not
        // `search`'s cosine ranking -- PQ decoding perturbs each record's
        // norm slightly, so the two orderings aren't always identical).
        let nprobe = self.inner.n_index_centroids();
        let mut by_distance: Vec<(u64, f32)> = self
            .inner
            .search(&anchor, self.inner.len(), nprobe)
            .into_iter()
            .filter_map(|h| {
                let approx = self.inner.get_approx_vector(h.id)?;
                Some((h.id, euclidean(&approx, &anchor)))
            })
            .collect();
        by_distance.sort_by(|a, b| a.1.total_cmp(&b.1));

        let n = neighborhood_size.max(1).min(by_distance.len());
        // Small epsilon so the n-th neighbor itself is included despite the
        // predicate below recomputing distance independently from
        // PQ-decoded coordinates (f32 rounding could otherwise exclude it).
        let radius = by_distance
            .get(n.saturating_sub(1))
            .map(|&(_, d)| d + 1e-4)
            .unwrap_or(f32::INFINITY);

        let anchor_for_predicate = anchor.clone();
        let predicate = move |v: &[f32]| euclidean(v, &anchor_for_predicate) <= radius;
        let graph = self.inner.build_viable_graph(predicate, k_nearest, false);

        // Keep only the distance-ranked ids that made it into the graph, so
        // the walk/geodesic anchors are guaranteed graph members.
        let in_graph_ids: Vec<u64> = by_distance
            .into_iter()
            .map(|(id, _)| id)
            .filter(|id| graph.contains(*id))
            .collect();

        let doc = |id: u64| -> serde_json::Value {
            serde_json::json!({ "id": id, "metadata": self.inner.get_metadata(id) })
        };

        let random_walk_json: Vec<serde_json::Value> = in_graph_ids
            .first()
            .map(|&start| {
                graph
                    .random_walk(start, steps, seed)
                    .into_iter()
                    .map(doc)
                    .collect()
            })
            .unwrap_or_default();

        let geodesic_json = match (in_graph_ids.first(), in_graph_ids.get(1)) {
            (Some(&from), Some(&to)) => graph.geodesic(from, to).map(|path| {
                serde_json::json!({
                    "from": from,
                    "to": to,
                    "hops": path.len().saturating_sub(1),
                    "path": path.into_iter().map(doc).collect::<Vec<_>>(),
                })
            }),
            _ => None,
        };

        let out = serde_json::json!({
            "nNodes": graph.n_nodes(),
            "nEdges": graph.n_edges(),
            "radius": radius,
            "randomWalk": random_walk_json,
            "geodesic": geodesic_json,
        });
        serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
