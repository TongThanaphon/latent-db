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

#[wasm_bindgen]
pub struct WasmLatentDb {
    inner: LatentDb,
}

#[wasm_bindgen]
impl WasmLatentDb {
    /// Deserialize a `LatentDb` previously produced by `bincode::serialize`
    /// (see `examples/build_index.rs`), e.g. from a `fetch()`'d byte buffer.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmLatentDb, JsValue> {
        let inner: LatentDb =
            bincode::deserialize(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
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
}
