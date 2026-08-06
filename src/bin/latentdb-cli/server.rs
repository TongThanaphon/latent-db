//! `latentdb-cli serve`: an HTTP surface over a loaded `LatentDb`, extending
//! the axum + tokio pattern `examples/embed_server.rs` already established
//! in this repo (one `Arc<Mutex<..>>`-guarded piece of state) rather than
//! introducing a second HTTP stack. Unlike `embed_server.rs` this has no
//! browser client (see `web/index.html`), so it skips that example's CORS
//! layer -- nothing here is meant to be called cross-origin from JS.
//!
//! Every lock scope below is synchronous (acquire, touch the DB, drop the
//! guard) with no `.await` while held, and any real work (`insert`'s
//! `save()` back to disk) is done inline rather than deferred -- simpler
//! than `embed_server.rs`'s `spawn_blocking` dance, which exists there only
//! because model inference is CPU-heavy; a PQ-encoded insert/search is not.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use latent_db::LatentDb;
use serde::{Deserialize, Serialize};

struct AppState {
    db: Mutex<LatentDb>,
    db_path: PathBuf,
}

#[derive(Deserialize)]
struct SearchRequest {
    embedding: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_nprobe")]
    nprobe: usize,
}

fn default_k() -> usize {
    10
}

fn default_nprobe() -> usize {
    4
}

#[derive(Serialize)]
struct SearchHitJson {
    id: u64,
    score: f32,
    metadata: String,
}

#[derive(Serialize)]
struct SearchResponse {
    hits: Vec<SearchHitJson>,
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let db = state.db.lock().unwrap();
    if req.embedding.len() != db.dim() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "embedding has dim {}, expected {}",
                req.embedding.len(),
                db.dim()
            ),
        ));
    }
    let hits = db
        .search(&req.embedding, req.k, req.nprobe)
        .into_iter()
        .map(|h| SearchHitJson {
            id: h.id,
            score: h.score,
            metadata: h.metadata,
        })
        .collect();
    Ok(Json(SearchResponse { hits }))
}

#[derive(Deserialize)]
struct InsertRequest {
    embedding: Vec<f32>,
    metadata: String,
}

#[derive(Serialize)]
struct InsertResponse {
    id: u64,
    len: usize,
}

async fn insert_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, (StatusCode, String)> {
    let (id, len) = {
        let mut db = state.db.lock().unwrap();
        let id = db
            .insert(&req.embedding, req.metadata)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        db.save(&state.db_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        (id, db.len())
    };
    Ok(Json(InsertResponse { id, len }))
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    len: usize,
    dim: usize,
}

async fn healthz_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let db = state.db.lock().unwrap();
    Json(HealthResponse {
        status: "ok",
        len: db.len(),
        dim: db.dim(),
    })
}

#[tokio::main]
pub async fn serve(db_path: PathBuf, port: u16) -> Result<()> {
    let db = LatentDb::load(&db_path).with_context(|| format!("loading DB from {db_path:?}"))?;
    println!(
        "loaded {:?}: {} records, dim={}",
        db_path,
        db.len(),
        db.dim()
    );

    let state = Arc::new(AppState {
        db: Mutex::new(db),
        db_path,
    });

    let app = Router::new()
        .route("/healthz", get(healthz_handler))
        .route("/search", post(search_handler))
        .route("/insert", post(insert_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("binding to port {port}"))?;
    println!("latentdb-cli serve listening on http://localhost:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
