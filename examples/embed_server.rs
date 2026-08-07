//! Minimal HTTP server exposing MiniLM text embedding over `POST /embed`.
//!
//! Part of the "hybrid" wasm + web UI demo (see `web/index.html`): the
//! browser side runs `LatentDb` search entirely in wasm (no server needed
//! for that), but embedding the user's live query text still needs a real
//! model forward pass, which candle/MiniLM can't do in-browser in this demo
//! -- so this tiny server does just that one thing.
//!
//! Run with:
//!     cargo run --release --example embed_server
//!
//! Then:
//!     curl -s -X POST localhost:8787/embed \
//!         -H 'Content-Type: application/json' \
//!         -d '{"text":"what temperature to bake bread"}'

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Error as E, Result};
use axum::extract::State;
use axum::http::Method;
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingParams, Tokenizer};
use tower_http::cors::{Any, CorsLayer};

const PORT: u16 = 8787;

struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: candle_core::Device,
}

impl Embedder {
    fn load() -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let client = HFClientSync::new()?;
        let repo = client.model("sentence-transformers", "all-MiniLM-L6-v2");
        let revision = "main";
        let config_filename = repo
            .download_file()
            .filename("config.json")
            .revision(revision)
            .send()?;
        let tokenizer_filename = repo
            .download_file()
            .filename("tokenizer.json")
            .revision(revision)
            .send()?;
        let weights_filename = repo
            .download_file()
            .filename("model.safetensors")
            .revision(revision)
            .send()?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed(&mut self, sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }

        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|t| Ok(Tensor::new(t.get_ids().to_vec().as_slice(), &self.device)?))
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|t| {
                Ok(Tensor::new(
                    t.get_attention_mask().to_vec().as_slice(),
                    &self.device,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        let mask = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = mask.sum(1)?;
        let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
        let pooled = summed.broadcast_div(&sum_mask)?;
        let normalized = normalize_l2(&pooled)?;

        let (n, _dim) = normalized.dims2()?;
        (0..n)
            .map(|i| Ok(normalized.get(i)?.to_vec1::<f32>()?))
            .collect()
    }
}

fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

#[derive(Deserialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

async fn embed_handler(
    State(embedder): State<Arc<Mutex<Embedder>>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (axum::http::StatusCode, String)> {
    let embedding = tokio::task::spawn_blocking(move || {
        let mut embedder = embedder.lock().unwrap();
        embedder
            .embed(&[req.text.as_str()])
            .map(|mut v| v.remove(0))
    })
    .await
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(EmbedResponse { embedding }))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading MiniLM...");
    let embedder = Arc::new(Mutex::new(Embedder::load()?));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/embed", post(embed_handler))
        .with_state(embedder)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", PORT)).await?;
    println!("embed_server listening on http://localhost:{PORT}/embed");
    axum::serve(listener, app).await?;
    Ok(())
}
