//! Walking-skeleton web demo (Issue #12): an axum HTTP server that boots
//! with real MiniLM embeddings (candle, reusing the `Embedder` pattern from
//! `examples/mini_rag.rs`), seeds an in-memory `LatentDb` from the same
//! 6-topic demo corpus, and serves both a small inline HTML/JS page and a
//! JSON API on one port.
//!
//! `LatentDb::build` is a batch-trained snapshot (PQ codebooks + centroid
//! index trained once over whatever data exists at build time), so this
//! server keeps every raw `(embedding, metadata)` pair it has ever accepted
//! in `Corpus::records` and rebuilds a fresh `LatentDb` from all of them on
//! every mutation (startup seed, and each ingest). A rebuild reassigns every
//! record's internal `LatentDb` id, so every record is also given a
//! `RecordKey` -- stable for the lifetime of the process, independent of
//! whatever the current `LatentDb` id happens to be -- so a client reference
//! survives a later rebuild triggered by someone else's ingest. Every later
//! ticket in this demo's batch (Search, Integrity, Boundary Classes,
//! Steering, Graph Explorer, Bandit Arena) builds on `Corpus`/`RecordKey`
//! rather than re-solving record identity itself.
//!
//! The page's first tab is **Ingest**: paste an English article URL, click
//! Ingest, and the server fetches the page (`ureq`, with a real
//! `User-Agent`), extracts the article's readable text (`scraper`, `<p>`
//! tags), splits it into sentence-group chunks capped to a safe token
//! length for MiniLM, embeds each chunk, and adds it into the accumulating
//! DB.
//!
//! The second tab is **Search** (Issue #13): type a free-text query and the
//! server embeds it, then runs both `LatentDb::search()` and
//! `LatentDb::search_blended()` (the query's own words, split on
//! whitespace, as the lexical terms) against the current accumulated DB,
//! returning both rankings -- keyed by `RecordKey`, not the internal id --
//! side by side, so a viewer can see blended fusion change the top hit.
//!
//! The third tab is **Integrity** (Issue #14): shows the current
//! `LatentDb::compression_ratio()` and `merkle_root()`, and gives every
//! record shown anywhere (Ingest's chunk list, Search's hit lists, or typed
//! directly into the tab's own box) a "Verify" action -- `Corpus::verify`
//! resolves the `RecordKey` to its current `LatentDb` id, then computes a
//! fresh `merkle_proof()` and `merkle_root()` from the same `&self` borrow
//! and checks `MerkleProof::verify()`, returning both the boolean and the
//! exact root it checked against. So a verification that happens to race a
//! rebuild (someone else's ingest) still checks -- and reports -- a root
//! taken *after* that rebuild, never a stale one.
//!
//! Run with:
//!     cargo run --release --example url_rag_server
//!
//! Then open http://localhost:8789/ in a browser. First run downloads
//! ~90MB of MiniLM weights from the Hugging Face Hub (cached after).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Error as E, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use latent_db::{Digest, LatentDb, LatentDbError, SearchHit};
use scraper::{Html as Document, Selector};
use serde::{Deserialize, Serialize};
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

const PORT: u16 = 8789;
const USER_AGENT: &str = "latent-db-url-rag-demo/0.1 (+https://github.com/TongThanaphon/latent-db)";
/// Sentences are batched into a chunk only up to this many tokens (leaving
/// room for `[CLS]`/`[SEP]`, added separately). `Embedder::load` also hard-
/// truncates at the tokenizer level to `max_position_embeddings`, so this
/// budget is about producing coherent chunks, not the only overflow guard.
const CHUNK_TOKEN_BUDGET_MARGIN: usize = 2;
/// How many chunks get embedded per `BertModel::forward` call.
const EMBED_BATCH_SIZE: usize = 32;
/// Paragraphs shorter than this are dropped before chunking -- filters most
/// nav/footer/byline boilerplate that a plain `<p>` selector also picks up.
const MIN_PARAGRAPH_CHARS: usize = 40;

/// Same 6-topic seed corpus as `examples/mini_rag.rs` (kept as a separate
/// copy -- see that file's module doc for why these examples don't share a
/// module).
const CORPUS: &[(&str, &str)] = &[
    (
        "cooking",
        "Preheat the oven to 200C before baking the bread.",
    ),
    (
        "cooking",
        "Add a pinch of salt to the pasta water while it boils.",
    ),
    (
        "cooking",
        "Simmer the tomato sauce for twenty minutes until it thickens.",
    ),
    (
        "cooking",
        "Whisk the eggs and sugar together until the mixture is pale.",
    ),
    (
        "cooking",
        "Marinate the chicken in soy sauce and garlic overnight.",
    ),
    (
        "cooking",
        "Knead the dough for ten minutes until it becomes smooth and elastic.",
    ),
    (
        "cooking",
        "Season the steak generously with salt and pepper before searing it.",
    ),
    (
        "cooking",
        "Fold the whipped cream gently into the chocolate mousse.",
    ),
    (
        "astronomy",
        "Jupiter is the largest planet in the solar system.",
    ),
    (
        "astronomy",
        "A neutron star can form when a massive star collapses.",
    ),
    (
        "astronomy",
        "The Hubble telescope has captured images of distant galaxies.",
    ),
    (
        "astronomy",
        "Saturn's rings are made mostly of ice and rock particles.",
    ),
    (
        "astronomy",
        "Astronomers detected a new exoplanet orbiting a red dwarf.",
    ),
    (
        "astronomy",
        "A total solar eclipse occurs when the moon fully blocks the sun.",
    ),
    (
        "astronomy",
        "Black holes warp spacetime so strongly that light cannot escape.",
    ),
    (
        "astronomy",
        "The Milky Way contains hundreds of billions of stars.",
    ),
    (
        "finance",
        "The central bank raised interest rates to curb inflation.",
    ),
    (
        "finance",
        "Diversifying a portfolio reduces exposure to a single asset.",
    ),
    (
        "finance",
        "The company reported quarterly earnings above analyst expectations.",
    ),
    (
        "finance",
        "Bond yields fell after the latest jobs report was released.",
    ),
    (
        "finance",
        "Investors moved capital into index funds during the downturn.",
    ),
    (
        "finance",
        "A rising unemployment rate often signals a slowing economy.",
    ),
    (
        "finance",
        "The stock market rallied after the merger was announced.",
    ),
    (
        "finance",
        "Compound interest lets savings grow faster over long time horizons.",
    ),
    (
        "travel",
        "Book a window seat for the best view during the flight.",
    ),
    (
        "travel",
        "The train departs from platform nine at half past six.",
    ),
    (
        "travel",
        "Pack light when hiking through the mountains for several days.",
    ),
    (
        "travel",
        "The old town's cobblestone streets are best explored on foot.",
    ),
    (
        "travel",
        "Renew your passport well before an international trip.",
    ),
    (
        "travel",
        "The ferry crosses the strait twice a day during summer.",
    ),
    (
        "travel",
        "Local guides recommend visiting the market early in the morning.",
    ),
    (
        "travel",
        "A long layover gives travelers time to explore the airport city.",
    ),
    (
        "health",
        "Drinking enough water throughout the day supports healthy digestion.",
    ),
    (
        "health",
        "Regular exercise lowers the risk of heart disease over time.",
    ),
    (
        "health",
        "A balanced diet includes plenty of vegetables and whole grains.",
    ),
    (
        "health",
        "Getting seven to eight hours of sleep improves concentration.",
    ),
    (
        "health",
        "Stretching before a workout helps prevent muscle strain.",
    ),
    (
        "health",
        "Doctors recommend an annual checkup to catch problems early.",
    ),
    (
        "health",
        "Chronic stress can weaken the immune system over months.",
    ),
    (
        "health",
        "Washing hands frequently reduces the spread of common colds.",
    ),
    (
        "technology",
        "The new processor doubles battery life compared to last year's model.",
    ),
    (
        "technology",
        "Cloud storage lets users access files from any device.",
    ),
    (
        "technology",
        "Encryption protects sensitive data from unauthorized access.",
    ),
    (
        "technology",
        "The software update fixed several long-standing security bugs.",
    ),
    (
        "technology",
        "Machine learning models improve as they train on more data.",
    ),
    (
        "technology",
        "Fiber-optic cables carry data faster than traditional copper wires.",
    ),
    (
        "technology",
        "A firmware bug caused the smart thermostat to overheat.",
    ),
    (
        "technology",
        "Open-source projects rely on contributions from volunteer developers.",
    ),
];

// ---------------------------------------------------------------------
// Embedding (candle + MiniLM) -- boilerplate shared in spirit with
// mini_rag.rs / embed_server.rs; see mini_rag.rs's module doc for why this
// isn't factored into the library crate.
// ---------------------------------------------------------------------

struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: candle_core::Device,
    /// Content token budget per chunk, leaving room for `[CLS]`/`[SEP]`
    /// (derived from the model's own `max_position_embeddings`, not
    /// hardcoded, so this tracks whatever checkpoint is actually loaded).
    max_chunk_tokens: usize,
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
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        // Hard safety net: whatever chunking produced, never hand the model
        // more tokens than its position-embedding table has room for.
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: config.max_position_embeddings,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_chunk_tokens: config
                .max_position_embeddings
                .saturating_sub(CHUNK_TOKEN_BUDGET_MARGIN),
        })
    }

    /// Embed a batch of sentences: tokenize -> BERT forward pass ->
    /// attention-mask-weighted mean pooling -> L2 normalize.
    fn embed(&mut self, sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
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

    /// Embed many chunks in fixed-size batches, so a long article doesn't
    /// pad every chunk to the length of its single longest one.
    fn embed_many(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH_SIZE) {
            let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
            out.extend(self.embed(&refs)?);
        }
        Ok(out)
    }

    /// Count how many tokens `text` would actually consume (no special
    /// tokens, no padding) -- used by the chunker to stay under budget.
    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(usize::MAX)
    }
}

fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

// ---------------------------------------------------------------------
// Text extraction + chunking (pure, no model/network -- see `mod tests`).
// ---------------------------------------------------------------------

/// Naive sentence splitter: breaks after `.`, `!`, or `?` when followed by
/// whitespace or end of string. Good enough for grouping article prose into
/// token-budgeted chunks; not real NLP sentence segmentation (e.g. "3.14"
/// or "Dr. Smith" won't split mid-abbreviation only by accident of not
/// being followed by whitespace).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let at_boundary = chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
            if at_boundary {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                current.clear();
            }
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }
    sentences
}

/// Groups `text`'s sentences into chunks whose token count (per
/// `count_tokens`) stays under `max_tokens`, never splitting a sentence
/// across chunks. `count_tokens` is injected so this stays testable without
/// a real tokenizer; the production call site measures with the loaded
/// MiniLM tokenizer via `Embedder::count_tokens`.
fn chunk_text(text: &str, max_tokens: usize, count_tokens: impl Fn(&str) -> usize) -> Vec<String> {
    let max_tokens = max_tokens.max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;

    for sentence in split_sentences(text) {
        let sentence_tokens = count_tokens(&sentence);
        if !current.is_empty() && current_tokens + sentence_tokens > max_tokens {
            chunks.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&sentence);
        current_tokens += sentence_tokens;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Extracts `<p>` text from an HTML document, dropping short paragraphs
/// (nav/footer/byline boilerplate) and collapsing internal whitespace.
fn extract_paragraphs(html: &str) -> Vec<String> {
    let document = Document::parse_document(html);
    let selector = Selector::parse("p").expect("static selector \"p\" always parses");

    document
        .select(&selector)
        .map(|el| el.text().collect::<Vec<_>>().join(" "))
        .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|text| text.chars().count() >= MIN_PARAGRAPH_CHARS)
        .collect()
}

fn fetch_article(url: &str) -> Result<String> {
    let mut response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("failed to fetch {url}"))?;
    response
        .body_mut()
        .read_to_string()
        .with_context(|| format!("failed to read response body from {url}"))
}

// ---------------------------------------------------------------------
// Corpus: the accumulating raw record store + the rebuilt LatentDb it
// trains, keyed so a rebuild never invalidates a key a client is holding.
// ---------------------------------------------------------------------

/// Stable external identifier for a record, independent of whatever
/// `LatentDb`-internal id a rebuild happens to assign it. Assigned once,
/// in append order, when a chunk first enters `Corpus::records`; never
/// reused or reassigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecordKey(u64);

/// One chunk ever accepted into `Corpus`, in the form it's stored for the
/// lifetime of the raw accumulator: full-precision embedding + its
/// original metadata string, tagged with the `RecordKey` it was assigned
/// on arrival.
struct RawRecord {
    key: RecordKey,
    embedding: Vec<f32>,
    metadata: String,
}

/// What a from-scratch `LatentDb::build` + full re-`insert` produces:
/// the freshly trained DB plus both key/id lookup directions (see
/// `Corpus::key_to_id`/`id_to_key`). Bundled into a struct rather than a
/// tuple so `Corpus::seed`/`rebuild` can destructure it by name.
struct Built {
    db: LatentDb,
    key_to_id: HashMap<RecordKey, u64>,
    id_to_key: HashMap<u64, RecordKey>,
}

/// LatentDb build hyperparameters, matching `mini_rag.rs`'s call
/// (`n_index_centroids` mirrors the seed corpus's 6 topics; ingest doesn't
/// retune this as the corpus grows past that -- fine for a demo).
const PQ_SUBSPACES: usize = 8;
const PQ_CENTROIDS: usize = 16;
const INDEX_CENTROIDS: usize = 6;
const SKETCH_DIM: usize = 8;
const BUILD_SEED: u64 = 42;

struct Corpus {
    /// Append-only source of truth: every chunk ever accepted, in the
    /// order it was accepted. Never reordered or truncated, so a
    /// `RecordKey` (an index into this in spirit) stays meaningful forever.
    records: Vec<RawRecord>,
    next_key: u64,
    db: LatentDb,
    /// Every key ever issued -> its id in the *current* `db`. Multiple
    /// keys can point at the same id (re-ingesting identical content
    /// dedupes inside `LatentDb::insert`), which is why this direction
    /// can't just be inverted from `id_to_key`.
    key_to_id: HashMap<RecordKey, u64>,
    /// Current `db` id -> the first (canonical) key that resolved to it,
    /// used to label search-style results with one key per live record.
    id_to_key: HashMap<u64, RecordKey>,
}

impl Corpus {
    fn seed(seed_records: Vec<(Vec<f32>, String)>) -> Result<Self, LatentDbError> {
        let mut records = Vec::with_capacity(seed_records.len());
        let mut next_key = 0u64;
        for (embedding, metadata) in seed_records {
            records.push(RawRecord {
                key: RecordKey(next_key),
                embedding,
                metadata,
            });
            next_key += 1;
        }
        let built = Self::build_db(&records)?;
        Ok(Corpus {
            records,
            next_key,
            db: built.db,
            key_to_id: built.key_to_id,
            id_to_key: built.id_to_key,
        })
    }

    fn build_db(records: &[RawRecord]) -> Result<Built, LatentDbError> {
        let training_vectors: Vec<Vec<f32>> = records.iter().map(|r| r.embedding.clone()).collect();
        let mut db = LatentDb::build(
            &training_vectors,
            PQ_SUBSPACES,
            PQ_CENTROIDS,
            INDEX_CENTROIDS,
            SKETCH_DIM,
            BUILD_SEED,
        );

        let mut key_to_id = HashMap::with_capacity(records.len());
        let mut id_to_key = HashMap::with_capacity(records.len());
        for r in records {
            let id = db.insert(&r.embedding, r.metadata.clone())?;
            key_to_id.insert(r.key, id);
            id_to_key.entry(id).or_insert(r.key);
        }
        Ok(Built {
            db,
            key_to_id,
            id_to_key,
        })
    }

    fn rebuild(&mut self) -> Result<(), LatentDbError> {
        let built = Self::build_db(&self.records)?;
        self.db = built.db;
        self.key_to_id = built.key_to_id;
        self.id_to_key = built.id_to_key;
        Ok(())
    }

    /// Appends every `(embedding, metadata)` pair as a new raw record (each
    /// gets a fresh key, even if a prior call already holds identical
    /// content -- `LatentDb::insert`'s own content-hash dedup is what keeps
    /// `db.len()` from growing on re-ingest, not any check here) and
    /// rebuilds. Returns the keys assigned, in the same order as `chunks`.
    fn ingest(&mut self, chunks: Vec<(Vec<f32>, String)>) -> Result<Vec<RecordKey>, LatentDbError> {
        let mut new_keys = Vec::with_capacity(chunks.len());
        for (embedding, metadata) in chunks {
            let key = RecordKey(self.next_key);
            self.next_key += 1;
            self.records.push(RawRecord {
                key,
                embedding,
                metadata,
            });
            new_keys.push(key);
        }
        self.rebuild()?;
        Ok(new_keys)
    }

    /// Resolves `key` to its id in the *current* `db` and checks a freshly
    /// built inclusion proof against a freshly read `merkle_root()` -- both
    /// read from the same `&self` borrow, so this is correct even called
    /// immediately after a rebuild (someone else's ingest) changed both the
    /// id `key` maps to and the root out from under any snapshot a caller
    /// might otherwise have cached. Returns the verification result paired
    /// with the exact root it was checked against, so a caller never has to
    /// fall back to a separately (and possibly staler) fetched root when
    /// reporting one. Returns `None` if `key` was never issued -- every
    /// issued key stays resolvable forever (`records` is append-only and
    /// every record is reinserted into `key_to_id` on every rebuild).
    fn verify(&self, key: RecordKey) -> Option<(bool, Digest)> {
        let id = *self.key_to_id.get(&key)?;
        let proof = self.db.merkle_proof(id)?;
        let root = self.db.merkle_root();
        Some((proof.verify(&root), root))
    }
}

// ---------------------------------------------------------------------
// Integrity (Issue #14): compression ratio, Merkle root, and per-record
// inclusion-proof verification over the current `Corpus::db`.
// ---------------------------------------------------------------------

/// Lowercase hex encoding, e.g. for rendering a `Digest` (`[u8; 32]`) in a
/// JSON response. Kept as a local copy rather than pulling in the `hex`
/// crate -- same one-liner `latentdb-cli`'s `main.rs` already carries.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------

struct AppState {
    embedder: Mutex<Embedder>,
    corpus: Mutex<Corpus>,
}

#[derive(Deserialize)]
struct IngestRequest {
    url: String,
}

#[derive(Serialize)]
struct IngestedChunk {
    key: RecordKey,
    preview: String,
}

#[derive(Serialize)]
struct IngestResponse {
    chunks_added: usize,
    /// How many of those chunks resulted in a genuinely new `LatentDb`
    /// record (i.e. `db.len()` delta) -- re-ingesting the same URL reports
    /// `chunks_added > 0` but `net_new_records == 0`, since content-hash
    /// dedup in `LatentDb::insert` collapses the duplicates.
    net_new_records: usize,
    total_records: usize,
    chunks: Vec<IngestedChunk>,
}

fn preview(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push('\u{2026}');
    }
    out
}

async fn ingest_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
    let url = req.url.trim().to_string();
    if url.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "url must not be empty".to_string()));
    }

    tokio::task::spawn_blocking(move || run_ingest(&state, &url))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        // `{e:#}` (not `{e}`/`to_string()`) so the readable error includes
        // the full anyhow context chain, e.g. "failed to fetch <url>: ...".
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("{e:#}")))
}

/// The whole ingest pipeline, run inside `spawn_blocking`: fetch -> extract
/// -> chunk -> embed -> accumulate -> rebuild. Locks are taken and released
/// entirely within this synchronous function, so no `MutexGuard` ever needs
/// to survive across an `.await`.
fn run_ingest(state: &AppState, url: &str) -> Result<IngestResponse> {
    let html = fetch_article(url)?;
    let paragraphs = extract_paragraphs(&html);
    if paragraphs.is_empty() {
        return Err(anyhow!(
            "no readable paragraph text found at {url} (paywall, JS-rendered page, or blocked)"
        ));
    }

    let mut embedder = state.embedder.lock().unwrap();
    let max_tokens = embedder.max_chunk_tokens;
    let chunk_texts: Vec<String> = paragraphs
        .iter()
        .flat_map(|p| chunk_text(p, max_tokens, |s| embedder.count_tokens(s)))
        .collect();
    if chunk_texts.is_empty() {
        return Err(anyhow!(
            "no readable paragraph text found at {url} (paywall, JS-rendered page, or blocked)"
        ));
    }
    let embeddings = embedder.embed_many(&chunk_texts)?;
    drop(embedder);

    let chunks: Vec<(Vec<f32>, String)> = embeddings
        .into_iter()
        .zip(chunk_texts.iter())
        .map(|(embedding, text)| (embedding, format!("[{url}] {text}")))
        .collect();
    let chunks_added = chunks.len();

    let mut corpus = state.corpus.lock().unwrap();
    let before = corpus.db.len();
    let new_keys = corpus.ingest(chunks)?;
    let total_records = corpus.db.len();
    let net_new_records = total_records.saturating_sub(before);

    let chunks = new_keys
        .into_iter()
        .zip(chunk_texts)
        .map(|(key, text)| IngestedChunk {
            key,
            preview: preview(&text, 160),
        })
        .collect();
    drop(corpus);

    Ok(IngestResponse {
        chunks_added,
        net_new_records,
        total_records,
        chunks,
    })
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
}

#[derive(Serialize)]
struct SearchHitView {
    key: RecordKey,
    score: f32,
    metadata: String,
}

#[derive(Serialize)]
struct SearchResponse {
    /// Plain-vector `LatentDb::search()` ranking.
    vector: Vec<SearchHitView>,
    /// `LatentDb::search_blended()` ranking, fusing the same vector
    /// candidates with a lexical/metadata term-overlap ranking over
    /// `query`'s own words via Reciprocal Rank Fusion. Its `score` is an
    /// RRF score, not a cosine similarity -- not comparable against
    /// `vector`'s scores, only against other `blended` entries.
    blended: Vec<SearchHitView>,
}

/// Translates `SearchHit::id` (a `LatentDb`-internal id, invalidated by the
/// next rebuild) to each hit's stable `RecordKey`, preserving rank order. A
/// hit whose id isn't in `id_to_key` is dropped rather than panicking, but
/// that's unreachable on the normal path: every id currently in `db` was
/// registered in `id_to_key` by the same `Corpus::build_db` call.
fn hits_to_views(hits: Vec<SearchHit>, id_to_key: &HashMap<u64, RecordKey>) -> Vec<SearchHitView> {
    hits.into_iter()
        .filter_map(|hit| {
            let key = *id_to_key.get(&hit.id)?;
            Some(SearchHitView {
                key,
                score: hit.score,
                metadata: hit.metadata,
            })
        })
        .collect()
}

/// How many hits each ranking returns.
const SEARCH_K: usize = 5;

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "query must not be empty".to_string(),
        ));
    }

    tokio::task::spawn_blocking(move || run_search(&state, &query))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        // `{e:#}` (not `{e}`/`to_string()`), same as `ingest_handler`, so a
        // chained anyhow error keeps its full context.
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))
}

/// Embeds `query`, then runs both rankings against the current db: the
/// plain-vector `search()` and the lexically blended `search_blended()`
/// (using `query`'s own words, split on whitespace, as the lexical terms --
/// `search_blended` lowercases them itself). Locks are taken and released
/// entirely within this synchronous function, mirroring `run_ingest`.
fn run_search(state: &AppState, query: &str) -> Result<SearchResponse> {
    let mut embedder = state.embedder.lock().unwrap();
    let embedding = embedder
        .embed(&[query])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("embedding a single query produced no vector"))?;
    drop(embedder);

    let terms: Vec<&str> = query.split_whitespace().collect();

    let corpus = state.corpus.lock().unwrap();
    let nprobe = corpus.db.n_index_centroids();
    let vector_hits = corpus.db.search(&embedding, SEARCH_K, nprobe);
    let blended_hits = corpus
        .db
        .search_blended(&terms, &embedding, SEARCH_K, nprobe);
    let vector = hits_to_views(vector_hits, &corpus.id_to_key);
    let blended = hits_to_views(blended_hits, &corpus.id_to_key);
    drop(corpus);

    Ok(SearchResponse { vector, blended })
}

#[derive(Serialize)]
struct IntegrityStatus {
    compression_ratio: f32,
    /// Lowercase hex encoding of the current `merkle_root()`.
    merkle_root: String,
}

/// Reads the current compression ratio and Merkle root -- cheap enough
/// (arithmetic plus a cached-tree lookup, no model/network I/O) to run
/// directly on the async handler rather than via `spawn_blocking`, unlike
/// `ingest_handler`/`search_handler`.
async fn integrity_handler(State(state): State<Arc<AppState>>) -> Json<IntegrityStatus> {
    let corpus = state.corpus.lock().unwrap();
    Json(IntegrityStatus {
        compression_ratio: corpus.db.compression_ratio(),
        merkle_root: hex(&corpus.db.merkle_root()),
    })
}

#[derive(Deserialize)]
struct VerifyRequest {
    key: RecordKey,
}

#[derive(Serialize)]
struct VerifyResponse {
    verified: bool,
    /// Hex encoding of the exact root `verified` was checked against (see
    /// `Corpus::verify`) -- returned rather than left for the caller to
    /// infer from a separately fetched `/integrity` response, so a client
    /// never has to pair this result with a root that might have gone
    /// stale between the two requests.
    merkle_root: String,
}

/// Same no-`spawn_blocking` reasoning as `integrity_handler`: `Corpus::verify`
/// is a proof build plus a hash comparison, no model/network I/O.
async fn integrity_verify_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (StatusCode, String)> {
    let corpus = state.corpus.lock().unwrap();
    let (verified, root) = corpus.verify(req.key).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("unknown record key {}", req.key.0),
        )
    })?;
    Ok(Json(VerifyResponse {
        verified,
        merkle_root: hex(&root),
    }))
}

async fn index_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    let total_records = state.corpus.lock().unwrap().db.len();
    Html(render_index_html(total_records))
}

fn render_index_html(total_records: usize) -> String {
    INDEX_HTML_TEMPLATE.replace("__TOTAL_RECORDS__", &total_records.to_string())
}

const INDEX_HTML_TEMPLATE: &str = include_str!("url_rag_server.html");

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading MiniLM (downloads ~90MB on first run, cached after)...");
    let mut embedder = Embedder::load()?;

    println!("Embedding {} seed corpus documents...", CORPUS.len());
    let corpus_texts: Vec<&str> = CORPUS.iter().map(|(_, text)| *text).collect();
    let corpus_embeddings = embedder.embed(&corpus_texts)?;
    let seed_records: Vec<(Vec<f32>, String)> = CORPUS
        .iter()
        .zip(corpus_embeddings)
        .map(|((topic, text), embedding)| (embedding, format!("[{topic}] {text}")))
        .collect();
    let corpus = Corpus::seed(seed_records)?;
    println!("seeded {} records into LatentDb\n", corpus.db.len());

    let state = Arc::new(AppState {
        embedder: Mutex::new(embedder),
        corpus: Mutex::new(corpus),
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ingest", post(ingest_handler))
        .route("/search", post(search_handler))
        .route("/integrity", get(integrity_handler))
        .route("/integrity/verify", post(integrity_verify_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", PORT)).await?;
    println!("url_rag_server listening on http://localhost:{PORT}/");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_breaks_on_terminal_punctuation() {
        let sentences = split_sentences("Hello world. This is a test! Is it working?");
        assert_eq!(
            sentences,
            vec!["Hello world.", "This is a test!", "Is it working?"]
        );
    }

    #[test]
    fn split_sentences_keeps_a_trailing_fragment_without_punctuation() {
        let sentences = split_sentences("First sentence. trailing fragment with no terminator");
        assert_eq!(
            sentences,
            vec!["First sentence.", "trailing fragment with no terminator"]
        );
    }

    #[test]
    fn split_sentences_does_not_split_mid_number() {
        let sentences = split_sentences("Pi is roughly 3.14 and that's well known.");
        assert_eq!(sentences, vec!["Pi is roughly 3.14 and that's well known."]);
    }

    #[test]
    fn split_sentences_on_empty_input_is_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }

    /// Fake token counter for chunk_text tests: one "token" per whitespace-
    /// separated word, so tests don't need a real tokenizer/network access.
    fn word_count(s: &str) -> usize {
        s.split_whitespace().count()
    }

    #[test]
    fn chunk_text_stays_under_the_token_budget_per_chunk() {
        let text = "one two three. four five six. seven eight nine. ten eleven twelve.";
        let chunks = chunk_text(text, 6, word_count);
        for chunk in &chunks {
            assert!(
                word_count(chunk) <= 6,
                "chunk {chunk:?} exceeded the 6-token budget"
            );
        }
        // No sentence content lost across chunking.
        let rejoined: String = chunks.join(" ");
        assert_eq!(word_count(&rejoined), word_count(text));
    }

    #[test]
    fn chunk_text_never_splits_a_single_sentence_across_chunks() {
        let chunks = chunk_text("a b c. d e f.", 3, word_count);
        assert_eq!(chunks, vec!["a b c.", "d e f."]);
    }

    #[test]
    fn chunk_text_keeps_an_oversized_single_sentence_as_its_own_chunk() {
        // A single sentence longer than the budget can't be split further
        // here (that's the tokenizer's hard-truncation job at embed time);
        // it still becomes its own chunk rather than being lost or looping.
        let chunks = chunk_text("one two three four five.", 2, word_count);
        assert_eq!(chunks, vec!["one two three four five."]);
    }

    #[test]
    fn chunk_text_on_empty_input_is_empty() {
        assert!(chunk_text("", 10, word_count).is_empty());
    }

    #[test]
    fn extract_paragraphs_reads_p_tags_and_drops_short_ones() {
        let html = format!(
            "<html><body><nav><p>Home</p></nav><p>{}</p><p>hi</p></body></html>",
            "This paragraph is long enough to survive the short-paragraph filter easily."
        );
        let paragraphs = extract_paragraphs(&html);
        assert_eq!(paragraphs.len(), 1);
        assert!(paragraphs[0].starts_with("This paragraph is long enough"));
    }

    #[test]
    fn extract_paragraphs_on_a_page_with_no_p_tags_is_empty() {
        let html = "<html><body><div>No paragraphs here, just a div.</div></body></html>";
        assert!(extract_paragraphs(html).is_empty());
    }

    #[test]
    fn extract_paragraphs_collapses_internal_whitespace_and_nested_tags() {
        let html = "<html><body><p>Hello,\n   <strong>bold</strong>   world with   enough length to pass the filter.</p></body></html>";
        let paragraphs = extract_paragraphs(html);
        assert_eq!(paragraphs.len(), 1);
        assert!(!paragraphs[0].contains('\n'));
        assert!(!paragraphs[0].contains("   "));
    }

    fn synthetic_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn synthetic_records(n: usize, dim: usize) -> Vec<(Vec<f32>, String)> {
        (0..n)
            .map(|i| (synthetic_vector(dim, i as u64), format!("doc-{i}")))
            .collect()
    }

    #[test]
    fn corpus_keys_survive_a_rebuild_and_resolve_to_the_same_metadata() {
        let seed = synthetic_records(20, 16);
        let mut corpus = Corpus::seed(seed).expect("seed build should succeed");

        let snapshot: Vec<(RecordKey, String)> = corpus
            .records
            .iter()
            .map(|r| (r.key, r.metadata.clone()))
            .collect();

        // A later ingest (someone else's) rebuilds the whole LatentDb and
        // reassigns every internal id.
        let more = synthetic_records(10, 16)
            .into_iter()
            .map(|(v, _)| (v, "new-doc".to_string()))
            .collect();
        corpus.ingest(more).expect("ingest should succeed");

        for (key, metadata) in snapshot {
            let id = *corpus
                .key_to_id
                .get(&key)
                .unwrap_or_else(|| panic!("key {key:?} should still be registered after rebuild"));
            assert_eq!(
                corpus.db.get_metadata(id),
                Some(metadata.as_str()),
                "key {key:?} should still resolve to its original metadata after rebuild"
            );
        }
    }

    #[test]
    fn corpus_reingesting_identical_content_does_not_grow_db_len() {
        let seed = synthetic_records(20, 16);
        let mut corpus = Corpus::seed(seed).expect("seed build should succeed");
        let before = corpus.db.len();

        // Re-"ingest" byte-identical embeddings (same seed -> same vectors),
        // simulating a re-ingested URL producing the same chunks again.
        let duplicate = synthetic_records(20, 16);
        corpus.ingest(duplicate).expect("ingest should succeed");

        assert_eq!(
            corpus.db.len(),
            before,
            "LatentDb::insert's content-hash dedup should keep len() stable"
        );
    }

    #[test]
    fn hits_to_views_translates_ids_to_stable_keys_and_preserves_rank_order() {
        let mut id_to_key = HashMap::new();
        id_to_key.insert(1u64, RecordKey(100));
        id_to_key.insert(2u64, RecordKey(200));
        let hits = vec![
            SearchHit {
                id: 2,
                score: 0.9,
                metadata: "b".to_string(),
            },
            SearchHit {
                id: 1,
                score: 0.5,
                metadata: "a".to_string(),
            },
        ];

        let views = hits_to_views(hits, &id_to_key);

        assert_eq!(views.len(), 2);
        assert_eq!(views[0].key, RecordKey(200));
        assert_eq!(views[0].score, 0.9);
        assert_eq!(views[0].metadata, "b");
        assert_eq!(views[1].key, RecordKey(100));
    }

    #[test]
    fn corpus_ingesting_genuinely_new_content_grows_db_len() {
        let seed = synthetic_records(20, 16);
        let mut corpus = Corpus::seed(seed).expect("seed build should succeed");
        let before = corpus.db.len();

        let fresh = synthetic_records(5, 16)
            .into_iter()
            .map(|(v, _)| (v, "distinct".to_string()))
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, (v, m))| (v.into_iter().map(|x| x + 100.0 + i as f32).collect(), m))
            .collect();
        corpus.ingest(fresh).expect("ingest should succeed");

        assert_eq!(corpus.db.len(), before + 5);
    }

    #[test]
    fn hex_encodes_bytes_as_lowercase_hex() {
        assert_eq!(hex(&[0u8, 255u8, 16u8]), "00ff10");
        assert_eq!(hex(&[]), "");
    }

    #[test]
    fn verify_is_none_for_a_key_that_was_never_issued() {
        let seed = synthetic_records(5, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        assert_eq!(corpus.verify(RecordKey(9999)), None);
    }

    #[test]
    fn verify_confirms_inclusion_of_every_currently_stored_record_against_the_current_root() {
        let seed = synthetic_records(8, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let root = corpus.db.merkle_root();

        for record in &corpus.records {
            assert_eq!(
                corpus.verify(record.key),
                Some((true, root)),
                "key {:?} should verify against the current root",
                record.key
            );
        }
    }

    /// The core correctness requirement from Issue #14's acceptance
    /// criteria: verifying a record that survives a rebuild (triggered by
    /// someone else's ingest) must check against the *current* root, not a
    /// root captured before that rebuild.
    #[test]
    fn verify_after_a_rebuild_checks_against_the_current_root_not_a_stale_one() {
        let seed = synthetic_records(5, 16);
        let mut corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key = corpus.records[0].key;
        let root_before = corpus.db.merkle_root();

        let more = synthetic_records(3, 16)
            .into_iter()
            .map(|(v, _)| (v, "new-doc".to_string()))
            .collect();
        corpus.ingest(more).expect("ingest should succeed");

        let root_after = corpus.db.merkle_root();
        assert_ne!(
            root_before, root_after,
            "an ingest that adds new records should change the root"
        );
        let id_after = *corpus.key_to_id.get(&key).unwrap();

        assert_eq!(
            corpus.verify(key),
            Some((true, root_after)),
            "verifying key {key:?} after a rebuild should still succeed against the current \
             root, and report that root back rather than one the caller had cached"
        );

        // A proof built against the pre-rebuild root would not verify
        // against the post-rebuild root -- confirming `verify` really is
        // reading the current root, not one it happened to have cached.
        let proof_after = corpus.db.merkle_proof(id_after).unwrap();
        assert!(!proof_after.verify(&root_before));
    }

    #[test]
    fn verify_is_stable_across_repeated_calls_with_no_mutation_between() {
        let seed = synthetic_records(6, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key = corpus.records[0].key;

        assert_eq!(corpus.verify(key), corpus.verify(key));
    }
}
