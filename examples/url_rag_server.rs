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
//! The fourth tab is **Boundary Classes** (Issue #15): builds a
//! `ViableGraph` over every currently-stored record (predicate: accept all)
//! and partitions it with `boundary_classes()` -- a structural-equivalence
//! partition (1-WL-style neighborhood-signature refinement, not a vector
//! clustering), so it's the *shape* of each record's neighborhood in the
//! kNN graph that groups records together, not raw embedding similarity.
//! Renders every record grouped by its class id, translated from the
//! `LatentDb`-internal id back to `RecordKey`, same as every other tab.
//!
//! The fifth tab is **Steering** (Issue #16): pick two records by key (e.g.
//! from a prior Search) and a query. The server builds a `SteeringVector`
//! whose direction is record B's full-precision embedding minus record A's,
//! unit-normalized (`Corpus::build_steering_vector` -- see that method's doc
//! comment for why it uses the raw embedding rather than
//! `get_approx_vector`), then runs `LatentDb::search()` and
//! `LatentDb::search_steered()` for the same query embedding side by side.
//! `Corpus::build_steering_vector` is the reusable piece this ticket owns:
//! Issue #17's Graph Explorer tab calls it directly for its steered-walk
//! toggle rather than reimplementing the construction. An alpha slider
//! re-runs both rankings live as it moves, so a viewer can watch
//! `search_steered()`'s ranking shift toward B's concept as alpha increases.
//!
//! The sixth tab is **Graph Explorer** (Issue #17): pick two records A and B
//! by key and build the same kind of "every currently-stored record"
//! `ViableGraph` Boundary Classes uses (its own `k_nearest`, see
//! `GRAPH_K_NEAREST`). **Path** mode shows `geodesic(A, B)` plus
//! `trajectory::path_geometry()`'s length/curvature/min-adjacent-cosine over
//! it, and a `bifurcation_ratio()` against a same-length comparison path --
//! a `random_walk()` anchored at the geodesic's own second node (one real
//! hop off A) rather than at A itself, since starting both paths at the
//! identical coordinate would put `bifurcation_ratio` in its zero-initial-
//! separation edge case (`separation_ratio` pinned to `1.0` or `+inf`,
//! `onset_step` always `None` -- see that function's doc comment) on every
//! request rather than only when the two paths genuinely start together.
//! **Walk** mode instead runs `weighted_random_walk()` from A, biased by a
//! `SteeringVector` built through `Corpus::build_steering_vector(A, B, ..)` --
//! the same reusable piece Steering owns, not reimplemented here -- next to
//! a plain `random_walk()` from the same start for comparison.
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
use latent_db::{
    bifurcation_ratio, cosine_sim, path_geometry, BifurcationResult, BoundaryClassId, Digest,
    LatentDb, LatentDbError, PathGeometry, SearchHit, SteeringError, SteeringVector,
};
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
/// Same value `examples/demo.rs`'s "Viable Manifold Graph" section
/// illustrates with. Chosen empirically against this file's own real
/// 6-topic/48-record seed corpus (embedded with the real MiniLM model, not
/// a synthetic stand-in): `cargo run --release --example url_rag_server`
/// then `curl localhost:8789/boundary-classes` on a freshly started server
/// currently produces 47 classes over the 48 seed records -- a real but
/// sparse partition where most records land in a singleton class, and the
/// one collision that does happen (`[cooking] Add a pinch of salt to the
/// pasta water...`, key 1, with `[cooking] Season the steak generously...`,
/// key 6) is between same-topic siblings, never across topics. This is a
/// live, re-checkable fact about the current corpus/model/algorithm, not a
/// guarantee -- `boundary_classes()` is a structural (graph-shape)
/// equivalence, not a vector-similarity clustering (see
/// `Corpus::boundary_classes`), so it isn't pinned by an automated test
/// against the real MiniLM-embedded corpus: the test suite is deliberately
/// network/model-free (see this file's own `[[example]]` entry in
/// `Cargo.toml`), so `boundary_classes_never_mixes_two_different_clusters_
/// into_one_class` below instead pins the same *shape* of behavior --
/// same-topic-only collisions, on a corpus with real cluster structure --
/// against a synthetic, network-free corpus.
const BOUNDARY_K_NEAREST: usize = 4;

/// `k_nearest` for the Graph Explorer tab's `ViableGraph` (Issue #17) -- a
/// separate constant from `BOUNDARY_K_NEAREST` even though they currently
/// hold the same value, since the two tabs want different (and in general
/// independently tunable) things from their graph: Boundary Classes wants a
/// *sparse* graph so `boundary_classes()` produces an interesting partition,
/// while Graph Explorer wants a *connected* one so `geodesic()` between two
/// arbitrarily-picked records (often from different topics) actually finds
/// a path rather than reporting "unreachable".
///
/// Verified empirically against the real seed corpus (embedded with the
/// real MiniLM model, not a synthetic stand-in): `cargo run --release
/// --example url_rag_server`, then driving `POST /graph/explore` over every
/// one of the 48 seed records' 2256 ordered `(key_a, key_b)` pairs on a
/// freshly started server, currently finds a `geodesic` for all 2256 pairs
/// -- zero unreachable, hop counts ranging 1-5 (mean ~2.5) -- and every one
/// of those geodesics' `bifurcation_ratio()` comparisons comes back with a
/// finite `separation_ratio` (never the `+-inf` edge case
/// `BifurcationView::separation_ratio`'s `None` guards against). This is a
/// live, re-checkable fact about the current corpus/model/algorithm, not a
/// guarantee -- unlike `BOUNDARY_K_NEAREST`'s own note, there's no
/// synthetic network-free test pinning full connectivity itself (that would
/// mean asserting a *global* graph property across a whole corpus, not a
/// *local* per-node property like `boundary_classes`' same-cluster-collision
/// shape), but `geodesic_diagnostics_reports_graph_stats_even_when_
/// unreachable` below still exercises the `None` (unreachable) branch
/// directly, on a synthetic corpus deliberately built to disconnect at this
/// constant's value (`per_cluster = GRAPH_K_NEAREST + 2`), so that code path
/// stays covered independent of whether the real corpus happens to trigger
/// it.
const GRAPH_K_NEAREST: usize = 4;

/// Numerical floor for the L2 norm of `key_b`'s embedding minus `key_a`'s,
/// checked before normalizing into a direction in
/// `Corpus::build_steering_vector` -- guards against dividing by (near)
/// zero when the two picked records have (near) identical embeddings, which
/// would otherwise silently produce a direction of NaN/Inf rather than a
/// clear error.
const STEERING_MIN_DIRECTION_NORM: f32 = 1e-6;
/// Passed to `SteeringVector::new`'s `norm_tol`: how far the direction's
/// normalized L2 norm may drift from 1.0 (float rounding from the
/// subtract-then-normalize pipeline) before being rejected. Generous
/// relative to the rounding actually involved, since a rejection here would
/// surface as a confusing internal error rather than the deliberate
/// "these two records are too similar" `DegenerateDirection` case above.
const STEERING_NORM_TOL: f32 = 1e-3;

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

    /// Builds a `ViableGraph` over every currently-stored record (predicate:
    /// accept all, per Issue #15) and partitions it via `boundary_classes()`,
    /// translating every graph node's `LatentDb`-internal id back to its
    /// stable `RecordKey` and current metadata. Ascending class id, and
    /// ascending id (so insertion order among ties) within each class,
    /// inherited from `BoundaryClasses::classes()`'s own ordering. Rebuilt
    /// fresh on every call -- cheap at demo corpus scale (same reasoning as
    /// `compression_ratio()`/`merkle_root()` in `integrity_handler`) -- so
    /// this is always current against whatever `db` currently holds, no
    /// separate invalidation needed after an ingest's rebuild.
    fn boundary_classes(&self) -> Vec<(BoundaryClassId, Vec<(RecordKey, String)>)> {
        let graph = self
            .db
            .build_viable_graph(|_| true, BOUNDARY_K_NEAREST, false);
        let partition = graph.boundary_classes();
        partition
            .classes()
            .into_iter()
            .enumerate()
            .map(|(class_id, ids)| (class_id as BoundaryClassId, self.translate_path(&ids)))
            .collect()
    }

    /// Looks up `key`'s full-precision embedding in `records` (append-only,
    /// so a key issued once stays resolvable forever). A linear scan rather
    /// than a `HashMap`, matching the demo's other O(n)-at-demo-scale reads
    /// (e.g. `RecordArena::lexical_rank`) -- adding a `RecordKey ->
    /// embedding` index isn't worth it at this corpus size, and would be
    /// another piece of state to keep in sync on every `ingest`.
    fn raw_embedding(&self, key: RecordKey) -> Option<&[f32]> {
        self.records
            .iter()
            .find(|r| r.key == key)
            .map(|r| r.embedding.as_slice())
    }

    /// Builds a `SteeringVector` whose direction is `key_b`'s embedding
    /// minus `key_a`'s, unit-normalized -- "steer search results from A's
    /// concept toward B's" (Issue #16). Reads each record's raw,
    /// full-precision embedding via `raw_embedding` rather than
    /// `LatentDb::get_approx_vector`'s PQ-decoded approximation: with this
    /// demo's PQ training budget (`PQ_CENTROIDS` quantization levels
    /// trained over however many records currently exist), two same-topic
    /// records can legitimately decode to identical codes, which would
    /// otherwise hand a viewer a spurious "no direction" error for a
    /// perfectly reasonable pick.
    ///
    /// This is the reusable piece Issue #16 owns: Issue #17's Graph
    /// Explorer tab calls this same method directly for its steered-walk
    /// toggle, rather than reimplementing the construction.
    fn build_steering_vector(
        &self,
        key_a: RecordKey,
        key_b: RecordKey,
        alpha: f32,
    ) -> Result<SteeringVector, SteeringBuildError> {
        let embedding_a = self
            .raw_embedding(key_a)
            .ok_or(SteeringBuildError::UnknownKey(key_a))?;
        let embedding_b = self
            .raw_embedding(key_b)
            .ok_or(SteeringBuildError::UnknownKey(key_b))?;

        let mut direction: Vec<f32> = embedding_b
            .iter()
            .zip(embedding_a.iter())
            .map(|(b, a)| b - a)
            .collect();
        let norm = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < STEERING_MIN_DIRECTION_NORM {
            return Err(SteeringBuildError::DegenerateDirection);
        }
        for x in &mut direction {
            *x /= norm;
        }

        SteeringVector::new(direction, alpha, STEERING_NORM_TOL)
            .map_err(SteeringBuildError::Steering)
    }

    /// Translates a list of `LatentDb`-internal ids (a `ViableGraph` path, or
    /// a `boundary_classes()` group) back to stable `(RecordKey, metadata)`
    /// pairs. Shared by `boundary_classes` and `geodesic_diagnostics` (which
    /// alone needs it twice: the geodesic path and its comparison walk).
    fn translate_path(&self, path: &[u64]) -> Vec<(RecordKey, String)> {
        path.iter()
            .filter_map(|&id| {
                let key = *self.id_to_key.get(&id)?;
                let metadata = self.db.get_metadata(id)?.to_string();
                Some((key, metadata))
            })
            .collect()
    }

    /// Builds a `ViableGraph` over every currently-stored record (predicate:
    /// accept all, same as `boundary_classes`, `k_nearest` = `GRAPH_K_NEAREST`
    /// -- Issue #17's own graph, deliberately denser than Boundary Classes'),
    /// then computes `geodesic(key_a, key_b)` plus its `trajectory` geometry
    /// and a `bifurcation_ratio()` comparison, all translated back to stable
    /// `RecordKey`s.
    ///
    /// Returns `Err(key)` naming whichever of `key_a`/`key_b` was never
    /// issued. `GraphExploreResult::geodesic` is `None` (not an error) when
    /// both keys are valid but `key_b` is unreachable from `key_a` in this
    /// graph -- an ordinary outcome for a sparsely-connected corpus, not a
    /// failure; `n_nodes`/`n_edges` are still reported in that case.
    ///
    /// The comparison path's anchor-at-`path[1]` construction is explained
    /// in this file's module doc. Walking `hops` steps from that anchor
    /// keeps it the same length as the geodesic (`bifurcation_ratio`
    /// requires equal lengths), so there's nothing to compare when the
    /// geodesic is trivial (`key_a == key_b`, zero hops) -- `comparison` is
    /// `None` in that case.
    fn geodesic_diagnostics(
        &self,
        key_a: RecordKey,
        key_b: RecordKey,
        seed: u64,
    ) -> Result<GraphExploreResult, RecordKey> {
        let id_a = *self.key_to_id.get(&key_a).ok_or(key_a)?;
        let id_b = *self.key_to_id.get(&key_b).ok_or(key_b)?;

        let graph = self.db.build_viable_graph(|_| true, GRAPH_K_NEAREST, false);
        let n_nodes = graph.n_nodes();
        let n_edges = graph.n_edges();

        let geodesic = graph.geodesic(id_a, id_b).map(|path| {
            let hops = path.len() - 1;
            let geometry = path_geometry(&graph, &path);

            let comparison = if hops == 0 {
                None
            } else {
                let anchor = path[1];
                let comparison_path = graph.random_walk(anchor, hops, seed);
                let comparison_geometry = path_geometry(&graph, &comparison_path);
                let bifurcation = bifurcation_ratio(&graph, &path, &comparison_path);
                Some(GraphComparisonResult {
                    path: self.translate_path(&comparison_path),
                    geometry: comparison_geometry,
                    bifurcation,
                })
            };

            GeodesicResult {
                path: self.translate_path(&path),
                hops,
                geometry,
                comparison,
            }
        });

        Ok(GraphExploreResult {
            n_nodes,
            n_edges,
            geodesic,
        })
    }

    /// Builds a `SteeringVector` from `key_a` -> `key_b` via
    /// `build_steering_vector` (Issue #16's reusable piece -- not
    /// reimplemented here), then runs a `weighted_random_walk()` from
    /// `key_a`, biased by that vector's direction, next to a plain
    /// `random_walk()` from the same start for comparison (Issue #17's Walk
    /// mode).
    ///
    /// The weight function scores each walk candidate by the cosine
    /// similarity between its graph coordinates and the steering direction
    /// -- the same construction `manifold::tests::
    /// weighted_random_walk_biased_toward_a_steering_direction_beats_a_plain_random_walk`
    /// validates biases a walk toward the steered-toward concept.
    fn steered_walk(
        &self,
        key_a: RecordKey,
        key_b: RecordKey,
        alpha: f32,
        steps: usize,
        seed: u64,
    ) -> Result<GraphWalkResult, SteeringBuildError> {
        let steering = self.build_steering_vector(key_a, key_b, alpha)?;
        // `build_steering_vector` already resolved `key_a` via
        // `raw_embedding` (an `Ok` above implies it exists in `records`),
        // and every key in `records` is re-registered into `key_to_id` on
        // every rebuild (`build_db`), so this lookup can't fail here.
        let id_a = *self
            .key_to_id
            .get(&key_a)
            .expect("key_a resolved by build_steering_vector must also be in key_to_id");

        let graph = self.db.build_viable_graph(|_| true, GRAPH_K_NEAREST, false);
        let biased = graph.weighted_random_walk(id_a, steps, seed, |_, cand| {
            graph
                .coords_of(cand)
                .map(|c| cosine_sim(c, steering.as_slice()))
                .unwrap_or(0.0)
        });
        let plain = graph.random_walk(id_a, steps, seed);

        Ok(GraphWalkResult {
            steering,
            biased: self.translate_path(&biased),
            plain: self.translate_path(&plain),
        })
    }
}

/// Return value of `Corpus::geodesic_diagnostics`, translated to stable
/// `RecordKey`s -- kept as a plain struct (not the HTTP wire view) so
/// `Corpus`'s own tests can assert on it without going through JSON.
#[derive(Debug)]
struct GraphExploreResult {
    n_nodes: usize,
    n_edges: usize,
    geodesic: Option<GeodesicResult>,
}

#[derive(Debug)]
struct GeodesicResult {
    path: Vec<(RecordKey, String)>,
    hops: usize,
    geometry: PathGeometry,
    comparison: Option<GraphComparisonResult>,
}

#[derive(Debug)]
struct GraphComparisonResult {
    path: Vec<(RecordKey, String)>,
    geometry: PathGeometry,
    bifurcation: BifurcationResult,
}

/// Return value of `Corpus::steered_walk`.
#[derive(Debug)]
struct GraphWalkResult {
    steering: SteeringVector,
    biased: Vec<(RecordKey, String)>,
    plain: Vec<(RecordKey, String)>,
}

/// Failure modes for `Corpus::build_steering_vector`, mapped to distinct
/// HTTP statuses by `steering_search_handler` (see
/// `SteeringSearchError::into_response`) rather than collapsed into one
/// generic 500 -- an unknown key or two near-identical picks are both
/// ordinary client-facing outcomes in this demo, not internal errors.
#[derive(Debug)]
enum SteeringBuildError {
    /// `key_a` or `key_b` was never issued by this `Corpus` (typo'd, or
    /// from a session against a different server instance).
    UnknownKey(RecordKey),
    /// `key_a` and `key_b` have (near) identical raw embeddings, so
    /// there's no meaningful direction to normalize between them.
    DegenerateDirection,
    /// `SteeringVector::new` itself rejected the (already unit-normalized)
    /// direction or `alpha` -- reachable only if `alpha` is out of
    /// `[0.0, 1.0]`, since the direction is normalized just above.
    Steering(SteeringError),
}

impl std::fmt::Display for SteeringBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SteeringBuildError::UnknownKey(key) => write!(f, "unknown record key {}", key.0),
            SteeringBuildError::DegenerateDirection => write!(
                f,
                "the two picked records have (near) identical embeddings -- no direction to \
                 steer along"
            ),
            SteeringBuildError::Steering(e) => write!(f, "{e}"),
        }
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

/// Embeds a single `query` string through `state.embedder`, locking and
/// releasing the embedder entirely within this call -- shared by
/// `run_search` and `run_steering_search`, the two handlers that each embed
/// exactly one free-text query before searching.
fn embed_single_query(state: &AppState, query: &str) -> Result<Vec<f32>> {
    let mut embedder = state.embedder.lock().unwrap();
    embedder
        .embed(&[query])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("embedding a single query produced no vector"))
}

/// Embeds `query`, then runs both rankings against the current db: the
/// plain-vector `search()` and the lexically blended `search_blended()`
/// (using `query`'s own words, split on whitespace, as the lexical terms --
/// `search_blended` lowercases them itself). Locks are taken and released
/// entirely within this synchronous function, mirroring `run_ingest`.
fn run_search(state: &AppState, query: &str) -> Result<SearchResponse> {
    let embedding = embed_single_query(state, query)?;

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

// ---------------------------------------------------------------------
// Boundary Classes (Issue #15): structural partition of the current
// `Corpus::db` into `manifold::BoundaryClasses`, keyed by `RecordKey` so a
// viewer sees stable identifiers even across a later rebuild.
// ---------------------------------------------------------------------

#[derive(Serialize)]
struct BoundaryClassRecordView {
    key: RecordKey,
    metadata: String,
}

#[derive(Serialize)]
struct BoundaryClassGroup {
    class_id: BoundaryClassId,
    records: Vec<BoundaryClassRecordView>,
}

#[derive(Serialize)]
struct BoundaryClassesResponse {
    groups: Vec<BoundaryClassGroup>,
}

/// No `spawn_blocking`, same as `integrity_handler`: no model/network I/O.
/// Unlike `integrity_handler`'s near-O(1) cached-tree read, this is an
/// O(n^2) pairwise-distance graph build plus a signature-refinement pass --
/// still cheap in wall-clock terms at demo corpus scale (tens to low
/// hundreds of records), just not free the way a Merkle root read is.
async fn boundary_classes_handler(
    State(state): State<Arc<AppState>>,
) -> Json<BoundaryClassesResponse> {
    let corpus = state.corpus.lock().unwrap();
    let groups = corpus
        .boundary_classes()
        .into_iter()
        .map(|(class_id, records)| BoundaryClassGroup {
            class_id,
            records: records
                .into_iter()
                .map(|(key, metadata)| BoundaryClassRecordView { key, metadata })
                .collect(),
        })
        .collect();
    Json(BoundaryClassesResponse { groups })
}

// ---------------------------------------------------------------------
// Steering (Issue #16): `Corpus::build_steering_vector` (defined above,
// alongside `Corpus`, since Issue #17's Graph Explorer tab calls it
// directly too) plus the HTTP layer comparing plain `search()` against
// `search_steered()` for the same query.
// ---------------------------------------------------------------------

/// Wire view of a built `SteeringVector`'s identity -- not its raw
/// direction floats (no reason to ship those to the browser). `commitment`
/// hashes `direction` *and* `alpha` together (see `compute_commitment` in
/// the `steering` module), so it changes on every `alpha` tick even for a
/// fixed `key_a`/`key_b` pair -- shown so a viewer can see each slider move
/// really did build a distinct `SteeringVector`, not just relabel the same
/// one.
#[derive(Serialize)]
struct SteeringVectorView {
    dim: usize,
    alpha: f32,
    /// Lowercase hex encoding of `SteeringVector::commitment()`.
    commitment: String,
}

/// Builds the wire view of `v`, mirroring `hits_to_views`'s plain-function
/// (not `From`) conversion style used elsewhere in this file.
fn steering_vector_view(v: &SteeringVector) -> SteeringVectorView {
    SteeringVectorView {
        dim: v.dim(),
        alpha: v.alpha(),
        commitment: hex(&v.commitment()),
    }
}

#[derive(Deserialize)]
struct SteeringSearchRequest {
    key_a: RecordKey,
    key_b: RecordKey,
    alpha: f32,
    query: String,
}

#[derive(Serialize)]
struct SteeringSearchResponse {
    /// The `SteeringVector` built from `key_a` -> `key_b` for this request.
    steering: SteeringVectorView,
    /// Plain `LatentDb::search()` ranking over `query`'s embedding,
    /// unsteered.
    plain: Vec<SearchHitView>,
    /// `LatentDb::search_steered()` ranking over the same query embedding,
    /// steered by `steering`.
    steered: Vec<SearchHitView>,
}

/// Maps a `SteeringBuildError` to its HTTP status: an unknown key is a 404
/// (nothing there to find), a degenerate pick or an out-of-range alpha is a
/// 400 (the request itself is malformed) -- shared between
/// `SteeringSearchError::into_response` and `graph_walk_handler`, the two
/// handlers that each wrap `Corpus::build_steering_vector` (directly, or via
/// `steered_walk`) and need to report the same failure the same way.
fn steering_build_error_response(e: SteeringBuildError) -> (StatusCode, String) {
    let status = match e {
        SteeringBuildError::UnknownKey(_) => StatusCode::NOT_FOUND,
        SteeringBuildError::DegenerateDirection | SteeringBuildError::Steering(_) => {
            StatusCode::BAD_REQUEST
        }
    };
    (status, e.to_string())
}

/// Everything that can go wrong building+running a steered search, kept
/// distinct from `anyhow::Error` (unlike `run_ingest`/`run_search`) so
/// `steering_search_handler` can report an unknown key or a degenerate pick
/// as a client-facing 4xx rather than collapsing every failure into a 500 --
/// both are ordinary outcomes of typing a stray key into this tab's inputs.
enum SteeringSearchError {
    Embed(anyhow::Error),
    Build(SteeringBuildError),
}

impl SteeringSearchError {
    fn into_response(self) -> (StatusCode, String) {
        match self {
            // `{e:#}`, same as `run_ingest`/`run_search`'s own mapping, so a
            // chained anyhow error keeps its full context.
            SteeringSearchError::Embed(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
            SteeringSearchError::Build(e) => steering_build_error_response(e),
        }
    }
}

async fn steering_search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SteeringSearchRequest>,
) -> Result<Json<SteeringSearchResponse>, (StatusCode, String)> {
    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "query must not be empty".to_string(),
        ));
    }

    tokio::task::spawn_blocking(move || {
        run_steering_search(&state, req.key_a, req.key_b, req.alpha, &query)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map(Json)
    .map_err(SteeringSearchError::into_response)
}

/// Embeds `query`, builds a `SteeringVector` from `key_a` -> `key_b` via
/// `Corpus::build_steering_vector`, then runs both `search()` and
/// `search_steered()` against it -- locks taken and released entirely
/// within this synchronous function, mirroring `run_ingest`/`run_search`.
fn run_steering_search(
    state: &AppState,
    key_a: RecordKey,
    key_b: RecordKey,
    alpha: f32,
    query: &str,
) -> Result<SteeringSearchResponse, SteeringSearchError> {
    let embedding = embed_single_query(state, query).map_err(SteeringSearchError::Embed)?;

    let corpus = state.corpus.lock().unwrap();
    let steering = corpus
        .build_steering_vector(key_a, key_b, alpha)
        .map_err(SteeringSearchError::Build)?;
    let nprobe = corpus.db.n_index_centroids();
    let plain_hits = corpus.db.search(&embedding, SEARCH_K, nprobe);
    let steered_hits = corpus
        .db
        .search_steered(&embedding, SEARCH_K, nprobe, &steering);
    let plain = hits_to_views(plain_hits, &corpus.id_to_key);
    let steered = hits_to_views(steered_hits, &corpus.id_to_key);
    drop(corpus);

    Ok(SteeringSearchResponse {
        steering: steering_vector_view(&steering),
        plain,
        steered,
    })
}

// ---------------------------------------------------------------------
// Graph Explorer (Issue #17): `Corpus::geodesic_diagnostics`/`steered_walk`
// (defined above, alongside `Corpus`) plus the HTTP layer for the tab's two
// modes -- Path (geodesic + trajectory geometry + bifurcation_ratio) and
// Walk (steered vs. plain random walk).
// ---------------------------------------------------------------------

/// Wire view of one record on a rendered graph path/walk.
#[derive(Serialize)]
struct GraphNodeView {
    key: RecordKey,
    metadata: String,
}

fn graph_node_views(path: Vec<(RecordKey, String)>) -> Vec<GraphNodeView> {
    path.into_iter()
        .map(|(key, metadata)| GraphNodeView { key, metadata })
        .collect()
}

/// Wire view of a `trajectory::PathGeometry`.
#[derive(Serialize)]
struct PathGeometryView {
    length: f32,
    mean_curvature: f32,
    min_adjacent_cosine: f32,
    n_steps: usize,
}

fn path_geometry_view(g: &PathGeometry) -> PathGeometryView {
    PathGeometryView {
        length: g.length,
        mean_curvature: g.mean_curvature,
        min_adjacent_cosine: g.min_adjacent_cosine,
        n_steps: g.n_steps,
    }
}

/// Wire view of a `trajectory::BifurcationResult`. `separation_ratio` is
/// `None` (rather than a raw `f32`) whenever it's non-finite -- `serde_json`
/// can't represent `+-inf` (that edge case fires when the comparison path's
/// first step lands exactly on the geodesic's own start coordinate, an
/// `initial_sep <= epsilon` `bifurcation_ratio` treats specially, see its
/// doc comment) and would otherwise silently serialize it as JSON `null`
/// for the UI's `.toFixed()` call to crash on, rather than a value this view
/// deliberately marks absent.
#[derive(Serialize)]
struct BifurcationView {
    separation_ratio: Option<f32>,
    onset_step: Option<usize>,
    final_separation: f32,
}

fn bifurcation_view(b: &BifurcationResult) -> BifurcationView {
    BifurcationView {
        separation_ratio: b.separation_ratio.is_finite().then_some(b.separation_ratio),
        onset_step: b.onset_step,
        final_separation: b.final_separation,
    }
}

#[derive(Deserialize)]
struct GraphExploreRequest {
    key_a: RecordKey,
    key_b: RecordKey,
    seed: u64,
}

#[derive(Serialize)]
struct GraphComparisonView {
    path: Vec<GraphNodeView>,
    geometry: PathGeometryView,
    bifurcation: BifurcationView,
}

#[derive(Serialize)]
struct GeodesicView {
    path: Vec<GraphNodeView>,
    hops: usize,
    geometry: PathGeometryView,
    /// `None` when `hops == 0` (`key_a == key_b`) -- no second node to
    /// anchor a same-length comparison path at, see
    /// `Corpus::geodesic_diagnostics`.
    comparison: Option<GraphComparisonView>,
}

#[derive(Serialize)]
struct GraphExploreResponse {
    n_nodes: usize,
    n_edges: usize,
    /// `None` when `key_b` is unreachable from `key_a` in this graph -- an
    /// ordinary outcome, not an error (see `Corpus::geodesic_diagnostics`).
    geodesic: Option<GeodesicView>,
}

/// No `spawn_blocking`, same reasoning as `boundary_classes_handler`: an
/// O(n^2) graph build plus a geodesic/geometry pass, no model/network I/O.
async fn graph_explore_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GraphExploreRequest>,
) -> Result<Json<GraphExploreResponse>, (StatusCode, String)> {
    let corpus = state.corpus.lock().unwrap();
    let result = corpus
        .geodesic_diagnostics(req.key_a, req.key_b, req.seed)
        .map_err(|bad_key| {
            (
                StatusCode::NOT_FOUND,
                format!("unknown record key {}", bad_key.0),
            )
        })?;
    drop(corpus);

    let geodesic = result.geodesic.map(|r| GeodesicView {
        path: graph_node_views(r.path),
        hops: r.hops,
        geometry: path_geometry_view(&r.geometry),
        comparison: r.comparison.map(|c| GraphComparisonView {
            path: graph_node_views(c.path),
            geometry: path_geometry_view(&c.geometry),
            bifurcation: bifurcation_view(&c.bifurcation),
        }),
    });

    Ok(Json(GraphExploreResponse {
        n_nodes: result.n_nodes,
        n_edges: result.n_edges,
        geodesic,
    }))
}

#[derive(Deserialize)]
struct GraphWalkRequest {
    key_a: RecordKey,
    key_b: RecordKey,
    alpha: f32,
    steps: usize,
    seed: u64,
}

#[derive(Serialize)]
struct GraphWalkResponse {
    steering: SteeringVectorView,
    biased: Vec<GraphNodeView>,
    plain: Vec<GraphNodeView>,
}

/// No `spawn_blocking`, same reasoning as `graph_explore_handler`.
async fn graph_walk_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GraphWalkRequest>,
) -> Result<Json<GraphWalkResponse>, (StatusCode, String)> {
    let corpus = state.corpus.lock().unwrap();
    let result = corpus
        .steered_walk(req.key_a, req.key_b, req.alpha, req.steps, req.seed)
        .map_err(steering_build_error_response)?;
    drop(corpus);

    Ok(Json(GraphWalkResponse {
        steering: steering_vector_view(&result.steering),
        biased: graph_node_views(result.biased),
        plain: graph_node_views(result.plain),
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
        .route("/boundary-classes", get(boundary_classes_handler))
        .route("/steering/search", post(steering_search_handler))
        .route("/graph/explore", post(graph_explore_handler))
        .route("/graph/walk", post(graph_walk_handler))
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

    /// `n_clusters` well-separated groups of `per_cluster` records each,
    /// tagged with the cluster index they belong to -- each cluster's center
    /// is spaced `5.0` apart per dimension per cluster index, jitter scaled
    /// down to `0.05x`, far enough apart that `BOUNDARY_K_NEAREST`'s kNN
    /// pass (verified empirically against this exact generator while
    /// building Issue #15) never crosses a cluster boundary.
    fn synthetic_clustered_records(
        n_clusters: usize,
        per_cluster: usize,
        dim: usize,
    ) -> Vec<(Vec<f32>, usize, String)> {
        let mut out = Vec::with_capacity(n_clusters * per_cluster);
        for c in 0..n_clusters {
            let center: Vec<f32> = (0..dim).map(|d| ((c * 37 + d) as f32) * 5.0).collect();
            for i in 0..per_cluster {
                let jitter = synthetic_vector(dim, (c * 1000 + i) as u64);
                let point: Vec<f32> = center
                    .iter()
                    .zip(jitter.iter())
                    .map(|(&cc, &j)| cc + j * 0.05)
                    .collect();
                out.push((point, c, format!("cluster-{c}-{i}")));
            }
        }
        out
    }

    /// The core correctness requirement from Issue #15's acceptance
    /// criteria: on a corpus with real cluster structure, boundary classes
    /// that group more than one record together are always siblings from
    /// the same cluster -- confirming the partition reflects the underlying
    /// data's real structure rather than being an arbitrary or fake
    /// grouping. `boundary_classes()` is a structural-equivalence partition
    /// (not vector-similarity clustering, see `Corpus::boundary_classes`),
    /// so a same-cluster collision isn't guaranteed for every cluster on
    /// every corpus -- but on this fixed-seed generator, at least one
    /// genuinely does collide (asserted below), so this test exercises real
    /// grouping, not a partition that only ever produces singletons.
    #[test]
    fn boundary_classes_never_mixes_two_different_clusters_into_one_class() {
        let clustered = synthetic_clustered_records(4, 8, 16);
        let seed: Vec<(Vec<f32>, String)> = clustered
            .iter()
            .map(|(v, _, m)| (v.clone(), m.clone()))
            .collect();
        let corpus = Corpus::seed(seed).expect("seed build should succeed");

        let cluster_of_key: HashMap<RecordKey, usize> = corpus
            .records
            .iter()
            .zip(clustered.iter())
            .map(|(r, (_, cluster, _))| (r.key, *cluster))
            .collect();

        let groups = corpus.boundary_classes();

        let total_grouped: usize = groups.iter().map(|(_, records)| records.len()).sum();
        assert_eq!(
            total_grouped,
            corpus.records.len(),
            "every record should appear in exactly one boundary class group"
        );

        let mut saw_a_same_cluster_collision = false;
        for (class_id, records) in &groups {
            let clusters_in_group: Vec<usize> =
                records.iter().map(|(key, _)| cluster_of_key[key]).collect();
            if clusters_in_group.len() > 1 {
                saw_a_same_cluster_collision = true;
                let first = clusters_in_group[0];
                assert!(
                    clusters_in_group.iter().all(|&c| c == first),
                    "boundary class {class_id} mixed records from different clusters: \
                     {clusters_in_group:?}"
                );
            }
        }
        assert!(
            saw_a_same_cluster_collision,
            "expected at least one boundary class to contain more than one record from the \
             same cluster on this fixed-seed corpus -- otherwise this test can't tell real \
             grouping from a partition that only ever produces singletons"
        );
    }

    /// The second acceptance criterion: a `boundary_classes()` computed
    /// after an ingest rebuilds `db` reflects the *current* corpus, not a
    /// stale pre-ingest snapshot -- exercised here by ingesting a pair of
    /// near-duplicate records far from every existing record and confirming
    /// they land in the same class post-ingest.
    #[test]
    fn boundary_classes_reflects_a_newly_ingested_cluster() {
        let seed = synthetic_records(5, 16);
        let mut corpus = Corpus::seed(seed).expect("seed build should succeed");

        let dim = 16;
        let far_center: Vec<f32> = (0..dim).map(|d| (d as f32) * 50.0).collect();
        let new_pair: Vec<(Vec<f32>, String)> = (0..2u64)
            .map(|i| {
                let jitter = synthetic_vector(dim, 9000 + i);
                let point: Vec<f32> = far_center
                    .iter()
                    .zip(jitter.iter())
                    .map(|(&c, &j)| c + j * 0.01)
                    .collect();
                (point, format!("new-{i}"))
            })
            .collect();
        let new_keys = corpus.ingest(new_pair).expect("ingest should succeed");
        let [key_a, key_b] = new_keys[..] else {
            panic!("ingest should have assigned exactly 2 keys");
        };

        let groups = corpus.boundary_classes();
        let class_of = |key: RecordKey| {
            groups
                .iter()
                .find(|(_, records)| records.iter().any(|(k, _)| *k == key))
                .map(|(class_id, _)| *class_id)
        };
        assert!(
            class_of(key_a).is_some() && class_of(key_a) == class_of(key_b),
            "the two newly-ingested near-duplicate records should share a boundary class after \
             the ingest-triggered rebuild"
        );
    }

    #[test]
    fn build_steering_vector_points_from_a_toward_b_and_is_unit_norm() {
        let seed = synthetic_records(10, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let key_b = corpus.records[1].key;

        let steering = corpus
            .build_steering_vector(key_a, key_b, 0.5)
            .expect("two distinct synthetic records should produce a valid direction");

        assert!(steering.verify(STEERING_NORM_TOL));
        assert_eq!(steering.alpha(), 0.5);

        // The built direction should point the same way as the raw
        // (un-normalized) B-minus-A difference -- a positive dot product,
        // not merely "some unit vector".
        let raw_diff: Vec<f32> = corpus.records[1]
            .embedding
            .iter()
            .zip(corpus.records[0].embedding.iter())
            .map(|(b, a)| b - a)
            .collect();
        let dot: f32 = steering
            .as_slice()
            .iter()
            .zip(raw_diff.iter())
            .map(|(s, d)| s * d)
            .sum();
        assert!(
            dot > 0.0,
            "direction should point from A toward B, not away from it (dot={dot})"
        );
    }

    #[test]
    fn build_steering_vector_rejects_an_unknown_key() {
        let seed = synthetic_records(5, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let bogus = RecordKey(9999);

        let err = corpus
            .build_steering_vector(key_a, bogus, 0.5)
            .expect_err("an unissued key should be rejected");
        assert!(matches!(err, SteeringBuildError::UnknownKey(k) if k == bogus));
    }

    #[test]
    fn build_steering_vector_rejects_two_identical_keys() {
        let seed = synthetic_records(5, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;

        let err = corpus
            .build_steering_vector(key_a, key_a, 0.5)
            .expect_err("the same key on both sides has a zero direction");
        assert!(matches!(err, SteeringBuildError::DegenerateDirection));
    }

    /// The core correctness requirement from Issue #16's acceptance
    /// criteria: steering toward record B's own direction should visibly
    /// raise B's rank/score relative to plain `search()`, using a
    /// `SteeringVector` built end-to-end through `Corpus::build_steering_vector`
    /// rather than constructed by hand -- `src/db.rs`'s own
    /// `search_steered_toward_a_records_own_direction_raises_its_score`
    /// already pins this at the `LatentDb` layer; this test pins that this
    /// demo's own construction (`Corpus::build_steering_vector`, using raw
    /// embeddings rather than PQ-decoded ones) wires correctly into it.
    #[test]
    fn corpus_built_steering_vector_raises_the_target_records_score() {
        let seed = synthetic_records(30, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let key_b = corpus.records[10].key;
        let id_b = *corpus.key_to_id.get(&key_b).unwrap();
        let query = corpus.records[0].embedding.clone();

        let baseline = corpus
            .build_steering_vector(key_a, key_b, 0.0)
            .expect("build should succeed");
        let strong = corpus
            .build_steering_vector(key_a, key_b, 1.0)
            .expect("build should succeed");

        let nprobe = corpus.db.n_index_centroids();
        let plain = corpus
            .db
            .search_steered(&query, corpus.db.len(), nprobe, &baseline);
        let steered = corpus
            .db
            .search_steered(&query, corpus.db.len(), nprobe, &strong);

        let plain_score = plain.iter().find(|h| h.id == id_b).unwrap().score;
        let steered_score = steered.iter().find(|h| h.id == id_b).unwrap().score;
        assert!(
            steered_score > plain_score,
            "steering the query from A toward B should raise B's score \
             ({steered_score} vs {plain_score})"
        );
    }

    /// Pins the specific claim the Steering tab's UI makes -- that the
    /// alpha slider "visibly re-ranks results as it moves" (Issue #16) --
    /// at the same `SEARCH_K`-truncated width the tab actually renders,
    /// rather than at `db.len()` like the score-comparison test above.
    /// Uses a query distinct from both picked records (unlike that test,
    /// which queries with A's own embedding), closer to how a viewer would
    /// actually drive this tab: type an unrelated query, then compare.
    #[test]
    fn corpus_built_steering_vector_visibly_reranks_the_rendered_top_k() {
        let seed = synthetic_records(40, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let key_b = corpus.records[20].key;
        let id_b = *corpus.key_to_id.get(&key_b).unwrap();
        let query = corpus.records[5].embedding.clone();

        let off = corpus
            .build_steering_vector(key_a, key_b, 0.0)
            .expect("build should succeed");
        let strong = corpus
            .build_steering_vector(key_a, key_b, 1.0)
            .expect("build should succeed");

        let nprobe = corpus.db.n_index_centroids();
        let plain_top = corpus.db.search_steered(&query, SEARCH_K, nprobe, &off);
        let steered_top = corpus.db.search_steered(&query, SEARCH_K, nprobe, &strong);

        let plain_has_b = plain_top.iter().any(|h| h.id == id_b);
        let steered_has_b = steered_top.iter().any(|h| h.id == id_b);
        assert!(
            !plain_has_b && steered_has_b,
            "steering strongly toward B should bring it into the rendered top-{SEARCH_K} \
             even though it isn't there in the plain ranking \
             (plain_has_b={plain_has_b}, steered_has_b={steered_has_b})"
        );
    }

    // Corpus::geodesic_diagnostics (Issue #17, Path mode)

    #[test]
    fn geodesic_diagnostics_rejects_an_unknown_key_a() {
        let seed = synthetic_records(10, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_b = corpus.records[1].key;
        let bogus = RecordKey(9999);

        let err = corpus
            .geodesic_diagnostics(bogus, key_b, 1)
            .expect_err("an unissued key_a should be rejected");
        assert_eq!(err, bogus);
    }

    #[test]
    fn geodesic_diagnostics_rejects_an_unknown_key_b() {
        let seed = synthetic_records(10, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let bogus = RecordKey(9999);

        let err = corpus
            .geodesic_diagnostics(key_a, bogus, 1)
            .expect_err("an unissued key_b should be rejected");
        assert_eq!(err, bogus);
    }

    #[test]
    fn geodesic_diagnostics_trivial_path_when_keys_are_identical() {
        let seed = synthetic_records(10, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;

        let result = corpus
            .geodesic_diagnostics(key_a, key_a, 1)
            .expect("both keys are valid");
        let geodesic = result
            .geodesic
            .expect("a record is trivially reachable from itself");
        assert_eq!(geodesic.hops, 0);
        assert_eq!(geodesic.path.len(), 1);
        assert_eq!(geodesic.path[0].0, key_a);
        assert!(
            geodesic.comparison.is_none(),
            "a zero-hop geodesic has no second node to anchor a comparison walk at"
        );
    }

    /// The core correctness requirement behind Issue #17's bifurcation
    /// diagnostic: the comparison path must *not* start at `key_a` itself
    /// (see this file's module doc for why -- that would put
    /// `bifurcation_ratio` in its zero-initial-separation edge case on every
    /// request) but at the geodesic's own second node, and it must be the
    /// same length as the geodesic path (`bifurcation_ratio` requires equal
    /// lengths to report anything but its default).
    #[test]
    fn geodesic_diagnostics_anchors_the_comparison_path_at_the_geodesics_second_node() {
        let seed = synthetic_records(25, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let key_b = corpus.records[15].key;

        let result = corpus
            .geodesic_diagnostics(key_a, key_b, 7)
            .expect("both keys are valid");
        let geodesic = result
            .geodesic
            .expect("a reasonably dense random synthetic graph should connect these two");
        assert!(
            geodesic.hops >= 1,
            "key_a != key_b should take at least one hop"
        );

        let comparison = geodesic
            .comparison
            .as_ref()
            .expect("a >=1-hop geodesic should always produce a comparison path");
        assert_eq!(
            comparison.path.len(),
            geodesic.path.len(),
            "bifurcation_ratio requires equal-length paths"
        );
        assert_eq!(
            comparison.path[0].0, geodesic.path[1].0,
            "the comparison path should start at the geodesic's own second node, not key_a"
        );
    }

    #[test]
    fn geodesic_diagnostics_reports_graph_stats_even_when_unreachable() {
        // Same reasoning as `boundary_classes_never_mixes_two_different_
        // clusters_into_one_class`'s generator, but with more same-cluster
        // neighbors than GRAPH_K_NEAREST so every node's kNN pass stays
        // entirely inside its own cluster -- four disconnected components.
        let per_cluster = GRAPH_K_NEAREST + 2;
        let clustered = synthetic_clustered_records(4, per_cluster, 16);
        let seed: Vec<(Vec<f32>, String)> = clustered
            .iter()
            .map(|(v, _, m)| (v.clone(), m.clone()))
            .collect();
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key; // cluster 0
        let key_b = corpus.records[per_cluster].key; // cluster 1

        let result = corpus
            .geodesic_diagnostics(key_a, key_b, 1)
            .expect("both keys are valid");
        assert_eq!(result.n_nodes, corpus.records.len());
        assert!(result.n_edges > 0);
        assert!(
            result.geodesic.is_none(),
            "the two picked records are in disconnected clusters"
        );
    }

    // Corpus::steered_walk (Issue #17, Walk mode)

    #[test]
    fn steered_walk_rejects_an_unknown_key() {
        let seed = synthetic_records(10, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let bogus = RecordKey(9999);

        let err = corpus
            .steered_walk(key_a, bogus, 1.0, 5, 1)
            .expect_err("an unissued key should be rejected");
        assert!(matches!(err, SteeringBuildError::UnknownKey(k) if k == bogus));
    }

    #[test]
    fn steered_walk_returns_a_valid_steering_vector_and_equal_length_walks() {
        let seed = synthetic_records(20, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let key_b = corpus.records[10].key;

        let result = corpus
            .steered_walk(key_a, key_b, 0.7, 6, 3)
            .expect("build should succeed");
        assert!(result.steering.verify(STEERING_NORM_TOL));
        assert_eq!(result.steering.alpha(), 0.7);
        assert_eq!(result.biased.len(), 7);
        assert_eq!(result.plain.len(), 7);
        assert_eq!(result.biased[0].0, key_a);
        assert_eq!(result.plain[0].0, key_a);
    }

    /// Pins that `steered_walk` wires `build_steering_vector`'s direction
    /// into `weighted_random_walk`'s weight function correctly (not, e.g.,
    /// inverted or ignored): steering from A toward a record that's already
    /// one of A's direct graph neighbors should make a single-step biased
    /// walk land on that neighbor far more often than a plain walk does,
    /// across many seeds -- the same "biased beats plain" claim
    /// `manifold::tests::weighted_random_walk_biased_toward_a_steering_
    /// direction_beats_a_plain_random_walk` pins at the library layer,
    /// exercised here end-to-end through this demo's own construction.
    #[test]
    fn steered_walk_biased_walk_favors_a_direct_neighbor_far_more_than_plain_walk() {
        let seed = synthetic_records(30, 16);
        let corpus = Corpus::seed(seed).expect("seed build should succeed");
        let key_a = corpus.records[0].key;
        let id_a = *corpus.key_to_id.get(&key_a).unwrap();

        let graph = corpus
            .db
            .build_viable_graph(|_| true, GRAPH_K_NEAREST, false);
        let neighbors = graph.neighbors(id_a);
        assert!(
            !neighbors.is_empty(),
            "a random 30-record graph should connect id_a"
        );
        let target_id = neighbors[0];
        let key_b = *corpus.id_to_key.get(&target_id).unwrap();

        let trials: u64 = 200;
        let mut biased_hits = 0u32;
        let mut plain_hits = 0u32;
        for trial_seed in 0..trials {
            let result = corpus
                .steered_walk(key_a, key_b, 1.0, 1, trial_seed)
                .expect("build should succeed");
            if result.biased[1].0 == key_b {
                biased_hits += 1;
            }
            if result.plain[1].0 == key_b {
                plain_hits += 1;
            }
        }
        let biased_rate = biased_hits as f32 / trials as f32;
        let plain_rate = plain_hits as f32 / trials as f32;
        assert!(
            biased_rate > plain_rate + 0.2,
            "a walk steered straight at a direct neighbor should land on it far more often \
             than a plain walk: biased_rate={biased_rate}, plain_rate={plain_rate}"
        );
    }
}
