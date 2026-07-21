//! Mini semantic-search demo: real text embeddings (via candle + a
//! pretrained MiniLM model, downloaded from the Hugging Face Hub on first
//! run) feeding into `LatentDb`.
//!
//! This is the "bring your own embedding step" half latent-db intentionally
//! doesn't provide itself (see README) -- everything in this file up to
//! `embed()`'s call site is generic sentence-embedding boilerplate, not
//! part of the latent-db crate. No LLM is used to generate an answer;
//! matched `metadata` (the original sentence) is shown directly, i.e. this
//! is semantic search, not a chatbot.
//!
//! Run with:
//!     cargo run --release --example mini_rag
//!
//! First run downloads ~90MB of model weights from the Hub and caches them
//! (subsequent runs are fast and fully offline).

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use latent_db::LatentDb;
use tokenizers::{PaddingParams, Tokenizer};

/// A small corpus spanning 3 distinct topics, so correct vs. random
/// retrieval is easy to tell apart at a glance.
const CORPUS: &[(&str, &str)] = &[
    ("cooking", "Preheat the oven to 200C before baking the bread."),
    ("cooking", "Add a pinch of salt to the pasta water while it boils."),
    ("cooking", "Simmer the tomato sauce for twenty minutes until it thickens."),
    ("cooking", "Whisk the eggs and sugar together until the mixture is pale."),
    ("cooking", "Marinate the chicken in soy sauce and garlic overnight."),
    ("astronomy", "Jupiter is the largest planet in the solar system."),
    ("astronomy", "A neutron star can form when a massive star collapses."),
    ("astronomy", "The Hubble telescope has captured images of distant galaxies."),
    ("astronomy", "Saturn's rings are made mostly of ice and rock particles."),
    ("astronomy", "Astronomers detected a new exoplanet orbiting a red dwarf."),
    ("finance", "The central bank raised interest rates to curb inflation."),
    ("finance", "Diversifying a portfolio reduces exposure to a single asset."),
    ("finance", "The company reported quarterly earnings above analyst expectations."),
    ("finance", "Bond yields fell after the latest jobs report was released."),
    ("finance", "Investors moved capital into index funds during the downturn."),
];

const QUERIES: &[(&str, &str)] = &[
    ("cooking", "What temperature should I set the oven to for baking?"),
    ("astronomy", "Tell me about planets and stars in space."),
    ("finance", "How do interest rates affect the stock market?"),
];

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
        let config_filename = repo.download_file().filename("config.json").revision(revision).send()?;
        let tokenizer_filename =
            repo.download_file().filename("tokenizer.json").revision(revision).send()?;
        let weights_filename =
            repo.download_file().filename("model.safetensors").revision(revision).send()?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self { model, tokenizer, device })
    }

    /// Embed a batch of sentences: tokenize -> BERT forward pass ->
    /// attention-mask-weighted mean pooling -> L2 normalize.
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
            .map(|t| Ok(Tensor::new(t.get_attention_mask().to_vec().as_slice(), &self.device)?))
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let hidden = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean-pool over tokens, excluding padding via the attention mask
        // (matches the `sentence_transformers` Python library's pooling).
        let mask = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = mask.sum(1)?;
        let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
        let pooled = summed.broadcast_div(&sum_mask)?;
        let normalized = normalize_l2(&pooled)?;

        let (n, _dim) = normalized.dims2()?;
        (0..n).map(|i| Ok(normalized.get(i)?.to_vec1::<f32>()?)).collect()
    }
}

fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

fn main() -> Result<()> {
    println!("Loading MiniLM (downloads ~90MB on first run, cached after)...");
    let mut embedder = Embedder::load()?;

    println!("Embedding {} corpus documents...", CORPUS.len());
    let corpus_texts: Vec<&str> = CORPUS.iter().map(|(_, text)| *text).collect();
    let corpus_embeddings = embedder.embed(&corpus_texts)?;
    let dim = corpus_embeddings[0].len();
    println!("embedding dimension: {dim}");

    // n_subspaces must divide dim (384 for MiniLM); n_pq_centroids /
    // n_index_centroids kept small since this corpus only has 15 documents.
    let mut db = LatentDb::build(&corpus_embeddings, 8, 8, 3, 8, 42);
    for ((topic, text), embedding) in CORPUS.iter().zip(corpus_embeddings.iter()) {
        db.insert(embedding, format!("[{topic}] {text}"))?;
    }
    println!("inserted {} records into LatentDb\n", db.len());

    let mut correct = 0;
    for (expected_topic, query) in QUERIES {
        let query_embedding = &embedder.embed(&[query])?[0];
        let hits = db.search(query_embedding, 3, db.n_index_centroids());

        println!("query ({expected_topic}): {query}");
        for hit in &hits {
            println!("  score={:.4}  {}", hit.score, hit.metadata);
        }
        let top_matches_topic = hits
            .first()
            .map(|h| h.metadata.starts_with(&format!("[{expected_topic}]")))
            .unwrap_or(false);
        println!("  top hit matches expected topic: {top_matches_topic}\n");
        if top_matches_topic {
            correct += 1;
        }
    }

    println!(
        "{correct}/{} queries retrieved a top hit from the correct topic \
         (this is what real trained embeddings look like -- contrast with \
         katgpt-rs's random-weight transformer, which could not do this)",
        QUERIES.len()
    );
    Ok(())
}
