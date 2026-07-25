//! Offline indexing step: embed the demo corpus with MiniLM, build a
//! `LatentDb`, and write the serialized bytes to `web/corpus.bin` for the
//! wasm-side `web/index.html` demo to `fetch()` and load directly (no
//! server needed for this part -- see `examples/embed_server.rs` for the
//! one piece that does need a server: embedding the live user query).
//!
//! Run with:
//!     cargo run --release --example build_index

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use latent_db::LatentDb;
use tokenizers::{PaddingParams, Tokenizer};

/// Same 48-document, 6-topic corpus as `examples/mini_rag.rs`, for
/// continuity with what's already been demonstrated working. Sized to give
/// the browser demo's `manifold`/`steering` panels (`web/index.html`)
/// enough same-topic neighbors per document for a non-trivial kNN graph.
const CORPUS: &[(&str, &str)] = &[
    ("cooking", "Preheat the oven to 200C before baking the bread."),
    ("cooking", "Add a pinch of salt to the pasta water while it boils."),
    ("cooking", "Simmer the tomato sauce for twenty minutes until it thickens."),
    ("cooking", "Whisk the eggs and sugar together until the mixture is pale."),
    ("cooking", "Marinate the chicken in soy sauce and garlic overnight."),
    ("cooking", "Knead the dough for ten minutes until it becomes smooth and elastic."),
    ("cooking", "Season the steak generously with salt and pepper before searing it."),
    ("cooking", "Fold the whipped cream gently into the chocolate mousse."),
    ("astronomy", "Jupiter is the largest planet in the solar system."),
    ("astronomy", "A neutron star can form when a massive star collapses."),
    ("astronomy", "The Hubble telescope has captured images of distant galaxies."),
    ("astronomy", "Saturn's rings are made mostly of ice and rock particles."),
    ("astronomy", "Astronomers detected a new exoplanet orbiting a red dwarf."),
    ("astronomy", "A total solar eclipse occurs when the moon fully blocks the sun."),
    ("astronomy", "Black holes warp spacetime so strongly that light cannot escape."),
    ("astronomy", "The Milky Way contains hundreds of billions of stars."),
    ("finance", "The central bank raised interest rates to curb inflation."),
    ("finance", "Diversifying a portfolio reduces exposure to a single asset."),
    ("finance", "The company reported quarterly earnings above analyst expectations."),
    ("finance", "Bond yields fell after the latest jobs report was released."),
    ("finance", "Investors moved capital into index funds during the downturn."),
    ("finance", "A rising unemployment rate often signals a slowing economy."),
    ("finance", "The stock market rallied after the merger was announced."),
    ("finance", "Compound interest lets savings grow faster over long time horizons."),
    ("travel", "Book a window seat for the best view during the flight."),
    ("travel", "The train departs from platform nine at half past six."),
    ("travel", "Pack light when hiking through the mountains for several days."),
    ("travel", "The old town's cobblestone streets are best explored on foot."),
    ("travel", "Renew your passport well before an international trip."),
    ("travel", "The ferry crosses the strait twice a day during summer."),
    ("travel", "Local guides recommend visiting the market early in the morning."),
    ("travel", "A long layover gives travelers time to explore the airport city."),
    ("health", "Drinking enough water throughout the day supports healthy digestion."),
    ("health", "Regular exercise lowers the risk of heart disease over time."),
    ("health", "A balanced diet includes plenty of vegetables and whole grains."),
    ("health", "Getting seven to eight hours of sleep improves concentration."),
    ("health", "Stretching before a workout helps prevent muscle strain."),
    ("health", "Doctors recommend an annual checkup to catch problems early."),
    ("health", "Chronic stress can weaken the immune system over months."),
    ("health", "Washing hands frequently reduces the spread of common colds."),
    ("technology", "The new processor doubles battery life compared to last year's model."),
    ("technology", "Cloud storage lets users access files from any device."),
    ("technology", "Encryption protects sensitive data from unauthorized access."),
    ("technology", "The software update fixed several long-standing security bugs."),
    ("technology", "Machine learning models improve as they train on more data."),
    ("technology", "Fiber-optic cables carry data faster than traditional copper wires."),
    ("technology", "A firmware bug caused the smart thermostat to overheat."),
    ("technology", "Open-source projects rely on contributions from volunteer developers."),
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
    println!("Loading MiniLM...");
    let mut embedder = Embedder::load()?;

    println!("Embedding {} corpus documents...", CORPUS.len());
    let corpus_texts: Vec<&str> = CORPUS.iter().map(|(_, text)| *text).collect();
    let corpus_embeddings = embedder.embed(&corpus_texts)?;

    let mut db = LatentDb::build(&corpus_embeddings, 8, 16, 6, 8, 42);
    for ((topic, text), embedding) in CORPUS.iter().zip(corpus_embeddings.iter()) {
        db.insert(embedding, format!("[{topic}] {text}"))?;
    }
    println!("inserted {} records into LatentDb", db.len());

    let bytes = bincode::serialize(&db)?;
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web/corpus.bin");
    std::fs::write(&out_path, &bytes)?;
    println!("wrote {} bytes to {}", bytes.len(), out_path.display());

    Ok(())
}
