//! Runnable demo of `latent-db`.
//!
//! Run with:
//!     cargo run --release --example demo
//!
//! Shows two things end to end:
//!   1. LatentDb: build from a training corpus, insert embeddings +
//!      metadata, run approximate nearest-neighbour search, check the
//!      compression ratio, and round-trip through save/load.
//!   2. SuperposedSlot: the more extreme MUX-Latent-style mode where many
//!      (key, value) pairs share a single vector, with retrieval quality
//!      shown degrading as more pairs are packed in.

use latent_db::{cosine_sim, EvictionPolicy, LatentDb, SuperposedSlot};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 32;

fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// A little "topic" generator: vectors clustered around a handful of
/// random centers, so nearest-neighbour search has real structure to find
/// (rather than pure noise where every result is equally (ir)relevant).
fn make_corpus(
    rng: &mut StdRng,
    n: usize,
    dim: usize,
    n_topics: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let topic_centers: Vec<Vec<f32>> = (0..n_topics).map(|_| random_vec(rng, dim)).collect();
    let mut vectors = Vec::with_capacity(n);
    let mut topics = Vec::with_capacity(n);
    for i in 0..n {
        let topic = i % n_topics;
        let center = &topic_centers[topic];
        let noisy: Vec<f32> = center
            .iter()
            .map(|c| c + rng.gen_range(-0.15..0.15))
            .collect();
        vectors.push(noisy);
        topics.push(topic);
    }
    (vectors, topics)
}

fn section(title: &str) {
    println!("\n=== {title} ===");
}

fn main() {
    let mut rng = StdRng::seed_from_u64(1234);

    section("1. Build LatentDb from a training corpus");
    let n_records = 600;
    let n_topics = 6;
    let (corpus, topics) = make_corpus(&mut rng, n_records, DIM, n_topics);
    println!("training corpus: {n_records} vectors, dim={DIM}, {n_topics} synthetic topics");

    let mut db = LatentDb::build(
        &corpus, // training vectors (also what we'll insert)
        8,       // n_subspaces for PQ  (32 / 8 = 4 floats per subspace)
        32,      // n_pq_centroids per subspace (5 bits/subspace)
        16,      // n_index_centroids (IVF-style buckets)
        8,       // sketch_dim for the random projector used by the index
        42,      // seed
    );

    section("2. Insert records");
    let mut ids = Vec::with_capacity(n_records);
    for (i, v) in corpus.iter().enumerate() {
        let id = db
            .insert(v, format!("doc-{i} (topic {})", topics[i]))
            .expect("insert should succeed for matching dimension");
        ids.push(id);
    }
    println!(
        "inserted {} records (deduped down from {} attempts)",
        db.len(),
        n_records
    );
    println!(
        "PQ compression ratio vs raw f32: {:.1}x  ({} bytes/vector -> {} bytes/vector)",
        db.compression_ratio(),
        DIM * 4,
        DIM * 4 / db.compression_ratio() as usize
    );

    section("3. Approximate nearest-neighbour search");
    let query_idx = 123;
    let query = &corpus[query_idx];
    let hits = db.search(query, 5, /* nprobe */ 4);
    println!("query = corpus[{query_idx}] (topic {})", topics[query_idx]);
    for hit in &hits {
        println!(
            "  id={:<4} score={:.4}  metadata={}",
            hit.id, hit.score, hit.metadata
        );
    }
    let top_is_self = hits
        .first()
        .map(|h| h.id == ids[query_idx])
        .unwrap_or(false);
    println!("top hit is the exact record we queried with: {top_is_self}");

    section("4. Approximate reconstruction from PQ codes");
    let approx = db.get_approx_vector(ids[query_idx]).unwrap();
    let sim = cosine_sim(query, &approx);
    println!("cosine similarity between original and PQ-decoded vector: {sim:.4}");

    section("5. Save / load round-trip");
    let path = std::env::temp_dir().join("latent-db_demo.bin");
    db.save(&path).expect("save should succeed");
    let reloaded = LatentDb::load(&path).expect("load should succeed");
    println!(
        "saved to {:?}, reloaded {} records (matches original: {})",
        path,
        reloaded.len(),
        reloaded.len() == db.len()
    );
    std::fs::remove_file(&path).ok();

    section("6. SuperposedSlot: many records, one vector (MUX-Latent style)");
    let slot_dim = 64;
    let target_key = random_vec(&mut rng, slot_dim);
    let target_value = random_vec(&mut rng, slot_dim);

    for &n_extra in &[0usize, 4, 16, 64, 256] {
        let mut slot = SuperposedSlot::new(slot_dim);
        slot.insert(&target_key, &target_value);
        for _ in 0..n_extra {
            slot.insert(
                &random_vec(&mut rng, slot_dim),
                &random_vec(&mut rng, slot_dim),
            );
        }
        let recovered = slot.expand(&target_key);
        let sim = cosine_sim(&recovered, &target_value);
        println!(
            "  pairs in slot={:<4} (compression {:.0}x)  retrieval cosine-sim={:.4}",
            slot.count, slot.count as f32, sim
        );
    }
    println!(
        "\nnote: retrieval quality degrades gracefully as more pairs share the slot --\n\
         this is the lossy HRR bind/bundle tradeoff, which does NOT mirror katgpt-rs's\n\
         real MUX-Latent (that scheme retains original tokens and is lossless; see README)."
    );

    section("7. Merkle inclusion proofs over stored records");
    let root = db.merkle_root();
    println!("merkle_root() over {} records: {}", db.len(), hex(&root));
    let proof_id = ids[query_idx];
    let proof = db.merkle_proof(proof_id).expect("record should be present");
    println!(
        "merkle_proof({proof_id}) verifies against the root: {}",
        proof.verify(&root)
    );
    let mut tampered = proof.clone();
    tampered.leaf_hash[0] ^= 0xFF;
    println!(
        "same proof with one flipped byte verifies: {}",
        tampered.verify(&root)
    );

    section("8. Bounded-memory operation: record budget + eviction");
    println!("before capping: {} records stored", db.len());
    db.set_eviction_policy(EvictionPolicy::LowestEnergy);
    db.set_record_budget(200);
    println!(
        "set_record_budget(200) with LowestEnergy policy -> {} records remain",
        db.len()
    );
    println!(
        "note: LowestEnergy evicts records closest to their assigned centroid first\n\
         (most redundant with their cluster), keeping the more distinctive ones."
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
