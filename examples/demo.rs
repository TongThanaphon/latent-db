//! Runnable demo of `latent-db`.
//!
//! Run with:
//!     cargo run --release --example demo
//!
//! Shows several things end to end:
//!   1. LatentDb: build from a training corpus, insert embeddings +
//!      metadata, run approximate nearest-neighbour search, check the
//!      compression ratio, and round-trip through save/load.
//!   2. SuperposedSlot: the more extreme MUX-Latent-style mode where many
//!      (key, value) pairs share a single vector, with retrieval quality
//!      shown degrading as more pairs are packed in.
//!   3. Latent Field Steering: biasing a query toward a topic axis before
//!      searching, without retraining anything.
//!   4. Viable Manifold Graph: navigating a predicate-filtered subset of
//!      records (geodesic + random walk) that never leaves the subset.

use latent_db::{cosine_sim, EvictionPolicy, LatentDb, SearchHit, SteeringVector, SuperposedSlot};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

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

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Mean vector of every `corpus[i]` whose `topics[i] == topic`.
fn topic_centroid(corpus: &[Vec<f32>], topics: &[usize], topic: usize, dim: usize) -> Vec<f32> {
    let mut sum = vec![0.0f32; dim];
    let mut count = 0usize;
    for (v, &t) in corpus.iter().zip(topics) {
        if t == topic {
            for i in 0..dim {
                sum[i] += v[i];
            }
            count += 1;
        }
    }
    for x in &mut sum {
        *x /= count.max(1) as f32;
    }
    sum
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

    // id -> topic lookup, reused by sections 8 and 9. Computed (and those
    // sections run) *before* the eviction demo in section 10 -- eviction
    // removes records, and we want the steering/graph demos to see the
    // full, un-thinned corpus.
    let id_to_topic: HashMap<u64, usize> = ids.iter().copied().zip(topics.iter().copied()).collect();
    let topic_centers: Vec<Vec<f32>> =
        (0..n_topics).map(|t| topic_centroid(&corpus, &topics, t, DIM)).collect();

    section("8. Latent Field Steering: bias a query toward a topic axis");
    let (topic_a, topic_b) = (0usize, 1usize);
    let neutral_query: Vec<f32> = topic_centers[topic_a]
        .iter()
        .zip(topic_centers[topic_b].iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();
    let diff: Vec<f32> = topic_centers[topic_b]
        .iter()
        .zip(topic_centers[topic_a].iter())
        .map(|(b, a)| b - a)
        .collect();
    let diff_norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
    let direction: Vec<f32> = diff.iter().map(|x| x / diff_norm).collect();
    let steering = SteeringVector::new(direction, 0.8, 1e-3).expect("unit-norm direction");

    let count_topic = |hits: &[SearchHit], topic: usize| -> usize {
        hits.iter()
            .filter(|h| id_to_topic.get(&h.id) == Some(&topic))
            .count()
    };
    let top_k = 10;
    let nprobe = db.n_index_centroids();
    let plain_hits = db.search(&neutral_query, top_k, nprobe);
    let steered_hits = db.search_steered(&neutral_query, top_k, nprobe, &steering);

    println!(
        "neutral query = midpoint of topic {topic_a} & topic {topic_b} centroids, top-{top_k} by topic:"
    );
    println!(
        "  plain search:   topic {topic_a}={:<3} topic {topic_b}={:<3} other={}",
        count_topic(&plain_hits, topic_a),
        count_topic(&plain_hits, topic_b),
        top_k - count_topic(&plain_hits, topic_a) - count_topic(&plain_hits, topic_b)
    );
    println!(
        "  steered search (alpha=0.8 toward topic {topic_b}): topic {topic_a}={:<3} topic {topic_b}={:<3} other={}",
        count_topic(&steered_hits, topic_a),
        count_topic(&steered_hits, topic_b),
        top_k - count_topic(&steered_hits, topic_a) - count_topic(&steered_hits, topic_b)
    );

    section("9. Viable Manifold Graph: navigate a predicate-filtered subset");
    let predicate = |v: &[f32]| {
        let d0 = euclidean(v, &topic_centers[0]);
        (0..n_topics).all(|t| t == 0 || euclidean(v, &topic_centers[t]) >= d0)
    };
    let graph = db.build_viable_graph(predicate, /* k_nearest */ 4, /* edge_midpoint_check */ false);
    println!(
        "predicate: 'nearest topic centroid is topic 0' -> kept {} of {} records as graph nodes, {} edges",
        graph.n_nodes(),
        db.len(),
        graph.n_edges()
    );

    let topic0_ids: Vec<u64> = ids
        .iter()
        .copied()
        .filter(|id| id_to_topic.get(id) == Some(&0))
        .collect();
    if topic0_ids.len() >= 2 {
        let (a, b) = (topic0_ids[0], topic0_ids[1]);
        match graph.geodesic(a, b) {
            Some(path) => println!(
                "geodesic({a}, {b}): {} hops, staying within the topic-0 neighborhood",
                path.len() - 1
            ),
            None => println!("geodesic({a}, {b}): unreachable (topic-0 subset split into separate components)"),
        }

        let walk = graph.random_walk(a, 10, 42);
        let all_topic0 = walk.iter().all(|id| id_to_topic.get(id) == Some(&0));
        println!(
            "random_walk({a}, steps=10, seed=42): visited {} ids, all still topic 0: {all_topic0}",
            walk.len()
        );
    }

    section("10. Bounded-memory operation: record budget + eviction");
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
