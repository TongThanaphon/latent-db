# latent-db

A small, embeddable Rust "database" for vector embeddings that stores every
record **compressed in latent space** — inspired by ideas described in
[katgpt-rs](https://github.com/katopz/katgpt-rs)'s README (MUX-Latent
superposition, ShardEmbedding random projection, Hybrid OCT+PQ compression,
Schema Centroid indexing).

> This started as an independent, from-scratch reimplementation, guessing at
> the underlying algorithms from katgpt-rs's public README alone (no source
> access): Johnson–Lindenstrauss random projection, product quantization,
> IVF-style centroid indexing, and Holographic Reduced Representations for
> superposition. katgpt-rs's actual source has since been reviewed (MIT
> licensed) — where the real implementation diverges from the original guess,
> that's now called out inline (see the MUX-Latent note below in particular).

## What it does

| Module          | Idea borrowed from katgpt-rs        | What it actually is                                             |
|------------------|--------------------------------------|------------------------------------------------------------------|
| `projector.rs`   | `ShardEmbedding` (JL projection)      | Random Gaussian projection matrix, `[f32;N] -> [f32;M]`, deterministic per seed, BLAKE3-committed (`commit()`/`verify()`) so a persisted matrix can be tamper/corruption-checked |
| `pq.rs`          | Hybrid OCT+PQ KV codec                | Product Quantization: split vector into subspaces, k-means per subspace, store 1 byte/subspace |
| `index.rs`       | Schema Centroid                       | IVF-style centroid index: bucket records by nearest centroid, probe only the closest `nprobe` buckets at query time |
| `superpose.rs`   | MUX-Latent naming, algorithm diverges — see note below | Holographic Reduced Representations: `bind` (circular convolution) + `bundle` (sum) + `unbind` (circular correlation) to pack many (key, value) pairs into one fixed-size vector, genuinely lossy by construction |
| `merkle.rs`      | `MerkleOctree` / `MerkleProof`        | General binary BLAKE3 Merkle tree (katgpt-rs's is a fixed 64-leaf octree; this crate's record count varies at runtime) — `LatentDb::merkle_root()` + `merkle_proof(id)` give per-record inclusion proofs over the whole DB |
| `db.rs`          | The overall pipeline, plus `LatentContextBuffer`-style budget/eviction | Ties the above together: insert embeddings, dedupe by content hash, PQ-compress, centroid-index, approximate search, save/load to disk, optional `record_budget` + `EvictionPolicy` (see note below) |

## Usage

```rust
use latent-db::LatentDb;

// 1. Build the DB from a training batch (learns PQ codebooks + centroids).
let training_vectors: Vec<Vec<f32>> = load_my_embeddings();
let mut db = LatentDb::build(
    &training_vectors,
    8,   // n_subspaces for PQ
    32,  // PQ centroids per subspace
    16,  // centroid-index buckets
    8,   // sketch dimension for the index projector
    42,  // seed
);

// 2. Insert records (can be the same vectors, or new ones of the same dim).
let id = db.insert(&embedding, "some metadata string")?;

// 3. Approximate nearest-neighbour search.
let hits = db.search(&query_embedding, /* k */ 5, /* nprobe */ 4);
for hit in hits {
    println!("{} (score={:.3}) -> {}", hit.id, hit.score, hit.metadata);
}

// 4. Persist to disk.
db.save("my_db.bin")?;
let reloaded = LatentDb::load("my_db.bin")?;

// 5. Prove a record is part of the current database state.
let root = db.merkle_root();               // publish/store this as a checkpoint
let proof = db.merkle_proof(id).unwrap();   // hand this to whoever needs to verify
assert!(proof.verify(&root));

// 6. Bound memory usage: cap the DB at N records, evicting the rest.
use latent_db::EvictionPolicy;
db.set_eviction_policy(EvictionPolicy::LowestEnergy);
db.set_record_budget(10_000); // 0 (the default) = unlimited
```

For the more extreme "many records in one vector" mode:

```rust
use latent-db::SuperposedSlot;

let mut slot = SuperposedSlot::new(64);
slot.insert(&key_a, &value_a);
slot.insert(&key_b, &value_b);
// ... pack in more pairs; storage stays a single 64-length vector ...

let recovered_a = slot.expand(&key_a); // approximately == value_a
```

Retrieval quality degrades as more pairs share a slot. That's the inherent
accuracy/compression tradeoff of true HRR-style superposition — but it does
**not** mirror what katgpt-rs's real MUX-Latent does, despite this module's
name being borrowed from it.

> **Note on katgpt-rs's real MUX-Latent (`crates/katgpt-core/src/mux_latent/`
> in the actual source):** it is not HRR at all — no bind/bundle/unbind, no
> circular convolution. A span of tokens is assigned one latent "slot" whose
> weights are just geometric-decay-weighted one-hot positions (`decay^j`,
> normalized to sum to 1), used to build a compact representation for the
> decoder's attention. Critically, the **original tokens are also stored
> verbatim** alongside those weights, and `EXPAND(segment_id)` is a literal
> lookup that returns them — not a mathematical inversion. So katgpt-rs's
> X4/X8/X16 "compression" modes describe attention-budget footprint, not
> information loss: they're **lossless by construction**, because the source
> tokens never leave storage. `SuperposedSlot` here is a different, harder
> technique (true HRR bind/bundle/unbind) that katgpt-rs doesn't implement —
> a legitimate thing to have, just not something katgpt-rs validates the
> lossy tradeoff of.

## Run the demo

```bash
cargo test                       # 10 unit tests
cargo run --release --example demo
```

The demo builds a 600-vector synthetic corpus with 6 clustered "topics",
inserts everything, runs a nearest-neighbour query (and confirms it finds
the exact record queried with as the top hit), reports the PQ compression
ratio, round-trips through save/load, and shows the superposition slot's
accuracy degrading as more pairs are packed in.

## Design notes & honest limitations

- **This is a research/embedded-use prototype, not a production vector DB.**
  There's no WAL, no transactions, no concurrent access control, and
  everything lives in memory except for the flat `save`/`load` snapshot.
- **PQ codebooks and the centroid index are trained once, in a batch,** at
  `LatentDb::build` time. A real deployment would need periodic retraining
  as the embedding distribution drifts.
- **The centroid index is a simple IVF, not HNSW/ANNOY/ScaNN.** It works
  well at the scale this demo runs (hundreds–thousands of vectors) but
  large-dimensional, large-N production workloads would want a proper ANN
  library.
- **`SuperposedSlot` retrieval is inherently lossy and noisy**, by
  construction — that's the whole point, and unlike katgpt-rs's real
  MUX-Latent (which is lossless — see the note above), there's no fallback
  copy of the original value retained here. Don't use it where exact recall
  matters; use the main `LatentDb` PQ + index path for that instead.
- **`merkle_root()`/`merkle_proof()` rebuild the whole Merkle tree from
  scratch on every call** (O(n log n) over the current record count) rather
  than maintaining it incrementally on insert/remove. Fine at this crate's
  prototype scale, same tradeoff as the batch-trained PQ codebooks and
  centroid index; a production version would want an incremental/append-only
  tree (e.g. a Merkle Mountain Range) instead.
- **`record_budget` + `EvictionPolicy` are named after katgpt-rs's
  `LatentContextBuffer` (`mux_latent/buffer.rs`), but are a from-scratch
  design, not a port** — a closer look at the real source turned up two
  things worth being explicit about: (1) katgpt-rs's own "eviction" never
  loses data — it demotes a compressed span back to the raw tokens it
  already keeps stored alongside it, so nothing is discarded. `LatentDb`
  keeps no such fallback copy, so eviction here is real, irreversible
  record removal. (2) katgpt-rs's `EvictionPolicy::LowestEnergy` is an
  unimplemented stub that silently falls back to `OldestFirst`
  ("would need spectral analysis"), and the `SpectralLOD` "energy" it
  gestures at isn't FFT-based despite the module name — it's a token-ID
  variance heuristic. This crate's `LowestEnergy` is a genuine, working
  implementation of the idea, adapted from token spans to vector records:
  a record's "energy" is its squared distance (in the projected sketch
  space) to its assigned centroid, so records that look like everything
  else in their bucket get evicted before distinctive outliers do.
- The content hash used for de-duplication is a simple FNV-1a rather than
  BLAKE3. This was originally a toolchain workaround (an older Rust
  toolchain here couldn't build BLAKE3's `cpufeatures` dependency); on a
  current toolchain (`rustc 1.95`+) BLAKE3 builds fine, and is now a direct
  dependency — `projector.rs` uses it for a `commit()`/`verify()` matrix
  integrity check, mirroring katgpt-rs's `JlProjectionMatrix`. The
  content-dedup hash in `db.rs` still uses FNV-1a for now; swapping it to
  BLAKE3 for a stronger collision bound is a natural, low-risk follow-up
  since the dependency is already in the tree.
