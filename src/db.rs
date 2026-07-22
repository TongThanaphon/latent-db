//! `LatentDb`: a small embeddable database whose records live compressed in
//! latent space, indexed approximately by centroid, with an optional
//! superposition mode for packing many records into a single vector.
//!
//! Design lineage (each piece is called out with what it borrows from
//! katgpt-rs's README, reimplemented independently here):
//!
//! | katgpt-rs concept        | Role in LatentDb                              |
//! |---------------------------|-----------------------------------------------|
//! | ShardEmbedding (JL proj.) | `Projector` -- cheap low-dim sketch for indexing |
//! | Hybrid OCT+PQ KV codec    | `PqCodec` -- per-record compressed storage    |
//! | Schema Centroid           | `CentroidIndex` -- IVF-style bucket index      |
//! | MUX-Latent + EXPAND(i)    | `SuperposedSlot` -- many records, one vector   |
//! | BLAKE3-committed vectors  | content hash used for de-duplication (FNV-1a) |
//! | MerkleOctree / MerkleProof | `merkle` module -- per-record inclusion proofs |
//! | Viable Manifold Graph     | `manifold::ViableGraph` -- kNN graph over a predicate-filtered record subset, `geodesic()` / `random_walk()` traversal (`build_viable_graph()`) |
//! | Latent Field Steering     | `steering::SteeringVector` -- frozen direction + strength shifts the query before search (`search_steered()`) |

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::index::CentroidIndex;
use crate::manifold::{self, ViableGraph};
use crate::merkle::{self, Digest, MerkleProof, MerkleTree};
use crate::pq::PqCodec;
use crate::projector::Projector;
use crate::steering::SteeringVector;

#[derive(Serialize, Deserialize)]
struct StoredRecord {
    hash: u64,
    codes: Vec<u8>,
    metadata: String,
}

/// Eviction policy applied once `record_budget` is exceeded.
///
/// Named after katgpt-rs's `mux_latent::buffer::EvictionPolicy`, but the
/// semantics differ in one important way: katgpt-rs's `LatentContextBuffer`
/// "evicts" a compressed span by demoting it back to the raw tokens it
/// already keeps alongside the compressed form, so nothing is ever actually
/// lost. `LatentDb` never keeps an uncompressed copy of a record, so
/// eviction here is real, irreversible removal of the record.
///
/// `LowestEnergy` is also a from-scratch implementation, not a port:
/// katgpt-rs's own `EvictionPolicy::LowestEnergy` is an unimplemented stub
/// that silently falls back to `OldestFirst` ("would need spectral
/// analysis"), and the `SpectralLOD` "energy" it gestures at doesn't use
/// FFT despite the module name -- it's a token-ID variance heuristic. See
/// `LatentDb::energy` for the vector-record analog implemented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Evict the oldest-inserted records first (ascending id).
    OldestFirst,
    /// Evict the lowest-"energy" records first (see `LatentDb::energy`).
    LowestEnergy,
}

#[derive(Serialize, Deserialize)]
pub struct LatentDb {
    dim: usize,
    projector: Projector,
    pq: PqCodec,
    index: CentroidIndex,
    records: HashMap<u64, StoredRecord>,
    /// content hash -> id, for de-duplication on insert
    hash_to_id: HashMap<u64, u64>,
    next_id: u64,
    /// Maximum number of stored records; 0 means unlimited. Enforced after
    /// every `insert()` and whenever changed via `set_record_budget()`.
    record_budget: usize,
    eviction_policy: EvictionPolicy,
}

pub struct SearchHit {
    pub id: u64,
    pub score: f32,
    pub metadata: String,
}

#[derive(Debug)]
pub enum LatentDbError {
    Io(std::io::Error),
    Serialize(String),
    DimMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for LatentDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatentDbError::Io(e) => write!(f, "io error: {e}"),
            LatentDbError::Serialize(e) => write!(f, "serialize error: {e}"),
            LatentDbError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for LatentDbError {}

impl LatentDb {
    /// Build a fresh DB. `training_vectors` are used once, up front, to
    /// learn the PQ codebooks and the centroid index; in a real system
    /// you'd retrain periodically as the data distribution shifts, but a
    /// one-shot batch fit is enough to demonstrate the design.
    pub fn build(
        training_vectors: &[Vec<f32>],
        n_subspaces: usize,
        n_pq_centroids: usize,
        n_index_centroids: usize,
        sketch_dim: usize,
        seed: u64,
    ) -> Self {
        assert!(
            !training_vectors.is_empty(),
            "need training data to build a LatentDb"
        );
        let dim = training_vectors[0].len();

        let pq = PqCodec::train(training_vectors, n_subspaces, n_pq_centroids, 15, seed);
        let projector = Projector::new(dim, sketch_dim, seed.wrapping_add(1));
        let projected_training: Vec<Vec<f32>> = training_vectors
            .iter()
            .map(|v| projector.project(v))
            .collect();
        let index = CentroidIndex::train(
            &projected_training,
            n_index_centroids,
            15,
            seed.wrapping_add(2),
        );

        LatentDb {
            dim,
            projector,
            pq,
            index,
            records: HashMap::new(),
            hash_to_id: HashMap::new(),
            next_id: 0,
            record_budget: 0,
            eviction_policy: EvictionPolicy::OldestFirst,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Content hash used purely for de-duplication (same role as the
    /// BLAKE3 commit on each MUX-Latent vector in katgpt-rs, just a
    /// lighter-weight FNV-1a here since this crate avoids extra
    /// dependencies with a Rust-edition MSRV newer than this sandbox).
    fn hash_vector(v: &[f32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for x in v {
            for byte in x.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }

    /// Insert a full-precision embedding + metadata string. Returns the
    /// assigned record id. If the exact same vector was already inserted
    /// (same BLAKE3 content hash), the existing id is returned instead of
    /// creating a duplicate row.
    pub fn insert(
        &mut self,
        embedding: &[f32],
        metadata: impl Into<String>,
    ) -> Result<u64, LatentDbError> {
        if embedding.len() != self.dim {
            return Err(LatentDbError::DimMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        let hash = Self::hash_vector(embedding);
        if let Some(&existing) = self.hash_to_id.get(&hash) {
            return Ok(existing);
        }

        let id = self.next_id;
        self.next_id += 1;

        let codes = self.pq.encode(embedding);
        let projected = self.projector.project(embedding);
        self.index.insert(id, &projected);

        self.records.insert(
            id,
            StoredRecord {
                hash,
                codes,
                metadata: metadata.into(),
            },
        );
        self.hash_to_id.insert(hash, id);
        self.enforce_budget();
        Ok(id)
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(rec) = self.records.remove(&id) {
            self.hash_to_id.remove(&rec.hash);
            self.index.remove(id);
        }
    }

    /// Reconstruct the (approximate, PQ-decoded) embedding for a record.
    pub fn get_approx_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.records.get(&id).map(|r| self.pq.decode(&r.codes))
    }

    pub fn get_metadata(&self, id: u64) -> Option<&str> {
        self.records.get(&id).map(|r| r.metadata.as_str())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Approximate nearest-neighbour search. `nprobe` controls how many
    /// centroid buckets get scanned (higher = more accurate, slower).
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchHit> {
        if query.len() != self.dim {
            return Vec::new();
        }
        let projected_query = self.projector.project(query);
        let candidates = self.index.candidates(&projected_query, nprobe);

        let mut scored: Vec<SearchHit> = candidates
            .into_iter()
            .filter_map(|id| {
                let rec = self.records.get(&id)?;
                let approx = self.pq.decode(&rec.codes);
                let score = crate::superpose::cosine_sim(query, &approx);
                Some(SearchHit {
                    id,
                    score,
                    metadata: rec.metadata.clone(),
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        scored
    }

    /// Like [`Self::search`], but first shifts a copy of `query` by
    /// `steering` (`state[i] += alpha * direction[i]`) before projecting and
    /// searching -- concept conditioning / query expansion toward (or away
    /// from, via the direction's sign) a frozen semantic axis, without
    /// retraining anything. See the `steering` module for how to build and
    /// persist a [`SteeringVector`].
    ///
    /// Returns an empty `Vec` if `query`'s or `steering`'s dimension doesn't
    /// match `self.dim()`, matching [`Self::search`]'s own dim-mismatch
    /// convention.
    pub fn search_steered(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        steering: &SteeringVector,
    ) -> Vec<SearchHit> {
        if query.len() != self.dim || steering.dim() != self.dim {
            return Vec::new();
        }
        let mut steered = query.to_vec();
        steering.apply(&mut steered);
        self.search(&steered, k, nprobe)
    }

    /// Build a [`ViableGraph`] over the subset of currently-stored records
    /// for which `predicate` (applied to each record's approximate decoded
    /// vector) is `true`, kNN-connected with `k_nearest` neighbors per node.
    /// See the `manifold` module for `edge_midpoint_check` and how to
    /// traverse the result (`geodesic`, `random_walk`).
    ///
    /// Ids are visited in ascending order before decoding so the resulting
    /// graph's node assignment -- and therefore `random_walk`'s output for a
    /// given seed -- is stable across rebuilds of the same record set, the
    /// same determinism guarantee `merkle_leaves()` makes for
    /// `merkle_root()`.
    pub fn build_viable_graph<F>(
        &self,
        predicate: F,
        k_nearest: usize,
        edge_midpoint_check: bool,
    ) -> ViableGraph
    where
        F: Fn(&[f32]) -> bool,
    {
        let mut ids: Vec<u64> = self.records.keys().copied().collect();
        ids.sort_unstable();
        let records = ids
            .into_iter()
            .map(|id| (id, self.pq.decode(&self.records[&id].codes)));
        manifold::build_viable_graph(records, predicate, k_nearest, edge_midpoint_check)
    }

    /// Compression ratio achieved by PQ storage vs. keeping raw f32 vectors.
    pub fn compression_ratio(&self) -> f32 {
        self.pq.raw_bytes() as f32 / self.pq.compressed_bytes() as f32
    }

    pub fn n_index_centroids(&self) -> usize {
        self.index.n_centroids()
    }

    pub fn record_budget(&self) -> usize {
        self.record_budget
    }

    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction_policy
    }

    pub fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    /// Cap the number of stored records at `budget` (0 = unlimited),
    /// evicting immediately per `eviction_policy` if the DB is already over
    /// budget. Also enforced automatically after every future `insert()`.
    pub fn set_record_budget(&mut self, budget: usize) {
        self.record_budget = budget;
        self.enforce_budget();
    }

    /// "Energy" of a stored record, used by `EvictionPolicy::LowestEnergy`:
    /// squared distance, in the projected sketch space, from the record to
    /// its assigned centroid. A record sitting right on its centroid looks
    /// like everything else already in that bucket (low energy, redundant,
    /// safe to drop first); a record far from its centroid is the odd one
    /// out in its bucket (high energy, worth keeping) -- the same
    /// "uniform/redundant vs. information-dense" framing katgpt-rs's
    /// `SpectralLOD` docs describe, just measured over vectors and their
    /// centroids instead of token-ID variance within a span.
    fn energy(&self, id: u64, rec: &StoredRecord) -> f32 {
        let centroid = self
            .index
            .assigned_centroid(id)
            .and_then(|c| self.index.centroid_vector(c));
        let Some(centroid) = centroid else {
            return f32::MAX; // not indexed (shouldn't happen) -- never evict first
        };
        let approx = self.pq.decode(&rec.codes);
        let projected = self.projector.project(&approx);
        projected
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Evict records, per `eviction_policy`, until `len() <= record_budget`.
    fn enforce_budget(&mut self) {
        if self.record_budget == 0 {
            return;
        }
        let excess = self.records.len().saturating_sub(self.record_budget);
        if excess == 0 {
            return;
        }

        let mut ids: Vec<u64> = self.records.keys().copied().collect();
        match self.eviction_policy {
            EvictionPolicy::OldestFirst => ids.sort_unstable(),
            EvictionPolicy::LowestEnergy => {
                ids.sort_by(|&a, &b| {
                    let ea = self.energy(a, &self.records[&a]);
                    let eb = self.energy(b, &self.records[&b]);
                    ea.total_cmp(&eb)
                });
            }
        }

        for id in ids.into_iter().take(excess) {
            self.remove(id);
        }
    }

    fn record_leaf_bytes(id: u64, rec: &StoredRecord) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + rec.codes.len() + rec.metadata.len());
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&rec.hash.to_le_bytes());
        bytes.extend_from_slice(&rec.codes);
        bytes.extend_from_slice(rec.metadata.as_bytes());
        bytes
    }

    /// Ids in ascending order paired with their Merkle leaf hash -- the
    /// canonical, deterministic leaf ordering. A `HashMap`'s own iteration
    /// order isn't stable across runs, so `merkle_root()` would otherwise
    /// change on every reload even with identical records.
    fn merkle_leaves(&self) -> Vec<(u64, Digest)> {
        let mut ids: Vec<u64> = self.records.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| {
                let rec = &self.records[&id];
                (id, merkle::hash_leaf(&Self::record_leaf_bytes(id, rec)))
            })
            .collect()
    }

    /// Build a fresh Merkle tree over every currently-stored record. This
    /// is rebuilt from scratch on every call rather than maintained
    /// incrementally on insert/remove -- fine at this crate's prototype
    /// scale (same tradeoff as the batch-trained PQ codebooks and centroid
    /// index, see the design notes), and it guarantees `merkle_root()` /
    /// `merkle_proof()` always reflect the live record set exactly, with no
    /// risk of a stale cached tree drifting out of sync.
    pub fn merkle_tree(&self) -> MerkleTree {
        MerkleTree::build(self.merkle_leaves().into_iter().map(|(_, h)| h).collect())
    }

    /// Root commitment over every currently-stored record. Publish or store
    /// this value out-of-band as a checkpoint; `merkle_proof(id)` plus
    /// `MerkleProof::verify()` then let a third party confirm a specific
    /// record was included in that checkpoint without needing the whole
    /// database.
    pub fn merkle_root(&self) -> Digest {
        self.merkle_tree().root()
    }

    /// Inclusion proof that record `id` is part of the current
    /// `merkle_root()`. Returns `None` if `id` isn't currently stored.
    pub fn merkle_proof(&self, id: u64) -> Option<MerkleProof> {
        let leaves = self.merkle_leaves();
        let position = leaves.iter().position(|(lid, _)| *lid == id)?;
        let tree = MerkleTree::build(leaves.into_iter().map(|(_, h)| h).collect());
        tree.proof(position)
    }

    /// Persist the whole DB (codebooks, index, compressed records) to disk.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), LatentDbError> {
        let bytes =
            bincode::serialize(self).map_err(|e| LatentDbError::Serialize(e.to_string()))?;
        let mut f = File::create(path).map_err(LatentDbError::Io)?;
        f.write_all(&bytes).map_err(LatentDbError::Io)?;
        Ok(())
    }

    /// Load a DB previously written with `save`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LatentDbError> {
        let mut f = File::open(path).map_err(LatentDbError::Io)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(LatentDbError::Io)?;
        bincode::deserialize(&bytes).map_err(|e| LatentDbError::Serialize(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn insert_and_exact_nearest_neighbor_roundtrip() {
        let corpus = synthetic_corpus(300, 32, 1);
        let mut db = LatentDb::build(&corpus, 4, 16, 8, 8, 42);

        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            let id = db.insert(v, format!("record-{i}")).unwrap();
            ids.push(id);
        }

        // Querying with a vector identical to one we inserted should
        // return that same record as the top hit (probing enough buckets).
        let probe_target = &corpus[10];
        let hits = db.search(probe_target, 5, db.n_index_centroids());
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, ids[10]);
    }

    #[test]
    fn duplicate_insert_deduplicates_by_content_hash() {
        let corpus = synthetic_corpus(50, 16, 2);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 7);
        let id1 = db.insert(&corpus[0], "first").unwrap();
        let id2 = db.insert(&corpus[0], "duplicate").unwrap();
        assert_eq!(id1, id2);
        assert_eq!(db.len(), 1);
        // metadata is not overwritten by the duplicate insert
        assert_eq!(db.get_metadata(id1), Some("first"));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let corpus = synthetic_corpus(80, 16, 3);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 11);
        for (i, v) in corpus.iter().enumerate() {
            db.insert(v, format!("r{i}")).unwrap();
        }
        let tmp = std::env::temp_dir().join("latent-db_test.bin");
        db.save(&tmp).unwrap();
        let loaded = LatentDb::load(&tmp).unwrap();
        assert_eq!(loaded.len(), db.len());
        assert_eq!(loaded.get_metadata(0), db.get_metadata(0));
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn compression_ratio_is_greater_than_one() {
        let corpus = synthetic_corpus(100, 64, 4);
        let db = LatentDb::build(&corpus, 8, 16, 8, 8, 5);
        assert!(db.compression_ratio() > 1.0);
    }

    #[test]
    fn merkle_root_changes_as_records_are_inserted() {
        let corpus = synthetic_corpus(20, 16, 5);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 9);
        let empty_root = db.merkle_root();
        db.insert(&corpus[0], "a").unwrap();
        let after_one = db.merkle_root();
        assert_ne!(empty_root, after_one);
        db.insert(&corpus[1], "b").unwrap();
        let after_two = db.merkle_root();
        assert_ne!(after_one, after_two);
    }

    #[test]
    fn merkle_proof_verifies_every_stored_record() {
        let corpus = synthetic_corpus(30, 16, 6);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 10);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }
        let root = db.merkle_root();
        for id in ids {
            let proof = db.merkle_proof(id).unwrap();
            assert!(proof.verify(&root));
        }
    }

    #[test]
    fn merkle_proof_rejects_a_tampered_leaf() {
        let corpus = synthetic_corpus(10, 16, 7);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 12);
        let id = db.insert(&corpus[0], "x").unwrap();
        let root = db.merkle_root();
        let mut proof = db.merkle_proof(id).unwrap();
        proof.leaf_hash[0] ^= 0xFF;
        assert!(!proof.verify(&root));
    }

    #[test]
    fn merkle_proof_is_none_for_unknown_or_removed_id() {
        let corpus = synthetic_corpus(10, 16, 8);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 13);
        assert!(db.merkle_proof(999).is_none());
        let id = db.insert(&corpus[0], "x").unwrap();
        db.remove(id);
        assert!(db.merkle_proof(id).is_none());
    }

    #[test]
    fn zero_budget_is_unlimited_by_default() {
        let corpus = synthetic_corpus(50, 16, 20);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 21);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        assert_eq!(db.len(), 50);
    }

    #[test]
    fn oldest_first_eviction_keeps_only_the_newest_records() {
        let corpus = synthetic_corpus(20, 16, 22);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 23);
        db.set_record_budget(5);
        assert_eq!(db.eviction_policy(), EvictionPolicy::OldestFirst);

        let mut ids = Vec::new();
        for v in &corpus {
            ids.push(db.insert(v, "x").unwrap());
        }
        assert_eq!(db.len(), 5);

        // The 5 highest (newest) ids should be the ones still present.
        let newest: Vec<u64> = ids[ids.len() - 5..].to_vec();
        for id in newest {
            assert!(db.get_metadata(id).is_some(), "newest record {id} should survive");
        }
        for id in &ids[..ids.len() - 5] {
            assert!(db.get_metadata(*id).is_none(), "oldest record {id} should be evicted");
        }
    }

    #[test]
    fn set_record_budget_evicts_immediately_when_already_over() {
        let corpus = synthetic_corpus(30, 16, 24);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 25);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        assert_eq!(db.len(), 30);
        db.set_record_budget(10);
        assert_eq!(db.len(), 10);
        assert_eq!(db.record_budget(), 10);
    }

    #[test]
    fn lowest_energy_eviction_prefers_dropping_records_near_their_centroid() {
        // Two tight clusters far apart: records near their cluster's own
        // centroid have low energy (redundant with their many near-twins);
        // a single far-flung outlier vector has high energy and should
        // survive even under a tight budget.
        let mut rng_seed = 30u64;
        let mut corpus = Vec::new();
        for center in [-5.0f32, 5.0] {
            for _ in 0..10 {
                rng_seed += 1;
                let mut v = vec![center; 16];
                v[0] += (rng_seed as f32 % 7.0) * 0.001; // tiny jitter, stays near centroid
                corpus.push(v);
            }
        }
        let outlier: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 50.0 } else { -50.0 }).collect();
        corpus.push(outlier.clone());

        let mut db = LatentDb::build(&corpus, 2, 8, 2, 8, 26);
        db.set_eviction_policy(EvictionPolicy::LowestEnergy);

        let mut ids = Vec::new();
        for v in &corpus {
            ids.push(db.insert(v, "x").unwrap());
        }
        let outlier_id = *ids.last().unwrap();

        db.set_record_budget(5);
        assert_eq!(db.len(), 5);
        assert!(
            db.get_metadata(outlier_id).is_some(),
            "the distinctive outlier should survive low-energy eviction"
        );
    }

    #[test]
    fn search_steered_with_zero_alpha_matches_plain_search() {
        let corpus = synthetic_corpus(100, 16, 40);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 41);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        let dir = SteeringVector::new(vec![1.0; 16], 0.0, 10.0).unwrap();
        // norm of an all-ones length-16 vector is 4, well outside default
        // tolerance -- use a generous norm_tol since this test only cares
        // about alpha=0 being a true no-op, not the direction's shape.
        let plain = db.search(&corpus[3], 5, db.n_index_centroids());
        let steered = db.search_steered(&corpus[3], 5, db.n_index_centroids(), &dir);
        assert_eq!(plain.len(), steered.len());
        for (a, b) in plain.iter().zip(steered.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.score, b.score);
        }
    }

    #[test]
    fn search_steered_rejects_dimension_mismatch() {
        let corpus = synthetic_corpus(50, 16, 42);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 43);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        let wrong_dim_steering = SteeringVector::new(vec![1.0, 0.0], 0.5, 1e-4).unwrap();
        assert!(db
            .search_steered(&corpus[0], 5, db.n_index_centroids(), &wrong_dim_steering)
            .is_empty());
    }

    #[test]
    fn search_steered_toward_a_records_own_direction_raises_its_score() {
        let corpus = synthetic_corpus(200, 32, 50);
        let mut db = LatentDb::build(&corpus, 4, 16, 8, 8, 51);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }

        let target = db.get_approx_vector(ids[0]).unwrap();
        let norm: f32 = target.iter().map(|x| x * x).sum::<f32>().sqrt();
        let direction: Vec<f32> = target.iter().map(|x| x / norm.max(1e-12)).collect();
        let baseline = SteeringVector::new(direction.clone(), 0.0, 1e-3).unwrap();
        let strong = SteeringVector::new(direction, 1.0, 1e-3).unwrap();

        let query = &corpus[7];
        let plain = db.search_steered(query, db.len(), db.n_index_centroids(), &baseline);
        let steered = db.search_steered(query, db.len(), db.n_index_centroids(), &strong);

        let plain_score = plain.iter().find(|h| h.id == ids[0]).unwrap().score;
        let steered_score = steered.iter().find(|h| h.id == ids[0]).unwrap().score;
        assert!(
            steered_score > plain_score,
            "steering the query toward record 0's own direction should raise its score \
             ({steered_score} vs {plain_score})"
        );
    }

    #[test]
    fn build_viable_graph_restricts_to_the_predicate_and_walks_stay_inside_it() {
        let corpus = synthetic_corpus(120, 8, 60);
        let mut db = LatentDb::build(&corpus, 4, 16, 8, 8, 61);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }

        let graph = db.build_viable_graph(|v| v[0] > 0.0, 4, false);
        let expected = ids
            .iter()
            .filter(|&&id| db.get_approx_vector(id).unwrap()[0] > 0.0)
            .count();
        assert_eq!(graph.n_nodes(), expected);

        for &id in &ids {
            let positive = db.get_approx_vector(id).unwrap()[0] > 0.0;
            assert_eq!(graph.contains(id), positive);
        }

        let start = *ids.iter().find(|&&id| graph.contains(id)).unwrap();
        let walk = graph.random_walk(start, 10, 7);
        assert_eq!(walk.len(), 11);
        for id in walk {
            assert!(db.get_approx_vector(id).unwrap()[0] > 0.0);
        }
    }
}
