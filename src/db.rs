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
//! | Manifold Bandit / LatentTaskTree | `bandit::RegionTree` -- PCA + recursive-k-means region tree over stored records, Thompson-sampled (`sample()`) and reward-updated (`observe()`) (`build_region_tree()`) |
//! | neuron-db's `recall_blended` | Reciprocal Rank Fusion of a lexical/metadata term-overlap ranking with the vector `search()` ranking (`search_blended()`) -- this crate's first borrowing from neuron-db rather than katgpt-rs |

use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::bandit::{RegionTree, RegionTreeConfig};
use crate::index::CentroidIndex;
use crate::manifold::{self, ViableGraph};
use crate::merkle::{self, Digest, MerkleProof, MerkleTree};
use crate::pq::PqCodec;
use crate::projector::Projector;
use crate::steering::SteeringVector;

/// Flat, pre-allocated storage for every record's PQ codes + metadata,
/// indexed directly by record id: `codes[id * code_len .. +code_len]`, a
/// parallel `hash`/`meta_span`/`live` entry per id. Ids are assigned once by
/// `LatentDb::next_id` and never reused, so `id` doubles as a stable slot
/// index -- no separate id-to-slot lookup is needed. `insert`/`remove`/reads
/// within capacity just index into already-allocated memory; the only
/// points this ever touches the global allocator are the four parallel
/// arrays' own geometric-doubling growth (like `Vec::push`'s amortized
/// growth), plus `meta_bytes` growing on that same amortized schedule, one
/// buffer below.
///
/// `meta_bytes` is a single append-only buffer holding every record's
/// metadata concatenated together, sliced per-record via `meta_span`
/// (offset, len) -- mirrors the "one big buffer, no per-record Vec/String"
/// pattern katgpt-rs uses for its own KV-cache and slot-table storage,
/// applied to metadata as well as codes. It grows independently of the
/// other four arrays (its own `Vec<u8>`, sized by total metadata bytes
/// rather than record count), via `Vec::extend_from_slice`'s own amortized
/// growth rather than `ensure_capacity`. Removing a record clears its
/// `live` bit but does not reclaim its `meta_bytes` span or its `codes`
/// slot; see the README for what that means for long-running
/// eviction-heavy DBs.
#[derive(Serialize, Deserialize)]
struct RecordArena {
    code_len: usize,
    codes: Vec<u8>,
    hash: Vec<u64>,
    /// (offset, len) into `meta_bytes` for each id's metadata.
    meta_span: Vec<(u32, u32)>,
    meta_bytes: Vec<u8>,
    live: Vec<bool>,
    len: usize,
}

impl RecordArena {
    fn new(code_len: usize) -> Self {
        RecordArena {
            code_len,
            codes: Vec::new(),
            hash: Vec::new(),
            meta_span: Vec::new(),
            meta_bytes: Vec::new(),
            live: Vec::new(),
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.live.len()
    }

    /// Grow every parallel array so slot `min_capacity - 1` is addressable.
    /// Doubles (like `Vec::push`'s own amortized growth) rather than
    /// growing to the exact minimum, so a run of sequential ids only hits
    /// the allocator O(log n) times, not once per id.
    fn ensure_capacity(&mut self, min_capacity: usize) {
        if self.capacity() >= min_capacity {
            return;
        }
        let new_capacity = min_capacity.max(self.capacity().saturating_mul(2)).max(16);
        self.codes.resize(new_capacity * self.code_len, 0);
        self.hash.resize(new_capacity, 0);
        self.meta_span.resize(new_capacity, (0, 0));
        self.live.resize(new_capacity, false);
    }

    /// Store record `id`: `write_codes` is applied directly to this id's
    /// arena slot (no intermediate `Vec<u8>` -- the caller encodes straight
    /// into arena memory), and `metadata`'s bytes are appended to the flat
    /// metadata buffer. The only allocations this can cause are the arena's
    /// own amortized growth: `ensure_capacity` growing to fit `id`, and/or
    /// `meta_bytes` growing to fit the appended metadata.
    fn insert(&mut self, id: u64, hash: u64, metadata: &str, write_codes: impl FnOnce(&mut [u8])) {
        let idx = id as usize;
        self.ensure_capacity(idx + 1);

        let stride = self.code_len;
        write_codes(&mut self.codes[idx * stride..idx * stride + stride]);
        self.hash[idx] = hash;

        let offset = self.meta_bytes.len() as u32;
        self.meta_bytes.extend_from_slice(metadata.as_bytes());
        self.meta_span[idx] = (offset, metadata.len() as u32);

        if !self.live[idx] {
            self.len += 1;
        }
        self.live[idx] = true;
    }

    fn is_live(&self, id: u64) -> bool {
        (id as usize) < self.live.len() && self.live[id as usize]
    }

    fn codes(&self, id: u64) -> Option<&[u8]> {
        if !self.is_live(id) {
            return None;
        }
        let idx = id as usize;
        let start = idx * self.code_len;
        Some(&self.codes[start..start + self.code_len])
    }

    fn hash(&self, id: u64) -> Option<u64> {
        self.is_live(id).then(|| self.hash[id as usize])
    }

    fn metadata(&self, id: u64) -> Option<&str> {
        if !self.is_live(id) {
            return None;
        }
        let (offset, len) = self.meta_span[id as usize];
        let bytes = &self.meta_bytes[offset as usize..offset as usize + len as usize];
        // Valid UTF-8 and in-bounds by construction: `insert` is the only
        // writer and always appends a whole `&str`'s own bytes, and
        // `LatentDb::validate_after_deserialize` (called by every
        // deserialization entry point -- `load`, `WasmLatentDb::from_bytes`)
        // re-checks both properties for arenas that didn't come from
        // `insert` at all. A panic here means one of those two guarantees
        // has a bug, not that untrusted bytes reached this unchecked.
        Some(std::str::from_utf8(bytes).expect("metadata bytes are valid utf8 by construction"))
    }

    /// Clear id's live bit and return its stored hash, if it was live.
    /// Leaves its `codes` slot and `meta_bytes` span as unreclaimed dead
    /// space (see the struct docs and README).
    fn remove(&mut self, id: u64) -> Option<u64> {
        if !self.is_live(id) {
            return None;
        }
        let idx = id as usize;
        self.live[idx] = false;
        self.len -= 1;
        Some(self.hash[idx])
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Every currently-live id, in ascending order (a side effect of
    /// scanning slots 0..capacity in order -- ids are never reused, so this
    /// is also insertion order among still-live records).
    fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.live
            .iter()
            .enumerate()
            .filter_map(|(idx, &live)| live.then_some(idx as u64))
    }

    /// Rank every currently-live record by how many distinct `query_terms`
    /// (already lowercased by the caller) appear as an exact,
    /// whitespace-delimited token in its metadata, descending by that
    /// overlap count, ties broken by ascending id for a deterministic
    /// order. Records with zero overlap are dropped -- a lexical ranker has
    /// nothing to say about a record it found no match in, the same way an
    /// ANN index has nothing to say about a bucket it never probed.
    fn lexical_rank(&self, query_terms: &HashSet<String>) -> Vec<(u64, usize)> {
        let mut scored: Vec<(u64, usize)> = self
            .ids()
            .filter_map(|id| {
                let metadata = self.metadata(id)?;
                let matched: HashSet<String> = metadata
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .filter(|w| query_terms.contains(w))
                    .collect();
                (!matched.is_empty()).then_some((id, matched.len()))
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
    }

    /// Cross-field invariant check for an arena that may not have come from
    /// `insert` -- i.e. one just produced by `bincode::deserialize`.
    /// `insert`/`remove` above are the only writers on the normal path and
    /// always keep these invariants true by construction, so this is never
    /// called there; it exists purely so a corrupted or hand-crafted byte
    /// stream fails here, at the deserialization boundary, instead of
    /// succeeding and then panicking later inside `codes()`/`metadata()`
    /// (e.g. from deep inside `search()` or `merkle_leaves()`). Mirrors
    /// what bincode's own `String`/`Vec<u8>` deserialization already
    /// guaranteed for the old per-record `HashMap<u64, StoredRecord>`
    /// storage this arena replaced.
    fn validate(&self) -> Result<(), String> {
        let capacity = self.live.len();
        if self.codes.len() != capacity * self.code_len {
            return Err(format!(
                "codes length {} does not match capacity {capacity} * code_len {}",
                self.codes.len(),
                self.code_len
            ));
        }
        if self.hash.len() != capacity || self.meta_span.len() != capacity {
            return Err(format!(
                "hash length {} / meta_span length {} does not match capacity {capacity}",
                self.hash.len(),
                self.meta_span.len()
            ));
        }
        let live_count = self.live.iter().filter(|&&live| live).count();
        if live_count != self.len {
            return Err(format!(
                "len {} does not match {live_count} live slots",
                self.len
            ));
        }
        for (idx, &live) in self.live.iter().enumerate() {
            if !live {
                continue;
            }
            let (offset, span_len) = self.meta_span[idx];
            let end = offset as usize + span_len as usize;
            let bytes = self.meta_bytes.get(offset as usize..end).ok_or_else(|| {
                format!(
                    "id {idx}: metadata span {offset}..{end} is out of bounds \
                     (meta_bytes len {})",
                    self.meta_bytes.len()
                )
            })?;
            std::str::from_utf8(bytes)
                .map_err(|e| format!("id {idx}: metadata bytes are not valid utf8: {e}"))?;
        }
        Ok(())
    }
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

/// Build products [`LatentDb::ensure_merkle_cache`] caches across
/// `merkle_root()`/`merkle_proof()` calls: the tree itself (see `merkle.rs`
/// -- it already holds every level, not just the root) plus the ascending
/// id list mapping a record id to its leaf index (`ids[i]` is the id of leaf
/// `i`), so `merkle_proof(id)` can binary-search straight to a leaf index
/// instead of the linear scan a fresh `merkle_leaves()` call would need.
struct MerkleCache {
    ids: Vec<u64>,
    tree: MerkleTree,
}

#[derive(Serialize, Deserialize)]
pub struct LatentDb {
    dim: usize,
    projector: Projector,
    pq: PqCodec,
    index: CentroidIndex,
    records: RecordArena,
    /// content hash -> id, for de-duplication on insert
    hash_to_id: HashMap<u64, u64>,
    next_id: u64,
    /// Maximum number of stored records; 0 means unlimited. Enforced after
    /// every `insert()` and whenever changed via `set_record_budget()`.
    record_budget: usize,
    eviction_policy: EvictionPolicy,
    /// Reusable `projector.project_into` output buffer for `insert`, so it
    /// doesn't need to allocate a fresh `Vec<f32>` every call just to
    /// immediately hand it to `index.insert` and discard it. Skipped in
    /// (de)serialization -- re-sized lazily on first use after `load()`.
    #[serde(skip)]
    scratch: Vec<f32>,
    /// Lazily-built Merkle tree cache, `RefCell`-wrapped so `merkle_root()`/
    /// `merkle_proof()` can stay `&self` while still filling it in on first
    /// use. `None` means "rebuild on next access" -- true right after
    /// construction/deserialization, and set by `invalidate_merkle_cache()`
    /// after every `insert`/`remove`. See `ensure_merkle_cache`. This is the
    /// one field that makes `LatentDb` no longer auto-`Sync` (every other
    /// field is a plain, `Sync` value) -- fine today since nothing in this
    /// crate shares a `LatentDb` across threads, but worth knowing if that
    /// ever changes.
    #[serde(skip)]
    merkle_cache: RefCell<Option<MerkleCache>>,
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

        let code_len = pq.code_len();
        LatentDb {
            dim,
            projector,
            pq,
            index,
            records: RecordArena::new(code_len),
            hash_to_id: HashMap::new(),
            next_id: 0,
            record_budget: 0,
            eviction_policy: EvictionPolicy::OldestFirst,
            scratch: vec![0.0; sketch_dim],
            merkle_cache: RefCell::new(None),
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

        let metadata = metadata.into();

        if self.scratch.len() != self.projector.out_dim() {
            self.scratch = vec![0.0; self.projector.out_dim()];
        }
        self.projector.project_into(embedding, &mut self.scratch);
        self.index.insert(id, &self.scratch);

        let pq = &self.pq;
        self.records
            .insert(id, hash, &metadata, |slot| pq.encode_into(embedding, slot));

        self.hash_to_id.insert(hash, id);
        self.invalidate_merkle_cache();
        self.enforce_budget();
        Ok(id)
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(hash) = self.records.remove(id) {
            self.hash_to_id.remove(&hash);
            self.index.remove(id);
            self.invalidate_merkle_cache();
        }
    }

    /// Drop the cached Merkle tree so the next `merkle_root()`/
    /// `merkle_proof()` call rebuilds it from the live record set. Called by
    /// `insert`/`remove` whenever they actually change the stored record
    /// set (not on `insert`'s de-dup fast path, which returns before this
    /// point without touching any record).
    fn invalidate_merkle_cache(&mut self) {
        *self.merkle_cache.get_mut() = None;
    }

    /// Reconstruct the (approximate, PQ-decoded) embedding for a record.
    pub fn get_approx_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.records.codes(id).map(|codes| self.pq.decode(codes))
    }

    pub fn get_metadata(&self, id: u64) -> Option<&str> {
        self.records.metadata(id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The vector-ranking half of [`Self::search`], factored out so
    /// `search_blended()` can feed the same ranking's rank *positions* (not
    /// its raw scores) into Reciprocal Rank Fusion. Returns `(id, score)`
    /// pairs sorted descending by score, *not* truncated to any `k` --
    /// truncation and metadata lookup are each caller's own concern.
    fn vector_ranked_candidates(&self, query: &[f32], nprobe: usize) -> Vec<(u64, f32)> {
        let projected_query = self.projector.project(query);
        let candidates = self.index.candidates(&projected_query, nprobe);

        let lut = self.pq.build_query_lut(query);
        let mut scored: Vec<(u64, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                let codes = self.records.codes(id)?;
                Some((id, lut.cosine_score(codes)))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
    }

    /// Approximate nearest-neighbour search. `nprobe` controls how many
    /// centroid buckets get scanned (higher = more accurate, slower).
    ///
    /// Candidates are scored via a per-query asymmetric-distance lookup
    /// table (`PqCodec::build_query_lut`): the query's dot product and norm
    /// against every centroid in every subspace is computed once
    /// (`vector_ranked_candidates`), then each candidate's stored codes are
    /// summed against that table (`QueryLut::cosine_score`) -- no
    /// per-candidate PQ decode, no per-candidate allocation.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchHit> {
        if query.len() != self.dim {
            return Vec::new();
        }
        let mut scored = self.vector_ranked_candidates(query, nprobe);
        scored.truncate(k);
        scored
            .into_iter()
            .filter_map(|(id, score)| {
                let metadata = self.records.metadata(id)?.to_string();
                Some(SearchHit {
                    id,
                    score,
                    metadata,
                })
            })
            .collect()
    }

    /// Reciprocal Rank Fusion: fold one ranking's ids, in rank order, into
    /// `scores` -- a record at rank `r` (1-indexed, i.e. `ranked_ids`'
    /// position + 1) contributes `1 / (RRF_K + r)`, and a record this
    /// ranking never mentions contributes nothing. Called once per ranking
    /// being fused, so a record present in both accumulates both
    /// contributions.
    fn accumulate_rrf(scores: &mut HashMap<u64, f32>, ranked_ids: impl Iterator<Item = u64>) {
        const RRF_K: f32 = 60.0;
        for (rank, id) in ranked_ids.enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f32);
        }
    }

    /// Like [`Self::search`], but fuses a lexical/metadata term-overlap
    /// ranking (`query_terms` against each stored record's `metadata`
    /// string, see `RecordArena::lexical_rank`) with the vector `search()`
    /// ranking via Reciprocal Rank Fusion -- the fusion technique
    /// neuron-db's `recall_blended` implements to combine its own lexical
    /// and semantic recall paths (this crate's first borrowing from
    /// neuron-db rather than katgpt-rs; see the module docs).
    ///
    /// RRF only needs each ranking's *rank position*, not comparable score
    /// scales, which is what makes it a clean way to combine two very
    /// differently-scaled rankings (cosine similarity vs. term overlap
    /// count) without hand-tuned weighting (`accumulate_rrf`). Fused scores
    /// are summed across both rankings, sorted descending (ties broken by
    /// ascending id), and truncated to `k`. The returned `SearchHit::score`
    /// is this fused RRF score, not a cosine similarity -- comparable only
    /// against other `search_blended` results, not against `search`'s
    /// scores.
    ///
    /// The vector half is still bounded by `nprobe` (only the probed
    /// buckets' candidates can contribute a vector rank), but the lexical
    /// half scans every live record's metadata regardless of `nprobe` --
    /// mirroring two independent recall paths (bounded ANN vs. full-text)
    /// being fused, rather than lexical matching being limited to whatever
    /// the vector index happened to probe.
    ///
    /// Degrades to exactly `search(query_embedding, k, nprobe)` (same ids,
    /// order, and scores) when `query_terms` is empty, since there's then
    /// nothing for a lexical ranking to contribute.
    ///
    /// Returns an empty `Vec` if `query_embedding`'s dimension doesn't match
    /// `self.dim()`, matching `search`'s own dim-mismatch convention.
    pub fn search_blended(
        &self,
        query_terms: &[&str],
        query_embedding: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Vec<SearchHit> {
        if query_embedding.len() != self.dim {
            return Vec::new();
        }
        if query_terms.is_empty() {
            return self.search(query_embedding, k, nprobe);
        }

        let query_terms: HashSet<String> = query_terms.iter().map(|t| t.to_lowercase()).collect();
        let vector_ranked = self.vector_ranked_candidates(query_embedding, nprobe);
        let lexical_ranked = self.records.lexical_rank(&query_terms);

        let mut fused_scores: HashMap<u64, f32> = HashMap::new();
        Self::accumulate_rrf(&mut fused_scores, vector_ranked.iter().map(|&(id, _)| id));
        Self::accumulate_rrf(&mut fused_scores, lexical_ranked.iter().map(|&(id, _)| id));

        let mut fused: Vec<SearchHit> = fused_scores
            .into_iter()
            .filter_map(|(id, score)| {
                let metadata = self.records.metadata(id)?.to_string();
                Some(SearchHit {
                    id,
                    score,
                    metadata,
                })
            })
            .collect();

        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        fused.truncate(k);
        fused
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
        // `RecordArena::ids()` already yields ascending order (a side
        // effect of scanning slots in order), so no separate sort is needed
        // here the way the old `HashMap`-backed version required.
        let records = self
            .records
            .ids()
            .map(|id| (id, self.pq.decode(self.records.codes(id).unwrap())));
        manifold::build_viable_graph(records, predicate, k_nearest, edge_midpoint_check)
    }

    /// Build a [`RegionTree`] over every currently-stored record's
    /// approximate decoded vector: a hierarchical clustering (PCA +
    /// recursive k-means, see the `bandit` module) whose leaves are
    /// individual records and whose internal nodes are regions. Thompson-
    /// sample a region to explore next via `RegionTree::sample()`, then feed
    /// back how useful it was via `RegionTree::observe()`.
    ///
    /// Ids are visited in ascending order before decoding, the same
    /// determinism guarantee `build_viable_graph()` makes -- so the
    /// resulting tree's k-means splits (and therefore `sample()`'s output
    /// for a given seed) are stable across rebuilds of the same record set.
    ///
    /// Like `build_viable_graph()`, this is meant to be rebuilt when the
    /// record set changes meaningfully rather than maintained incrementally.
    ///
    /// # Panics
    /// Panics if the database is empty -- see [`RegionTree::build`].
    pub fn build_region_tree(&self, config: RegionTreeConfig) -> RegionTree {
        let records: Vec<(u64, Vec<f32>)> = self
            .records
            .ids()
            .map(|id| (id, self.pq.decode(self.records.codes(id).unwrap())))
            .collect();
        RegionTree::build(&records, config)
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
    fn energy(&self, id: u64) -> f32 {
        let centroid = self
            .index
            .assigned_centroid(id)
            .and_then(|c| self.index.centroid_vector(c));
        let Some(centroid) = centroid else {
            return f32::MAX; // not indexed (shouldn't happen) -- never evict first
        };
        let Some(codes) = self.records.codes(id) else {
            return f32::MAX; // not stored (shouldn't happen) -- never evict first
        };
        let approx = self.pq.decode(codes);
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

        let mut ids: Vec<u64> = self.records.ids().collect();
        match self.eviction_policy {
            // `RecordArena::ids()` already yields ascending (oldest-first)
            // order, so there's nothing left to do here -- kept as an
            // explicit arm (rather than folding this into an `if`) so the
            // compiler still flags this match as non-exhaustive if a third
            // `EvictionPolicy` variant is ever added.
            EvictionPolicy::OldestFirst => {}
            EvictionPolicy::LowestEnergy => {
                // Precompute every candidate's energy once -- O(n) energy
                // evaluations -- then sort the precomputed list, instead of
                // recomputing `energy()` (a decode + project) inside the
                // sort comparator, which would evaluate it O(n log n) times.
                let mut scored: Vec<(u64, f32)> =
                    ids.iter().map(|&id| (id, self.energy(id))).collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                ids = scored.into_iter().map(|(id, _)| id).collect();
            }
        }

        for id in ids.into_iter().take(excess) {
            self.remove(id);
        }
    }

    fn record_leaf_bytes(id: u64, hash: u64, codes: &[u8], metadata: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + codes.len() + metadata.len());
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&hash.to_le_bytes());
        bytes.extend_from_slice(codes);
        bytes.extend_from_slice(metadata.as_bytes());
        bytes
    }

    /// Ids in ascending order paired with their Merkle leaf hash -- the
    /// canonical, deterministic leaf ordering. `RecordArena::ids()` already
    /// yields ascending order (arena slots are scanned in order), so this
    /// stays stable across reloads the same way the old explicit sort over
    /// a `HashMap`'s keys did.
    fn merkle_leaves(&self) -> Vec<(u64, Digest)> {
        self.records
            .ids()
            .map(|id| {
                let hash = self.records.hash(id).unwrap();
                let codes = self.records.codes(id).unwrap();
                let metadata = self.records.metadata(id).unwrap();
                (
                    id,
                    merkle::hash_leaf(&Self::record_leaf_bytes(id, hash, codes, metadata)),
                )
            })
            .collect()
    }

    /// Fill `merkle_cache` from the live record set if it's currently empty
    /// (construction/deserialization, or the most recent `insert`/`remove`
    /// invalidated it via `invalidate_merkle_cache`). A no-op otherwise, so
    /// any number of `merkle_root()`/`merkle_proof()` calls between two
    /// mutations pay this O(n log n) `merkle_leaves()` rehash + tree build
    /// exactly once, not once per call.
    fn ensure_merkle_cache(&self) {
        if self.merkle_cache.borrow().is_some() {
            return;
        }
        let leaves = self.merkle_leaves();
        let ids = leaves.iter().map(|(id, _)| *id).collect();
        let tree = MerkleTree::build(leaves.into_iter().map(|(_, h)| h).collect());
        *self.merkle_cache.borrow_mut() = Some(MerkleCache { ids, tree });
    }

    /// Fill the cache if needed (`ensure_merkle_cache`), then hand back a
    /// borrow of it -- the one place `merkle_tree`/`merkle_root`/
    /// `merkle_proof` all go through, so there's a single ensure-then-borrow
    /// path instead of three copies of it.
    fn cached_merkle(&self) -> Ref<'_, MerkleCache> {
        self.ensure_merkle_cache();
        Ref::map(self.merkle_cache.borrow(), |cache| cache.as_ref().unwrap())
    }

    /// The Merkle tree over every currently-stored record, reusing the
    /// cached build (see `ensure_merkle_cache`) rather than rebuilding from
    /// scratch when nothing has changed since the last call.
    pub fn merkle_tree(&self) -> MerkleTree {
        self.cached_merkle().tree.clone()
    }

    /// Root commitment over every currently-stored record. Publish or store
    /// this value out-of-band as a checkpoint; `merkle_proof(id)` plus
    /// `MerkleProof::verify()` then let a third party confirm a specific
    /// record was included in that checkpoint without needing the whole
    /// database.
    pub fn merkle_root(&self) -> Digest {
        self.cached_merkle().tree.root()
    }

    /// Inclusion proof that record `id` is part of the current
    /// `merkle_root()`. Returns `None` if `id` isn't currently stored.
    ///
    /// Reuses the cached tree and id list when nothing has changed since the
    /// last call (see `ensure_merkle_cache`): once cached, this is an O(log
    /// n) binary search for `id`'s leaf index followed by an O(log n)
    /// sibling-path lookup against the tree's cached levels (`MerkleTree`
    /// keeps every level, not just the root -- see `merkle.rs`), instead of
    /// the full O(n log n) rehash + rebuild the old always-rebuild
    /// implementation paid on every single call.
    pub fn merkle_proof(&self, id: u64) -> Option<MerkleProof> {
        let cache = self.cached_merkle();
        let position = cache.ids.binary_search(&id).ok()?;
        cache.tree.proof(position)
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
        let db: LatentDb =
            bincode::deserialize(&bytes).map_err(|e| LatentDbError::Serialize(e.to_string()))?;
        db.validate_after_deserialize()?;
        Ok(db)
    }

    /// Check `records`'s cross-field invariants (see
    /// `RecordArena::validate`'s doc comment). Every deserialization entry
    /// point -- `load` above, and `wasm::WasmLatentDb::from_bytes`, which
    /// deserializes independently since `std::fs` (and therefore `load`)
    /// has no real backing on `wasm32-unknown-unknown` -- must call this
    /// before treating a freshly-deserialized `LatentDb` as trustworthy.
    pub(crate) fn validate_after_deserialize(&self) -> Result<(), LatentDbError> {
        self.records.validate().map_err(LatentDbError::Serialize)
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

    /// Issue 2 acceptance criterion: `search`'s LUT-scored top-k must match
    /// the pre-change (decode + `cosine_sim`) baseline's ids, order, and
    /// scores (within float tolerance) on a fixed synthetic corpus.
    /// `nprobe = n_index_centroids()` makes `search`'s candidate set
    /// exhaustive -- every stored id -- so the "baseline" computed here by
    /// decoding every record and scoring it directly is exactly what
    /// `search` itself scores, just via the old decode path instead of the
    /// new LUT path.
    #[test]
    fn search_lut_scoring_matches_decode_baseline_topk_and_order() {
        let corpus = synthetic_corpus(500, 32, 70);
        let mut db = LatentDb::build(&corpus, 4, 16, 8, 8, 71);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }

        let query = &corpus[123];
        let k = 10;
        let hits = db.search(query, k, db.n_index_centroids());

        let mut reference: Vec<(u64, f32)> = ids
            .iter()
            .map(|&id| {
                let approx = db.get_approx_vector(id).unwrap();
                (id, crate::superpose::cosine_sim(query, &approx))
            })
            .collect();
        reference.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        reference.truncate(k);

        assert_eq!(hits.len(), reference.len());
        for (hit, &(ref_id, ref_score)) in hits.iter().zip(reference.iter()) {
            assert_eq!(
                hit.id, ref_id,
                "top-k ordering diverged from decode baseline"
            );
            assert!(
                (hit.score - ref_score).abs() < 1e-4,
                "score diverged beyond tolerance: lut={} decode={}",
                hit.score,
                ref_score
            );
        }
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
        let root_before_save = db.merkle_root();
        let tmp = std::env::temp_dir().join("latent-db_test.bin");
        db.save(&tmp).unwrap();
        let loaded = LatentDb::load(&tmp).unwrap();
        assert_eq!(loaded.len(), db.len());
        assert_eq!(loaded.get_metadata(0), db.get_metadata(0));
        // `merkle_cache` is `#[serde(skip)]`, so this exercises a fresh
        // post-load build of the cache, not a (nonexistent) deserialized one.
        assert_eq!(loaded.merkle_root(), root_before_save);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn record_arena_validate_accepts_a_freshly_built_arena() {
        let mut arena = RecordArena::new(2);
        arena.insert(0, 42, "hello", |slot| slot.copy_from_slice(&[1, 2]));
        arena.insert(1, 7, "world", |slot| slot.copy_from_slice(&[3, 4]));
        assert!(arena.validate().is_ok());
    }

    #[test]
    fn record_arena_validate_rejects_an_out_of_bounds_metadata_span() {
        let mut arena = RecordArena::new(2);
        arena.insert(0, 42, "hello", |slot| slot.copy_from_slice(&[1, 2]));
        arena.meta_span[0] = (0, 9999);
        assert!(arena.validate().is_err());
    }

    #[test]
    fn record_arena_validate_rejects_invalid_utf8_metadata() {
        let mut arena = RecordArena::new(2);
        arena.insert(0, 42, "hello", |slot| slot.copy_from_slice(&[1, 2]));
        let (offset, len) = arena.meta_span[0];
        let start = offset as usize;
        // Same length as the original "hello" span, but not valid UTF-8.
        for b in &mut arena.meta_bytes[start..start + len as usize] {
            *b = 0xFF;
        }
        assert!(arena.validate().is_err());
    }

    /// The invariant `validate` checks used to be enforced for free by
    /// bincode's own `String` deserialization (the old per-record
    /// `HashMap<u64, StoredRecord>` storage this arena replaced would fail
    /// `load()` with a `Result::Err` on corrupted metadata bytes, since a
    /// `String` field can't deserialize invalid UTF-8 at all). This proves
    /// `load()` still fails the same way -- at the deserialization
    /// boundary, not later with a panic inside `search()`/`merkle_leaves()`
    /// -- now that metadata lives in a raw `Vec<u8>` arena buffer bincode
    /// itself doesn't validate.
    #[test]
    fn load_rejects_a_corrupted_arena_instead_of_deserializing_successfully() {
        let corpus = synthetic_corpus(20, 16, 500);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 501);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        db.records.meta_span[0] = (0, u32::MAX);

        let tmp = std::env::temp_dir().join("latent-db_corrupt_test.bin");
        db.save(&tmp).unwrap();
        let result = LatentDb::load(&tmp);
        assert!(
            result.is_err(),
            "load() should reject a corrupted arena rather than succeeding and \
             leaving a later, unrelated call to panic"
        );
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

    /// Repeated `merkle_root()`/`merkle_proof()` calls with no mutation in
    /// between must keep returning the exact same values as the cached tree
    /// is reused -- proves reusing the cache doesn't silently drift from
    /// what a fresh rebuild would produce.
    #[test]
    fn repeated_merkle_calls_without_mutation_are_stable() {
        let corpus = synthetic_corpus(25, 16, 14);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 15);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }

        let root_a = db.merkle_root();
        let root_b = db.merkle_root();
        assert_eq!(root_a, root_b);

        for &id in &ids {
            let proof_a = db.merkle_proof(id).unwrap();
            let proof_b = db.merkle_proof(id).unwrap();
            assert_eq!(proof_a.leaf_index, proof_b.leaf_index);
            assert_eq!(proof_a.leaf_hash, proof_b.leaf_hash);
            assert_eq!(proof_a.siblings, proof_b.siblings);
            assert!(proof_a.verify(&root_a));
        }
    }

    /// A record inserted after the cache was already warmed by an earlier
    /// `merkle_root()`/`merkle_proof()` call must still show up: the cache
    /// has to be invalidated and rebuilt on `insert`, not served stale.
    #[test]
    fn merkle_proof_sees_records_inserted_after_the_cache_was_warmed() {
        let corpus = synthetic_corpus(10, 16, 16);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 17);
        let first_id = db.insert(&corpus[0], "first").unwrap();

        // Warm the cache before the second insert.
        let _ = db.merkle_root();
        let _ = db.merkle_proof(first_id);

        let second_id = db.insert(&corpus[1], "second").unwrap();
        let root = db.merkle_root();
        let proof = db
            .merkle_proof(second_id)
            .expect("record inserted after cache warm-up should still be provable");
        assert!(proof.verify(&root));
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
            assert!(
                db.get_metadata(id).is_some(),
                "newest record {id} should survive"
            );
        }
        for id in &ids[..ids.len() - 5] {
            assert!(
                db.get_metadata(*id).is_none(),
                "oldest record {id} should be evicted"
            );
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
        let outlier: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 50.0 } else { -50.0 })
            .collect();
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

    /// Issue 9 acceptance criterion: a record whose `metadata` unambiguously
    /// matches the query terms, but whose embedding is equally
    /// (dis)similar to two far-apart clusters, should rank above plain
    /// vector search's top hit once lexical and vector rankings are fused.
    ///
    /// Two tight clusters sit at opposite constant vectors (-5s and +5s);
    /// the ambiguous record's embedding alternates +3/-3 so its *raw* dot
    /// product against an all-(-5) query is exactly zero -- the same "no
    /// lean toward either cluster" signal a genuinely in-between embedding
    /// would give a cosine-based ranker (PQ quantization can nudge the
    /// actual `cosine_score` slightly off zero, but not toward either
    /// cluster in particular). Only that record's metadata contains the
    /// query terms.
    #[test]
    fn search_blended_ranks_lexically_unambiguous_but_vector_ambiguous_record_above_plain_search() {
        let dim = 16;
        let mut corpus = Vec::new();
        for center in [-5.0f32, 5.0] {
            for i in 0..8 {
                let mut v = vec![center; dim];
                v[1] += i as f32 * 0.001; // tiny jitter so cluster members aren't identical
                corpus.push(v);
            }
        }
        let ambiguous: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        corpus.push(ambiguous);

        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 100);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            let metadata = if i == corpus.len() - 1 {
                "widget catalog".to_string()
            } else {
                format!("record-{i}")
            };
            ids.push(db.insert(v, metadata).unwrap());
        }
        let ambiguous_id = *ids.last().unwrap();

        // Dead-center on cluster A: plain vector search should strongly
        // prefer pure cluster-A members over the equally-(dis)similar
        // ambiguous record.
        let query = vec![-5.0f32; dim];
        let query_terms = ["widget", "catalog"];
        let nprobe = db.n_index_centroids();

        let plain = db.search(&query, 5, nprobe);
        assert_ne!(
            plain[0].id, ambiguous_id,
            "plain vector search shouldn't favor the vector-ambiguous record"
        );

        let blended = db.search_blended(&query_terms, &query, 5, nprobe);
        assert_eq!(
            blended[0].id, ambiguous_id,
            "an unambiguous lexical match should win the fused ranking"
        );
    }

    /// Issue 9 acceptance criterion: with no query terms, there's nothing
    /// for a lexical ranking to contribute, so `search_blended` must
    /// degrade to exactly `search`'s ids, order, and scores.
    #[test]
    fn search_blended_with_no_query_terms_matches_plain_vector_search() {
        let corpus = synthetic_corpus(120, 16, 200);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 201);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        let query = &corpus[5];
        let nprobe = db.n_index_centroids();

        let plain = db.search(query, 5, nprobe);
        let blended = db.search_blended(&[], query, 5, nprobe);

        assert_eq!(plain.len(), blended.len());
        for (a, b) in plain.iter().zip(blended.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.score, b.score);
            assert_eq!(a.metadata, b.metadata);
        }
    }

    #[test]
    fn search_blended_rejects_dimension_mismatch() {
        let corpus = synthetic_corpus(50, 16, 300);
        let mut db = LatentDb::build(&corpus, 2, 8, 4, 8, 301);
        for v in &corpus {
            db.insert(v, "x").unwrap();
        }
        let wrong_dim_query = vec![0.0f32; 8];
        assert!(db
            .search_blended(&["x"], &wrong_dim_query, 5, db.n_index_centroids())
            .is_empty());
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

    #[test]
    fn build_region_tree_covers_every_stored_record() {
        let corpus = synthetic_corpus(60, 16, 90);
        let mut db = LatentDb::build(&corpus, 4, 16, 8, 8, 91);
        let mut ids = Vec::new();
        for (i, v) in corpus.iter().enumerate() {
            ids.push(db.insert(v, format!("r{i}")).unwrap());
        }

        let tree = db.build_region_tree(RegionTreeConfig::default());
        assert_eq!(tree.num_arms(), db.len());
        for &id in &ids {
            assert!(tree.leaf_belief(id).is_some());
        }

        let arm = tree.sample(1);
        assert!(ids.contains(&arm));
    }
}
