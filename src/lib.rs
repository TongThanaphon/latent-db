//! # latent-db
//!
//! A small embeddable "database" for vector embeddings that keeps every
//! record compressed in latent space, inspired by ideas from
//! [katgpt-rs](https://github.com/katopz/katgpt-rs) (MUX-Latent superposition,
//! ShardEmbedding random projection, Hybrid OCT+PQ compression, and Schema
//! Centroid indexing) -- reimplemented independently here as general,
//! well-established techniques (Johnson-Lindenstrauss random projection,
//! product quantization, IVF-style centroid indexing, and Holographic
//! Reduced Representations for superposition) rather than copied code.
//!
//! Two ways to use it:
//!
//! 1. **`LatentDb`** -- the main path. Insert full embeddings + metadata,
//!    get back an id; each record is stored as a handful of PQ code bytes
//!    (not raw f32s) and indexed by nearest centroid for fast approximate
//!    search.
//! 2. **`SuperposedSlot`** -- an extra, more extreme mode where *many*
//!    (key, value) pairs are packed into a *single* vector of fixed size,
//!    named after but not algorithmically equivalent to katgpt-rs's
//!    MUX-Latent span compression (that scheme is lossless; this one isn't
//!    -- see `superpose` module docs). Retrieval is approximate and
//!    degrades as more pairs share a slot.
//!
//! `LatentDb` also exposes `merkle_root()` / `merkle_proof(id)` for
//! per-record inclusion proofs over the whole database, adapted from
//! katgpt-rs's `MerkleOctree`/`MerkleProof` (see the `merkle` module).
//!
//! Two further pieces, reimplemented independently from katgpt-rs's game-AI
//! primitives of the same conceptual lineage:
//!
//! - **`manifold`** -- `LatentDb::build_viable_graph()` builds a kNN
//!   navigation graph over the subset of stored records passing a caller
//!   predicate, then `ViableGraph::geodesic()` / `random_walk()` traverse it
//!   without ever visiting a record that fails the predicate. Inspired by
//!   katgpt-rs's Viable Manifold Graph (arXiv:2206.00106 distillation).
//!   `ViableGraph::weighted_random_walk()` is the same traversal with a
//!   caller-supplied weight function biasing which neighbor gets sampled
//!   next (e.g. by cosine similarity to a `steering::SteeringVector`
//!   direction) instead of `random_walk()`'s uniform draw -- ported from
//!   katgpt-rs's `manifold_curiosity_walk`, an omission from the earlier
//!   port rather than a deliberate drop. `ViableGraph::boundary_classes()`
//!   complements that with a structural view of the same graph: a
//!   signature-refinement partition of its nodes
//!   into equivalence classes, so structurally redundant records collapse
//!   together and structurally distinct ones separate out -- a from-scratch
//!   reinterpretation of katgpt-rs's bisimulation-refinement idea for a kNN
//!   graph rather than a labeled transition system.
//! - **`steering`** -- `LatentDb::search_steered()` shifts a query vector by
//!   a BLAKE3-committed, unit-norm direction before searching, biasing
//!   results toward a concept axis without retraining anything. Inspired by
//!   katgpt-rs's Latent Field Steering (Plan 309).
//! - **`trajectory`** -- `path_geometry()` scores a `ViableGraph::geodesic()`
//!   / `random_walk()` path's length, curvature, and minimum adjacent-step
//!   cosine similarity; `bifurcation_ratio()` measures how much two such
//!   paths diverge from a shared start. Ported from katgpt-rs's
//!   `latent_trajectory_geometry` (arXiv:2606.09287 distillation), which is
//!   pure geometry over a vector sequence with no game-specific assumptions.
//! - **`bandit`** -- `LatentDb::build_region_tree()` builds a `RegionTree`:
//!   a hierarchical clustering (PCA + recursive k-means) of currently-stored
//!   record embeddings into regions, with `RegionTree::sample()` descending
//!   it via Thompson sampling to pick a region under uncertainty and
//!   `RegionTree::observe()` updating a region's reward belief from
//!   feedback -- reward-driven exploration, distinct from
//!   `random_walk`/`weighted_random_walk`'s distance- or predicate-driven
//!   traversal. Ported from katgpt-rs's `manifold_bandit::LatentTaskTree`
//!   (arXiv:2606.19750 distillation).
//!
//! `LatentDb::search_blended()` is this crate's first borrowing from
//! **neuron-db** rather than katgpt-rs: it fuses a lexical/metadata
//! term-overlap ranking with the vector `search()` ranking via Reciprocal
//! Rank Fusion, the technique neuron-db's `recall_blended` actually
//! implements to combine its own lexical and semantic recall paths. Unlike
//! katgpt-rs, neuron-db isn't itself a vector database, so this multi-signal
//! recall/fusion pattern is its own genuine contribution rather than shared
//! lineage with the rest of this crate. Purely additive: `search()` and
//! `search_steered()` are unchanged.

pub mod alloc;
pub mod bandit;
pub mod db;
pub mod index;
pub mod manifold;
pub mod merkle;
pub mod pq;
pub mod projector;
pub mod simd;
pub mod steering;
pub mod superpose;
pub mod trajectory;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use bandit::{RegionTree, RegionTreeConfig};
pub use db::{EvictionPolicy, LatentDb, LatentDbError, SearchHit};
pub use index::CentroidIndex;
pub use manifold::{build_viable_graph, BoundaryClassId, BoundaryClasses, ViableGraph};
pub use merkle::{Digest, MerkleProof, MerkleTree};
pub use pq::{PqCodec, QueryLut};
pub use projector::Projector;
pub use steering::{SteeringEnvelope, SteeringError, SteeringVector};
pub use superpose::{circular_convolve, circular_correlate, cosine_sim, SuperposedSlot};
pub use trajectory::{bifurcation_ratio, path_geometry, BifurcationResult, PathGeometry};

// Debug-only global allocator (see `alloc` module docs): tracks per-thread
// allocation count/bytes so `*_alloc_check` tests and benches can assert a
// hot path's allocation behavior. Installed here, at the crate root, rather
// than gated on `cfg(test)`, so it's also active for `tests/` integration
// tests and `benches/` binaries in this package -- both link this crate as a
// plain rlib dependency, where `cfg(test)` is never set on *this* crate.
// Compiles away entirely in release builds (`cargo bench` defaults to the
// release profile, so timing benches see zero counting overhead). Excluded
// on wasm32 since the `wasm` feature's cdylib output isn't a benchmarking
// target and doesn't need allocation tracking.
#[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;
