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

pub mod db;
pub mod index;
pub mod merkle;
pub mod pq;
pub mod projector;
pub mod superpose;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use db::{EvictionPolicy, LatentDb, LatentDbError, SearchHit};
pub use index::CentroidIndex;
pub use merkle::{Digest, MerkleProof, MerkleTree};
pub use pq::PqCodec;
pub use projector::Projector;
pub use superpose::{circular_convolve, circular_correlate, cosine_sim, SuperposedSlot};
