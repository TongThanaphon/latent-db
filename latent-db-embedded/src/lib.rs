//! Fixed-size, zero-allocation vector/graph primitives for embedded targets
//! (Issue #20). Coexists with, and shares no code with, the dynamic-dimension
//! `latent-db` crate one level up -- see
//! `docs/adr/0001-latent-db-embedded-coexistence.md` for why this is a
//! separate crate rather than a `no_std` conversion of that one.
#![no_std]

/// Fixed embedding dimension every vector in this crate holds.
pub const DIM: usize = 768;

/// Upper bound on outgoing edges a single graph node may hold.
pub const MAX_NEIGHBORS: usize = 16;
