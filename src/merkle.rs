//! General-purpose binary Merkle tree (BLAKE3), giving `LatentDb` tamper
//! evidence and per-record inclusion proofs.
//!
//! Adapted from katgpt-rs's `MerkleOctree` / `MerkleProof`
//! (`crates/katgpt-types/src/merkle.rs`): same core idea (bottom-up BLAKE3
//! commitment tree + sibling-path inclusion proofs), but that type is a
//! *fixed* depth-3, 8-way octree sized for exactly 64 leaves (knowledge-graph
//! nodes). `LatentDb`'s record count grows and shrinks at runtime, so this
//! is a standard binary tree over an arbitrary number of leaves instead.
//!
//! Leaf hashes are prefixed `0x00` and internal-node hashes `0x01` before
//! BLAKE3-hashing (the domain separation Certificate Transparency / RFC 6962
//! uses), so a leaf hash can never be replayed as a forged internal node
//! and vice versa.

use serde::{Deserialize, Serialize};

pub const HASH_SIZE: usize = 32;
pub type Digest = [u8; HASH_SIZE];

const LEAF_PREFIX: u8 = 0x00;
const NODE_PREFIX: u8 = 0x01;

/// Hash a leaf's raw content into a `Digest` suitable for `MerkleTree::build`.
pub fn hash_leaf(data: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[LEAF_PREFIX]);
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

fn hash_pair(left: &Digest, right: &Digest) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[NODE_PREFIX]);
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// A binary Merkle tree built bottom-up from pre-hashed leaves (see
/// [`hash_leaf`]). An unpaired trailing node at any level is paired with
/// itself, the same convention Bitcoin-style Merkle trees use.
#[derive(Clone, Debug, Default)]
pub struct MerkleTree {
    /// `levels[0]` = leaf hashes, `levels.last()` = `[root]`.
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    pub fn build(leaf_hashes: Vec<Digest>) -> Self {
        if leaf_hashes.is_empty() {
            return MerkleTree {
                levels: vec![Vec::new()],
            };
        }
        let mut levels = vec![leaf_hashes];
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let next = prev
                .chunks(2)
                .map(|pair| match pair {
                    [a, b] => hash_pair(a, b),
                    [a] => hash_pair(a, a),
                    _ => unreachable!(),
                })
                .collect();
            levels.push(next);
        }
        MerkleTree { levels }
    }

    /// Root commitment over every leaf. For an empty tree this is all
    /// zero bytes; for a single-leaf tree it's that leaf's hash unchanged.
    pub fn root(&self) -> Digest {
        self.levels
            .last()
            .and_then(|l| l.first())
            .copied()
            .unwrap_or([0u8; HASH_SIZE])
    }

    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate an inclusion proof for the leaf at `leaf_index`.
    pub fn proof(&self, leaf_index: usize) -> Option<MerkleProof> {
        if leaf_index >= self.len() {
            return None;
        }
        let leaf_hash = self.levels[0][leaf_index];
        let mut siblings = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut idx = leaf_index;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = idx ^ 1;
            let sibling = *level.get(sibling_idx).unwrap_or(&level[idx]);
            siblings.push(sibling);
            idx /= 2;
        }
        Some(MerkleProof {
            leaf_index: leaf_index as u64,
            leaf_hash,
            siblings,
        })
    }
}

/// Inclusion proof that a leaf is part of a `MerkleTree`'s `root()`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: u64,
    pub leaf_hash: Digest,
    pub siblings: Vec<Digest>,
}

impl MerkleProof {
    /// Recompute the root from `leaf_hash` + `siblings` and compare against
    /// `expected_root`.
    pub fn verify(&self, expected_root: &Digest) -> bool {
        let mut current = self.leaf_hash;
        let mut idx = self.leaf_index;
        for sibling in &self.siblings {
            current = if idx.is_multiple_of(2) {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            idx /= 2;
        }
        current == *expected_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Digest> {
        (0..n).map(|i| hash_leaf(&i.to_le_bytes())).collect()
    }

    #[test]
    fn empty_tree_has_zero_root_and_no_proofs() {
        let tree = MerkleTree::build(Vec::new());
        assert_eq!(tree.root(), [0u8; HASH_SIZE]);
        assert!(tree.proof(0).is_none());
    }

    #[test]
    fn single_leaf_tree_root_is_the_leaf_hash() {
        let leaf = hash_leaf(b"only-record");
        let tree = MerkleTree::build(vec![leaf]);
        assert_eq!(tree.root(), leaf);
        let proof = tree.proof(0).unwrap();
        assert!(proof.verify(&tree.root()));
    }

    #[test]
    fn every_leaf_proof_verifies_against_the_root() {
        for n in [1, 2, 3, 4, 5, 8, 13, 17, 64] {
            let tree = MerkleTree::build(leaves(n));
            let root = tree.root();
            for i in 0..n {
                let proof = tree.proof(i).unwrap();
                assert!(proof.verify(&root), "leaf {i} of {n} failed to verify");
            }
        }
    }

    #[test]
    fn tampered_leaf_hash_fails_verification() {
        let tree = MerkleTree::build(leaves(10));
        let root = tree.root();
        let mut proof = tree.proof(3).unwrap();
        proof.leaf_hash[0] ^= 0xFF;
        assert!(!proof.verify(&root));
    }

    #[test]
    fn tampered_sibling_fails_verification() {
        let tree = MerkleTree::build(leaves(10));
        let root = tree.root();
        let mut proof = tree.proof(3).unwrap();
        proof.siblings[0][0] ^= 0xFF;
        assert!(!proof.verify(&root));
    }

    #[test]
    fn out_of_range_leaf_index_returns_none() {
        let tree = MerkleTree::build(leaves(5));
        assert!(tree.proof(5).is_none());
    }

    #[test]
    fn root_changes_when_leaves_change() {
        let a = MerkleTree::build(leaves(5)).root();
        let b = MerkleTree::build(leaves(6)).root();
        assert_ne!(a, b);
    }
}
