//! `latentdb-cli` -- a thin command-line wrapper around the public
//! `LatentDb` API (see `latent_db::db::LatentDb`), plus a `bench` subcommand
//! that runs this crate's `benches/` suite and a `serve` subcommand that
//! exposes `LatentDb::search`/`insert` over HTTP (see `server.rs`).
//!
//! Each of `build`/`insert`/`search`/`save`/`load` maps directly onto the
//! `LatentDb` method of the same name -- `build` trains a fresh DB and
//! writes it to `--db`, `insert`/`search` load `--db`, call the matching
//! method, and (for `insert`) save the result back, and `save`/`load` are
//! explicit wrappers around `LatentDb::save`/`load` for scripting/debugging
//! (e.g. `load` alone to print a DB file's summary without mutating it).
//!
//! Vectors are passed as comma-separated floats (`--embedding
//! "0.1,0.2,0.3"`) rather than a richer format, since that's all a shell
//! one-liner or the round-trip demo in `tests/cli.rs` needs; `build`'s
//! training set is the one place with enough data that a flat JSON array of
//! arrays (`--training corpus.json`) is worth it instead.
//!
//! Run with `cargo run --features cli --bin latentdb-cli -- <subcommand>`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use latent_db::{EvictionPolicy, LatentDb};

mod server;

#[derive(Parser)]
#[command(name = "latentdb-cli", about = "Command-line interface for latent-db")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a fresh LatentDb from a JSON training set and write it to `--db`.
    Build {
        /// Path to write the newly-built (empty) DB to.
        #[arg(long)]
        db: PathBuf,
        /// JSON file: an array of arrays of f32, e.g. `[[0.1,0.2],[0.3,0.4]]`.
        #[arg(long)]
        training: PathBuf,
        #[arg(long, default_value_t = 8)]
        n_subspaces: usize,
        #[arg(long, default_value_t = 32)]
        n_pq_centroids: usize,
        #[arg(long, default_value_t = 16)]
        n_index_centroids: usize,
        #[arg(long, default_value_t = 8)]
        sketch_dim: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Insert one embedding + metadata into an existing DB, saving it back.
    Insert {
        #[arg(long)]
        db: PathBuf,
        /// Comma-separated floats, e.g. "0.1,-0.2,0.3".
        #[arg(long, allow_hyphen_values = true)]
        embedding: String,
        #[arg(long)]
        metadata: String,
    },
    /// Approximate nearest-neighbour search against an existing DB.
    Search {
        #[arg(long)]
        db: PathBuf,
        /// Comma-separated floats, e.g. "0.1,-0.2,0.3".
        #[arg(long, allow_hyphen_values = true)]
        query: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 4)]
        nprobe: usize,
    },
    /// Load a DB and re-save it (a copy if `--out` differs from `--db`).
    Save {
        #[arg(long)]
        db: PathBuf,
        /// Destination path; defaults to `--db` (re-save in place).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Load a DB and print a summary (dim, record count, compression ratio, ...).
    Load {
        #[arg(long)]
        db: PathBuf,
    },
    /// Run (or report how to run) this crate's `benches/` suite.
    Bench {
        /// Only run the bench with this `[[bench]]` name (see Cargo.toml).
        #[arg(long)]
        name: Option<String>,
        /// Print the available bench names and how to run them, without running anything.
        #[arg(long, default_value_t = false)]
        list: bool,
    },
    /// Boot an HTTP server exposing /healthz, /search, /insert over a loaded DB.
    Serve {
        #[arg(long)]
        db: PathBuf,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

fn eviction_policy_name(p: EvictionPolicy) -> &'static str {
    match p {
        EvictionPolicy::OldestFirst => "oldest-first",
        EvictionPolicy::LowestEnergy => "lowest-energy",
    }
}

fn parse_floats(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|tok| {
            tok.trim()
                .parse::<f32>()
                .with_context(|| format!("invalid float {tok:?} in {s:?}"))
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn cmd_build(
    db: PathBuf,
    training: PathBuf,
    n_subspaces: usize,
    n_pq_centroids: usize,
    n_index_centroids: usize,
    sketch_dim: usize,
    seed: u64,
) -> Result<()> {
    let raw = std::fs::read_to_string(&training)
        .with_context(|| format!("reading training file {training:?}"))?;
    let training_vectors: Vec<Vec<f32>> =
        serde_json::from_str(&raw).with_context(|| format!("parsing {training:?} as JSON"))?;
    if training_vectors.is_empty() {
        bail!("training file {training:?} contains no vectors");
    }

    let latent_db = LatentDb::build(
        &training_vectors,
        n_subspaces,
        n_pq_centroids,
        n_index_centroids,
        sketch_dim,
        seed,
    );
    latent_db
        .save(&db)
        .with_context(|| format!("saving new DB to {db:?}"))?;

    println!(
        "{}",
        serde_json::json!({
            "db": db.to_string_lossy(),
            "dim": latent_db.dim(),
            "trainingVectors": training_vectors.len(),
        })
    );
    Ok(())
}

fn cmd_insert(db: PathBuf, embedding: String, metadata: String) -> Result<()> {
    let embedding = parse_floats(&embedding)?;
    let mut latent_db = LatentDb::load(&db).with_context(|| format!("loading DB from {db:?}"))?;
    let id = latent_db.insert(&embedding, metadata)?;
    latent_db
        .save(&db)
        .with_context(|| format!("saving DB back to {db:?}"))?;

    println!(
        "{}",
        serde_json::json!({ "id": id, "len": latent_db.len() })
    );
    Ok(())
}

fn cmd_search(db: PathBuf, query: String, k: usize, nprobe: usize) -> Result<()> {
    let query = parse_floats(&query)?;
    let latent_db = LatentDb::load(&db).with_context(|| format!("loading DB from {db:?}"))?;
    if query.len() != latent_db.dim() {
        bail!(
            "query has dim {}, expected {}",
            query.len(),
            latent_db.dim()
        );
    }
    let hits = latent_db.search(&query, k, nprobe);

    let json: Vec<serde_json::Value> = hits
        .into_iter()
        .map(|h| serde_json::json!({ "id": h.id, "score": h.score, "metadata": h.metadata }))
        .collect();
    println!("{}", serde_json::to_string(&json)?);
    Ok(())
}

fn cmd_save(db: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let latent_db = LatentDb::load(&db).with_context(|| format!("loading DB from {db:?}"))?;
    let out = out.unwrap_or_else(|| db.clone());
    latent_db
        .save(&out)
        .with_context(|| format!("saving DB to {out:?}"))?;

    println!(
        "{}",
        serde_json::json!({ "db": out.to_string_lossy(), "len": latent_db.len() })
    );
    Ok(())
}

fn cmd_load(db: PathBuf) -> Result<()> {
    let latent_db = LatentDb::load(&db).with_context(|| format!("loading DB from {db:?}"))?;
    println!(
        "{}",
        serde_json::json!({
            "db": db.to_string_lossy(),
            "dim": latent_db.dim(),
            "len": latent_db.len(),
            "compressionRatio": latent_db.compression_ratio(),
            "nIndexCentroids": latent_db.n_index_centroids(),
            "recordBudget": latent_db.record_budget(),
            "evictionPolicy": eviction_policy_name(latent_db.eviction_policy()),
            "merkleRoot": hex(&latent_db.merkle_root()),
        })
    );
    Ok(())
}

/// Names must match the `[[bench]]` entries in Cargo.toml.
const BENCH_NAMES: &[&str] = &[
    "search_bench",
    "merkle_proof_bench",
    "viable_graph_bench",
    "projector_bench",
    "euclidean_kernel_bench",
];

/// Directory containing this crate's `Cargo.toml`, baked in at compile time.
/// Anchors the `cargo bench` invocation below so `latentdb-cli bench` works
/// regardless of the caller's current directory -- without this, running
/// the installed binary from anywhere outside the repo fails with "could
/// not find Cargo.toml in ... or any parent directory".
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

fn cmd_bench(name: Option<String>, list: bool) -> Result<()> {
    if list {
        println!("Available benches (run with `cargo bench --bench <name>`):");
        for n in BENCH_NAMES {
            println!("  {n}");
        }
        println!(
            "Alloc-tracking tests from the same harness (run with `cargo test`):\n  \
             tests/search_alloc_check.rs\n  tests/merkle_alloc_check.rs"
        );
        return Ok(());
    }

    if let Some(name) = &name {
        if !BENCH_NAMES.contains(&name.as_str()) {
            bail!(
                "unknown bench {name:?}; run `latentdb-cli bench --list` for the available names"
            );
        }
    }

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench").current_dir(MANIFEST_DIR);
    if let Some(name) = &name {
        cmd.args(["--bench", name]);
    }

    println!("running: {cmd:?}");
    let status = cmd
        .status()
        .context("spawning `cargo bench` -- is cargo on PATH?")?;
    if !status.success() {
        bail!("cargo bench exited with {status}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build {
            db,
            training,
            n_subspaces,
            n_pq_centroids,
            n_index_centroids,
            sketch_dim,
            seed,
        } => cmd_build(
            db,
            training,
            n_subspaces,
            n_pq_centroids,
            n_index_centroids,
            sketch_dim,
            seed,
        ),
        Command::Insert {
            db,
            embedding,
            metadata,
        } => cmd_insert(db, embedding, metadata),
        Command::Search {
            db,
            query,
            k,
            nprobe,
        } => cmd_search(db, query, k, nprobe),
        Command::Save { db, out } => cmd_save(db, out),
        Command::Load { db } => cmd_load(db),
        Command::Bench { name, list } => cmd_bench(name, list),
        Command::Serve { db, port } => server::serve(db, port),
    }
}
