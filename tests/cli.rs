//! Integration tests for `latentdb-cli` (Issue 10 acceptance criteria).
//!
//! Runs the actual compiled binary via `CARGO_BIN_EXE_latentdb-cli` rather
//! than calling library code directly, so these tests exercise the same
//! surface a user invokes from a shell: argument parsing (including
//! negative-float embeddings, which look like flags to a naive parser),
//! JSON stdout, and on-disk persistence across separate process
//! invocations.
//!
//! `#![cfg(feature = "cli")]` (plus `required-features = ["cli"]` on this
//! `[[test]]` in Cargo.toml): only meaningful with `cargo test --features
//! cli`, matching how `Command::Serve` needs axum/tokio, which aren't built
//! at all for a plain `cargo test`.

#![cfg(feature = "cli")]

use std::io::Read;
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_latentdb-cli")
}

fn run(args: &[&str]) -> serde_json::Value {
    let output = Command::new(bin())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run latentdb-cli {args:?}: {e}"));
    assert!(
        output.status.success(),
        "latentdb-cli {args:?} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("non-JSON stdout from {args:?}: {e}"))
}

fn write_training_file(path: &Path, n: usize, dim: usize) {
    let mut rng_state = 12345u64;
    let mut next = move || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
    std::fs::write(path, serde_json::to_string(&vectors).unwrap()).unwrap();
}

/// Acceptance criterion: `build ... && insert ... && search ...` round-trips
/// correctly against a saved DB file.
#[test]
fn build_insert_search_round_trips_against_a_saved_db_file() {
    let dir = std::env::temp_dir().join(format!("latentdb-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("round_trip.db");
    let training_path = dir.join("training.json");
    write_training_file(&training_path, 40, 8);

    let build_out = run(&[
        "build",
        "--db",
        db_path.to_str().unwrap(),
        "--training",
        training_path.to_str().unwrap(),
        "--n-subspaces",
        "2",
        "--n-pq-centroids",
        "8",
        "--n-index-centroids",
        "4",
        "--sketch-dim",
        "8",
        "--seed",
        "7",
    ]);
    assert_eq!(build_out["dim"], 8);

    let embedding = "0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8";
    let insert_out = run(&[
        "insert",
        "--db",
        db_path.to_str().unwrap(),
        "--embedding",
        embedding,
        "--metadata",
        "round-trip-doc",
    ]);
    let id = insert_out["id"].as_u64().unwrap();
    assert_eq!(insert_out["len"], 1);

    let search_out = run(&[
        "search",
        "--db",
        db_path.to_str().unwrap(),
        "--query",
        embedding,
        "--k",
        "5",
        "--nprobe",
        "4",
    ]);
    let hits = search_out.as_array().unwrap();
    assert!(!hits.is_empty(), "expected at least one search hit");
    assert_eq!(hits[0]["id"].as_u64().unwrap(), id);
    assert_eq!(hits[0]["metadata"], "round-trip-doc");

    let load_out = run(&["load", "--db", db_path.to_str().unwrap()]);
    assert_eq!(load_out["len"], 1);
    assert_eq!(load_out["dim"], 8);

    let copy_path = dir.join("round_trip_copy.db");
    run(&[
        "save",
        "--db",
        db_path.to_str().unwrap(),
        "--out",
        copy_path.to_str().unwrap(),
    ]);
    let copy_load_out = run(&["load", "--db", copy_path.to_str().unwrap()]);
    assert_eq!(copy_load_out["len"], 1);
    assert_eq!(copy_load_out["merkleRoot"], load_out["merkleRoot"]);

    std::fs::remove_dir_all(&dir).ok();
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Minimal blocking HTTP/1.1 client, just enough to POST JSON and read a
/// response body without pulling in a dev-dependency HTTP client purely for
/// one test file.
fn http_post(port: u16, path: &str, body: &str) -> String {
    use std::io::Write;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
}

fn http_get(port: u16, path: &str) -> String {
    use std::io::Write;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
}

/// Acceptance criterion: `latentdb-cli serve` responds correctly to a
/// `/search` request against a loaded DB.
#[test]
fn serve_responds_to_healthz_search_and_insert() {
    let dir = std::env::temp_dir().join(format!("latentdb-cli-serve-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("serve.db");
    let training_path = dir.join("training.json");
    write_training_file(&training_path, 40, 8);

    run(&[
        "build",
        "--db",
        db_path.to_str().unwrap(),
        "--training",
        training_path.to_str().unwrap(),
        "--n-subspaces",
        "2",
        "--n-pq-centroids",
        "8",
        "--n-index-centroids",
        "4",
        "--sketch-dim",
        "8",
        "--seed",
        "7",
    ]);
    run(&[
        "insert",
        "--db",
        db_path.to_str().unwrap(),
        "--embedding",
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        "--metadata",
        "served-doc",
    ]);

    // Port chosen to avoid colliding with common dev-server defaults; not
    // fully collision-proof but good enough for a single-process test run.
    let port: u16 = 18_812;
    let mut child = Command::new(bin())
        .args([
            "serve",
            "--db",
            db_path.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn latentdb-cli serve");

    let up = wait_for_port(port, Duration::from_secs(10));
    let result = std::panic::catch_unwind(|| {
        assert!(up, "serve did not start listening within timeout");

        let health = http_get(port, "/healthz");
        assert!(health.contains("\"status\":\"ok\""), "health={health}");
        assert!(health.contains("\"len\":1"), "health={health}");

        let search = http_post(
            port,
            "/search",
            r#"{"embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"k":5,"nprobe":4}"#,
        );
        let search_json: serde_json::Value = serde_json::from_str(&search)
            .unwrap_or_else(|e| panic!("non-JSON /search body {search:?}: {e}"));
        let hits = search_json["hits"].as_array().unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0]["metadata"], "served-doc");

        let insert = http_post(
            port,
            "/insert",
            r#"{"embedding":[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"metadata":"served-doc-2"}"#,
        );
        let insert_json: serde_json::Value = serde_json::from_str(&insert)
            .unwrap_or_else(|e| panic!("non-JSON /insert body {insert:?}: {e}"));
        assert_eq!(insert_json["len"], 2);
    });

    let _ = child.kill();
    let _ = child.wait();
    std::fs::remove_dir_all(&dir).ok();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
