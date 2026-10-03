//! Native vs WebAssembly speed: the app's `bench_fixed` (fixed-depth iterative deepening, TT on)
//! over the identity fixture's positions, timed here; `bench_wasm.mjs` times the app's .wasm on
//! the same positions. The node counts must match (same tree), so the time ratio is pure speed.
//!
//!   cargo run --release -p tdg-gess-negamax --example bench -- <depth>

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::Read;
use std::time::Instant;
use tdg_gess_negamax::{Board, Searcher, Weights};

fn main() {
    let depth: i32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/identity.json.gz");
    let mut text = String::new();
    GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    let recs: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut s = Searcher::new(Weights::default());
    let (mut nodes, t0) = (0u64, Instant::now());
    let mut per: Vec<u64> = Vec::new();
    for rec in &recs {
        let st = &rec["state"];
        let (mut w, mut b) = ([0u32; 20], [0u32; 20]);
        for (i, ch) in st["board"].as_str().unwrap().chars().enumerate() {
            match ch {
                'W' => w[i / 20] |= 1 << (i % 20),
                'B' => b[i / 20] |= 1 << (i % 20),
                _ => {}
            }
        }
        let board = Board {
            w,
            b,
            white_to_move: st["to_move"].as_i64() == Some(1),
            quiet: st["quiet_moves"].as_i64().unwrap() as i32,
        };
        let n = s.bench_fixed(&board, depth);
        per.push(n);
        nodes += n;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "native: depth {depth}, {} positions, {nodes} nodes in {secs:.2} s = {:.0} knodes/s; node counts {}",
        recs.len(),
        nodes as f64 / secs / 1000.0,
        serde_json::to_string(&per).unwrap()
    );
}
