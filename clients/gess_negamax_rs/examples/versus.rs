//! One set of evaluation weights against another, in process.
//!
//!   cargo run --release -p tdg-gess-negamax --example versus -- <budget> <openings> <threads> <a.json|default> [<b.json|default>]
//!
//! `budget` is `d3` (depth), `n40000` (nodes) or `250` (ms; only on an otherwise idle machine).
//! Openings are 2 random plies from the start, both colours each; reported from `a`'s side.

#[path = "common/arena.rs"]
mod arena;

use arena::{describe, openings, run_match, Budget, Engine};
use std::time::Instant;
use tdg_gess_negamax::Weights;

fn weights(arg: Option<&String>) -> Weights {
    match arg.map(String::as_str) {
        None | Some("default") => Weights::default(),
        Some(path) => Weights::load(std::path::Path::new(path)).expect("weights"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget = Budget::parse(args.first().map_or("d3", String::as_str));
    let count: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(8);
    let a = Engine {
        weights: weights(args.get(3)),
        budget,
    };
    let b = Engine {
        weights: weights(args.get(4)),
        budget,
    };
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let ops = openings(count, 2, seed);
    let t0 = Instant::now();
    let tally = run_match(&a, &b, &ops, threads);
    println!(
        "a {:?}\nb {:?}\n{budget:?}, {} openings: {}  ({:.0} s)",
        a.weights,
        b.weights,
        ops.len(),
        describe(&tally),
        t0.elapsed().as_secs_f64()
    );
}
