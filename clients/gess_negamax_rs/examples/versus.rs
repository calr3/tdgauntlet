//! One set of evaluation weights against another, in process.
//!
//!   cargo run --release -p tdg-gess-negamax --example versus -- <budget> <openings> <threads> <a.json|default> [<b.json|default>]
//!
//! `budget` is `d3` (depth), `n40000` (nodes) or `250` (ms; only on an otherwise idle machine).
//! A weights argument may end in `@q<n>` to add n plies of quiescence search (e.g. `v1.json@q4`).
//! Openings are 2 random plies from the start, both colours each; reported from `a`'s side.

#[path = "common/arena.rs"]
mod arena;

use arena::{describe, openings, run_match, Budget, Engine};
use std::time::Instant;
use tdg_gess_negamax::Weights;

/// Weights and quiescence plies from `path[@q<n>]` (`default` for the app's weights).
fn engine_spec(arg: Option<&String>, budget: Budget) -> Engine {
    let arg = arg.map_or("default", String::as_str);
    let (path, q) = match arg.split_once("@q") {
        Some((p, q)) => (p, q.parse().expect("quiescence plies")),
        None => (arg, 0),
    };
    let weights = match path {
        "default" => Weights::default(),
        path => Weights::load(std::path::Path::new(path)).expect("weights"),
    };
    Engine {
        weights,
        budget,
        quiescence: q,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget = Budget::parse(args.first().map_or("d3", String::as_str));
    let count: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(8);
    let a = engine_spec(args.get(3), budget);
    let b = engine_spec(args.get(4), budget);
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let ops = openings(count, 2, seed);
    let t0 = Instant::now();
    let tally = run_match(&a, &b, &ops, threads);
    println!(
        "a {:?} q{}\nb {:?} q{}\n{budget:?}, {} openings: {}  ({:.0} s)",
        a.weights,
        a.quiescence,
        b.weights,
        b.quiescence,
        ops.len(),
        describe(&tally),
        t0.elapsed().as_secs_f64()
    );
}
