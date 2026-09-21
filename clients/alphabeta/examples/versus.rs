//! Search features, or evaluation weights, against each other at a time control:
//!
//!   cargo run --release -p tdg-alphabeta --example versus -- <ms> <openings> <threads>
//!   cargo run --release -p tdg-alphabeta --example versus -- <ms> <openings> <threads> solve
//!   cargo run --release -p tdg-alphabeta --example versus -- <ms> <openings> <threads> \
//!       weights <a.json> [<b.json> ...]
//!
//! With `solve`, the heuristic-free search (`tdg_alphabeta::solve`) plays both weight sets.
//!
//! With `weights`, each file plays pgx's weights (`Weights::default()`), both with the plain
//! search.

#[path = "common/arena.rs"]
mod arena;

use arena::{describe, openings, run_match, Budget, Engine};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::Features;

/// Usage: versus <budget> <openings> <threads> [ablate | solve | weights <file>...], where
/// budget is `250` (ms), `n40000` (nodes) or `d3` (depth).
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let spec = args.first().map(String::as_str).unwrap_or("100");
    let budget = match (spec.strip_prefix('n'), spec.strip_prefix('d')) {
        (Some(n), _) => Budget::Nodes(n.parse().expect("nodes")),
        (_, Some(d)) => Budget::Depth(d.parse().expect("depth")),
        _ => Budget::Millis(spec.parse().expect("milliseconds")),
    };
    let count: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(40);
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(16);
    let ops = openings(count, 2, 7);
    let engine = |features| Engine::heuristic(Weights::default(), features, budget);
    if args.get(3).map(String::as_str) == Some("solve") {
        // The heuristic-free search against the evaluation it is the floor for. Its own
        // refinements are only speed, so there is nothing else to ablate here.
        for (name, other) in [
            ("solve v plain", engine(Features::NONE)),
            (
                "solve v recommended",
                Engine::heuristic(Weights::recommended(), Features::NONE, budget),
            ),
        ] {
            let started = std::time::Instant::now();
            let t = run_match(&Engine::solver(budget), &other, &ops, threads);
            println!(
                "{name:<24} {spec}: {}  [{:.0}s]",
                describe(&t),
                started.elapsed().as_secs_f64()
            );
        }
        return;
    }
    if args.get(3).map(String::as_str) == Some("weights") {
        let default = engine(Features::NONE);
        for path in &args[4..] {
            let challenger = Engine {
                weights: Weights::load(std::path::Path::new(path)).expect("weights"),
                ..engine(Features::NONE)
            };
            let started = std::time::Instant::now();
            let t = run_match(&challenger, &default, &ops, threads);
            println!(
                "{path} v default {spec}: {}  [{:.0}s]",
                describe(&t),
                started.elapsed().as_secs_f64()
            );
        }
        return;
    }
    let none = Features::NONE;
    let ordering = Features {
        incursions: false,
        ..Features::ALL
    };
    let comparisons: Vec<(&str, Features, Features)> = match args.get(3).map(String::as_str) {
        Some("ablate") => vec![
            ("ordering v none", ordering, none),
            ("incursions v ordering", Features::ALL, ordering),
        ],
        _ => vec![("all v none", Features::ALL, none)],
    };
    for (name, a, b) in comparisons {
        let started = std::time::Instant::now();
        let t = run_match(&engine(a), &engine(b), &ops, threads);
        println!(
            "{name:<24} {spec}: {}  [{:.0}s]",
            describe(&t),
            started.elapsed().as_secs_f64()
        );
    }
}
