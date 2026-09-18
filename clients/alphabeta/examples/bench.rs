//! Fixed-depth search over a fixed set of midgame positions, for measuring move ordering and
//! search speed: fewer nodes for the same depth means better ordering.
//!
//!   cargo run --release -p tdg-alphabeta --example bench -- [depth]

use std::time::Instant;
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search_with, Features, Limits};
use tdg_alphabeta::tt::Table;
use tdg_games::epaminondas::Position;

/// Positions from seeded random games, a spread of depths into the game.
pub fn positions() -> Vec<Position> {
    let mut out = Vec::new();
    let mut state: u64 = 0x5EED;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for game in 0..6 {
        let mut p = Position::start();
        for ply in 0..60 {
            if p.outcome().is_some() {
                break;
            }
            if ply >= 10 && ply % 12 == 10 - game % 2 {
                out.push(p.clone());
            }
            let moves = p.legal_moves();
            p = p.apply(moves[next() as usize % moves.len()]);
        }
    }
    out
}

fn run(name: &str, features: Features, depth: u32, positions: &[Position]) {
    let w = Weights::default();
    let (mut nodes, mut qnodes, mut cut, mut first) = (0u64, 0u64, 0u64, 0u64);
    let started = Instant::now();
    for p in positions {
        let tt = Table::new(64);
        let limits = Limits {
            deadline: None,
            nodes: None,
            depth,
        };
        let r = search_with(p, &tt, &w, limits, features).unwrap();
        nodes += r.nodes;
        qnodes += r.stats.quiescence_nodes;
        cut += r.stats.cutoffs;
        first += r.stats.first_move_cutoffs;
    }
    let secs = started.elapsed().as_secs_f64();
    println!(
        "{name:<18} {:>11} nodes ({:>4.1}% quiescence) {:>7.2}s {:>6.0}k nodes/s  first-move cutoffs {:>5.1}%",
        nodes,
        100.0 * qnodes as f64 / nodes as f64,
        secs,
        nodes as f64 / secs / 1000.0,
        100.0 * first as f64 / cut.max(1) as f64,
    );
}

/// A bench row: its label, and the feature it switches on.
type Step = (&'static str, fn(&mut Features));

pub fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    let ps = positions();
    println!("{} positions, depth {depth}", ps.len());
    // Each row adds one feature to the row before it.
    let mut f = Features::NONE;
    run("none", f, depth, &ps);
    let steps: [Step; 6] = [
        ("+capture order", |f| f.capture_order = true),
        ("+history", |f| f.history = true),
        ("+pvs", |f| f.pvs = true),
        ("+aspiration", |f| f.aspiration = true),
        ("+incursions", |f| f.incursions = true),
        ("+delta pruning", |f| f.delta_pruning = true),
    ];
    for (name, add) in steps {
        add(&mut f);
        run(name, f, depth, &ps);
    }
    assert_eq!(
        f,
        Features {
            delta_pruning: true,
            ..Features::ALL
        }
    );
}
