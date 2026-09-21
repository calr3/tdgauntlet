//! What the rules alone can see: how often the heuristic-free search reaches a verdict, and what
//! its ordering is worth.
//!
//!   cargo run --release -p tdg-alphabeta --example solve -- [depth]
//!
//! Two sets of positions. The midgame ones are the bench's, where nothing is decided within any
//! depth this search can reach; the endgame ones are the last plies of random games, where it can
//! reach one. The point of the first table is that "don't know" is the honest answer almost
//! everywhere, and of the second that the verdicts arrive exactly when the depth reaches them.

use std::time::Instant;
use tdg_alphabeta::search::Limits;
use tdg_alphabeta::solve::{solve_with, Features};
use tdg_alphabeta::tt::Table;
use tdg_games::epaminondas::Position;

#[path = "bench.rs"]
#[allow(dead_code)]
mod bench;

/// The last few positions of seeded random games: close enough to the end that a shallow search
/// can prove something, which almost nowhere else is.
fn endgames(per_game: usize) -> Vec<Position> {
    let mut out = Vec::new();
    let mut state: u64 = 0xE4D5;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..8 {
        let mut p = Position::start();
        let mut tail: Vec<Position> = Vec::new();
        for _ in 0..12_000 {
            if p.outcome().is_some() {
                break;
            }
            tail.push(p.clone());
            let moves = p.legal_moves();
            p = p.apply(moves[next() as usize % moves.len()]);
        }
        let from = tail.len().saturating_sub(per_game);
        out.extend_from_slice(&tail[from..]);
    }
    out
}

/// Search every position to `depth` and report what came back.
fn run(label: &str, positions: &[Position], depth: u32, features: Features) {
    let (mut decided, mut nodes, mut cut, mut first) = (0usize, 0u64, 0u64, 0u64);
    let started = Instant::now();
    for p in positions {
        let tt = Table::new(64);
        let limits = Limits {
            deadline: None,
            nodes: None,
            depth,
        };
        let r = solve_with(p, &tt, limits, features).unwrap();
        decided += r.verdict.is_proof() as usize;
        nodes += r.nodes;
        cut += r.stats.cutoffs;
        first += r.stats.first_move_cutoffs;
    }
    let secs = started.elapsed().as_secs_f64();
    println!(
        "{label:<22} decided {:>3}/{:<3} {:>12} nodes {:>7.2}s {:>6.0}k nodes/s  first-move cutoffs {:>5.1}%",
        decided,
        positions.len(),
        nodes,
        secs,
        nodes as f64 / secs / 1000.0,
        100.0 * first as f64 / cut.max(1) as f64,
    );
}

/// A row of the ordering table: its label, and the refinement it switches on.
type Step = (&'static str, fn(&mut Features));

fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    let midgame = bench::positions();
    let endgame = endgames(6);
    println!(
        "{} midgame positions, {} endgame positions",
        midgame.len(),
        endgame.len()
    );
    for d in 1..=depth {
        run(&format!("midgame depth {d}"), &midgame, d, Features::ALL);
    }
    for d in 1..=depth {
        run(&format!("endgame depth {d}"), &endgame, d, Features::ALL);
    }
    // Each row adds one refinement to the row before it. All of them are speed only: the verdict
    // column must not move.
    println!("\nordering, at depth {depth} over the endgame positions");
    let mut f = Features::NONE;
    run("none", &endgame, depth, f);
    let steps: [Step; 4] = [
        ("+killers", |f| f.killers = true),
        ("+history", |f| f.history = true),
        ("+pvs", |f| f.pvs = true),
        ("+proof promotion", |f| f.proof_promotion = true),
    ];
    for (name, add) in steps {
        add(&mut f);
        run(name, &endgame, depth, f);
    }
    assert_eq!(f, Features::ALL);
}
