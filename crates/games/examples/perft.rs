//! `cargo run --release -p tdg-games --example perft -- 4` — move-tree node counts from the start.

use std::time::Instant;
use tdg_games::epaminondas::{Epaminondas, Position};
use tdg_games::Game;

fn perft(g: &Epaminondas, pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = g.legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|m| perft(g, &g.apply(pos, m), depth - 1))
        .sum()
}

fn main() {
    let max: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);
    for d in 1..=max {
        let t = Instant::now();
        let n = perft(&Epaminondas, &Position::start(), d);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "perft({d}) = {n}  ({secs:.2}s, {:.1}M nodes/s)",
            n as f64 / secs / 1e6
        );
    }
}
