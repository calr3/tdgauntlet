//! `cargo run --release -p tdg-games --example perft -- 4 [game]` — move-tree node counts from
//! the start of any game this build knows. The game defaults to `epaminondas/1`.

use std::time::Instant;
use tdg_games::{epaminondas::Epaminondas, gess::Gess, Game};

fn perft<G: Game>(g: &G, pos: &G::State, depth: u32) -> u64 {
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

fn run<G: Game>(g: &G, max: u32) {
    println!("{}", g.id());
    for d in 1..=max {
        let t = Instant::now();
        let n = perft(g, &g.initial(), d);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "perft({d}) = {n}  ({secs:.2}s, {:.1}M nodes/s)",
            n as f64 / secs / 1e6
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let max: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let id = args.next().unwrap_or_else(|| Epaminondas.id().into());
    match id.as_str() {
        id if id == Epaminondas.id() => run(&Epaminondas, max),
        id if id == Gess.id() => run(&Gess, max),
        other => eprintln!("unknown game {other:?}; try {:?}", tdg_games::all_ids()),
    }
}
