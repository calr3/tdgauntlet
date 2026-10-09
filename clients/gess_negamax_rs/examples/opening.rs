//! Every first move of the starting position scored by a fixed-depth search with a full
//! window (`search_root` narrows its window as it goes, so only its best move is exact).
//!
//!   cargo run --release -p tdg-gess-negamax --example opening -- <depth> <threads> [<weights.json>|default] [<top>]
//!
//! Each move is played and the reply searched to depth - 1, so a move's score is from the
//! mover's (black's) side, in hundredths of a stone; wins and losses are +/-1,000,000 less the
//! distance. Moves are written as tdgauntlet does (centre, then destination; column a-t, row
//! 20 at the top).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tdg_games::gess::Position;
use tdg_games::Outcome;
use tdg_gess_negamax::{Board, Searcher, Weights};

fn square(c: usize) -> String {
    format!("{}{}", (b'a' + (c % 20) as u8) as char, 20 - c / 20)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let depth: i32 = args.first().and_then(|a| a.parse().ok()).unwrap_or(5);
    let threads: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(8);
    let weights = match args.get(2).map(String::as_str) {
        None | Some("default") => Weights::default(),
        Some(path) => Weights::load(std::path::Path::new(path)).expect("weights"),
    };
    let top: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(15);

    let start = Position::start();
    let mover = start.to_move;
    let moves = start.legal_moves();
    let next = AtomicUsize::new(0);
    let scores = Mutex::new(Vec::with_capacity(moves.len()));
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut searcher = Searcher::new(weights);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= moves.len() {
                        break;
                    }
                    let child = start.apply(moves[i]);
                    let score = match child.outcome() {
                        Some(Outcome::Win { side }) if side == mover => 1_000_000,
                        Some(Outcome::Win { .. }) => -1_000_000,
                        Some(Outcome::Draw) => 0,
                        None => -searcher.search_to_depth(
                            &Board::from_position(&child),
                            depth - 1,
                            true,
                        ),
                    };
                    scores.lock().unwrap().push((score, moves[i]));
                }
            });
        }
    });
    let mut scores = scores.into_inner().unwrap();
    scores.sort_by_key(|(s, _)| -s);
    let w = weights.as_array();
    println!(
        "{} first moves, depth {depth}, weights stone {} ring {} ring_pair {} deep {} pressure {}: {:.0} s",
        moves.len(),
        w[0],
        w[1],
        w[2],
        w[3],
        w[4],
        t0.elapsed().as_secs_f64()
    );
    for (rank, (score, mv)) in scores.iter().take(top).enumerate() {
        println!(
            "{:>3}. {}-{:<4} {:+6}",
            rank + 1,
            square(mv.from as usize),
            square(mv.to as usize),
            score
        );
    }
}
