//! Positions, evaluation terms and game results, as JSON lines, for fitting the weights.
//!
//!   cargo run --release -p tdg-gess-negamax --example dump -- <out.jsonl> [--weights <w.json>] [--selfplay <games> <depth> <threads>] [results/*.json ...]
//!
//! Each line is one position: `{"game", "source", "ply", "quiet", "features": [...], "result"}`,
//! with the features (see `NAMES`) and the result (1 win, 0.5 draw, 0 loss) both from the side to
//! move's point of view. Games come from tdgauntlet result files (replayed with tdgauntlet's
//! rules from their openings) and, with `--selfplay`, from the default engine playing itself at a
//! fixed depth from random 2-ply openings, breaking ties at random so the games vary.

#[path = "common/arena.rs"]
mod arena;

use serde_json::{json, Value};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tdg_games::gess::{Gess, Move, Position};
use tdg_games::{Game, Outcome};
use tdg_gess_negamax::{Board, Limits, Searcher, Weights};

fn score(outcome: Outcome, side: tdg_games::Side) -> f64 {
    match outcome {
        Outcome::Draw => 0.5,
        Outcome::Win { side: s } if s == side => 1.0,
        Outcome::Win { .. } => 0.0,
    }
}

/// Non-zero pattern counts as [class, count] pairs.
fn sparse(counts: &[i32]) -> Vec<[i32; 2]> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, &n)| n != 0)
        .map(|(i, &n)| [i as i32, n])
        .collect()
}

fn lines(game: &str, source: &str, positions: &[Position], outcome: Outcome) -> Vec<String> {
    positions
        .iter()
        .enumerate()
        .map(|(ply, p)| {
            // The next position's quiet count resets iff the move from this one captured
            // (counted as a capture for the last position, whose move ends the game).
            let captured = positions.get(ply + 1).is_none_or(|n| n.quiet_moves == 0);
            json!({
                "game": game, "source": source, "ply": ply, "quiet": p.quiet_moves, "captured": captured,
                "features": Board::from_position(p).features(), "result": score(outcome, p.to_move),
                "patterns": sparse(&Board::from_position(p).pattern_counts()),
            })
            .to_string()
        })
        .collect()
}

fn from_results(path: &str) -> Vec<String> {
    let d: Value = serde_json::from_str(&std::fs::read_to_string(path).expect(path)).expect(path);
    let openings: std::collections::HashMap<&str, &Value> = d["openings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| (o["id"].as_str().unwrap(), &o["state"]))
        .collect();
    let mut out = Vec::new();
    for g in d["games"].as_array().unwrap() {
        let mut pos = Gess
            .decode_state(openings[g["opening"].as_str().unwrap()])
            .unwrap();
        let mut seen = Vec::new();
        for m in g["moves"].as_array().unwrap() {
            seen.push(pos.clone());
            let mv = Move::new(
                m["move"]["from"].as_u64().unwrap() as usize,
                m["move"]["to"].as_u64().unwrap() as usize,
            );
            pos = pos.apply(mv);
        }
        let outcome = pos.outcome().unwrap_or(Outcome::Draw);
        out.extend(lines(
            &format!("{path}:{}", g["id"].as_str().unwrap()),
            path,
            &seen,
            outcome,
        ));
    }
    out
}

fn selfplay(games: usize, depth: i32, threads: usize, weights: Weights) -> Vec<String> {
    let ops = arena::openings(games, 2, 7);
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(opening) = ops.get(i) else { break };
                let mut searcher = Searcher::new(weights).with_seed(1000 + i as u64);
                let mut pos = opening.clone();
                let mut seen = Vec::new();
                while pos.outcome().is_none() && seen.len() < 4000 {
                    seen.push(pos.clone());
                    let r = searcher
                        .best_move(&Board::from_position(&pos), Limits::depth(depth))
                        .unwrap();
                    pos = pos.apply(Move::new(r.from, r.to));
                }
                let outcome = pos.outcome().unwrap_or(Outcome::Draw);
                out.lock().unwrap().extend(lines(
                    &format!("selfplay:{i}"),
                    "selfplay",
                    &seen,
                    outcome,
                ));
            });
        }
    });
    out.into_inner().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = &args[0];
    let mut rows = Vec::new();
    let mut weights = Weights::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--weights" {
            weights = Weights::load(std::path::Path::new(&args[i + 1])).expect("weights");
            i += 2;
        } else if args[i] == "--selfplay" {
            let (n, d, t) = (
                args[i + 1].parse().unwrap(),
                args[i + 2].parse().unwrap(),
                args[i + 3].parse().unwrap(),
            );
            let r = selfplay(n, d, t, weights);
            eprintln!("self-play: {n} games at depth {d}, {} positions", r.len());
            rows.extend(r);
            i += 4;
        } else {
            let r = from_results(&args[i]);
            eprintln!("{}: {} positions", args[i], r.len());
            rows.extend(r);
            i += 1;
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    for r in &rows {
        writeln!(f, "{r}").unwrap();
    }
    eprintln!("wrote {} positions to {out_path}", rows.len());
}
