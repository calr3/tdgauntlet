//! Tune the evaluation weights by SPSA self-play.
//!
//!   cargo run --release -p tdg-alphabeta --example spsa -- <iterations> <openings> <threads> <out.json>
//!
//! If `<out.json>` already exists, tuning resumes from the weights in it, at the iteration after
//! the last one in its log (`<out>.log.jsonl`), so the step schedules carry on where they left off.
//!
//! Each iteration perturbs every tuned weight by ±c at random, plays the two perturbed engines
//! against each other over fresh openings (both colours) at a fixed node budget, and steps the
//! weights towards whichever side won. Material is not tuned: it is fixed at its default, which
//! anchors the scale everything else is measured against.

#[path = "common/arena.rs"]
mod arena;

use arena::{openings, play, Budget, Engine};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tdg_alphabeta::eval::{Weights, NAMES};
use tdg_alphabeta::search::Features;
use tdg_games::{Outcome, Side};

/// Perturbation size per weight, roughly a fifth of its plausible range.
const C: [f64; 9] = [0.0, 40.0, 3.0, 0.3, 2.0, 1.0, 2.0, 20.0, 40.0];
const NODES: u64 = 40_000;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iterations: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(150);
    let per_iter: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(16);
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(16);
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "results/spsa_weights.json".into());
    let log_path = out.replace(".json", ".log.jsonl");
    let (mut theta, start) = match std::fs::read_to_string(&log_path) {
        Ok(text) if std::path::Path::new(&out).exists() => {
            let done = text.lines().count();
            let w = Weights::load(std::path::Path::new(&out)).expect("resume weights");
            println!("resuming from {out} after {done} iterations");
            (w.to_vec(), done)
        }
        _ => (Weights::default().to_vec(), 0),
    };
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("log");
    let mut rng: u64 = 0x5_75A ^ (start as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut coin = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        if rng & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    };

    for k in start..iterations {
        let delta: [f64; 9] = std::array::from_fn(|i| if C[i] == 0.0 { 0.0 } else { coin() });
        let ck = 1.0 / (1.0 + k as f64).powf(0.101);
        let plus = Weights::from_vec(std::array::from_fn(|i| theta[i] + ck * C[i] * delta[i]));
        let minus = Weights::from_vec(std::array::from_fn(|i| theta[i] - ck * C[i] * delta[i]));
        let engine = |w| Engine {
            weights: w,
            features: Features::ALL,
            budget: Budget::Nodes(NODES),
        };
        let (a, b) = (engine(plus), engine(minus));

        // Plus's net wins over every game, both colours of each opening.
        let ops = openings(per_iter, 2, 1000 + k as u64);
        let next = AtomicUsize::new(0);
        let net = Mutex::new(0i32);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= 2 * ops.len() {
                        break;
                    }
                    let (o, leg) = (i / 2, i % 2);
                    let first = ops[o].to_move;
                    let (engines, plus_side) = if leg == 0 {
                        ([&a, &b], first)
                    } else {
                        ([&b, &a], first.other())
                    };
                    let r = match play(engines, &ops[o], 12_000) {
                        Outcome::Draw => 0,
                        Outcome::Win { side } if side == plus_side => 1,
                        Outcome::Win { .. } => -1,
                    };
                    *net.lock().unwrap() += r;
                });
            }
        });
        let net = net.into_inner().unwrap();
        let result = net as f64 / (2 * ops.len()) as f64; // in [-1, 1]

        // Step towards the winner, shrinking as the run goes on.
        let rk = 0.5 / (1.0 + k as f64 / 50.0);
        for i in 0..9 {
            theta[i] += rk * ck * C[i] * delta[i] * result;
        }
        let w = Weights::from_vec(theta);
        std::fs::write(&out, serde_json::to_string_pretty(&w.to_json()).unwrap()).unwrap();
        let line = serde_json::json!({"iteration": k, "net": net, "weights": w.to_json()});
        writeln!(log, "{line}").unwrap();
        let shown: Vec<String> = NAMES
            .iter()
            .zip(theta)
            .skip(1)
            .map(|(n, x)| format!("{n}={x:.2}"))
            .collect();
        println!("[{k:>3}] net {net:+3}  {}", shown.join(" "));
    }
    let _ = Side::First;
}
