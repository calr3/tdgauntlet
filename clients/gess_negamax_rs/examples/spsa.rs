//! Tune the evaluation weights by SPSA self-play, as `tdg-alphabeta`'s `spsa` does for
//! Epaminondas.
//!
//!   cargo run --release -p tdg-gess-negamax --example spsa -- <iterations> <openings> <threads> <start.json> <out.json> [<budget>]
//!
//! Each iteration perturbs every tuned weight by ±c at random, plays the two perturbed engines
//! against each other over fresh 2-ply openings (both colours) at a fixed budget (default depth
//! 3), and steps the weights towards whichever side won. The stone weight is not tuned: it is
//! the unit everything else is measured in. Progress goes to `<out>.log.jsonl`.

#[path = "common/arena.rs"]
mod arena;

use arena::{openings, play, Budget, Engine, Seats};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tdg_games::Outcome;
use tdg_gess_negamax::{Weights, NAMES, NUM_FEATURES};

/// Perturbation size per weight (stone fixed; candidate terms untouched).
const C: [f64; NUM_FEATURES] = {
    let mut c = [0.0; NUM_FEATURES];
    c[1] = 40.0; // ring
    c[2] = 30.0; // ring_pair
    c[3] = 2.0; // deep
    c[4] = 15.0; // pressure
    c
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iterations: usize = args[0].parse().unwrap();
    let per_iter: usize = args[1].parse().unwrap();
    let threads: usize = args[2].parse().unwrap();
    let start = Weights::load(std::path::Path::new(&args[3])).expect("start weights");
    let out = args[4].clone();
    let budget = Budget::parse(args.get(5).map_or("d3", String::as_str));
    let mut log = std::fs::File::create(out.replace(".json", ".log.jsonl")).expect("log");
    let mut theta: [f64; NUM_FEATURES] = start.as_array().map(|x| x as f64);
    let mut rng: u64 = 0x5_75A;
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
    let round = |t: [f64; NUM_FEATURES]| Weights::from_array(t.map(|x| x.round() as i32));

    for k in 0..iterations {
        let delta: [f64; NUM_FEATURES] =
            std::array::from_fn(|i| if C[i] == 0.0 { 0.0 } else { coin() });
        let ck = 1.0 / (1.0 + k as f64).powf(0.101);
        let a = Engine {
            weights: round(std::array::from_fn(|i| theta[i] + ck * C[i] * delta[i])),
            budget,
            quiescence: 0,
        };
        let b = Engine {
            weights: round(std::array::from_fn(|i| theta[i] - ck * C[i] * delta[i])),
            budget,
            quiescence: 0,
        };
        let ops = openings(per_iter, 2, 1000 + k as u64);
        let next = AtomicUsize::new(0);
        let net = Mutex::new(0i32);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut seats = Seats::new();
                    loop {
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
                        let r = match play(&mut seats, engines, &ops[o], 4000).0 {
                            Outcome::Draw => 0,
                            Outcome::Win { side } if side == plus_side => 1,
                            Outcome::Win { .. } => -1,
                        };
                        *net.lock().unwrap() += r;
                    }
                });
            }
        });
        let net = net.into_inner().unwrap();
        let result = net as f64 / (2 * ops.len()) as f64;
        let rk = 0.5 / (1.0 + k as f64 / 50.0);
        for i in 0..NUM_FEATURES {
            theta[i] += rk * ck * C[i] * delta[i] * result;
        }
        let w = round(theta);
        std::fs::write(&out, serde_json::to_string_pretty(&w).unwrap()).unwrap();
        writeln!(
            log,
            "{}",
            serde_json::json!({"iteration": k, "net": net, "theta": theta})
        )
        .unwrap();
        let shown: Vec<String> = NAMES
            .iter()
            .zip(theta)
            .filter(|(n, _)| C[NAMES.iter().position(|m| m == *n).unwrap()] > 0.0)
            .map(|(n, x)| format!("{n}={x:.1}"))
            .collect();
        println!("[{k:>3}] net {net:+3}  {}", shown.join(" "));
    }
}
