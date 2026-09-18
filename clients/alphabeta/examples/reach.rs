//! How deep each feature set gets at a time control, over the bench positions.
//!
//!   cargo run --release -p tdg-alphabeta --example reach -- <ms>

use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search_with, Features, Limits};
use tdg_alphabeta::tt::Table;

#[path = "bench.rs"]
#[allow(dead_code)]
mod bench;

fn main() {
    let ms: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);
    let ps = bench::positions();
    let ordering = Features {
        incursions: false,
        delta_pruning: false,
        ..Features::ALL
    };
    for (name, f) in [
        ("none", Features::NONE),
        (
            "capture order+history",
            Features {
                capture_order: true,
                history: true,
                ..Features::NONE
            },
        ),
        (
            "+pvs",
            Features {
                capture_order: true,
                history: true,
                pvs: true,
                ..Features::NONE
            },
        ),
        ("ordering (+aspiration)", ordering),
        ("all", Features::ALL),
    ] {
        let (mut depth, mut nodes, mut hist) = (0u32, 0u64, [0u32; 8]);
        for p in &ps {
            let tt = Table::new(64);
            let limits = Limits {
                deadline: Some(Instant::now() + Duration::from_millis(ms)),
                nodes: None,
                depth: 64,
            };
            let r = search_with(p, &tt, &Weights::default(), limits, f).unwrap();
            depth += r.depth;
            nodes += r.nodes;
            hist[r.depth.min(7) as usize] += 1;
        }
        println!(
            "{name:<24} mean depth {:.2}  depth histogram {:?}  {:.0}k nodes/search",
            depth as f64 / ps.len() as f64,
            &hist[..6],
            nodes as f64 / ps.len() as f64 / 1000.0
        );
    }
}
