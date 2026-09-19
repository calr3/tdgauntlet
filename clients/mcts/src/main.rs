//! The MCTS client with UCT bounds.

use clap::Parser;
use serde_json::json;
use std::time::{Duration, Instant};
use tdg_client_kit::{serve, Answer, Engine, Info};
use tdg_games::epaminondas::{self, Epaminondas};
use tdg_games::Game;
use tdg_mcts::search::{search, Limits, MctsConfig};
use tdg_proto::{MoveInfo, MoveRequest};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9300)]
    port: u16,
    #[arg(long, default_value = "mcts")]
    name: String,
    /// Think time when a request states no budget.
    #[arg(long, default_value_t = 1000)]
    default_ms: u64,
    /// Exploration constant for UCT.
    #[arg(long, default_value_t = std::f64::consts::SQRT_2)]
    c_uct: f64,
    /// Maximum simulations per search, if set.
    #[arg(long)]
    max_sims: Option<u64>,
    /// Searches to run at once. Defaults to number of CPUs.
    #[arg(long)]
    threads: Option<u32>,
}

struct MctsEngine {
    name: String,
    c_uct: f64,
    default_ms: u64,
    max_sims: Option<u64>,
    threads: u32,
}

impl Engine for MctsEngine {
    fn info(&self) -> Info {
        Info {
            name: self.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            games: vec![epaminondas::ID.into()],
            max_concurrency: self.threads,
        }
    }

    fn choose(&self, req: &MoveRequest) -> Result<Answer, String> {
        let started = Instant::now();
        let pos = Epaminondas
            .decode_state(&req.state)
            .map_err(|e| e.to_string())?;

        let ms = req.budget.ms.unwrap_or(self.default_ms);
        let think = Duration::from_millis((ms * 9 / 10).saturating_sub(15).max(5));

        let sims_budget = req.budget.sims.or(req.budget.nodes).or(self.max_sims);

        let limits = Limits {
            deadline: Some(started + think),
            sims: sims_budget,
        };
        let config = MctsConfig { c_uct: self.c_uct };

        let r = search(&pos, limits, config).ok_or("no legal moves")?;

        let mut p = pos.clone();
        let pv =
            r.pv.iter()
                .map(|m| {
                    let san = Epaminondas.describe_move(&p, m);
                    p = p.apply(*m);
                    json!(san)
                })
                .collect();

        Ok(Answer {
            mv: Epaminondas.encode_move(&pos, &r.best),
            info: Some(MoveInfo {
                eval: Some(r.win_rate),
                depth: Some(r.depth),
                nodes: Some(r.sims),
                pv,
                ..MoveInfo::default()
            }),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as u32));

    let engine = MctsEngine {
        name: args.name,
        c_uct: args.c_uct,
        default_ms: args.default_ms,
        max_sims: args.max_sims,
        threads,
    };

    serve(engine, ([127, 0, 0, 1], args.port).into()).await
}
