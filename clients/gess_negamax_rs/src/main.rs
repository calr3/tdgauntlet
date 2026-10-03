//! The Gess negamax client: the web app's search, natively, with chosen evaluation weights.
//! Each concurrent request gets its own searcher (transposition table and scratch space); the
//! table is invalidated per search, as in the app, so a searcher carries nothing between moves.

use clap::Parser;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tdg_client_kit::{serve, Answer, Engine, Info};
use tdg_games::gess::{self, Gess, Move};
use tdg_games::Game;
use tdg_gess_negamax::{Board, Limits, Searcher, Weights};
use tdg_proto::{MoveInfo, MoveRequest};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9500)]
    port: u16,
    #[arg(long, default_value = "gess-negamax")]
    name: String,
    /// Think time when a request states no budget.
    #[arg(long, default_value_t = 1000)]
    default_ms: u64,
    /// Search to exactly this depth, whatever the request's budget (for matches on a shared
    /// machine, where time budgets are unreliable).
    #[arg(long)]
    depth: Option<i32>,
    /// Searches to run at once.
    #[arg(long, default_value_t = 4)]
    threads: u32,
    /// A JSON file of evaluation weights (see `Weights`); missing names keep the app's values.
    #[arg(long)]
    weights: Option<std::path::PathBuf>,
    /// Seed for breaking ties between equally good moves at random, as the app does. Without it
    /// the first such move is played.
    #[arg(long)]
    seed: Option<u64>,
}

struct Negamax {
    name: String,
    default_ms: u64,
    depth: Option<i32>,
    threads: u32,
    pool: Mutex<Vec<Searcher>>,
}

impl Engine for Negamax {
    fn info(&self) -> Info {
        Info {
            name: self.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            games: vec![gess::ID.into()],
            max_concurrency: self.threads,
        }
    }

    fn choose(&self, req: &MoveRequest) -> Result<Answer, String> {
        let started = Instant::now();
        let pos = Gess.decode_state(&req.state).map_err(|e| e.to_string())?;
        let limits = match self.depth {
            Some(d) => Limits::depth(d),
            None => {
                // Leave a margin for the round trip, as the alpha-beta client does.
                let ms = req.budget.ms.unwrap_or(self.default_ms);
                let think = Duration::from_millis((ms * 9 / 10).saturating_sub(15).max(5));
                Limits {
                    depth: 64,
                    deadline: Some(started + think),
                    nodes: req.budget.nodes,
                }
            }
        };
        let mut searcher = self
            .pool
            .lock()
            .unwrap()
            .pop()
            .ok_or("more requests than --threads")?;
        let result = searcher.best_move(&Board::from_position(&pos), limits);
        self.pool.lock().unwrap().push(searcher);
        let r = result.ok_or("no legal moves")?;
        let mv = Move::new(r.from, r.to);
        let info = MoveInfo {
            eval: Some(r.score as f64 / 100.0),
            depth: Some(r.depth as u32),
            nodes: Some(r.nodes),
            ..MoveInfo::default()
        };
        Ok(Answer {
            mv: Gess.encode_move(&pos, &mv),
            info: Some(info),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let weights = match &args.weights {
        Some(path) => Weights::load(path).map_err(anyhow::Error::msg)?,
        None => Weights::default(),
    };
    let pool = (0..args.threads)
        .map(|i| {
            let s = Searcher::new(weights);
            match args.seed {
                Some(seed) => s.with_seed(seed.wrapping_add(i as u64)),
                None => s,
            }
        })
        .collect();
    let engine = Negamax {
        name: args.name,
        default_ms: args.default_ms,
        depth: args.depth,
        threads: args.threads,
        pool: Mutex::new(pool),
    };
    serve(engine, ([127, 0, 0, 1], args.port).into()).await
}
