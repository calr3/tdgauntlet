//! The alpha-beta client. One transposition table is shared by every request, however many games
//! they come from: it is keyed by position, so that is exactly the kind of state the protocol
//! allows a client to keep.

use clap::Parser;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search_with, Features, Limits};
use tdg_alphabeta::solve::{self, solve_with};
use tdg_alphabeta::tt::Table;
use tdg_client_kit::{serve, Answer, Engine, Info};
use tdg_games::epaminondas::{self, Epaminondas, Move, Position};
use tdg_games::Game;
use tdg_proto::{MoveInfo, MoveRequest};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9200)]
    port: u16,
    #[arg(long, default_value = "alphabeta")]
    name: String,
    /// Transposition table size.
    #[arg(long, default_value_t = 256)]
    hash_mb: usize,
    /// Think time when a request states no millisecond budget.
    #[arg(long, default_value_t = 1000)]
    default_ms: u64,
    /// Deepest iteration to search, whatever the budget.
    #[arg(long, default_value_t = 64)]
    max_depth: u32,
    /// Searches to run at once. Defaults to the number of CPUs.
    #[arg(long)]
    threads: Option<u32>,
    /// Evaluation weights: `recommended`, `pgx` for exactly pgx's evaluator, or a JSON file of
    /// weights (see `eval::NAMES`) in which missing names keep pgx's values. Ignored by
    /// `--search solve`, which has no evaluation.
    #[arg(long, default_value = "recommended")]
    weights: String,
    /// `plain` alpha-beta, `all` for every refinement in `search::Features::ALL`, or `solve` for
    /// the search with no game knowledge at all: to the depth the budget allows, and a win, a
    /// loss or nothing. Plain is the default because it has measured stronger at real time
    /// controls; see TUNING.md.
    #[arg(long, default_value = "plain")]
    search: String,
    /// Seed for breaking ties: when several moves share the best score, one of them is drawn at
    /// random. The draw depends on the seed and the position only, so under one seed a position
    /// always gets the same move; give each run its own seed to vary them. Not used by
    /// `--search solve`.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// Which search answers a request.
enum Mode {
    /// Alpha-beta over the LEONIDAS evaluation.
    Heuristic(Weights, Features),
    /// Alpha-beta over the rules alone.
    Solve(solve::Features),
}

struct AlphaBeta {
    name: String,
    tt: Table,
    mode: Mode,
    default_ms: u64,
    max_depth: u32,
    threads: u32,
    seed: u64,
}

/// The moves of a principal variation, in this game's notation.
fn pv_json(pos: &Position, pv: &[Move]) -> Vec<Value> {
    let mut p = pos.clone();
    pv.iter()
        .map(|m| {
            let san = Epaminondas.describe_move(&p, m);
            p = p.apply(*m);
            json!(san)
        })
        .collect()
}

impl Engine for AlphaBeta {
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
        // Leave a margin for the round trip, and never plan on less than a few milliseconds.
        let ms = req.budget.ms.unwrap_or(self.default_ms);
        let think = Duration::from_millis((ms * 9 / 10).saturating_sub(15).max(5));
        let limits = Limits {
            deadline: Some(started + think),
            nodes: req.budget.nodes,
            depth: self.max_depth,
            seed: Some(self.seed),
        };
        let (best, info) = match &self.mode {
            Mode::Heuristic(weights, features) => {
                let r = search_with(&pos, &self.tt, weights, limits, *features)
                    .ok_or("no legal moves")?;
                let mut info = MoveInfo {
                    eval: Some(r.score as f64 / 100.0),
                    depth: Some(r.depth),
                    nodes: Some(r.nodes),
                    pv: pv_json(&pos, &r.pv),
                    ..MoveInfo::default()
                };
                info.extra.insert("ties".into(), json!(r.ties));
                (r.best, info)
            }
            Mode::Solve(features) => {
                let r = solve_with(&pos, &self.tt, limits, *features).ok_or("no legal moves")?;
                // There is no evaluation to report, so `eval` is the verdict as 1, -1 or 0, and
                // the verdict itself goes alongside it in the result file.
                let mut info = MoveInfo {
                    eval: Some(r.verdict.eval()),
                    depth: Some(r.depth),
                    nodes: Some(r.nodes),
                    pv: pv_json(&pos, &r.pv),
                    ..MoveInfo::default()
                };
                info.extra
                    .insert("verdict".into(), json!(r.verdict.label()));
                if let Some(plies) = r.verdict.plies() {
                    info.extra.insert("verdict_plies".into(), json!(plies));
                }
                (r.best, info)
            }
        };
        Ok(Answer {
            mv: Epaminondas.encode_move(&pos, &best),
            info: Some(info),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as u32));
    let weights = || match args.weights.as_str() {
        "recommended" => Ok(Weights::recommended()),
        "pgx" => Ok(Weights::default()),
        path => Weights::load(std::path::Path::new(path)).map_err(anyhow::Error::msg),
    };
    let mode = match args.search.as_str() {
        "plain" => Mode::Heuristic(weights()?, Features::NONE),
        "all" => Mode::Heuristic(weights()?, Features::ALL),
        "solve" => Mode::Solve(solve::Features::ALL),
        other => anyhow::bail!("--search must be plain, all or solve, not {other:?}"),
    };
    let engine = AlphaBeta {
        name: args.name,
        tt: Table::new(args.hash_mb),
        mode,
        default_ms: args.default_ms,
        max_depth: args.max_depth,
        threads,
        seed: args.seed,
    };
    serve(engine, ([127, 0, 0, 1], args.port).into()).await
}
