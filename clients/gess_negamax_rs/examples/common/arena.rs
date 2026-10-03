#![allow(dead_code)] // shared by several examples, each of which uses only part of it
//! Engine-against-engine Gess matches in process, for measuring evaluation changes and tuning.
//! Scored as tournaments are: minimatches over shared openings, 1 win 1 loss excluded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tdg_games::gess::{Gess, Move, Position};
use tdg_games::{DynGame, Game, Outcome, Side};
use tdg_gess_negamax::{Board, Limits, Searcher, Weights};
use tdg_tourney::config::OpeningSpec;
use tdg_tourney::score::{GameResult, Tally};

#[derive(Debug, Clone, Copy)]
pub enum Budget {
    Millis(u64),
    Nodes(u64),
    Depth(i32),
}

impl Budget {
    /// `250` (ms), `n40000` (nodes) or `d3` (depth).
    pub fn parse(s: &str) -> Budget {
        if let Some(n) = s.strip_prefix('n') {
            Budget::Nodes(n.parse().expect("node budget"))
        } else if let Some(d) = s.strip_prefix('d') {
            Budget::Depth(d.parse().expect("depth"))
        } else {
            Budget::Millis(s.parse().expect("milliseconds"))
        }
    }

    pub fn limits(&self) -> Limits {
        match *self {
            Budget::Millis(ms) => Limits {
                depth: 64,
                deadline: Some(Instant::now() + Duration::from_millis(ms)),
                nodes: None,
            },
            Budget::Nodes(n) => Limits {
                depth: 64,
                deadline: None,
                nodes: Some(n),
            },
            Budget::Depth(d) => Limits::depth(d),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Engine {
    pub weights: Weights,
    pub budget: Budget,
}

/// Searchers for one thread, reused across its games (a searcher carries nothing between moves).
pub struct Seats {
    searchers: [Searcher; 2],
}

impl Seats {
    pub fn new() -> Self {
        Seats {
            searchers: [
                Searcher::new(Weights::default()),
                Searcher::new(Weights::default()),
            ],
        }
    }
}

/// The move `engine` plays in `pos`.
pub fn choose(searcher: &mut Searcher, engine: &Engine, pos: &Position) -> Move {
    searcher.weights = engine.weights;
    let r = searcher
        .best_move(&Board::from_position(pos), engine.budget.limits())
        .expect("a legal move");
    Move::new(r.from, r.to)
}

/// Play one game; `engines[0]` moves first. Returns the outcome and the positions played through.
pub fn play(
    seats: &mut Seats,
    engines: [&Engine; 2],
    opening: &Position,
    max_plies: usize,
) -> (Outcome, Vec<Position>) {
    let mut pos = opening.clone();
    let mut seen = Vec::new();
    for _ in 0..max_plies {
        if let Some(o) = pos.outcome() {
            return (o, seen);
        }
        seen.push(pos.clone());
        let i = pos.to_move.index() ^ opening.to_move.index();
        let mv = choose(&mut seats.searchers[i], engines[i], &pos);
        pos = pos.apply(mv);
    }
    (Outcome::Draw, seen)
}

pub fn openings(count: usize, plies: u32, seed: u64) -> Vec<Position> {
    let spec = OpeningSpec::Random { plies, count };
    tdg_tourney::openings::generate(&Gess as &dyn DynGame, &spec, seed)
        .expect("openings")
        .into_iter()
        .map(|o| Gess.decode_state(&o.state).expect("own state"))
        .collect()
}

/// `a` against `b` over every opening, both colours, on `threads` threads. From `a`'s side.
pub fn run_match(a: &Engine, b: &Engine, openings: &[Position], threads: usize) -> Tally {
    let next = AtomicUsize::new(0);
    let games: Vec<(usize, usize)> = (0..openings.len()).flat_map(|o| [(o, 0), (o, 1)]).collect();
    let per_game: Mutex<Vec<Option<GameResult>>> = Mutex::new(vec![None; games.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut seats = Seats::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(o, leg)) = games.get(i) else { break };
                    let first = openings[o].to_move;
                    let (engines, a_side) = if leg == 0 {
                        ([a, b], first)
                    } else {
                        ([b, a], first.other())
                    };
                    let (outcome, _) = play(&mut seats, engines, &openings[o], 4000);
                    per_game.lock().unwrap()[i] = Some(GameResult::for_side(outcome, a_side));
                }
            });
        }
    });
    let per_game = per_game.into_inner().unwrap();
    let mut tally = Tally::default();
    for pair in per_game.chunks(2) {
        tally.add(pair[0].unwrap(), pair[1].unwrap());
    }
    tally
}

/// Per-game score of `a` (wins + half draws over games) and its 95% interval half-width.
pub fn per_game(t: &Tally) -> (f64, f64) {
    let n = (t.wins + t.draws + t.losses) as f64;
    let s = (t.wins as f64 + 0.5 * t.draws as f64) / n;
    let var = (t.wins as f64 * (1.0 - s).powi(2)
        + t.draws as f64 * (0.5 - s).powi(2)
        + t.losses as f64 * s * s)
        / n;
    (s, 1.96 * (var / n).sqrt())
}

pub fn describe(t: &Tally) -> String {
    let (s, ci) = per_game(t);
    let elo = if s > 0.0 && s < 1.0 {
        format!("{:+.0}", -400.0 * (1.0 / s - 1.0).log10())
    } else {
        "-".into()
    };
    format!("per game {s:.3} +/- {ci:.3} ({elo} Elo)  games {}-{}-{}  minimatches counted {}, excluded {}", t.wins, t.draws, t.losses, t.counted, t.excluded)
}

#[allow(dead_code)]
pub fn side_name(s: Side) -> &'static str {
    ["black", "white"][s.index()]
}
