#![allow(dead_code)] // shared by several examples, each of which uses only part of it
//! Engine-against-engine matches in process, for measuring changes to the search and evaluation
//! and for tuning. Scored exactly as a tournament is: minimatches over shared openings, with
//! 1 win 1 loss excluded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search_with, Features, Limits};
use tdg_alphabeta::solve::{self, solve_with};
use tdg_alphabeta::tt::Table;
use tdg_games::epaminondas::{Epaminondas, Position};
use tdg_games::{DynGame, Game, Outcome, Side};
use tdg_tourney::config::OpeningSpec;
use tdg_tourney::score::{GameResult, Tally};

#[derive(Debug, Clone, Copy)]
pub enum Budget {
    Millis(u64),
    Nodes(u64),
    Depth(u32),
}

#[derive(Debug, Clone)]
pub struct Engine {
    pub weights: Weights,
    pub features: Features,
    pub budget: Budget,
    /// Search the rules alone rather than the evaluation, leaving `weights` and `features`
    /// unused: `tdg_alphabeta::solve`.
    pub solve: Option<solve::Features>,
}

impl Engine {
    /// The heuristic search, with these weights and refinements.
    pub fn heuristic(weights: Weights, features: Features, budget: Budget) -> Engine {
        Engine {
            weights,
            features,
            budget,
            solve: None,
        }
    }

    /// The search with no game knowledge, at the same budget.
    pub fn solver(budget: Budget) -> Engine {
        Engine {
            weights: Weights::default(),
            features: Features::NONE,
            budget,
            solve: Some(solve::Features::ALL),
        }
    }

    fn choose(&self, pos: &Position, tt: &Table) -> tdg_games::epaminondas::Move {
        let limits = match self.budget {
            Budget::Millis(ms) => Limits {
                deadline: Some(Instant::now() + Duration::from_millis(ms)),
                nodes: None,
                depth: 64,
                seed: None,
            },
            Budget::Nodes(n) => Limits {
                deadline: None,
                nodes: Some(n),
                depth: 64,
                seed: None,
            },
            Budget::Depth(d) => Limits {
                deadline: None,
                nodes: None,
                depth: d,
                seed: None,
            },
        };
        match self.solve {
            Some(f) => solve_with(pos, tt, limits, f).expect("a legal move").best,
            None => {
                search_with(pos, tt, &self.weights, limits, self.features)
                    .expect("a legal move")
                    .best
            }
        }
    }
}

/// Play one game; `engines[0]` moves first.
pub fn play(engines: [&Engine; 2], opening: &Position, max_plies: usize) -> Outcome {
    let tables = [Table::new(16), Table::new(16)];
    let mut pos = opening.clone();
    for _ in 0..max_plies {
        if let Some(o) = pos.outcome() {
            return o;
        }
        let i = pos.to_move.index() ^ opening.to_move.index();
        pos = pos.apply(engines[i].choose(&pos, &tables[i]));
    }
    Outcome::Draw
}

pub fn openings(count: usize, plies: u32, seed: u64) -> Vec<Position> {
    let spec = OpeningSpec::Random { plies, count };
    tdg_tourney::openings::generate(&Epaminondas as &dyn DynGame, &spec, seed)
        .expect("openings")
        .into_iter()
        .map(|o| Epaminondas.decode_state(&o.state).expect("own state"))
        .collect()
}

/// `a` against `b` over every opening, both colours, on `threads` threads. From `a`'s side.
pub fn run_match(a: &Engine, b: &Engine, openings: &[Position], threads: usize) -> Tally {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, [GameResult; 2])>> = Mutex::new(Vec::new());
    // Each minimatch's two games are independent; schedule them individually.
    let games: Vec<(usize, usize)> = (0..openings.len()).flat_map(|o| [(o, 0), (o, 1)]).collect();
    let per_game: Mutex<Vec<Option<GameResult>>> = Mutex::new(vec![None; games.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(o, leg)) = games.get(i) else { break };
                let first = openings[o].to_move;
                let (engines, a_side) = if leg == 0 {
                    ([a, b], first)
                } else {
                    ([b, a], first.other())
                };
                let outcome = play(engines, &openings[o], 12_000);
                per_game.lock().unwrap()[i] = Some(GameResult::for_side(outcome, a_side));
            });
        }
    });
    let per_game = per_game.into_inner().unwrap();
    let mut tally = Tally::default();
    for (o, pair) in per_game.chunks(2).enumerate() {
        let (x, y) = (pair[0].unwrap(), pair[1].unwrap());
        tally.add(x, y);
        results.lock().unwrap().push((o, [x, y]));
    }
    tally
}

pub fn describe(t: &Tally) -> String {
    let pct = t
        .percentage()
        .map_or("   -  ".into(), |p| format!("{p:5.1}%"));
    let elo = t
        .percentage()
        .filter(|p| *p > 0.0 && *p < 100.0)
        .map_or("   -".into(), |p| {
            format!("{:+4.0}", tdg_tourney::score::elo_difference(p))
        });
    format!(
        "{pct} ({elo} Elo)  counted {:>3}, excluded {:>3}  games {}-{}-{}",
        t.counted, t.excluded, t.wins, t.draws, t.losses
    )
}

#[allow(dead_code)]
pub fn side_name(s: Side) -> &'static str {
    ["white", "black"][s.index()]
}
