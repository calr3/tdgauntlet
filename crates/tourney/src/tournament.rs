//! Scheduling minimatches and assembling the result.

use crate::engine::{play_game, GameRecord, GameSpec, Seat, Termination};
use crate::openings::Opening;
use crate::score::{elo_difference, minimatch_score, GameResult, Tally};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tdg_games::DynGame;
use tdg_proto::{Identify, Side};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

pub struct Entrant {
    pub seat: Arc<Seat>,
    pub identify: Identify,
    pub source: String,
}

pub struct Plan {
    pub game: Arc<dyn DynGame>,
    pub entrants: Vec<Entrant>,
    pub openings: Vec<Opening>,
    pub pairings: Vec<(usize, usize)>,
    /// Games in flight at once.
    pub concurrency: usize,
    pub max_plies: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntrantInfo {
    pub name: String,
    pub source: String,
    pub identify: Identify,
    pub budget: tdg_proto::Budget,
    pub on_error: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimatchRecord {
    pub match_id: String,
    pub opening: String,
    pub a: String,
    pub b: String,
    /// `a`'s results in the game where `a` moved first, then the one where `b` did.
    pub a_results: [GameResult; 2],
    /// `a`'s score, or `None` for a 1 win 1 loss minimatch, which does not count.
    pub a_score: Option<f64>,
    pub games: [String; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairingSummary {
    pub a: String,
    pub b: String,
    /// From `a`'s side.
    pub tally: Tally,
    pub a_percentage: Option<f64>,
    pub a_elo: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Standing {
    pub name: String,
    pub tally: Tally,
    pub percentage: Option<f64>,
    pub exclusion_rate: Option<f64>,
    pub incidents: u32,
    pub forfeits: u32,
    pub server_moves: u32,
    pub moves: u32,
    pub mean_think_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TournamentResult {
    pub tdgauntlet: String,
    pub game: String,
    pub seed: u64,
    pub entrants: Vec<EntrantInfo>,
    pub openings: Vec<Opening>,
    pub standings: Vec<Standing>,
    pub pairings: Vec<PairingSummary>,
    pub minimatches: Vec<MinimatchRecord>,
    pub games: Vec<GameRecord>,
}

impl TournamentResult {
    /// The result with everything that depends on wall-clock time zeroed, so two runs of the
    /// same tournament can be compared exactly.
    pub fn without_timings(mut self) -> Self {
        for g in &mut self.games {
            for m in &mut g.moves {
                m.elapsed_ms = 0;
                if let Some(info) = &mut m.info {
                    info.elapsed_ms = None;
                }
            }
        }
        for s in &mut self.standings {
            s.mean_think_ms = None;
        }
        self
    }
}

/// How often to report progress, and to whom.
pub type Progress = Box<dyn Fn(&GameRecord, usize, usize) + Send + Sync>;

pub async fn run(plan: Plan, progress: Option<Progress>) -> TournamentResult {
    let openings: Vec<Arc<Opening>> = plan.openings.iter().cloned().map(Arc::new).collect();
    let slots = Arc::new(Semaphore::new(plan.concurrency));

    // Every game gets a sort key, so the result is in the same order however they finish.
    let mut tasks = JoinSet::new();
    let mut total = 0;
    for (p, &(a, b)) in plan.pairings.iter().enumerate() {
        for (o, opening) in openings.iter().enumerate() {
            let match_id = format!(
                "{}-v-{}-{}",
                plan.entrants[a].seat.name, plan.entrants[b].seat.name, opening.id
            );
            for leg in 0..2 {
                let (first, second) = if leg == 0 { (a, b) } else { (b, a) };
                let spec = GameSpec {
                    id: format!("{match_id}-{}", ["a", "b"][leg]),
                    match_id: match_id.clone(),
                    opening: opening.clone(),
                    seats: [
                        plan.entrants[first].seat.clone(),
                        plan.entrants[second].seat.clone(),
                    ],
                    max_plies: plan.max_plies,
                    seed: plan.seed,
                };
                let game = plan.game.clone();
                let slots = slots.clone();
                tasks.spawn(async move {
                    let _slot = slots.acquire_owned().await.expect("never closed");
                    ((p, o, leg), play_game(game, spec).await)
                });
                total += 1;
            }
        }
    }

    let mut done = Vec::with_capacity(total);
    while let Some(joined) = tasks.join_next().await {
        let (key, record) = joined.expect("a game task panicked");
        if let Some(report) = &progress {
            report(&record, done.len() + 1, total);
        }
        done.push((key, record));
    }
    done.sort_by_key(|(key, _)| *key);
    let games: Vec<GameRecord> = done.into_iter().map(|(_, g)| g).collect();

    assemble(&plan, games)
}

fn assemble(plan: &Plan, games: Vec<GameRecord>) -> TournamentResult {
    let names: Vec<String> = plan.entrants.iter().map(|e| e.seat.name.clone()).collect();

    let mut minimatches = Vec::new();
    let mut pair_tallies: BTreeMap<(usize, usize), Tally> = BTreeMap::new();
    let mut player_tallies = vec![Tally::default(); names.len()];
    // Games arrive as consecutive (a first, b first) legs of each minimatch.
    for legs in games.chunks_exact(2) {
        let (g0, g1) = (&legs[0], &legs[1]);
        let a = names
            .iter()
            .position(|n| *n == g0.players[0])
            .expect("known player");
        let b = names
            .iter()
            .position(|n| *n == g0.players[1])
            .expect("known player");
        let a_results = [
            GameResult::for_side(g0.outcome, Side::First),
            GameResult::for_side(g1.outcome, Side::Second),
        ];
        pair_tallies
            .entry((a, b))
            .or_default()
            .add(a_results[0], a_results[1]);
        player_tallies[a].add(a_results[0], a_results[1]);
        player_tallies[b].add(a_results[0].reversed(), a_results[1].reversed());
        minimatches.push(MinimatchRecord {
            match_id: g0.match_id.clone(),
            opening: g0.opening.clone(),
            a: names[a].clone(),
            b: names[b].clone(),
            a_results,
            a_score: minimatch_score(a_results[0], a_results[1]),
            games: [g0.id.clone(), g1.id.clone()],
        });
    }

    let pairings = plan
        .pairings
        .iter()
        .map(|&(a, b)| {
            let tally = pair_tallies.get(&(a, b)).copied().unwrap_or_default();
            PairingSummary {
                a: names[a].clone(),
                b: names[b].clone(),
                tally,
                a_percentage: tally.percentage(),
                // None for a clean sweep, which has no finite Elo.
                a_elo: tally
                    .percentage()
                    .filter(|p| *p > 0.0 && *p < 100.0)
                    .map(elo_difference),
            }
        })
        .collect();

    let mut standings: Vec<Standing> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let mut s = Standing {
                name: name.clone(),
                tally: player_tallies[i],
                percentage: player_tallies[i].percentage(),
                exclusion_rate: player_tallies[i].exclusion_rate(),
                incidents: 0,
                forfeits: 0,
                server_moves: 0,
                moves: 0,
                mean_think_ms: None,
            };
            let mut think = 0u64;
            for g in &games {
                s.incidents += g.incidents.iter().filter(|x| x.player == *name).count() as u32;
                if let Termination::Forfeit { side } = g.termination {
                    s.forfeits += (g.players[side.index()] == *name) as u32;
                }
                for m in g
                    .moves
                    .iter()
                    .filter(|m| g.players[m.side.index()] == *name)
                {
                    s.moves += 1;
                    s.server_moves += m.by_server as u32;
                    think += m.elapsed_ms;
                }
            }
            s.mean_think_ms = (s.moves > 0).then(|| think as f64 / s.moves as f64);
            s
        })
        .collect();
    // Best first; players with nothing counted go last, in entry order.
    standings.sort_by(|x, y| {
        let key = |s: &Standing| s.percentage.unwrap_or(f64::NEG_INFINITY);
        key(y).total_cmp(&key(x))
    });

    TournamentResult {
        tdgauntlet: env!("CARGO_PKG_VERSION").to_string(),
        game: plan.game.id().to_string(),
        seed: plan.seed,
        entrants: plan
            .entrants
            .iter()
            .map(|e| EntrantInfo {
                name: e.seat.name.clone(),
                source: e.source.clone(),
                identify: e.identify.clone(),
                budget: e.seat.budget,
                on_error: e.seat.on_error.to_string(),
            })
            .collect(),
        openings: plan.openings.clone(),
        standings,
        pairings,
        minimatches,
        games,
    }
}
