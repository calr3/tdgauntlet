//! Iterative-deepening negamax with alpha-beta pruning, a transposition table, and a quiescence
//! search over captures and incursions.
//!
//! Scores are integers: the evaluation times 100, so the table can hold them compactly, and wins
//! near `WIN` with the ply subtracted so a quicker win is preferred.
//!
//! Each refinement over plain alpha-beta can be switched off through [`Features`], so the bench
//! (`cargo run --release -p tdg-alphabeta --example bench`) can measure what each one is worth.

use crate::eval::{evaluate, Weights};
use crate::tt::{key, Bound, Entry, Table};
use std::time::Instant;
use tdg_games::epaminondas::{home_row, Move, Position, N, WIDTH};
use tdg_games::{Outcome, Side};

pub const WIN: i32 = 1_000_000_000;
const INF: i32 = WIN + 1;
const MAX_PLY: usize = 128;
const QUIESCENCE_DEPTH: u32 = 6;
/// Slack for delta pruning, in search units: what a capture may be worth beyond its material.
const DELTA_MARGIN: i32 = 2_000;
/// Half-width of the first aspiration window, in search units (evaluation times 100).
const ASPIRATION: i32 = 2_500;

/// The refinements over plain alpha-beta, individually switchable for measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    /// Try captures in order of how many pieces they take.
    pub capture_order: bool,
    /// Order quiet moves by how often they have caused cutoffs elsewhere in the tree.
    pub history: bool,
    /// Principal variation search: prove later moves worse with a null window first.
    pub pvs: bool,
    /// Start each iteration with a narrow window around the last score.
    pub aspiration: bool,
    /// In quiescence, never stand pat while the opponent has an unanswered incursion, and look at
    /// incursions as well as captures on its first ply.
    pub incursions: bool,
    /// In quiescence, skip captures that could not raise the score to alpha even with a margin.
    ///
    /// Off by default: it cost about 150 Elo in self-play. It bounds a capture's gain by its
    /// material, but under these weights a piece on a home rank is worth far more than its
    /// material (crossing is 250 against material's 30), so it prunes exactly the captures that
    /// remove an incursion.
    pub delta_pruning: bool,
}

impl Features {
    /// Everything that helps: all but delta pruning.
    pub const ALL: Features = Features {
        capture_order: true,
        history: true,
        pvs: true,
        aspiration: true,
        incursions: true,
        delta_pruning: false,
    };
    pub const NONE: Features = Features {
        capture_order: false,
        history: false,
        pvs: false,
        aspiration: false,
        incursions: false,
        delta_pruning: false,
    };
}

impl Default for Features {
    fn default() -> Self {
        Features::ALL
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub deadline: Option<Instant>,
    pub nodes: Option<u64>,
    pub depth: u32,
}

/// How well the search is pruning. `first_move_cutoffs / cutoffs` is the usual measure of move
/// ordering: near 1 means the refuting move is almost always tried first.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub quiescence_nodes: u64,
    pub cutoffs: u64,
    pub first_move_cutoffs: u64,
    pub researches: u64,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: Move,
    /// From the side to move's point of view, in evaluation units times 100.
    pub score: i32,
    /// The deepest iteration that completed.
    pub depth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub stats: Stats,
}

struct Search<'a> {
    tt: &'a Table,
    weights: &'a Weights,
    features: Features,
    limits: Limits,
    nodes: u64,
    stopped: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Cutoff credit for quiet moves, by (lead, destination).
    history: Vec<i32>,
    stats: Stats,
}

/// How far the lead ends up towards the opponent: a cheap proxy for an aggressive move.
fn advance(pos: &Position, m: Move) -> i32 {
    let row = (m.dest as usize / WIDTH) as i32;
    match pos.to_move {
        Side::First => row,
        Side::Second => 11 - row,
    }
}

fn is_incursion(pos: &Position, m: Move) -> bool {
    m.dest as usize / WIDTH == home_row(pos.to_move.other())
}

fn terminal_score(o: Outcome, pos: &Position, ply: usize) -> i32 {
    match o {
        Outcome::Draw => 0,
        Outcome::Win { side } if side == pos.to_move => WIN - ply as i32,
        Outcome::Win { .. } => -(WIN - ply as i32),
    }
}

impl Search<'_> {
    /// Count a node, and every 1024th one check the budget. Every node in both searches comes
    /// through here, so no check is ever skipped.
    fn visit(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            let late = self.limits.deadline.is_some_and(|d| Instant::now() >= d);
            let big = self.limits.nodes.is_some_and(|n| self.nodes >= n);
            self.stopped |= late || big;
        }
        self.stopped
    }

    fn order(&self, pos: &Position, moves: &mut [Move], tt_move: Option<Move>, ply: usize) {
        let killers = self.killers[ply.min(MAX_PLY - 1)];
        let f = self.features;
        moves.sort_by_cached_key(|&m| {
            let captured = pos.captured_count(m) as i64;
            let score: i64 = if Some(m) == tt_move {
                1 << 40
            } else if captured > 0 {
                (1 << 36) + if f.capture_order { captured } else { 0 }
            } else if killers.contains(&Some(m)) {
                1 << 32
            } else if f.history {
                // History dominates; advancement breaks ties among moves with none.
                ((self.history[m.lead as usize * N + m.dest as usize] as i64) << 4)
                    + advance(pos, m) as i64
            } else {
                advance(pos, m) as i64
            };
            -score
        });
    }

    fn quiesce(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        qdepth: u32,
    ) -> i32 {
        if self.visit() {
            return 0;
        }
        self.stats.quiescence_nodes += 1;
        if let Some(o) = pos.outcome() {
            return terminal_score(o, pos, ply);
        }
        let me = pos.to_move;
        // The opponent has more pieces on our home rank than we have on theirs: unless this
        // move deals with it, they win at the start of their turn. Standing pat would be a lie.
        let threatened = self.features.incursions && pos.crossed(me.other()) > pos.crossed(me);
        let stand = (evaluate(pos, self.weights) * 100.0).round() as i32;
        if qdepth == 0 {
            return stand;
        }
        let mut best = -INF;
        let mut moves = if threatened {
            pos.legal_moves()
        } else {
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
            best = stand;
            let mut noisy = if self.features.incursions && qdepth == QUIESCENCE_DEPTH {
                pos.legal_moves()
                    .into_iter()
                    .filter(|&m| pos.captured_count(m) > 0 || is_incursion(pos, m))
                    .collect()
            } else {
                pos.legal_captures()
            };
            if self.features.delta_pruning {
                // A capture gains at most its pieces' material; if even that, plus a margin for
                // everything else it might change, leaves us below alpha, it cannot matter.
                let per_piece = (self.weights.material * 100.0) as i32;
                noisy.retain(|&m| {
                    is_incursion(pos, m)
                        || stand + per_piece * (pos.captured_count(m) as i32 + 1) + DELTA_MARGIN
                            > alpha
                });
            }
            noisy
        };
        self.order(pos, &mut moves, None, ply);
        for m in moves {
            let score = -self.quiesce(&pos.apply(m), -beta, -alpha, ply + 1, qdepth - 1);
            if self.stopped {
                return best.max(stand);
            }
            if score > best {
                best = score;
                alpha = alpha.max(score);
                if alpha >= beta {
                    break;
                }
            }
        }
        best
    }

    fn negamax(
        &mut self,
        pos: &Position,
        depth: u32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
    ) -> i32 {
        if depth == 0 || ply >= MAX_PLY - 1 {
            return self.quiesce(pos, alpha, beta, ply, QUIESCENCE_DEPTH);
        }
        if self.visit() {
            return 0;
        }
        if let Some(o) = pos.outcome() {
            return terminal_score(o, pos, ply);
        }

        let k = key(pos);
        let entry = self.tt.probe(k);
        if let Some(e) = entry.filter(|e| e.depth as u32 >= depth && ply > 0) {
            match e.bound {
                Bound::Exact => return e.score,
                Bound::Lower => alpha = alpha.max(e.score),
                Bound::Upper => beta = beta.min(e.score),
            }
            if alpha >= beta {
                return e.score;
            }
        }

        let original_alpha = alpha;
        let mut moves = pos.legal_moves();
        self.order(pos, &mut moves, entry.and_then(|e| e.mv), ply);
        let (mut best, mut best_move) = (-INF, None);
        for (i, &m) in moves.iter().enumerate() {
            let child = pos.apply(m);
            let score = if i == 0 || !self.features.pvs {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
            } else {
                let probe = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, ply + 1);
                if probe > alpha && probe < beta && !self.stopped {
                    self.stats.researches += 1;
                    -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
                } else {
                    probe
                }
            };
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                best_move = Some(m);
                alpha = alpha.max(score);
                if alpha >= beta {
                    self.stats.cutoffs += 1;
                    self.stats.first_move_cutoffs += (i == 0) as u64;
                    if pos.captured_count(m) == 0 {
                        let k = &mut self.killers[ply];
                        if k[0] != Some(m) {
                            k[1] = k[0];
                            k[0] = Some(m);
                        }
                        let h = &mut self.history[m.lead as usize * N + m.dest as usize];
                        *h = (*h + (depth * depth) as i32).min(1 << 20);
                    }
                    break;
                }
            }
        }
        let bound = if best <= original_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt.store(
            k,
            Entry {
                score: best,
                depth: depth.min(63) as u8,
                bound,
                mv: best_move,
            },
        );
        best
    }

    /// One iteration over the root moves within (alpha, beta). `None` if the budget ran out.
    fn root(
        &mut self,
        pos: &Position,
        root: &[Move],
        depth: u32,
        mut alpha: i32,
        beta: i32,
    ) -> Option<(Move, i32)> {
        let mut best: Option<(Move, i32)> = None;
        for (i, &m) in root.iter().enumerate() {
            let child = pos.apply(m);
            let score = if i == 0 || !self.features.pvs {
                -self.negamax(&child, depth - 1, -beta, -alpha, 1)
            } else {
                let probe = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, 1);
                if probe > alpha && probe < beta && !self.stopped {
                    self.stats.researches += 1;
                    -self.negamax(&child, depth - 1, -beta, -alpha, 1)
                } else {
                    probe
                }
            };
            if self.stopped {
                return None;
            }
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((m, score));
                alpha = alpha.max(score);
                if alpha >= beta {
                    break;
                }
            }
        }
        best
    }

    /// The principal variation, read back out of the table.
    fn pv(&self, pos: &Position, first: Move, depth: u32) -> Vec<Move> {
        let mut out = vec![first];
        let mut p = pos.apply(first);
        let mut seen = std::collections::HashSet::new();
        while out.len() < depth as usize && p.outcome().is_none() && seen.insert(key(&p)) {
            let Some(m) = self.tt.probe(key(&p)).and_then(|e| e.mv) else {
                break;
            };
            if !p.legal_moves().contains(&m) {
                break;
            }
            out.push(m);
            p = p.apply(m);
        }
        out
    }
}

/// Search `pos` until the limits run out; the result is from the deepest complete iteration.
/// Returns `None` only if there is no legal move.
pub fn search(
    pos: &Position,
    tt: &Table,
    weights: &Weights,
    limits: Limits,
) -> Option<SearchResult> {
    search_with(pos, tt, weights, limits, Features::default())
}

pub fn search_with(
    pos: &Position,
    tt: &Table,
    weights: &Weights,
    limits: Limits,
    features: Features,
) -> Option<SearchResult> {
    let mut root = pos.legal_moves();
    let first = *root.first()?;
    if root.len() == 1 || pos.outcome().is_some() {
        return Some(SearchResult {
            best: first,
            score: 0,
            depth: 0,
            nodes: 0,
            pv: vec![first],
            stats: Stats::default(),
        });
    }
    let mut s = Search {
        tt,
        weights,
        features,
        limits,
        nodes: 0,
        stopped: false,
        killers: [[None; 2]; MAX_PLY],
        history: vec![0; N * N],
        stats: Stats::default(),
    };
    s.order(pos, &mut root, tt.probe(key(pos)).and_then(|e| e.mv), 0);
    let mut result = SearchResult {
        best: root[0],
        score: 0,
        depth: 0,
        nodes: 0,
        pv: vec![root[0]],
        stats: Stats::default(),
    };

    for depth in 1..=limits.depth.max(1) {
        let found = if features.aspiration && depth > 1 && result.score.abs() < WIN / 2 {
            // Search a narrow window around the last score, widening on failure.
            let mut width = ASPIRATION;
            loop {
                let (lo, hi) = (result.score - width, result.score + width);
                match s.root(pos, &root, depth, lo, hi) {
                    None => break None,
                    Some((m, score)) if score > lo && score < hi => break Some((m, score)),
                    Some(_) if width >= WIN => break s.root(pos, &root, depth, -INF, INF),
                    Some(_) => width = width.saturating_mul(4),
                }
                s.stats.researches += 1;
            }
        } else {
            s.root(pos, &root, depth, -INF, INF)
        };
        let Some((m, score)) = found else { break };
        s.tt.store(
            key(pos),
            Entry {
                score,
                depth: depth.min(63) as u8,
                bound: Bound::Exact,
                mv: Some(m),
            },
        );
        result = SearchResult {
            best: m,
            score,
            depth,
            nodes: s.nodes,
            pv: s.pv(pos, m, depth),
            stats: s.stats,
        };
        // Search the best move first next time.
        if let Some(i) = root.iter().position(|&r| r == m) {
            root[..=i].rotate_right(1);
        }
        if score.abs() >= WIN - MAX_PLY as i32 {
            break; // a forced result; deeper search cannot change it
        }
    }
    result.nodes = s.nodes;
    result.stats = s.stats;
    Some(result)
}
