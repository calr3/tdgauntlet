//! Alpha-beta over the rules alone: search to the prescribed depth, and call what is there a win,
//! a loss, or "don't know".
//!
//! [`crate::search`] is only ever as good as the LEONIDAS evaluation it leans on, and a match
//! between two of its settings measures that evaluation as much as it measures the search. This
//! mode has no evaluation. A position is worth `WIN - ply` when the rules have already given the
//! game to the side to move, `-(WIN - ply)` when they have given it away, and [`UNKNOWN`] — zero
//! — everywhere else, the depth limit included. Nothing in this file knows what a capture, a
//! phalanx or a home rank is: it asks the position for its legal moves, plays them, and asks
//! whether the game is over.
//!
//! What that costs: until a result is inside the horizon every move scores the same, and which
//! of them it then returns means nothing. What it buys: every non-zero score is a proof rather
//! than an opinion, so it plays out a decided game exactly, and it is the floor any evaluation
//! has to beat.
//!
//! Move ordering obeys the same rule, so the capture and advancement keys of the heuristic search
//! are out; what is left is general purpose and would work in any game: the table's move, the
//! killers of this ply, then history. There is no quiescence — it searches captures, and a capture
//! is game knowledge — and with no evaluation to stabilise there is nothing for it to do.
//!
//! Like [`crate::search`], each refinement can be switched off through [`Features`] so the
//! `solve` example can measure what it is worth.

use crate::search::{terminal_score, Limits, INF, MAX_PLY, WIN};
use crate::tt::{exact_key, Bound, Entry, Table};
use std::time::Instant;
use tdg_games::epaminondas::{Move, Position, N};

/// The score of a position the search cannot decide. Zero, so that alpha-beta sorts it between
/// a win and a loss without anything further.
pub const UNKNOWN: i32 = 0;

/// Scores at least this big are proofs. Every score this search produces is either exactly
/// [`UNKNOWN`] or `WIN` less the plies to the end of the game, so the threshold has only to tell
/// those two apart; half of `WIN` leaves room for a proof arbitrarily far from its terminal
/// position, which keeping proofs in the table across a whole game can make it.
const PROVEN: i32 = WIN / 2;

/// The most a slot can hold, and where a proof is kept: a proof is true at any depth.
const FOREVER: u8 = 63;

/// What the search made of a position, from the side to move's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The side to move wins, in this many plies against the best defence the search saw.
    Win { plies: u32 },
    /// The side to move loses, in this many plies.
    Loss { plies: u32 },
    /// Nothing was decided inside the depth searched.
    Unknown,
}

impl Verdict {
    fn of(score: i32) -> Verdict {
        if score >= PROVEN {
            Verdict::Win {
                plies: (WIN - score) as u32,
            }
        } else if score <= -PROVEN {
            Verdict::Loss {
                plies: (WIN + score) as u32,
            }
        } else {
            Verdict::Unknown
        }
    }

    /// Whether the rules settled it, so that searching deeper cannot change it.
    pub fn is_proof(self) -> bool {
        self != Verdict::Unknown
    }

    /// How far off the end of the game is, when it is known.
    pub fn plies(self) -> Option<u32> {
        match self {
            Verdict::Win { plies } | Verdict::Loss { plies } => Some(plies),
            Verdict::Unknown => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Verdict::Win { .. } => "win",
            Verdict::Loss { .. } => "loss",
            Verdict::Unknown => "unknown",
        }
    }

    /// 1, -1 or 0: the verdict as the one number the protocol has room for.
    pub fn eval(self) -> f64 {
        match self {
            Verdict::Win { .. } => 1.0,
            Verdict::Loss { .. } => -1.0,
            Verdict::Unknown => 0.0,
        }
    }
}

/// The refinements over plain alpha-beta, all of them game-agnostic, individually switchable.
/// The table's move always leads, since the table is not optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    /// Try the moves that last caused a cutoff at this ply before the rest.
    pub killers: bool,
    /// Order what is left by how often it has caused a cutoff anywhere in the tree.
    pub history: bool,
    /// Principal variation search: prove later moves no better with a null window first.
    pub pvs: bool,
    /// Keep a proven win or loss in the table at its greatest depth. A proof does not depend on
    /// how much depth was left when it was found, so a later, deeper search can have it for free;
    /// this is what stops each iteration from re-proving what the last one settled.
    pub proof_promotion: bool,
}

impl Features {
    pub const ALL: Features = Features {
        killers: true,
        history: true,
        pvs: true,
        proof_promotion: true,
    };
    pub const NONE: Features = Features {
        killers: false,
        history: false,
        pvs: false,
        proof_promotion: false,
    };
}

impl Default for Features {
    fn default() -> Self {
        Features::ALL
    }
}

/// How the search spent itself. `first_move_cutoffs / cutoffs` is the usual measure of ordering.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub cutoffs: u64,
    pub first_move_cutoffs: u64,
    /// Nodes answered out of the table without searching a move.
    pub tt_cutoffs: u64,
    /// Nodes the rules decided: the only places a score comes from.
    pub terminal_nodes: u64,
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub best: Move,
    pub verdict: Verdict,
    /// The deepest iteration that completed.
    pub depth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub stats: Stats,
}

struct Solver<'a> {
    tt: &'a Table,
    features: Features,
    limits: Limits,
    nodes: u64,
    stopped: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Cutoff credit by (lead, destination): from-square to to-square, no game knowledge in it.
    history: Vec<i32>,
    stats: Stats,
}

/// Scores count plies from the root, entries count them from their own node, so an entry found at
/// a different ply than it was stored at still says how far off the end of the game is.
fn to_tt(score: i32, ply: usize) -> i32 {
    match score {
        s if s >= PROVEN => s + ply as i32,
        s if s <= -PROVEN => s - ply as i32,
        s => s,
    }
}

fn from_tt(score: i32, ply: usize) -> i32 {
    match score {
        s if s >= PROVEN => s - ply as i32,
        s if s <= -PROVEN => s + ply as i32,
        s => s,
    }
}

/// Whether an entry settles its position whatever depth is asked of it later. A bound only counts
/// when it points away from "don't know": a value at least a win is a win, and a value at most a
/// loss is a loss, however deep the search that asks.
fn is_proof(score: i32, bound: Bound) -> bool {
    match bound {
        Bound::Exact => score.abs() >= PROVEN,
        Bound::Lower => score >= PROVEN,
        Bound::Upper => score <= -PROVEN,
    }
}

impl Solver<'_> {
    /// Count a node, and every 1024th one check the budget.
    fn visit(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            let late = self.limits.deadline.is_some_and(|d| Instant::now() >= d);
            let big = self.limits.nodes.is_some_and(|n| self.nodes >= n);
            self.stopped |= late || big;
        }
        self.stopped
    }

    fn order(&self, moves: &mut [Move], tt_move: Option<Move>, ply: usize) {
        let killers = self.killers[ply.min(MAX_PLY - 1)];
        let f = self.features;
        // A stable sort, so moves nothing distinguishes — most of them, most of the time — keep
        // the order the rules generated them in, and the same position always answers the same.
        moves.sort_by_cached_key(|&m| {
            let score: i64 = if Some(m) == tt_move {
                1 << 40
            } else if f.killers && killers.contains(&Some(m)) {
                1 << 32
            } else if f.history {
                self.history[m.lead as usize * N + m.dest as usize] as i64
            } else {
                0
            };
            -score
        });
    }

    fn negamax(
        &mut self,
        pos: &Position,
        depth: u32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
    ) -> i32 {
        if self.visit() {
            return UNKNOWN;
        }
        // The whole evaluation: the rules, or nothing.
        if let Some(o) = pos.outcome() {
            self.stats.terminal_nodes += 1;
            return terminal_score(o, pos, ply);
        }
        if depth == 0 || ply >= MAX_PLY - 1 {
            return UNKNOWN;
        }

        let k = exact_key(pos);
        let entry = self.tt.probe(k);
        if let Some(e) = entry.filter(|e| e.depth as u32 >= depth && ply > 0) {
            let score = from_tt(e.score, ply);
            match e.bound {
                Bound::Exact => {
                    self.stats.tt_cutoffs += 1;
                    return score;
                }
                Bound::Lower => alpha = alpha.max(score),
                Bound::Upper => beta = beta.min(score),
            }
            if alpha >= beta {
                self.stats.tt_cutoffs += 1;
                return score;
            }
        }

        let original_alpha = alpha;
        let mut moves = pos.legal_moves();
        self.order(&mut moves, entry.and_then(|e| e.mv), ply);
        let (mut best, mut best_move) = (-INF, None);
        for (i, &m) in moves.iter().enumerate() {
            let child = pos.apply(m);
            let score = if i == 0 || !self.features.pvs {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
            } else {
                let probe = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, ply + 1);
                if probe > alpha && probe < beta && !self.stopped {
                    -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
                } else {
                    probe
                }
            };
            if self.stopped {
                return UNKNOWN;
            }
            if score > best {
                best = score;
                best_move = Some(m);
                alpha = alpha.max(score);
                if alpha >= beta {
                    self.stats.cutoffs += 1;
                    self.stats.first_move_cutoffs += (i == 0) as u64;
                    // Every move is credited, not just the quiet ones: telling a capture from a
                    // quiet move is knowing what a capture is.
                    let killers = &mut self.killers[ply];
                    if killers[0] != Some(m) {
                        killers[1] = killers[0];
                        killers[0] = Some(m);
                    }
                    let h = &mut self.history[m.lead as usize * N + m.dest as usize];
                    *h = (*h + (depth * depth) as i32).min(1 << 20);
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
        let keep = self.features.proof_promotion && is_proof(best, bound);
        self.tt.store(
            k,
            Entry {
                score: to_tt(best, ply),
                depth: if keep {
                    FOREVER
                } else {
                    depth.min(FOREVER as u32) as u8
                },
                bound,
                mv: best_move,
            },
        );
        best
    }

    /// One iteration over the root moves. `None` if the budget ran out first.
    fn root(&mut self, pos: &Position, root: &[Move], depth: u32) -> Option<(Move, i32)> {
        let (mut alpha, beta) = (-INF, INF);
        let mut best: Option<(Move, i32)> = None;
        for (i, &m) in root.iter().enumerate() {
            let child = pos.apply(m);
            let score = if i == 0 || !self.features.pvs {
                -self.negamax(&child, depth - 1, -beta, -alpha, 1)
            } else {
                let probe = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, 1);
                if probe > alpha && !self.stopped {
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
            }
        }
        best
    }
}

/// The principal variation, read back out of the table.
fn principal_variation(tt: &Table, pos: &Position, first: Move, depth: u32) -> Vec<Move> {
    let mut out = vec![first];
    let mut p = pos.apply(first);
    let mut seen = std::collections::HashSet::new();
    while out.len() < depth as usize && p.outcome().is_none() && seen.insert(exact_key(&p)) {
        let Some(m) = tt.probe(exact_key(&p)).and_then(|e| e.mv) else {
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

/// Search `pos` to the prescribed depth, or until the limits run out, with no game knowledge at
/// all. Returns `None` only if there is no legal move.
pub fn solve(pos: &Position, tt: &Table, limits: Limits) -> Option<SolveResult> {
    solve_with(pos, tt, limits, Features::default())
}

pub fn solve_with(
    pos: &Position,
    tt: &Table,
    limits: Limits,
    features: Features,
) -> Option<SolveResult> {
    let mut root = pos.legal_moves();
    let first = *root.first()?;
    if root.len() == 1 || pos.outcome().is_some() {
        return Some(SolveResult {
            best: first,
            verdict: Verdict::Unknown,
            depth: 0,
            nodes: 0,
            pv: vec![first],
            stats: Stats::default(),
        });
    }
    let mut s = Solver {
        tt,
        features,
        limits,
        nodes: 0,
        stopped: false,
        killers: [[None; 2]; MAX_PLY],
        history: vec![0; N * N],
        stats: Stats::default(),
    };
    s.order(&mut root, tt.probe(exact_key(pos)).and_then(|e| e.mv), 0);
    let mut result = SolveResult {
        best: root[0],
        verdict: Verdict::Unknown,
        depth: 0,
        nodes: 0,
        pv: vec![root[0]],
        stats: Stats::default(),
    };

    // Iterative deepening, for the ordering the shallower iterations leave behind: the last ply
    // is most of the tree, so the ones before it are nearly free, and an interrupted iteration
    // still leaves the one below it to answer with.
    for depth in 1..=limits.depth.max(1) {
        let Some((m, score)) = s.root(pos, &root, depth) else {
            break;
        };
        s.tt.store(
            exact_key(pos),
            Entry {
                score: to_tt(score, 0),
                depth: if features.proof_promotion && is_proof(score, Bound::Exact) {
                    FOREVER
                } else {
                    depth.min(FOREVER as u32) as u8
                },
                bound: Bound::Exact,
                mv: Some(m),
            },
        );
        result = SolveResult {
            best: m,
            verdict: Verdict::of(score),
            depth,
            nodes: s.nodes,
            pv: principal_variation(tt, pos, m, depth),
            stats: s.stats,
        };
        // Search the best move first next time.
        if let Some(i) = root.iter().position(|&r| r == m) {
            root[..=i].rotate_right(1);
        }
        if result.verdict.is_proof() {
            break; // the rules have decided it; depth cannot.
        }
    }
    result.nodes = s.nodes;
    result.stats = s.stats;
    Some(result)
}
