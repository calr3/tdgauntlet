//! Monte Carlo Tree Search (MCTS) with Upper Confidence Bound applied to Trees (UCT).

use rand::seq::IndexedRandom;
use std::time::Instant;
use tdg_games::epaminondas::{Epaminondas, Move, Position};
use tdg_games::{Game, Outcome};

/// Limits on MCTS search budget.
#[derive(Debug, Clone, Copy, Default)]
pub struct Limits {
    pub deadline: Option<Instant>,
    pub sims: Option<u64>,
}

/// Configuration parameters for MCTS search.
#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    /// Exploration constant C in UCT formula: Q + C * sqrt(ln(N_parent) / N_child).
    pub c_uct: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_uct: std::f64::consts::SQRT_2,
        }
    }
}

/// Result of an MCTS search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: Move,
    /// Expected win rate for the side to move at root in [0.0, 1.0].
    pub win_rate: f64,
    /// Total MCTS simulations executed.
    pub sims: u64,
    /// Maximum tree depth reached.
    pub depth: u32,
    /// Principal variation (sequence of most visited moves).
    pub pv: Vec<Move>,
}

#[derive(Debug)]
struct Node {
    state: Position,
    visits: u64,
    total_reward: f64,
    unvisited_moves: Vec<Move>,
    children: Vec<(Move, usize)>,
    outcome: Option<Outcome>,
    depth: u32,
}

/// Perform MCTS with UCT bounds on position `pos`.
pub fn search(pos: &Position, limits: Limits, config: MctsConfig) -> Option<SearchResult> {
    let initial_moves = Epaminondas.legal_moves(pos);
    if initial_moves.is_empty() {
        return None;
    }

    let mut nodes: Vec<Node> = Vec::with_capacity(10_000);
    nodes.push(Node {
        state: pos.clone(),
        visits: 0,
        total_reward: 0.0,
        unvisited_moves: initial_moves,
        children: Vec::new(),
        outcome: Epaminondas.outcome(pos),
        depth: 1,
    });

    let mut total_sims = 0u64;
    let mut rng = rand::rng();

    loop {
        if let Some(s) = limits.sims {
            if total_sims >= s {
                break;
            }
        }
        // Every simulation, not every 128th: one simulation is a descent plus a full random
        // playout, so reading the clock costs nothing beside it, and a stride of 128 meant
        // the overshoot was 128 playouts - over half a second in an unoptimised build.
        if total_sims > 0 {
            if let Some(dl) = limits.deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
        }

        run_iteration(&mut nodes, config.c_uct, &mut rng);
        total_sims += 1;
    }

    let root = &nodes[0];
    if root.children.is_empty() {
        run_iteration(&mut nodes, config.c_uct, &mut rng);
        total_sims += 1;
    }

    let root = &nodes[0];
    if root.children.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut max_visits = u64::MIN;
    let mut max_win_rate = f64::NEG_INFINITY;

    for (i, &(_mv, child_idx)) in root.children.iter().enumerate() {
        let child = &nodes[child_idx];
        let win_rate = if child.visits > 0 {
            1.0 - (child.total_reward / child.visits as f64)
        } else {
            0.5
        };

        if child.visits > max_visits || (child.visits == max_visits && win_rate > max_win_rate) {
            max_visits = child.visits;
            max_win_rate = win_rate;
            best_idx = i;
        }
    }

    let (best_move, best_child_node_idx) = root.children[best_idx];
    let best_child = &nodes[best_child_node_idx];
    let win_rate = if best_child.visits > 0 {
        1.0 - (best_child.total_reward / best_child.visits as f64)
    } else {
        0.5
    };

    let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(1);

    let mut pv = Vec::new();
    let mut curr = 0;
    while !nodes[curr].children.is_empty() {
        let mut most_visited_child = None;
        let mut child_max_visits = 0u64;

        for &(mv, child_idx) in &nodes[curr].children {
            let child = &nodes[child_idx];
            if child.visits > child_max_visits {
                child_max_visits = child.visits;
                most_visited_child = Some((mv, child_idx));
            }
        }

        if let Some((mv, next_curr)) = most_visited_child {
            pv.push(mv);
            curr = next_curr;
        } else {
            break;
        }
    }

    Some(SearchResult {
        best: best_move,
        win_rate,
        sims: total_sims,
        depth: max_depth,
        pv,
    })
}

fn run_iteration(nodes: &mut Vec<Node>, c_uct: f64, rng: &mut impl rand::Rng) {
    let mut path = Vec::with_capacity(64);
    let mut curr = 0;
    path.push(curr);

    loop {
        if nodes[curr].outcome.is_some() {
            break;
        }

        if !nodes[curr].unvisited_moves.is_empty() {
            let mv = nodes[curr].unvisited_moves.pop().unwrap();
            let next_state = Epaminondas.apply(&nodes[curr].state, &mv);
            let outcome = Epaminondas.outcome(&next_state);
            let next_moves = if outcome.is_none() {
                Epaminondas.legal_moves(&next_state)
            } else {
                Vec::new()
            };
            let child_depth = nodes[curr].depth + 1;
            let child_idx = nodes.len();
            nodes.push(Node {
                state: next_state,
                visits: 0,
                total_reward: 0.0,
                unvisited_moves: next_moves,
                children: Vec::new(),
                outcome,
                depth: child_depth,
            });
            nodes[curr].children.push((mv, child_idx));
            curr = child_idx;
            path.push(curr);
            break;
        }

        if nodes[curr].children.is_empty() {
            break;
        }

        let parent_visits = nodes[curr].visits;
        let ln_n = (parent_visits as f64).ln();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_child_idx = nodes[curr].children[0].1;

        for &(_mv, child_idx) in &nodes[curr].children {
            let child = &nodes[child_idx];
            let score = if child.visits == 0 {
                f64::INFINITY
            } else {
                let q = 1.0 - (child.total_reward / child.visits as f64);
                let u = c_uct * (ln_n / child.visits as f64).sqrt();
                q + u
            };

            if score > best_score {
                best_score = score;
                best_child_idx = child_idx;
            }
        }

        curr = best_child_idx;
        path.push(curr);
    }

    let outcome = if let Some(o) = nodes[curr].outcome {
        o
    } else {
        simulate(&nodes[curr].state, rng)
    };

    for &node_idx in &path {
        let n = &mut nodes[node_idx];
        n.visits += 1;
        n.total_reward += outcome.score_for(n.state.to_move);
    }
}

fn simulate(pos: &Position, rng: &mut impl rand::Rng) -> Outcome {
    let mut state = pos.clone();
    loop {
        if let Some(outcome) = Epaminondas.outcome(&state) {
            return outcome;
        }
        let moves = Epaminondas.legal_moves(&state);
        if moves.is_empty() {
            return Outcome::Draw;
        }
        let mv = moves.choose(rng).unwrap();
        state = Epaminondas.apply(&state, mv);
    }
}
