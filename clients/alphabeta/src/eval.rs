//! The six LEONIDAS heuristics from King and Peterson, "Epaminondas: Exploring Combat Tactics",
//! ICGA Journal 37(3): material, crossing, home row defence, territory, centrality, mobility.
//!
//! A term-for-term port of `EpaminondasEvaluator` in pgx's `examples/alphazero/negamax.py`,
//! including its readings of where the paper is ambiguous and its weights, which are not the
//! paper's (the paper never published any). `tests/eval.rs` holds it to pgx's numbers, so a match
//! between this client and `negamax.py` compares search, not evaluation.
//!
//! Every term is "ours minus theirs" for the side to move.
//!
//! Three terms are this engine's own and absent from LEONIDAS, because under the v6 rules most
//! games that reach the quiet-move clock are decided on advancement: `advancement`, `tiebreak`
//! and `tiebreak_clock`. Their default weight is zero, so `Weights::default()` is exactly pgx's
//! evaluator; `Weights::load` reads tuned ones.

use tdg_games::epaminondas::{home_row, stone, Position, EMPTY, HEIGHT, MAX_QUIET_MOVES, N, WIDTH};
use tdg_games::Side;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub material: f64,
    pub crossing: f64,
    pub home_row_defense: f64,
    pub territory: f64,
    pub centrality: f64,
    pub mobility: f64,
    pub advancement: f64,
    pub tiebreak: f64,
    pub tiebreak_clock: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            material: 30.0,
            crossing: 250.0,
            home_row_defense: 8.0,
            territory: 0.5,
            centrality: 4.0,
            mobility: 2.0,
            advancement: 0.0,
            tiebreak: 0.0,
            tiebreak_clock: 0.0,
        }
    }
}

/// The weight names, in the order of [`Weights::to_vec`].
pub const NAMES: [&str; 9] = [
    "material",
    "crossing",
    "home_row_defense",
    "territory",
    "centrality",
    "mobility",
    "advancement",
    "tiebreak",
    "tiebreak_clock",
];

impl Weights {
    pub fn to_vec(&self) -> [f64; 9] {
        [
            self.material,
            self.crossing,
            self.home_row_defense,
            self.territory,
            self.centrality,
            self.mobility,
            self.advancement,
            self.tiebreak,
            self.tiebreak_clock,
        ]
    }

    pub fn from_vec(v: [f64; 9]) -> Self {
        let [material, crossing, home_row_defense, territory, centrality, mobility, advancement, tiebreak, tiebreak_clock] =
            v;
        Self {
            material,
            crossing,
            home_row_defense,
            territory,
            centrality,
            mobility,
            advancement,
            tiebreak,
            tiebreak_clock,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let v = self.to_vec();
        serde_json::Value::Object(
            NAMES
                .iter()
                .zip(v)
                .map(|(k, x)| (k.to_string(), serde_json::json!(x)))
                .collect(),
        )
    }

    /// Weights from a JSON object of name to value. Names left out keep their default.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let obj = value.as_object().ok_or("weights must be a JSON object")?;
        let mut v = Weights::default().to_vec();
        for (k, x) in obj {
            let i = NAMES
                .iter()
                .position(|n| n == k)
                .ok_or(format!("unknown weight {k:?}"))?;
            v[i] = x.as_f64().ok_or(format!("{k}: not a number"))?;
        }
        Ok(Weights::from_vec(v))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Weights::from_json(&value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Terms {
    pub material: f64,
    pub crossing: f64,
    pub home_row_defense: f64,
    pub territory: f64,
    pub centrality: f64,
    pub mobility: f64,
    pub advancement: f64,
    pub tiebreak: f64,
    pub tiebreak_clock: f64,
}

impl Terms {
    pub fn weighted(&self, w: &Weights) -> f64 {
        w.advancement * self.advancement
            + w.tiebreak * self.tiebreak
            + w.tiebreak_clock * self.tiebreak_clock
            + w.material * self.material
            + w.crossing * self.crossing
            + w.home_row_defense * self.home_row_defense
            + w.territory * self.territory
            + w.centrality * self.centrality
            + w.mobility * self.mobility
    }
}

const DIRS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// `NEXT[d][s]`: the square one step from `s` along `d`, or `NONE`.
const NONE: u8 = u8::MAX;
static NEXT: [[u8; N]; 8] = {
    let mut t = [[NONE; N]; 8];
    let mut d = 0;
    while d < 8 {
        let mut s = 0;
        while s < N {
            let r = (s / WIDTH) as i32 + DIRS[d].0;
            let c = (s % WIDTH) as i32 + DIRS[d].1;
            if r >= 0 && r < HEIGHT as i32 && c >= 0 && c < WIDTH as i32 {
                t[d][s] = (r as usize * WIDTH + c as usize) as u8;
            }
            s += 1;
        }
        d += 1;
    }
    t
};

/// Squares from `s` inclusive along `d` holding `want`.
#[inline]
fn run(board: &[u8; N], mut s: u8, d: usize, want: u8) -> u32 {
    let mut n = 0;
    while s != NONE && board[s as usize] == want {
        n += 1;
        s = NEXT[d][s as usize];
    }
    n
}

fn row_count(board: &[u8; N], row: usize, want: u8) -> f64 {
    board[row * WIDTH..(row + 1) * WIDTH]
        .iter()
        .filter(|&&c| c == want)
        .count() as f64
}

fn longest_run(board: &[u8; N], row: usize, want: u8) -> f64 {
    let (mut best, mut cur) = (0, 0);
    for &c in &board[row * WIDTH..(row + 1) * WIDTH] {
        cur = if c == want { cur + 1 } else { 0 };
        best = best.max(cur);
    }
    best as f64
}

/// Per piece, the adjacent squares that are on the board and not held by the enemy.
fn territory(board: &[u8; N], mine: u8, theirs: u8) -> f64 {
    let mut n = 0;
    for s in 0..N {
        if board[s] != mine {
            continue;
        }
        for next in NEXT.iter().map(|t| t[s]) {
            n += (next != NONE && board[next as usize] != theirs) as u32;
        }
    }
    n as f64
}

/// Mean Euclidean distance of a side's pieces from the centre of the board; 0 with none.
fn mean_distance(board: &[u8; N], mine: u8) -> f64 {
    let (cr, cc) = ((HEIGHT - 1) as f64 / 2.0, (WIDTH - 1) as f64 / 2.0);
    let (mut total, mut n) = (0.0, 0);
    for (s, _) in board.iter().enumerate().filter(|&(_, &c)| c == mine) {
        let (r, c) = ((s / WIDTH) as f64, (s % WIDTH) as f64);
        total += ((r - cr).powi(2) + (c - cc).powi(2)).sqrt();
        n += 1;
    }
    if n > 0 {
        total / n as f64
    } else {
        0.0
    }
}

/// For every (piece, direction): how far the group it leads that way could travel, which is the
/// group size capped by the empty squares ahead. The mean over all of them plus the maximum.
fn mobility(board: &[u8; N], mine: u8) -> f64 {
    let (mut total, mut count, mut best) = (0u32, 0u32, 0u32);
    for s in 0..N {
        if board[s] != mine {
            continue;
        }
        for (d, next) in NEXT.iter().enumerate() {
            // Own pieces extending back against d: the group `s` can lead along d.
            let group = run(board, s as u8, 7 - d, mine);
            let ahead = run(board, next[s], d, EMPTY);
            let travel = group.min(ahead);
            total += travel;
            count += 1;
            best = best.max(travel);
        }
    }
    let mean = if count > 0 {
        total as f64 / count as f64
    } else {
        0.0
    };
    mean + best as f64
}

pub fn terms(pos: &Position) -> Terms {
    let b = &pos.board;
    let me = pos.to_move;
    let them = me.other();
    let (mine, theirs) = (stone(me), stone(them));
    let tiebreak = tiebreak(b, me);
    Terms {
        material: pos.count(me) as f64 - pos.count(them) as f64,
        crossing: row_count(b, home_row(them), mine) - row_count(b, home_row(me), theirs),
        home_row_defense: longest_run(b, home_row(me), mine)
            - longest_run(b, home_row(them), theirs),
        territory: territory(b, mine, theirs) - territory(b, theirs, mine),
        centrality: mean_distance(b, theirs) - mean_distance(b, mine),
        mobility: mobility(b, mine) - mobility(b, theirs),
        advancement: progress(b, me) - progress(b, them),
        tiebreak,
        tiebreak_clock: tiebreak * (pos.quiet_moves.min(MAX_QUIET_MOVES) as f64)
            / MAX_QUIET_MOVES as f64,
    }
}

/// Total rows advanced from its home rank, over all of `side`'s pieces.
fn progress(board: &[u8; N], side: Side) -> f64 {
    let want = stone(side);
    let home = home_row(side);
    board
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == want)
        .map(|(s, _)| (s / WIDTH).abs_diff(home) as f64)
        .sum()
}

/// A smooth stand-in for the advancement tiebreak: rank pair `i` (i rows short of the
/// opponent's home rank, for both sides) contributes its count difference times 2^-i, so deeper
/// pairs count for more, as in the rule. It is not strictly lexicographic like the rule: a large
/// difference on a later pair can outweigh a small one earlier. Within sight of the clock the
/// search sees the real tiebreak in its terminal scores anyway.
fn tiebreak(board: &[u8; N], me: Side) -> f64 {
    let (them, mine, theirs) = (me.other(), stone(me), stone(me.other()));
    let row = |r: usize, want: u8| {
        board[r * WIDTH..(r + 1) * WIDTH]
            .iter()
            .filter(|&&c| c == want)
            .count() as f64
    };
    let (target, own_home) = (home_row(them), home_row(me));
    let toward = |from: usize, i: usize| if from == 0 { from + i } else { from - i };
    (0..HEIGHT)
        .map(|i| {
            let diff = row(toward(target, i), mine) - row(toward(own_home, i), theirs);
            diff * 0.5f64.powi(i as i32)
        })
        .sum()
}

/// The weighted evaluation from the side to move's point of view.
pub fn evaluate(pos: &Position, w: &Weights) -> f64 {
    terms(pos).weighted(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Terms, b: Terms) {
        let pairs = [
            (a.material, b.material),
            (a.crossing, b.crossing),
            (a.home_row_defense, b.home_row_defense),
            (a.territory, b.territory),
            (a.centrality, b.centrality),
            (a.mobility, b.mobility),
        ];
        assert!(
            pairs.iter().all(|(x, y)| (x - y).abs() < 1e-9),
            "{a:?}\n{b:?}"
        );
    }

    #[test]
    fn the_start_position_is_level() {
        // Zero on every term, up to the order floating-point sums happen in.
        assert_close(terms(&Position::start()), Terms::default());
    }

    #[test]
    fn evaluation_is_antisymmetric_under_a_colour_flip() {
        // Flip the board vertically and swap colours: every term should negate... except that the
        // side to move changes too, so the evaluation of the flipped position from its mover's
        // side equals the original from its mover's side.
        let p = Position::from_pieces(Side::First, &[30, 31, 60, 100], &[140, 141, 90]);
        let mut flipped = [EMPTY; N];
        for s in 0..N {
            let target = (HEIGHT - 1 - s / WIDTH) * WIDTH + s % WIDTH;
            flipped[target] = match p.board[s] {
                1 => 2,
                2 => 1,
                c => c,
            };
        }
        let q = Position {
            board: flipped,
            to_move: Side::Second,
            ..p.clone()
        };
        assert_close(terms(&p), terms(&q));
    }

    #[test]
    fn default_weights_ignore_the_new_terms() {
        let w = Weights::default();
        assert_eq!(
            (w.advancement, w.tiebreak, w.tiebreak_clock),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn weights_round_trip_through_json() {
        let mut v = Weights::default().to_vec();
        v[7] = 3.5;
        let w = Weights::from_vec(v);
        assert_eq!(Weights::from_json(&w.to_json()).unwrap(), w);
        let partial = Weights::from_json(&serde_json::json!({"tiebreak": 2.0})).unwrap();
        assert_eq!(partial.tiebreak, 2.0);
        assert_eq!(partial.material, 30.0, "names left out keep their default");
        assert!(Weights::from_json(&serde_json::json!({"typo": 1.0})).is_err());
    }

    #[test]
    fn advancement_terms_favour_the_side_further_up() {
        // White on rank 9, black on rank 7 (5 short of white's home rank): white is further up
        // by the tiebreak's measure, and has advanced more rows.
        let p = Position::from_pieces(Side::First, &[8 * WIDTH + 3], &[6 * WIDTH + 3]);
        let t = terms(&p);
        assert!(t.advancement > 0.0 && t.tiebreak > 0.0, "{t:?}");
        assert_eq!(t.tiebreak_clock, 0.0, "no quiet moves yet");
        let late = Position {
            quiet_moves: 50,
            ..p.clone()
        };
        assert!((terms(&late).tiebreak_clock - t.tiebreak / 2.0).abs() < 1e-12);
        // From black's side, everything negates.
        let q = Position {
            to_move: Side::Second,
            ..p
        };
        let u = terms(&q);
        assert_eq!((u.advancement, u.tiebreak), (-t.advancement, -t.tiebreak));
    }

    #[test]
    fn a_lone_free_piece_has_mobility_two() {
        // negamax.py's docstring: one space each way, average 1, plus a greatest distance of 1.
        let p = Position::from_pieces(Side::First, &[5 * WIDTH + 6], &[]);
        assert_eq!(mobility(&p.board, 1), 2.0);
    }
}
