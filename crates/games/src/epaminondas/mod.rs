//! Epaminondas (Robert Abbott, 1975) on a 14 x 12 board.
//!
//! A scalar port of pgx's `pgx/_src/games/epaminondas.py` (env version v6), whose header comment
//! is the specification. The two implementations are held together by the conformance suite in
//! `tests/conformance.rs`, which replays positions dumped from pgx; any rule change there must
//! regenerate those vectors.
//!
//! Squares are row-major, `idx = row * WIDTH + col`. Row 0 is white's home rank.

mod notation;
#[cfg(test)]
mod tests;

pub use notation::{parse_move, parse_square, square_label};

use crate::{DecodeError, Game, Outcome, Side};
use serde_json::{json, Value};

pub const ID: &str = "epaminondas/1";

pub const WIDTH: usize = 14;
pub const HEIGHT: usize = 12;
pub const N: usize = WIDTH * HEIGHT;

/// The game ends once this many consecutive moves pass without a capture, and is then decided on
/// advancement. Every capture puts the count back to zero, so there is no absolute bound: at most
/// 55 captures can happen, so the worst case is about 56 * 100 moves.
pub const MAX_QUIET_MOVES: u32 = 100;

pub const EMPTY: u8 = 0;
pub const WHITE: u8 = 1;
pub const BLACK: u8 = 2;

/// The eight directions as (row, col) steps, in pgx's order. `DIRS[7 - d]` is the opposite of d.
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

const fn opposite(d: usize) -> usize {
    7 - d
}

/// The square `k` steps from `s` along direction `d`, if it is on the board.
#[inline]
fn step(s: usize, d: usize, k: usize) -> Option<usize> {
    let (dr, dc) = DIRS[d];
    let r = (s / WIDTH) as i32 + dr * k as i32;
    let c = (s % WIDTH) as i32 + dc * k as i32;
    if (0..HEIGHT as i32).contains(&r) && (0..WIDTH as i32).contains(&c) {
        Some(r as usize * WIDTH + c as usize)
    } else {
        None
    }
}

/// The direction from `from` towards `to`, which must share a row, column or diagonal, and the
/// number of steps between them.
fn direction(from: usize, to: usize) -> Option<(usize, usize)> {
    let (fr, fc) = ((from / WIDTH) as i32, (from % WIDTH) as i32);
    let (tr, tc) = ((to / WIDTH) as i32, (to % WIDTH) as i32);
    let (dr, dc) = (tr - fr, tc - fc);
    if (dr, dc) == (0, 0) || !(dr == 0 || dc == 0 || dr.abs() == dc.abs()) {
        return None;
    }
    let d = DIRS
        .iter()
        .position(|&(r, c)| (r, c) == (dr.signum(), dc.signum()))?;
    Some((d, dr.abs().max(dc.abs()) as usize))
}

/// Consecutive squares from `s` (inclusive) along `d` holding `want`.
#[inline]
fn run(board: &[u8; N], s: usize, d: usize, want: u8) -> usize {
    let mut k = 0;
    while let Some(t) = step(s, d, k) {
        if board[t] != want {
            break;
        }
        k += 1;
    }
    k
}

pub fn stone(side: Side) -> u8 {
    match side {
        Side::First => WHITE,
        Side::Second => BLACK,
    }
}

/// The rank a side starts on; the one its opponent is trying to reach.
pub fn home_row(side: Side) -> usize {
    match side {
        Side::First => 0,
        Side::Second => HEIGHT - 1,
    }
}

/// A move: the phalanx from `rear` to `lead` moves so the lead lands on `dest`. `rear == lead`
/// is a single piece. This is pgx's three-action encoding, one action per field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub lead: u8,
    pub rear: u8,
    pub dest: u8,
}

impl Move {
    pub fn new(lead: usize, rear: usize, dest: usize) -> Self {
        Self {
            lead: lead as u8,
            rear: rear as u8,
            dest: dest as u8,
        }
    }

    pub fn is_single(self) -> bool {
        self.lead == self.rear
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    pub board: [u8; N],
    pub to_move: Side,
    /// Completed moves, for reporting only.
    pub moves: u32,
    /// Moves since the last capture; the game ends at [`MAX_QUIET_MOVES`].
    pub quiet_moves: u32,
    /// Who captured most recently: the tiebreak for an exact advancement mirror.
    pub last_capturer: Option<Side>,
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

impl Position {
    /// Each side fills the two rows nearest it: 28 pieces each.
    pub fn start() -> Self {
        let mut board = [EMPTY; N];
        board[..2 * WIDTH].fill(WHITE);
        board[N - 2 * WIDTH..].fill(BLACK);
        Self {
            board,
            to_move: Side::First,
            moves: 0,
            quiet_moves: 0,
            last_capturer: None,
        }
    }

    /// An empty board with nothing but the pieces given, for tests and puzzles.
    pub fn from_pieces(to_move: Side, white: &[usize], black: &[usize]) -> Self {
        let mut board = [EMPTY; N];
        for &s in white {
            board[s] = WHITE;
        }
        for &s in black {
            board[s] = BLACK;
        }
        Self {
            board,
            to_move,
            moves: 0,
            quiet_moves: 0,
            last_capturer: None,
        }
    }

    pub fn count(&self, side: Side) -> usize {
        let want = stone(side);
        self.board.iter().filter(|&&c| c == want).count()
    }

    /// Calls `f` with every move the side to move could make, before the symmetry rule is
    /// applied. Stops early, returning `true`, as soon as `f` does.
    fn for_each_pseudo_move(&self, mut f: impl FnMut(Move) -> bool) -> bool {
        let own = stone(self.to_move);
        let enemy = stone(self.to_move.other());
        let b = &self.board;
        for s in 0..N {
            if b[s] != own {
                continue;
            }
            for d in 0..8 {
                // A single piece steps one square onto an empty one. It can never capture: that
                // needs an enemy line shorter than one.
                if let Some(t) = step(s, d, 1) {
                    if b[t] == EMPTY && f(Move::new(s, s, t)) {
                        return true;
                    }
                }
                // Phalanxes of every size m >= 2 that `s` can lead along d: the own pieces
                // extending back from s against d.
                let longest = run(b, s, opposite(d), own);
                for m in 2..=longest {
                    let rear = step(s, opposite(d), m - 1).expect("inside the run");
                    for k in 1..=m {
                        let Some(t) = step(s, d, k) else { break };
                        match b[t] {
                            EMPTY => {
                                if f(Move::new(s, rear, t)) {
                                    return true;
                                }
                            }
                            c if c == enemy => {
                                if run(b, t, d, enemy) < m && f(Move::new(s, rear, t)) {
                                    return true;
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                }
            }
        }
        false
    }

    /// Whether the side to move has any move at all, ignoring the symmetry rule. This is pgx's
    /// "stuck" test, which a side with no move at all loses.
    fn has_pseudo_move(&self) -> bool {
        self.for_each_pseudo_move(|_| true)
    }

    /// A move may not put a piece on the opponent's back rank if that leaves the whole position
    /// left-to-right symmetric. Only the lead can newly arrive there, so only moves whose
    /// destination is on that rank need simulating.
    fn symmetry_allows(&self, mv: Move) -> bool {
        if mv.dest as usize / WIDTH != home_row(self.to_move.other()) {
            return true;
        }
        !is_mirror_symmetric(&do_move(&self.board, self.to_move, mv))
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::with_capacity(128);
        self.for_each_pseudo_move(|mv| {
            if self.symmetry_allows(mv) {
                out.push(mv);
            }
            false
        });
        out
    }

    /// The legal moves that capture: the ones a quiescence search looks at.
    pub fn legal_captures(&self) -> Vec<Move> {
        let enemy = stone(self.to_move.other());
        let mut out = Vec::new();
        self.for_each_pseudo_move(|mv| {
            if self.board[mv.dest as usize] == enemy && self.symmetry_allows(mv) {
                out.push(mv);
            }
            false
        });
        out
    }

    /// How many enemy pieces `mv` would capture: the enemy line from its destination onwards.
    /// Zero for a quiet move. `mv` must be a legal move here.
    pub fn captured_count(&self, mv: Move) -> usize {
        let enemy = stone(self.to_move.other());
        let dest = mv.dest as usize;
        if self.board[dest] != enemy || mv.is_single() {
            return 0;
        }
        let (d, _) = direction(mv.rear as usize, mv.lead as usize).expect("a phalanx is a line");
        run(&self.board, dest, d, enemy)
    }

    /// Pieces of `side` on its opponent's home rank.
    pub fn crossed(&self, side: Side) -> usize {
        let row = home_row(side.other());
        let want = stone(side);
        self.board[row * WIDTH..(row + 1) * WIDTH]
            .iter()
            .filter(|&&c| c == want)
            .count()
    }

    fn has_legal_move(&self) -> bool {
        self.for_each_pseudo_move(|mv| self.symmetry_allows(mv))
    }

    /// Play a legal move and hand the turn over.
    pub fn apply(&self, mv: Move) -> Position {
        let board = do_move(&self.board, self.to_move, mv);
        // A move never adds a piece and a capture always removes one, so the count alone says
        // whether this move captured.
        let captured = occupied(&board) < occupied(&self.board);
        Position {
            board,
            to_move: self.to_move.other(),
            moves: self.moves + 1,
            quiet_moves: if captured { 0 } else { self.quiet_moves + 1 },
            last_capturer: if captured {
                Some(self.to_move)
            } else {
                self.last_capturer
            },
        }
    }

    /// True if the side to move has more pieces on the opponent's back rank than the opponent
    /// has on theirs — the win condition, tested at the start of a turn.
    pub fn wins_at_turn_start(&self) -> bool {
        let me = self.to_move;
        let row_count = |row: usize, want: u8| {
            self.board[row * WIDTH..(row + 1) * WIDTH]
                .iter()
                .filter(|&&c| c == want)
                .count()
        };
        row_count(home_row(me.other()), stone(me)) > row_count(home_row(me), stone(me.other()))
    }

    /// Checked in pgx's order, which matters where two apply at once: an incursion win, then a
    /// side with no move at all losing, then the quiet-move clock.
    pub fn outcome(&self) -> Option<Outcome> {
        if self.wins_at_turn_start() {
            return Some(Outcome::win(self.to_move));
        }
        if !self.has_pseudo_move() {
            return Some(Outcome::win(self.to_move.other()));
        }
        if self.quiet_moves >= MAX_QUIET_MOVES {
            let winner = advancement_winner(&self.board)
                .or(self.last_capturer)
                .unwrap_or(Side::Second);
            return Some(Outcome::win(winner));
        }
        // Moves exist, but the symmetry rule forbids all of them. pgx never resolves this — its
        // action mask dead-ends at the destination stage — so it cannot occur in anything the two
        // engines are compared on. Treating it as having no move keeps every game finite.
        if !self.has_legal_move() {
            return Some(Outcome::win(self.to_move.other()));
        }
        None
    }
}

fn occupied(board: &[u8; N]) -> usize {
    board.iter().filter(|&&c| c != EMPTY).count()
}

pub fn is_mirror_symmetric(board: &[u8; N]) -> bool {
    board
        .chunks_exact(WIDTH)
        .all(|row| (0..WIDTH / 2).all(|c| row[c] == row[WIDTH - 1 - c]))
}

/// The board after moving the phalanx `rear..lead` so the lead lands on `dest`, capturing the
/// enemy line there if it lands on one.
fn do_move(board: &[u8; N], side: Side, mv: Move) -> [u8; N] {
    let (lead, rear, dest) = (mv.lead as usize, mv.rear as usize, mv.dest as usize);
    let (d, size) = if mv.is_single() {
        (direction(lead, dest).expect("a move goes somewhere").0, 1)
    } else {
        let (d, span) = direction(rear, lead).expect("a phalanx is a straight line");
        (d, span + 1)
    };
    let back = opposite(d);
    let own = stone(side);
    let enemy = stone(side.other());

    let mut out = *board;
    for i in 0..size {
        out[step(lead, back, i).expect("phalanx is on the board")] = EMPTY;
    }
    if board[dest] == enemy {
        for i in 0..run(board, dest, d, enemy) {
            out[step(dest, d, i).expect("inside the run")] = EMPTY;
        }
    }
    for i in 0..size {
        out[step(dest, back, i).expect("phalanx lands on the board")] = own;
    }
    out
}

/// Whoever has come further up the board, or `None` for an exact mirror.
///
/// Compares white's count on row `HEIGHT-1-i` against black's on row `i`, for i = 0, 1, 2, ...:
/// "i rows short of the opponent's home rank" for both. The first pair that differs decides.
pub fn advancement_winner(board: &[u8; N]) -> Option<Side> {
    let count = |row: usize, want: u8| {
        board[row * WIDTH..(row + 1) * WIDTH]
            .iter()
            .filter(|&&c| c == want)
            .count()
    };
    for i in 0..HEIGHT {
        let (w, b) = (count(HEIGHT - 1 - i, WHITE), count(i, BLACK));
        if w != b {
            return Some(if w > b { Side::First } else { Side::Second });
        }
    }
    None
}

/// The [`Game`] implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Epaminondas;

impl Game for Epaminondas {
    type State = Position;
    type Move = Move;

    fn id(&self) -> &'static str {
        ID
    }

    fn initial(&self) -> Position {
        Position::start()
    }

    fn side_to_move(&self, s: &Position) -> Side {
        s.to_move
    }

    fn legal_moves(&self, s: &Position) -> Vec<Move> {
        if s.outcome().is_some() {
            return Vec::new();
        }
        s.legal_moves()
    }

    fn apply(&self, s: &Position, mv: &Move) -> Position {
        s.apply(*mv)
    }

    fn outcome(&self, s: &Position) -> Option<Outcome> {
        s.outcome()
    }

    fn encode_state(&self, s: &Position) -> Value {
        let board: String = s
            .board
            .iter()
            .map(|&c| match c {
                WHITE => 'W',
                BLACK => 'B',
                _ => '.',
            })
            .collect();
        json!({
            "board": board,
            "to_move": s.to_move.index(),
            "moves": s.moves,
            "quiet_moves": s.quiet_moves,
            "last_capturer": s.last_capturer.map_or(-1, |side| side.index() as i64),
        })
    }

    fn decode_state(&self, v: &Value) -> Result<Position, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("state must be an object"))?;
        let field = |k: &str| {
            obj.get(k)
                .ok_or_else(|| DecodeError::new(format!("missing {k}")))
        };
        let text = field("board")?
            .as_str()
            .ok_or_else(|| DecodeError::new("board: not a string"))?;
        if text.len() != N {
            return Err(DecodeError::new(format!(
                "board: {} squares, expected {N}",
                text.len()
            )));
        }
        let mut board = [EMPTY; N];
        for (i, ch) in text.bytes().enumerate() {
            board[i] = match ch {
                b'.' => EMPTY,
                b'W' => WHITE,
                b'B' => BLACK,
                _ => {
                    return Err(DecodeError::new(format!(
                        "board: bad square {:?}",
                        ch as char
                    )))
                }
            };
        }
        let int = |k: &str| -> Result<i64, DecodeError> {
            field(k)?
                .as_i64()
                .ok_or_else(|| DecodeError::new(format!("{k}: not an integer")))
        };
        let counter = |k: &str| -> Result<u32, DecodeError> {
            u32::try_from(int(k)?).map_err(|_| DecodeError::new(format!("{k}: out of range")))
        };
        let to_move = match int("to_move")? {
            0 => Side::First,
            1 => Side::Second,
            other => return Err(DecodeError::new(format!("to_move: {other}"))),
        };
        let last_capturer = match int("last_capturer")? {
            -1 => None,
            0 => Some(Side::First),
            1 => Some(Side::Second),
            other => return Err(DecodeError::new(format!("last_capturer: {other}"))),
        };
        Ok(Position {
            board,
            to_move,
            moves: counter("moves")?,
            quiet_moves: counter("quiet_moves")?,
            last_capturer,
        })
    }

    fn encode_move(&self, s: &Position, mv: &Move) -> Value {
        json!({
            "lead": mv.lead,
            "rear": mv.rear,
            "dest": mv.dest,
            "san": self.describe_move(s, mv),
        })
    }

    fn decode_move(&self, v: &Value) -> Result<Move, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("move must be an object"))?;
        let square = |k: &str| -> Result<usize, DecodeError> {
            let n = obj
                .get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| DecodeError::new(format!("{k}: missing or not a square index")))?;
            if n as usize >= N {
                return Err(DecodeError::new(format!("{k}: {n} is off the board")));
            }
            Ok(n as usize)
        };
        Ok(Move::new(square("lead")?, square("rear")?, square("dest")?))
    }

    fn render(&self, s: &Position) -> String {
        notation::render(s)
    }

    fn describe_move(&self, s: &Position, mv: &Move) -> String {
        notation::describe(s, *mv)
    }
}
