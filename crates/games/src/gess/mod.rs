//! Gess (Cambridge University Games Society, 1993) on a 20 x 20 grid.
//!
//! A scalar port of pgx's `pgx/_src/games/gess.py` (env version v1), whose header comment is the
//! specification. The two implementations are held together by the conformance suite in
//! `tests/gess_conformance.rs`, which replays positions dumped from pgx; any rule change there
//! must regenerate those vectors.
//!
//! Cells are row-major, `idx = row * SIZE + col`. Row 0 is the top of the board, on white's side;
//! row 19 is the bottom, on black's. The outer ring (row or column 0 or 19) is a border: it is
//! always empty, and a stone pushed onto it is lost. The playing area is the inner 18 x 18,
//! [`MIN_IDX`]..=[`MAX_IDX`] on each axis.
//!
//! **Black moves first**, so black is [`Side::First`] and its stones are [`BLACK`] — the opposite
//! pairing to Epaminondas, and the one pgx uses for this game.

mod notation;
#[cfg(test)]
mod tests;

pub use notation::{parse_move, parse_square, square_label};

use crate::{DecodeError, Game, Outcome, Side};
use serde_json::{json, Value};
use std::cmp::Ordering;

pub const ID: &str = "gess/1";

/// The side of the whole grid, border ring included.
pub const SIZE: usize = 20;
pub const N: usize = SIZE * SIZE;
/// First and last row/column of the playing area; everything else is border.
pub const MIN_IDX: usize = 1;
pub const MAX_IDX: usize = SIZE - 2;

pub const EMPTY: u8 = 0;
/// Black, who moves first: [`Side::First`].
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;

/// The game ends once this many consecutive moves take no stone off the board, and is then
/// decided on the number of stones left. Both players' moves count towards it, and anything that
/// removes a stone — a capture, a self-capture, a stone pushed onto the border — resets it.
pub const MAX_QUIET_MOVES: u32 = 20;

/// The eight directions as (row, col) steps, in pgx's order.
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

/// How far a piece may travel when its centre cell is empty. With a stone in the centre it is
/// unlimited, which on this grid means [`SIZE`] - 1.
const SHORT_RANGE: i32 = 3;

#[inline]
fn row_col(cell: usize) -> (i32, i32) {
    ((cell / SIZE) as i32, (cell % SIZE) as i32)
}

#[inline]
fn cell_at(r: i32, c: i32) -> usize {
    r as usize * SIZE + c as usize
}

#[inline]
fn on_grid(r: i32, c: i32) -> bool {
    (0..SIZE as i32).contains(&r) && (0..SIZE as i32).contains(&c)
}

/// The border ring, which holds no stones: anything pushed onto it is removed.
#[inline]
fn on_border(r: i32, c: i32) -> bool {
    r == 0 || c == 0 || r == SIZE as i32 - 1 || c == SIZE as i32 - 1
}

/// Whether (r, c) is inside the 3 x 3 footprint centred on (cr, cc).
#[inline]
fn in_footprint(r: i32, c: i32, cr: i32, cc: i32) -> bool {
    (r - cr).abs() <= 1 && (c - cc).abs() <= 1
}

pub fn stone(side: Side) -> u8 {
    match side {
        Side::First => BLACK,
        Side::Second => WHITE,
    }
}

/// A move: the piece centred on `from` slides so that its centre lands on `to`. Both are cell
/// indices on the 20 x 20 grid, which is pgx's two-action encoding one field at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub from: u16,
    pub to: u16,
}

impl Move {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from: from as u16,
            to: to as u16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    pub board: [u8; N],
    pub to_move: Side,
    /// Completed moves, for reporting only.
    pub moves: u32,
    /// Moves since a stone was last taken off the board; the game is decided at
    /// [`MAX_QUIET_MOVES`].
    pub quiet_moves: u32,
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

impl Position {
    /// The standard opening: 43 stones a side, one ring each.
    pub fn start() -> Self {
        // White's rows, from the top; black mirrors each at SIZE - 1 - row.
        const WHITE_ROWS: [(usize, &[usize]); 4] = [
            (1, &[2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
            (2, &[1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 16, 17, 18]),
            (3, &[2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
            (6, &[2, 5, 8, 11, 14, 17]),
        ];
        let mut board = [EMPTY; N];
        for (row, cols) in WHITE_ROWS {
            for &col in cols {
                board[row * SIZE + col] = WHITE;
                board[(SIZE - 1 - row) * SIZE + col] = BLACK;
            }
        }
        Self {
            board,
            to_move: Side::First,
            moves: 0,
            quiet_moves: 0,
        }
    }

    /// An empty board with nothing but the stones given, for tests and puzzles. Black first,
    /// since black moves first.
    pub fn from_stones(to_move: Side, black: &[usize], white: &[usize]) -> Self {
        let mut board = [EMPTY; N];
        for &s in black {
            board[s] = BLACK;
        }
        for &s in white {
            board[s] = WHITE;
        }
        Self {
            board,
            to_move,
            moves: 0,
            quiet_moves: 0,
        }
    }

    /// Stones of `side` on the playing area. The border ring never holds any, so this is every
    /// stone a side has.
    pub fn count(&self, side: Side) -> usize {
        let want = stone(side);
        let mut n = 0;
        for r in MIN_IDX..=MAX_IDX {
            for c in MIN_IDX..=MAX_IDX {
                n += (self.board[r * SIZE + c] == want) as usize;
            }
        }
        n
    }

    /// Whether the 3 x 3 footprint centred on `centre` is a piece the side to move may pick up:
    /// no enemy stone anywhere in it, and at least one of its own in the eight cells around the
    /// centre. A footprint may hang over the border, whose cells are empty and count for nothing.
    pub fn is_piece(&self, centre: usize) -> bool {
        let (cr, cc) = row_col(centre);
        let (own, enemy) = (stone(self.to_move), stone(self.to_move.other()));
        let mut has_own = false;
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (cr + dr, cc + dc);
                if !on_grid(r, c) {
                    continue;
                }
                match self.board[cell_at(r, c)] {
                    v if v == enemy => return false,
                    v if v == own && (dr, dc) != (0, 0) => has_own = true,
                    _ => {}
                }
            }
        }
        has_own
    }

    /// Whether the piece at `from`, moved so its centre is on (r, c), would run into something.
    /// Only cells outside the piece's own footprint count: it carries that with it.
    fn obstructed(&self, from: usize, r: i32, c: i32) -> bool {
        let (fr, fc) = row_col(from);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (rr, cc) = (r + dr, c + dc);
                if !on_grid(rr, cc) || in_footprint(rr, cc, fr, fc) {
                    continue;
                }
                if self.board[cell_at(rr, cc)] != EMPTY {
                    return true;
                }
            }
        }
        false
    }

    /// Calls `f` with every cell the piece centred on `from` can move its centre to. Stops early,
    /// returning `true`, as soon as `f` does.
    ///
    /// A direction is open only when the piece has a stone on that side of its centre, and the
    /// piece travels at most three cells unless it has a stone in the centre as well. It may
    /// stop on the first cell where its footprint meets a stone — that is how a capture is made —
    /// but not go past it.
    pub fn for_each_destination(&self, from: usize, mut f: impl FnMut(usize) -> bool) -> bool {
        let own = stone(self.to_move);
        let (fr, fc) = row_col(from);
        let range = if self.board[from] == own {
            SIZE as i32 - 1
        } else {
            SHORT_RANGE
        };
        for (dr, dc) in DIRS {
            if !on_grid(fr + dr, fc + dc) || self.board[cell_at(fr + dr, fc + dc)] != own {
                continue;
            }
            for k in 1..=range {
                let (r, c) = (fr + dr * k, fc + dc * k);
                if !on_grid(r, c) {
                    break;
                }
                if f(cell_at(r, c)) {
                    return true;
                }
                if self.obstructed(from, r, c) {
                    break;
                }
            }
        }
        false
    }

    /// Every destination for one piece, in the order the rules generate them.
    pub fn destinations(&self, from: usize) -> Vec<usize> {
        let mut out = Vec::new();
        self.for_each_destination(from, |to| {
            out.push(to);
            false
        });
        out
    }

    /// Calls `f` with every legal move. Stops early, returning `true`, as soon as `f` does.
    pub fn for_each_move(&self, mut f: impl FnMut(Move) -> bool) -> bool {
        for from in 0..N {
            if !self.is_piece(from) {
                continue;
            }
            if self.for_each_destination(from, |to| f(Move::new(from, to))) {
                return true;
            }
        }
        false
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::new();
        self.for_each_move(|m| {
            out.push(m);
            false
        });
        out
    }

    /// Play a legal move and hand the turn over.
    pub fn apply(&self, mv: Move) -> Position {
        let own = stone(self.to_move);
        let (fr, fc) = row_col(mv.from as usize);
        let (tr, tc) = row_col(mv.to as usize);
        let mut board = self.board;
        // Lift the piece. Only the mover's own stones come up; legality keeps the enemy out of
        // the footprint, so nothing else is in it.
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (fr + dr, fc + dc);
                if on_grid(r, c) && board[cell_at(r, c)] == own {
                    board[cell_at(r, c)] = EMPTY;
                }
            }
        }
        // Everything standing where it lands is taken, the mover's own stones included.
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (tr + dr, tc + dc);
                if on_grid(r, c) {
                    board[cell_at(r, c)] = EMPTY;
                }
            }
        }
        // Set the piece down cell for cell. A stone that lands on the border is lost.
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (sr, sc) = (fr + dr, fc + dc);
                let (r, c) = (tr + dr, tc + dc);
                if on_grid(sr, sc)
                    && self.board[cell_at(sr, sc)] == own
                    && on_grid(r, c)
                    && !on_border(r, c)
                {
                    board[cell_at(r, c)] = own;
                }
            }
        }
        // A move never puts a stone on the board, so the total falls exactly when one was taken.
        let before = self.board.iter().filter(|&&v| v != EMPTY).count();
        let after = board.iter().filter(|&&v| v != EMPTY).count();
        Position {
            board,
            to_move: self.to_move.other(),
            moves: self.moves + 1,
            quiet_moves: if after < before {
                0
            } else {
                self.quiet_moves + 1
            },
        }
    }

    /// Whether `side` still has a ring: an empty cell of the playing area with eight of its own
    /// stones around it. Losing your last one loses the game.
    pub fn has_ring(&self, side: Side) -> bool {
        let own = stone(side);
        for r in MIN_IDX as i32..=MAX_IDX as i32 {
            for c in MIN_IDX as i32..=MAX_IDX as i32 {
                if self.board[cell_at(r, c)] != EMPTY {
                    continue;
                }
                let ringed = (-1..=1).all(|dr| {
                    (-1..=1)
                        .all(|dc| (dr, dc) == (0, 0) || self.board[cell_at(r + dr, c + dc)] == own)
                });
                if ringed {
                    return true;
                }
            }
        }
        false
    }

    /// Who has more stones left: the tiebreak when the game runs out of captures.
    pub fn material_result(&self) -> Outcome {
        match self.count(Side::First).cmp(&self.count(Side::Second)) {
            Ordering::Greater => Outcome::win(Side::First),
            Ordering::Less => Outcome::win(Side::Second),
            Ordering::Equal => Outcome::Draw,
        }
    }

    /// `None` while the game is still going.
    ///
    /// Rings are checked in pgx's order, which matters when one move ends both: the side that
    /// has just moved is asked first, so breaking your own last ring loses even if the same move
    /// took your opponent's. A ring can be lost without a capture, so a ring loss and the
    /// quiet-move count can fall on the same move; the ring decides it.
    pub fn outcome(&self) -> Option<Outcome> {
        let mover = self.to_move.other();
        if !self.has_ring(mover) {
            return Some(Outcome::win(self.to_move));
        }
        if !self.has_ring(self.to_move) {
            return Some(Outcome::win(mover));
        }
        if self.quiet_moves >= MAX_QUIET_MOVES {
            return Some(self.material_result());
        }
        None
    }
}

/// The [`Game`] implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gess;

impl Game for Gess {
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
                BLACK => 'B',
                WHITE => 'W',
                _ => '.',
            })
            .collect();
        json!({
            "board": board,
            "to_move": s.to_move.index(),
            "moves": s.moves,
            "quiet_moves": s.quiet_moves,
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
                "board: {} cells, expected {N}",
                text.len()
            )));
        }
        let mut board = [EMPTY; N];
        for (i, ch) in text.bytes().enumerate() {
            board[i] = match ch {
                b'.' => EMPTY,
                b'B' => BLACK,
                b'W' => WHITE,
                _ => {
                    return Err(DecodeError::new(format!(
                        "board: bad cell {:?}",
                        ch as char
                    )))
                }
            };
            if board[i] != EMPTY && on_border((i / SIZE) as i32, (i % SIZE) as i32) {
                return Err(DecodeError::new(format!(
                    "board: a stone on the border ring, at {}",
                    square_label(i)
                )));
            }
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
        Ok(Position {
            board,
            to_move,
            moves: counter("moves")?,
            quiet_moves: counter("quiet_moves")?,
        })
    }

    fn encode_move(&self, s: &Position, mv: &Move) -> Value {
        json!({
            "from": mv.from,
            "to": mv.to,
            "san": self.describe_move(s, mv),
        })
    }

    fn decode_move(&self, v: &Value) -> Result<Move, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("move must be an object"))?;
        let cell = |k: &str| -> Result<usize, DecodeError> {
            let n = obj
                .get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| DecodeError::new(format!("{k}: missing or not a cell index")))?;
            if n as usize >= N {
                return Err(DecodeError::new(format!("{k}: {n} is off the grid")));
            }
            Ok(n as usize)
        };
        Ok(Move::new(cell("from")?, cell("to")?))
    }

    fn render(&self, s: &Position) -> String {
        notation::render(s)
    }

    fn describe_move(&self, s: &Position, mv: &Move) -> String {
        notation::describe(s, *mv)
    }
}
