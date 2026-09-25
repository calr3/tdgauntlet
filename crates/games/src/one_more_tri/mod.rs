//! One More Tri: two players place numbered tiles on a board of 21 triangles, and the one
//! triangle left empty at the end decides the game.
//!
//! A scalar port of pgx's `pgx/_src/games/g_hex.py` (env id `g_hex`, version v1), which is the
//! specification. The two implementations are held together by the conformance suite in
//! `tests/one_more_tri_conformance.rs`, which replays positions dumped from pgx; any rule change
//! there must regenerate those vectors.
//!
//! Triangles are numbered 0..=20 exactly as in pgx's diagram (see [`render`](OneMoreTri)), and
//! [`ADJ`] is pgx's `_ADJ` with its padding entry dropped.
//!
//! **Black moves first**: black is [`Side::First`], and its tiles are stored as positive values,
//! white's as negative ones, as pgx stores them.

#[cfg(test)]
mod tests;

use crate::{DecodeError, Game, Outcome, Side};
use serde_json::{json, Value};

pub const ID: &str = "one-more-tri/1";

/// Triangles on the board.
pub const N: usize = 21;
/// Tiles in each hand, valued 1..=TILES.
pub const TILES: usize = 10;

/// Each triangle's neighbours: the triangles whose tiles count when it is the one left empty.
/// From pgx's `_ADJ`, whose padding index 21 (an always-empty triangle standing in for "no
/// neighbour") is left out. Symmetric: pgx v0 left 4 out of 10's list, and v1 put it back.
pub const ADJ: [&[usize]; N] = [
    &[1, 6],       //  0
    &[0, 2],       //  1
    &[1, 3, 8],    //  2
    &[2, 4],       //  3
    &[3, 10],      //  4
    &[6, 12],      //  5
    &[0, 5, 7],    //  6
    &[6, 8, 14],   //  7
    &[2, 7, 9],    //  8
    &[8, 10],      //  9
    &[4, 9, 11],   // 10
    &[10],         // 11
    &[5, 13],      // 12
    &[12, 14, 16], // 13
    &[7, 13, 15],  // 14
    &[14, 18],     // 15
    &[13, 17],     // 16
    &[16, 18],     // 17
    &[15, 17, 19], // 18
    &[18, 20],     // 19
    &[19],         // 20
];

/// Place tile `tile` (1..=10) on triangle `tri` (0..=20): pgx's single action
/// `tri * 10 + tile - 1`, one field at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub tri: u8,
    pub tile: u8,
}

impl Move {
    pub fn new(tri: usize, tile: usize) -> Self {
        Self {
            tri: tri as u8,
            tile: tile as u8,
        }
    }
}

/// The sign a side's tiles carry on the board.
fn sign(side: Side) -> i8 {
    match side {
        Side::First => 1,
        Side::Second => -1,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    /// Tile values, positive for black, negative for white, 0 for an empty triangle.
    pub board: [i8; N],
    pub to_move: Side,
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

impl Position {
    /// An empty board, black to move.
    pub fn start() -> Self {
        Self {
            board: [0; N],
            to_move: Side::First,
        }
    }

    /// Whether `side` still holds tile `tile`.
    pub fn in_hand(&self, side: Side, tile: usize) -> bool {
        let v = sign(side) * tile as i8;
        !self.board.contains(&v)
    }

    /// The tiles `side` still holds, smallest first.
    pub fn hand(&self, side: Side) -> Vec<usize> {
        (1..=TILES).filter(|&t| self.in_hand(side, t)).collect()
    }

    /// Tiles on the board.
    pub fn placed(&self) -> usize {
        self.board.iter().filter(|&&v| v != 0).count()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let hand = self.hand(self.to_move);
        let mut out = Vec::new();
        for tri in (0..N).filter(|&t| self.board[t] == 0) {
            for &tile in &hand {
                out.push(Move::new(tri, tile));
            }
        }
        out
    }

    pub fn apply(&self, mv: Move) -> Self {
        let mut board = self.board;
        board[mv.tri as usize] = sign(self.to_move) * mv.tile as i8;
        Self {
            board,
            to_move: self.to_move.other(),
        }
    }

    /// The signed sum of every empty triangle's neighbours: positive favours black. Once the
    /// game is over exactly one triangle is empty, so it is that triangle's neighbours alone.
    pub fn score(&self) -> i32 {
        (0..N)
            .filter(|&t| self.board[t] == 0)
            .flat_map(|t| ADJ[t].iter())
            .map(|&n| self.board[n] as i32)
            .sum()
    }

    /// Over once both hands are empty: then the score decides, and 0 is a draw.
    pub fn outcome(&self) -> Option<Outcome> {
        if self.placed() < 2 * TILES {
            return None;
        }
        Some(match self.score() {
            s if s > 0 => Outcome::Win { side: Side::First },
            s if s < 0 => Outcome::Win { side: Side::Second },
            _ => Outcome::Draw,
        })
    }
}

/// `7 on 13`: the tile, then the triangle, as pgx's interactive tournament reads and writes them.
pub fn describe(mv: Move) -> String {
    format!("{} on {}", mv.tile, mv.tri)
}

/// The inverse of [`describe`]. `None` unless it names a tile 1..=10 and a triangle 0..=20.
pub fn parse_move(text: &str) -> Option<Move> {
    let (tile, tri) = text.trim().split_once(" on ")?;
    let (tile, tri): (usize, usize) = (tile.trim().parse().ok()?, tri.trim().parse().ok()?);
    ((1..=TILES).contains(&tile) && tri < N).then(|| Move::new(tri, tile))
}

/// Two characters for a triangle: `B7`, `W3`, `BT` / `WT` for a 10, or the triangle's own number
/// when it is empty, so the diagram also says what to type.
fn cell(p: &Position, tri: usize) -> String {
    let v = p.board[tri];
    let colour = if v > 0 { 'B' } else { 'W' };
    match v.unsigned_abs() {
        0 => format!("{tri:>2}"),
        10 => format!("{colour}T"),
        n => format!("{colour}{n}"),
    }
}

fn render(p: &Position) -> String {
    let c = |t| cell(p, t);
    let hand = |side| {
        (1..=TILES)
            .map(|t| {
                if p.in_hand(side, t) {
                    format!("{t:2}")
                } else {
                    "  ".into()
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_string()
    };
    // pgx's diagram (`Game.pretty_game`), line for line.
    let mut out = String::new();
    out += "      ____________\n";
    out += "     /\\    /\\    /\\\n";
    out += &format!("    /{}\\{}/{}\\{}/{}\\\n", c(0), c(1), c(2), c(3), c(4));
    out += "   /____\\/____\\/____\\\n";
    out += "  /\\    /\\    /\\    /\\\n";
    out += &format!(
        " /{}\\{}/{}\\{}/{}\\{}/{}\\\n",
        c(5),
        c(6),
        c(7),
        c(8),
        c(9),
        c(10),
        c(11)
    );
    out += "/____\\/____\\/____\\/____\\\n";
    out += "\\    /\\    /\\\n";
    out += &format!(" \\{}/{}\\{}/{}\\\n", c(12), c(13), c(14), c(15));
    out += "  \\/____\\/____\\______\n";
    out += "   \\    /\\    /\\    /\n";
    out += &format!(
        "    \\{}/{}\\{}/{}\\{}/\n",
        c(16),
        c(17),
        c(18),
        c(19),
        c(20)
    );
    out += "     \\/____\\/____\\/\n";
    out += &format!("\nBlack tiles: {}\n", hand(Side::First));
    out += &format!("White tiles: {}\n", hand(Side::Second));
    let mover = match p.to_move {
        Side::First => "Black",
        Side::Second => "White",
    };
    match p.outcome() {
        None => out += &format!("{mover} to move\n"),
        Some(_) => out += &format!("Game over, score {}\n", p.score()),
    }
    out
}

pub struct OneMoreTri;

impl Game for OneMoreTri {
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
        json!({
            "board": s.board.to_vec(),
            "to_move": s.to_move.index(),
        })
    }

    fn decode_state(&self, v: &Value) -> Result<Position, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("state must be an object"))?;
        let cells = obj
            .get("board")
            .and_then(Value::as_array)
            .ok_or_else(|| DecodeError::new("board: missing or not an array"))?;
        if cells.len() != N {
            return Err(DecodeError::new(format!(
                "board: {} triangles, expected {N}",
                cells.len()
            )));
        }
        let mut board = [0i8; N];
        let mut seen = [[false; TILES + 1]; 2];
        for (i, c) in cells.iter().enumerate() {
            let v = c
                .as_i64()
                .filter(|v| v.unsigned_abs() as usize <= TILES)
                .ok_or_else(|| {
                    DecodeError::new(format!("board: triangle {i} is not a tile value: {c}"))
                })?;
            if v != 0 {
                let side = (v < 0) as usize;
                let tile = v.unsigned_abs() as usize;
                if std::mem::replace(&mut seen[side][tile], true) {
                    let colour = ["black", "white"][side];
                    return Err(DecodeError::new(format!(
                        "board: {colour}'s {tile} is on the board twice"
                    )));
                }
            }
            board[i] = v as i8;
        }
        let to_move = match obj.get("to_move").and_then(Value::as_i64) {
            Some(0) => Side::First,
            Some(1) => Side::Second,
            _ => return Err(DecodeError::new("to_move: missing, or not 0 or 1")),
        };
        // Black moves first and nobody passes, so the counts say whose turn it is.
        let (black, white) = (
            board.iter().filter(|&&v| v > 0).count(),
            board.iter().filter(|&&v| v < 0).count(),
        );
        let expected = match black.checked_sub(white) {
            Some(0) => Side::First,
            Some(1) => Side::Second,
            _ => {
                return Err(DecodeError::new(format!(
                    "board: {black} black tiles and {white} white cannot happen, black moves first"
                )))
            }
        };
        if to_move != expected {
            return Err(DecodeError::new(format!(
                "to_move: {} with {black} black tiles and {white} white on the board",
                to_move.index()
            )));
        }
        Ok(Position { board, to_move })
    }

    fn encode_move(&self, _s: &Position, mv: &Move) -> Value {
        json!({
            "tri": mv.tri,
            "tile": mv.tile,
            "san": describe(*mv),
        })
    }

    fn decode_move(&self, v: &Value) -> Result<Move, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("move must be an object"))?;
        let field = |k: &str, range: std::ops::RangeInclusive<u64>| {
            obj.get(k)
                .and_then(Value::as_u64)
                .filter(|n| range.contains(n))
                .map(|n| n as usize)
                .ok_or_else(|| {
                    DecodeError::new(format!(
                        "{k}: missing, or not an integer in {}..={}",
                        range.start(),
                        range.end()
                    ))
                })
        };
        Ok(Move::new(
            field("tri", 0..=N as u64 - 1)?,
            field("tile", 1..=TILES as u64)?,
        ))
    }

    fn render(&self, s: &Position) -> String {
        render(s)
    }

    fn describe_move(&self, _s: &Position, mv: &Move) -> String {
        describe(*mv)
    }
}
