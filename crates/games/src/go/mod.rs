//! Go on a 19 x 19 board: area scoring, komi 7.5, simple ko, and positional superko as a loss.
//!
//! A scalar port of pgx's `pgx/_src/games/go.py` (env id `go_19x19`, version v1), which is the
//! specification. The two implementations are held together by the conformance suite in
//! `tests/go_conformance.rs`, which replays games dumped from pgx; any rule change there must
//! regenerate those vectors.
//!
//! Points are row-major, `p = row * SIZE + col`, as pgx indexes them. Row 0 is rank 1 and column
//! 0 is file A, so point 0 is `A1` and point 60 is `D4`. (The board is symmetric, so pgx has no
//! opinion; this is the usual diagram with black's first corner at the bottom left.)
//!
//! A move is a pgx action: a point `0..=360`, or [`PASS`] (361). pgx's rules, which are not quite
//! any human ruleset:
//!
//! - Suicide is illegal, and so is retaking a ko at once (a capture of one stone by a stone that
//!   is then alone in atari).
//! - A move that repeats an earlier position is **legal, but ends the game and loses it** for the
//!   player who made it (positional superko: any earlier position, whoever was to move).
//! - Two passes in a row end the game, and so does the 722nd move (`SIZE * SIZE * 2`).
//! - Area scoring: your stones, plus the empty points from which no path through empty points
//!   reaches one of the opponent's stones. An empty area that touches neither colour counts for
//!   both; one that touches both counts for neither. Black wins if its score, less 7.5, is more
//!   than white's, so there are no draws.
//!
//! **Black moves first**, so black is [`Side::First`], stored as `1`; white is `-1`.

#[cfg(test)]
mod tests;

use crate::{DecodeError, Game, Outcome, Side};
use serde_json::{json, Value};

pub const ID: &str = "go-19x19/1";

pub const SIZE: usize = 19;
/// Points on the board.
pub const N: usize = SIZE * SIZE;
/// pgx's action for a pass. Every other action is the point played.
pub const PASS: u16 = N as u16;
/// The game ends when this many moves (passes included) have been played.
pub const MAX_MOVES: usize = 2 * N;
/// Komi, doubled so that scores stay integers: black wins if `2 * (black - white) > 15`.
pub const KOMI_X2: i32 = 15;

/// File letters, without I.
const FILES: &[u8; SIZE] = b"ABCDEFGHJKLMNOPQRST";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Move {
    Play(u16),
    Pass,
}

impl Move {
    /// The pgx action.
    pub fn action(self) -> u16 {
        match self {
            Move::Play(p) => p,
            Move::Pass => PASS,
        }
    }

    pub fn from_action(a: u16) -> Option<Move> {
        match a {
            PASS => Some(Move::Pass),
            p if (p as usize) < N => Some(Move::Play(p)),
            _ => None,
        }
    }
}

/// The sign a side's stones carry on the board.
fn sign(side: Side) -> i8 {
    match side {
        Side::First => 1,
        Side::Second => -1,
    }
}

/// The up to four points next to `p`.
fn neighbours(p: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (p / SIZE, p % SIZE);
    [
        (r > 0).then(|| p - SIZE),
        (r + 1 < SIZE).then(|| p + SIZE),
        (c > 0).then(|| p - 1),
        (c + 1 < SIZE).then(|| p + 1),
    ]
    .into_iter()
    .flatten()
}

const fn splitmix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Zobrist keys for a black and a white stone on each point, for spotting repeated positions.
const ZOBRIST: [[u64; 2]; N] = {
    let mut t = [[0; 2]; N];
    let mut i = 0;
    while i < N {
        t[i] = [splitmix(2 * i as u64), splitmix(2 * i as u64 + 1)];
        i += 1;
    }
    t
};

fn zobrist(p: usize, stone: i8) -> u64 {
    ZOBRIST[p][(stone < 0) as usize]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    /// `1` black, `-1` white, `0` empty, by point.
    pub board: [i8; N],
    pub to_move: Side,
    /// The point the side to move may not play on, having just lost a ko there.
    pub ko: Option<u16>,
    /// Every move since the empty board, as pgx actions.
    pub history: Vec<u16>,
    /// Passes in a row at the end of `history`.
    passes: u8,
    /// Whether the last move repeated an earlier position.
    superko: bool,
    /// Hash of the board.
    hash: u64,
    /// The board hash after each move in `history`. pgx's list likewise leaves out the empty
    /// board the game started from.
    seen: Vec<u64>,
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

/// The stones of one chain and how many distinct liberties it has.
struct Chain {
    stones: Vec<usize>,
    liberties: usize,
}

impl Position {
    /// An empty board, black to move.
    pub fn start() -> Self {
        Self {
            board: [0; N],
            to_move: Side::First,
            ko: None,
            history: Vec::new(),
            passes: 0,
            superko: false,
            hash: 0,
            seen: Vec::new(),
        }
    }

    /// Play `moves` from the start, or say which one is illegal (counting from 0).
    pub fn from_history(moves: &[u16]) -> Result<Self, String> {
        let mut p = Self::start();
        for (i, &a) in moves.iter().enumerate() {
            let mv = Move::from_action(a).ok_or_else(|| format!("move {i}: no action {a}"))?;
            if p.outcome().is_some() {
                return Err(format!("move {i}: the game was already over"));
            }
            if !p.is_legal(mv) {
                return Err(format!("move {i}: {} is illegal", describe(mv)));
            }
            p = p.apply(mv);
        }
        Ok(p)
    }

    /// Passes in a row at the end of the game so far.
    pub fn passes(&self) -> u8 {
        self.passes
    }

    /// Whether the last move repeated an earlier position, which ended the game.
    pub fn superko(&self) -> bool {
        self.superko
    }

    fn chain(&self, from: usize) -> Chain {
        let colour = self.board[from];
        let mut seen = [false; N];
        let mut stones = vec![from];
        let mut liberties = 0;
        seen[from] = true;
        let mut i = 0;
        while i < stones.len() {
            for q in neighbours(stones[i]) {
                if std::mem::replace(&mut seen[q], true) {
                    continue;
                }
                match self.board[q] {
                    0 => liberties += 1,
                    c if c == colour => stones.push(q),
                    _ => {}
                }
            }
            i += 1;
        }
        Chain { stones, liberties }
    }

    /// Whether the side to move may play `mv`. Says nothing about whether the game is over.
    pub fn is_legal(&self, mv: Move) -> bool {
        let Move::Play(p) = mv else {
            return true;
        };
        let p = p as usize;
        if self.board[p] != 0 || self.ko == Some(p as u16) {
            return false;
        }
        let me = sign(self.to_move);
        neighbours(p).any(|q| match self.board[q] {
            0 => true,
            c if c == me => self.chain(q).liberties >= 2,
            _ => self.chain(q).liberties == 1,
        })
    }

    /// Every legal move, points ascending and then the pass: pgx's action order.
    pub fn legal_moves(&self) -> Vec<Move> {
        // Label each chain once, with its liberty count, rather than walk it from every point.
        const NONE: u16 = u16::MAX;
        let mut label = [NONE; N];
        let mut liberties: Vec<usize> = Vec::new();
        let mut counted = [NONE; N];
        for p in 0..N {
            if self.board[p] == 0 || label[p] != NONE {
                continue;
            }
            let id = liberties.len() as u16;
            let (colour, mut stack, mut libs) = (self.board[p], vec![p], 0);
            label[p] = id;
            while let Some(s) = stack.pop() {
                for q in neighbours(s) {
                    if self.board[q] == 0 {
                        if counted[q] != id {
                            counted[q] = id;
                            libs += 1;
                        }
                    } else if self.board[q] == colour && label[q] == NONE {
                        label[q] = id;
                        stack.push(q);
                    }
                }
            }
            liberties.push(libs);
        }
        let me = sign(self.to_move);
        let mut out: Vec<Move> = (0..N)
            .filter(|&p| self.board[p] == 0 && self.ko != Some(p as u16))
            .filter(|&p| {
                neighbours(p).any(|q| match self.board[q] {
                    0 => true,
                    c if c == me => liberties[label[q] as usize] >= 2,
                    _ => liberties[label[q] as usize] == 1,
                })
            })
            .map(|p| Move::Play(p as u16))
            .collect();
        out.push(Move::Pass);
        out
    }

    /// Play a legal move.
    pub fn apply(&self, mv: Move) -> Self {
        let mut next = self.clone();
        next.ko = None;
        next.superko = false;
        next.history.push(mv.action());
        match mv {
            Move::Pass => next.passes += 1,
            Move::Play(p) => {
                let p = p as usize;
                let me = sign(self.to_move);
                next.passes = 0;
                let (mut captured, mut last) = (0, 0);
                for q in neighbours(p) {
                    // A chain touching p twice is gone after the first time.
                    if next.board[q] != -me {
                        continue;
                    }
                    let chain = next.chain(q);
                    if chain.liberties == 1 {
                        for &s in &chain.stones {
                            next.board[s] = 0;
                            next.hash ^= zobrist(s, -me);
                        }
                        captured += chain.stones.len();
                        last = q;
                    }
                }
                // A ko: one stone taken by a stone played where the opponent had every
                // neighbouring point, so that it could be taken straight back.
                if captured == 1 && neighbours(p).all(|q| self.board[q] == -me) {
                    next.ko = Some(last as u16);
                }
                next.board[p] = me;
                next.hash ^= zobrist(p, me);
                next.superko = self.seen.contains(&next.hash);
            }
        }
        next.seen.push(next.hash);
        next.to_move = self.to_move.other();
        next
    }

    /// Area scores, (black, white): stones, plus the empty points from which no path through
    /// empty points reaches the other colour. Komi is not included.
    pub fn score(&self) -> (i32, i32) {
        let mut score = [0i32; 2];
        for &c in &self.board {
            match c {
                1 => score[0] += 1,
                -1 => score[1] += 1,
                _ => {}
            }
        }
        let mut seen = [false; N];
        for p in 0..N {
            if self.board[p] != 0 || seen[p] {
                continue;
            }
            let (mut stack, mut size, mut touches) = (vec![p], 0, [false; 2]);
            seen[p] = true;
            while let Some(s) = stack.pop() {
                size += 1;
                for q in neighbours(s) {
                    match self.board[q] {
                        0 if !seen[q] => {
                            seen[q] = true;
                            stack.push(q);
                        }
                        0 => {}
                        c => touches[(c < 0) as usize] = true,
                    }
                }
            }
            for side in 0..2 {
                if !touches[1 - side] {
                    score[side] += size;
                }
            }
        }
        (score[0], score[1])
    }

    pub fn outcome(&self) -> Option<Outcome> {
        if self.superko {
            // The side to move is the one that did not repeat the position.
            return Some(Outcome::win(self.to_move));
        }
        if self.passes < 2 && self.history.len() < MAX_MOVES {
            return None;
        }
        let (black, white) = self.score();
        Some(Outcome::win(if 2 * (black - white) > KOMI_X2 {
            Side::First
        } else {
            Side::Second
        }))
    }
}

/// `D4`, or `pass`.
pub fn describe(mv: Move) -> String {
    match mv {
        Move::Pass => "pass".into(),
        Move::Play(p) => point_label(p as usize),
    }
}

pub fn point_label(p: usize) -> String {
    format!("{}{}", FILES[p % SIZE] as char, p / SIZE + 1)
}

/// The inverse of [`describe`], ignoring case.
pub fn parse_move(text: &str) -> Option<Move> {
    let text = text.trim().to_ascii_uppercase();
    if text == "PASS" {
        return Some(Move::Pass);
    }
    let first = *text.as_bytes().first()?;
    let file = FILES.iter().position(|&f| f == first)?;
    let rank: usize = text[1..].parse().ok()?;
    (1..=SIZE)
        .contains(&rank)
        .then(|| Move::Play(((rank - 1) * SIZE + file) as u16))
}

fn render(p: &Position) -> String {
    const STARS: [usize; 3] = [3, 9, 15];
    let files: String = FILES.iter().map(|&f| format!(" {}", f as char)).collect();
    let last = match p.history.last() {
        Some(&a) if a != PASS => Some(a as usize),
        _ => None,
    };
    let mut out = format!("   {files}\n");
    for r in (0..SIZE).rev() {
        out += &format!("{:>2} ", r + 1);
        for c in 0..SIZE {
            let q = r * SIZE + c;
            let cell = match p.board[q] {
                1 => 'X',
                -1 => 'O',
                _ if STARS.contains(&r) && STARS.contains(&c) => '+',
                _ => '.',
            };
            // The last stone played is bracketed.
            let (before, after) = match last {
                Some(l) if l == q => ('(', ')'),
                Some(l) if l + 1 == q && c > 0 => (')', ' '),
                _ => (' ', ' '),
            };
            out.push(before);
            out.push(cell);
            if c == SIZE - 1 {
                out.push(after);
            }
        }
        out += &format!(" {}\n", r + 1);
    }
    out += &format!("   {files}\n\n");
    let colour = |side| match side {
        Side::First => "Black (X)",
        Side::Second => "White (O)",
    };
    let (black, white) = p.score();
    out += &format!(
        "Move {}. Area: black {black}, white {white} + 7.5 komi.",
        p.history.len()
    );
    if let Some(ko) = p.ko {
        out += &format!(" Ko: {} is barred.", point_label(ko as usize));
    }
    out += "\n";
    match p.outcome() {
        None if p.passes == 1 => {
            out += &format!("{} to move; a pass ends the game\n", colour(p.to_move))
        }
        None => out += &format!("{} to move\n", colour(p.to_move)),
        Some(_) if p.superko => {
            out += &format!(
                "Game over: {} repeated a position and loses\n",
                colour(p.to_move.other())
            )
        }
        Some(o) => {
            let winner = if o == Outcome::win(Side::First) {
                Side::First
            } else {
                Side::Second
            };
            out += &format!("Game over: {} wins\n", colour(winner));
        }
    }
    out
}

pub struct Go;

impl Game for Go {
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
            "ko": s.ko,
            "history": s.history,
        })
    }

    /// The history is the state: it is replayed, and the other fields must agree with it.
    fn decode_state(&self, v: &Value) -> Result<Position, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("state must be an object"))?;
        let history: Vec<u16> = obj
            .get("history")
            .and_then(Value::as_array)
            .ok_or_else(|| DecodeError::new("history: missing or not an array"))?
            .iter()
            .enumerate()
            .map(|(i, a)| {
                a.as_u64()
                    .filter(|&a| a <= PASS as u64)
                    .map(|a| a as u16)
                    .ok_or_else(|| {
                        DecodeError::new(format!("history: move {i} is not an action 0..=361: {a}"))
                    })
            })
            .collect::<Result<_, _>>()?;
        if history.len() > MAX_MOVES {
            return Err(DecodeError::new(format!(
                "history: {} moves, the game ends at {MAX_MOVES}",
                history.len()
            )));
        }
        let pos = Position::from_history(&history)
            .map_err(|e| DecodeError::new(format!("history: {e}")))?;

        let cells = obj
            .get("board")
            .and_then(Value::as_array)
            .ok_or_else(|| DecodeError::new("board: missing or not an array"))?;
        if cells.len() != N {
            return Err(DecodeError::new(format!(
                "board: {} points, expected {N}",
                cells.len()
            )));
        }
        for (i, c) in cells.iter().enumerate() {
            let v = c.as_i64().filter(|v| (-1..=1).contains(v)).ok_or_else(|| {
                DecodeError::new(format!("board: point {i} is not -1, 0 or 1: {c}"))
            })?;
            if v != pos.board[i] as i64 {
                return Err(DecodeError::new(format!(
                    "board: {} is {v}, but the history leaves {}",
                    point_label(i),
                    pos.board[i]
                )));
            }
        }
        let to_move = obj.get("to_move").and_then(Value::as_u64);
        if to_move != Some(pos.to_move.index() as u64) {
            return Err(DecodeError::new(format!(
                "to_move: {}, but the history leaves {} to move",
                obj.get("to_move").unwrap_or(&Value::Null),
                pos.to_move.index()
            )));
        }
        let ko = obj.get("ko").unwrap_or(&Value::Null);
        if *ko != json!(pos.ko) {
            return Err(DecodeError::new(format!(
                "ko: {ko}, but the history leaves {}",
                json!(pos.ko)
            )));
        }
        Ok(pos)
    }

    fn encode_move(&self, _s: &Position, mv: &Move) -> Value {
        match mv {
            Move::Play(p) => json!({"point": p, "san": describe(*mv)}),
            Move::Pass => json!({"pass": true, "san": "pass"}),
        }
    }

    fn decode_move(&self, v: &Value) -> Result<Move, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("move must be an object"))?;
        match (obj.get("pass"), obj.get("point")) {
            (Some(Value::Bool(true)), None) => Ok(Move::Pass),
            (None | Some(Value::Bool(false)), Some(p)) => p
                .as_u64()
                .filter(|&p| p < N as u64)
                .map(|p| Move::Play(p as u16))
                .ok_or_else(|| {
                    DecodeError::new(format!("point: not an integer in 0..={}: {p}", N - 1))
                }),
            (Some(Value::Bool(true)), Some(_)) => {
                Err(DecodeError::new("a move is a point or a pass, not both"))
            }
            (Some(p), _) if !p.is_boolean() => Err(DecodeError::new("pass: must be true")),
            _ => Err(DecodeError::new("a move needs a point, or \"pass\": true")),
        }
    }

    fn render(&self, s: &Position) -> String {
        render(s)
    }

    fn describe_move(&self, _s: &Position, mv: &Move) -> String {
        describe(*mv)
    }
}
