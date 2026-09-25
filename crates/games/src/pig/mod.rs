//! Pig: roll a die as often as you dare, and bank the turn's total before a 1 wipes it out.
//! First to bank 100 wins.
//!
//! A scalar port of pgx's `pgx/_src/games/pig.py` (env id `pig`, version v0), which is the
//! specification. The two implementations are held together by the conformance suite in
//! `tests/pig_conformance.rs`, which replays positions dumped from pgx with every roll of the
//! die after every move; any rule change there must regenerate those vectors.
//!
//! The one game here with chance in it. As in pgx, the die is rolled as part of the move before
//! it — the start rolls for the first player, `stop` for the next, `roll` for the same one — so
//! a position always shows the roll its player is deciding on. [`Position::play`] takes that roll
//! as an argument; the [`Game`] impl draws it from the server's generator.

#[cfg(test)]
mod tests;

use crate::{DecodeError, Game, Outcome, Side};
use rand::{Rng, RngCore};
use serde_json::{json, Value};

pub const ID: &str = "pig/1";

/// The banked total that wins.
pub const TARGET: u32 = 100;
/// Faces of the die, 1..=FACES.
pub const FACES: u8 = 6;

/// pgx's actions: 0 is `stop`, 1 is `roll`, and moves sort the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Move {
    /// Bank the turn total (nothing, after a 1) and pass the die.
    Stop,
    /// Roll again. Not allowed after a 1.
    Roll,
}

impl Move {
    pub fn action(self) -> usize {
        self as usize
    }

    pub fn from_action(action: usize) -> Option<Self> {
        match action {
            0 => Some(Move::Stop),
            1 => Some(Move::Roll),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Move::Stop => "stop",
            Move::Roll => "roll",
        }
    }

    /// The inverse of [`Move::name`].
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "stop" => Some(Move::Stop),
            "roll" => Some(Move::Roll),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    /// Banked, indexed by side.
    pub totals: [u32; 2],
    /// This turn's rolls so far, the last one included, even when it is a 1 and about to be lost.
    pub turn_total: u32,
    /// The roll `to_move` is deciding on, 1..=6.
    pub last_roll: u8,
    pub to_move: Side,
}

impl Position {
    /// Nothing banked, and the first player has rolled `roll`.
    pub fn start(roll: u8) -> Self {
        Self {
            totals: [0, 0],
            turn_total: roll as u32,
            last_roll: roll,
            to_move: Side::First,
        }
    }

    /// A 1 was rolled: the turn total is lost and the only move is to stop.
    pub fn pigged_out(&self) -> bool {
        self.last_roll == 1
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        if self.pigged_out() {
            vec![Move::Stop]
        } else {
            vec![Move::Stop, Move::Roll]
        }
    }

    /// Play `mv`, after which the die shows `roll`: the next player's first roll after a stop,
    /// or the same player's next roll.
    pub fn play(&self, mv: Move, roll: u8) -> Self {
        debug_assert!((1..=FACES).contains(&roll));
        let mut next = self.clone();
        next.last_roll = roll;
        match mv {
            Move::Stop => {
                if !self.pigged_out() {
                    next.totals[self.to_move.index()] += self.turn_total;
                }
                next.turn_total = roll as u32;
                next.to_move = self.to_move.other();
            }
            Move::Roll => next.turn_total += roll as u32,
        }
        next
    }

    /// Won by whoever has banked [`TARGET`]. Only a stop banks, so that is the player who has
    /// just passed the die: pgx still rolls it for the other one, and so does this.
    pub fn outcome(&self) -> Option<Outcome> {
        [Side::First, Side::Second]
            .into_iter()
            .find(|s| self.totals[s.index()] >= TARGET)
            .map(|side| Outcome::Win { side })
    }
}

fn roll_die(rng: &mut dyn RngCore) -> u8 {
    rng.random_range(1..=FACES)
}

fn render(p: &Position) -> String {
    let who = |s: Side| format!("P{}", s.index());
    let mut out = format!("Totals: P0 {}  P1 {}\n", p.totals[0], p.totals[1]);
    let me = who(p.to_move);
    if let Some(Outcome::Win { side }) = p.outcome() {
        out += &format!("Game over: {} won\n", who(side));
    } else if p.pigged_out() {
        out += &format!(
            "{me} rolled a 1: pig out, must stop (turn total {} is lost)\n",
            p.turn_total
        );
    } else {
        out += &format!(
            "{me} to move: rolled {}, turn total {}\n",
            p.last_roll, p.turn_total
        );
    }
    out
}

pub struct Pig;

impl Game for Pig {
    type State = Position;
    type Move = Move;

    fn id(&self) -> &'static str {
        ID
    }

    fn initial(&self) -> Position {
        panic!("Pig starts with a roll of the die: use initial_with")
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

    fn apply(&self, _s: &Position, _mv: &Move) -> Position {
        panic!("every Pig move rolls the die: use apply_with, or Position::play with the roll")
    }

    fn outcome(&self, s: &Position) -> Option<Outcome> {
        s.outcome()
    }

    fn has_chance(&self) -> bool {
        true
    }

    fn initial_with(&self, rng: &mut dyn RngCore) -> Position {
        Position::start(roll_die(rng))
    }

    fn apply_with(&self, s: &Position, mv: &Move, rng: &mut dyn RngCore) -> Position {
        s.play(*mv, roll_die(rng))
    }

    fn encode_state(&self, s: &Position) -> Value {
        json!({
            "totals": s.totals,
            "turn_total": s.turn_total,
            "last_roll": s.last_roll,
            "to_move": s.to_move.index(),
        })
    }

    fn decode_state(&self, v: &Value) -> Result<Position, DecodeError> {
        let obj = v
            .as_object()
            .ok_or_else(|| DecodeError::new("state must be an object"))?;
        // pgx keeps these in int32s.
        let count = |k: &str, v: Option<&Value>| {
            v.and_then(Value::as_u64)
                .filter(|&n| n <= i32::MAX as u64)
                .map(|n| n as u32)
                .ok_or_else(|| DecodeError::new(format!("{k}: missing, or not a count")))
        };
        let totals = match obj.get("totals").and_then(Value::as_array) {
            Some(t) if t.len() == 2 => [count("totals", t.first())?, count("totals", t.get(1))?],
            _ => return Err(DecodeError::new("totals: missing, or not two counts")),
        };
        let turn_total = count("turn_total", obj.get("turn_total"))?;
        let last_roll = obj
            .get("last_roll")
            .and_then(Value::as_u64)
            .filter(|n| (1..=FACES as u64).contains(n))
            .ok_or_else(|| DecodeError::new("last_roll: missing, or not 1..=6"))?
            as u8;
        let to_move = match obj.get("to_move").and_then(Value::as_i64) {
            Some(0) => Side::First,
            Some(1) => Side::Second,
            _ => return Err(DecodeError::new("to_move: missing, or not 0 or 1")),
        };
        // The earlier rolls of a turn are all 2 or more: a 1 would have ended it.
        match turn_total.checked_sub(last_roll as u32) {
            None | Some(1) => {
                return Err(DecodeError::new(format!(
                "turn_total: {turn_total} cannot follow earlier rolls of 2 to 6 and a {last_roll}"
            )))
            }
            _ => {}
        }
        if totals[to_move.index()] >= TARGET {
            return Err(DecodeError::new(format!(
                "totals: player {} has {TARGET} or more, but only the player who just stopped can",
                to_move.index()
            )));
        }
        Ok(Position {
            totals,
            turn_total,
            last_roll,
            to_move,
        })
    }

    fn encode_move(&self, _s: &Position, mv: &Move) -> Value {
        json!({"action": mv.name(), "san": mv.name()})
    }

    fn decode_move(&self, v: &Value) -> Result<Move, DecodeError> {
        v.as_object()
            .and_then(|o| o.get("action"))
            .and_then(Value::as_str)
            .and_then(Move::parse)
            .ok_or_else(|| DecodeError::new("action: missing, or not \"stop\" or \"roll\""))
    }

    fn render(&self, s: &Position) -> String {
        render(s)
    }

    fn describe_move(&self, _s: &Position, mv: &Move) -> String {
        mv.name().into()
    }
}
