//! Game rules behind a game-agnostic interface.
//!
//! Each game implements [`Game`] with its own concrete state and move types, which is what an
//! engine (a search client, say) links against directly. The tournament never sees those types:
//! it goes through [`DynGame`], which every [`Game`] gets for free and which speaks only the
//! opaque JSON that travels on the wire.

pub mod epaminondas;
pub mod gess;
pub mod go;
pub mod one_more_tri;
pub mod pig;

use rand::RngCore;
use serde_json::Value;
use std::sync::Arc;
pub use tdg_proto::{Outcome, Side};

/// A two-player, perfect-information, turn-based game, possibly with dice.
///
/// A game with chance in it (Pig) overrides [`Game::initial_with`] and [`Game::apply_with`],
/// which draw it from the server's generator, and says so in [`Game::has_chance`]. Its
/// [`Game::initial`] and [`Game::apply`] have no dice to roll and panic.
pub trait Game: Send + Sync + 'static {
    type State: Clone + Send + Sync + std::fmt::Debug;
    type Move: Clone + PartialEq + Send + Sync + std::fmt::Debug;

    /// `"<name>/<version>"`. The version changes whenever a rule or the encoding does.
    fn id(&self) -> &'static str;
    fn initial(&self) -> Self::State;
    fn side_to_move(&self, state: &Self::State) -> Side;
    /// Every legal move. Empty only when the position is terminal.
    fn legal_moves(&self, state: &Self::State) -> Vec<Self::Move>;
    /// Play a move already known to be legal. Behaviour on an illegal move is unspecified.
    fn apply(&self, state: &Self::State, mv: &Self::Move) -> Self::State;
    /// `None` while the game is still going.
    fn outcome(&self, state: &Self::State) -> Option<Outcome>;

    /// Whether playing a move can involve chance, so that the position it leads to is not
    /// known until the server has played it.
    fn has_chance(&self) -> bool {
        false
    }
    /// The starting position, with any chance in it drawn from `rng`.
    fn initial_with(&self, rng: &mut dyn RngCore) -> Self::State {
        let _ = rng;
        self.initial()
    }
    /// [`Game::apply`], with any chance in the move drawn from `rng`.
    fn apply_with(
        &self,
        state: &Self::State,
        mv: &Self::Move,
        rng: &mut dyn RngCore,
    ) -> Self::State {
        let _ = rng;
        self.apply(state, mv)
    }

    fn encode_state(&self, state: &Self::State) -> Value;
    fn decode_state(&self, value: &Value) -> Result<Self::State, DecodeError>;
    fn encode_move(&self, state: &Self::State, mv: &Self::Move) -> Value;
    fn decode_move(&self, value: &Value) -> Result<Self::Move, DecodeError>;

    /// A board diagram for logs and the keyboard client.
    fn render(&self, state: &Self::State) -> String;
    /// Human notation for a move, e.g. `d3-d5`.
    fn describe_move(&self, state: &Self::State, mv: &Self::Move) -> String;
}

/// A value on the wire that does not describe a valid state or move.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DecodeError(pub String);

impl DecodeError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// Why a move offered by a client was not played.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MoveError {
    #[error("malformed move: {0}")]
    Malformed(DecodeError),
    #[error("illegal move: {0}")]
    Illegal(String),
    #[error("the game is already over")]
    GameOver,
}

/// The type-erased face of a [`Game`], in terms of wire JSON only.
///
/// States are decoded on every call. That costs microseconds against a network round trip of
/// milliseconds per move, and in exchange the tournament holds nothing but opaque JSON.
///
/// Anything that can involve chance takes the generator to draw it from; games without any
/// ignore it.
pub trait DynGame: Send + Sync {
    fn id(&self) -> &'static str;
    fn has_chance(&self) -> bool;
    fn initial(&self, rng: &mut dyn RngCore) -> Value;
    fn side_to_move(&self, state: &Value) -> Result<Side, DecodeError>;
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, DecodeError>;
    /// Every legal move with its human notation, sorted by notation.
    fn described_moves(&self, state: &Value) -> Result<Vec<(Value, String)>, DecodeError>;
    /// Validate `mv` against `state` and play it, returning the new state and the move in
    /// canonical form (as the server would encode it, whatever extra fields the client sent).
    fn play(&self, state: &Value, mv: &Value, rng: &mut dyn RngCore) -> Result<Played, MoveError>;
    fn outcome(&self, state: &Value) -> Result<Option<Outcome>, DecodeError>;
    fn render(&self, state: &Value) -> Result<String, DecodeError>;
}

/// The result of [`DynGame::play`].
#[derive(Debug, Clone, PartialEq)]
pub struct Played {
    pub state: Value,
    pub mv: Value,
    pub notation: String,
}

impl<G: Game> DynGame for G {
    fn id(&self) -> &'static str {
        Game::id(self)
    }

    fn has_chance(&self) -> bool {
        Game::has_chance(self)
    }

    fn initial(&self, rng: &mut dyn RngCore) -> Value {
        self.encode_state(&self.initial_with(rng))
    }

    fn side_to_move(&self, state: &Value) -> Result<Side, DecodeError> {
        Ok(Game::side_to_move(self, &self.decode_state(state)?))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, DecodeError> {
        let s = self.decode_state(state)?;
        Ok(Game::legal_moves(self, &s)
            .iter()
            .map(|m| self.encode_move(&s, m))
            .collect())
    }

    fn described_moves(&self, state: &Value) -> Result<Vec<(Value, String)>, DecodeError> {
        let s = self.decode_state(state)?;
        let mut out: Vec<(Value, String)> = Game::legal_moves(self, &s)
            .iter()
            .map(|m| (self.encode_move(&s, m), self.describe_move(&s, m)))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(out)
    }

    fn play(&self, state: &Value, mv: &Value, rng: &mut dyn RngCore) -> Result<Played, MoveError> {
        let s = self.decode_state(state).map_err(MoveError::Malformed)?;
        if Game::outcome(self, &s).is_some() {
            return Err(MoveError::GameOver);
        }
        let m = self.decode_move(mv).map_err(MoveError::Malformed)?;
        if !Game::legal_moves(self, &s).contains(&m) {
            return Err(MoveError::Illegal(self.describe_move(&s, &m)));
        }
        let notation = self.describe_move(&s, &m);
        let next = self.apply_with(&s, &m, rng);
        Ok(Played {
            mv: self.encode_move(&s, &m),
            state: self.encode_state(&next),
            notation,
        })
    }

    fn outcome(&self, state: &Value) -> Result<Option<Outcome>, DecodeError> {
        Ok(Game::outcome(self, &self.decode_state(state)?))
    }

    fn render(&self, state: &Value) -> Result<String, DecodeError> {
        Ok(Game::render(self, &self.decode_state(state)?))
    }
}

/// Look up a game by its id.
pub fn by_id(id: &str) -> Option<Arc<dyn DynGame>> {
    match id {
        epaminondas::ID => Some(Arc::new(epaminondas::Epaminondas)),
        gess::ID => Some(Arc::new(gess::Gess)),
        one_more_tri::ID => Some(Arc::new(one_more_tri::OneMoreTri)),
        go::ID => Some(Arc::new(go::Go)),
        pig::ID => Some(Arc::new(pig::Pig)),
        _ => None,
    }
}

/// Every game this build knows.
pub fn all_ids() -> &'static [&'static str] {
    &[epaminondas::ID, gess::ID, one_more_tri::ID, go::ID, pig::ID]
}
