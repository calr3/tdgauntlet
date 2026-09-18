//! Wire types for the tdgauntlet client protocol, version 1.
//!
//! The envelope defined here is fixed; the `state` and `move` payloads inside it are opaque JSON
//! owned by the game, so this crate has no notion of any particular game. See `PROTOCOL.md`.

mod side;
mod version;

pub use side::{Outcome, Side};
pub use version::{ProtocolVersion, PROTOCOL_VERSION};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response to `GET /identify`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identify {
    pub protocol: ProtocolVersion,
    /// How the client calls itself. A tournament config may override it.
    pub name: String,
    /// Free-form; recorded in the result file so a run can be traced back to a build.
    #[serde(default)]
    pub version: String,
    /// Game ids this client can play, e.g. `epaminondas/1`.
    pub games: Vec<String>,
    /// Most simultaneous `/move` requests the server may have in flight against this client.
    #[serde(default = "default_concurrency")]
    pub max_concurrency: u32,
}

fn default_concurrency() -> u32 {
    1
}

impl Identify {
    /// Whether this client advertises support for `game`.
    pub fn plays(&self, game: &str) -> bool {
        self.games.iter().any(|g| g == game)
    }
}

/// How much thinking the client is being granted. Every field is optional: a client honours
/// whichever it understands and ignores the rest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Wall-clock milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ms: Option<u64>,
    /// Search nodes, for engines that count them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes: Option<u64>,
    /// MCTS simulations, for engines that count those instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sims: Option<u64>,
}

impl Budget {
    pub fn from_ms(ms: u64) -> Self {
        Self {
            ms: Some(ms),
            ..Self::default()
        }
    }
}

/// Where this position sits in the tournament. Informational only — a conformant client never
/// needs it to choose a move, and must not carry state between requests that share a `game_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub match_id: String,
    pub game_id: String,
    /// Completed plies before this position.
    pub ply: u32,
    /// Which side the client is playing, always the side to move in `state`.
    pub side: Side,
}

/// Request body of `POST /move`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveRequest {
    pub protocol: ProtocolVersion,
    pub request_id: String,
    pub game: String,
    /// Opaque, game-defined. Always a complete position.
    pub state: Value,
    #[serde(default)]
    pub budget: Budget,
    pub context: Context,
}

/// Response body of `POST /move`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveResponse {
    /// Must equal the request's `request_id`.
    pub request_id: String,
    /// Opaque, game-defined.
    #[serde(rename = "move")]
    pub mv: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<MoveInfo>,
}

/// Optional self-reporting from the client. Recorded, never acted on.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MoveInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pv: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    /// Anything else the client wants recorded.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Body a client returns alongside a non-2xx status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

impl ErrorResponse {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { error: msg.into() }
    }
}

#[cfg(test)]
mod tests;
