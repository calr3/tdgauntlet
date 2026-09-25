//! Playing one game: ask the side to move, check the answer, apply it, repeat.

use crate::config::ErrorPolicy;
use crate::openings::Opening;
use crate::player::{Player, PlayerError};
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tdg_games::{DynGame, MoveError};
use tdg_proto::{Budget, Context, MoveInfo, MoveRequest, Outcome, ProtocolVersion, Side};
use tokio::sync::Semaphore;

/// One entrant, as the game loop needs it.
pub struct Seat {
    pub name: String,
    pub player: Arc<dyn Player>,
    /// Caps this client's requests in flight across every game it is playing.
    pub limit: Arc<Semaphore>,
    pub budget: Budget,
    pub timeout: Duration,
    pub on_error: ErrorPolicy,
}

pub struct GameSpec {
    pub id: String,
    pub match_id: String,
    pub opening: Arc<Opening>,
    /// Indexed by side: `seats[0]` moves first.
    pub seats: [Arc<Seat>; 2],
    pub max_plies: u32,
    /// Seeds the server's own random moves under the `random` error policy, and the dice of a
    /// game of chance.
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveRecord {
    pub side: Side,
    pub san: String,
    #[serde(rename = "move")]
    pub mv: Value,
    /// The server chose this move itself because the client failed (`on_error = "random"`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_server: bool,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<MoveInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Termination {
    /// The rules decided it.
    Rules,
    /// A client failed and its policy was to lose.
    Forfeit { side: Side },
    /// The game hit the server's `max_plies` safety net and was scored a draw.
    PlyCap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Incident {
    /// Plies played in this game when it happened.
    pub ply: u32,
    pub side: Side,
    pub player: String,
    pub error: IncidentKind,
    /// What the server did about it.
    pub action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum IncidentKind {
    Client(PlayerError),
    WrongRequestId(String),
    MalformedMove(String),
    IllegalMove(String),
    PlyCap(u32),
}

impl std::fmt::Display for IncidentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncidentKind::Client(e) => write!(f, "{e}"),
            IncidentKind::WrongRequestId(id) => write!(f, "answered request {id:?}"),
            IncidentKind::MalformedMove(m) => write!(f, "malformed move: {m}"),
            IncidentKind::IllegalMove(m) => write!(f, "illegal move {m}"),
            IncidentKind::PlyCap(n) => write!(f, "still going after {n} plies"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameRecord {
    pub id: String,
    pub match_id: String,
    pub opening: String,
    /// Names, indexed by side.
    pub players: [String; 2],
    pub moves: Vec<MoveRecord>,
    pub outcome: Outcome,
    pub termination: Termination,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incidents: Vec<Incident>,
    pub final_state: Value,
}

/// A stable 64-bit hash of a string, for deriving per-game seeds.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

pub async fn play_game(game: Arc<dyn DynGame>, spec: GameSpec) -> GameRecord {
    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed ^ fnv(&spec.id));
    // Seeded by the minimatch, not the game, so both of its games roll the same dice in turn.
    let mut dice = ChaCha8Rng::seed_from_u64(spec.seed ^ fnv(&spec.match_id));
    let mut state = spec.opening.state.clone();
    let mut moves: Vec<MoveRecord> = Vec::new();
    let mut incidents: Vec<Incident> = Vec::new();
    let opening_plies = spec.opening.moves.len() as u32;

    let (outcome, termination) = 'game: loop {
        if let Some(outcome) = game
            .outcome(&state)
            .expect("the server's own states decode")
        {
            break (outcome, Termination::Rules);
        }
        let ply = moves.len() as u32;
        let side = game
            .side_to_move(&state)
            .expect("the server's own states decode");
        let seat = &spec.seats[side.index()];
        if ply >= spec.max_plies {
            incidents.push(Incident {
                ply,
                side,
                player: seat.name.clone(),
                error: IncidentKind::PlyCap(ply),
                action: "scored as a draw".into(),
            });
            break (Outcome::Draw, Termination::PlyCap);
        }

        let mut attempt = 0;
        loop {
            let req = MoveRequest {
                protocol: ProtocolVersion::CURRENT,
                request_id: format!("{}-p{ply}-a{attempt}", spec.id),
                game: game.id().to_string(),
                state: state.clone(),
                budget: seat.budget,
                context: Context {
                    match_id: spec.match_id.clone(),
                    game_id: spec.id.clone(),
                    ply: opening_plies + ply,
                    side,
                },
            };
            // Timed from when the client actually has the request: waiting for a free slot on a
            // client busy with other games is not this move's thinking time.
            let (answer, elapsed_ms) = {
                let _permit = seat.limit.acquire().await.expect("never closed");
                let started = Instant::now();
                let answer = seat.player.choose(&req, seat.timeout).await;
                (answer, started.elapsed().as_millis() as u64)
            };

            let result = answer.map_err(IncidentKind::Client).and_then(|resp| {
                if resp.request_id != req.request_id {
                    return Err(IncidentKind::WrongRequestId(resp.request_id));
                }
                let played = game
                    .play(&state, &resp.mv, &mut dice)
                    .map_err(|e| match e {
                        MoveError::Malformed(d) => IncidentKind::MalformedMove(d.to_string()),
                        MoveError::Illegal(m) => IncidentKind::IllegalMove(m),
                        MoveError::GameOver => unreachable!("checked before asking"),
                    })?;
                Ok((played, resp.info))
            });

            let error = match result {
                Ok((played, info)) => {
                    moves.push(MoveRecord {
                        side,
                        san: played.notation,
                        mv: played.mv,
                        by_server: false,
                        elapsed_ms,
                        info,
                    });
                    state = played.state;
                    break;
                }
                Err(e) => e,
            };

            let mut incident = |action: &str| {
                incidents.push(Incident {
                    ply,
                    side,
                    player: seat.name.clone(),
                    error: error.clone(),
                    action: action.into(),
                })
            };
            match seat.on_error {
                ErrorPolicy::Retry(n) if attempt < n => {
                    incident("asked again");
                    attempt += 1;
                }
                ErrorPolicy::Random => {
                    incident("played a random move instead");
                    let legal = game.legal_moves(&state).expect("decodes");
                    let mv = legal.choose(&mut rng).expect("an undecided game has moves");
                    let played = game
                        .play(&state, mv, &mut dice)
                        .expect("a generated move is legal");
                    moves.push(MoveRecord {
                        side,
                        san: played.notation,
                        mv: played.mv,
                        by_server: true,
                        elapsed_ms,
                        info: None,
                    });
                    state = played.state;
                    break;
                }
                ErrorPolicy::Forfeit | ErrorPolicy::Retry(_) => {
                    incident("forfeited the game");
                    break 'game (Outcome::win(side.other()), Termination::Forfeit { side });
                }
            }
        }
    };

    GameRecord {
        id: spec.id,
        match_id: spec.match_id,
        opening: spec.opening.id.clone(),
        players: [spec.seats[0].name.clone(), spec.seats[1].name.clone()],
        moves,
        outcome,
        termination,
        incidents,
        final_state: state,
    }
}
