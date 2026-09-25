//! A human in a web browser.
//!
//! One port serves both sides: the protocol endpoints the tournament calls, and a page at `/`
//! where a person answers. A `/move` request is held open until they choose a move on that page,
//! resign, or the tournament gives up on it (a timeout closes the connection, and the position
//! disappears from the page).
//!
//! Like any client it keeps no game state: each position on the page is exactly the one the
//! request carried. The page remembers the position after each of your moves, per game, purely
//! to highlight what the opponent changed since.

use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tdg_games::DynGame;
use tdg_proto::{
    Budget, Context, ErrorResponse, Identify, MoveInfo, MoveRequest, MoveResponse, Outcome,
    ProtocolVersion,
};
use tokio::sync::{oneshot, watch};

/// How long `GET /api/pending` waits for something to change before answering anyway.
const LONG_POLL: Duration = Duration::from_secs(25);

#[derive(Debug, Clone)]
pub struct Options {
    pub name: String,
    /// Positions that may wait on the page at once; advertised as `max_concurrency`.
    pub concurrency: u32,
}

/// What the person decided.
enum Reply {
    Move(Value),
    Resign,
}

struct Waiting {
    view: PendingView,
    game: Arc<dyn DynGame>,
    received: Instant,
    reply: oneshot::Sender<Reply>,
}

/// A position waiting for an answer, as the page sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingView {
    pub id: u64,
    pub game: String,
    pub state: Value,
    pub budget: Budget,
    pub context: Context,
    pub moves: Vec<LegalMove>,
    /// The game's own text diagram, for games the page has no board for.
    pub text: String,
    /// How long it has been waiting, as of this snapshot.
    #[serde(default)]
    pub waited_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalMove {
    #[serde(rename = "move")]
    pub mv: Value,
    pub notation: String,
}

/// Response to `GET /api/pending`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Changes whenever the set of waiting positions does; pass it back as `since`.
    pub version: u64,
    pub name: String,
    pub pending: Vec<PendingView>,
}

/// Body of `POST /api/answer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerRequest {
    pub id: u64,
    #[serde(default, rename = "move")]
    pub mv: Option<Value>,
    #[serde(default)]
    pub resign: bool,
}

/// Response to `POST /api/answer` for a move: where it leads, so the page can show it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answered {
    pub notation: String,
    pub state: Value,
    pub text: String,
    pub outcome: Option<Outcome>,
}

struct Desk {
    opts: Options,
    inner: Mutex<Inner>,
    changed: watch::Sender<u64>,
}

#[derive(Default)]
struct Inner {
    next_id: u64,
    waiting: BTreeMap<u64, Waiting>,
}

impl Desk {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn bump(&self) {
        self.changed.send_modify(|v| *v += 1);
    }

    fn snapshot(&self) -> Snapshot {
        let version = *self.changed.borrow();
        let pending = self
            .lock()
            .waiting
            .values()
            .map(|w| PendingView {
                waited_ms: w.received.elapsed().as_millis() as u64,
                ..w.view.clone()
            })
            .collect();
        Snapshot {
            version,
            name: self.opts.name.clone(),
            pending,
        }
    }
}

/// Takes a position off the page when its request ends, however it ends: answered, or dropped
/// because the tournament closed the connection.
struct Withdraw<'a> {
    desk: &'a Desk,
    id: u64,
}

impl Drop for Withdraw<'_> {
    fn drop(&mut self) {
        if self.desk.lock().waiting.remove(&self.id).is_some() {
            self.desk.bump();
        }
    }
}

pub fn router(opts: Options) -> Router {
    let desk = Arc::new(Desk {
        opts,
        inner: Mutex::default(),
        changed: watch::channel(0).0,
    });
    Router::new()
        .route("/identify", get(identify))
        .route("/health", get(|| async { "ok" }))
        .route("/move", post(choose))
        .route("/api/pending", get(pending))
        .route("/api/answer", post(answer))
        .route(
            "/",
            get(|| asset("text/html", include_str!("ui/index.html"))),
        )
        .route(
            "/app.js",
            get(|| asset("text/javascript", include_str!("ui/app.js"))),
        )
        .route(
            "/style.css",
            get(|| asset("text/css", include_str!("ui/style.css"))),
        )
        .with_state(desk)
}

async fn asset(kind: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, kind),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(ErrorResponse::new(msg))).into_response()
}

async fn identify(State(desk): State<Arc<Desk>>) -> Json<Identify> {
    Json(Identify {
        protocol: ProtocolVersion::CURRENT,
        name: desk.opts.name.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        games: tdg_games::all_ids().iter().map(|s| s.to_string()).collect(),
        max_concurrency: desk.opts.concurrency,
    })
}

async fn choose(State(desk): State<Arc<Desk>>, body: axum::body::Bytes) -> Response {
    let req: MoveRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad request: {e}")),
    };
    let Some(game) = tdg_games::by_id(&req.game) else {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("unsupported game: {}", req.game),
        );
    };
    let (moves, text) = match (game.described_moves(&req.state), game.render(&req.state)) {
        (Ok(m), Ok(t)) => (m, t),
        (Err(e), _) | (_, Err(e)) => return error(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()),
    };
    let started = Instant::now();
    let reply = match moves.as_slice() {
        [] => return error(StatusCode::UNPROCESSABLE_ENTITY, "no legal moves"),
        // Nothing to decide.
        [(only, _)] => Reply::Move(only.clone()),
        _ => {
            let (tx, rx) = oneshot::channel();
            let id = {
                let mut inner = desk.lock();
                inner.next_id += 1;
                let id = inner.next_id;
                let view = PendingView {
                    id,
                    game: req.game.clone(),
                    state: req.state.clone(),
                    budget: req.budget,
                    context: req.context.clone(),
                    moves: moves
                        .into_iter()
                        .map(|(mv, notation)| LegalMove { mv, notation })
                        .collect(),
                    text,
                    waited_ms: 0,
                };
                inner.waiting.insert(
                    id,
                    Waiting {
                        view,
                        game,
                        received: started,
                        reply: tx,
                    },
                );
                id
            };
            desk.bump();
            let _withdraw = Withdraw { desk: &desk, id };
            match rx.await {
                Ok(reply) => reply,
                Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "withdrawn"),
            }
        }
    };
    match reply {
        Reply::Move(mv) => Json(MoveResponse {
            request_id: req.request_id,
            mv,
            info: Some(MoveInfo {
                elapsed_ms: Some(started.elapsed().as_millis() as u64),
                ..MoveInfo::default()
            }),
        })
        .into_response(),
        Reply::Resign => error(StatusCode::UNPROCESSABLE_ENTITY, "resigned"),
    }
}

#[derive(Deserialize)]
struct Since {
    since: Option<u64>,
}

/// The waiting positions. With `since`, waits until the set differs from that version (or a
/// while passes), so the page can poll in a loop without hammering.
async fn pending(State(desk): State<Arc<Desk>>, Query(q): Query<Since>) -> Json<Snapshot> {
    if let Some(since) = q.since {
        let mut rx = desk.changed.subscribe();
        let _ = tokio::time::timeout(LONG_POLL, rx.wait_for(|v| *v != since)).await;
    }
    Json(desk.snapshot())
}

async fn answer(State(desk): State<Arc<Desk>>, Json(a): Json<AnswerRequest>) -> Response {
    let gone = || {
        error(
            StatusCode::GONE,
            "that position is no longer waiting: the tournament stopped waiting for it",
        )
    };
    let mut inner = desk.lock();
    let Some(w) = inner.waiting.get(&a.id) else {
        return gone();
    };
    let (reply, answered) = if a.resign {
        (Reply::Resign, None)
    } else {
        let Some(mv) = &a.mv else {
            return error(StatusCode::BAD_REQUEST, "send a move or resign");
        };
        let played = match w.game.play(&w.view.state, mv, &mut rand::rng()) {
            Ok(p) => p,
            Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
        };
        // Where a move of chance leads is up to the server's dice, not the ones rolled here, so
        // the page is shown the position the move was played in until the next one arrives.
        let answered = if w.game.has_chance() {
            Answered {
                notation: played.notation,
                text: w.view.text.clone(),
                outcome: None,
                state: w.view.state.clone(),
            }
        } else {
            Answered {
                notation: played.notation,
                text: w.game.render(&played.state).unwrap_or_default(),
                outcome: w.game.outcome(&played.state).ok().flatten(),
                state: played.state,
            }
        };
        (Reply::Move(played.mv), Some(answered))
    };
    let w = inner.waiting.remove(&a.id).expect("looked up above");
    drop(inner);
    desk.bump();
    if w.reply.send(reply).is_err() {
        return gone();
    }
    match answered {
        Some(a) => Json(a).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}
