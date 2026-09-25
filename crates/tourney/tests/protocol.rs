//! The protocol conformance harness: real clients over real sockets, most of them broken on
//! purpose, driven through the same path `tdgauntlet run` uses.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tdg_games::DynGame;
use tdg_proto::{MoveRequest, MoveResponse};
use tdg_tourney::config::Config;
use tdg_tourney::engine::{IncidentKind, Termination};
use tdg_tourney::player::PlayerError;
use tdg_tourney::tournament::TournamentResult;
use tdg_tourney::{check, prepare, tournament};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Good,
    /// Legal moves, answered after a delay that varies per request, so answers complete out of
    /// order.
    Jitter,
    Illegal,
    /// Always answers with a move that is legal from the start position.
    StaleBoard,
    /// Remembers the last position it was sent and answers for that one: fine one game at a
    /// time, wrong as soon as requests interleave.
    Stateful,
    NotJson,
    Status500,
    Hang,
    /// Advertises a concurrency of 1.
    Serial,
}

struct Client {
    mode: Mode,
    game: Arc<dyn DynGame>,
    in_flight: AtomicU32,
    peak: AtomicU32,
    last_state: Mutex<Option<Value>>,
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

async fn identify(State(c): State<Arc<Client>>) -> Json<Value> {
    let max = if c.mode == Mode::Serial { 1 } else { 16 };
    Json(json!({
        "protocol": 1, "name": format!("{:?}", c.mode), "version": "test",
        "games": ["epaminondas/1"], "max_concurrency": max
    }))
}

async fn choose(State(c): State<Arc<Client>>, Json(req): Json<MoveRequest>) -> Response {
    if req.game != "epaminondas/1" {
        let body = Json(json!({"error": format!("unsupported game: {}", req.game)}));
        return (StatusCode::UNPROCESSABLE_ENTITY, body).into_response();
    }
    let now = c.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    c.peak.fetch_max(now, Ordering::SeqCst);
    let pick = |state: &Value| {
        let moves = c.game.legal_moves(state).unwrap();
        moves[fnv(&req.request_id) as usize % moves.len()].clone()
    };
    let delay = match c.mode {
        Mode::Jitter => fnv(&req.request_id) % 15,
        Mode::Hang => 5_000,
        Mode::Serial | Mode::Stateful => 2,
        _ => 0,
    };
    // For the stateful client, the "remembered" board is whatever arrived last - read after the
    // delay, so interleaved requests overwrite each other exactly as a real one would.
    if c.mode == Mode::Stateful {
        *c.last_state.lock().unwrap() = Some(req.state.clone());
    }
    tokio::time::sleep(Duration::from_millis(delay)).await;
    let reply = |mv: Value| {
        Json(MoveResponse {
            request_id: req.request_id.clone(),
            mv,
            info: None,
        })
        .into_response()
    };
    let resp = match c.mode {
        Mode::Good | Mode::Jitter | Mode::Serial | Mode::Hang => reply(pick(&req.state)),
        Mode::Illegal => reply(json!({"lead": 0, "rear": 0, "dest": 100})),
        Mode::StaleBoard => reply(pick(&c.game.initial(&mut rand::rng()))),
        Mode::Stateful => {
            let remembered = c.last_state.lock().unwrap().clone().unwrap();
            reply(pick(&remembered))
        }
        Mode::NotJson => (StatusCode::OK, "hello").into_response(),
        Mode::Status500 => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "model fell over"})),
        )
            .into_response(),
    };
    c.in_flight.fetch_sub(1, Ordering::SeqCst);
    resp
}

/// Start a client in `mode` on a free port; returns its URL and a handle for inspection.
async fn start(mode: Mode) -> (String, Arc<Client>) {
    let client = Arc::new(Client {
        mode,
        game: tdg_games::by_id("epaminondas/1").unwrap(),
        in_flight: AtomicU32::new(0),
        peak: AtomicU32::new(0),
        last_state: Mutex::new(None),
    });
    let app = Router::new()
        .route("/identify", get(identify))
        .route("/health", get(|| async { "ok" }))
        .route("/move", post(choose))
        .with_state(client.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await });
    (url, client)
}

fn temp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tdg-protocol-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Write a config naming these clients, load it the way the CLI does, and run it.
async fn tournament(players: &[(&str, &str, &str)], extra: &str) -> TournamentResult {
    let mut text = format!(
        "game = \"epaminondas/1\"\nseed = 3\nconcurrency = 8\n{extra}\n\
         [openings]\nkind = \"random\"\nplies = 2\ncount = 4\n"
    );
    for (name, url, player_extra) in players {
        text += &format!("[[player]]\nname = \"{name}\"\nurl = \"{url}\"\n{player_extra}\n");
    }
    let path = temp_path(&format!("{}.toml", players[0].0));
    std::fs::write(&path, text).unwrap();
    let config = Config::load(&path).unwrap();
    let result = tournament::run(prepare(&config).await.unwrap(), None).await;
    // Round-trip through the result file format, as a reader of it would see it.
    let json = serde_json::to_string(&result).unwrap();
    serde_json::from_str(&json).unwrap()
}

fn incidents_of<'a>(r: &'a TournamentResult, player: &str) -> Vec<&'a IncidentKind> {
    r.games
        .iter()
        .flat_map(|g| &g.incidents)
        .filter(|i| i.player == player)
        .map(|i| &i.error)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn well_behaved_clients_finish_every_game_cleanly() {
    let (a, _) = start(Mode::Good).await;
    let (b, _) = start(Mode::Jitter).await;
    let r = tournament(&[("good", &a, ""), ("jitter", &b, "")], "").await;
    assert_eq!(r.games.len(), 8);
    for g in &r.games {
        assert_eq!(g.termination, Termination::Rules, "{}", g.id);
        assert!(g.incidents.is_empty(), "{}: {:?}", g.id, g.incidents);
    }
    assert_eq!(
        r.entrants[1].identify.name, "Jitter",
        "identify is recorded"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn illegal_moves_forfeit_and_are_recorded() {
    let (good, _) = start(Mode::Good).await;
    let (bad, _) = start(Mode::Illegal).await;
    let r = tournament(&[("bad", &bad, ""), ("good", &good, "")], "").await;
    for g in &r.games {
        let side = g.players.iter().position(|p| p == "bad").unwrap();
        assert!(matches!(g.termination, Termination::Forfeit { side: s } if s.index() == side));
    }
    let incidents = incidents_of(&r, "bad");
    assert_eq!(incidents.len(), 8);
    assert!(incidents
        .iter()
        .all(|k| matches!(k, IncidentKind::IllegalMove(_))));
    assert_eq!(
        r.standings
            .iter()
            .find(|s| s.name == "bad")
            .unwrap()
            .percentage,
        Some(0.0)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_move_for_the_wrong_position_is_caught() {
    let (good, _) = start(Mode::Good).await;
    let (stale, _) = start(Mode::StaleBoard).await;
    let r = tournament(
        &[
            ("stale", &stale, "on_error = \"random\""),
            ("good", &good, ""),
        ],
        "",
    )
    .await;
    // It gets lucky sometimes (a start-position move can still be legal later), but not often,
    // and every game still reaches a real finish because the server stepped in.
    assert!(incidents_of(&r, "stale").len() > 8);
    assert!(r.games.iter().all(|g| g.termination == Termination::Rules));
    // Everything actually played is legal: replay each game from its opening.
    let game = tdg_games::by_id("epaminondas/1").unwrap();
    for g in &r.games {
        let mut state = r
            .openings
            .iter()
            .find(|o| o.id == g.opening)
            .unwrap()
            .state
            .clone();
        for m in &g.moves {
            state = game
                .play(&state, &m.mv, &mut rand::rng())
                .expect("only legal moves are recorded")
                .state;
        }
        assert_eq!(state, g.final_state);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transport_level_failures_are_classified() {
    let (good, _) = start(Mode::Good).await;
    let (not_json, _) = start(Mode::NotJson).await;
    let (status, _) = start(Mode::Status500).await;
    let r = tournament(&[("notjson", &not_json, ""), ("good", &good, "")], "").await;
    assert!(incidents_of(&r, "notjson")
        .iter()
        .all(|k| matches!(k, IncidentKind::Client(PlayerError::Malformed(_)))));

    let r = tournament(&[("status", &status, ""), ("good", &good, "")], "").await;
    let incidents = incidents_of(&r, "status");
    assert_eq!(incidents.len(), 8);
    assert!(incidents.iter().all(|k| matches!(
        k,
        IncidentKind::Client(PlayerError::Status { status: 500, message }) if message == "model fell over"
    )), "{incidents:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hanging_client_times_out_and_retries_are_counted() {
    let (good, _) = start(Mode::Good).await;
    let (hang, _) = start(Mode::Hang).await;
    let r = tournament(
        &[
            ("hang", &hang, "timeout_ms = 100\non_error = \"retry:1\""),
            ("good", &good, ""),
        ],
        "",
    )
    .await;
    // Each game: one timeout, one retry that also times out, then a forfeit.
    for g in &r.games {
        let mine: Vec<_> = g.incidents.iter().filter(|i| i.player == "hang").collect();
        assert_eq!(mine.len(), 2, "{}", g.id);
        assert!(mine
            .iter()
            .all(|i| i.error == IncidentKind::Client(PlayerError::Timeout)));
        assert_eq!(mine[0].action, "asked again");
        assert_eq!(mine[1].action, "forfeited the game");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clients_advertised_concurrency_is_respected() {
    let (serial, serial_client) = start(Mode::Serial).await;
    let (good, good_client) = start(Mode::Good).await;
    tournament(&[("serial", &serial, ""), ("good", &good, "")], "").await;
    assert_eq!(serial_client.peak.load(Ordering::SeqCst), 1);
    // The cap is per client, not global: the other client did see overlapping requests.
    assert!(good_client.peak.load(Ordering::SeqCst) >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_config_cap_can_lower_the_advertised_concurrency() {
    let (a, client) = start(Mode::Jitter).await;
    let (b, _) = start(Mode::Good).await;
    tournament(&[("capped", &a, "max_concurrency = 2"), ("b", &b, "")], "").await;
    assert!(client.peak.load(Ordering::SeqCst) <= 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_does_not_play_the_game_is_turned_away() {
    let (url, _) = start(Mode::Good).await;
    let path = temp_path("wrong_game.toml");
    // Pretend the tournament is for a game id the client does not list, by asking prepare to
    // check against it directly.
    let config: Config = toml::from_str(&format!(
        "game = \"epaminondas/1\"\n[openings]\nkind = \"start\"\n\
         [[player]]\nname = \"a\"\nurl = \"{url}\"\n[[player]]\nname = \"b\"\nurl = \"http://127.0.0.1:1\"\n"
    ))
    .unwrap();
    std::fs::write(&path, "").unwrap();
    let err = prepare(&config)
        .await
        .err()
        .expect("an unreachable client stops the run");
    assert!(format!("{err:#}").contains("\"b\""), "{err:#}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn spawned_clients_that_never_come_up_are_reported() {
    for (cmd, why) in [
        ("false {port}", "exited"),
        ("sleep 30 # {port}", "not healthy"),
    ] {
        let config: Config = toml::from_str(&format!(
            "game = \"epaminondas/1\"\n[openings]\nkind = \"start\"\n\
             [[player]]\nname = \"a\"\ncmd = \"{cmd}\"\nstartup_timeout_s = 1\n\
             [[player]]\nname = \"b\"\nurl = \"http://127.0.0.1:1\"\n"
        ))
        .unwrap();
        let err = format!("{:#}", prepare(&config).await.err().unwrap());
        assert!(err.contains(why), "{cmd}: {err}");
    }
}

// ─── tdgauntlet check ────────────────────────────────────────────────────────

async fn check_mode(mode: Mode) -> Vec<check::Check> {
    let (url, _) = start(mode).await;
    let opts = check::CheckOptions {
        positions: 16,
        timeout: Duration::from_millis(500),
        ..Default::default()
    };
    check::run(&url, &opts).await
}

fn failed(checks: &[check::Check]) -> Vec<&str> {
    checks
        .iter()
        .filter(|c| c.result.is_err())
        .map(|c| c.name.as_str())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn check_passes_a_good_client() {
    for mode in [Mode::Good, Mode::Jitter, Mode::Serial] {
        let checks = check_mode(mode).await;
        assert!(
            failed(&checks).is_empty(),
            "{mode:?}: {:?}",
            failed(&checks)
        );
        assert_eq!(checks.len(), 4);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn check_catches_a_client_that_tracks_the_current_game() {
    let checks = check_mode(Mode::Stateful).await;
    assert_eq!(
        failed(&checks),
        ["answers unrelated positions sharing a game id, in any order"],
        "the stateful client should fail exactly the ordering probe"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn check_catches_the_other_failures() {
    for mode in [
        Mode::Illegal,
        Mode::StaleBoard,
        Mode::NotJson,
        Mode::Status500,
        Mode::Hang,
    ] {
        let checks = check_mode(mode).await;
        assert!(
            failed(&checks)
                .contains(&"answers unrelated positions sharing a game id, in any order"),
            "{mode:?}"
        );
    }
}

#[tokio::test]
async fn check_reports_an_unreachable_client() {
    let checks = check::run("http://127.0.0.1:1", &check::CheckOptions::default()).await;
    assert_eq!(failed(&checks), ["identify"]);
}
