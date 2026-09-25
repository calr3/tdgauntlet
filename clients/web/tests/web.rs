//! tdg-web over real sockets: the tournament's side (`/move`) and the page's side (`/api/...`)
//! against one another.

use serde_json::{json, Value};
use std::time::Duration;
use tdg_games::DynGame;
use tdg_proto::{Identify, MoveResponse};
use tdg_web::{router, Answered, Options, Snapshot};

async fn start(concurrency: u32) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(Options {
        name: "someone".into(),
        concurrency,
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{addr}")
}

fn request(id: &str, state: Value) -> Value {
    json!({
        "protocol": 1, "request_id": id, "game": "epaminondas/1", "state": state,
        "budget": {"ms": 5000},
        "context": {"match_id": "m1", "game_id": "m1-a", "ply": 0, "side": 0}
    })
}

fn start_position() -> Value {
    tdg_games::epaminondas::Epaminondas.initial(&mut rand::rng())
}

/// Posts to `/move` in the background, as the tournament would.
fn ask(base: &str, body: Value) -> tokio::task::JoinHandle<reqwest::Response> {
    let url = format!("{base}/move");
    tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .unwrap()
    })
}

async fn snapshot(base: &str) -> Snapshot {
    reqwest::get(format!("{base}/api/pending"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Waits until exactly `n` positions are on the page.
async fn until_pending(base: &str, n: usize) -> Snapshot {
    for _ in 0..100 {
        let s = snapshot(base).await;
        if s.pending.len() == n {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("never saw {n} waiting positions");
}

async fn answer(base: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}/api/answer"))
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn identifies_and_serves_the_page() {
    let base = start(3).await;
    let id: Identify = reqwest::get(format!("{base}/identify"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!((id.name.as_str(), id.max_concurrency), ("someone", 3));
    assert!(id.plays("epaminondas/1"));
    assert!(reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .status()
        .is_success());

    let page = reqwest::get(format!("{base}/")).await.unwrap();
    assert!(page.text().await.unwrap().contains("app.js"));
    for (path, kind) in [("/app.js", "text/javascript"), ("/style.css", "text/css")] {
        let r = reqwest::get(format!("{base}{path}")).await.unwrap();
        assert!(r.status().is_success(), "{path}");
        assert_eq!(r.headers()["content-type"], kind);
    }
}

#[tokio::test]
async fn a_move_chosen_on_the_page_answers_the_request() {
    let base = start(1).await;
    let reply = ask(&base, request("r-1", start_position()));
    let snap = until_pending(&base, 1).await;
    let view = &snap.pending[0];
    assert_eq!(view.context.game_id, "m1-a");
    assert_eq!(view.budget.ms, Some(5000));
    assert!(view.text.contains("White to move"));
    let d2d3 = view
        .moves
        .iter()
        .find(|m| m.notation == "d2-d3")
        .expect("d2-d3 is offered")
        .mv
        .clone();

    // An illegal move is refused and leaves the position waiting.
    let r = answer(
        &base,
        json!({"id": view.id, "move": {"lead": 0, "rear": 0, "dest": 14}}),
    )
    .await;
    assert_eq!(r.status(), 400);
    assert_eq!(snapshot(&base).await.pending.len(), 1);

    let r = answer(&base, json!({"id": view.id, "move": d2d3})).await;
    assert!(r.status().is_success());
    let played: Answered = r.json().await.unwrap();
    assert_eq!(played.notation, "d2-d3");
    assert_eq!(played.state["to_move"], 1);
    assert!(played.outcome.is_none());

    let resp = reply.await.unwrap();
    assert!(resp.status().is_success());
    let body: MoveResponse = resp.json().await.unwrap();
    assert_eq!(body.request_id, "r-1");
    assert_eq!(body.mv["san"], "d2-d3");
    assert!(body.info.unwrap().elapsed_ms.is_some());
    assert!(snapshot(&base).await.pending.is_empty());

    // Answering twice finds nothing waiting.
    let r = answer(&base, json!({"id": view.id, "move": d2d3})).await;
    assert_eq!(r.status(), 410);
}

#[tokio::test]
async fn resigning_is_an_error_response() {
    let base = start(1).await;
    let reply = ask(&base, request("r-2", start_position()));
    let id = until_pending(&base, 1).await.pending[0].id;
    let r = answer(&base, json!({"id": id, "resign": true})).await;
    assert!(r.status().is_success());
    let resp = reply.await.unwrap();
    assert_eq!(resp.status(), 422);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "resigned");
}

#[tokio::test]
async fn a_request_the_tournament_abandons_leaves_the_page() {
    let base = start(1).await;
    let url = format!("{base}/move");
    let gave_up = tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .json(&request("r-3", start_position()))
            .timeout(Duration::from_millis(300))
            .send()
            .await
    });
    let id = until_pending(&base, 1).await.pending[0].id;
    assert!(gave_up.await.unwrap().is_err(), "the request timed out");
    until_pending(&base, 0).await;
    let r = answer(&base, json!({"id": id, "resign": true})).await;
    assert_eq!(r.status(), 410);
}

#[tokio::test]
async fn several_positions_wait_side_by_side() {
    let base = start(2).await;
    let first = ask(&base, request("r-a", start_position()));
    until_pending(&base, 1).await;
    let mut other = request("r-b", start_position());
    other["context"]["game_id"] = json!("m1-b");
    let second = ask(&base, other);
    let snap = until_pending(&base, 2).await;
    let games: Vec<&str> = snap
        .pending
        .iter()
        .map(|v| v.context.game_id.as_str())
        .collect();
    assert_eq!(games, ["m1-a", "m1-b"], "oldest first");

    // Answer them out of order: each reply goes to its own request.
    for (view, want) in snap.pending.iter().rev().zip(["m1-b", "m1-a"]) {
        assert_eq!(view.context.game_id, want);
        let mv = view.moves[0].mv.clone();
        assert!(answer(&base, json!({"id": view.id, "move": mv}))
            .await
            .status()
            .is_success());
    }
    let a: MoveResponse = first.await.unwrap().json().await.unwrap();
    let b: MoveResponse = second.await.unwrap().json().await.unwrap();
    assert_eq!(
        (a.request_id.as_str(), b.request_id.as_str()),
        ("r-a", "r-b")
    );
}

#[tokio::test]
async fn the_page_can_wait_for_a_change() {
    let base = start(1).await;
    let version = snapshot(&base).await.version;
    let url = format!("{base}/api/pending?since={version}");
    let waiting = tokio::spawn(async move {
        reqwest::get(url)
            .await
            .unwrap()
            .json::<Snapshot>()
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!waiting.is_finished(), "nothing has changed yet");
    let _reply = ask(&base, request("r-4", start_position()));
    let snap = tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .expect("woken by the new position")
        .unwrap();
    assert_ne!(snap.version, version);
    assert_eq!(snap.pending.len(), 1);
}

#[tokio::test]
async fn a_forced_move_is_played_without_asking() {
    // White's lone piece on a3 is boxed in by black on a2, b2, b3 and a4: only a3-b4 is legal.
    let mut board = vec![b'.'; 168];
    board[28] = b'W';
    for s in [14, 15, 29, 42] {
        board[s] = b'B';
    }
    let state = json!({
        "board": String::from_utf8(board).unwrap(), "to_move": 0, "moves": 40,
        "quiet_moves": 3, "last_capturer": 1
    });
    let base = start(1).await;
    let resp = ask(&base, request("r-5", state)).await.unwrap();
    assert!(resp.status().is_success());
    let body: MoveResponse = resp.json().await.unwrap();
    assert_eq!(body.mv["san"], "a3-b4");
    assert!(snapshot(&base).await.pending.is_empty());
}

#[tokio::test]
async fn refuses_what_it_cannot_play() {
    let base = start(1).await;
    let c = reqwest::Client::new();
    let mut chess = request("r-6", start_position());
    chess["game"] = json!("chess/1");
    let r = c
        .post(format!("{base}/move"))
        .json(&chess)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 422);
    let r = c
        .post(format!("{base}/move"))
        .json(&json!({"protocol": 2}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = c
        .post(format!("{base}/move"))
        .json(&request("r-7", json!({"board": "nonsense"})))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 422);
}

#[tokio::test]
async fn a_move_of_chance_shows_the_position_it_was_played_in() {
    // The roll that comes with it is the server's to make, not the page's.
    let state = json!({"totals": [42, 17], "turn_total": 9, "last_roll": 5, "to_move": 0});
    let mut req = request("r-pig", state.clone());
    req["game"] = json!("pig/1");
    let base = start(1).await;
    let reply = ask(&base, req);
    let view = until_pending(&base, 1).await.pending[0].clone();
    assert!(
        view.text.contains("rolled 5, turn total 9"),
        "{}",
        view.text
    );
    let roll = view.moves.iter().find(|m| m.notation == "roll").unwrap();
    let r = answer(&base, json!({"id": view.id, "move": roll.mv})).await;
    let played: Answered = r.json().await.unwrap();
    assert_eq!(played.notation, "roll");
    assert_eq!(played.state, state);
    assert_eq!(played.text, view.text);
    assert!(played.outcome.is_none());
    let body: MoveResponse = reply.await.unwrap().json().await.unwrap();
    assert_eq!(body.mv["action"], "roll");
}
