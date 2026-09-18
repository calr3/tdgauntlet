//! Serve a tdgauntlet client.
//!
//! Implement [`Engine`] and call [`serve`]. The kit handles the three endpoints, rejects requests
//! for games the engine does not play, echoes the request id, fills in `elapsed_ms`, and runs
//! each move on a blocking thread so a CPU-bound search does not stall the server.
//!
//! An engine is shared across every request and may be asked about many games at once: it can
//! hold anything keyed by position, but nothing about "the current game".

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tdg_proto::{ErrorResponse, Identify, MoveInfo, MoveRequest, MoveResponse, ProtocolVersion};

/// What a client says about itself.
#[derive(Debug, Clone)]
pub struct Info {
    pub name: String,
    pub version: String,
    pub games: Vec<String>,
    pub max_concurrency: u32,
}

/// A chosen move and, optionally, what the engine wants recorded about it.
pub struct Answer {
    pub mv: serde_json::Value,
    pub info: Option<MoveInfo>,
}

impl Answer {
    pub fn new(mv: serde_json::Value) -> Self {
        Self { mv, info: None }
    }
}

pub trait Engine: Send + Sync + 'static {
    fn info(&self) -> Info;
    /// Choose a move for `req.state`. Called concurrently, on blocking threads.
    fn choose(&self, req: &MoveRequest) -> Result<Answer, String>;
}

pub fn router(engine: Arc<dyn Engine>) -> Router {
    Router::new()
        .route("/identify", get(identify))
        .route("/health", get(|| async { "ok" }))
        .route("/move", post(choose))
        .with_state(engine)
}

/// Serve `engine` on `addr` until the process is stopped.
pub async fn serve(engine: impl Engine, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let info = engine.info();
    eprintln!(
        "{} {} listening on http://{}",
        info.name,
        info.version,
        listener.local_addr()?
    );
    axum::serve(listener, router(Arc::new(engine))).await?;
    Ok(())
}

async fn identify(State(engine): State<Arc<dyn Engine>>) -> Json<Identify> {
    let info = engine.info();
    Json(Identify {
        protocol: ProtocolVersion::CURRENT,
        name: info.name,
        version: info.version,
        games: info.games,
        max_concurrency: info.max_concurrency,
    })
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(ErrorResponse::new(msg))).into_response()
}

async fn choose(State(engine): State<Arc<dyn Engine>>, body: axum::body::Bytes) -> Response {
    let req: MoveRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad request: {e}")),
    };
    if !engine.info().games.contains(&req.game) {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("unsupported game: {}", req.game),
        );
    }
    let started = Instant::now();
    let request_id = req.request_id.clone();
    let answer = tokio::task::spawn_blocking(move || engine.choose(&req)).await;
    match answer {
        Ok(Ok(Answer { mv, info })) => {
            let mut info = info.unwrap_or_default();
            info.elapsed_ms
                .get_or_insert(started.elapsed().as_millis() as u64);
            Json(MoveResponse {
                request_id,
                mv,
                info: Some(info),
            })
            .into_response()
        }
        Ok(Err(msg)) => error(StatusCode::UNPROCESSABLE_ENTITY, msg),
        Err(panic) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("engine panicked: {panic}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Echo;

    impl Engine for Echo {
        fn info(&self) -> Info {
            Info {
                name: "echo".into(),
                version: "1".into(),
                games: vec!["g/1".into()],
                max_concurrency: 3,
            }
        }
        fn choose(&self, req: &MoveRequest) -> Result<Answer, String> {
            match req.state["say"].as_str() {
                Some("fail") => Err("cannot".into()),
                Some("panic") => panic!("oh no"),
                _ => Ok(Answer::new(req.state.clone())),
            }
        }
    }

    async fn start() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(Arc::new(Echo))).await });
        format!("http://{addr}")
    }

    fn request(game: &str, say: &str) -> serde_json::Value {
        json!({
            "protocol": 1, "request_id": "r-17", "game": game, "state": {"say": say},
            "context": {"match_id": "m", "game_id": "g", "ply": 0, "side": 0}
        })
    }

    #[tokio::test]
    async fn serves_the_three_endpoints() {
        let base = start().await;
        let c = reqwest::Client::new();
        let id: Identify = c
            .get(format!("{base}/identify"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!((id.name.as_str(), id.max_concurrency), ("echo", 3));
        assert!(c
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .status()
            .is_success());

        let resp = c
            .post(format!("{base}/move"))
            .json(&request("g/1", "hi"))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: MoveResponse = resp.json().await.unwrap();
        assert_eq!(body.request_id, "r-17", "the request id is echoed");
        assert_eq!(body.mv, json!({"say": "hi"}));
        assert!(
            body.info.unwrap().elapsed_ms.is_some(),
            "elapsed time is filled in"
        );
    }

    #[tokio::test]
    async fn refuses_what_it_cannot_do() {
        let base = start().await;
        let c = reqwest::Client::new();
        let post = |body: serde_json::Value| c.post(format!("{base}/move")).json(&body).send();

        let r = post(request("chess/1", "hi")).await.unwrap();
        assert_eq!(r.status(), 422);
        let e: ErrorResponse = r.json().await.unwrap();
        assert!(e.error.contains("unsupported game"));

        let r = post(request("g/1", "fail")).await.unwrap();
        assert_eq!(r.status(), 422);

        let r = post(request("g/1", "panic")).await.unwrap();
        assert_eq!(
            r.status(),
            500,
            "a panicking engine is an error response, not a dead server"
        );
        // And the server is still up.
        assert!(post(request("g/1", "hi"))
            .await
            .unwrap()
            .status()
            .is_success());

        let r = post(json!({"protocol": 2})).await.unwrap();
        assert_eq!(r.status(), 400);
    }
}
