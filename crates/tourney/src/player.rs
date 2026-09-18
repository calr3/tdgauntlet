//! Talking to clients.

use crate::config::PlayerConfig;
use anyhow::{bail, Context as _, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tdg_proto::{ErrorResponse, Identify, MoveRequest, MoveResponse};

/// Something that can be asked for moves. Real clients are [`HttpPlayer`]s; tests substitute
/// in-process ones.
#[async_trait]
pub trait Player: Send + Sync {
    async fn identify(&self) -> Result<Identify, PlayerError>;
    async fn choose(
        &self,
        req: &MoveRequest,
        timeout: Duration,
    ) -> Result<MoveResponse, PlayerError>;
}

/// Why a client did not produce a usable response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum PlayerError {
    #[error("timed out")]
    Timeout,
    #[error("could not reach the client: {0}")]
    Transport(String),
    #[error("status {status}: {message}")]
    Status { status: u16, message: String },
    #[error("unreadable response: {0}")]
    Malformed(String),
}

pub struct HttpPlayer {
    base: String,
    client: reqwest::Client,
}

impl HttpPlayer {
    pub fn new(base: &str) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    async fn send<T: for<'de> Deserialize<'de>>(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<T, PlayerError> {
        let resp = req.send().await.map_err(classify)?;
        let status = resp.status();
        let body = resp.bytes().await.map_err(classify)?;
        if !status.is_success() {
            let message = serde_json::from_slice::<ErrorResponse>(&body)
                .map(|e| e.error)
                .unwrap_or_else(|_| String::from_utf8_lossy(&body).chars().take(200).collect());
            return Err(PlayerError::Status {
                status: status.as_u16(),
                message,
            });
        }
        serde_json::from_slice(&body).map_err(|e| PlayerError::Malformed(e.to_string()))
    }

    pub async fn healthy(&self) -> bool {
        let req = self
            .client
            .get(format!("{}/health", self.base))
            .timeout(Duration::from_secs(2));
        matches!(req.send().await, Ok(r) if r.status().is_success())
    }
}

fn classify(e: reqwest::Error) -> PlayerError {
    if e.is_timeout() {
        PlayerError::Timeout
    } else if e.is_decode() {
        PlayerError::Malformed(e.to_string())
    } else {
        PlayerError::Transport(e.to_string())
    }
}

#[async_trait]
impl Player for HttpPlayer {
    async fn identify(&self) -> Result<Identify, PlayerError> {
        let req = self
            .client
            .get(format!("{}/identify", self.base))
            .timeout(Duration::from_secs(10));
        self.send(req).await
    }

    async fn choose(
        &self,
        req: &MoveRequest,
        timeout: Duration,
    ) -> Result<MoveResponse, PlayerError> {
        let http = self
            .client
            .post(format!("{}/move", self.base))
            .json(req)
            .timeout(timeout);
        self.send(http).await
    }
}

/// A client the server started itself, and stops when this is dropped.
pub struct SpawnedPlayer {
    http: HttpPlayer,
    _child: tokio::process::Child,
}

impl SpawnedPlayer {
    pub async fn start(cfg: &PlayerConfig, cmd: &str, dir: &std::path::Path) -> Result<Self> {
        let port = free_port()?;
        let line = cmd.replace("{port}", &port.to_string());
        let mut child = shell(&line)
            .current_dir(dir)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {:?}: {line}", cfg.name))?;
        let http = HttpPlayer::new(&format!("http://127.0.0.1:{port}"));
        let deadline = Instant::now() + Duration::from_secs(cfg.startup_timeout_s);
        loop {
            if http.healthy().await {
                return Ok(Self {
                    http,
                    _child: child,
                });
            }
            if let Some(status) = child.try_wait()? {
                bail!(
                    "{:?} exited with {status} before becoming healthy: {line}",
                    cfg.name
                );
            }
            if Instant::now() > deadline {
                bail!(
                    "{:?} was not healthy after {}s: {line}",
                    cfg.name,
                    cfg.startup_timeout_s
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

#[cfg(unix)]
fn shell(line: &str) -> tokio::process::Command {
    let mut c = tokio::process::Command::new("sh");
    c.arg("-c").arg(format!("exec {line}"));
    c
}

#[cfg(windows)]
fn shell(line: &str) -> tokio::process::Command {
    let mut c = tokio::process::Command::new("cmd");
    c.arg("/C").arg(line);
    c
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

#[async_trait]
impl Player for SpawnedPlayer {
    async fn identify(&self) -> Result<Identify, PlayerError> {
        self.http.identify().await
    }

    async fn choose(
        &self,
        req: &MoveRequest,
        timeout: Duration,
    ) -> Result<MoveResponse, PlayerError> {
        self.http.choose(req, timeout).await
    }
}
