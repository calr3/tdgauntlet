//! `tournament.toml`.

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tdg_proto::Budget;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Game id, e.g. `epaminondas/1`.
    pub game: String,
    #[serde(default)]
    pub seed: u64,
    /// Games in flight at once, across all pairings.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Where to write the result file. Relative paths are relative to the config file.
    #[serde(default)]
    pub out: Option<PathBuf>,
    /// A safety net against a game that will not end. A game reaching it is recorded as a draw
    /// and flagged. Epaminondas cannot exceed about 5,600 moves, so the default never fires there.
    #[serde(default = "default_max_plies")]
    pub max_plies: u32,
    pub openings: OpeningSpec,
    #[serde(rename = "player")]
    pub players: Vec<PlayerConfig>,
    /// Pairings to play, by name. Empty means a full round robin.
    #[serde(default)]
    pub pairs: Vec<[String; 2]>,
    /// The directory holding the config file. Relative paths in it, and spawned players' working
    /// directories, are all taken from here, so a config means the same wherever it is run from.
    #[serde(skip, default = "current_dir")]
    pub base_dir: PathBuf,
}

fn current_dir() -> PathBuf {
    PathBuf::from(".")
}

fn default_concurrency() -> usize {
    1
}

fn default_max_plies() -> u32 {
    12_000
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OpeningSpec {
    /// Only the game's normal starting position.
    Start,
    /// `count` distinct positions reached by `plies` uniformly random moves from the start.
    Random { plies: u32, count: usize },
    /// A JSON-lines book. Each line is `{"state": ...}` or `{"moves": [...]}`, optionally with
    /// an `"id"`.
    File { path: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerConfig {
    pub name: String,
    /// Base URL of a client that is already running.
    #[serde(default)]
    pub url: Option<String>,
    /// A command that starts the client, run from the config file's directory. `{port}` is
    /// replaced with a free port; the server waits for `/health`, and kills the process when the
    /// tournament ends.
    #[serde(default)]
    pub cmd: Option<String>,
    #[serde(default)]
    pub budget_ms: Option<u64>,
    #[serde(default)]
    pub budget_nodes: Option<u64>,
    #[serde(default)]
    pub budget_sims: Option<u64>,
    #[serde(default)]
    pub on_error: ErrorPolicy,
    /// How long to wait for a move before treating it as a timeout. Defaults to twice the
    /// millisecond budget plus two seconds, or a minute when there is no millisecond budget.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Cap on concurrent requests, below what the client advertises.
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    /// How long a spawned client may take to become healthy.
    #[serde(default = "default_startup_s")]
    pub startup_timeout_s: u64,
}

fn default_startup_s() -> u64 {
    300
}

impl PlayerConfig {
    pub fn budget(&self) -> Budget {
        Budget {
            ms: self.budget_ms,
            nodes: self.budget_nodes,
            sims: self.budget_sims,
        }
    }

    pub fn timeout(&self) -> Duration {
        Duration::from_millis(match (self.timeout_ms, self.budget_ms) {
            (Some(t), _) => t,
            (None, Some(b)) => 2 * b + 2_000,
            (None, None) => 60_000,
        })
    }
}

/// What to do when a client fails to produce a legal move: an illegal or malformed move, an
/// error status, a broken connection, or a timeout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorPolicy {
    /// Lose the game.
    #[default]
    Forfeit,
    /// The server plays a uniformly random legal move instead and the game continues.
    Random,
    /// Ask again up to this many times, then forfeit.
    Retry(u32),
}

impl std::str::FromStr for ErrorPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "forfeit" => Ok(ErrorPolicy::Forfeit),
            "random" => Ok(ErrorPolicy::Random),
            _ => s
                .strip_prefix("retry:")
                .and_then(|n| n.parse().ok())
                .map(ErrorPolicy::Retry)
                .ok_or_else(|| format!("on_error must be forfeit, random or retry:N, not {s:?}")),
        }
    }
}

impl std::fmt::Display for ErrorPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorPolicy::Forfeit => write!(f, "forfeit"),
            ErrorPolicy::Random => write!(f, "random"),
            ErrorPolicy::Retry(n) => write!(f, "retry:{n}"),
        }
    }
}

impl Serialize for ErrorPolicy {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ErrorPolicy {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut config: Config =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let base = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        config.base_dir = base.to_path_buf();
        if let Some(out) = &config.out {
            config.out = Some(base.join(out));
        }
        if let OpeningSpec::File { path } = &config.openings {
            config.openings = OpeningSpec::File {
                path: base.join(path),
            };
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if tdg_games::by_id(&self.game).is_none() {
            bail!(
                "unknown game {:?}; this build knows {:?}",
                self.game,
                tdg_games::all_ids()
            );
        }
        if self.players.len() < 2 {
            bail!("a tournament needs at least two players");
        }
        if self.concurrency == 0 {
            bail!("concurrency must be at least 1");
        }
        let mut names = HashSet::new();
        for p in &self.players {
            if !names.insert(p.name.as_str()) {
                bail!("two players are called {:?}", p.name);
            }
            match (&p.url, &p.cmd) {
                (Some(_), None) | (None, Some(_)) => {}
                _ => bail!("player {:?} needs exactly one of url or cmd", p.name),
            }
            if let Some(cmd) = &p.cmd {
                if !cmd.contains("{port}") {
                    bail!(
                        "player {:?}: cmd must contain {{port}} for the server to fill in",
                        p.name
                    );
                }
            }
        }
        for [a, b] in &self.pairs {
            for n in [a, b] {
                if !names.contains(n.as_str()) {
                    bail!("pairs mentions {n:?}, who is not a player");
                }
            }
            if a == b {
                bail!("{a:?} cannot play themselves");
            }
        }
        if let OpeningSpec::Random { count: 0, .. } = self.openings {
            bail!("openings.count must be at least 1");
        }
        Ok(())
    }

    /// The pairings to play, as indices into `players`.
    pub fn pairings(&self) -> Vec<(usize, usize)> {
        let index = |n: &str| {
            self.players
                .iter()
                .position(|p| p.name == n)
                .expect("validated")
        };
        if self.pairs.is_empty() {
            let n = self.players.len();
            (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .collect()
        } else {
            self.pairs
                .iter()
                .map(|[a, b]| (index(a), index(b)))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        game = "epaminondas/1"
        [openings]
        kind = "start"
        [[player]]
        name = "a"
        url = "http://127.0.0.1:9001"
        [[player]]
        name = "b"
        cmd = "prog --port {port}"
    "#;

    fn parse(text: &str) -> Result<Config> {
        let c: Config = toml::from_str(text)?;
        c.validate()?;
        Ok(c)
    }

    #[test]
    fn a_minimal_config_gets_sensible_defaults() {
        let c = parse(MINIMAL).unwrap();
        assert_eq!(c.concurrency, 1);
        assert_eq!(c.max_plies, 12_000);
        assert_eq!(c.openings, OpeningSpec::Start);
        assert_eq!(c.players[0].on_error, ErrorPolicy::Forfeit);
        assert_eq!(c.players[0].timeout(), Duration::from_secs(60));
        assert_eq!(c.pairings(), vec![(0, 1)]);
    }

    #[test]
    fn error_policies_parse() {
        for (s, want) in [
            ("forfeit", ErrorPolicy::Forfeit),
            ("random", ErrorPolicy::Random),
            ("retry:3", ErrorPolicy::Retry(3)),
        ] {
            assert_eq!(s.parse::<ErrorPolicy>().unwrap(), want);
            assert_eq!(want.to_string(), s);
        }
        for bad in ["", "retry", "retry:", "retry:x", "Forfeit", "ignore"] {
            assert!(bad.parse::<ErrorPolicy>().is_err(), "{bad}");
        }
    }

    #[test]
    fn timeouts_follow_the_budget() {
        let mut p = parse(MINIMAL).unwrap().players.remove(0);
        p.budget_ms = Some(1000);
        assert_eq!(p.timeout(), Duration::from_millis(4000));
        p.timeout_ms = Some(750);
        assert_eq!(p.timeout(), Duration::from_millis(750));
    }

    #[test]
    fn round_robin_covers_every_pair_once() {
        let mut c = parse(MINIMAL).unwrap();
        let extra = c.players[0].clone();
        for name in ["c", "d"] {
            c.players.push(PlayerConfig {
                name: name.into(),
                ..extra.clone()
            });
        }
        let pairs = c.pairings();
        assert_eq!(pairs.len(), 6);
        assert!(pairs.iter().all(|(a, b)| a < b));
    }

    #[test]
    fn explicit_pairs_are_honoured() {
        // Top-level keys go before the first table, or TOML files them under it.
        let c = parse(&format!("pairs = [[\"b\", \"a\"]]\n{MINIMAL}")).unwrap();
        assert_eq!(c.pairings(), vec![(1, 0)]);
    }

    #[test]
    fn bad_configs_are_rejected_with_a_reason() {
        let cases: &[(&str, &str, &str)] = &[
            ("epaminondas/1", "epaminondas/9", "unknown game"),
            ("name = \"b\"", "name = \"a\"", "two players"),
            ("cmd = \"prog --port {port}\"", "cmd = \"prog\"", "{port}"),
            (
                "cmd = \"prog --port {port}\"",
                "cmd = \"prog --port {port}\"\nurl = \"http://x\"",
                "exactly one",
            ),
        ];
        for (from, to, why) in cases {
            let err = parse(&MINIMAL.replace(from, to)).unwrap_err().to_string();
            assert!(err.contains(why), "{why}: {err}");
        }
        let err = parse(&format!("pairs = [[\"a\", \"z\"]]\n{MINIMAL}")).unwrap_err();
        assert!(err.to_string().contains("not a player"));
        let err = parse(&format!("pairs = [[\"a\", \"a\"]]\n{MINIMAL}")).unwrap_err();
        assert!(err.to_string().contains("themselves"));
        let err = parse(&format!("concurrency = 0\n{MINIMAL}")).unwrap_err();
        assert!(err.to_string().contains("concurrency"));
        assert!(
            parse(&format!("typo_field = 1\n{MINIMAL}")).is_err(),
            "unknown keys are errors"
        );
    }
}
