//! `tdgauntlet check`: test a running client against the protocol before entering it.
//!
//! Beyond the endpoints themselves, this probes the one rule that is easy to break by accident:
//! a client must answer each position on its own. Every position is sent with the *same* game id,
//! in shuffled order and concurrently, so a client that tracks "the current game" by id, or by
//! the order requests arrive in, answers some of them with moves for the wrong board.

use crate::player::{HttpPlayer, Player};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::time::Duration;
use tdg_games::DynGame;
use tdg_proto::{Budget, Context, MoveRequest, ProtocolVersion, PROTOCOL_VERSION};
use tokio::task::JoinSet;

pub struct Check {
    pub name: String,
    pub result: Result<String, String>,
}

pub struct CheckOptions {
    pub game: String,
    pub positions: usize,
    pub budget_ms: u64,
    pub timeout: Duration,
    pub seed: u64,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            game: "epaminondas/1".into(),
            positions: 32,
            budget_ms: 200,
            timeout: Duration::from_secs(30),
            seed: 1,
        }
    }
}

/// Undecided positions from random games, at a spread of depths.
fn sample_positions(game: &dyn DynGame, n: usize, seed: u64) -> Vec<serde_json::Value> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut state = game.initial();
        let depth = out.len() * 3;
        for _ in 0..depth {
            let moves = game.legal_moves(&state).expect("own state");
            let Some(mv) = moves.choose(&mut rng) else {
                break;
            };
            state = game.play(&state, mv).expect("legal").state;
        }
        if game.outcome(&state).expect("own state").is_none() {
            out.push(state);
        }
    }
    out
}

pub async fn run(url: &str, opts: &CheckOptions) -> Vec<Check> {
    let mut checks = Vec::new();
    let mut record = |name: &str, result: Result<String, String>| {
        checks.push(Check {
            name: name.into(),
            result,
        });
    };
    let player = Arc::new(HttpPlayer::new(url));

    let identify = match player.identify().await {
        Ok(id) => {
            record(
                "identify",
                Ok(format!(
                    "{} {} (protocol {PROTOCOL_VERSION}), plays {:?}, up to {} at once",
                    id.name, id.version, id.games, id.max_concurrency
                )),
            );
            id
        }
        Err(e) => {
            record("identify", Err(e.to_string()));
            return checks;
        }
    };
    record(
        "health",
        if player.healthy().await {
            Ok("ready".into())
        } else {
            Err("not 200".into())
        },
    );
    if !identify.plays(&opts.game) {
        record(
            "plays the game",
            Err(format!("does not list {}", opts.game)),
        );
        return checks;
    }
    let Some(game) = tdg_games::by_id(&opts.game) else {
        record(
            "plays the game",
            Err(format!("this server does not know {}", opts.game)),
        );
        return checks;
    };

    // Unsupported games should be refused, not answered.
    let mut bogus = request(&opts.game, "check-bogus", game.initial(), opts, &*game);
    bogus.game = "no-such-game/1".into();
    record(
        "refuses an unknown game",
        match player.choose(&bogus, opts.timeout).await {
            Err(_) => Ok("refused".into()),
            Ok(_) => Err("answered a request for a game it does not play".into()),
        },
    );

    // The main probe: many unrelated positions, one game id, shuffled, concurrent.
    let mut positions = sample_positions(&*game, opts.positions, opts.seed);
    // Ask about one position twice, far apart.
    positions.push(positions[0].clone());
    let mut order: Vec<usize> = (0..positions.len()).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(opts.seed));

    let slots = Arc::new(tokio::sync::Semaphore::new(
        identify.max_concurrency.max(1) as usize
    ));
    let mut tasks = JoinSet::new();
    for i in order {
        let req = request(
            &opts.game,
            &format!("check-{i}"),
            positions[i].clone(),
            opts,
            &*game,
        );
        let (player, game, slots, timeout) =
            (player.clone(), game.clone(), slots.clone(), opts.timeout);
        tasks.spawn(async move {
            let _slot = slots.acquire_owned().await.expect("open");
            let answer = player.choose(&req, timeout).await;
            let verdict = match answer {
                Err(e) => Err(e.to_string()),
                Ok(r) if r.request_id != req.request_id => Err(format!(
                    "answered {} with request_id {:?}",
                    req.request_id, r.request_id
                )),
                Ok(r) => game
                    .play(&req.state, &r.mv)
                    .map(|p| p.notation)
                    .map_err(|e| format!("{}: {e}", req.request_id)),
            };
            (req.request_id, verdict)
        });
    }
    let mut failures = Vec::new();
    let mut answered = 0;
    while let Some(joined) = tasks.join_next().await {
        match joined.expect("check task") {
            (_, Ok(_)) => answered += 1,
            (_, Err(e)) => failures.push(e),
        }
    }
    let total = positions.len();
    record(
        "answers unrelated positions sharing a game id, in any order",
        if failures.is_empty() {
            Ok(format!("{answered}/{total} legal"))
        } else {
            failures.sort();
            Err(format!(
                "{}/{total} wrong, e.g. {}",
                failures.len(),
                failures
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ))
        },
    );
    checks
}

fn request(
    game_id: &str,
    request_id: &str,
    state: serde_json::Value,
    opts: &CheckOptions,
    game: &dyn DynGame,
) -> MoveRequest {
    let side = game.side_to_move(&state).expect("own state");
    MoveRequest {
        protocol: ProtocolVersion::CURRENT,
        request_id: request_id.into(),
        game: game_id.into(),
        state,
        budget: Budget::from_ms(opts.budget_ms),
        context: Context {
            match_id: "check".into(),
            game_id: "check".into(),
            ply: 0,
            side,
        },
    }
}
