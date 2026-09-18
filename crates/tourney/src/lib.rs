//! The tdgauntlet tournament server.

pub mod check;
pub mod config;
pub mod engine;
pub mod openings;
pub mod player;
pub mod report;
pub mod score;
pub mod tournament;

use anyhow::{bail, Context as _, Result};
use config::Config;
use engine::Seat;
use player::{HttpPlayer, Player, SpawnedPlayer};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tournament::{Entrant, Plan};

/// Connect to (or start) every player, check each one can play the game, and build the plan.
pub async fn prepare(config: &Config) -> Result<Plan> {
    let game = tdg_games::by_id(&config.game).expect("validated");
    let mut entrants = Vec::new();
    for p in &config.players {
        let (player, source): (Arc<dyn Player>, String) = match (&p.url, &p.cmd) {
            (Some(url), _) => (Arc::new(HttpPlayer::new(url)), url.clone()),
            (_, Some(cmd)) => (
                Arc::new(SpawnedPlayer::start(p, cmd, &config.base_dir).await?),
                cmd.clone(),
            ),
            _ => unreachable!("validated"),
        };
        let identify = player
            .identify()
            .await
            .with_context(|| format!("asking {:?} to identify itself at {source}", p.name))?;
        if !identify.plays(&config.game) {
            bail!(
                "{:?} does not play {} (it plays {:?})",
                p.name,
                config.game,
                identify.games
            );
        }
        let limit = identify
            .max_concurrency
            .max(1)
            .min(p.max_concurrency.unwrap_or(u32::MAX));
        eprintln!(
            "{}: {} {} at {source}, up to {limit} at once",
            p.name, identify.name, identify.version
        );
        entrants.push(Entrant {
            seat: Arc::new(Seat {
                name: p.name.clone(),
                player,
                limit: Arc::new(Semaphore::new(limit as usize)),
                budget: p.budget(),
                timeout: p.timeout(),
                on_error: p.on_error,
            }),
            identify,
            source,
        });
    }
    let openings = openings::generate(game.as_ref(), &config.openings, config.seed)?;
    Ok(Plan {
        game,
        entrants,
        openings,
        pairings: config.pairings(),
        concurrency: config.concurrency,
        max_plies: config.max_plies,
        seed: config.seed,
    })
}
