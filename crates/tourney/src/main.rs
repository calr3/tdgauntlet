use anyhow::{bail, Context as _, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use tdg_tourney::config::Config;
use tdg_tourney::{check, prepare, report, tournament};

#[derive(Parser)]
#[command(version, about = "The tdgauntlet tournament server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a tournament.
    Run {
        /// The tournament config.
        config: PathBuf,
        /// Write the result file here, overriding `out` in the config.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Do not print each game as it finishes.
        #[arg(long, short)]
        quiet: bool,
    },
    /// Check that a running client follows the protocol.
    Check {
        /// The client's base URL, e.g. http://127.0.0.1:9001
        url: String,
        #[arg(long, default_value = "epaminondas/1")]
        game: String,
        /// How many positions to ask about.
        #[arg(long, default_value_t = 32)]
        positions: usize,
        /// Budget to state in each request.
        #[arg(long, default_value_t = 200)]
        budget_ms: u64,
        /// How long to wait for each answer.
        #[arg(long, default_value_t = 30)]
        timeout_s: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Args::parse().command {
        Command::Run { config, out, quiet } => run(config, out, quiet).await,
        Command::Check {
            url,
            game,
            positions,
            budget_ms,
            timeout_s,
        } => {
            let opts = check::CheckOptions {
                game,
                positions,
                budget_ms,
                timeout: Duration::from_secs(timeout_s),
                ..Default::default()
            };
            let checks = check::run(&url, &opts).await;
            let mut failed = 0;
            for c in &checks {
                match &c.result {
                    Ok(detail) => println!("  ok    {}: {detail}", c.name),
                    Err(detail) => {
                        failed += 1;
                        println!("  FAIL  {}: {detail}", c.name)
                    }
                }
            }
            if failed > 0 {
                bail!("{failed} of {} checks failed", checks.len());
            }
            println!("{url} follows the protocol");
            Ok(())
        }
    }
}

async fn run(config_path: PathBuf, out: Option<PathBuf>, quiet: bool) -> Result<()> {
    let config = Config::load(&config_path)?;
    let plan = prepare(&config).await?;
    eprintln!(
        "{} openings x {} pairings x 2 colours = {} games, {} at a time",
        plan.openings.len(),
        plan.pairings.len(),
        2 * plan.openings.len() * plan.pairings.len(),
        plan.concurrency
    );

    let progress: Option<tournament::Progress> = (!quiet).then(|| {
        Box::new(
            |g: &tdg_tourney::engine::GameRecord, n: usize, total: usize| {
                let result = match g.outcome {
                    tdg_proto::Outcome::Draw => "draw".to_string(),
                    tdg_proto::Outcome::Win { side } => format!("{} won", g.players[side.index()]),
                };
                eprintln!("[{n}/{total}] {} ({} moves): {result}", g.id, g.moves.len());
            },
        ) as tournament::Progress
    });
    let result = tournament::run(plan, progress).await;

    if let Some(out) = out.or(config.out) {
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&out, serde_json::to_string_pretty(&result)?)
            .with_context(|| format!("writing {}", out.display()))?;
        eprintln!("wrote {}", out.display());
    }
    print!("{}", report::render(&result));
    Ok(())
}
