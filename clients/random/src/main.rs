//! The reference client: a uniformly random legal move, for any game this build knows.
//!
//! It is also the shortest example of a conformant client. It holds no state at all, so it is
//! trivially safe to ask about any number of games at once.

use clap::Parser;
use rand::seq::IndexedRandom;
use tdg_client_kit::{serve, Answer, Engine, Info};
use tdg_proto::MoveRequest;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9000)]
    port: u16,
    #[arg(long, default_value = "random")]
    name: String,
}

struct Random {
    name: String,
}

impl Engine for Random {
    fn info(&self) -> Info {
        Info {
            name: self.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            games: tdg_games::all_ids().iter().map(|s| s.to_string()).collect(),
            max_concurrency: 64,
        }
    }

    fn choose(&self, req: &MoveRequest) -> Result<Answer, String> {
        let game = tdg_games::by_id(&req.game).ok_or("unknown game")?;
        let moves = game.legal_moves(&req.state).map_err(|e| e.to_string())?;
        let mv = moves
            .choose(&mut rand::rng())
            .ok_or("no legal moves: the game is over")?;
        Ok(Answer::new(mv.clone()))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    serve(
        Random { name: args.name },
        ([127, 0, 0, 1], args.port).into(),
    )
    .await
}
