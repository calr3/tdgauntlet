//! A human at the terminal.
//!
//! Each request shows the board it was sent and waits for a move. Like any client it keeps no
//! game state: the board on screen is always exactly the position asked about. It advertises a
//! concurrency of one, so the server asks one question at a time.
//!
//! Start it yourself in a terminal and give the tournament its URL; do not let the server spawn
//! it, or it will have no terminal to read from. Give it a long `timeout_ms` in the config.

mod input;

use clap::Parser;
use std::io::{BufRead, Write};
use std::sync::Mutex;
use tdg_client_kit::{serve, Answer, Engine, Info};
use tdg_proto::MoveRequest;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9100)]
    port: u16,
    #[arg(long, default_value = "human")]
    name: String,
}

struct Keyboard {
    name: String,
    terminal: Mutex<()>,
}

impl Engine for Keyboard {
    fn info(&self) -> Info {
        Info {
            name: self.name.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
            games: tdg_games::all_ids().iter().map(|s| s.to_string()).collect(),
            max_concurrency: 1,
        }
    }

    fn choose(&self, req: &MoveRequest) -> Result<Answer, String> {
        let _terminal = self.terminal.lock().map_err(|_| "terminal lock poisoned")?;
        let game = tdg_games::by_id(&req.game).ok_or("unknown game")?;
        let moves = game
            .described_moves(&req.state)
            .map_err(|e| e.to_string())?;
        let board = game.render(&req.state).map_err(|e| e.to_string())?;
        let stdin = std::io::stdin();
        let mut out = std::io::stdout();
        let _ = writeln!(
            out,
            "\n{} (ply {}), you are {}:\n\n{board}",
            req.context.game_id,
            req.context.ply,
            ["first", "second"][req.context.side.index()]
        );
        if let [(only, san)] = moves.as_slice() {
            let _ = writeln!(out, "Only one legal move: {san}");
            return Ok(Answer::new(only.clone()));
        }
        loop {
            let _ = write!(out, "Your move (? for the list, resign to give up): ");
            let _ = out.flush();
            let mut line = String::new();
            if stdin
                .lock()
                .read_line(&mut line)
                .map_err(|e| e.to_string())?
                == 0
            {
                return Err("the terminal closed".into());
            }
            match input::interpret(&line, &moves) {
                input::Choice::Move(i) => return Ok(Answer::new(moves[i].0.clone())),
                input::Choice::Resign => return Err("resigned".into()),
                input::Choice::List => {
                    let names: Vec<&str> = moves.iter().map(|(_, s)| s.as_str()).collect();
                    let _ = writeln!(out, "{}", input::columns(&names, 100));
                }
                input::Choice::Unknown(why) => {
                    let _ = writeln!(out, "{why}");
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let engine = Keyboard {
        name: args.name,
        terminal: Mutex::new(()),
    };
    serve(engine, ([127, 0, 0, 1], args.port).into()).await
}
