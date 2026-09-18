//! A human in a web browser: start it, open the printed address, and give the tournament the
//! same address as the player's `url` (or let the server spawn it with `--port {port}`). Give the
//! player a long `timeout_ms` in the config: the default is sized for engines.
//!
//! There is no login: anyone who can reach the port can move for you. It listens on localhost
//! unless `--host` says otherwise.

use clap::Parser;
use std::net::IpAddr;
use tdg_web::{router, Options};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 9200)]
    port: u16,
    /// Address to listen on; 0.0.0.0 to play from another machine.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value = "human")]
    name: String,
    /// Positions that may wait on the page at once.
    #[arg(long, default_value_t = 1)]
    concurrency: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let listener = tokio::net::TcpListener::bind((args.host, args.port)).await?;
    eprintln!(
        "{} {}: play at http://{}/",
        args.name,
        env!("CARGO_PKG_VERSION"),
        listener.local_addr()?
    );
    let app = router(Options {
        name: args.name,
        concurrency: args.concurrency.max(1),
    });
    axum::serve(listener, app).await?;
    Ok(())
}
