//! The human-readable summary printed at the end of a run.

use crate::engine::Termination;
use crate::tournament::TournamentResult;
use std::fmt::Write;

fn pct(p: Option<f64>) -> String {
    p.map_or_else(|| "     -".into(), |p| format!("{p:5.1}%"))
}

pub fn render(r: &TournamentResult) -> String {
    let mut out = String::new();
    let games = r.games.len();
    let by_rules = r
        .games
        .iter()
        .filter(|g| g.termination == Termination::Rules)
        .count();
    let _ = writeln!(
        out,
        "{} — {} games, {} minimatches, {} openings ({} decided by the rules)\n",
        r.game,
        games,
        r.minimatches.len(),
        r.openings.len(),
        by_rules
    );

    let width = r
        .standings
        .iter()
        .map(|s| s.name.len())
        .max()
        .unwrap_or(4)
        .max(6);
    let _ = writeln!(
        out,
        "{:<width$}  {:>6}  {:>5}  {:>8}  {:>11}  {:>9}  {:>9}",
        "player", "score", "Elo", "counted", "excluded", "W-D-L", "think ms"
    );
    for s in &r.standings {
        let elo = match s.percentage {
            None => "    -".to_string(),
            // A clean sweep has no finite Elo; show the clamp as a bound, not a measurement.
            Some(p) if p >= 100.0 => ">+999".into(),
            Some(p) if p <= 0.0 => "<-999".into(),
            Some(p) => format!(
                "{:+5.0}",
                crate::score::elo_difference(p).clamp(-999.0, 999.0)
            ),
        };
        let excluded = match s.exclusion_rate {
            Some(rate) => format!("{} ({rate:.0}%)", s.tally.excluded),
            None => "-".into(),
        };
        let wdl = format!("{}-{}-{}", s.tally.wins, s.tally.draws, s.tally.losses);
        let think = s
            .mean_think_ms
            .map_or_else(|| "-".into(), |t| format!("{t:.0}"));
        let _ = writeln!(
            out,
            "{:<width$}  {}  {elo}  {:>8}  {excluded:>11}  {wdl:>9}  {think:>9}",
            s.name,
            pct(s.percentage),
            s.tally.counted,
        );
        if s.incidents > 0 {
            let _ = writeln!(
                out,
                "{:<width$}  ! {} incidents: {} forfeits, {} moves played by the server",
                "", s.incidents, s.forfeits, s.server_moves
            );
        }
    }

    if r.pairings.len() > 1 {
        let _ = writeln!(out, "\npairings (score of the first-named player):");
        for p in &r.pairings {
            let _ = writeln!(
                out,
                "  {:<width$} v {:<width$}  {}  over {} counted, {} excluded",
                p.a,
                p.b,
                pct(p.a_percentage),
                p.tally.counted,
                p.tally.excluded
            );
        }
    }

    let incidents: Vec<_> = r
        .games
        .iter()
        .flat_map(|g| g.incidents.iter().map(move |i| (g, i)))
        .collect();
    if !incidents.is_empty() {
        let _ = writeln!(out, "\nincidents:");
        for (g, i) in incidents.iter().take(20) {
            let _ = writeln!(
                out,
                "  {} ply {}: {} {} — {}",
                g.id, i.ply, i.player, i.error, i.action
            );
        }
        if incidents.len() > 20 {
            let _ = writeln!(
                out,
                "  ... and {} more in the result file",
                incidents.len() - 20
            );
        }
    }
    out
}
