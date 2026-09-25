//! Openings: the positions each minimatch starts from.
//!
//! Generated once, up front, from the tournament seed, and written into the result file, so a run
//! is reproducible and every pairing sees exactly the same set.

use crate::config::OpeningSpec;
use anyhow::{bail, Context as _, Result};
use rand::seq::IndexedRandom;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use tdg_games::DynGame;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Opening {
    pub id: String,
    pub state: Value,
    /// The moves that lead to it from the start, in notation, when known.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub moves: Vec<String>,
}

pub fn generate(game: &dyn DynGame, spec: &OpeningSpec, seed: u64) -> Result<Vec<Opening>> {
    // Only a game of chance draws on it outside `random`: for the first roll of Pig, say.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    match spec {
        OpeningSpec::Start => Ok(vec![Opening {
            id: "start".into(),
            state: game.initial(&mut rng),
            moves: vec![],
        }]),
        OpeningSpec::Random { plies, count } => random(game, *plies, *count, seed),
        OpeningSpec::File { path } => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading openings from {}", path.display()))?;
            from_book(game, &text, &mut rng)
        }
    }
}

/// Distinct positions after `plies` random moves. With few plies there may be fewer distinct
/// positions than asked for; then all of them are returned, which is noted on stderr.
fn random(game: &dyn DynGame, plies: u32, count: usize, seed: u64) -> Result<Vec<Opening>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let attempts = count.saturating_mul(50).max(1000);
    for _ in 0..attempts {
        if out.len() == count {
            break;
        }
        let mut state = game.initial(&mut rng);
        let mut moves = Vec::new();
        for _ in 0..plies {
            let legal = game.legal_moves(&state)?;
            let Some(mv) = legal.choose(&mut rng) else {
                break;
            };
            let played = game
                .play(&state, mv, &mut rng)
                .expect("a generated move is legal");
            moves.push(played.notation);
            state = played.state;
        }
        // An opening must leave a game to play.
        if game.outcome(&state)?.is_some() || !seen.insert(state.to_string()) {
            continue;
        }
        out.push(Opening {
            id: format!("o{}", out.len()),
            state,
            moves,
        });
    }
    if out.len() < count {
        eprintln!(
            "note: only {} distinct openings exist at {plies} plies; using all of them",
            out.len()
        );
    }
    if out.is_empty() {
        bail!("could not generate any opening");
    }
    Ok(out)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BookLine {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    state: Option<Value>,
    #[serde(default)]
    moves: Option<Vec<Value>>,
}

/// A JSON-lines book: `{"state": ...}` or `{"moves": [...]}` per line. Blank lines and lines
/// starting with `#` are skipped. In a game of chance, `rng` rolls the dice of a move list.
pub fn from_book(game: &dyn DynGame, text: &str, rng: &mut dyn RngCore) -> Result<Vec<Opening>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let where_ = || format!("openings line {}", n + 1);
        let entry: BookLine = serde_json::from_str(line).with_context(where_)?;
        let id = entry.id.unwrap_or_else(|| format!("o{}", out.len()));
        let (state, moves) = match (entry.state, entry.moves) {
            (Some(state), None) => {
                game.side_to_move(&state).with_context(where_)?;
                (state, vec![])
            }
            (None, Some(moves)) => {
                let mut state = game.initial(rng);
                let mut notation = Vec::new();
                for (i, mv) in moves.iter().enumerate() {
                    let played = game
                        .play(&state, mv, rng)
                        .with_context(|| format!("{}, move {}", where_(), i + 1))?;
                    notation.push(played.notation);
                    state = played.state;
                }
                (state, notation)
            }
            _ => bail!("{}: give exactly one of state or moves", where_()),
        };
        if game.outcome(&state).with_context(where_)?.is_some() {
            bail!("{}: the position is already decided", where_());
        }
        out.push(Opening { id, state, moves });
    }
    if out.is_empty() {
        bail!("the openings book is empty");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tdg_games::epaminondas::Epaminondas;

    fn epam() -> &'static dyn DynGame {
        &Epaminondas
    }

    #[test]
    fn start_is_the_initial_position() {
        let o = generate(epam(), &OpeningSpec::Start, 1).unwrap();
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].state, epam().initial(&mut rand::rng()));
    }

    #[test]
    fn zero_random_plies_is_just_the_start() {
        let o = generate(
            epam(),
            &OpeningSpec::Random {
                plies: 0,
                count: 10,
            },
            1,
        )
        .unwrap();
        assert_eq!(o.len(), 1, "there is only one position zero plies deep");
        assert_eq!(o[0].state, epam().initial(&mut rand::rng()));
    }

    #[test]
    fn the_same_seed_gives_the_same_openings() {
        let spec = OpeningSpec::Random {
            plies: 4,
            count: 20,
        };
        let a = generate(epam(), &spec, 42).unwrap();
        assert_eq!(a, generate(epam(), &spec, 42).unwrap());
        assert_ne!(a, generate(epam(), &spec, 43).unwrap());
    }

    #[test]
    fn random_openings_are_distinct_legal_and_undecided() {
        let o = generate(
            epam(),
            &OpeningSpec::Random {
                plies: 3,
                count: 50,
            },
            7,
        )
        .unwrap();
        assert_eq!(o.len(), 50);
        let distinct: HashSet<String> = o.iter().map(|x| x.state.to_string()).collect();
        assert_eq!(distinct.len(), 50);
        for x in &o {
            assert_eq!(x.moves.len(), 3);
            assert_eq!(epam().outcome(&x.state).unwrap(), None);
            // Replaying the recorded notation from the start reaches the recorded state.
            let mut state = epam().initial(&mut rand::rng());
            for san in &x.moves {
                let mv = epam()
                    .legal_moves(&state)
                    .unwrap()
                    .into_iter()
                    .find(|m| m["san"] == san.as_str())
                    .unwrap();
                state = epam().play(&state, &mv, &mut rand::rng()).unwrap().state;
            }
            assert_eq!(state, x.state);
        }
    }

    #[test]
    fn a_small_space_yields_everything_it_has() {
        // 114 first moves, so asking for more returns all 114.
        let o = generate(
            epam(),
            &OpeningSpec::Random {
                plies: 1,
                count: 500,
            },
            7,
        )
        .unwrap();
        assert_eq!(o.len(), 114);
    }

    #[test]
    fn books_accept_states_and_move_lists() {
        let start = epam().initial(&mut rand::rng());
        let book = format!(
            "# a comment\n\n{{\"id\": \"quiet\", \"state\": {start}}}\n\
             {{\"moves\": [{{\"lead\": 17, \"rear\": 3, \"dest\": 45}}]}}\n"
        );
        let o = from_book(epam(), &book, &mut rand::rng()).unwrap();
        assert_eq!(o.len(), 2);
        assert_eq!(o[0].id, "quiet");
        assert_eq!(o[1].id, "o1");
        assert_eq!(o[1].moves, ["d1d2-d4"]);
    }

    #[test]
    fn bad_books_are_rejected_with_the_line_number() {
        let start = epam().initial(&mut rand::rng());
        let illegal = "{\"moves\": [{\"lead\": 140, \"rear\": 140, \"dest\": 126}]}";
        let both = format!("{{\"state\": {start}, \"moves\": []}}");
        for (book, why) in [
            ("", "empty"),
            ("not json", "line 1"),
            (illegal, "line 1, move 1"),
            (both.as_str(), "exactly one"),
            ("{\"state\": {\"board\": \"...\"}}", "line 1"),
        ] {
            let err = format!(
                "{:#}",
                from_book(epam(), book, &mut rand::rng()).unwrap_err()
            );
            assert!(err.contains(why), "{why}: {err}");
        }
    }
}
