//! The conformance gate for Go: the Rust rules must agree exactly with pgx's `go_19x19`.
//!
//! `conformance/dump_go.py` plays seeded and scripted games in pgx and records each game's
//! actions, then for every position reached a digest of it, its outcome, its legal actions, a
//! combined digest of every successor and each successor's outcome. This replays the games and
//! checks all of it. It also checks perft counts computed by pgx.
//!
//! When pgx's rules change, regenerate the vectors (see the script's docstring) and commit them.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tdg_games::go::{describe, Go, Move, Position, MAX_MOVES, N};
use tdg_games::{Game, Outcome, Side};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance")
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u32 {
    let mut h: u32 = 0x811C9DC5;
    for b in bytes {
        h = (h ^ b as u32).wrapping_mul(0x01000193);
    }
    h
}

/// FNV-1a (32-bit) of the state's wire fields as bytes, as the dump script computes it: each
/// point as colour + 1, `to_move`, then the ko point + 1 as two bytes, low first (0 for none).
fn digest(p: &Position) -> String {
    let ko = p.ko.map_or(0, |k| k + 1);
    let bytes = p.board.iter().map(|&v| (v + 1) as u8).chain([
        p.to_move.index() as u8,
        ko as u8,
        (ko >> 8) as u8,
    ]);
    format!("{:08x}", fnv(bytes))
}

fn outcome_code(o: Option<Outcome>) -> char {
    match o {
        None => '-',
        Some(Outcome::Win { side: Side::First }) => '0',
        Some(Outcome::Win { side: Side::Second }) => '1',
        Some(Outcome::Draw) => 'd',
    }
}

/// The actions whose bits are set in the dump's hex bitmask.
fn unmask(hex: &str) -> Vec<u16> {
    (0..hex.len() / 2)
        .flat_map(|i| {
            let byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
            (0..8)
                .filter(move |b| byte >> b & 1 == 1)
                .map(move |b| (8 * i + b) as u16)
        })
        .collect()
}

struct Record {
    game: Vec<u16>,
    policy: String,
    ply: usize,
    line: Value,
}

fn read_vectors() -> (Value, Vec<Record>) {
    let path = vectors_dir().join("go.jsonl.gz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate it with conformance/dump_go.py.",
            path.display()
        )
    });
    let mut lines = BufReader::new(GzDecoder::new(file)).lines();
    let meta: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(meta["kind"], "meta");
    let mut records = Vec::new();
    let (mut game, mut policy) = (Vec::new(), String::new());
    for line in lines {
        let line: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if line["kind"] == "game" {
            game = line["actions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as u16)
                .collect();
            policy = line["policy"].as_str().unwrap().to_string();
            continue;
        }
        let ply = line["ply"].as_u64().unwrap() as usize;
        records.push(Record {
            game: game.clone(),
            policy: policy.clone(),
            ply,
            line,
        });
    }
    (meta, records)
}

#[test]
fn vectors_come_from_the_rules_version_this_engine_implements() {
    let (meta, _) = read_vectors();
    assert_eq!(
        meta["pgx_env_version"], "v1",
        "regenerate the vectors, or port the new rules"
    );
    assert_eq!(
        meta["pgx_rules_dirty"], false,
        "vectors were dumped from uncommitted pgx rules; regenerate from a commit"
    );
}

#[test]
fn every_position_agrees_with_pgx() {
    let g = Go;
    let (_, records) = read_vectors();
    let mut failures: Vec<String> = Vec::new();
    let mut checked_moves = 0usize;
    let mut pos = Position::start();

    for (i, r) in records.iter().enumerate() {
        if r.ply == 0 {
            pos = Position::start();
        } else {
            pos = pos.apply(Move::from_action(r.game[r.ply - 1]).unwrap());
        }
        let here = format!(
            "record {i} ({} game, move {}, after {})",
            r.policy,
            r.ply,
            r.game[..r.ply]
                .iter()
                .rev()
                .take(4)
                .rev()
                .map(|&a| describe(Move::from_action(a).unwrap()))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let wire = g.encode_state(&pos);
        assert_eq!(
            g.decode_state(&wire).as_ref(),
            Ok(&pos),
            "{here}: encoding does not round-trip"
        );
        if digest(&pos) != r.line["digest"].as_str().unwrap() {
            failures.push(format!("{here}: a different position\n{}", g.render(&pos)));
            break; // Everything after this in the game would differ too.
        }

        let want = r.line["outcome"].as_str().unwrap().chars().next().unwrap();
        let got = outcome_code(g.outcome(&pos));
        if got != want {
            failures.push(format!(
                "{here}: outcome {got}, pgx says {want}\n{}",
                g.render(&pos)
            ));
            continue;
        }
        let Some(legal) = r.line.get("legal").and_then(Value::as_str) else {
            continue;
        };

        let pgx_moves = unmask(legal);
        let ours: Vec<u16> = g.legal_moves(&pos).iter().map(|m| m.action()).collect();
        if ours != pgx_moves {
            let only = |a: &[u16], b: &[u16]| -> Vec<String> {
                a.iter()
                    .filter(|m| !b.contains(m))
                    .map(|&m| describe(Move::from_action(m).unwrap()))
                    .collect()
            };
            failures.push(format!(
                "{here}: move sets differ ({} ours, {} pgx)\n  only rust: {:?}\n  only pgx:  {:?}\n{}",
                ours.len(),
                pgx_moves.len(),
                only(&ours, &pgx_moves),
                only(&pgx_moves, &ours),
                g.render(&pos)
            ));
            continue;
        }

        let successors: Vec<Position> = ours
            .iter()
            .map(|&a| pos.apply(Move::from_action(a).unwrap()))
            .collect();
        let each: String = successors.iter().map(digest).collect();
        if format!("{:08x}", fnv(each.bytes())) != r.line["successors"].as_str().unwrap() {
            failures.push(format!(
                "{here}: some move leads to a different position than in pgx (rerun the dump \
                 for this one to see which)\n{}",
                g.render(&pos)
            ));
        }
        let outcomes: Vec<char> = r.line["successor_outcomes"]
            .as_str()
            .unwrap()
            .chars()
            .collect();
        for ((&a, next), &want) in ours.iter().zip(&successors).zip(&outcomes) {
            let o = outcome_code(g.outcome(next));
            if o != want {
                failures.push(format!(
                    "{here}: after {} the outcome is {o}, pgx says {want}\n{}",
                    describe(Move::from_action(a).unwrap()),
                    g.render(next)
                ));
            }
        }
        checked_moves += ours.len();
        if failures.len() > 20 {
            break;
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements with pgx (showing up to 5):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
    eprintln!(
        "agreed with pgx on {} positions and {checked_moves} moves",
        records.len()
    );
}

/// The vectors are only as good as what they cover: captures, suicide, ko, superko by each side,
/// every way the game ends, and both winners. The scripted games and the dump's policies are
/// what reach them; keep them.
#[test]
fn vectors_cover_the_rules_that_matter() {
    let (_, records) = read_vectors();
    let (mut captures, mut suicides, mut kos) = (0, 0, 0);
    let mut ends = std::collections::BTreeMap::<String, usize>::new();
    let mut pos = Position::start();
    for r in &records {
        if r.ply == 0 {
            pos = Position::start();
        } else {
            let before = pos.board.iter().filter(|&&v| v != 0).count();
            pos = pos.apply(Move::from_action(r.game[r.ply - 1]).unwrap());
            let after = pos.board.iter().filter(|&&v| v != 0).count();
            captures += (after < before) as usize;
        }
        if let Some(legal) = r.line.get("legal").and_then(Value::as_str) {
            let legal = unmask(legal);
            let empty = (0..N as u16).filter(|&p| pos.board[p as usize] == 0);
            suicides += empty
                .filter(|&p| Some(p) != pos.ko && !legal.contains(&p))
                .count();
            kos += pos.ko.is_some() as usize;
        }
        if let Some(o) = Go.outcome(&pos) {
            let how = if pos.superko() {
                "superko"
            } else if pos.history.len() == MAX_MOVES && pos.passes() < 2 {
                "move cap"
            } else {
                "two passes"
            };
            let who = if o == Outcome::win(Side::First) {
                "black"
            } else {
                "white"
            };
            *ends.entry(format!("{how}, {who} won")).or_default() += 1;
        }
    }
    eprintln!(
        "{} positions: {captures} captures, {suicides} suicide points, {kos} kos; ends {ends:?}",
        records.len()
    );
    assert!(captures >= 100, "too few captures: {captures}");
    assert!(suicides >= 100, "too few suicide points: {suicides}");
    assert!(kos >= 5, "too few kos: {kos}");
    for end in [
        "superko, black won",
        "superko, white won",
        "two passes, black won",
        "two passes, white won",
    ] {
        assert!(ends.contains_key(end), "no game ends with {end}: {ends:?}");
    }
    let capped: usize = ends
        .iter()
        .filter(|(k, _)| k.starts_with("move cap"))
        .map(|(_, n)| n)
        .sum();
    assert!(capped >= 1, "no game reaches the move cap: {ends:?}");
}

fn perft(g: &Go, pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = g.legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|m| perft(g, &g.apply(pos, m), depth - 1))
        .sum()
}

#[test]
fn perft_agrees_with_pgx() {
    let g = Go;
    let path = vectors_dir().join("go_perft.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() >= 4, "expected the start plus later cases");
    for case in cases {
        let actions: Vec<u16> = case["actions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap() as u16)
            .collect();
        let pos = Position::from_history(&actions).unwrap();
        let depth = case["depth"].as_u64().unwrap() as u32;
        assert_eq!(
            perft(&g, &pos, depth),
            case["nodes"].as_u64().unwrap(),
            "perft({depth}) after {} moves\n{}",
            actions.len(),
            g.render(&pos)
        );
    }
}
