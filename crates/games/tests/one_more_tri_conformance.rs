//! The conformance gate for One More Tri: the Rust rules must agree exactly with pgx's `g_hex`.
//!
//! `conformance/dump_one_more_tri.py` walks seeded games in pgx and records, for every position,
//! the full set of legal moves, a digest of each successor state, and each successor's outcome.
//! This replays every one of them. It also checks perft counts computed by pgx.
//!
//! When pgx's rules change, regenerate the vectors (see the script's docstring) and commit them.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tdg_games::one_more_tri::{Move, OneMoreTri, Position};
use tdg_games::{Game, Outcome, Side};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance")
}

/// FNV-1a (32-bit) of the same canonical rendering the dump script uses.
fn digest(state: &Value) -> String {
    let board: Vec<String> = state["board"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap().to_string())
        .collect();
    let text = format!("{}|{}", board.join(","), state["to_move"]);
    let mut h: u32 = 0x811C9DC5;
    for b in text.bytes() {
        h = (h ^ b as u32).wrapping_mul(0x01000193);
    }
    format!("{h:08x}")
}

fn outcome_code(o: Option<Outcome>) -> char {
    match o {
        None => '-',
        Some(Outcome::Win { side: Side::First }) => '0',
        Some(Outcome::Win { side: Side::Second }) => '1',
        Some(Outcome::Draw) => 'd',
    }
}

fn read_vectors() -> (Value, Vec<Value>) {
    let path = vectors_dir().join("one_more_tri.jsonl.gz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate it with conformance/dump_one_more_tri.py.",
            path.display()
        )
    });
    let mut lines = BufReader::new(GzDecoder::new(file)).lines();
    let meta: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(meta["kind"], "meta");
    let records = lines
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
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
    let g = OneMoreTri;
    let (_, records) = read_vectors();
    let mut failures: Vec<String> = Vec::new();
    let mut checked_moves = 0usize;

    for (i, r) in records.iter().enumerate() {
        let wire = &r["state"];
        let pos: Position = g.decode_state(wire).expect("vector state decodes");
        let here = format!("record {i} (game {}, {} placed)", r["game"], pos.placed());

        assert_eq!(
            g.encode_state(&pos),
            *wire,
            "{here}: encoding does not round-trip"
        );

        let want = r["outcome"].as_str().unwrap().chars().next().unwrap();
        let got = outcome_code(g.outcome(&pos));
        if got != want {
            failures.push(format!(
                "{here}: outcome {got}, pgx says {want}\n{}",
                g.render(&pos)
            ));
            continue;
        }
        let Some(pgx_moves) = r.get("moves").and_then(Value::as_array) else {
            continue;
        };

        let pgx_moves: Vec<Move> = pgx_moves
            .iter()
            .map(|m| {
                let n = |k: usize| m[k].as_u64().unwrap() as usize;
                Move::new(n(0), n(1))
            })
            .collect();
        let mut ours = g.legal_moves(&pos);
        ours.sort();
        if ours != pgx_moves {
            let only = |a: &[Move], b: &[Move]| -> Vec<String> {
                a.iter()
                    .filter(|m| !b.contains(m))
                    .map(|m| g.describe_move(&pos, m))
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

        let digests = r["successors"].as_array().unwrap();
        let outcomes: Vec<char> = r["successor_outcomes"].as_str().unwrap().chars().collect();
        for (j, mv) in ours.iter().enumerate() {
            let next = g.apply(&pos, mv);
            let d = digest(&g.encode_state(&next));
            if d != digests[j].as_str().unwrap() {
                failures.push(format!(
                    "{here}: {} leads to a different state than in pgx\n{}",
                    g.describe_move(&pos, mv),
                    g.render(&next)
                ));
            }
            let o = outcome_code(g.outcome(&next));
            if o != outcomes[j] {
                failures.push(format!(
                    "{here}: after {} the outcome is {o}, pgx says {}\n{}",
                    g.describe_move(&pos, mv),
                    outcomes[j],
                    g.render(&next)
                ));
            }
            checked_moves += 1;
        }
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

/// The vectors are only as good as what they cover. Every game reaches its end, so what matters
/// is which ends: all three results, and games decided on triangle 4 and on triangle 10, where
/// the two count each other (pgx v0 had 10 ignore 4). The dump's `last_4_10` policy is what
/// reaches those; keep it.
#[test]
fn vectors_cover_the_rules_that_matter() {
    let g = OneMoreTri;
    let (_, records) = read_vectors();
    let finals: Vec<Position> = records
        .iter()
        .filter_map(|r| {
            let pos = g.decode_state(&r["state"]).unwrap();
            let moves = r.get("moves")?.as_array()?.clone();
            (pos.placed() == 19).then_some((pos, moves))
        })
        .flat_map(|(pos, moves)| {
            moves
                .into_iter()
                .map(move |m| {
                    let n = |k: usize| m[k].as_u64().unwrap() as usize;
                    pos.apply(Move::new(n(0), n(1)))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let count = |o: Outcome| finals.iter().filter(|p| p.outcome() == Some(o)).count();
    let empty_at = |t: usize| finals.iter().filter(|p| p.board[t] == 0).count();
    let (black, white, draws) = (
        count(Outcome::Win { side: Side::First }),
        count(Outcome::Win { side: Side::Second }),
        count(Outcome::Draw),
    );
    eprintln!(
        "{} positions, {} finished boards: {black} black, {white} white, {draws} drawn; {} with 4 \
         empty, {} with 10 empty",
        records.len(),
        finals.len(),
        empty_at(4),
        empty_at(10)
    );
    assert!(records.len() >= 400, "too few positions: {}", records.len());
    assert!(black >= 5 && white >= 5, "both sides should win somewhere");
    assert!(draws >= 1, "no draw is reached");
    assert!(
        empty_at(4) >= 5 && empty_at(10) >= 5,
        "the 4-10 adjacency is barely exercised"
    );
}

fn perft(g: &OneMoreTri, pos: &Position, depth: u32) -> u64 {
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
    let g = OneMoreTri;
    let path = vectors_dir().join("one_more_tri_perft.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() >= 4, "expected the start plus later cases");
    for case in cases {
        let pos = g.decode_state(&case["state"]).unwrap();
        let depth = case["depth"].as_u64().unwrap() as u32;
        let want = case["nodes"].as_u64().unwrap();
        assert_eq!(
            perft(&g, &pos, depth),
            want,
            "perft({depth}) from\n{}",
            g.render(&pos)
        );
    }
}
