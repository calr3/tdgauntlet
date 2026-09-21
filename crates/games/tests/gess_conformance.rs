//! The conformance gate for Gess: the Rust rules must agree exactly with pgx's.
//!
//! `conformance/dump_gess.py` walks seeded games in pgx and records, for every position, the full
//! set of complete legal moves, a digest of each successor state, and each successor's outcome.
//! This replays every one of them. It also checks perft counts computed by pgx.
//!
//! When pgx's rules change, regenerate the vectors (see the script's docstring) and commit them.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tdg_games::gess::{Gess, Move, Position};
use tdg_games::{Game, Outcome, Side};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance")
}

/// FNV-1a (32-bit) of the same canonical rendering the dump script uses.
fn digest(state: &Value) -> String {
    let text = format!(
        "{}|{}|{}|{}",
        state["board"].as_str().unwrap(),
        state["to_move"],
        state["moves"],
        state["quiet_moves"],
    );
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
    let path = vectors_dir().join("gess.jsonl.gz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate it with conformance/dump_gess.py.",
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
    let g = Gess;
    let (_, records) = read_vectors();
    let mut failures: Vec<String> = Vec::new();
    let mut checked_moves = 0usize;

    for (i, r) in records.iter().enumerate() {
        let wire = &r["state"];
        let pos: Position = g.decode_state(wire).expect("vector state decodes");
        let here = format!("record {i} (game {}, move {})", r["game"], wire["moves"]);

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

/// The vectors are only as good as what they cover; make sure they reach the interesting parts.
/// Plain random play here is nearly all self-destruction — a piece lifted from around your own
/// ring loses on the spot — so the dump also plays two policies that keep their ring. Both must
/// stay in the vectors, or everything past move three stops being tested.
#[test]
fn vectors_cover_the_rules_that_matter() {
    let g = Gess;
    let (_, records) = read_vectors();
    let positions = records.len();
    let stones = |r: &Value| {
        r["state"]["board"]
            .as_str()
            .unwrap()
            .chars()
            .filter(|&c| c != '.')
            .count()
    };
    let captures = records
        .windows(2)
        .filter(|w| w[0]["game"] == w[1]["game"])
        .filter(|w| stones(&w[1]) < stones(&w[0]))
        .count();
    let deep_into_the_clock = records
        .iter()
        .filter(|r| r["state"]["quiet_moves"].as_u64().unwrap() >= 15)
        .count();
    let terminal_successors: String = records
        .iter()
        .filter_map(|r| r["successor_outcomes"].as_str())
        .collect::<String>()
        .chars()
        .filter(|&c| c != '-')
        .collect();
    let ended = |by_clock: bool| {
        records
            .iter()
            .filter(|r| r["outcome"] != "-")
            .filter(|r| {
                (r["state"]["quiet_moves"].as_u64().unwrap()
                    >= tdg_games::gess::MAX_QUIET_MOVES as u64)
                    == by_clock
            })
            .count()
    };
    let (by_clock, by_ring) = (ended(true), ended(false));
    // Pieces whose footprint hangs over the border, and moves that push a stone off it: two
    // rules nothing else in the suite would notice.
    let (mut border_pieces, mut over_the_edge) = (0, 0);
    for r in records.iter().filter(|r| r.get("moves").is_some()) {
        let pos = g.decode_state(&r["state"]).unwrap();
        let mover = pos.to_move;
        for m in r["moves"].as_array().unwrap() {
            let (from, to) = (
                m[0].as_u64().unwrap() as usize,
                m[1].as_u64().unwrap() as usize,
            );
            let edge = |cell: usize| {
                let (row, col) = (cell / tdg_games::gess::SIZE, cell % tdg_games::gess::SIZE);
                [row, col]
                    .iter()
                    .any(|&x| x == 0 || x == tdg_games::gess::SIZE - 1)
            };
            border_pieces += edge(from) as usize;
            let next = g.apply(&pos, &Move::new(from, to));
            over_the_edge += (next.count(mover) < pos.count(mover)) as usize;
        }
    }

    eprintln!(
        "{positions} positions, {captures} captures, {deep_into_the_clock} past 15 quiet moves, \
         {} terminal successors, {by_clock} games ended on the count-back, {by_ring} on a ring, \
         {border_pieces} pieces centred on the border, {over_the_edge} moves costing the mover a \
         stone",
        terminal_successors.len()
    );
    assert!(positions >= 400, "too few positions: {positions}");
    assert!(captures >= 50, "too few captures: {captures}");
    assert!(
        deep_into_the_clock >= 40,
        "the quiet-move clock is barely exercised"
    );
    assert!(by_clock >= 3, "too few games reach the quiet-move clock");
    assert!(by_ring >= 3, "too few games end on a ring: {by_ring}");
    assert!(border_pieces >= 50, "no pieces centred on the border ring");
    assert!(
        over_the_edge >= 50,
        "no moves that cost the mover a stone (self-capture or the border)"
    );
    assert!(
        terminal_successors.contains('0')
            && terminal_successors.contains('1')
            && terminal_successors.contains('d'),
        "both wins and a draw should be reachable in one move somewhere: {terminal_successors:?}"
    );
}

fn perft(g: &Gess, pos: &Position, depth: u32) -> u64 {
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
    let g = Gess;
    let path = vectors_dir().join("gess_perft.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() >= 2, "expected the start plus deeper cases");
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
