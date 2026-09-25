//! The conformance gate for Pig: the Rust rules must agree exactly with pgx's `pig`.
//!
//! `conformance/dump_pig.py` walks seeded games in pgx and records, for every position, its legal
//! moves and, for each move and each face of the die, a digest of the successor state and that
//! successor's outcome. This replays every one of them, and the start after each first roll.
//!
//! When pgx's rules change, regenerate the vectors (see the script's docstring) and commit them.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tdg_games::pig::{Move, Pig, Position, FACES, TARGET};
use tdg_games::{Game, Outcome, Side};

/// FNV-1a (32-bit) of the same canonical rendering the dump script uses.
fn digest(state: &Value) -> String {
    let text = format!(
        "{},{}|{}|{}|{}",
        state["totals"][0],
        state["totals"][1],
        state["turn_total"],
        state["last_roll"],
        state["to_move"]
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
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/pig.jsonl.gz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Generate it with conformance/dump_pig.py.",
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

fn faces() -> impl Iterator<Item = u8> {
    1..=FACES
}

#[test]
fn vectors_come_from_the_rules_version_this_engine_implements() {
    let (meta, _) = read_vectors();
    // v1 only appended an observation feature: its vectors are v0's, position for position.
    assert!(
        ["v0", "v1"].contains(&meta["pgx_env_version"].as_str().unwrap()),
        "{}: regenerate the vectors, or port the new rules",
        meta["pgx_env_version"]
    );
    assert_eq!(
        meta["pgx_rules_dirty"], false,
        "vectors were dumped from uncommitted pgx rules; regenerate from a commit"
    );
}

#[test]
fn the_start_after_each_first_roll_agrees_with_pgx() {
    let (meta, _) = read_vectors();
    let starts: Vec<Value> = faces()
        .map(|f| Pig.encode_state(&Position::start(f)))
        .collect();
    assert_eq!(meta["starts"], Value::from(starts));
}

#[test]
fn every_position_agrees_with_pgx() {
    let g = Pig;
    let (_, records) = read_vectors();
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for (i, r) in records.iter().enumerate() {
        let wire = &r["state"];
        let pos = g.decode_state(wire).expect("vector state decodes");
        let here = format!("record {i} (game {})", r["game"]);
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
            .map(|a| Move::from_action(a.as_u64().unwrap() as usize).unwrap())
            .collect();
        let ours = g.legal_moves(&pos);
        if ours != pgx_moves {
            failures.push(format!(
                "{here}: moves {ours:?}, pgx has {pgx_moves:?}\n{}",
                g.render(&pos)
            ));
            continue;
        }
        for (j, &mv) in ours.iter().enumerate() {
            let digests = r["successors"][j].as_array().unwrap();
            let outcomes: Vec<char> = r["successor_outcomes"][j]
                .as_str()
                .unwrap()
                .chars()
                .collect();
            for f in faces() {
                let next = pos.play(mv, f);
                let k = f as usize - 1;
                let d = digest(&g.encode_state(&next));
                if d != digests[k].as_str().unwrap() {
                    failures.push(format!(
                        "{here}: {} then a {f} leads to a different state than in pgx\n{}",
                        mv.name(),
                        g.render(&next)
                    ));
                }
                let o = outcome_code(g.outcome(&next));
                if o != outcomes[k] {
                    failures.push(format!(
                        "{here}: after {} then a {f} the outcome is {o}, pgx says {}\n{}",
                        mv.name(),
                        outcomes[k],
                        g.render(&next)
                    ));
                }
                checked += 1;
            }
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
        "agreed with pgx on {} positions and {checked} (move, roll) successors",
        records.len()
    );
}

/// The vectors are only as good as what they cover: wins by each side, one exactly on the
/// target, a 1 that costs points, and the two ways a turn total past the target is not a win - a
/// 1 rolled on it, and rolling on instead of banking. The dump's `overshoot` policy is what
/// reaches the last two; keep it.
#[test]
fn vectors_cover_the_rules_that_matter() {
    let g = Pig;
    let (_, records) = read_vectors();
    let positions: Vec<Position> = records
        .iter()
        .map(|r| g.decode_state(&r["state"]).unwrap())
        .collect();
    let mover = |p: &Position| p.totals[p.to_move.index()];
    let count = |f: &dyn Fn(&Position) -> bool| positions.iter().filter(|p| f(p)).count();
    let wins = |side: Side| count(&|p| p.outcome() == Some(Outcome::Win { side }));
    let exact = count(&|p| p.outcome().is_some() && p.totals.contains(&TARGET));
    let costly_ones = count(&|p| p.pigged_out() && p.turn_total > 1);
    let ones_past = count(&|p| p.pigged_out() && mover(p) + p.turn_total >= TARGET);
    // A roll played with the target already in the turn total: the same player again next.
    let rolls_past = records
        .windows(2)
        .zip(positions.windows(2))
        .filter(|(r, p)| {
            r[0]["game"] == r[1]["game"]
                && p[0].to_move == p[1].to_move
                && mover(&p[0]) + p[0].turn_total >= TARGET
        })
        .count();
    eprintln!(
        "{} positions: {} and {} wins, {exact} on exactly {TARGET}; {costly_ones} costly 1s, \
         {ones_past} past the target; {rolls_past} positions rolled past it",
        positions.len(),
        wins(Side::First),
        wins(Side::Second)
    );
    assert!(
        positions.len() >= 1000,
        "too few positions: {}",
        positions.len()
    );
    assert!(
        wins(Side::First) >= 5 && wins(Side::Second) >= 5,
        "both sides should win"
    );
    assert!(exact >= 1, "no game is won on exactly the target");
    assert!(costly_ones >= 20, "a 1 rarely costs anything");
    assert!(
        ones_past >= 1,
        "no 1 is rolled with the target in the turn total"
    );
    assert!(
        rolls_past >= 5,
        "the target is rarely passed without banking"
    );
}
