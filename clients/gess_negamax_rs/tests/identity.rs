//! The port plays exactly like the Gess web app's engine with the default weights: the same root
//! moves, outcomes and keys, the same per-move scores on the exact TT-off path, and the same best
//! score at every depth on the production TT-on path, for 150 positions from strong games. The
//! reference values come from the app's own WebAssembly build (`make_identity_fixture.mjs`).

use flate2::read::GzDecoder;
use serde_json::Value;
use std::io::Read;
use tdg_gess_negamax::{Board, Searcher, Weights};

fn board(state: &Value) -> Board {
    let text = state["board"].as_str().unwrap();
    let (mut w, mut b) = ([0u32; 20], [0u32; 20]);
    for (i, ch) in text.chars().enumerate() {
        match ch {
            'W' => w[i / 20] |= 1 << (i % 20),
            'B' => b[i / 20] |= 1 << (i % 20),
            _ => {}
        }
    }
    Board {
        w,
        b,
        white_to_move: state["to_move"].as_i64() == Some(1),
        quiet: state["quiet_moves"].as_i64().unwrap() as i32,
    }
}

fn fixture() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/identity.json.gz");
    let mut text = String::new();
    GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn root_moves_match_the_app() {
    let mut s = Searcher::new(Weights::default());
    for rec in fixture() {
        let b = board(&rec["state"]);
        let ours: Vec<Vec<i64>> = s
            .root_moves(&b)
            .into_iter()
            .map(|(f, t, o, k)| vec![f as i64, t as i64, o as i64, k as i64])
            .collect();
        let theirs: Vec<Vec<i64>> = rec["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                m.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(ours, theirs, "root moves of {}", rec["state"]);
    }
}

#[test]
fn exact_search_matches_the_app() {
    let mut s = Searcher::new(Weights::default());
    let mut checked = 0;
    for rec in fixture() {
        let Some(by_depth) = rec["search_root"].as_object() else {
            continue;
        };
        let b = board(&rec["state"]);
        for (d, want) in by_depth {
            let (best, scores) = s.search_root(&b, d.parse().unwrap(), false);
            let scores: Vec<i64> = scores.iter().map(|x| x.2 as i64).collect();
            let want_scores: Vec<i64> = want["scores"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect();
            assert_eq!(
                best as i64,
                want["best"].as_i64().unwrap(),
                "depth {d} best of {}",
                rec["state"]
            );
            assert_eq!(scores, want_scores, "depth {d} scores of {}", rec["state"]);
            checked += 1;
        }
    }
    assert!(checked >= 150, "{checked}");
}

#[test]
fn production_search_matches_the_app() {
    let mut s = Searcher::new(Weights::default());
    for rec in fixture() {
        let b = board(&rec["state"]);
        for (d, want) in rec["search_to_depth"].as_object().unwrap() {
            let got = s.search_to_depth(&b, d.parse().unwrap(), true);
            assert_eq!(
                got as i64,
                want.as_i64().unwrap(),
                "depth {d} of {}",
                rec["state"]
            );
        }
    }
}
