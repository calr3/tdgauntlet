//! The evaluation must match pgx's `EpaminondasEvaluator` term for term, so that matches between
//! this client and `negamax.py` measure the search and nothing else.
//! The reference numbers come from `conformance/dump_eval.py`.

use serde_json::Value;
use tdg_alphabeta::eval::{terms, Weights};
use tdg_games::epaminondas::Epaminondas;
use tdg_games::Game;

#[test]
fn every_term_matches_pgx() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/epaminondas_eval.json"
    );
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    let w = &data["weights"];
    let ours = Weights::default();
    for (name, value) in [
        ("material", ours.material),
        ("crossing", ours.crossing),
        ("home_row_defense", ours.home_row_defense),
        ("territory", ours.territory),
        ("centrality", ours.centrality),
        ("mobility", ours.mobility),
    ] {
        assert_eq!(w[name].as_f64().unwrap(), value, "weight {name}");
    }

    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() > 100);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
    let mut nonzero = [0usize; 6];
    for (i, case) in cases.iter().enumerate() {
        let pos = Epaminondas.decode_state(&case["state"]).unwrap();
        let t = terms(&pos);
        let want = &case["terms"];
        for (k, (name, got)) in [
            ("material", t.material),
            ("crossing", t.crossing),
            ("home_row_defense", t.home_row_defense),
            ("territory", t.territory),
            ("centrality", t.centrality),
            ("mobility", t.mobility),
        ]
        .into_iter()
        .enumerate()
        {
            let expected = want[name].as_f64().unwrap();
            assert!(
                close(got, expected),
                "case {i}, {name}: ours {got}, pgx {expected}"
            );
            nonzero[k] += (expected != 0.0) as usize;
        }
        let total = t.weighted(&ours);
        assert!(
            close(total, case["total"].as_f64().unwrap()),
            "case {i} total"
        );
    }
    // The comparison is only meaningful if each term actually varies across the cases.
    assert!(
        nonzero.iter().all(|&n| n > 0),
        "some term is zero in every case: {nonzero:?}"
    );
}

#[test]
fn the_committed_weight_files_load() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/weights");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let w = tdg_alphabeta::eval::Weights::load(&path).unwrap();
        assert_eq!(
            w.material,
            30.0,
            "{}: material anchors the scale",
            path.display()
        );
    }
}
