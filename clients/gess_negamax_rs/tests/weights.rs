//! Weights files and the evaluation's terms.

use tdg_games::gess::Position;
use tdg_gess_negamax::{evaluate, features, Board, Weights, NUM_FEATURES};

#[test]
fn a_weights_file_overrides_only_the_names_it_gives() {
    let dir = std::env::temp_dir().join("tdg-gess-negamax-weights-test.json");
    std::fs::write(&dir, r#"{"ring": 195, "pressure": 94}"#).unwrap();
    let w = Weights::load(&dir).unwrap();
    assert_eq!(
        w,
        Weights {
            ring: 195,
            pressure: 94,
            ..Weights::default()
        }
    );
    let v1 = Weights::load(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/weights/v1.json"
    )))
    .unwrap();
    assert_eq!(v1.stone, 100);
}

#[test]
fn the_evaluation_is_the_weighted_sum_of_the_terms_from_either_side() {
    let mut pos = Position::start();
    let candidates = Weights {
        threat: 3,
        defenders: -2,
        mobile: 5,
        edge: -7,
        quiet_stone: 11,
        ..Weights::default()
    };
    for _ in 0..12 {
        for side_white in [false, true] {
            let b = Board::from_position(&pos);
            for w in [Weights::default(), candidates] {
                let f = features(&b.w, &b.b, side_white, b.quiet);
                let sum: i32 = f.iter().zip(w.as_array()).map(|(a, k)| a * k).sum();
                assert_eq!(evaluate(&b.w, &b.b, side_white, b.quiet, &w), sum);
                // Zero-sum: one side's terms are the other's, negated.
                let g = features(&b.w, &b.b, !side_white, b.quiet);
                for i in 0..NUM_FEATURES {
                    if i != 5 {
                        assert_eq!(f[i], -g[i], "term {i}");
                    }
                }
            }
        }
        let mv = pos.legal_moves()[pos.legal_moves().len() / 3];
        pos = pos.apply(mv);
    }
}
