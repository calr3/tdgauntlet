//! Piece pattern classes and counts.

use tdg_games::gess::Position;
use tdg_gess_negamax::{
    evaluate, pattern_class, pattern_counts, Board, Patterns, Weights, NUM_PATTERNS,
};

/// The pattern of a ring: all eight edge cells, no centre.
const RING: usize = 0b111_101_111;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Positions from a few random games, stopping each before it ends.
fn positions() -> Vec<Position> {
    let mut rng = 7u64;
    let mut out = Vec::new();
    for _ in 0..20 {
        let mut pos = Position::start();
        for _ in 0..60 {
            if pos.outcome().is_some() {
                break;
            }
            out.push(pos.clone());
            let legal = pos.legal_moves();
            pos = pos.apply(legal[(splitmix(&mut rng) % legal.len() as u64) as usize]);
        }
    }
    out
}

#[test]
fn classes_merge_each_pattern_with_its_left_right_mirror_only() {
    let t = pattern_class();
    let mirror = |p: usize| {
        (0..3).fold(0, |m, row| {
            m | ((p >> (row * 3) & 7).reverse_bits() >> 61) << (row * 3)
        })
    };
    let up_down = |p: usize| (p & 0o070) | (p & 0o007) << 6 | (p & 0o700) >> 6;
    assert_eq!(*t.iter().max().unwrap() as usize, NUM_PATTERNS - 1);
    for p in 0..512 {
        assert_eq!(t[p], t[mirror(p)], "pattern {p:o}");
    }
    // A piece with a stone only on its forward edge is not the same as one only on its back edge.
    assert_ne!(t[0o007], t[up_down(0o007)]);
    // fit_eval.py numbers the classes the same way (it leaves the ring's out of its fit).
    assert_eq!(t[RING], 281);
}

#[test]
fn the_start_is_symmetric_and_counts_flip_with_the_side() {
    let start = Board::from_position(&Position::start());
    assert_eq!(pattern_counts(&start.w, &start.b, true), [0; NUM_PATTERNS]);
    for pos in positions() {
        let b = Board::from_position(&pos);
        let (w, k) = (
            pattern_counts(&b.w, &b.b, true),
            pattern_counts(&b.w, &b.b, false),
        );
        assert!(w.iter().zip(k.iter()).all(|(x, y)| *x == -*y));
    }
}

#[test]
fn the_ring_class_counts_the_rings() {
    let ring = pattern_class()[RING] as usize;
    let mut seen = 0;
    // Random play never leaves the sides with different numbers of rings (losing one ends the
    // game), so each position also gets a second white ring on the empty middle rows.
    for pos in positions() {
        let mut b = Board::from_position(&pos);
        if (8..=10).all(|r| (b.w[r] | b.b[r]) & 0b111 << 8 == 0) {
            b.w[8] |= 0b111 << 8;
            b.w[9] |= 0b101 << 8;
            b.w[10] |= 0b111 << 8;
        }
        let f = b.features();
        assert_eq!(b.pattern_counts()[ring], f[1], "rings differ");
        seen += (f[1] != 0) as i32;
    }
    assert!(seen > 0, "no position with unequal rings");
}

#[test]
fn pattern_weights_add_to_the_evaluation_and_round_trip() {
    let mut patterns = Patterns::default();
    for (i, p) in patterns.0.iter_mut().enumerate() {
        *p = (i as i32 % 7) - 3;
    }
    let weights = Weights {
        patterns,
        ..Weights::default()
    };
    for pos in positions().into_iter().step_by(17) {
        let b = Board::from_position(&pos);
        let n = b.pattern_counts();
        let extra: i32 = n.iter().zip(patterns.0.iter()).map(|(a, w)| a * w).sum();
        let plain = evaluate(&b.w, &b.b, b.white_to_move, b.quiet, &Weights::default());
        assert_eq!(
            evaluate(&b.w, &b.b, b.white_to_move, b.quiet, &weights),
            plain + extra
        );
    }
    let json = serde_json::to_string(&weights).unwrap();
    assert_eq!(serde_json::from_str::<Weights>(&json).unwrap(), weights);
    // Weights without patterns write no pattern list, so older files stay as they were.
    assert!(!serde_json::to_string(&Weights::default())
        .unwrap()
        .contains("patterns"));
}
