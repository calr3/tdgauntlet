//! The port's move generation and game outcomes agree with tdgauntlet's own Gess rules
//! (`tdg_games::gess`, which is conformance-tested against pgx), over random games that keep
//! going long enough to reach captures, broken rings and the quiet-move rule.

use std::collections::BTreeMap;
use tdg_games::gess::{Move, Position};
use tdg_gess_negamax::{Board, Searcher, Weights};
use tdg_proto::Outcome;

/// The engine's outcome code (mover's view) for tdgauntlet's outcome after a move.
fn code(after: &Position, mover: tdg_proto::Side) -> i32 {
    match after.outcome() {
        None => 0,
        Some(Outcome::Draw) => 1,
        Some(Outcome::Win { side }) if side == mover => 2,
        Some(Outcome::Win { .. }) => -1,
    }
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn moves_and_outcomes_match_tdgauntlets_rules() {
    let mut s = Searcher::new(Weights::default());
    let mut rng = 7u64;
    let (mut positions, mut moves_checked, mut ends) = (0, 0, BTreeMap::new());
    for _game in 0..40 {
        let mut pos = Position::start();
        for _ply in 0..120 {
            if pos.outcome().is_some() {
                break;
            }
            let mover = pos.to_move;
            let ours: BTreeMap<(usize, usize), i32> = s
                .root_moves(&Board::from_position(&pos))
                .into_iter()
                .map(|(f, t, o, _)| ((f, t), o))
                .collect();
            let legal = pos.legal_moves();
            let theirs: BTreeMap<(usize, usize), i32> = legal
                .iter()
                .map(|m| {
                    (
                        (m.from as usize, m.to as usize),
                        code(&pos.apply(*m), mover),
                    )
                })
                .collect();
            assert_eq!(ours, theirs, "{pos:?}");
            positions += 1;
            moves_checked += legal.len();
            for o in theirs.values() {
                *ends.entry(*o).or_insert(0usize) += 1;
            }
            // Mostly avoid moves that lose on the spot, so games last; in every other game also
            // prefer moves that capture nothing, so the quiet-move rule decides some of them.
            let quiet_game = _game % 2 == 1;
            let mut keep: Vec<&Move> = legal
                .iter()
                .filter(|m| code(&pos.apply(**m), mover) != -1)
                .collect();
            if quiet_game {
                let quiet: Vec<&Move> = keep
                    .iter()
                    .copied()
                    .filter(|m| pos.apply(**m).quiet_moves > pos.quiet_moves)
                    .collect();
                if !quiet.is_empty() {
                    keep = quiet;
                }
            }
            let pick = if keep.is_empty() || splitmix(&mut rng) % 50 == 0 {
                legal[(splitmix(&mut rng) % legal.len() as u64) as usize]
            } else {
                *keep[(splitmix(&mut rng) % keep.len() as u64) as usize]
            };
            pos = pos.apply(pick);
        }
    }
    assert!(positions > 2000, "{positions}");
    // Every kind of outcome was exercised.
    for code in [-1, 0, 1, 2] {
        assert!(
            ends.get(&code).copied().unwrap_or(0) > 0,
            "no move with outcome {code}: {ends:?} over {moves_checked}"
        );
    }
}
