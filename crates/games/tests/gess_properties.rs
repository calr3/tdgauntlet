//! Invariants that must hold in every reachable Gess position, checked along seeded random games.
//!
//! The conformance vectors say the rules match pgx where pgx has been asked; these say the things
//! that must hold everywhere, and they catch the kind of bug a vector set can walk straight past:
//! a direction indexed the wrong way round, or one colour treated unlike the other.

use proptest::prelude::*;
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use tdg_games::gess::{Gess, Move, Position, BLACK, EMPTY, MIN_IDX, N, SIZE, WHITE};
use tdg_games::{Game, Outcome};

/// Positions along a random game from the start, including the last (possibly terminal) one.
fn random_game(seed: u64, plies: usize) -> Vec<Position> {
    let g = Gess;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pos = Position::start();
    let mut out = vec![pos.clone()];
    for _ in 0..plies {
        let moves = g.legal_moves(&pos);
        let Some(mv) = moves.choose(&mut rng) else {
            break;
        };
        pos = g.apply(&pos, mv);
        out.push(pos.clone());
    }
    out
}

/// A cell transform, applied to the board and to moves together.
struct Transform {
    cell: fn(usize) -> usize,
    swap_colours: bool,
}

const MIRROR_LEFT_RIGHT: Transform = Transform {
    cell: |s| (s / SIZE) * SIZE + (SIZE - 1 - s % SIZE),
    swap_colours: false,
};

const FLIP_AND_SWAP_COLOURS: Transform = Transform {
    cell: |s| (SIZE - 1 - s / SIZE) * SIZE + s % SIZE,
    swap_colours: true,
};

impl Transform {
    fn position(&self, p: &Position) -> Position {
        let mut board = [EMPTY; N];
        for s in 0..N {
            board[(self.cell)(s)] = match (p.board[s], self.swap_colours) {
                (BLACK, true) => WHITE,
                (WHITE, true) => BLACK,
                (c, _) => c,
            };
        }
        Position {
            board,
            to_move: if self.swap_colours {
                p.to_move.other()
            } else {
                p.to_move
            },
            ..p.clone()
        }
    }

    fn mv(&self, m: Move) -> Move {
        Move::new((self.cell)(m.from as usize), (self.cell)(m.to as usize))
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn every_legal_move_applies_cleanly(seed in any::<u64>(), plies in 0usize..40) {
        let g = Gess;
        for pos in random_game(seed, plies) {
            let moves = g.legal_moves(&pos);
            // The game offers moves exactly when it is not over. In Gess that needs no stalemate
            // rule: a side with a ring can always play its centre.
            prop_assert_eq!(moves.is_empty(), g.outcome(&pos).is_some());
            let unique: HashSet<_> = moves.iter().collect();
            prop_assert_eq!(unique.len(), moves.len(), "duplicate moves");

            let mover = pos.to_move;
            for mv in &moves {
                let next = g.apply(&pos, mv);
                prop_assert!(next.board.iter().all(|&c| c <= WHITE));
                prop_assert_eq!(next.to_move, mover.other());
                prop_assert_eq!(next.moves, pos.moves + 1);
                // The border ring never holds a stone.
                for s in 0..N {
                    let (r, c) = (s / SIZE, s % SIZE);
                    let border = !(MIN_IDX..SIZE - 1).contains(&r)
                        || !(MIN_IDX..SIZE - 1).contains(&c);
                    prop_assert!(!border || next.board[s] == EMPTY);
                }
                // A move never puts a stone on the board, and it can only cost the opponent
                // stones that stood where it landed.
                prop_assert!(next.count(mover) <= pos.count(mover));
                prop_assert!(next.count(mover.other()) <= pos.count(mover.other()));
                let taken = (pos.count(mover) - next.count(mover))
                    + (pos.count(mover.other()) - next.count(mover.other()));
                if taken > 0 {
                    prop_assert_eq!(next.quiet_moves, 0);
                } else {
                    prop_assert_eq!(next.quiet_moves, pos.quiet_moves + 1);
                }
            }
        }
    }

    #[test]
    fn states_round_trip_through_the_wire(seed in any::<u64>(), plies in 0usize..40) {
        let g = Gess;
        for pos in random_game(seed, plies) {
            let wire = g.encode_state(&pos);
            prop_assert_eq!(g.decode_state(&wire).unwrap(), pos.clone());
            for mv in g.legal_moves(&pos).iter().take(40) {
                prop_assert_eq!(g.decode_move(&g.encode_move(&pos, mv)).unwrap(), *mv);
            }
        }
    }

    /// The rules do not care which way round the board is.
    #[test]
    fn the_rules_are_left_right_symmetric(seed in any::<u64>(), plies in 0usize..30) {
        let g = Gess;
        let t = MIRROR_LEFT_RIGHT;
        for pos in random_game(seed, plies) {
            let mirrored = t.position(&pos);
            let mut expected: Vec<Move> = g.legal_moves(&pos).into_iter().map(|m| t.mv(m)).collect();
            let mut got = g.legal_moves(&mirrored);
            expected.sort();
            got.sort();
            prop_assert_eq!(got, expected);
            prop_assert_eq!(g.outcome(&mirrored), g.outcome(&pos));
        }
    }

    /// Nor which colour is which: Gess has no rule that favours a side, not even its tiebreak.
    #[test]
    fn the_rules_are_colour_symmetric(seed in any::<u64>(), plies in 0usize..30) {
        let g = Gess;
        let t = FLIP_AND_SWAP_COLOURS;
        for pos in random_game(seed, plies) {
            let flipped = t.position(&pos);
            let mut expected: Vec<Move> = g.legal_moves(&pos).into_iter().map(|m| t.mv(m)).collect();
            let mut got = g.legal_moves(&flipped);
            expected.sort();
            got.sort();
            prop_assert_eq!(got, expected);
            prop_assert_eq!(g.outcome(&flipped), g.outcome(&pos).map(Outcome::flipped));
        }
    }
}

#[test]
fn random_games_finish() {
    // The quiet-move clock guarantees termination: 20 moves without a stone leaving the board end
    // it, and only so many stones can leave.
    let g = Gess;
    for seed in 0..20 {
        let game = random_game(seed, 4000);
        let last = game.last().unwrap();
        assert!(
            g.outcome(last).is_some(),
            "seed {seed} did not finish in 4000 plies"
        );
    }
}
