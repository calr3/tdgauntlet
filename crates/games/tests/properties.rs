//! Invariants that must hold in every reachable position, checked along seeded random games.

use proptest::prelude::*;
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use tdg_games::epaminondas::{Epaminondas, Move, Position, BLACK, EMPTY, HEIGHT, N, WHITE, WIDTH};
use tdg_games::{Game, Outcome, Side};

/// Positions along a random game from the start, including the last (possibly terminal) one.
fn random_game(seed: u64, plies: usize) -> Vec<Position> {
    let g = Epaminondas;
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

/// A board transform, applied to squares, pieces and sides together.
struct Transform {
    square: fn(usize) -> usize,
    swap_colours: bool,
}

const REFLECT_LEFT_RIGHT: Transform = Transform {
    square: |s| (s / WIDTH) * WIDTH + (WIDTH - 1 - s % WIDTH),
    swap_colours: false,
};

const FLIP_AND_SWAP_COLOURS: Transform = Transform {
    square: |s| (HEIGHT - 1 - s / WIDTH) * WIDTH + s % WIDTH,
    swap_colours: true,
};

impl Transform {
    fn side(&self, side: Side) -> Side {
        if self.swap_colours {
            side.other()
        } else {
            side
        }
    }

    fn position(&self, p: &Position) -> Position {
        let mut board = [EMPTY; N];
        for s in 0..N {
            board[(self.square)(s)] = match (p.board[s], self.swap_colours) {
                (WHITE, true) => BLACK,
                (BLACK, true) => WHITE,
                (c, _) => c,
            };
        }
        Position {
            board,
            to_move: self.side(p.to_move),
            last_capturer: p.last_capturer.map(|s| self.side(s)),
            ..p.clone()
        }
    }

    fn mv(&self, m: Move) -> Move {
        let f = |s: u8| (self.square)(s as usize);
        Move::new(f(m.lead), f(m.rear), f(m.dest))
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn every_legal_move_applies_cleanly(seed in any::<u64>(), plies in 0usize..160) {
        let g = Epaminondas;
        for pos in random_game(seed, plies) {
            let moves = g.legal_moves(&pos);
            // The game offers moves exactly when it is not over.
            prop_assert_eq!(moves.is_empty(), g.outcome(&pos).is_some());
            let unique: HashSet<_> = moves.iter().collect();
            prop_assert_eq!(unique.len(), moves.len(), "duplicate moves");

            let mover = pos.to_move;
            for mv in &moves {
                let next = g.apply(&pos, mv);
                prop_assert!(next.board.iter().all(|&c| c <= BLACK));
                prop_assert_eq!(next.to_move, mover.other());
                prop_assert_eq!(next.moves, pos.moves + 1);
                // Moving never costs the mover a piece; it can only cost the opponent some.
                prop_assert_eq!(next.count(mover), pos.count(mover));
                let lost = pos.count(mover.other()) - next.count(mover.other());
                if lost > 0 {
                    prop_assert_eq!(next.quiet_moves, 0);
                    prop_assert_eq!(next.last_capturer, Some(mover));
                    // A capture takes a line strictly shorter than the moving group.
                    let group = if mv.is_single() { 1 } else {
                        let (l, r) = (mv.lead as i32, mv.rear as i32);
                        let (dr, dc) = (l / 14 - r / 14, l % 14 - r % 14);
                        dr.abs().max(dc.abs()) as usize + 1
                    };
                    prop_assert!(lost < group, "captured {} with a group of {}", lost, group);
                } else {
                    prop_assert_eq!(next.quiet_moves, pos.quiet_moves + 1);
                    prop_assert_eq!(next.last_capturer, pos.last_capturer);
                }
                // The mover's pieces end up on the destination (the lead lands there).
                prop_assert_eq!(next.board[mv.dest as usize], if mover == Side::First { WHITE } else { BLACK });
            }
        }
    }

    #[test]
    fn states_round_trip_through_the_wire(seed in any::<u64>(), plies in 0usize..160) {
        let g = Epaminondas;
        for pos in random_game(seed, plies) {
            let wire = g.encode_state(&pos);
            prop_assert_eq!(g.decode_state(&wire).unwrap(), pos.clone());
            for mv in g.legal_moves(&pos) {
                prop_assert_eq!(g.decode_move(&g.encode_move(&pos, &mv)).unwrap(), mv);
            }
        }
    }

    /// The rules do not care which way round the board is. A generator bug that treats one
    /// direction or one colour differently shows up here even where pgx vectors happen not to.
    #[test]
    fn the_rules_are_left_right_symmetric(seed in any::<u64>(), plies in 0usize..120) {
        let g = Epaminondas;
        let t = REFLECT_LEFT_RIGHT;
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

    #[test]
    fn the_rules_are_colour_symmetric(seed in any::<u64>(), plies in 0usize..120) {
        let g = Epaminondas;
        let t = FLIP_AND_SWAP_COLOURS;
        for pos in random_game(seed, plies) {
            let flipped = t.position(&pos);
            let mut expected: Vec<Move> = g.legal_moves(&pos).into_iter().map(|m| t.mv(m)).collect();
            let mut got = g.legal_moves(&flipped);
            expected.sort();
            got.sort();
            prop_assert_eq!(got, expected);
            // Outcomes swap sides too - except the one deliberate asymmetry, the final fallback
            // of the clock tiebreak, which goes to black whichever side that is.
            let fallback = pos.quiet_moves >= tdg_games::epaminondas::MAX_QUIET_MOVES
                && tdg_games::epaminondas::advancement_winner(&pos.board).is_none()
                && pos.last_capturer.is_none();
            if !fallback {
                prop_assert_eq!(g.outcome(&flipped), g.outcome(&pos).map(Outcome::flipped));
            }
        }
    }
}

#[test]
fn random_games_finish() {
    // The quiet-move clock guarantees termination; check it holds in practice with no cap on
    // plies beyond the theoretical worst case.
    let g = Epaminondas;
    for seed in 0..20 {
        let game = random_game(seed, 6000);
        let last = game.last().unwrap();
        assert!(
            g.outcome(last).is_some(),
            "seed {seed} did not finish in 6000 plies"
        );
    }
}
