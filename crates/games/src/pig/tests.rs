//! Unit tests for the rules. The conformance suite (`tests/pig_conformance.rs`) is what holds
//! them to pgx; these say in one position each what the rules are supposed to be.

use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn pos(to_move: Side, totals: [u32; 2], turn_total: u32, last_roll: u8) -> Position {
    Position {
        totals,
        turn_total,
        last_roll,
        to_move,
    }
}

#[test]
fn the_start_has_rolled_for_the_first_player() {
    let p = Position::start(4);
    assert_eq!(p, pos(Side::First, [0, 0], 4, 4));
    assert_eq!(p.legal_moves(), [Move::Stop, Move::Roll]);
    assert!(p.outcome().is_none());
}

#[test]
fn rolling_adds_to_the_turn_and_keeps_the_die() {
    let p = pos(Side::First, [20, 30], 5, 4).play(Move::Roll, 6);
    assert_eq!(p, pos(Side::First, [20, 30], 11, 6));
}

#[test]
fn stopping_banks_the_turn_and_rolls_for_the_other_player() {
    let p = pos(Side::Second, [30, 40], 7, 5).play(Move::Stop, 3);
    assert_eq!(p, pos(Side::First, [30, 47], 3, 3));
}

#[test]
fn a_one_loses_the_turn_total_and_forces_a_stop() {
    let p = pos(Side::First, [30, 40], 15, 1);
    assert!(p.pigged_out());
    assert_eq!(p.legal_moves(), [Move::Stop]);
    assert_eq!(p.play(Move::Stop, 2), pos(Side::Second, [30, 40], 2, 2));
}

#[test]
fn banking_the_target_wins_and_nothing_else_does() {
    let won = pos(Side::First, [99, 0], 3, 3).play(Move::Stop, 6);
    assert_eq!(won.outcome(), Some(Outcome::Win { side: Side::First }));
    assert_eq!(won.to_move, Side::Second, "pgx rolls for the loser anyway");
    assert!(Pig.legal_moves(&won).is_empty());
    assert_eq!(
        pos(Side::Second, [0, 97], 3, 3)
            .play(Move::Stop, 1)
            .outcome(),
        Some(Outcome::Win { side: Side::Second })
    );
    // One short, and the roll that comes with the stop is the next player's.
    assert!(pos(Side::First, [93, 50], 6, 3)
        .play(Move::Stop, 6)
        .outcome()
        .is_none());
    // Past the target but not banked: still going, and a 1 loses it all.
    let over = pos(Side::First, [90, 10], 8, 2).play(Move::Roll, 4);
    assert_eq!(over.turn_total, 12);
    assert!(over.outcome().is_none());
    let lost = over.play(Move::Roll, 1).play(Move::Stop, 5);
    assert_eq!(lost.totals, [90, 10]);
    assert!(lost.outcome().is_none());
}

#[test]
fn the_server_draws_every_face_of_the_die() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut seen = [false; FACES as usize + 1];
    let mut p = Pig.initial_with(&mut rng);
    seen[p.last_roll as usize] = true;
    for _ in 0..200 {
        if p.outcome().is_some() {
            p = Pig.initial_with(&mut rng);
        }
        let mv = *p.legal_moves().last().unwrap();
        p = Pig.apply_with(&p, &mv, &mut rng);
        seen[p.last_roll as usize] = true;
    }
    assert_eq!(seen, [false, true, true, true, true, true, true]);
}

#[test]
#[should_panic(expected = "apply_with")]
fn apply_without_dice_is_refused() {
    Pig.apply(&Position::start(3), &Move::Roll);
}

#[test]
fn states_round_trip_and_bad_ones_are_rejected() {
    let g = Pig;
    let p = pos(Side::Second, [42, 17], 9, 5);
    assert_eq!(
        g.encode_state(&p),
        json!({"totals": [42, 17], "turn_total": 9, "last_roll": 5, "to_move": 1})
    );
    assert_eq!(g.decode_state(&g.encode_state(&p)).unwrap(), p);
    let won = pos(Side::Second, [104, 17], 2, 2);
    assert_eq!(g.decode_state(&g.encode_state(&won)).unwrap(), won);

    let bad = |v: Value| g.decode_state(&v).unwrap_err().0;
    let state = |totals: [i64; 2], turn: i64, roll: i64, to_move: i64| json!({"totals": totals, "turn_total": turn, "last_roll": roll, "to_move": to_move});
    assert!(bad(state([0, 0], 7, 0, 0)).contains("last_roll"));
    assert!(bad(state([0, 0], 7, 7, 0)).contains("last_roll"));
    assert!(bad(state([-1, 0], 7, 4, 0)).contains("totals"));
    assert!(bad(state([0, 0], 3, 4, 0)).contains("turn_total"));
    assert!(
        bad(state([0, 0], 5, 4, 0)).contains("turn_total"),
        "a 1 ends a turn"
    );
    assert!(bad(state([100, 0], 4, 4, 0)).contains("just stopped"));
    assert!(bad(state([100, 100], 4, 4, 1)).contains("just stopped"));
    assert!(bad(state([0, 0], 4, 4, 2)).contains("to_move"));
    assert!(
        bad(json!({"totals": [0], "turn_total": 4, "last_roll": 4, "to_move": 0}))
            .contains("totals")
    );
}

#[test]
fn moves_round_trip_in_notation_and_json() {
    let g = Pig;
    let p = Position::start(3);
    for mv in [Move::Stop, Move::Roll] {
        assert_eq!(Move::parse(mv.name()), Some(mv));
        assert_eq!(Move::from_action(mv.action()), Some(mv));
        assert_eq!(g.decode_move(&g.encode_move(&p, &mv)).unwrap(), mv);
    }
    assert_eq!(
        g.encode_move(&p, &Move::Roll),
        json!({"action": "roll", "san": "roll"})
    );
    assert_eq!(Move::Stop.action(), 0);
    assert!(g.decode_move(&json!({"action": "hold"})).is_err());
    assert!(g.decode_move(&json!({"action": 1})).is_err());
}

#[test]
fn the_diagram_says_whose_turn_and_what_is_at_stake() {
    let text = Pig.render(&pos(Side::Second, [42, 17], 9, 5));
    assert!(text.contains("P0 42") && text.contains("P1 17"), "{text}");
    assert!(
        text.contains("P1 to move: rolled 5, turn total 9"),
        "{text}"
    );
    let text = Pig.render(&pos(Side::First, [42, 17], 9, 1));
    assert!(
        text.contains("pig out") && text.contains("9 is lost"),
        "{text}"
    );
    let text = Pig.render(&pos(Side::First, [42, 101], 3, 3));
    assert!(text.contains("P1 won"), "{text}");
}

#[test]
fn the_rules_documents_examples_are_what_the_engine_reads_and_writes() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../games/pig.md");
    let doc = std::fs::read_to_string(path).unwrap();
    let blocks: Vec<Value> = doc
        .split("```json")
        .skip(1)
        .map(|b| serde_json::from_str(b.split("```").next().unwrap()).unwrap())
        .collect();
    let g = Pig;
    let p = g.decode_state(&blocks[0]).unwrap();
    assert_eq!(p, pos(Side::Second, [42, 17], 9, 5));
    assert_eq!(g.encode_state(&p), blocks[0]);
    assert_eq!(g.encode_move(&p, &Move::Roll), blocks[1]);
}
