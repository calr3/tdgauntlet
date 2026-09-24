//! Unit tests for the rules. The conformance suite (`tests/go_conformance.rs`) is what holds them
//! to pgx; these say in one position each what the rules are supposed to be.

use super::*;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::SeedableRng;

fn mv(text: &str) -> Move {
    parse_move(text).unwrap_or_else(|| panic!("{text} is not a move"))
}

/// A position from moves written in notation, played in turn from the start.
fn play(moves: &[&str]) -> Position {
    moves.iter().fold(Position::start(), |p, &m| {
        assert!(p.outcome().is_none(), "the game is over before {m}");
        assert!(
            p.legal_moves().contains(&mv(m)),
            "{m} is illegal\n{}",
            render(&p)
        );
        p.apply(mv(m))
    })
}

/// Black's stones and white's, placed alternately with the side that has run out passing, then
/// `to_move` to play. Nothing may be captured on the way.
fn setup(black: &str, white: &str, to_move: Side) -> Position {
    let (b, w): (Vec<&str>, Vec<&str>) = (
        black.split_whitespace().collect(),
        white.split_whitespace().collect(),
    );
    let mut moves = Vec::new();
    for i in 0..b.len().max(w.len()) {
        moves.push(*b.get(i).unwrap_or(&"pass"));
        moves.push(*w.get(i).unwrap_or(&"pass"));
    }
    if moves.len() >= 2 && moves[moves.len() - 1] == "pass" && moves[moves.len() - 2] == "pass" {
        unreachable!("both lists ran out together");
    }
    if moves.last() == Some(&"pass") {
        moves.pop();
    }
    let mut p = play(&moves);
    if p.to_move != to_move {
        p = p.apply(Move::Pass);
    }
    let stones = |c| p.board.iter().filter(|&&v| v == c).count();
    assert_eq!(
        (stones(1), stones(-1)),
        (b.len(), w.len()),
        "setup captured"
    );
    p
}

fn at(p: &Position, point: &str) -> i8 {
    let Move::Play(q) = mv(point) else {
        panic!("{point} is a pass")
    };
    p.board[q as usize]
}

#[test]
fn notation_is_files_without_i_and_ranks_from_the_bottom() {
    assert_eq!(mv("A1"), Move::Play(0));
    assert_eq!(mv("D4"), Move::Play(60));
    assert_eq!(mv("j1"), Move::Play(8), "J follows H");
    assert_eq!(mv("T19"), Move::Play(360));
    assert_eq!(mv("Pass"), Move::Pass);
    assert_eq!(parse_move("I5"), None);
    assert_eq!(parse_move("A20"), None);
    assert_eq!(parse_move("A0"), None);
    for p in 0..N as u16 {
        assert_eq!(parse_move(&describe(Move::Play(p))), Some(Move::Play(p)));
    }
    assert_eq!(Move::from_action(PASS), Some(Move::Pass));
    assert_eq!(Move::from_action(362), None);
}

#[test]
fn the_start_is_an_empty_board_with_every_point_and_the_pass() {
    let p = Position::start();
    assert_eq!(p.to_move, Side::First);
    let moves = p.legal_moves();
    assert_eq!(moves.len(), N + 1);
    assert_eq!(moves.last(), Some(&Move::Pass));
    assert!(p.outcome().is_none());
}

#[test]
fn a_stone_with_no_liberties_is_captured() {
    // White's A1 has one liberty, A2.
    let p = setup("B1", "A1", Side::First);
    let p = p.apply(mv("A2"));
    assert_eq!(at(&p, "A1"), 0);
    assert_eq!(
        p.ko, None,
        "black's A2 has a liberty at A3, so white cannot take back"
    );
    // A chain goes all at once.
    let p = setup("A3 B2", "A1 A2 B1", Side::First).apply(mv("C1"));
    assert_eq!((at(&p, "A1"), at(&p, "A2"), at(&p, "B1")), (0, 0, 0));
    assert_eq!(p.ko, None);
}

#[test]
fn suicide_is_illegal_unless_it_captures() {
    // A1 is surrounded by black: white may not play there.
    let p = setup("A2 B1", "T19", Side::Second);
    assert!(!p.is_legal(mv("A1")));
    assert!(!p.legal_moves().contains(&mv("A1")));
    // Filling a chain's last liberty is suicide too.
    let p = setup("A3 B2 C1", "A2 B1", Side::Second);
    assert!(
        !p.is_legal(mv("A1")),
        "white A1 would leave A2 B1 A1 with no liberty"
    );
    // But if it captures, it is a capture: black's B1 has only A1 left.
    let p = setup("A2 B1", "A3 B2 C1", Side::Second);
    assert!(p.is_legal(mv("A1")));
    let q = p.apply(mv("A1"));
    assert_eq!((at(&q, "A2"), at(&q, "B1"), at(&q, "A1")), (0, 0, -1));
    // And own-colour neighbours with a liberty to spare make it no suicide at all.
    let p = setup("A2 B1", "T19", Side::First);
    assert!(p.is_legal(mv("A1")));
}

/// A ko: black's B4 C5 C3 around C4 and white's D5 D3 E4 around D4, with a white stone on C4
/// that black can take by playing D4.
fn ko() -> Position {
    setup("B4 C5 C3", "D5 D3 E4 C4", Side::First)
}

#[test]
fn a_ko_cannot_be_taken_straight_back() {
    let p = ko().apply(mv("D4"));
    assert_eq!(at(&p, "C4"), 0);
    assert_eq!(p.ko, Some(60 - 1), "C4 is barred");
    assert!(!p.is_legal(mv("C4")));
    assert!(!p.legal_moves().contains(&mv("C4")));
    // After an exchange elsewhere, it may.
    let p = p.apply(mv("Q16")).apply(mv("R16"));
    assert_eq!(p.ko, None);
    assert!(p.is_legal(mv("C4")));
    let p = p.apply(mv("C4"));
    assert_eq!((at(&p, "D4"), p.ko), (0, Some(60)));
    // A pass also lifts it.
    let p = ko().apply(mv("D4")).apply(Move::Pass).apply(Move::Pass);
    assert!(p.outcome().is_some(), "two passes end the game");
}

#[test]
fn taking_two_stones_or_leaving_liberties_is_no_ko() {
    // Black takes C4 with D4, but D4 also touches an empty E4: no ko.
    let p = setup("B4 C5 C3", "D5 D3 C4", Side::First).apply(mv("D4"));
    assert_eq!((at(&p, "C4"), p.ko), (0, None));
    // Two stones at once.
    let p = setup("B4 B5 C6 C3 D5", "C4 C5 D3 E4", Side::First).apply(mv("D4"));
    assert_eq!((at(&p, "C4"), at(&p, "C5"), p.ko), (0, 0, None));
}

/// Two kos, the first with a white stone to take and the second a black one, as in
/// `conformance/dump_go.py`.
fn double_ko() -> Position {
    setup(
        "E6 E4 D5 N14 N12 M13 O13",
        "F6 F4 G5 O14 O12 P13 E5",
        Side::First,
    )
}

#[test]
fn repeating_a_position_ends_the_game_and_loses_it() {
    let start = double_ko();
    // A capture each, a pass to lift the ban, a capture each: black's last capture puts the
    // board back as it was.
    let mut p = start.clone();
    for m in ["F5", "N13", "pass", "E5"] {
        p = p.apply(mv(m));
        assert!(p.outcome().is_none());
    }
    assert!(p.is_legal(mv("O13")), "superko is not a ban, it is a loss");
    let p = p.apply(mv("O13"));
    assert_eq!(p.board, start.board);
    assert!(p.superko());
    assert_eq!(p.outcome(), Some(Outcome::win(Side::Second)));
    assert!(Go.legal_moves(&p).is_empty());
    // A pass never repeats: the position after one is always the one before.
    assert!(!start.apply(Move::Pass).superko());
}

#[test]
fn a_repeat_by_white_loses_for_white() {
    let mut p = double_ko();
    for m in ["pass", "N13", "F5", "pass", "O13", "E5"] {
        assert!(p.outcome().is_none(), "before {m}");
        p = p.apply(mv(m));
    }
    assert_eq!(p.board, double_ko().board);
    assert_eq!(p.outcome(), Some(Outcome::win(Side::First)));
}

#[test]
fn area_scoring_counts_stones_and_what_only_they_reach() {
    // An empty board is 361 each, and komi gives it to white.
    let p = play(&["pass", "pass"]);
    assert_eq!(p.score(), (361, 361));
    assert_eq!(p.outcome(), Some(Outcome::win(Side::Second)));
    // One stone reaches everything.
    let p = play(&["K10", "pass", "pass"]);
    assert_eq!(p.score(), (361, 0));
    assert_eq!(p.outcome(), Some(Outcome::win(Side::First)));
    // Black walls off the A1 corner: A1 A2 B1 reach only black, so black has them and its wall
    // A3 B2 C1. The rest reaches both colours and counts for neither: white has just its stone.
    let p = setup("A3 B2 C1", "T19", Side::First);
    assert_eq!(p.score(), (6, 1));
    // Empty points reaching both count for neither.
    let p = setup("A1", "T19", Side::First);
    assert_eq!(p.score(), (1, 1));
}

#[test]
fn komi_is_seven_and_a_half() {
    let full = |black: usize| {
        let mut p = Position::start();
        for (q, v) in p.board.iter_mut().enumerate() {
            *v = if q < black { 1 } else { -1 };
        }
        p.passes = 2;
        p
    };
    assert_eq!(full(184).score(), (184, 177));
    assert_eq!(
        full(184).outcome(),
        Some(Outcome::win(Side::Second)),
        "7 < 7.5"
    );
    assert_eq!(
        full(185).outcome(),
        Some(Outcome::win(Side::First)),
        "9 > 7.5"
    );
}

#[test]
fn two_passes_end_the_game_but_one_does_not() {
    let p = play(&["D4", "pass"]);
    assert_eq!(p.passes(), 1);
    assert!(p.outcome().is_none());
    let p = p.apply(mv("Q16"));
    assert_eq!(p.passes(), 0);
    let p = p.apply(Move::Pass).apply(Move::Pass);
    assert!(p.outcome().is_some());
    assert!(Go.legal_moves(&p).is_empty());
}

#[test]
fn the_game_ends_at_move_722() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut p = Position::start();
    while p.history.len() < MAX_MOVES - 1 {
        let moves: Vec<Move> = p
            .legal_moves()
            .into_iter()
            .filter(|m| *m != Move::Pass)
            .collect();
        p = p.apply(*moves.choose(&mut rng).unwrap_or(&Move::Pass));
        assert!(
            p.outcome().is_none(),
            "over early after {} moves",
            p.history.len()
        );
    }
    let p = p.apply(p.legal_moves()[0]);
    assert!(p.outcome().is_some());
}

#[test]
fn the_labelled_move_list_agrees_with_the_one_point_check() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..3 {
        let mut p = Position::start();
        while p.outcome().is_none() {
            let moves = p.legal_moves();
            let all = (0..N as u16).map(Move::Play).chain([Move::Pass]);
            let checked: Vec<Move> = all.filter(|&m| p.is_legal(m)).collect();
            assert_eq!(moves, checked, "\n{}", render(&p));
            p = p.apply(*moves.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn states_round_trip_and_bad_ones_are_rejected() {
    let g = Go;
    let p = ko().apply(mv("D4"));
    let wire = g.encode_state(&p);
    assert_eq!(g.decode_state(&wire).unwrap(), p);

    let bad = |edit: &dyn Fn(&mut Value)| {
        let mut v = wire.clone();
        edit(&mut v);
        g.decode_state(&v).unwrap_err().0
    };
    assert!(bad(&|v| v["board"][0] = json!(1)).contains("board: A1 is 1"));
    assert!(bad(&|v| v["board"][0] = json!(2)).contains("not -1, 0 or 1"));
    assert!(bad(&|v| v["to_move"] = json!(0)).contains("to_move"));
    assert!(bad(&|v| v["ko"] = json!(null)).contains("ko"));
    assert!(bad(&|v| v["history"].as_array_mut().unwrap().push(json!(59))).contains("illegal"));
    assert!(bad(&|v| v["history"][0] = json!(362)).contains("not an action"));
    assert!(
        bad(&|v| v["board"].as_array_mut().unwrap().pop().map(drop).unwrap())
            .contains("expected 361")
    );
    let over = g.encode_state(&play(&["pass", "pass"]));
    let mut late = over.clone();
    late["history"].as_array_mut().unwrap().push(json!(PASS));
    assert!(g
        .decode_state(&late)
        .unwrap_err()
        .0
        .contains("already over"));
}

#[test]
fn moves_round_trip_in_json() {
    let g = Go;
    let p = Position::start();
    for m in p.legal_moves() {
        assert_eq!(g.decode_move(&g.encode_move(&p, &m)).unwrap(), m);
    }
    assert_eq!(
        g.decode_move(&json!({"point": 60})).unwrap(),
        Move::Play(60)
    );
    assert_eq!(g.decode_move(&json!({"pass": true})).unwrap(), Move::Pass);
    for bad in [
        json!({"point": 361}),
        json!({"point": -1}),
        json!({"pass": true, "point": 3}),
        json!({"pass": false}),
        json!({"pass": "yes"}),
        json!({}),
    ] {
        assert!(g.decode_move(&bad).is_err(), "{bad}");
    }
}

#[test]
fn the_diagram_shows_stones_the_last_move_and_the_ko() {
    let p = ko().apply(mv("D4"));
    let text = Go.render(&p);
    assert!(text.contains("(X)"), "{text}");
    assert!(text.contains("Ko: C4 is barred"), "{text}");
    assert!(text.contains("White (O) to move"), "{text}");
    assert!(
        text.contains(" 4 ") && text.contains(" A B C D E F G H J K"),
        "{text}"
    );
    // The file letters stand over their points.
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0].find('A'), Some(4), "{text}");
    assert_eq!(lines[SIZE].get(..5), Some(" 1  ."), "{text}");
}

#[test]
fn the_rules_documents_examples_are_what_the_engine_reads_and_writes() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../games/go-19x19.md");
    let doc = std::fs::read_to_string(path).unwrap();
    let blocks: Vec<Value> = doc
        .split("```json")
        .skip(1)
        .map(|b| serde_json::from_str(b.split("```").next().unwrap()).unwrap())
        .collect();
    let g = Go;
    let p = play(&["D4", "Q16", "pass"]);
    assert_eq!(g.decode_state(&blocks[0]).unwrap(), p);
    assert_eq!(g.encode_move(&p, &mv("D16")), blocks[1]);
    assert_eq!(g.encode_move(&p, &Move::Pass), blocks[2]);
}
