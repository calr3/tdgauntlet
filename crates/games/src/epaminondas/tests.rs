//! Per-rule tests, each from a hand-built position. Squares are written in notation (`sq("c3")`)
//! so a failing test reads the way the board does.

use super::*;
use crate::{DynGame, Game, MoveError};

fn sq(label: &str) -> usize {
    parse_square(label).unwrap_or_else(|| panic!("bad square {label}"))
}

fn sqs(labels: &str) -> Vec<usize> {
    labels.split_whitespace().map(sq).collect()
}

fn pos(to_move: Side, white: &str, black: &str) -> Position {
    Position::from_pieces(to_move, &sqs(white), &sqs(black))
}

fn mv(text: &str) -> Move {
    parse_move(text).unwrap_or_else(|| panic!("bad move {text}"))
}

/// Legal moves as sorted notation, which makes assertions and failures readable.
fn legal(p: &Position) -> Vec<String> {
    let mut v: Vec<String> = p
        .legal_moves()
        .iter()
        .map(|&m| notation::describe(p, m))
        .collect();
    v.sort();
    v
}

fn is_legal(p: &Position, text: &str) -> bool {
    p.legal_moves().contains(&mv(text))
}

// ─── Notation ────────────────────────────────────────────────────────────────

#[test]
fn squares_are_labelled_from_whites_home_corner() {
    assert_eq!(square_label(0), "a1");
    assert_eq!(square_label(13), "n1");
    assert_eq!(square_label(14), "a2");
    assert_eq!(square_label(167), "n12");
    for s in 0..N {
        assert_eq!(parse_square(&square_label(s)), Some(s));
    }
    for bad in ["", "a", "o1", "a0", "a13", "a1x", "1a", "A1b"] {
        assert_eq!(parse_square(bad), None, "{bad}");
    }
}

#[test]
fn move_notation_round_trips() {
    assert_eq!(mv("c2-c3"), Move::new(sq("c2"), sq("c2"), sq("c3")));
    // Rear first, then lead.
    assert_eq!(mv("c1c2-c4"), Move::new(sq("c2"), sq("c1"), sq("c4")));
    assert_eq!(
        mv("C1C2XC4"),
        mv("c1c2-c4"),
        "case and the separator are not significant"
    );
    assert_eq!(mv("a11a12-a10"), Move::new(sq("a12"), sq("a11"), sq("a10")));
    for bad in ["", "c2", "c2c3", "z1-a1", "c2-", "-c3", "c2-c13"] {
        assert_eq!(parse_move(bad), None, "{bad}");
    }
    // Every legal move from the start describes itself in a way that parses back.
    let p = Position::start();
    for m in p.legal_moves() {
        assert_eq!(parse_move(&notation::describe(&p, m)), Some(m));
    }
}

// ─── The starting position ───────────────────────────────────────────────────

#[test]
fn start_position() {
    let p = Position::start();
    assert_eq!(p.count(Side::First), 28);
    assert_eq!(p.count(Side::Second), 28);
    assert!(sqs("a1 n1 a2 n2").iter().all(|&s| p.board[s] == WHITE));
    assert!(sqs("a11 n11 a12 n12").iter().all(|&s| p.board[s] == BLACK));
    assert_eq!(p.to_move, Side::First);
    assert_eq!(p.outcome(), None);
    assert!(is_mirror_symmetric(&p.board));
}

// ─── Single pieces ───────────────────────────────────────────────────────────

#[test]
fn a_single_piece_steps_one_square_in_any_direction() {
    let p = pos(Side::First, "f6", "a12");
    assert_eq!(
        legal(&p),
        ["f6-e5", "f6-e6", "f6-e7", "f6-f5", "f6-f7", "f6-g5", "f6-g6", "f6-g7"]
    );
}

#[test]
fn a_single_piece_in_a_corner_has_three_moves() {
    let p = pos(Side::First, "a1", "n12");
    assert_eq!(legal(&p), ["a1-a2", "a1-b1", "a1-b2"]);
}

#[test]
fn a_single_piece_cannot_capture() {
    // An enemy line is never shorter than one piece.
    let p = pos(Side::First, "f6", "f7 a12");
    assert!(!is_legal(&p, "f6-f7"));
    assert!(!legal(&p).iter().any(|m| m.contains('x')));
}

#[test]
fn a_single_piece_cannot_move_onto_a_friend() {
    let p = pos(Side::First, "f6 f7", "a12");
    assert!(!is_legal(&p, "f6-f7"));
}

// ─── Phalanxes ───────────────────────────────────────────────────────────────

#[test]
fn a_phalanx_moves_up_to_as_many_squares_as_it_has_pieces() {
    let p = pos(Side::First, "c1 c2 c3", "n12");
    for k in 1..=3 {
        let dest = format!("c{}", 3 + k);
        assert!(
            is_legal(&p, &format!("c1c3-{dest}")),
            "3 pieces, {k} squares"
        );
    }
    assert!(!is_legal(&p, "c1c3-c7"), "3 pieces may not move 4 squares");
    // The two-piece sub-phalanx led by c3 may move at most two.
    assert!(is_legal(&p, "c2c3-c5"));
    assert!(!is_legal(&p, "c2c3-c6"));
}

#[test]
fn a_phalanx_can_move_backwards_by_leading_with_the_other_end() {
    let p = pos(Side::First, "c4 c5 c6", "n12");
    assert!(is_legal(&p, "c6c4-c1"), "led by c4 heading down the board");
    assert!(is_legal(&p, "c4c6-c9"), "led by c6 heading up the board");
}

#[test]
fn a_phalanx_may_split_but_only_as_a_run_including_the_lead() {
    let p = pos(Side::First, "c1 c2 c3 c4", "n12");
    // Led by c4: the rear may be c3, c2 or c1.
    assert!(is_legal(&p, "c3c4-c5"));
    assert!(is_legal(&p, "c2c4-c6"));
    assert!(is_legal(&p, "c1c4-c8"));
    // A rear that is not in line behind the lead is not a phalanx.
    assert!(!is_legal(&p, "d3c4-c5"));
    assert!(
        !is_legal(&p, "c1c3-c4"),
        "c3 cannot lead upwards: c4 is in the way"
    );
}

#[test]
fn phalanxes_move_in_every_direction() {
    // A diagonal phalanx d4-e5-f6, and the same shape on each line through e5.
    for (line, forward) in [
        ("d4 e5 f6", "d4f6-h8"),
        ("f4 e5 d6", "f4d6-b8"),
        ("e4 e5 e6", "e4e6-e9"),
        ("d5 e5 f5", "d5f5-i5"),
        ("f6 e5 d4", "f6d4-a1"),
        ("d6 e5 f4", "d6f4-i1"),
        ("e6 e5 e4", "e6e4-e1"),
        ("f5 e5 d5", "f5d5-a5"),
    ] {
        let p = pos(Side::First, line, "n12");
        assert!(
            is_legal(&p, forward),
            "{line}: {forward} in {:?}",
            legal(&p)
        );
    }
}

#[test]
fn a_phalanx_is_blocked_by_a_friendly_piece_ahead() {
    let p = pos(Side::First, "c1 c2 c3 c5", "n12");
    assert!(is_legal(&p, "c1c3-c4"));
    assert!(!is_legal(&p, "c1c3-c5"), "may not land on a friend");
    assert!(!is_legal(&p, "c1c3-c6"), "may not jump a friend");
}

#[test]
fn a_phalanx_may_not_pass_over_an_enemy_piece() {
    // Two whites against a line of three blacks: no capture, and no passing.
    let p = pos(Side::First, "c1 c2", "c4 c5 c6");
    assert!(is_legal(&p, "c1c2-c3"));
    assert!(!is_legal(&p, "c1c2-c4"), "the enemy line is not shorter");
}

// ─── Capture ─────────────────────────────────────────────────────────────────

#[test]
fn capture_needs_the_enemy_line_strictly_shorter() {
    let three = "c1 c2 c3";
    // Enemy line of two: capturable by three.
    let p = pos(Side::First, three, "c5 c6");
    assert!(is_legal(&p, "c1c3xc5"));
    // Enemy line of three: equal, so not.
    let p = pos(Side::First, three, "c5 c6 c7");
    assert!(!is_legal(&p, "c1c3xc5"));
    // Enemy line of four: longer, so not.
    let p = pos(Side::First, three, "c5 c6 c7 c8");
    assert!(!is_legal(&p, "c1c3xc5"));
}

#[test]
fn only_the_enemy_line_continuing_away_counts() {
    // Black on c5 and c6, plus d5 and b5 beside it and c4... c4 would block. A black piece
    // *behind* the target along the line of attack (c4) is impossible without blocking, so check
    // the sideways ones do not count.
    let p = pos(Side::First, "c1 c2 c3", "c5 c6 b5 d5 b6 d6");
    assert!(
        is_legal(&p, "c1c3xc5"),
        "b5/d5 are beside the line, not on it"
    );
}

#[test]
fn capture_counts_the_capturing_group_not_the_whole_phalanx() {
    let p = pos(Side::First, "c1 c2 c3", "c5 c6");
    // The two-piece sub-phalanx c2c3 is no longer than the enemy line.
    assert!(!is_legal(&p, "c2c3xc5"));
}

#[test]
fn capture_must_be_within_reach() {
    // Three pieces, enemy four squares ahead: out of reach.
    let p = pos(Side::First, "c1 c2 c3", "c7");
    assert!(!is_legal(&p, "c1c3xc7"));
    assert!(is_legal(&p, "c1c3-c6"));
    // Three squares ahead is in reach.
    let p = pos(Side::First, "c1 c2 c3", "c6");
    assert!(is_legal(&p, "c1c3xc6"));
}

#[test]
fn a_capture_removes_the_whole_enemy_line_and_stops_there() {
    let p = pos(Side::First, "c1 c2 c3 c4", "c6 c7 c8 a12");
    let after = p.apply(mv("c1c4xc6"));
    // White: the four pieces moved two squares, lead on c6.
    for s in sqs("c3 c4 c5 c6") {
        assert_eq!(after.board[s], WHITE, "{}", square_label(s));
    }
    for s in sqs("c1 c2 c7 c8") {
        assert_eq!(after.board[s], EMPTY, "{}", square_label(s));
    }
    assert_eq!(after.count(Side::Second), 1, "only a12 survives");
    assert_eq!(after.last_capturer, Some(Side::First));
    assert_eq!(after.quiet_moves, 0);
}

#[test]
fn a_capture_does_not_take_pieces_beyond_a_gap() {
    let p = pos(Side::First, "c1 c2 c3", "c5 c6 c8");
    let after = p.apply(mv("c1c3xc5"));
    assert_eq!(after.board[sq("c8")], BLACK);
    assert_eq!(after.count(Side::Second), 1);
}

#[test]
fn quiet_moves_and_captures_drive_the_clock() {
    let p = pos(Side::First, "c1 c2 c3", "c5 c6 n11");
    let p = Position {
        quiet_moves: 40,
        moves: 50,
        ..p
    };
    let quiet = p.apply(mv("c1c3-c4"));
    assert_eq!((quiet.moves, quiet.quiet_moves), (51, 41));
    assert_eq!(quiet.last_capturer, None);
    assert_eq!(quiet.to_move, Side::Second);
    let capture = p.apply(mv("c1c3xc5"));
    assert_eq!((capture.moves, capture.quiet_moves), (51, 0));
}

// ─── Winning ─────────────────────────────────────────────────────────────────

#[test]
fn an_incursion_wins_if_unanswered_at_the_start_of_the_next_turn() {
    // White steps onto black's home rank; black to move is not the winner.
    let p = pos(Side::First, "c11", "a5 n12");
    let after = p.apply(mv("c11-c12"));
    assert_eq!(after.outcome(), None, "black gets a turn to answer");
    // Black does nothing about it; at the start of white's turn, white wins.
    let after = after.apply(mv("n12-n11"));
    assert_eq!(after.outcome(), Some(Outcome::win(Side::First)));
}

#[test]
fn an_incursion_matched_in_time_is_not_a_win() {
    // White has one piece on rank 12, black one on rank 1: level, so nobody wins.
    let p = pos(Side::First, "c12 n5", "a1 n9");
    assert!(!p.wins_at_turn_start());
    assert_eq!(p.outcome(), None);
    let p = pos(Side::Second, "c12 n5", "a1 n9");
    assert_eq!(p.outcome(), None);
}

#[test]
fn only_the_side_to_move_can_win_by_incursion() {
    // Black has more on white's home rank, but it is white's turn: that is not white's win, and
    // it is not tested on white's behalf either.
    let p = pos(Side::First, "f6", "a1 b1");
    assert_eq!(p.outcome(), None);
    let p = pos(Side::Second, "f6", "a1 b1");
    assert_eq!(p.outcome(), Some(Outcome::win(Side::Second)));
}

#[test]
fn a_side_with_no_move_at_all_loses() {
    // White's only piece is boxed into a corner by black pairs it cannot capture.
    let p = pos(Side::First, "a1", "a2 a3 b1 c1 b2 c3");
    assert!(p.legal_moves().is_empty());
    assert_eq!(p.outcome(), Some(Outcome::win(Side::Second)));
}

#[test]
fn the_symmetry_rule_forbids_a_back_rank_entry_that_mirrors_the_board() {
    // Symmetric about the centre line except white's g11, which is about to enter g12 or h12.
    // g11 -> h12 lands on h12 and leaves g11 empty: not symmetric (h12 alone). Arrange instead
    // for a mirror-completing move: white h12 already there, g11 steps to g12.
    let p = pos(Side::First, "h12 g11", "a1 n1");
    let completes_mirror = mv("g11-g12");
    let after = p.apply(completes_mirror);
    assert!(
        is_mirror_symmetric(&after.board),
        "the setup must actually produce a mirror"
    );
    assert!(
        !is_legal(&p, "g11-g12"),
        "forbidden: enters the back rank into a mirror"
    );
    // The same step with an asymmetry elsewhere on the board is fine.
    let p = pos(Side::First, "h12 g11 c5", "a1 n1");
    assert!(is_legal(&p, "g11-g12"));
}

#[test]
fn the_symmetry_rule_only_applies_to_the_opponents_back_rank() {
    // A mirror created away from the back rank is allowed.
    let p = pos(Side::First, "g5 h6", "a1 n1");
    let after = p.apply(mv("h6-h5"));
    assert!(is_mirror_symmetric(&after.board));
    assert!(is_legal(&p, "h6-h5"));
}

// ─── The quiet-move clock and the advancement tiebreak ───────────────────────

fn at_clock(p: Position) -> Position {
    Position {
        quiet_moves: MAX_QUIET_MOVES,
        ..p
    }
}

#[test]
fn the_game_ends_after_the_quiet_move_limit() {
    let p = pos(Side::First, "c5", "c8");
    assert_eq!(
        Position {
            quiet_moves: MAX_QUIET_MOVES - 1,
            ..p.clone()
        }
        .outcome(),
        None
    );
    assert!(at_clock(p).outcome().is_some());
}

#[test]
fn advancement_is_compared_from_the_opponents_home_rank_down() {
    // White has a piece on rank 11 (one short of black's home), black none on rank 2.
    let p = pos(Side::First, "c11 a1", "c2 n12 m12");
    // Rank pair 0: white on rank 12 (0) vs black on rank 1 (0) - level.
    // Rank pair 1: white on rank 11 (1) vs black on rank 2 (1) - level.
    // Rank pair 2..: white on rank 10 (0) vs black on rank 3 (0) ... white's a1 is on rank 1,
    // compared at pair 11 against black's rank 12 (2). Black is further up.
    assert_eq!(advancement_winner(&p.board), Some(Side::Second));
    assert_eq!(at_clock(p).outcome(), Some(Outcome::win(Side::Second)));
}

#[test]
fn the_first_differing_rank_decides_even_against_a_larger_count_later() {
    // White: one piece on rank 10. Black: nothing on rank 3, but five on rank 4.
    let p = pos(Side::First, "c10", "a4 b4 c4 d4 e4");
    // Pairs 0 and 1 are level (empty), pair 2 is white rank 10 (1) vs black rank 3 (0).
    assert_eq!(advancement_winner(&p.board), Some(Side::First));
}

#[test]
fn an_exact_mirror_falls_back_to_the_last_capturer_then_to_black() {
    let p = pos(Side::First, "c5", "c8");
    assert_eq!(advancement_winner(&p.board), None, "rank 5 mirrors rank 8");
    assert_eq!(
        at_clock(p.clone()).outcome(),
        Some(Outcome::win(Side::Second)),
        "black by default"
    );
    let captured = Position {
        last_capturer: Some(Side::First),
        ..at_clock(p.clone())
    };
    assert_eq!(captured.outcome(), Some(Outcome::win(Side::First)));
    let captured = Position {
        last_capturer: Some(Side::Second),
        ..at_clock(p)
    };
    assert_eq!(captured.outcome(), Some(Outcome::win(Side::Second)));
}

#[test]
fn an_incursion_is_decided_before_the_clock() {
    // At the limit with white to move and a piece on rank 12. An incursion always leads on
    // advancement too (it is the first rank pair compared), so the two agree; what this pins is
    // that a won incursion is reported rather than, say, falling through to the mirror tiebreak.
    let p = at_clock(pos(Side::First, "c12", "a5 b5 c6"));
    assert_eq!(p.outcome(), Some(Outcome::win(Side::First)));
}

// ─── The wire encoding ───────────────────────────────────────────────────────

#[test]
fn state_encoding_round_trips() {
    let g = Epaminondas;
    let mut p = Position::start();
    for m in ["c2-c3", "c11-c10", "b1b2-b4"] {
        p = p.apply(mv(m));
    }
    p.last_capturer = Some(Side::Second);
    p.quiet_moves = 17;
    let v = g.encode_state(&p);
    assert_eq!(g.decode_state(&v).unwrap(), p);
    assert_eq!(v["board"].as_str().unwrap().len(), N);
    assert_eq!(
        &v["board"].as_str().unwrap()[..2],
        "W.",
        "a1 first, then b1 (which moved)"
    );
}

#[test]
fn the_start_state_encodes_as_documented() {
    let v = Epaminondas.encode_state(&Position::start());
    let board = v["board"].as_str().unwrap();
    assert_eq!(
        board,
        format!("{}{}{}", "W".repeat(28), ".".repeat(112), "B".repeat(28))
    );
    assert_eq!(v["to_move"], 0);
    assert_eq!(v["moves"], 0);
    assert_eq!(v["quiet_moves"], 0);
    assert_eq!(v["last_capturer"], -1);
}

#[test]
fn malformed_states_are_rejected_not_panicked_on() {
    let g = Epaminondas;
    let good = g.encode_state(&Position::start());
    let with = |k: &str, v: serde_json::Value| {
        let mut s = good.clone();
        s[k] = v;
        s
    };
    let without = |k: &str| {
        let mut s = good.clone();
        s.as_object_mut().unwrap().remove(k);
        s
    };
    for bad in [
        serde_json::json!(null),
        serde_json::json!("board"),
        with("board", serde_json::json!(".".repeat(167))),
        with("board", serde_json::json!(".".repeat(169))),
        with("board", serde_json::json!("x".repeat(168))),
        with("board", serde_json::json!(7)),
        with("to_move", serde_json::json!(2)),
        with("to_move", serde_json::json!(-1)),
        with("to_move", serde_json::json!("white")),
        with("moves", serde_json::json!(-3)),
        with("quiet_moves", serde_json::json!(1.5)),
        with("last_capturer", serde_json::json!(2)),
        without("board"),
        without("quiet_moves"),
        without("last_capturer"),
    ] {
        assert!(g.decode_state(&bad).is_err(), "accepted {bad}");
    }
}

#[test]
fn move_encoding_round_trips_and_ignores_san() {
    let g = Epaminondas;
    let p = Position::start();
    for m in p.legal_moves() {
        let v = g.encode_move(&p, &m);
        assert_eq!(g.decode_move(&v).unwrap(), m);
        assert_eq!(v["san"], notation::describe(&p, m));
    }
    let wrong_san = serde_json::json!({"lead": 17, "rear": 17, "dest": 31, "san": "nonsense"});
    assert_eq!(g.decode_move(&wrong_san).unwrap(), Move::new(17, 17, 31));
}

#[test]
fn malformed_moves_are_rejected_not_panicked_on() {
    let g = Epaminondas;
    for bad in [
        serde_json::json!(null),
        serde_json::json!([1, 2, 3]),
        serde_json::json!({"lead": 1, "rear": 1}),
        serde_json::json!({"lead": 1, "rear": 1, "dest": 168}),
        serde_json::json!({"lead": -1, "rear": 1, "dest": 2}),
        serde_json::json!({"lead": "a1", "rear": 1, "dest": 2}),
    ] {
        assert!(g.decode_move(&bad).is_err(), "accepted {bad}");
    }
}

// ─── Through the type-erased interface the tournament uses ───────────────────

#[test]
fn dyn_game_plays_legal_moves_and_refuses_everything_else() {
    let g: &dyn DynGame = &Epaminondas;
    let start = g.initial(&mut rand::rng());
    // d1d2-d4: rear d1 (3), lead d2 (17), lead lands on d4 (45).
    let played = g
        .play(
            &start,
            &serde_json::json!({"lead": 17, "rear": 3, "dest": 45}),
            &mut rand::rng(),
        )
        .unwrap();
    assert_eq!(played.notation, "d1d2-d4");
    assert_eq!(
        played.mv["san"], "d1d2-d4",
        "the canonical move carries the notation"
    );
    assert_eq!(g.side_to_move(&played.state).unwrap(), Side::Second);

    // Well-formed, but not a legal move here: the rear is not in line.
    let err = g
        .play(
            &start,
            &serde_json::json!({"lead": 17, "rear": 0, "dest": 31}),
            &mut rand::rng(),
        )
        .unwrap_err();
    assert!(matches!(err, MoveError::Illegal(_)), "{err}");
    // A black piece, on white's turn.
    let black = g.play(
        &start,
        &serde_json::json!({"lead": 140, "rear": 140, "dest": 126}),
        &mut rand::rng(),
    );
    assert!(matches!(black, Err(MoveError::Illegal(_))));
    // Not a move at all.
    let err = g
        .play(&start, &serde_json::json!({"from": "d3"}), &mut rand::rng())
        .unwrap_err();
    assert!(matches!(err, MoveError::Malformed(_)), "{err}");
    // A move in a finished game.
    let over = Epaminondas.encode_state(&pos(Side::First, "c12", "a5"));
    let err = g
        .play(
            &over,
            &serde_json::json!({"lead": 0, "rear": 0, "dest": 1}),
            &mut rand::rng(),
        )
        .unwrap_err();
    assert_eq!(err, MoveError::GameOver);
}

#[test]
fn described_moves_pair_each_move_with_its_notation() {
    let g: &dyn DynGame = &Epaminondas;
    let described = g.described_moves(&g.initial(&mut rand::rng())).unwrap();
    assert_eq!(described.len(), 114);
    assert!(
        described.windows(2).all(|w| w[0].1 <= w[1].1),
        "sorted by notation"
    );
    for (mv, san) in &described {
        assert_eq!(
            g.play(&g.initial(&mut rand::rng()), mv, &mut rand::rng())
                .unwrap()
                .notation,
            *san
        );
    }
}

#[test]
fn a_finished_game_has_no_legal_moves() {
    let over = pos(Side::First, "c12 f6", "a5");
    assert!(
        !over.legal_moves().is_empty(),
        "the raw generator still sees moves"
    );
    assert!(
        Game::legal_moves(&Epaminondas, &over).is_empty(),
        "but the game offers none"
    );
}

#[test]
fn the_registry_finds_epaminondas() {
    assert_eq!(crate::by_id("epaminondas/1").unwrap().id(), "epaminondas/1");
    assert!(crate::by_id("epaminondas/2").is_none());
    assert!(crate::all_ids().contains(&ID));
}

#[test]
fn render_shows_rank_twelve_at_the_top() {
    let text = Game::render(&Epaminondas, &Position::start());
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[1].starts_with("12 ") && lines[1].contains('B'),
        "{text}"
    );
    assert!(
        lines[12].starts_with(" 1 ") && lines[12].contains('W'),
        "{text}"
    );
    assert!(text.contains("White to move"));
}

// ─── Counting ────────────────────────────────────────────────────────────────

#[test]
fn the_start_position_has_the_expected_move_count() {
    // Pinned independently of the generator; the conformance suite also checks it against pgx.
    let n = Position::start().legal_moves().len();
    assert_eq!(
        n,
        super::tests::expected_start_moves(),
        "start position move count"
    );
}

/// Worked out by hand from the rules rather than taken from the generator.
pub(super) fn expected_start_moves() -> usize {
    // Singles from row 2 into row 3: 14 pieces x 3 targets, minus 2 at the edges.
    let singles = 14 * 3 - 2;
    // Vertical pairs (c1, c2) led by c2: 1 or 2 squares up. 14 columns x 2.
    let vertical = 14 * 2;
    // Diagonal pairs, rear on rank 1 and lead on rank 2, 13 per direction. Up-right, the pair
    // with its rear in column x lands on column x+2 (one step) or x+3 (two), which must be on the
    // board: x <= 11 and x <= 10 respectively. Up-left mirrors it.
    let diagonal = 2 * (12 + 11);
    singles + vertical + diagonal
}

#[test]
fn captures_are_listed_and_counted() {
    let p = pos(Side::First, "c1 c2 c3 h1", "c5 c6 m12");
    let caps = p.legal_captures();
    assert_eq!(caps, vec![mv("c1c3xc5")]);
    assert_eq!(p.captured_count(mv("c1c3xc5")), 2);
    assert_eq!(p.captured_count(mv("c1c3-c4")), 0);
    let all = p.legal_moves();
    let expected: Vec<Move> = all
        .iter()
        .copied()
        .filter(|&m| p.captured_count(m) > 0)
        .collect();
    assert_eq!(caps, expected, "exactly the legal moves that capture");
}

#[test]
fn crossed_counts_pieces_on_the_opponents_home_rank() {
    let p = pos(Side::First, "c12 d12 a5", "b1 n6");
    assert_eq!(p.crossed(Side::First), 2);
    assert_eq!(p.crossed(Side::Second), 1);
}
