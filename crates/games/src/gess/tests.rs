//! Unit tests for the rules. The conformance suite (`tests/gess_conformance.rs`) is what holds
//! them to pgx; these say in one position each what the rules are supposed to be.

use super::*;

fn cell(label: &str) -> usize {
    parse_square(label).unwrap_or_else(|| panic!("{label} is not a cell"))
}

fn cells(text: &str) -> Vec<usize> {
    text.split_whitespace().map(cell).collect()
}

/// A position from stones written in notation, black first.
fn pos(to_move: Side, black: &str, white: &str) -> Position {
    Position::from_stones(to_move, &cells(black), &cells(white))
}

/// The eight stones around `centre`, as cell indices: a ring.
fn ring_around(centre: usize) -> Vec<usize> {
    let (r, c) = row_col(centre);
    let mut out = Vec::new();
    for dr in -1..=1 {
        for dc in -1..=1 {
            if (dr, dc) != (0, 0) {
                out.push(cell_at(r + dr, c + dc));
            }
        }
    }
    out
}

#[test]
fn the_opening_is_symmetric_and_has_a_ring_each() {
    let p = Position::start();
    assert_eq!(p.count(Side::First), 43, "black stones");
    assert_eq!(p.count(Side::Second), 43, "white stones");
    assert!(p.has_ring(Side::First) && p.has_ring(Side::Second));
    assert_eq!(p.to_move, Side::First, "black moves first");
    assert!(p.outcome().is_none());
    // Mirrored top to bottom, with the colours swapped.
    for r in 0..SIZE {
        for c in 0..SIZE {
            let (top, bottom) = (p.board[r * SIZE + c], p.board[(SIZE - 1 - r) * SIZE + c]);
            let swapped = match top {
                BLACK => WHITE,
                WHITE => BLACK,
                _ => EMPTY,
            };
            assert_eq!(bottom, swapped, "row {r} col {c}");
        }
    }
    // The border ring is empty, and stays that way.
    for i in 0..N {
        if on_border((i / SIZE) as i32, (i % SIZE) as i32) {
            assert_eq!(p.board[i], EMPTY, "{} holds a stone", square_label(i));
        }
    }
}

#[test]
fn a_piece_needs_a_stone_around_its_centre_and_no_enemy_in_it() {
    let p = pos(Side::First, "f15 f16", "k10");
    assert!(p.is_piece(cell("f15")), "own stone above the centre");
    assert!(
        p.is_piece(cell("g15")),
        "a centre next to a stone, itself empty"
    );
    assert!(!p.is_piece(cell("a1")), "nothing near it");
    // An enemy stone anywhere in the footprint rules the whole piece out.
    let blocked = pos(Side::First, "f15 f16", "g15");
    assert!(!blocked.is_piece(cell("f15")));
    // A lone stone is not a piece: the centre alone does not count.
    let lone = pos(Side::First, "f15", "");
    assert!(!lone.is_piece(cell("f15")));
}

#[test]
fn a_direction_opens_only_where_the_piece_has_a_stone() {
    // One stone directly above the centre: north only, and three cells at most since the centre
    // is empty.
    let p = pos(Side::First, "f16", "");
    let from = cell("f15");
    assert_eq!(
        p.destinations(from),
        cells("f16 f17 f18"),
        "three cells north and nothing else"
    );
    // With a stone in the centre as well, the same direction runs as far as the grid allows.
    let far = pos(Side::First, "f15 f16", "");
    assert!(far.destinations(from).contains(&cell("f20")));
    assert_eq!(far.destinations(from).len(), 5, "f16..f20");
}

#[test]
fn a_piece_stops_where_its_footprint_first_meets_a_stone() {
    // Black at g15 (centre) and g16 runs north without a range limit. A white stone at h18 is
    // outside the footprint the piece would occupy centred on g16, and inside the one it would
    // occupy centred on g17.
    let p = pos(Side::First, "g15 g16", "h18");
    let dests = p.destinations(cell("g15"));
    assert!(dests.contains(&cell("g16")), "one step, nothing in the way");
    assert!(
        dests.contains(&cell("g17")),
        "it may finish on the square where it meets the stone: that is the capture"
    );
    assert!(!dests.contains(&cell("g18")), "but not carry on past it");
    // Landing there does take the stone: it stood inside the footprint.
    let next = p.apply(Move::new(cell("g15"), cell("g17")));
    assert_eq!(next.count(Side::Second), 0);
    assert_eq!(next.quiet_moves, 0);
}

#[test]
fn a_move_carries_the_whole_footprint_and_takes_what_it_lands_on() {
    let p = pos(Side::First, "f15 f16", "f17 g17");
    let before = p.count(Side::Second);
    let next = p.apply(Move::new(cell("f15"), cell("f16")));
    // Each stone shifts by the same step: f16 -> f17, f15 -> f16.
    assert_eq!(next.board[cell("f17")], BLACK);
    assert_eq!(next.board[cell("f16")], BLACK);
    assert_eq!(next.board[cell("f15")], EMPTY);
    // Both white stones stood in the destination footprint, so both are gone.
    assert_eq!(next.count(Side::Second), before - 2);
    assert_eq!(next.quiet_moves, 0, "a capture resets the count");
    assert_eq!(next.to_move, Side::Second);
    assert_eq!(next.moves, 1);
}

#[test]
fn a_stone_pushed_over_the_border_is_lost() {
    // b19 is the top-left of the playing area; moving north puts a stone on the border.
    let p = pos(Side::First, "c19 c18", "");
    let next = p.apply(Move::new(cell("c18"), cell("c19")));
    assert_eq!(next.board[cell("c20")], EMPTY, "the border stays empty");
    assert_eq!(next.board[cell("c19")], BLACK);
    assert_eq!(next.count(Side::First), 1, "the other stone is gone");
    assert_eq!(next.quiet_moves, 0, "losing a stone counts as a capture");
}

#[test]
fn a_quiet_move_advances_the_count_and_a_capture_resets_it() {
    let p = Position {
        quiet_moves: 7,
        ..pos(Side::First, "f15 f16", "")
    };
    assert_eq!(p.apply(Move::new(cell("f15"), cell("f16"))).quiet_moves, 8);
}

#[test]
fn losing_your_last_ring_loses_the_game() {
    let ring = ring_around(cell("f15"));
    let mut black = ring.clone();
    black.extend(cells("m5 m6"));
    let white = ring_around(cell("m15"));
    let p = Position::from_stones(Side::First, &black, &white);
    assert!(p.has_ring(Side::First) && p.has_ring(Side::Second));
    assert!(p.outcome().is_none());

    // White, to move, with no ring: whoever moved last left it that way, so black has won.
    let no_white_ring = Position::from_stones(Side::Second, &black, &cells("m15 m16"));
    assert_eq!(
        no_white_ring.outcome(),
        Some(Outcome::win(Side::First)),
        "a side without a ring has already lost"
    );
    // And a mover who breaks their own last ring loses, even having taken the other one.
    let neither = Position::from_stones(Side::Second, &cells("f15 f16"), &cells("m15 m16"));
    assert_eq!(
        neither.outcome(),
        Some(Outcome::win(Side::Second)),
        "the side that has just moved is asked first"
    );
}

#[test]
fn a_ring_needs_an_empty_centre_inside_the_playing_area() {
    let centre = cell("f15");
    let p = Position::from_stones(Side::First, &ring_around(centre), &[]);
    assert!(p.has_ring(Side::First));
    // Fill the centre and it is no longer a ring.
    let mut filled = p.clone();
    filled.board[centre] = BLACK;
    assert!(!filled.has_ring(Side::First));
    // Take a corner out and it is no longer a ring either.
    let mut broken = p.clone();
    broken.board[cell("e16")] = EMPTY;
    assert!(!broken.has_ring(Side::First));
}

#[test]
fn twenty_quiet_moves_decide_the_game_on_stones() {
    let black = ring_around(cell("f15"));
    let white = ring_around(cell("m15"));
    let level = Position {
        quiet_moves: MAX_QUIET_MOVES,
        ..Position::from_stones(Side::First, &black, &white)
    };
    assert_eq!(level.outcome(), Some(Outcome::Draw), "same stones, a draw");

    let mut ahead = black.clone();
    ahead.push(cell("c3"));
    let p = Position {
        quiet_moves: MAX_QUIET_MOVES,
        ..Position::from_stones(Side::First, &ahead, &white)
    };
    assert_eq!(p.outcome(), Some(Outcome::win(Side::First)));
    // One move short of the limit it is still anybody's game.
    let p = Position {
        quiet_moves: MAX_QUIET_MOVES - 1,
        ..p
    };
    assert_eq!(p.outcome(), None);
}

#[test]
fn a_side_with_a_ring_always_has_a_move() {
    // The ring centre is a piece with all eight directions open, so there is always something to
    // play: no stalemate rule is needed, and `legal_moves` is empty only when the game is over.
    let p = Position::from_stones(
        Side::First,
        &ring_around(cell("f15")),
        &ring_around(cell("m15")),
    );
    assert!(!p.legal_moves().is_empty());
    assert_eq!(
        p.destinations(cell("f15")).len(),
        8 * 3,
        "three cells each way"
    );
}

#[test]
fn moves_are_distinct_and_every_one_of_them_applies() {
    let p = Position::start();
    let moves = p.legal_moves();
    let unique: std::collections::HashSet<_> = moves.iter().collect();
    assert_eq!(unique.len(), moves.len(), "duplicate moves");
    assert!(!moves.is_empty());
    for m in &moves {
        let next = p.apply(*m);
        assert_eq!(next.to_move, Side::Second);
        assert!(next.count(Side::First) <= 43);
    }
}

#[test]
fn notation_round_trips() {
    assert_eq!(square_label(0), "a20");
    assert_eq!(square_label(N - 1), "t1");
    assert_eq!(cell("b19"), SIZE + 1, "the top left of the playing area");
    for i in 0..N {
        assert_eq!(parse_square(&square_label(i)), Some(i), "cell {i}");
    }
    let p = pos(Side::First, "f15 f16", "f17");
    let m = Move::new(cell("f15"), cell("f16"));
    assert_eq!(parse_move("f15xf16"), Some(m));
    assert_eq!(parse_move("f15-f16"), Some(m));
    assert_eq!(notation::describe(&p, m), "f15xf16", "it takes a stone");
    let quiet = pos(Side::First, "f15 f16", "");
    assert_eq!(notation::describe(&quiet, m), "f15-f16");
    assert_eq!(parse_square("u1"), None);
    assert_eq!(parse_square("a21"), None);
}

#[test]
fn the_wire_state_round_trips_and_rejects_nonsense() {
    let g = Gess;
    let p = Position::start();
    let wire = g.encode_state(&p);
    assert_eq!(g.decode_state(&wire).unwrap(), p);
    assert_eq!(wire["board"].as_str().unwrap().len(), N);
    for m in g.legal_moves(&p).iter().take(50) {
        assert_eq!(g.decode_move(&g.encode_move(&p, m)).unwrap(), *m);
    }
    let bad = |board: String| {
        let mut v = wire.clone();
        v["board"] = serde_json::json!(board);
        g.decode_state(&v)
    };
    let mut border = ".".repeat(N);
    border.replace_range(0..1, "B");
    assert!(bad(border).is_err(), "a stone on the border ring");
    assert!(bad(".".repeat(N - 1)).is_err(), "too few cells");
    assert!(bad("x".repeat(N)).is_err(), "unknown cell character");
}
