//! Unit tests for the rules. The conformance suite (`tests/one_more_tri_conformance.rs`) is what
//! holds them to pgx; these say in one position each what the rules are supposed to be.

use super::*;

/// A position from moves written in notation, played in turn from the start.
fn play(moves: &[&str]) -> Position {
    moves.iter().fold(Position::start(), |p, m| {
        let mv = parse_move(m).unwrap_or_else(|| panic!("{m} is not a move"));
        assert!(p.legal_moves().contains(&mv), "{m} is illegal in {p:?}");
        p.apply(mv)
    })
}

/// A finished board: every triangle but `empty` holds a tile, and `around` gives the tiles on
/// `empty`'s neighbours, in [`ADJ`] order and signed. The remaining tiles fill the rest.
fn finished(empty: usize, around: &[i8]) -> Position {
    assert_eq!(around.len(), ADJ[empty].len());
    let mut board = [0i8; N];
    for (&n, &v) in ADJ[empty].iter().zip(around) {
        board[n] = v;
    }
    let mut rest = (1..=TILES as i8)
        .flat_map(|t| [t, -t])
        .filter(|v| !around.contains(v));
    for (t, v) in board.iter_mut().enumerate() {
        if t != empty && *v == 0 {
            *v = rest.next().unwrap();
        }
    }
    assert!(rest.next().is_none());
    Position {
        board,
        to_move: Side::First,
    }
}

#[test]
fn adjacency_is_pgxs_symmetric_but_for_4_and_10() {
    let mut one_way = Vec::new();
    for (t, ns) in ADJ.iter().enumerate() {
        for &n in *ns {
            assert_ne!(n, t);
            if !ADJ[n].contains(&t) {
                one_way.push((t, n));
            }
        }
    }
    assert_eq!(
        one_way,
        [(4, 10)],
        "pgx lists 10 as 4's neighbour, and not the reverse"
    );
    let count = |k| (0..N).filter(|&t| ADJ[t].len() == k).collect::<Vec<_>>();
    assert_eq!(count(1), [11, 20]);
    assert_eq!(count(3), [2, 6, 7, 8, 13, 14, 18]);
}

#[test]
fn the_start_is_empty_with_every_tile_on_every_triangle() {
    let p = Position::start();
    assert_eq!(p.to_move, Side::First);
    assert_eq!(p.legal_moves().len(), N * TILES);
    assert_eq!(p.hand(Side::Second), (1..=10).collect::<Vec<_>>());
    assert!(p.outcome().is_none());
}

#[test]
fn a_tile_is_used_once_and_a_triangle_filled_once() {
    let p = play(&["7 on 13"]);
    assert_eq!(p.board[13], 7);
    assert_eq!(p.to_move, Side::Second);
    // White may still play its own 7, but nothing goes on 13.
    let moves = p.legal_moves();
    assert_eq!(moves.len(), (N - 1) * TILES);
    assert!(moves.contains(&Move::new(0, 7)));
    assert!(moves.iter().all(|m| m.tri != 13));
    let p = play(&["7 on 13", "7 on 0"]);
    assert_eq!(p.board[0], -7);
    assert!(!p.in_hand(Side::First, 7) && !p.in_hand(Side::Second, 7));
    assert!(p.legal_moves().iter().all(|m| m.tile != 7));
}

#[test]
fn the_empty_triangles_neighbours_decide() {
    // Triangle 7 touches 6, 8 and 14.
    let black = finished(7, &[10, -3, 2]);
    assert_eq!(black.score(), 9);
    assert_eq!(black.outcome(), Some(Outcome::Win { side: Side::First }));
    let white = finished(7, &[1, -5, 2]);
    assert_eq!(white.score(), -2);
    assert_eq!(white.outcome(), Some(Outcome::Win { side: Side::Second }));
    let draw = finished(7, &[4, -9, 5]);
    assert_eq!(draw.outcome(), Some(Outcome::Draw));
    // 4 counts the tile on 10, but 10 does not count the tile on 4 (see ADJ).
    let four = finished(4, &[-1, 3]);
    assert_eq!(four.board[10], 3);
    assert_eq!(four.score(), 2);
    let ten = finished(10, &[-1, -2]);
    assert!(
        ten.board[4] > 0,
        "a black tile on 4 that does not count: {:?}",
        ten.board
    );
    assert_eq!(ten.score(), -3);
    // Triangle 11 has a single neighbour, 10.
    assert_eq!(
        finished(11, &[-6]).outcome(),
        Some(Outcome::Win { side: Side::Second })
    );
}

#[test]
fn nothing_is_decided_until_both_hands_are_empty() {
    let mut p = Position::start();
    for _ in 0..2 * TILES - 1 {
        assert!(p.outcome().is_none());
        p = p.apply(p.legal_moves()[0]);
    }
    assert!(p.outcome().is_none());
    p = p.apply(p.legal_moves()[0]);
    assert!(p.outcome().is_some());
    assert_eq!(p.placed(), 2 * TILES);
    assert!(OneMoreTri.legal_moves(&p).is_empty());
}

#[test]
fn states_round_trip_and_bad_ones_are_rejected() {
    let g = OneMoreTri;
    let p = play(&["7 on 13", "10 on 0", "1 on 20"]);
    assert_eq!(g.decode_state(&g.encode_state(&p)).unwrap(), p);

    let bad = |board: Vec<i64>, to_move: i64| {
        g.decode_state(&json!({"board": board, "to_move": to_move}))
            .unwrap_err()
            .0
    };
    let mut board = vec![0i64; N];
    board[3] = 11;
    assert!(bad(board.clone(), 1).contains("not a tile value"));
    board[3] = 4;
    board[5] = -4;
    board[6] = 4;
    assert!(bad(board.clone(), 1).contains("twice"));
    board[6] = 0;
    assert!(
        bad(board.clone(), 1).contains("to_move"),
        "one each is black to move"
    );
    board[5] = 0;
    board[6] = -2;
    board[7] = -3;
    assert!(bad(board, 0).contains("black moves first"));
    assert!(bad(vec![0; N - 1], 0).contains("expected 21"));
}

#[test]
fn moves_round_trip_in_notation_and_json() {
    let g = OneMoreTri;
    let p = Position::start();
    for mv in p.legal_moves() {
        assert_eq!(parse_move(&describe(mv)), Some(mv));
        assert_eq!(g.decode_move(&g.encode_move(&p, &mv)).unwrap(), mv);
    }
    assert_eq!(describe(Move::new(13, 7)), "7 on 13");
    assert_eq!(parse_move("11 on 3"), None);
    assert_eq!(parse_move("3 on 21"), None);
    assert!(g.decode_move(&json!({"tri": 21, "tile": 3})).is_err());
    assert!(g.decode_move(&json!({"tri": 2, "tile": 0})).is_err());
}

#[test]
fn the_diagram_shows_tiles_and_the_numbers_of_empty_triangles() {
    let text = OneMoreTri.render(&play(&["7 on 13", "10 on 0"]));
    assert!(text.contains("B7") && text.contains("WT"), "{text}");
    assert!(text.contains("20") && !text.contains(" 0\\"), "{text}");
    assert!(text.contains("Black to move"), "{text}");
}

#[test]
fn the_rules_documents_examples_are_what_the_engine_reads_and_writes() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../games/one-more-tri.md");
    let doc = std::fs::read_to_string(path).unwrap();
    let blocks: Vec<Value> = doc
        .split("```json")
        .skip(1)
        .map(|b| serde_json::from_str(b.split("```").next().unwrap()).unwrap())
        .collect();
    let g = OneMoreTri;
    let p = play(&["7 on 13", "10 on 0", "1 on 20"]);
    assert_eq!(g.encode_state(&p), blocks[0]);
    assert_eq!(g.encode_move(&p, &Move::new(13, 7)), blocks[1]);
    // And the table of neighbours is ADJ.
    for (t, ns) in ADJ.iter().enumerate() {
        let list = ns
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        assert!(doc.contains(&format!("| {t} | {list} |")), "triangle {t}");
    }
}
