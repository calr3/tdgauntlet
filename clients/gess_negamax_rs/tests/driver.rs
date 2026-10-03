//! The move-choosing driver (`best_move`), which the app only runs on a clock, agrees with the
//! fixed-depth search the identity test checks: at a fixed depth, the chosen move's reported score
//! is the position's value at that depth, and the move is legal.

use tdg_games::gess::Position;
use tdg_gess_negamax::{Board, Limits, Searcher, Weights};

#[test]
fn best_move_at_a_fixed_depth_scores_the_positions_value() {
    let mut s = Searcher::new(Weights::default());
    let mut pos = Position::start();
    let mut checked = 0;
    for ply in 0..60 {
        if pos.outcome().is_some() {
            break;
        }
        let b = Board::from_position(&pos);
        let d = 1 + (ply % 3);
        let r = s.best_move(&b, Limits::depth(d)).expect("a legal move");
        assert_eq!(r.score, s.search_to_depth(&b, r.depth, true), "ply {ply}");
        assert!(r.depth >= 1 && r.depth <= d);
        let mv = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.from as usize == r.from && m.to as usize == r.to);
        let mv = mv.expect("the chosen move is legal");
        checked += 1;
        pos = pos.apply(mv);
    }
    assert!(checked >= 10, "{checked}");
}

#[test]
fn a_node_budget_stops_the_search() {
    let mut s = Searcher::new(Weights::default());
    let b = Board::from_position(&Position::start());
    let r = s
        .best_move(
            &b,
            Limits {
                depth: 64,
                deadline: None,
                nodes: Some(20_000),
            },
        )
        .unwrap();
    assert!(r.nodes < 200_000, "{}", r.nodes);
    assert!(r.depth >= 1);
}
