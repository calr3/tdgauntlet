//! The heuristic-free search: what it proves, what it refuses to guess at, and that its results
//! do not depend on any of its refinements.

use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search, Limits};
use tdg_alphabeta::solve::{solve, solve_with, Features, Verdict};
use tdg_alphabeta::tt::Table;
use tdg_games::epaminondas::{parse_move, parse_square, Position};
use tdg_games::Side;

fn sq(s: &str) -> Vec<usize> {
    s.split_whitespace()
        .map(|x| parse_square(x).unwrap())
        .collect()
}

fn pos(to_move: Side, white: &str, black: &str) -> Position {
    Position::from_pieces(to_move, &sq(white), &sq(black))
}

fn depth(d: u32) -> Limits {
    Limits {
        deadline: None,
        nodes: None,
        depth: d,
        seed: None,
    }
}

#[test]
fn it_proves_the_win_it_can_see() {
    // White steps onto rank 12; black's lone piece can neither capture it nor match it, so white
    // wins at the start of its next turn: two plies away.
    let p = pos(Side::First, "c11 a3", "n7");
    let r = solve(&p, &Table::new(4), depth(4)).unwrap();
    assert_eq!(r.verdict, Verdict::Win { plies: 2 }, "pv {:?}", r.pv);
    assert!(sq("b12 c12 d12").contains(&(r.best.dest as usize)));
    assert_eq!(r.depth, 2, "it stops as soon as the rules have decided it");
}

#[test]
fn it_avoids_the_only_loss_without_knowing_what_a_capture_is() {
    // Black has stepped onto rank 1 at c1 and wins at the start of its turn unless white removes
    // it. Only the c2 c3 phalanx can, and nothing here tells the search that a capture is special.
    let p = pos(Side::First, "c2 c3 k6", "c1 n11");
    let r = solve(&p, &Table::new(4), depth(2)).unwrap();
    assert_eq!(r.best, parse_move("c3c2xc1").unwrap(), "pv {:?}", r.pv);
    assert_eq!(r.verdict, Verdict::Unknown, "surviving is not yet winning");
}

#[test]
fn it_proves_a_loss_it_cannot_escape() {
    // The same incursion, with white's pieces too far away to answer it at all.
    let p = pos(Side::First, "k6 k7", "c1 n11");
    let r = solve(&p, &Table::new(4), depth(3)).unwrap();
    assert_eq!(r.verdict, Verdict::Loss { plies: 1 });
    assert!(p.legal_moves().contains(&r.best), "still plays on");
}

#[test]
fn an_undecided_position_gets_no_opinion() {
    // Nothing is settled within three plies of the start, and with no evaluation there is nothing
    // else to say about it.
    let p = Position::start();
    let r = solve(&p, &Table::new(16), depth(3)).unwrap();
    assert_eq!(r.verdict, Verdict::Unknown);
    assert_eq!(r.depth, 3);
    assert!(p.legal_moves().contains(&r.best));
    let mut q = p.clone();
    for m in &r.pv {
        assert!(q.legal_moves().contains(m), "pv {:?}", r.pv);
        q = q.apply(*m);
    }
}

#[test]
fn the_refinements_only_change_the_node_count() {
    // Ordering, PVS and keeping proofs are speed; at a fixed depth the verdict must be the same.
    let positions = [
        pos(Side::First, "c11 a3", "n7"),
        pos(Side::First, "c2 c3 k6", "c1 n11"),
        pos(Side::First, "k6 k7", "c1 n11"),
        Position::start(),
    ];
    for p in &positions {
        let all = solve_with(p, &Table::new(8), depth(3), Features::ALL).unwrap();
        let none = solve_with(p, &Table::new(8), depth(3), Features::NONE).unwrap();
        assert_eq!(all.verdict, none.verdict, "verdicts differ on {p:?}");
    }
}

#[test]
fn deepening_never_unsays_a_proof() {
    let p = pos(Side::First, "c11 a3", "n7");
    let tt = Table::new(8);
    for d in 2..=6 {
        let r = solve(&p, &tt, depth(d)).unwrap();
        assert_eq!(r.verdict, Verdict::Win { plies: 2 }, "at depth {d}");
    }
}

#[test]
fn it_respects_the_deadline_and_a_node_budget() {
    let tt = Table::new(16);
    let started = Instant::now();
    let limits = Limits {
        deadline: Some(started + Duration::from_millis(100)),
        nodes: None,
        depth: 64,
        seed: None,
    };
    let r = solve(&Position::start(), &tt, limits).unwrap();
    assert!(started.elapsed() < Duration::from_millis(400));
    assert!(r.depth >= 1, "always completes at least one iteration");

    let r = solve(&Position::start(), &Table::new(4), {
        Limits {
            deadline: None,
            nodes: Some(20_000),
            depth: 64,
            seed: None,
        }
    })
    .unwrap();
    // Checked every 1024 nodes, so it may overshoot by about that much.
    assert!(r.nodes < 20_000 + 2048, "{} nodes", r.nodes);
}

#[test]
fn the_two_searches_can_share_one_table() {
    // Their scores mean different things, so the keys are tagged apart; neither may read the
    // other's entries as its own.
    let tt = Table::new(16);
    let win = pos(Side::First, "c11 a3", "n7");
    let w = Weights::recommended();
    for _ in 0..3 {
        let s = search(&win, &tt, &w, depth(3)).unwrap();
        assert!(win.legal_moves().contains(&s.best));
        let r = solve(&win, &tt, depth(4)).unwrap();
        assert_eq!(r.verdict, Verdict::Win { plies: 2 });
        let s = search(&Position::start(), &tt, &w, depth(2)).unwrap();
        assert!(Position::start().legal_moves().contains(&s.best));
        let r = solve(&Position::start(), &tt, depth(2)).unwrap();
        assert_eq!(r.verdict, Verdict::Unknown);
    }
}

#[test]
fn concurrent_solves_sharing_a_table_all_return_legal_moves() {
    let tt = Table::new(16);
    let positions: Vec<Position> = {
        let mut out = vec![Position::start()];
        let mut p = Position::start();
        for i in 0..12 {
            let moves = p.legal_moves();
            p = p.apply(moves[(i * 7) % moves.len()]);
            out.push(p.clone());
        }
        out
    };
    std::thread::scope(|s| {
        for p in &positions {
            let tt = &tt;
            s.spawn(move || {
                for _ in 0..3 {
                    let r = solve(p, tt, depth(2)).unwrap();
                    assert!(p.legal_moves().contains(&r.best));
                }
            });
        }
    });
}
