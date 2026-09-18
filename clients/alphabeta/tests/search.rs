use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search, Limits, WIN};
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
    }
}

#[test]
fn it_walks_into_an_unanswerable_incursion() {
    // White steps onto rank 12; black's lone piece is far away and cannot capture or match it.
    let p = pos(Side::First, "c11 a3", "n7");
    let r = search(&p, &Table::new(4), &Weights::default(), depth(3)).unwrap();
    // b12, c12 and d12 all win; any of them will do.
    let rank_12 = sq("b12 c12 d12");
    assert!(rank_12.contains(&(r.best.dest as usize)), "pv {:?}", r.pv);
    assert!(
        r.score >= WIN - 10,
        "should see the forced win: {}",
        r.score
    );
}

#[test]
fn it_answers_an_incursion_it_can_capture() {
    // Black has stepped onto rank 1 at c1. White's c3-c2 pair... cannot capture downwards onto
    // c1 with a single piece, but the vertical pair c2 c3 led by c2 can take a lone c1.
    let p = pos(Side::First, "c2 c3 k6", "c1 n11");
    let r = search(&p, &Table::new(4), &Weights::default(), depth(2)).unwrap();
    assert_eq!(r.best, parse_move("c3c2xc1").unwrap(), "pv {:?}", r.pv);
}

#[test]
fn it_takes_free_material() {
    let p = pos(Side::First, "c1 c2 c3 h1", "c5 c6 m12 n12");
    let r = search(&p, &Table::new(4), &Weights::default(), depth(2)).unwrap();
    assert_eq!(r.best, parse_move("c1c3xc5").unwrap(), "pv {:?}", r.pv);
}

#[test]
fn it_respects_the_deadline() {
    let tt = Table::new(16);
    for ms in [20u64, 150] {
        let started = Instant::now();
        let limits = Limits {
            deadline: Some(started + Duration::from_millis(ms)),
            nodes: None,
            depth: 64,
        };
        let r = search(&Position::start(), &tt, &Weights::default(), limits).unwrap();
        let took = started.elapsed();
        assert!(
            took < Duration::from_millis(ms + 100),
            "{ms}ms budget took {took:?}"
        );
        assert!(r.depth >= 1, "always completes at least one iteration");
        assert!(Position::start().legal_moves().contains(&r.best));
    }
}

#[test]
fn it_respects_a_node_budget() {
    let limits = Limits {
        deadline: None,
        nodes: Some(20_000),
        depth: 64,
    };
    let r = search(
        &Position::start(),
        &Table::new(4),
        &Weights::default(),
        limits,
    )
    .unwrap();
    // Checked every 1024 nodes, so it may overshoot by at most about that much.
    assert!(r.nodes < 20_000 + 2048, "{} nodes", r.nodes);
}

#[test]
fn deeper_iterations_are_reported_and_the_pv_is_legal() {
    let p = Position::start();
    let r = search(&p, &Table::new(16), &Weights::default(), depth(3)).unwrap();
    assert_eq!(r.depth, 3);
    let mut q = p.clone();
    for m in &r.pv {
        assert!(q.legal_moves().contains(m));
        q = q.apply(*m);
    }
}

#[test]
fn a_forced_move_is_returned_without_searching() {
    // One white piece in a corner with only one empty neighbour.
    let p = pos(Side::First, "a1", "a2 a3 b2 c3 n12");
    assert_eq!(p.legal_moves().len(), 1);
    let r = search(&p, &Table::new(1), &Weights::default(), depth(10)).unwrap();
    assert_eq!(r.nodes, 0);
}

#[test]
fn concurrent_searches_sharing_a_table_all_return_legal_moves() {
    // Many unrelated positions at once, one shared table: the situation the protocol creates.
    let tt = Table::new(16);
    let positions: Vec<Position> = {
        let mut out = vec![Position::start()];
        let mut p = Position::start();
        for i in 0..24 {
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
                    let r = search(p, tt, &Weights::default(), depth(3)).unwrap();
                    assert!(p.legal_moves().contains(&r.best));
                }
            });
        }
    });
}
