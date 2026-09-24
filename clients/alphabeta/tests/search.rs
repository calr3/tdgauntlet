use std::time::{Duration, Instant};
use tdg_alphabeta::eval::Weights;
use tdg_alphabeta::search::{search, search_with, Features, Limits, WIN};
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

fn seeded(d: u32, seed: u64) -> Limits {
    Limits {
        seed: Some(seed),
        ..depth(d)
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
    // The clock is only read every 1024 nodes, so the overshoot to allow is however long this
    // machine takes to search 1024 nodes - a property of the machine, not a constant. An
    // unoptimised build on a loaded CI runner needs well over 100ms for that, which a fixed
    // slack calls a failure. Measure it instead.
    let started = Instant::now();
    let cal = search(
        &Position::start(),
        &tt,
        &Weights::default(),
        Limits {
            deadline: None,
            nodes: Some(4096),
            depth: 64,
            seed: None,
        },
    )
    .unwrap();
    let per_1024 = started.elapsed() / (cal.nodes / 1024).max(1) as u32;
    let slack = 2 * per_1024 + Duration::from_millis(100);

    for ms in [20u64, 150] {
        let started = Instant::now();
        let limits = Limits {
            deadline: Some(started + Duration::from_millis(ms)),
            nodes: None,
            depth: 64,
            seed: None,
        };
        let r = search(&Position::start(), &tt, &Weights::default(), limits).unwrap();
        let took = started.elapsed();
        assert!(
            took < Duration::from_millis(ms) + slack,
            "{ms}ms budget took {took:?}, allowing {slack:?} over"
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
        seed: None,
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

#[test]
fn equally_good_moves_are_chosen_between_by_the_seed() {
    // As above: b12, c12 and d12 all win on the spot, and nothing else does.
    let p = pos(Side::First, "c11 a3", "n7");
    let winners = sq("b12 c12 d12");
    let mut chosen = std::collections::BTreeSet::new();
    for seed in 0..32 {
        let r = search(&p, &Table::new(4), &Weights::default(), seeded(3, seed)).unwrap();
        assert_eq!(r.ties, 3, "seed {seed}");
        assert!(winners.contains(&(r.best.dest as usize)), "seed {seed}");
        chosen.insert(r.best);
    }
    assert_eq!(
        chosen.len(),
        3,
        "32 seeds should reach every winner: {chosen:?}"
    );
    // Without a seed nothing is drawn: the first winner in search order, as before.
    let r = search(&p, &Table::new(4), &Weights::default(), depth(3)).unwrap();
    assert_eq!(r.ties, 1);
}

#[test]
fn a_seed_always_picks_the_same_move_whatever_the_search_order() {
    // The start is symmetric left to right, so every move has a mirror image that scores the
    // same, and the best is always tied.
    let p = Position::start();
    let w = Weights::default();
    let mut chosen = std::collections::BTreeSet::new();
    for seed in 0..8 {
        let all = search_with(&p, &Table::new(16), &w, seeded(3, seed), Features::ALL).unwrap();
        let none = search_with(&p, &Table::new(16), &w, seeded(3, seed), Features::NONE).unwrap();
        assert!(all.ties >= 2, "the mirror image ties: {}", all.ties);
        assert_eq!(
            (all.best, all.score, all.ties),
            (none.best, none.score, none.ties)
        );
        let again = search_with(&p, &Table::new(16), &w, seeded(3, seed), Features::ALL).unwrap();
        assert_eq!(again.best, all.best);
        chosen.insert(all.best);
    }
    assert!(chosen.len() >= 2, "every seed chose {chosen:?}");
}

#[test]
fn a_seed_does_not_change_a_clear_best_move_or_its_score() {
    let p = pos(Side::First, "c1 c2 c3 h1", "c5 c6 m12 n12");
    let plain = search(&p, &Table::new(4), &Weights::default(), depth(2)).unwrap();
    for seed in 0..4 {
        let r = search(&p, &Table::new(4), &Weights::default(), seeded(2, seed)).unwrap();
        assert_eq!((r.best, r.score, r.ties), (plain.best, plain.score, 1));
    }
}
