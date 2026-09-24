use std::time::{Duration, Instant};
use tdg_games::epaminondas::{parse_move, parse_square, Position};
use tdg_games::Side;
use tdg_mcts::search::{search, Limits, MctsConfig};

fn sq(s: &str) -> Vec<usize> {
    s.split_whitespace()
        .map(|x| parse_square(x).unwrap())
        .collect()
}

fn pos(to_move: Side, white: &str, black: &str) -> Position {
    Position::from_pieces(to_move, &sq(white), &sq(black))
}

fn sims(n: u64) -> Limits {
    Limits {
        deadline: None,
        sims: Some(n),
    }
}

#[test]
fn it_finds_immediate_winning_incursion() {
    // White steps onto rank 12; black's lone piece is far away and cannot capture or match it.
    let p = pos(Side::First, "c11 a3", "n7");
    let r = search(&p, sims(2000), MctsConfig::default()).unwrap();
    let rank_12 = sq("b12 c12 d12");
    assert!(rank_12.contains(&(r.best.dest as usize)), "pv {:?}", r.pv);
    assert!(
        r.win_rate > 0.8,
        "win rate should be near 1.0: {}",
        r.win_rate
    );
}

#[test]
fn it_answers_an_incursion_it_can_capture() {
    // Black has stepped onto rank 1 at c1. White's c2 c3 vertical pair can capture c1.
    let p = pos(Side::First, "c2 c3 k6", "c1 n11");
    let r = search(&p, sims(3000), MctsConfig::default()).unwrap();
    assert_eq!(r.best, parse_move("c3c2xc1").unwrap(), "pv {:?}", r.pv);
}

#[test]
fn it_respects_the_deadline() {
    // At least one simulation always runs to the end, and the clock is read between them, so
    // the overshoot is one simulation plus scheduling noise. How long a simulation takes is a
    // property of the machine - an unoptimised build on a loaded CI runner is far slower than
    // this one - so measure it rather than assuming a constant.
    let started = Instant::now();
    let sample = search(
        &Position::start(),
        Limits {
            deadline: None,
            sims: Some(64),
        },
        MctsConfig::default(),
    )
    .unwrap();
    let per_sim = started.elapsed() / sample.sims as u32;
    let slack = 4 * per_sim + Duration::from_millis(100);

    for ms in [20u64, 100] {
        let started = Instant::now();
        let limits = Limits {
            deadline: Some(started + Duration::from_millis(ms)),
            sims: None,
        };
        let r = search(&Position::start(), limits, MctsConfig::default()).unwrap();
        let took = started.elapsed();
        assert!(
            took < Duration::from_millis(ms) + slack,
            "{ms}ms budget took {took:?}, allowing {slack:?} over"
        );
        assert!(r.sims >= 1, "runs at least one simulation");
    }
}

#[test]
fn it_respects_a_simulation_budget() {
    let limits = Limits {
        deadline: None,
        sims: Some(500),
    };
    let r = search(&Position::start(), limits, MctsConfig::default()).unwrap();
    assert_eq!(r.sims, 500);
}
