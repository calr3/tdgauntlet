//! The scheduler and game loop, driven by in-process stub players.

use async_trait::async_trait;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tdg_games::DynGame;
use tdg_proto::{Budget, Identify, MoveRequest, MoveResponse, Outcome, ProtocolVersion, Side};
use tdg_tourney::config::{ErrorPolicy, OpeningSpec};
use tdg_tourney::engine::{play_game, GameSpec, IncidentKind, Seat, Termination};
use tdg_tourney::openings::{self, Opening};
use tdg_tourney::player::{Player, PlayerError};
use tdg_tourney::tournament::{run, Entrant, Plan, TournamentResult};
use tokio::sync::Semaphore;

fn epam() -> Arc<dyn DynGame> {
    tdg_games::by_id("epaminondas/1").unwrap()
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// How a stub behaves on a given request.
#[derive(Clone, Copy)]
enum Behaviour {
    /// A legal move chosen by hashing the request id with a salt: deterministic whatever order
    /// requests arrive in, and different players (salts) play differently.
    Legal(u64),
    /// Prefer a legal move that captures, else the first legal move: a stronger stub.
    Greedy,
    /// A well-formed move that is not legal.
    Illegal,
    /// Fail with a 500.
    ServerError,
    /// Echo the wrong request id.
    WrongId,
    /// Fail on the first `n` requests, then play legally.
    FailFirst(u32),
    /// Take this long before answering legally.
    Slow(u64),
}

struct Stub {
    game: Arc<dyn DynGame>,
    behaviour: Behaviour,
    max_concurrency: u32,
    in_flight: AtomicU32,
    peak: AtomicU32,
    calls: AtomicU32,
    seen: Mutex<Vec<String>>,
}

impl Stub {
    fn new(behaviour: Behaviour) -> Arc<Stub> {
        Self::limited(behaviour, 64)
    }

    fn limited(behaviour: Behaviour, max_concurrency: u32) -> Arc<Stub> {
        Arc::new(Stub {
            game: epam(),
            behaviour,
            max_concurrency,
            in_flight: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            calls: AtomicU32::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn legal(&self, req: &MoveRequest, salt: u64) -> serde_json::Value {
        let moves = self.game.legal_moves(&req.state).unwrap();
        moves[(fnv(&req.request_id) ^ salt) as usize % moves.len()].clone()
    }
}

#[async_trait]
impl Player for Stub {
    async fn identify(&self) -> Result<Identify, PlayerError> {
        Ok(Identify {
            protocol: ProtocolVersion::CURRENT,
            name: "stub".into(),
            version: "0".into(),
            games: vec!["epaminondas/1".into()],
            max_concurrency: self.max_concurrency,
        })
    }

    async fn choose(&self, req: &MoveRequest, _: Duration) -> Result<MoveResponse, PlayerError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen.lock().unwrap().push(req.request_id.clone());
        // Yield so that concurrent requests genuinely overlap.
        tokio::time::sleep(Duration::from_millis(1)).await;
        let reply = |mv| {
            Ok(MoveResponse {
                request_id: req.request_id.clone(),
                mv,
                info: None,
            })
        };
        let result = match self.behaviour {
            Behaviour::Legal(salt) => reply(self.legal(req, salt)),
            Behaviour::Greedy => {
                let moves = self.game.legal_moves(&req.state).unwrap();
                let capture = moves
                    .iter()
                    .find(|m| m["san"].as_str().unwrap().contains('x'));
                reply(capture.unwrap_or(&moves[0]).clone())
            }
            Behaviour::Illegal => reply(serde_json::json!({"lead": 0, "rear": 0, "dest": 167})),
            Behaviour::ServerError => Err(PlayerError::Status {
                status: 500,
                message: "boom".into(),
            }),
            Behaviour::WrongId => Ok(MoveResponse {
                request_id: "something-else".into(),
                mv: self.legal(req, 0),
                info: None,
            }),
            Behaviour::FailFirst(k) if n < k => Err(PlayerError::Timeout),
            Behaviour::FailFirst(_) => reply(self.legal(req, 0)),
            Behaviour::Slow(ms) => {
                tokio::time::sleep(Duration::from_millis(ms)).await;
                reply(self.legal(req, 0))
            }
        };
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        result
    }
}

fn seat(name: &str, stub: Arc<Stub>, on_error: ErrorPolicy) -> Arc<Seat> {
    Arc::new(Seat {
        name: name.into(),
        limit: Arc::new(Semaphore::new(stub.max_concurrency as usize)),
        player: stub,
        budget: Budget::from_ms(10),
        timeout: Duration::from_secs(5),
        on_error,
    })
}

async fn plan(players: Vec<(&str, Arc<Stub>)>, openings: Vec<Opening>, concurrency: usize) -> Plan {
    let mut entrants = Vec::new();
    for (name, stub) in players {
        entrants.push(Entrant {
            identify: stub.identify().await.unwrap(),
            seat: seat(name, stub, ErrorPolicy::Forfeit),
            source: "stub".into(),
        });
    }
    let n = entrants.len();
    Plan {
        game: epam(),
        entrants,
        openings,
        pairings: (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect(),
        concurrency,
        max_plies: 12_000,
        seed: 5,
    }
}

fn some_openings(count: usize) -> Vec<Opening> {
    openings::generate(epam().as_ref(), &OpeningSpec::Random { plies: 2, count }, 9).unwrap()
}

async fn round_robin(concurrency: usize) -> TournamentResult {
    let players = vec![
        ("alice", Stub::new(Behaviour::Legal(1))),
        ("bob", Stub::new(Behaviour::Legal(2))),
        ("carol", Stub::new(Behaviour::Greedy)),
    ];
    run(plan(players, some_openings(4), concurrency).await, None).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_round_robin_plays_every_minimatch_both_ways() {
    let r = round_robin(8).await;
    // 3 pairings x 4 openings x 2 colours.
    assert_eq!(r.games.len(), 24);
    assert_eq!(r.minimatches.len(), 12);
    assert_eq!(r.pairings.len(), 3);
    for m in &r.minimatches {
        let [g0, g1] =
            [&m.games[0], &m.games[1]].map(|id| r.games.iter().find(|g| g.id == *id).unwrap());
        // Same opening, colours swapped.
        assert_eq!(g0.opening, g1.opening);
        assert_eq!(g0.players, [m.a.clone(), m.b.clone()]);
        assert_eq!(g1.players, [m.b.clone(), m.a.clone()]);
        let opening = r.openings.iter().find(|o| o.id == g0.opening).unwrap();
        // Both legs replay from the recorded opening to the recorded final state.
        for g in [g0, g1] {
            let mut state = opening.state.clone();
            for mv in &g.moves {
                state = epam().play(&state, &mv.mv, &mut rand::rng()).unwrap().state;
            }
            assert_eq!(state, g.final_state, "{}", g.id);
            assert_eq!(g.termination, Termination::Rules);
            assert_eq!(epam().outcome(&state).unwrap(), Some(g.outcome));
        }
    }
    // Every player appears in the standings, and pairing scores are complementary.
    assert_eq!(r.standings.len(), 3);
    for p in &r.pairings {
        let t = p.tally;
        assert_eq!(t.counted + t.excluded, 4);
        assert_eq!(t.wins + t.draws + t.losses, 8);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_result_does_not_depend_on_concurrency() {
    let serial = round_robin(1).await.without_timings();
    let parallel = round_robin(16).await.without_timings();
    assert_eq!(serial, parallel);
    let json = |r: &TournamentResult| serde_json::to_string_pretty(r).unwrap();
    assert_eq!(json(&serial), json(&parallel));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_never_sees_more_requests_than_it_allows() {
    let limited = Stub::limited(Behaviour::Legal(1), 2);
    let free = Stub::new(Behaviour::Legal(2));
    let p = plan(
        vec![("limited", limited.clone()), ("free", free.clone())],
        some_openings(6),
        12,
    )
    .await;
    let r = run(p, None).await;
    assert_eq!(r.games.len(), 12);
    assert!(
        limited.peak.load(Ordering::SeqCst) <= 2,
        "peak {}",
        limited.peak.load(Ordering::SeqCst)
    );
    assert!(
        free.peak.load(Ordering::SeqCst) > 2,
        "the unlimited client should see overlap"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_from_different_games_interleave() {
    // The protocol's premise: a client is asked about many games at once. Make sure the server
    // really does interleave them rather than serialising per client.
    let a = Stub::new(Behaviour::Legal(1));
    let p = plan(
        vec![("a", a.clone()), ("b", Stub::new(Behaviour::Legal(2)))],
        some_openings(3),
        6,
    )
    .await;
    run(p, None).await;
    let seen = a.seen.lock().unwrap();
    let game_of = |id: &String| id.rsplitn(3, '-').nth(2).unwrap().to_string();
    let switches = seen
        .windows(2)
        .filter(|w| game_of(&w[0]) != game_of(&w[1]))
        .count();
    assert!(
        switches > seen.len() / 4,
        "only {switches} switches in {} requests",
        seen.len()
    );
}

#[tokio::test]
async fn the_progress_callback_sees_every_game() {
    let count = Arc::new(AtomicU32::new(0));
    let c = count.clone();
    let p = plan(
        vec![
            ("a", Stub::new(Behaviour::Legal(1))),
            ("b", Stub::new(Behaviour::Legal(2))),
        ],
        some_openings(2),
        2,
    )
    .await;
    run(
        p,
        Some(Box::new(move |_, n, total| {
            assert!(n <= total);
            c.fetch_add(1, Ordering::SeqCst);
        })),
    )
    .await;
    assert_eq!(count.load(Ordering::SeqCst), 4);
}

// ─── One game at a time, for the failure policies ─────────────────────────────

async fn one_game(
    white: Arc<Seat>,
    black: Arc<Seat>,
    max_plies: u32,
) -> tdg_tourney::engine::GameRecord {
    let opening = Arc::new(Opening {
        id: "start".into(),
        state: epam().initial(&mut rand::rng()),
        moves: vec![],
    });
    play_game(
        epam(),
        GameSpec {
            id: "g".into(),
            match_id: "m".into(),
            opening,
            seats: [white, black],
            max_plies,
            seed: 3,
        },
    )
    .await
}

fn honest(name: &str) -> Arc<Seat> {
    seat(name, Stub::new(Behaviour::Legal(7)), ErrorPolicy::Forfeit)
}

#[tokio::test]
async fn an_illegal_move_forfeits_by_default() {
    let r = one_game(
        seat("bad", Stub::new(Behaviour::Illegal), ErrorPolicy::Forfeit),
        honest("ok"),
        12_000,
    )
    .await;
    assert_eq!(r.outcome, Outcome::win(Side::Second));
    assert_eq!(r.termination, Termination::Forfeit { side: Side::First });
    assert_eq!(r.moves.len(), 0);
    assert_eq!(r.incidents.len(), 1);
    assert!(
        matches!(r.incidents[0].error, IncidentKind::IllegalMove(_)),
        "{:?}",
        r.incidents
    );
    assert_eq!(r.incidents[0].player, "bad");
}

#[tokio::test]
async fn every_kind_of_failure_is_caught() {
    for (behaviour, check) in [
        (
            Behaviour::ServerError,
            (|k: &IncidentKind| {
                matches!(
                    k,
                    IncidentKind::Client(PlayerError::Status { status: 500, .. })
                )
            }) as fn(&IncidentKind) -> bool,
        ),
        (
            Behaviour::WrongId,
            |k| matches!(k, IncidentKind::WrongRequestId(id) if id == "something-else"),
        ),
        (Behaviour::FailFirst(99), |k| {
            matches!(k, IncidentKind::Client(PlayerError::Timeout))
        }),
    ] {
        let r = one_game(
            honest("ok"),
            seat("bad", Stub::new(behaviour), ErrorPolicy::Forfeit),
            12_000,
        )
        .await;
        assert_eq!(r.termination, Termination::Forfeit { side: Side::Second });
        assert_eq!(r.moves.len(), 1, "white moved once, then black failed");
        assert!(check(&r.incidents[0].error), "{:?}", r.incidents[0]);
    }
}

#[tokio::test]
async fn a_malformed_move_is_distinguished_from_an_illegal_one() {
    struct Garbage;
    #[async_trait]
    impl Player for Garbage {
        async fn identify(&self) -> Result<Identify, PlayerError> {
            unreachable!()
        }
        async fn choose(
            &self,
            req: &MoveRequest,
            _: Duration,
        ) -> Result<MoveResponse, PlayerError> {
            Ok(MoveResponse {
                request_id: req.request_id.clone(),
                mv: serde_json::json!("e4"),
                info: None,
            })
        }
    }
    let bad = Arc::new(Seat {
        name: "garbage".into(),
        player: Arc::new(Garbage),
        limit: Arc::new(Semaphore::new(1)),
        budget: Budget::default(),
        timeout: Duration::from_secs(1),
        on_error: ErrorPolicy::Forfeit,
    });
    let r = one_game(bad, honest("ok"), 12_000).await;
    assert!(
        matches!(r.incidents[0].error, IncidentKind::MalformedMove(_)),
        "{:?}",
        r.incidents
    );
}

#[tokio::test]
async fn retry_asks_exactly_n_more_times_then_forfeits() {
    let stub = Stub::new(Behaviour::Illegal);
    let r = one_game(
        seat("bad", stub.clone(), ErrorPolicy::Retry(3)),
        honest("ok"),
        12_000,
    )
    .await;
    assert_eq!(
        stub.calls.load(Ordering::SeqCst),
        4,
        "the first ask plus three retries"
    );
    assert_eq!(r.termination, Termination::Forfeit { side: Side::First });
    assert_eq!(r.incidents.len(), 4);
    assert!(r.incidents[..3].iter().all(|i| i.action == "asked again"));
    assert_eq!(r.incidents[3].action, "forfeited the game");
    // Each attempt had its own request id.
    let seen = stub.seen.lock().unwrap();
    assert_eq!(*seen, ["g-p0-a0", "g-p0-a1", "g-p0-a2", "g-p0-a3"]);
}

#[tokio::test]
async fn retry_recovers_when_the_client_does() {
    let stub = Stub::new(Behaviour::FailFirst(2));
    let r = one_game(
        seat("flaky", stub, ErrorPolicy::Retry(2)),
        honest("ok"),
        12_000,
    )
    .await;
    assert_eq!(
        r.termination,
        Termination::Rules,
        "two failures fit in two retries"
    );
    assert_eq!(r.incidents.len(), 2);
}

#[tokio::test]
async fn random_policy_keeps_the_game_going() {
    let r = one_game(
        seat("bad", Stub::new(Behaviour::Illegal), ErrorPolicy::Random),
        honest("ok"),
        12_000,
    )
    .await;
    assert_eq!(
        r.termination,
        Termination::Rules,
        "the game ran to a real finish"
    );
    let white_moves: Vec<_> = r.moves.iter().filter(|m| m.side == Side::First).collect();
    assert!(!white_moves.is_empty());
    assert!(
        white_moves.iter().all(|m| m.by_server),
        "every white move was the server's"
    );
    assert!(r
        .moves
        .iter()
        .filter(|m| m.side == Side::Second)
        .all(|m| !m.by_server));
    assert_eq!(r.incidents.len(), white_moves.len());
}

#[tokio::test]
async fn random_policy_is_reproducible() {
    let play = || {
        one_game(
            seat("bad", Stub::new(Behaviour::Illegal), ErrorPolicy::Random),
            honest("ok"),
            12_000,
        )
    };
    let (a, b) = (play().await, play().await);
    assert_eq!(
        a.moves.iter().map(|m| &m.san).collect::<Vec<_>>(),
        b.moves.iter().map(|m| &m.san).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn the_ply_cap_ends_a_game_as_a_flagged_draw() {
    let r = one_game(honest("a"), honest("b"), 10).await;
    assert_eq!(r.moves.len(), 10);
    assert_eq!(r.outcome, Outcome::Draw);
    assert_eq!(r.termination, Termination::PlyCap);
    assert!(matches!(r.incidents[0].error, IncidentKind::PlyCap(10)));
}

#[tokio::test]
async fn a_decided_game_stops_immediately() {
    // An opening that is already decided (white has an unanswered incursion) asks nobody.
    let stub = Stub::new(Behaviour::Legal(1));
    let mut board = vec!['.'; 168];
    board[160] = 'W';
    board[70] = 'B'; // mid-board, not on white's home rank
    let state = serde_json::json!({
        "board": board.iter().collect::<String>(), "to_move": 0, "moves": 9,
        "quiet_moves": 9, "last_capturer": -1
    });
    let opening = Arc::new(Opening {
        id: "won".into(),
        state,
        moves: vec![],
    });
    let s = seat("a", stub.clone(), ErrorPolicy::Forfeit);
    let r = play_game(
        epam(),
        GameSpec {
            id: "g".into(),
            match_id: "m".into(),
            opening,
            seats: [s.clone(), s],
            max_plies: 100,
            seed: 0,
        },
    )
    .await;
    assert_eq!(r.outcome, Outcome::win(Side::First));
    assert_eq!(stub.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn think_time_is_recorded() {
    let r = one_game(
        seat("slow", Stub::new(Behaviour::Slow(30)), ErrorPolicy::Forfeit),
        honest("b"),
        4,
    )
    .await;
    let slow: Vec<_> = r.moves.iter().filter(|m| m.side == Side::First).collect();
    assert!(
        slow.iter().all(|m| m.elapsed_ms >= 30),
        "{:?}",
        slow.iter().map(|m| m.elapsed_ms).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn standings_count_incidents_and_forfeits() {
    let mut p = plan(
        vec![
            ("bad", Stub::new(Behaviour::Illegal)),
            ("ok", Stub::new(Behaviour::Legal(1))),
        ],
        some_openings(3),
        4,
    )
    .await;
    // `plan` uses forfeit for everyone.
    p.concurrency = 4;
    let r = run(p, None).await;
    let bad = r.standings.iter().find(|s| s.name == "bad").unwrap();
    assert_eq!(bad.forfeits, 6, "forfeits every game it plays");
    assert_eq!(bad.incidents, 6);
    assert_eq!(bad.percentage, Some(0.0));
    let ok = r.standings.iter().find(|s| s.name == "ok").unwrap();
    assert_eq!(ok.percentage, Some(100.0));
    assert_eq!(r.standings[0].name, "ok", "best first");
    assert!(tdg_tourney::report::render(&r).contains("6 forfeits"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn think_time_excludes_waiting_for_a_busy_client() {
    // A client that takes 20ms a move and serves one request at a time, playing four games at
    // once: requests queue, but each move's recorded time is its own 20ms, not the queue's.
    let slow = Stub::limited(Behaviour::Slow(20), 1);
    let p = plan(
        vec![("slow", slow), ("fast", Stub::new(Behaviour::Legal(1)))],
        some_openings(2),
        4,
    )
    .await;
    let mut p = p;
    p.max_plies = 12;
    let r = run(p, None).await;
    let times: Vec<u64> = r
        .games
        .iter()
        .flat_map(|g| {
            g.moves
                .iter()
                .filter(|m| g.players[m.side.index()] == "slow")
                .map(|m| m.elapsed_ms)
        })
        .collect();
    assert!(times.iter().all(|&t| (20..60).contains(&t)), "{times:?}");
}

/// Plays Pig by holding at 20: its move depends on the position alone.
struct HoldAt20;

#[async_trait]
impl Player for HoldAt20 {
    async fn identify(&self) -> Result<Identify, PlayerError> {
        Ok(Identify {
            protocol: ProtocolVersion::CURRENT,
            name: "hold-at-20".into(),
            version: "0".into(),
            games: vec!["pig/1".into()],
            max_concurrency: 1,
        })
    }

    async fn choose(&self, req: &MoveRequest, _: Duration) -> Result<MoveResponse, PlayerError> {
        let s = &req.state;
        let roll = s["last_roll"] != 1 && s["turn_total"].as_u64().unwrap() < 20;
        Ok(MoveResponse {
            request_id: req.request_id.clone(),
            mv: serde_json::json!({"action": if roll { "roll" } else { "stop" }}),
            info: None,
        })
    }
}

#[tokio::test]
async fn both_games_of_a_pig_minimatch_roll_the_same_dice() {
    let pig = tdg_games::by_id("pig/1").unwrap();
    let seat = |name: &str| {
        Arc::new(Seat {
            name: name.into(),
            player: Arc::new(HoldAt20),
            limit: Arc::new(Semaphore::new(1)),
            budget: Budget::from_ms(10),
            timeout: Duration::from_secs(5),
            on_error: ErrorPolicy::Forfeit,
        })
    };
    let (a, b) = (seat("a"), seat("b"));
    let opening = Arc::new(Opening {
        id: "start".into(),
        state: serde_json::json!({"totals": [0, 0], "turn_total": 4, "last_roll": 4, "to_move": 0}),
        moves: vec![],
    });
    let play = |id: &str, match_id: &str, seats: [Arc<Seat>; 2]| {
        play_game(
            pig.clone(),
            GameSpec {
                id: id.into(),
                match_id: match_id.into(),
                opening: opening.clone(),
                seats,
                max_plies: 12_000,
                seed: 5,
            },
        )
    };
    let first = play("m-a", "m", [a.clone(), b.clone()]).await;
    let second = play("m-b", "m", [b.clone(), a.clone()]).await;
    let sans = |g: &tdg_tourney::engine::GameRecord| {
        g.moves.iter().map(|m| m.san.clone()).collect::<Vec<_>>()
    };
    // The same dice and the same strategy on both sides: the same game, won by the same seat.
    assert!(first.moves.len() > 20, "{} moves", first.moves.len());
    assert_eq!(first.termination, Termination::Rules);
    assert_eq!(sans(&first), sans(&second));
    assert_eq!(first.final_state, second.final_state);
    assert_eq!(first.outcome, second.outcome);
    // Another minimatch rolls other dice.
    let other = play("n-a", "n", [a, b]).await;
    assert_ne!(first.final_state, other.final_state);
}
