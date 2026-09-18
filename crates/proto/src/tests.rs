use super::*;
use serde_json::json;

fn sample_request() -> MoveRequest {
    MoveRequest {
        protocol: ProtocolVersion::CURRENT,
        request_id: "req-1".into(),
        game: "epaminondas/1".into(),
        state: json!({"board": "...", "to_move": 0}),
        budget: Budget::from_ms(1000),
        context: Context {
            match_id: "m1".into(),
            game_id: "m1-a".into(),
            ply: 26,
            side: Side::First,
        },
    }
}

fn roundtrip<T>(value: &T)
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let text = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(value, &back, "round-trip changed the value (via {text})");
}

#[test]
fn every_message_round_trips() {
    roundtrip(&Identify {
        protocol: ProtocolVersion::CURRENT,
        name: "rando".into(),
        version: "0.1.0".into(),
        games: vec!["epaminondas/1".into()],
        max_concurrency: 8,
    });
    roundtrip(&sample_request());
    roundtrip(&MoveResponse {
        request_id: "req-1".into(),
        mv: json!({"lead": 31, "rear": 17, "dest": 59}),
        info: Some(MoveInfo {
            depth: Some(9),
            nodes: Some(184203),
            ..MoveInfo::default()
        }),
    });
    roundtrip(&ErrorResponse::new("unsupported game"));
    for side in [Side::First, Side::Second] {
        roundtrip(&side);
        roundtrip(&Outcome::win(side));
    }
    roundtrip(&Outcome::Draw);
}

#[test]
fn move_is_serialized_under_its_json_name() {
    let text = serde_json::to_string(&MoveResponse {
        request_id: "r".into(),
        mv: json!(7),
        info: None,
    })
    .unwrap();
    // `move` is a Rust keyword, so the field is `mv` in Rust and must be renamed on the wire.
    assert!(text.contains("\"move\":7"), "{text}");
    assert!(!text.contains("\"mv\""), "{text}");
    // An absent `info` is omitted rather than sent as null.
    assert!(!text.contains("info"), "{text}");
}

#[test]
fn unknown_protocol_versions_are_rejected() {
    for v in [0, 2, 99] {
        let text = format!(
            r#"{{"protocol":{v},"request_id":"r","game":"g","state":{{}},
                "context":{{"match_id":"m","game_id":"g","ply":0,"side":0}}}}"#
        );
        let err = serde_json::from_str::<MoveRequest>(&text)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported protocol version"), "{v}: {err}");
    }
}

#[test]
fn missing_required_fields_are_rejected() {
    // No `game`.
    let text = r#"{"protocol":1,"request_id":"r","state":{},
                   "context":{"match_id":"m","game_id":"g","ply":0,"side":0}}"#;
    assert!(serde_json::from_str::<MoveRequest>(text).is_err());
    // No `move`.
    assert!(serde_json::from_str::<MoveResponse>(r#"{"request_id":"r"}"#).is_err());
    // No `games`.
    assert!(serde_json::from_str::<Identify>(r#"{"protocol":1,"name":"x"}"#).is_err());
}

#[test]
fn optional_fields_have_workable_defaults() {
    let req: MoveRequest = serde_json::from_str(
        r#"{"protocol":1,"request_id":"r","game":"g","state":{},
            "context":{"match_id":"m","game_id":"g","ply":0,"side":1}}"#,
    )
    .unwrap();
    assert_eq!(
        req.budget,
        Budget::default(),
        "an absent budget means no stated limit"
    );
    assert_eq!(req.context.side, Side::Second);

    let id: Identify = serde_json::from_str(r#"{"protocol":1,"name":"x","games":["g"]}"#).unwrap();
    assert_eq!(
        id.max_concurrency, 1,
        "a client that says nothing gets one request at a time"
    );
    assert_eq!(id.version, "");
}

#[test]
fn explicit_nulls_in_a_budget_mean_unset() {
    let b: Budget = serde_json::from_str(r#"{"ms":1000,"nodes":null,"sims":null}"#).unwrap();
    assert_eq!(b, Budget::from_ms(1000));
    assert_eq!(serde_json::to_string(&b).unwrap(), r#"{"ms":1000}"#);
}

#[test]
fn unknown_fields_are_tolerated_for_forward_compatibility() {
    let req: MoveRequest = serde_json::from_str(
        r#"{"protocol":1,"request_id":"r","game":"g","state":{},"future_field":42,
            "context":{"match_id":"m","game_id":"g","ply":0,"side":0}}"#,
    )
    .unwrap();
    assert_eq!(req.request_id, "r");
}

#[test]
fn extra_info_keys_are_kept() {
    let info: MoveInfo =
        serde_json::from_str(r#"{"depth":3,"tbhits":17,"note":"tricky"}"#).unwrap();
    assert_eq!(info.depth, Some(3));
    assert_eq!(info.extra["tbhits"], json!(17));
    assert_eq!(info.extra["note"], json!("tricky"));
    roundtrip(&info);
}

#[test]
fn side_must_be_zero_or_one() {
    assert!(serde_json::from_str::<Side>("2").is_err());
    assert!(serde_json::from_str::<Side>("-1").is_err());
    assert_eq!(serde_json::from_str::<Side>("0").unwrap(), Side::First);
}

#[test]
fn sides_and_outcomes_are_consistent() {
    assert_eq!(Side::First.other(), Side::Second);
    assert_eq!(Side::Second.other().other(), Side::Second);
    for side in [Side::First, Side::Second] {
        let win = Outcome::win(side);
        assert_eq!(win.score_for(side), 1.0);
        assert_eq!(win.score_for(side.other()), 0.0);
        assert_eq!(win.flipped(), Outcome::win(side.other()));
        assert_eq!(Outcome::Draw.score_for(side), 0.5);
        // The two players' scores always sum to 1.
        for outcome in [win, Outcome::Draw] {
            assert_eq!(
                outcome.score_for(side) + outcome.score_for(side.other()),
                1.0
            );
        }
    }
    assert_eq!(Outcome::Draw.flipped(), Outcome::Draw);
}

#[test]
fn identify_reports_which_games_it_plays() {
    let id = Identify {
        protocol: ProtocolVersion::CURRENT,
        name: "x".into(),
        version: String::new(),
        games: vec!["epaminondas/1".into()],
        max_concurrency: 1,
    };
    assert!(id.plays("epaminondas/1"));
    assert!(
        !id.plays("epaminondas/2"),
        "the version is part of the game id"
    );
    assert!(!id.plays("gess/1"));
}

/// The examples in `PROTOCOL.md` are the thing client authors copy, so they are parsed here as
/// part of the test suite: the document cannot drift away from the types without failing CI.
#[test]
fn protocol_md_examples_parse() {
    let doc = include_str!("../../../PROTOCOL.md");
    let mut seen = (false, false, false, false);
    for block in json_blocks(doc) {
        let value: Value = serde_json::from_str(&block)
            .unwrap_or_else(|e| panic!("PROTOCOL.md has an unparseable json block: {e}\n{block}"));
        let obj = value.as_object().expect("every example is an object");
        if obj.contains_key("games") {
            serde_json::from_str::<Identify>(&block).expect("identify example");
            seen.0 = true;
        } else if obj.contains_key("state") {
            serde_json::from_str::<MoveRequest>(&block).expect("move request example");
            seen.1 = true;
        } else if obj.contains_key("move") {
            serde_json::from_str::<MoveResponse>(&block).expect("move response example");
            seen.2 = true;
        } else if obj.contains_key("error") {
            serde_json::from_str::<ErrorResponse>(&block).expect("error example");
            seen.3 = true;
        } else {
            panic!("unrecognised example block in PROTOCOL.md:\n{block}");
        }
    }
    assert_eq!(
        seen,
        (true, true, true, true),
        "PROTOCOL.md lost one of its examples"
    );
}

/// Every ```json fenced block in a markdown document.
fn json_blocks(doc: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;
    for line in doc.lines() {
        match (&mut current, line.trim_start()) {
            (None, l) if l.starts_with("```json") => current = Some(String::new()),
            (Some(_), l) if l.starts_with("```") => blocks.push(current.take().unwrap()),
            (Some(buf), _) => {
                buf.push_str(line);
                buf.push('\n');
            }
            _ => {}
        }
    }
    assert!(current.is_none(), "unterminated code fence");
    blocks
}
