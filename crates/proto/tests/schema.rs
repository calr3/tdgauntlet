//! The JSON Schemas in `schema/` are the normative machine-readable form of the protocol, and the
//! examples in the markdown are what client authors copy. Both are checked here against each
//! other and against the Rust types, so none of the three can drift alone.

use serde_json::Value;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn schema(name: &str) -> Value {
    let path = repo_root().join("schema").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", name))
}

/// All schemas, keyed by their `$id`, so sibling `$ref`s resolve without touching the network.
fn registry() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(repo_root().join("schema")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        let value = schema(&name);
        let id = value["$id"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} has no $id"))
            .to_string();
        assert!(id.ends_with(&name), "{name} has a mismatched $id: {id}");
        out.push((id, value));
    }
    out
}

fn compile(name: &str) -> jsonschema::Validator {
    let mut options = jsonschema::options();
    for (id, value) in registry() {
        options = options.with_resource(
            id.clone(),
            jsonschema::Resource::from_contents(value)
                .unwrap_or_else(|e| panic!("{id} is not a usable resource: {e}")),
        );
    }
    options
        .build(&schema(name))
        .unwrap_or_else(|e| panic!("{name} is not a valid schema: {e}"))
}

fn assert_valid(validator: &jsonschema::Validator, value: &Value, what: &str) {
    if let Err(e) = validator.validate(value) {
        panic!(
            "{what} failed schema validation: {e}\n{}",
            serde_json::to_string_pretty(value).unwrap()
        );
    }
}

fn assert_invalid(validator: &jsonschema::Validator, value: &Value, what: &str) {
    assert!(
        validator.validate(value).is_err(),
        "{what} should have been rejected"
    );
}

/// Every ```json fenced block in a markdown document.
fn json_blocks(doc: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in doc.lines() {
        match (&mut current, line.trim_start()) {
            (None, l) if l.starts_with("```json") => current = Some(String::new()),
            (Some(buf), l) if l.starts_with("```") => {
                let text = std::mem::take(buf);
                current = None;
                out.push(
                    serde_json::from_str(&text)
                        .unwrap_or_else(|e| panic!("unparseable json block: {e}\n{text}")),
                );
            }
            (Some(buf), _) => {
                buf.push_str(line);
                buf.push('\n');
            }
            _ => {}
        }
    }
    assert!(current.is_none(), "unterminated code fence");
    out
}

#[test]
fn every_schema_compiles() {
    for entry in std::fs::read_dir(repo_root().join("schema")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        compile(&name);
    }
}

#[test]
fn protocol_md_examples_match_their_schemas() {
    let doc = std::fs::read_to_string(repo_root().join("PROTOCOL.md")).unwrap();
    let (identify, request, response, error) = (
        compile("identify.schema.json"),
        compile("move_request.schema.json"),
        compile("move_response.schema.json"),
        compile("error.schema.json"),
    );
    let mut seen = 0;
    for value in json_blocks(&doc) {
        let obj = value.as_object().expect("examples are objects");
        let (v, what) = if obj.contains_key("games") {
            (&identify, "identify example")
        } else if obj.contains_key("state") {
            (&request, "move request example")
        } else if obj.contains_key("move") {
            (&response, "move response example")
        } else if obj.contains_key("error") {
            (&error, "error example")
        } else {
            panic!("unrecognised example block:\n{value}");
        };
        assert_valid(v, &value, what);
        seen += 1;
    }
    assert_eq!(
        seen, 4,
        "PROTOCOL.md should carry one example of each message"
    );
}

#[test]
fn epaminondas_doc_examples_match_their_schemas() {
    let doc = std::fs::read_to_string(repo_root().join("games/epaminondas.md")).unwrap();
    let (state, mv) = (
        compile("epaminondas_1.state.schema.json"),
        compile("epaminondas_1.move.schema.json"),
    );
    let mut seen = (false, false);
    for value in json_blocks(&doc) {
        let obj = value.as_object().unwrap();
        if obj.contains_key("board") {
            // The document elides the middle of the board string for readability, so check the
            // fields that are really there rather than the length.
            for key in ["to_move", "moves", "quiet_moves", "last_capturer"] {
                assert!(obj.contains_key(key), "the state example lost {key}");
            }
            seen.0 = true;
        } else {
            assert_valid(&mv, &value, "epaminondas move example");
            seen.1 = true;
        }
    }
    assert_eq!(seen, (true, true));
    // A real full-length board does validate.
    let board: String = ".".repeat(168);
    assert_valid(
        &state,
        &serde_json::json!({
            "board": board, "to_move": 0, "moves": 0, "quiet_moves": 0, "last_capturer": -1
        }),
        "empty board",
    );
}

#[test]
fn gess_doc_examples_match_their_schemas() {
    let doc = std::fs::read_to_string(repo_root().join("games/gess.md")).unwrap();
    let (state, mv) = (
        compile("gess_1.state.schema.json"),
        compile("gess_1.move.schema.json"),
    );
    let mut seen = (false, false);
    for value in json_blocks(&doc) {
        let obj = value.as_object().unwrap();
        if obj.contains_key("board") {
            // The document elides the middle of the board string for readability, so check the
            // fields that are really there rather than the length.
            for key in ["to_move", "moves", "quiet_moves"] {
                assert!(obj.contains_key(key), "the state example lost {key}");
            }
            seen.0 = true;
        } else {
            assert_valid(&mv, &value, "gess move example");
            seen.1 = true;
        }
    }
    assert_eq!(seen, (true, true));
    // A real full-length board does validate, and a stone off the 20 x 20 grid does not.
    let board: String = ".".repeat(400);
    assert_valid(
        &state,
        &serde_json::json!({"board": board, "to_move": 0, "moves": 0, "quiet_moves": 0}),
        "empty board",
    );
    assert_invalid(
        &state,
        &serde_json::json!({"board": ".".repeat(399), "to_move": 0, "moves": 0, "quiet_moves": 0}),
        "399 cells",
    );
    assert_invalid(
        &mv,
        &serde_json::json!({"from": 0, "to": 400}),
        "a cell off the grid",
    );
}

#[test]
fn one_more_tri_doc_examples_match_their_schemas() {
    let doc = std::fs::read_to_string(repo_root().join("games/one-more-tri.md")).unwrap();
    let (state, mv) = (
        compile("one-more-tri_1.state.schema.json"),
        compile("one-more-tri_1.move.schema.json"),
    );
    let mut seen = (false, false);
    for value in json_blocks(&doc) {
        if value.get("board").is_some() {
            assert_valid(&state, &value, "one-more-tri state example");
            seen.0 = true;
        } else {
            assert_valid(&mv, &value, "one-more-tri move example");
            seen.1 = true;
        }
    }
    assert_eq!(seen, (true, true));
    let empty = serde_json::json!(vec![0; 21]);
    assert_valid(
        &state,
        &serde_json::json!({"board": empty, "to_move": 0}),
        "the empty board",
    );
    for (board, why) in [
        (serde_json::json!(vec![0; 20]), "20 triangles"),
        (serde_json::json!(vec![11; 21]), "a tile of 11"),
    ] {
        assert_invalid(
            &state,
            &serde_json::json!({"board": board, "to_move": 0}),
            why,
        );
    }
    for (value, why) in [
        (serde_json::json!({"tri": 21, "tile": 3}), "triangle 21"),
        (serde_json::json!({"tri": 3, "tile": 0}), "tile 0"),
        (serde_json::json!({"tri": 3}), "no tile"),
    ] {
        assert_invalid(&mv, &value, why);
    }
}

#[test]
fn schemas_reject_what_the_rust_types_reject() {
    let request = compile("move_request.schema.json");
    let base = serde_json::json!({
        "protocol": 1, "request_id": "r", "game": "epaminondas/1", "state": {},
        "context": {"match_id": "m", "game_id": "g", "ply": 0, "side": 0}
    });
    assert_valid(&request, &base, "the baseline request");

    for (field, bad) in [
        ("protocol", serde_json::json!(2)),
        ("game", serde_json::json!("Epaminondas")),
        ("request_id", serde_json::json!("")),
    ] {
        let mut v = base.clone();
        v[field] = bad.clone();
        assert_invalid(&request, &v, &format!("{field} = {bad}"));
        // And the Rust types agree, at least where they can express the same constraint.
        if field == "protocol" {
            assert!(serde_json::from_value::<tdg_proto::MoveRequest>(v).is_err());
        }
    }

    let mut v = base.clone();
    v["context"]["side"] = serde_json::json!(2);
    assert_invalid(&request, &v, "side = 2");
    assert!(serde_json::from_value::<tdg_proto::MoveRequest>(v).is_err());

    let mut v = base.clone();
    v.as_object_mut().unwrap().remove("state");
    assert_invalid(&request, &v, "a request with no state");
}

#[test]
fn epaminondas_state_schema_enforces_the_board_shape() {
    let state = compile("epaminondas_1.state.schema.json");
    let ok: String = ".".repeat(168);
    for (board, why) in [
        (ok.clone(), ""),
        (ok[1..].to_string(), "167 squares"),
        (format!("{ok}."), "169 squares"),
        (ok.replace('.', "x"), "unknown piece character"),
    ] {
        let v = serde_json::json!({
            "board": board, "to_move": 0, "moves": 0, "quiet_moves": 0, "last_capturer": -1
        });
        if why.is_empty() {
            assert_valid(&state, &v, "a legal empty board");
        } else {
            assert_invalid(&state, &v, why);
        }
    }
    assert_invalid(
        &state,
        &serde_json::json!({"board": ok, "to_move": 0, "moves": 3, "last_capturer": -1}),
        "no quiet_moves",
    );
    assert_invalid(
        &state,
        &serde_json::json!({
            "board": ok, "to_move": 0, "moves": 3, "quiet_moves": -1, "last_capturer": -1
        }),
        "negative quiet_moves",
    );
}
