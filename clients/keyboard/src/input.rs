//! Turning what was typed into a move, with no knowledge of any particular game: the typed text
//! is matched against each legal move's notation.

use serde_json::Value;

#[derive(Debug, PartialEq)]
pub enum Choice {
    Move(usize),
    List,
    Resign,
    Unknown(String),
}

/// Capture and quiet moves are written with `x` and `-`; accept either.
fn normalise(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('x', "-")
}

pub fn interpret(line: &str, moves: &[(Value, String)]) -> Choice {
    let text = line.trim();
    match text {
        "" => return Choice::Unknown("Type a move.".into()),
        "?" | "help" | "list" => return Choice::List,
        "resign" => return Choice::Resign,
        _ => {}
    }
    let want = normalise(text);
    let exact: Vec<usize> = moves
        .iter()
        .enumerate()
        .filter(|(_, (_, s))| normalise(s) == want)
        .map(|(i, _)| i)
        .collect();
    if let [i] = exact.as_slice() {
        return Choice::Move(*i);
    }
    // A unique prefix is enough, e.g. just the rear and lead of the only phalanx move from there.
    let prefixed: Vec<usize> = moves
        .iter()
        .enumerate()
        .filter(|(_, (_, s))| normalise(s).starts_with(&want))
        .map(|(i, _)| i)
        .collect();
    match prefixed.as_slice() {
        [i] => Choice::Move(*i),
        [] => Choice::Unknown(format!("{text} is not a legal move here.")),
        many => Choice::Unknown(format!(
            "{text} could be any of: {}",
            many.iter()
                .map(|&i| moves[i].1.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Lay names out in columns to fit `width`.
pub fn columns(names: &[&str], width: usize) -> String {
    let cell = names.iter().map(|n| n.len()).max().unwrap_or(0) + 2;
    let per_row = (width / cell.max(1)).max(1);
    names
        .chunks(per_row)
        .map(|row| {
            row.iter()
                .map(|n| format!("{n:<cell$}"))
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tdg_games::DynGame;

    fn start_moves() -> Vec<(Value, String)> {
        let g = tdg_games::epaminondas::Epaminondas;
        use tdg_games::Game;
        g.described_moves(&g.encode_state(&Game::initial(&g)))
            .unwrap()
    }

    fn san(moves: &[(Value, String)], c: Choice) -> String {
        match c {
            Choice::Move(i) => moves[i].1.clone(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn exact_notation_picks_that_move() {
        let m = start_moves();
        assert_eq!(san(&m, interpret("d2-d3\n", &m)), "d2-d3");
        assert_eq!(san(&m, interpret("  D1D2-D4 ", &m)), "d1d2-d4");
        assert_eq!(
            san(&m, interpret("d1d2xd4", &m)),
            "d1d2-d4",
            "x and - are interchangeable"
        );
    }

    #[test]
    fn a_unique_prefix_is_enough() {
        let m = start_moves();
        // Only one move starts with the single piece on n2 heading to m3.
        assert_eq!(san(&m, interpret("n2-m", &m)), "n2-m3");
    }

    #[test]
    fn ambiguity_and_nonsense_are_explained() {
        let m = start_moves();
        match interpret("d1d2", &m) {
            Choice::Unknown(msg) => assert!(msg.contains("d1d2-d3") && msg.contains("d1d2-d4")),
            other => panic!("{other:?}"),
        }
        assert!(
            matches!(interpret("a1-a2", &m), Choice::Unknown(msg) if msg.contains("not a legal"))
        );
        assert!(matches!(interpret("", &m), Choice::Unknown(_)));
    }

    #[test]
    fn commands() {
        let m = start_moves();
        assert_eq!(interpret("?", &m), Choice::List);
        assert_eq!(interpret("resign\n", &m), Choice::Resign);
    }

    #[test]
    fn columns_fit_the_width() {
        let text = columns(&["a1-a2", "b1-b2", "c1c2-c4"], 20);
        assert!(text.lines().all(|l| l.len() <= 20), "{text}");
        assert_eq!(text.lines().count(), 2);
    }
}
