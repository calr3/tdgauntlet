//! A Gess negamax engine for tdgauntlet: a native port of the Gess web app's search, with its
//! evaluation's weights made tunable. See `engine` for what was ported and what changed.

pub mod engine;

pub use engine::{
    evaluate, features, Board, Limits, SearchResult, Searcher, Switches, Weights, NAMES,
    NUM_FEATURES,
};
