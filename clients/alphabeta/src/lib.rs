//! Alpha-beta search for Epaminondas.
//!
//! Two searches share the transposition table and the client: [`search`], over King and
//! Peterson's LEONIDAS heuristics, and [`solve`], which has no game knowledge at all and scores
//! a position as a win, a loss or "don't know".

pub mod eval;
pub mod search;
pub mod solve;
pub mod tt;
