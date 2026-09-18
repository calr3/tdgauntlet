//! Minimatch scoring.
//!
//! A minimatch is one opening played twice with colours swapped. From one player's point of view
//! it scores:
//!
//! | result        | score | counted |
//! |---------------|-------|---------|
//! | 2 wins        | +1    | yes     |
//! | 1 win 1 draw  | +0.5  | yes     |
//! | 2 draws       | 0     | yes     |
//! | 1 win 1 loss  | —     | no      |
//! | 1 draw 1 loss | −0.5  | yes     |
//! | 2 losses      | −1    | yes     |
//!
//! A win and a loss means each player won from the same side of the same opening: the opening
//! decided it, and it says nothing about either player, so it leaves the denominator entirely.
//! The headline figure is `100 * mean((s + 1) / 2)` over counted minimatches: 50% is par, and the
//! two players of a pairing always sum to 100%.

use serde::{Deserialize, Serialize};
use tdg_proto::Outcome;
use tdg_proto::Side;

/// One game of a minimatch from a given player's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameResult {
    Win,
    Draw,
    Loss,
}

impl GameResult {
    pub fn for_side(outcome: Outcome, side: Side) -> GameResult {
        match outcome {
            Outcome::Draw => GameResult::Draw,
            Outcome::Win { side: w } if w == side => GameResult::Win,
            Outcome::Win { .. } => GameResult::Loss,
        }
    }

    pub fn reversed(self) -> GameResult {
        match self {
            GameResult::Win => GameResult::Loss,
            GameResult::Draw => GameResult::Draw,
            GameResult::Loss => GameResult::Win,
        }
    }

    fn points(self) -> i32 {
        match self {
            GameResult::Win => 1,
            GameResult::Draw => 0,
            GameResult::Loss => -1,
        }
    }
}

/// A minimatch score for one player, or `None` when the minimatch does not count.
pub fn minimatch_score(a: GameResult, b: GameResult) -> Option<f64> {
    let (wins, losses) = [a, b].iter().fold((0, 0), |(w, l), r| match r {
        GameResult::Win => (w + 1, l),
        GameResult::Loss => (w, l + 1),
        GameResult::Draw => (w, l),
    });
    if wins == 1 && losses == 1 {
        return None;
    }
    Some((a.points() + b.points()) as f64 / 2.0)
}

/// Running totals for one player, against one opponent or against the field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Tally {
    /// Minimatches that counted.
    pub counted: u32,
    /// Minimatches dropped as 1 win 1 loss.
    pub excluded: u32,
    /// Sum of minimatch scores over the counted ones, each in [-1, 1].
    pub score: f64,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl Tally {
    pub fn add(&mut self, a: GameResult, b: GameResult) {
        for r in [a, b] {
            match r {
                GameResult::Win => self.wins += 1,
                GameResult::Draw => self.draws += 1,
                GameResult::Loss => self.losses += 1,
            }
        }
        match minimatch_score(a, b) {
            Some(s) => {
                self.counted += 1;
                self.score += s;
            }
            None => self.excluded += 1,
        }
    }

    pub fn merge(&mut self, other: &Tally) {
        self.counted += other.counted;
        self.excluded += other.excluded;
        self.score += other.score;
        self.wins += other.wins;
        self.draws += other.draws;
        self.losses += other.losses;
    }

    /// The headline figure, or `None` if no minimatch counted — which is not the same as 0%.
    pub fn percentage(&self) -> Option<f64> {
        (self.counted > 0).then(|| 100.0 * (self.score / self.counted as f64 + 1.0) / 2.0)
    }

    /// Share of minimatches dropped as 1 win 1 loss. High means the openings are deciding games.
    pub fn exclusion_rate(&self) -> Option<f64> {
        let total = self.counted + self.excluded;
        (total > 0).then(|| 100.0 * self.excluded as f64 / total as f64)
    }

    /// The same tally from the opponent's side.
    pub fn reversed(&self) -> Tally {
        Tally {
            score: -self.score,
            wins: self.losses,
            losses: self.wins,
            ..*self
        }
    }
}

/// The Elo difference implied by a percentage score, clamped away from the infinities at 0% and
/// 100%. The formula is the one `pgx/examples/alphazero/model_tournament.py::_elo` uses.
pub fn elo_difference(percentage: f64) -> f64 {
    let p = (percentage / 100.0).clamp(0.001, 0.999);
    -400.0 * (1.0 / p - 1.0).log10()
}

#[cfg(test)]
mod tests {
    use super::GameResult::{Draw, Loss, Win};
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn the_six_minimatch_cases() {
        // (first game, second game, score, counted)
        let table = [
            (Win, Win, Some(1.0)),
            (Win, Draw, Some(0.5)),
            (Draw, Draw, Some(0.0)),
            (Win, Loss, None),
            (Draw, Loss, Some(-0.5)),
            (Loss, Loss, Some(-1.0)),
        ];
        for (a, b, want) in table {
            assert_eq!(minimatch_score(a, b), want, "{a:?} {b:?}");
            // The order of the two games does not matter.
            assert_eq!(minimatch_score(b, a), want, "{b:?} {a:?}");
            // And the opponent sees the negation (or also nothing).
            assert_eq!(
                minimatch_score(a.reversed(), b.reversed()),
                want.map(|s| -s)
            );
        }
    }

    #[test]
    fn one_win_one_loss_touches_neither_numerator_nor_denominator() {
        let mut t = Tally::default();
        t.add(Win, Win);
        assert_eq!(t.percentage(), Some(100.0));
        t.add(Win, Loss);
        t.add(Loss, Win);
        assert_eq!(
            t.percentage(),
            Some(100.0),
            "excluded minimatches must not dilute the score"
        );
        assert_eq!((t.counted, t.excluded), (1, 2));
        // The raw game counts still include them.
        assert_eq!((t.wins, t.losses), (4, 2));
    }

    #[test]
    fn two_draws_count_in_both() {
        let mut t = Tally::default();
        t.add(Win, Win);
        t.add(Draw, Draw);
        assert_eq!(t.counted, 2);
        assert_eq!(t.percentage(), Some(75.0), "(100% + 50%) / 2");
    }

    #[test]
    fn percentages_for_each_case_alone() {
        for (a, b, want) in [
            (Win, Win, Some(100.0)),
            (Win, Draw, Some(75.0)),
            (Draw, Draw, Some(50.0)),
            (Win, Loss, None),
            (Draw, Loss, Some(25.0)),
            (Loss, Loss, Some(0.0)),
        ] {
            let mut t = Tally::default();
            t.add(a, b);
            assert_eq!(t.percentage(), want, "{a:?} {b:?}");
        }
    }

    #[test]
    fn an_all_excluded_tally_has_no_percentage_rather_than_zero() {
        let mut t = Tally::default();
        assert_eq!(t.percentage(), None, "nothing played");
        assert_eq!(t.exclusion_rate(), None);
        t.add(Win, Loss);
        t.add(Loss, Win);
        assert_eq!(t.percentage(), None, "nothing counted");
        assert_eq!(t.exclusion_rate(), Some(100.0));
        assert_eq!(t.reversed().percentage(), None);
    }

    #[test]
    fn results_for_sides() {
        let white_won = Outcome::win(Side::First);
        assert_eq!(GameResult::for_side(white_won, Side::First), Win);
        assert_eq!(GameResult::for_side(white_won, Side::Second), Loss);
        assert_eq!(GameResult::for_side(Outcome::Draw, Side::Second), Draw);
    }

    #[test]
    fn elo_is_zero_at_par_and_antisymmetric() {
        assert!(elo_difference(50.0).abs() < 1e-9);
        assert!((elo_difference(75.0) + elo_difference(25.0)).abs() < 1e-9);
        assert!((elo_difference(75.0) - 190.848).abs() < 0.01);
        assert!(elo_difference(100.0).is_finite() && elo_difference(0.0).is_finite());
    }

    fn result() -> impl Strategy<Value = GameResult> {
        prop_oneof![Just(Win), Just(Draw), Just(Loss)]
    }

    proptest! {
        #[test]
        fn a_pairing_always_sums_to_one_hundred(
            games in proptest::collection::vec((result(), result()), 1..60)
        ) {
            let mut a = Tally::default();
            let mut b = Tally::default();
            for (x, y) in &games {
                a.add(*x, *y);
                b.add(x.reversed(), y.reversed());
            }
            prop_assert_eq!(a.reversed(), b);
            match (a.percentage(), b.percentage()) {
                (Some(pa), Some(pb)) => prop_assert!((pa + pb - 100.0).abs() < 1e-9),
                (None, None) => prop_assert_eq!(a.counted, 0),
                other => prop_assert!(false, "only one side has a percentage: {:?}", other),
            }
            let p = a.percentage().unwrap_or(50.0);
            prop_assert!((0.0..=100.0).contains(&p));
        }

        #[test]
        fn merging_is_the_same_as_adding(
            xs in proptest::collection::vec((result(), result()), 0..30),
            ys in proptest::collection::vec((result(), result()), 0..30),
        ) {
            let (mut a, mut b, mut all) = (Tally::default(), Tally::default(), Tally::default());
            for (x, y) in &xs { a.add(*x, *y); all.add(*x, *y); }
            for (x, y) in &ys { b.add(*x, *y); all.add(*x, *y); }
            a.merge(&b);
            prop_assert_eq!(a.counted, all.counted);
            prop_assert_eq!(a.excluded, all.excluded);
            prop_assert!((a.score - all.score).abs() < 1e-9);
        }
    }
}
