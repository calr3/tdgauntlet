use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Which of the two players. Serialized as `0` or `1`; for Epaminondas 0 is white.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    First,
    Second,
}

impl Side {
    pub fn index(self) -> usize {
        match self {
            Side::First => 0,
            Side::Second => 1,
        }
    }

    pub fn other(self) -> Side {
        match self {
            Side::First => Side::Second,
            Side::Second => Side::First,
        }
    }

    pub fn from_index(i: usize) -> Option<Side> {
        match i {
            0 => Some(Side::First),
            1 => Some(Side::Second),
            _ => None,
        }
    }
}

impl Serialize for Side {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u8(self.index() as u8)
    }
}

impl<'de> Deserialize<'de> for Side {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = u8::deserialize(d)?;
        Side::from_index(v as usize)
            .ok_or_else(|| serde::de::Error::custom(format!("side must be 0 or 1, got {v}")))
    }
}

/// How a game finished. Modelled with a draw even for games that cannot produce one, so the
/// tournament and scoring layers stay game-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "result")]
pub enum Outcome {
    Win { side: Side },
    Draw,
}

impl Outcome {
    pub fn win(side: Side) -> Outcome {
        Outcome::Win { side }
    }

    /// The outcome as seen by `side`: `1.0` win, `0.5` draw, `0.0` loss.
    pub fn score_for(self, side: Side) -> f64 {
        match self {
            Outcome::Draw => 0.5,
            Outcome::Win { side: w } if w == side => 1.0,
            Outcome::Win { .. } => 0.0,
        }
    }

    /// The same outcome with the two sides exchanged.
    pub fn flipped(self) -> Outcome {
        match self {
            Outcome::Draw => Outcome::Draw,
            Outcome::Win { side } => Outcome::Win { side: side.other() },
        }
    }
}
