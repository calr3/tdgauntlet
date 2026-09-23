//! A transposition table shared by every search running at once.
//!
//! Lockless in Hyatt's style: each slot stores `key ^ data` next to `data`, so a slot torn by two
//! threads writing at once fails the key check on read and is treated as a miss, never as a wrong
//! hit. Positions are the key, which is what makes sharing it across concurrent requests from
//! unrelated games legitimate under the protocol.

use std::sync::atomic::{AtomicU64, Ordering};
use tdg_games::epaminondas::{Move, Position, MAX_QUIET_MOVES, N};
use tdg_games::Side;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub score: i32,
    pub depth: u8,
    pub bound: Bound,
    pub mv: Option<Move>,
}

impl Entry {
    fn pack(self) -> u64 {
        let bound = match self.bound {
            Bound::Exact => 0,
            Bound::Lower => 1,
            Bound::Upper => 2,
        };
        let mv = self.mv.map_or(0xFF_FFFF, |m| {
            (m.lead as u64) | (m.rear as u64) << 8 | (m.dest as u64) << 16
        });
        (self.score as u32 as u64) | (self.depth.min(63) as u64) << 32 | bound << 38 | mv << 40
    }

    fn unpack(data: u64) -> Entry {
        let mv = data >> 40;
        Entry {
            score: data as u32 as i32,
            depth: ((data >> 32) & 63) as u8,
            bound: match (data >> 38) & 3 {
                0 => Bound::Exact,
                1 => Bound::Lower,
                _ => Bound::Upper,
            },
            mv: (mv != 0xFF_FFFF).then(|| {
                Move::new(
                    (mv & 0xFF) as usize,
                    (mv >> 8 & 0xFF) as usize,
                    (mv >> 16 & 0xFF) as usize,
                )
            }),
        }
    }
}

pub struct Table {
    slots: Vec<[AtomicU64; 2]>,
    mask: usize,
}

impl Table {
    /// A table of about `mb` megabytes, rounded down to a power of two entries.
    pub fn new(mb: usize) -> Self {
        let wanted = (mb.max(1) << 20) / 16;
        let n = 1usize << (usize::BITS - 1 - wanted.leading_zeros());
        Self {
            slots: (0..n)
                .map(|_| [AtomicU64::new(0), AtomicU64::new(0)])
                .collect(),
            mask: n - 1,
        }
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let slot = &self.slots[key as usize & self.mask];
        let (check, data) = (
            slot[0].load(Ordering::Relaxed),
            slot[1].load(Ordering::Relaxed),
        );
        (data != 0 && check ^ data == key).then(|| Entry::unpack(data))
    }

    pub fn store(&self, key: u64, entry: Entry) {
        let slot = &self.slots[key as usize & self.mask];
        // Keep a deeper result for a different position rather than overwrite it with a shallow
        // one; always replace the same position.
        let (check, data) = (
            slot[0].load(Ordering::Relaxed),
            slot[1].load(Ordering::Relaxed),
        );
        if data != 0 && check ^ data != key && Entry::unpack(data).depth > entry.depth + 2 {
            return;
        }
        let data = entry.pack();
        slot[0].store(key ^ data, Ordering::Relaxed);
        slot[1].store(data, Ordering::Relaxed);
    }
}

/// Zobrist keys, built from a fixed seed so every process agrees.
struct Keys {
    piece: [[u64; N]; 2],
    black_to_move: u64,
    /// The quiet-move clock only changes a position's value close to its end, so it joins the
    /// key only then: far from it, the same board at different counts shares an entry.
    clock: [u64; 32],
    /// The clock at its exact value, for [`exact_key`].
    clock_exact: [u64; MAX_QUIET_MOVES as usize + 1],
    /// Who captured last, by side, for [`exact_key`].
    last_capturer: [u64; 2],
    /// Separates [`exact_key`]'s entries from [`key`]'s in a shared table.
    solver: u64,
}

pub(crate) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

static KEYS: std::sync::LazyLock<Keys> = std::sync::LazyLock::new(|| {
    let mut s = 0x7D6_A5E5_u64;
    let mut piece = [[0; N]; 2];
    for side in &mut piece {
        for k in side.iter_mut() {
            *k = splitmix(&mut s);
        }
    }
    let black_to_move = splitmix(&mut s);
    let mut clock = [0; 32];
    for k in &mut clock {
        *k = splitmix(&mut s);
    }
    let mut clock_exact = [0; MAX_QUIET_MOVES as usize + 1];
    for k in &mut clock_exact {
        *k = splitmix(&mut s);
    }
    let last_capturer = [splitmix(&mut s), splitmix(&mut s)];
    let solver = splitmix(&mut s);
    Keys {
        piece,
        black_to_move,
        clock,
        clock_exact,
        last_capturer,
        solver,
    }
});

/// Moves before the clock runs out at which the clock starts to matter to the key.
pub const CLOCK_HORIZON: u32 = 31;

/// The board and the side to move: everything both keys below have in common.
fn board_key(pos: &Position) -> u64 {
    let k = &*KEYS;
    let mut h = 0;
    for (s, &c) in pos.board.iter().enumerate() {
        if c != 0 {
            h ^= k.piece[c as usize - 1][s];
        }
    }
    if pos.to_move == Side::Second {
        h ^= k.black_to_move;
    }
    h
}

pub fn key(pos: &Position) -> u64 {
    let left = MAX_QUIET_MOVES.saturating_sub(pos.quiet_moves);
    let clock = if left <= CLOCK_HORIZON {
        KEYS.clock[left as usize]
    } else {
        0
    };
    board_key(pos) ^ clock
}

/// The key for a search whose only evaluation is the game's own result: [`crate::solve`].
///
/// It separates everything the outcome depends on, which [`key`] does not have to: the clock at
/// its exact value, because how much of it is left decides the game by advancement, and who
/// captured last, because that settles the tiebreak when advancement is level. `solve` proves
/// wins and losses and then keeps them in the table at any depth, so a key that merged two
/// positions the rules decide differently would not cost it a slightly wrong score — it would
/// spread a wrong proof. The tag keeps these entries from ever reading as `key`'s, so one table
/// can serve both searches.
pub fn exact_key(pos: &Position) -> u64 {
    let k = &*KEYS;
    let capturer = pos.last_capturer.map_or(0, |s| k.last_capturer[s.index()]);
    board_key(pos)
        ^ k.clock_exact[pos.quiet_moves.min(MAX_QUIET_MOVES) as usize]
        ^ capturer
        ^ k.solver
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip() {
        for e in [
            Entry {
                score: -1_000_000_000,
                depth: 63,
                bound: Bound::Upper,
                mv: Some(Move::new(167, 0, 3)),
            },
            Entry {
                score: 42,
                depth: 0,
                bound: Bound::Exact,
                mv: None,
            },
            Entry {
                score: 7,
                depth: 12,
                bound: Bound::Lower,
                mv: Some(Move::new(0, 0, 1)),
            },
        ] {
            assert_eq!(Entry::unpack(e.pack()), e);
        }
    }

    #[test]
    fn a_stored_entry_is_found_and_a_different_key_misses() {
        let t = Table::new(1);
        let e = Entry {
            score: 5,
            depth: 3,
            bound: Bound::Exact,
            mv: None,
        };
        t.store(12345, e);
        assert_eq!(t.probe(12345), Some(e));
        assert_eq!(
            t.probe(12345 + t.slots.len() as u64),
            None,
            "same slot, different key"
        );
    }

    #[test]
    fn the_clock_enters_the_key_only_near_its_end() {
        let p = Position::start();
        let at = |q: u32| {
            key(&Position {
                quiet_moves: q,
                ..p.clone()
            })
        };
        assert_eq!(at(0), at(40), "far from the end the count is ignored");
        assert_ne!(at(90), at(91), "near it, every move counts");
        assert_ne!(at(0), at(99));
        let black = Position {
            to_move: tdg_games::Side::Second,
            ..p.clone()
        };
        assert_ne!(key(&p), key(&black));
    }

    #[test]
    fn the_exact_key_separates_everything_the_outcome_depends_on() {
        let p = Position::start();
        let with = |q: u32, c: Option<Side>| {
            exact_key(&Position {
                quiet_moves: q,
                last_capturer: c,
                ..p.clone()
            })
        };
        assert_ne!(with(0, None), with(40, None), "every count is its own key");
        assert_ne!(with(40, None), with(41, None));
        assert_ne!(with(0, None), with(0, Some(Side::First)), "the tiebreak");
        assert_ne!(with(0, Some(Side::First)), with(0, Some(Side::Second)));
        assert_ne!(exact_key(&p), key(&p), "tagged apart in a shared table");
    }
}
