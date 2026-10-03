//! A native port of the Gess web app's negamax engine (`~/calr3gh/gess/wasm-engine/src/lib.rs`,
//! which runs in the browser as WebAssembly).
//!
//! The rules, move generation, make/unmake, transposition table and search (iterative deepening,
//! alpha-beta, a lazy TT-on path with killers, null move and late-move reductions, and an eager
//! TT-off path kept as a self-check) are the app's, line for line. What changed:
//!
//! - The browser parts are gone: the host imports (`now()` is `Instant`; `random()`, used only to
//!   break ties between equally good root moves, is a seeded generator or "take the first"), the
//!   shared-memory I/O buffers, and the Lazy-SMP helper threads. One search runs on one thread;
//!   tuning gets its parallelism from playing many games at once.
//! - The app keeps its transposition table, move scratch and generation counter in statics. Here
//!   they belong to a [`Searcher`], so many engines with different weights can play in one
//!   process.
//! - The evaluation's constants are [`Weights`], applied to [`features`]. With the default weights
//!   it is exactly the app's evaluation; `tests/identity.rs` checks the port against the app's
//!   `.wasm` move for move and score for score.
//! - The full-evaluation move generator scores each child with the same evaluation function
//!   rather than the app's incremental per-row update of it (the same value, so a weight change
//!   reaches both places).
//! - Searches can also stop on a node budget, not just a clock.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Instant;

// ─── Constants (as the app) ──────────────────────────────────────────────────
pub const SIZE: i32 = 20;
const MIN: i32 = 1;
const MAX: i32 = 18;
// A ring centre can only sit in rows 2..17 (its neighbours would otherwise be in the empty
// border rows), so ring scans skip rows 1 and 18.
const RING_LO: i32 = MIN + 1;
const RING_HI: i32 = MAX - 1;
pub const WIN_SCORE: i32 = 1_000_000;
pub const MAX_DEPTH: i32 = 64;
const PLAY_MASK: u32 = 0x7FFFE; // columns 1..18
const NEG_INF: i32 = -2_000_000_000;
/// After this many consecutive moves without a capture the game is decided on stones.
pub const QUIET_LIMIT: i32 = 20;

// Move outcomes from the mover's view, in the order moves are tried.
const OUT_WIN: i32 = 2;
const OUT_DRAW: i32 = 1;
const OUT_CONTINUE: i32 = 0;
const OUT_LOSS: i32 = -1;

const ARENA_MOVES: usize = 1 << 15;
const TT_BITS: usize = 19;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = (TT_SIZE as u64) - 1;
const TF_EXACT: u8 = 0;
const TF_LOWER: u8 = 1;
const TF_UPPER: u8 = 2;
const KILLER_PLIES: usize = (MAX_DEPTH + 2) as usize;
const UNDO_CAP: usize = MAX_DEPTH as usize + 8;
// Check the clock / node budget once every this many nodes, as the app does.
const CLOCK_CHECK_MASK: u32 = 0xFF;

#[inline]
fn terminal_score(outcome: i32, ply: i32) -> Option<i32> {
    match outcome {
        OUT_WIN => Some(WIN_SCORE - (ply + 1)),
        OUT_LOSS => Some(-(WIN_SCORE - (ply + 1))),
        OUT_DRAW => Some(0),
        _ => None,
    }
}

#[inline]
fn stone_outcome(mine: i32, theirs: i32) -> i32 {
    if mine > theirs {
        OUT_WIN
    } else if mine < theirs {
        OUT_LOSS
    } else {
        OUT_DRAW
    }
}

// [dr, dc] for the 8 footprint positions; this order with the stable sort pins the move order.
const DIRS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

// COL3[c] = mask of columns {c-1, c, c+1} clamped to [0, 19].
const COL3: [u32; 20] = {
    let mut t = [0u32; 20];
    let mut c = 0usize;
    while c < 20 {
        let mut m = 1u32 << c;
        if c > 0 {
            m |= 1u32 << (c - 1);
        }
        if c < 19 {
            m |= 1u32 << (c + 1);
        }
        t[c] = m;
        c += 1;
    }
    t
};

// ─── Evaluation ──────────────────────────────────────────────────────────────

/// The evaluation's terms, from the side to move's point of view (see [`features`]). The first
/// five are the app's evaluation; the rest are candidates, off (weight 0) by default.
pub const NAMES: [&str; NUM_FEATURES] = [
    "stone",
    "ring",
    "ring_pair",
    "deep",
    "pressure",
    "threat",
    "defenders",
    "mobile",
    "edge",
    "quiet_stone",
];
pub const NUM_FEATURES: usize = 10;
/// The app's terms: the rest are only computed when one of their weights is non-zero.
const NUM_BASE: usize = 5;

/// Evaluation weights: `eval = Σ weight · feature`, in hundredths of a stone. The defaults are the
/// app's evaluation exactly: 100 per stone; 700 per ring (the app counts rings in tenths at 70 per
/// tenth) less 210 for each ring with another ring two files to its left on the same row (the
/// app's 3-tenths discount); 1 per row of advancement of the deepest stone; 10 per stone of ring
/// pressure (opposing stones in the 5x5 box around each ring).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Weights {
    pub stone: i32,
    pub ring: i32,
    pub ring_pair: i32,
    pub deep: i32,
    pub pressure: i32,
    pub threat: i32,
    pub defenders: i32,
    pub mobile: i32,
    pub edge: i32,
    pub quiet_stone: i32,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            stone: 100,
            ring: 700,
            ring_pair: -210,
            deep: 1,
            pressure: 10,
            threat: 0,
            defenders: 0,
            mobile: 0,
            edge: 0,
            quiet_stone: 0,
        }
    }
}

impl Weights {
    pub fn as_array(&self) -> [i32; NUM_FEATURES] {
        [
            self.stone,
            self.ring,
            self.ring_pair,
            self.deep,
            self.pressure,
            self.threat,
            self.defenders,
            self.mobile,
            self.edge,
            self.quiet_stone,
        ]
    }

    pub fn from_array(a: [i32; NUM_FEATURES]) -> Self {
        Weights {
            stone: a[0],
            ring: a[1],
            ring_pair: a[2],
            deep: a[3],
            pressure: a[4],
            threat: a[5],
            defenders: a[6],
            mobile: a[7],
            edge: a[8],
            quiet_stone: a[9],
        }
    }

    /// Whether any candidate term beyond the app's five is switched on.
    fn extended(&self) -> bool {
        self.as_array()[NUM_BASE..].iter().any(|&x| x != 0)
    }

    /// Weights from a JSON object of name -> number; missing names keep the defaults.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[inline]
fn row_ring_mask_w(w: &[u32; 20], b: &[u32; 20], r: usize) -> u32 {
    let wu = w[r - 1];
    let wm = w[r];
    let wd = w[r + 1];
    let empty = !(w[r] | b[r]);
    (wu & (wu << 1) & (wu >> 1))
        & ((wm << 1) & (wm >> 1))
        & (wd & (wd << 1) & (wd >> 1))
        & empty
        & PLAY_MASK
}

#[inline]
fn row_ring_mask_b(w: &[u32; 20], b: &[u32; 20], r: usize) -> u32 {
    let bu = b[r - 1];
    let bm = b[r];
    let bd = b[r + 1];
    let empty = !(w[r] | b[r]);
    (bu & (bu << 1) & (bu >> 1))
        & ((bm << 1) & (bm >> 1))
        & (bd & (bd << 1) & (bd >> 1))
        & empty
        & PLAY_MASK
}

fn white_deepest(w: &[u32; 20]) -> i32 {
    let mut r = MAX;
    while r >= MIN {
        if w[r as usize] & PLAY_MASK != 0 {
            return r - 1;
        }
        r -= 1;
    }
    0
}

fn black_deepest(b: &[u32; 20]) -> i32 {
    let mut r = MIN;
    while r <= MAX {
        if b[r as usize] & PLAY_MASK != 0 {
            return MAX - r;
        }
        r += 1;
    }
    0
}

/// The evaluation's terms for a board, from `white`'s point of view if `white`, else black's,
/// with `quiet` the moves since the last capture:
///
/// - `stone`: stone difference;
/// - `ring`: ring difference;
/// - `ring_pair`: difference in rings with another ring two files left on their row;
/// - `deep`: difference in the advancement of each side's deepest stone;
/// - `pressure`: own stones in the 5x5 boxes round the opponent's rings minus the opponent's
///   round one's own (the app's ring safety);
/// - `threat`: the opponent's stones round one's own rings (with `pressure`, this lets attack
///   and defence weigh differently);
/// - `defenders`: own stones round one's own rings, minus the opponent's round theirs;
/// - `mobile`: own stones that centre a piece (an own stone beside them, no opponent stone in
///   the 3x3), which can slide any distance, minus the opponent's;
/// - `edge`: own stones on the outermost playing rows and columns, minus the opponent's;
/// - `quiet_stone`: stone difference times the moves since the last capture (the game is
///   decided on stones at 20).
///
/// The candidates (from `threat` on) are computed only when `extended`; otherwise they are 0.
pub fn features_with(
    w: &[u32; 20],
    b: &[u32; 20],
    white: bool,
    quiet: i32,
    extended: bool,
) -> [i32; NUM_FEATURES] {
    let (mut wn, mut bn, mut wp, mut bp, mut ws, mut bs) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut pw, mut pb) = (0i32, 0i32); // opposing stones round white's / black's rings
    let (mut dw, mut db) = (0i32, 0i32); // own stones round white's / black's rings
    let mut r = MIN;
    while r <= MAX {
        let ri = r as usize;
        let mw = row_ring_mask_w(w, b, ri);
        let mb = row_ring_mask_b(w, b, ri);
        wn += mw.count_ones() as i32;
        bn += mb.count_ones() as i32;
        wp += (mw & (mw << 2)).count_ones() as i32;
        bp += (mb & (mb << 2)).count_ones() as i32;
        ws += (w[ri] & PLAY_MASK).count_ones() as i32;
        bs += (b[ri] & PLAY_MASK).count_ones() as i32;
        if mw != 0 {
            let band = mw | (mw << 1) | (mw >> 1) | (mw << 2) | (mw >> 2);
            let mut rr = r - 2;
            while rr <= r + 2 {
                if (0..SIZE).contains(&rr) {
                    pw += (b[rr as usize] & band).count_ones() as i32;
                    if extended {
                        dw += (w[rr as usize] & band).count_ones() as i32;
                    }
                }
                rr += 1;
            }
        }
        if mb != 0 {
            let band = mb | (mb << 1) | (mb >> 1) | (mb << 2) | (mb >> 2);
            let mut rr = r - 2;
            while rr <= r + 2 {
                if (0..SIZE).contains(&rr) {
                    pb += (w[rr as usize] & band).count_ones() as i32;
                    if extended {
                        db += (b[rr as usize] & band).count_ones() as i32;
                    }
                }
                rr += 1;
            }
        }
        r += 1;
    }
    let (wd, bd) = (white_deepest(w), black_deepest(b));
    let mut f = [ws - bs, wn - bn, wp - bp, wd - bd, pb - pw, 0, 0, 0, 0, 0];
    if extended {
        let mobile = |own: &[u32; 20], opp: &[u32; 20]| -> i32 {
            let mut n = 0;
            for r in MIN..=MAX {
                let (u, m, d) = (
                    own[(r - 1) as usize],
                    own[r as usize],
                    own[(r + 1) as usize],
                );
                let beside =
                    u | d | (u << 1) | (u >> 1) | (d << 1) | (d >> 1) | (m << 1) | (m >> 1);
                let o = opp[(r - 1) as usize] | opp[r as usize] | opp[(r + 1) as usize];
                let blocked = o | (o << 1) | (o >> 1);
                n += (m & beside & !blocked & PLAY_MASK).count_ones() as i32;
            }
            n
        };
        let edge = |own: &[u32; 20]| -> i32 {
            let sides = (1u32 << MIN) | (1u32 << MAX);
            let mut n = (own[MIN as usize] & PLAY_MASK).count_ones() as i32
                + (own[MAX as usize] & PLAY_MASK).count_ones() as i32;
            for r in MIN + 1..MAX {
                n += (own[r as usize] & sides).count_ones() as i32;
            }
            n
        };
        f[5] = -pw; // white's view: black stones round white's rings, as a cost
        f[6] = dw - db;
        f[7] = mobile(w, b) - mobile(b, w);
        f[8] = edge(w) - edge(b);
        f[9] = (ws - bs) * quiet.clamp(0, QUIET_LIMIT);
        if !white {
            f[5] = -pb; // black's view: white stones round black's rings
            f[6] = -f[6];
            f[7] = -f[7];
            f[8] = -f[8];
            f[9] = -f[9];
            for x in f.iter_mut().take(NUM_BASE) {
                *x = -*x;
            }
            return f;
        }
        return f;
    }
    if white {
        f
    } else {
        f.map(|x| -x)
    }
}

/// All the evaluation's terms (candidates included) for `white`'s side.
pub fn features(w: &[u32; 20], b: &[u32; 20], white: bool, quiet: i32) -> [i32; NUM_FEATURES] {
    features_with(w, b, white, quiet, true)
}

/// The static evaluation: Σ weight · feature.
#[inline]
pub fn evaluate(w: &[u32; 20], b: &[u32; 20], white: bool, quiet: i32, weights: &Weights) -> i32 {
    let f = features_with(w, b, white, quiet, weights.extended());
    let k = weights.as_array();
    f.iter().zip(k.iter()).map(|(a, b)| a * b).sum()
}

// ─── Zobrist keys (as the app's: same seed, same order) ──────────────────────
struct Zobrist {
    w: [u64; 400],
    b: [u64; 400],
    side: u64,
    quiet: [u64; (QUIET_LIMIT + 1) as usize],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn zobrist() -> &'static Zobrist {
    static Z: OnceLock<Zobrist> = OnceLock::new();
    Z.get_or_init(|| {
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let mut z = Zobrist {
            w: [0; 400],
            b: [0; 400],
            side: 0,
            quiet: [0; (QUIET_LIMIT + 1) as usize],
        };
        for i in 0..400 {
            z.w[i] = splitmix64(&mut s);
        }
        for i in 0..400 {
            z.b[i] = splitmix64(&mut s);
        }
        z.side = splitmix64(&mut s);
        for i in 0..=QUIET_LIMIT as usize {
            z.quiet[i] = splitmix64(&mut s);
        }
        z
    })
}

#[inline]
fn zob_cell(white: bool, idx: usize) -> u64 {
    let z = zobrist();
    if white {
        z.w[idx]
    } else {
        z.b[idx]
    }
}

#[inline]
fn zob_quiet(quiet: i32) -> u64 {
    zobrist().quiet[quiet.clamp(0, QUIET_LIMIT) as usize]
}

fn compute_hash(w: &[u32; 20], b: &[u32; 20], white: bool) -> u64 {
    let z = zobrist();
    let mut h = 0u64;
    for r in 0..20usize {
        let mut m = w[r];
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            h ^= z.w[r * 20 + c];
            m &= m - 1;
        }
        let mut m = b[r];
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            h ^= z.b[r * 20 + c];
            m &= m - 1;
        }
    }
    if white {
        h ^= z.side;
    }
    h
}

// ─── Transposition table (one per Searcher; generation-invalidated per search) ──
struct Tt {
    data: Vec<u64>,
    lock: Vec<u64>,
    mv: Vec<u32>,
    gen: u32,
}

struct TtHit {
    val: i32,
    flag: u8,
    depth: i32,
    mv: u32,
}

#[inline]
fn pack_entry(val: i32, depth: i32, flag: u8, gen: u32) -> u64 {
    (val as u32 as u64)
        | (((depth as u64) & 0xFF) << 32)
        | (((flag as u64) & 0x3) << 40)
        | (((gen as u64) & 0xFFFF) << 42)
}

impl Tt {
    fn new() -> Self {
        Tt {
            data: vec![0; TT_SIZE],
            lock: vec![0; TT_SIZE],
            mv: vec![0; TT_SIZE],
            gen: 0,
        }
    }

    /// Start a new search: a fresh generation makes every older entry read as empty.
    fn new_search(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if (self.gen & 0xFFFF) == 0 {
            self.data.iter_mut().for_each(|x| *x = 0);
            self.lock.iter_mut().for_each(|x| *x = 0);
            self.mv.iter_mut().for_each(|x| *x = 0);
            self.gen = 1;
        }
    }

    #[inline]
    fn probe(&self, key: u64) -> Option<TtHit> {
        let idx = (key & TT_MASK) as usize;
        let data = self.data[idx];
        if (self.lock[idx] ^ data) != key {
            return None;
        }
        if ((data >> 42) & 0xFFFF) as u32 != (self.gen & 0xFFFF) {
            return None;
        }
        Some(TtHit {
            val: data as u32 as i32,
            flag: ((data >> 40) & 0x3) as u8,
            depth: ((data >> 32) & 0xFF) as i32,
            mv: self.mv[idx],
        })
    }

    #[inline]
    fn store(&mut self, key: u64, depth: i32, val: i32, flag: u8, mv: u32) {
        let idx = (key & TT_MASK) as usize;
        let data = pack_entry(val, depth, flag, self.gen);
        self.data[idx] = data;
        self.lock[idx] = key ^ data;
        self.mv[idx] = mv;
    }
}

// ─── Moves ───────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct EMove {
    from_r: i32,
    from_c: i32,
    to_r: i32,
    to_c: i32,
    outcome: i32,
    key: i32,
}

const NO_MOVE: EMove = EMove {
    from_r: 0,
    from_c: 0,
    to_r: 0,
    to_c: 0,
    outcome: 0,
    key: 0,
};

#[derive(Clone, Copy)]
struct UndoFrame {
    r: [usize; 6],
    w: [u32; 6],
    b: [u32; 6],
    n: usize,
    quiet: i32,
}

#[inline]
fn move_before(a: &EMove, b: &EMove) -> bool {
    a.outcome > b.outcome || (a.outcome == b.outcome && a.key > b.key)
}

#[inline]
fn pack_mv(m: &EMove) -> u32 {
    (m.from_r as u32) | ((m.from_c as u32) << 5) | ((m.to_r as u32) << 10) | ((m.to_c as u32) << 15)
}

#[inline]
fn has_ring_w(w: &[u32; 20], b: &[u32; 20]) -> bool {
    (RING_LO..=RING_HI).any(|r| row_ring_mask_w(w, b, r as usize) != 0)
}

#[inline]
fn has_ring_b(w: &[u32; 20], b: &[u32; 20]) -> bool {
    (RING_LO..=RING_HI)
        .rev()
        .any(|r| row_ring_mask_b(w, b, r as usize) != 0)
}

/// Move `target` to slot `front` of `moves[..count]` if present at or after it.
#[inline]
fn promote(moves: &mut [EMove], front: usize, target: u32) -> usize {
    if target == 0 {
        return front;
    }
    for i in front..moves.len() {
        if pack_mv(&moves[i]) == target {
            moves.swap(front, i);
            return front + 1;
        }
    }
    front
}

/// The app's stable bottom-up merge sort: outcome desc, then key desc, ties in insertion order.
fn stable_sort_moves(a: &mut [EMove], tmp: &mut [EMove]) {
    // A stable sort by the same order gives the same result as the app's merge sort.
    let n = a.len();
    if n < 2 {
        return;
    }
    let tmp = &mut tmp[..n];
    tmp.copy_from_slice(a);
    tmp.sort_by(|x, y| {
        if move_before(y, x) {
            std::cmp::Ordering::Greater
        } else if move_before(x, y) {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    });
    a.copy_from_slice(tmp);
}

// ─── Board and positions ─────────────────────────────────────────────────────

/// A position as the engine sees it: a bitboard row per side (bit c of row r = cell r*20+c), the
/// side to move, and the moves since the last capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    pub w: [u32; 20],
    pub b: [u32; 20],
    pub white_to_move: bool,
    pub quiet: i32,
}

impl Board {
    /// From tdgauntlet's Gess position (`tdg_games::gess`: black = 1, white = 2; black, side 0,
    /// moves first).
    pub fn from_position(p: &tdg_games::gess::Position) -> Self {
        let mut w = [0u32; 20];
        let mut b = [0u32; 20];
        for (i, &v) in p.board.iter().enumerate() {
            let (r, c) = (i / 20, i % 20);
            if v == tdg_games::gess::WHITE {
                w[r] |= 1 << c;
            } else if v == tdg_games::gess::BLACK {
                b[r] |= 1 << c;
            }
        }
        Board {
            w,
            b,
            white_to_move: p.to_move == tdg_games::Side::Second,
            quiet: (p.quiet_moves as i32).clamp(0, QUIET_LIMIT),
        }
    }

    /// The static evaluation for the side to move.
    pub fn evaluate(&self, weights: &Weights) -> i32 {
        evaluate(&self.w, &self.b, self.white_to_move, self.quiet, weights)
    }

    /// The evaluation's terms for the side to move.
    pub fn features(&self) -> [i32; NUM_FEATURES] {
        features(&self.w, &self.b, self.white_to_move, self.quiet)
    }
}

/// How long a search may run. Unlimited fields are `None`; `depth` caps the iterations.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub depth: i32,
    pub deadline: Option<Instant>,
    pub nodes: Option<u64>,
}

impl Limits {
    pub fn depth(depth: i32) -> Self {
        Limits {
            depth,
            deadline: None,
            nodes: None,
        }
    }
}

/// A search's choice: the move (source and destination cells), its score at the deepest
/// completed iteration, that depth, and the nodes searched.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub from: usize,
    pub to: usize,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

/// Search switches, all on by default as in the app (null move and LMR are off only for the
/// exact self-checks; `force_eager` routes a TT search through the eager path).
#[derive(Debug, Clone, Copy)]
pub struct Switches {
    pub null_move: bool,
    pub lmr: bool,
    pub force_eager: bool,
}

impl Default for Switches {
    fn default() -> Self {
        Switches {
            null_move: true,
            lmr: true,
            force_eager: false,
        }
    }
}

/// Everything a search needs that outlives one search: the transposition table, the move scratch
/// space, the weights, the switches and the tie-break generator.
pub struct Searcher {
    tt: Tt,
    arena: Vec<EMove>,
    pub weights: Weights,
    pub switches: Switches,
    /// `None`: of the equally good root moves, play the first (deterministic). `Some(seed)`: one
    /// at random, as the app does with `Math.random()`.
    rng: Option<u64>,
}

impl Searcher {
    pub fn new(weights: Weights) -> Self {
        Searcher {
            tt: Tt::new(),
            arena: vec![NO_MOVE; ARENA_MOVES],
            weights,
            switches: Switches::default(),
            rng: None,
        }
    }

    /// Break ties between equally good root moves at random, from this seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Some(seed);
        self
    }

    /// The legal moves (source cell, destination cell) with their outcomes (2 win, 1 draw,
    /// 0 play on, -1 loss) and static keys, best first - the app's `gen_root`.
    pub fn root_moves(&mut self, board: &Board) -> Vec<(usize, usize, i32, i32)> {
        let Searcher {
            tt,
            arena,
            weights,
            switches,
            ..
        } = self;
        let mut eng = Engine::new(tt, arena, *weights, *switches, board, false);
        collect_root(&mut eng, board.white_to_move)
            .iter()
            .map(|m| {
                (
                    cell(m.from_r, m.from_c),
                    cell(m.to_r, m.to_c),
                    m.outcome,
                    m.key,
                )
            })
            .collect()
    }

    /// The app's `search_root`: phase-1 alpha-beta at a fixed depth (no tie-break, no root
    /// reordering), returning the best score and every root move's score in generated order.
    /// With the TT off this is the exact, deterministic self-check path.
    pub fn search_root(
        &mut self,
        board: &Board,
        depth: i32,
        tt_on: bool,
    ) -> (i32, Vec<(usize, usize, i32)>) {
        let white = board.white_to_move;
        let Searcher {
            tt,
            arena,
            weights,
            switches,
            ..
        } = self;
        if tt_on {
            tt.new_search();
        }
        let mut eng = Engine::new(tt, arena, *weights, *switches, board, tt_on);
        let root = collect_root(&mut eng, white);
        let mut alpha = NEG_INF;
        let mut out = Vec::with_capacity(root.len());
        for ch in &root {
            let score = if let Some(t) = terminal_score(ch.outcome, 0) {
                t
            } else if depth == 1 {
                ch.key
            } else {
                let h = eng.make_move(ch.from_r, ch.from_c, ch.to_r, ch.to_c, white, tt_on);
                eng.hash ^= h;
                let s = -eng.search(!white, depth - 1, NEG_INF, -alpha, 1);
                eng.unmake_move();
                eng.hash ^= h;
                s
            };
            out.push((cell(ch.from_r, ch.from_c), cell(ch.to_r, ch.to_c), score));
            alpha = alpha.max(score);
        }
        (alpha, out)
    }

    /// The app's `search_to_depth`: iterative-deepening phase-1 search to `depth` (no time
    /// limit, no tie-break), returning the best score.
    pub fn search_to_depth(&mut self, board: &Board, depth: i32, tt_on: bool) -> i32 {
        let white = board.white_to_move;
        let Searcher {
            tt,
            arena,
            weights,
            switches,
            ..
        } = self;
        if tt_on {
            tt.new_search();
        }
        let mut eng = Engine::new(tt, arena, *weights, *switches, board, tt_on);
        let root = collect_root(&mut eng, white);
        let mut final_best = NEG_INF;
        for d in 1..=depth {
            let mut alpha = NEG_INF;
            for ch in &root {
                let score = if let Some(t) = terminal_score(ch.outcome, 0) {
                    t
                } else if d == 1 {
                    ch.key
                } else {
                    let h = eng.make_move(ch.from_r, ch.from_c, ch.to_r, ch.to_c, white, tt_on);
                    eng.hash ^= h;
                    let s = -eng.search(!white, d - 1, NEG_INF, -alpha, 1);
                    eng.unmake_move();
                    eng.hash ^= h;
                    s
                };
                alpha = alpha.max(score);
            }
            final_best = alpha;
        }
        final_best
    }

    /// The app's `bench_fixed`: fixed-depth iterative deepening on the lazy path, returning the
    /// nodes searched. The same tree as the app's, so the node count must match it exactly.
    pub fn bench_fixed(&mut self, board: &Board, depth: i32) -> u64 {
        let white = board.white_to_move;
        let Searcher {
            tt,
            arena,
            weights,
            switches,
            ..
        } = self;
        tt.new_search();
        let mut eng = Engine::new(tt, arena, *weights, *switches, board, true);
        let root = collect_root(&mut eng, white);
        for d in 1..=depth {
            let mut alpha = NEG_INF;
            for ch in &root {
                if ch.outcome != OUT_CONTINUE {
                    continue;
                }
                if d == 1 {
                    alpha = alpha.max(ch.key);
                    continue;
                }
                let h = eng.make_move(ch.from_r, ch.from_c, ch.to_r, ch.to_c, white, true);
                eng.hash ^= h;
                let s = -eng.search(!white, d - 1, NEG_INF, -alpha, 1);
                eng.unmake_move();
                eng.hash ^= h;
                alpha = alpha.max(s);
            }
        }
        eng.nodes_total
    }

    /// The app's `best_move`: iterative deepening with alpha-beta, the full-window re-search of
    /// tied root moves, mate-distance scoring and a tie-break, within `limits`. `None` if there is
    /// no legal move.
    pub fn best_move(&mut self, board: &Board, limits: Limits) -> Option<SearchResult> {
        let white = board.white_to_move;
        let Searcher {
            tt,
            arena,
            weights,
            switches,
            rng,
        } = self;
        tt.new_search();
        let max_depth = limits.depth.clamp(1, MAX_DEPTH);
        let mut eng = Engine::new(tt, arena, *weights, *switches, board, true);
        eng.deadline = limits.deadline;
        eng.node_limit = limits.nodes.unwrap_or(u64::MAX);
        eng.limited = limits.deadline.is_some() || limits.nodes.is_some();
        let mut root = collect_root(&mut eng, white);
        if root.is_empty() {
            return None;
        }
        let mut best = root[0];
        let mut best_depth = 0;
        let mut last_best = 0i32;

        let mut depth = 1;
        while depth <= max_depth {
            eng.timed_out = false;
            let mut alpha = NEG_INF;
            let mut scored: Vec<(EMove, i32, bool)> = Vec::with_capacity(root.len());
            let mut aborted = false;
            for ch in &root {
                let (score, exact);
                if let Some(t) = terminal_score(ch.outcome, 0) {
                    score = t;
                    exact = true;
                } else if depth == 1 {
                    score = ch.key;
                    exact = true;
                } else {
                    let h = eng.make_move(ch.from_r, ch.from_c, ch.to_r, ch.to_c, white, true);
                    eng.hash ^= h;
                    let s = -eng.search(!white, depth - 1, NEG_INF, -alpha, 1);
                    eng.unmake_move();
                    eng.hash ^= h;
                    if eng.timed_out {
                        aborted = true;
                        break;
                    }
                    score = s;
                    exact = false;
                }
                scored.push((*ch, score, exact));
                alpha = alpha.max(score);
            }
            if aborted {
                best = choose_on_abort(best, &scored);
                break;
            }
            let best_score = alpha;
            // Phase 2: full-window re-search of the moves tied at the best score, so a loss that
            // only tied it through a loose fail-low bound drops out.
            for entry in scored.iter_mut() {
                if entry.1 != best_score || entry.2 {
                    continue;
                }
                let ch = entry.0;
                let h = eng.make_move(ch.from_r, ch.from_c, ch.to_r, ch.to_c, white, true);
                eng.hash ^= h;
                let v = -eng.search(!white, depth - 1, NEG_INF, -NEG_INF, 1);
                eng.unmake_move();
                eng.hash ^= h;
                if eng.timed_out {
                    aborted = true;
                    break;
                }
                entry.1 = v;
                entry.2 = true;
            }
            if aborted {
                best = choose_on_abort(best, &scored);
                break;
            }
            let best_score = scored.iter().map(|s| s.1).max().unwrap();
            let mut candidates: Vec<EMove> = scored
                .iter()
                .filter(|s| s.1 == best_score)
                .map(|s| s.0)
                .collect();
            if candidates.is_empty() {
                candidates.push(root[0]);
            }
            let mut idx = (random(rng) * candidates.len() as f64) as usize;
            if idx >= candidates.len() {
                idx = candidates.len() - 1;
            }
            let chosen = candidates[idx];
            best = chosen;
            best_depth = depth;
            last_best = best_score;

            // A forced win or loss will not change with depth.
            if best_score >= WIN_SCORE - MAX_DEPTH || best_score <= -(WIN_SCORE - MAX_DEPTH) {
                break;
            }
            // Best first for the next iteration; drop moves proven lost; the chosen move leads.
            let mut order: Vec<usize> = (0..root.len()).collect();
            order.sort_by(|&a, &b| scored[b].1.cmp(&scored[a].1));
            root = order
                .iter()
                .filter(|&&i| scored[i].1 > -(WIN_SCORE - MAX_DEPTH))
                .map(|&i| root[i])
                .collect();
            if let Some(pos) = root.iter().position(|m| *m == chosen) {
                if pos > 0 {
                    let c = root.remove(pos);
                    root.insert(0, c);
                }
            }
            if eng.out_of_budget() {
                break;
            }
            depth += 1;
        }
        Some(SearchResult {
            from: cell(best.from_r, best.from_c),
            to: cell(best.to_r, best.to_c),
            score: last_best,
            depth: best_depth,
            nodes: eng.nodes_total,
        })
    }
}

fn random(rng: &mut Option<u64>) -> f64 {
    match rng {
        None => 0.0,
        Some(s) => (splitmix64(s) >> 11) as f64 / (1u64 << 53) as f64,
    }
}

#[inline]
fn cell(r: i32, c: i32) -> usize {
    (r * SIZE + c) as usize
}

fn choose_on_abort(prev_best: EMove, scored: &[(EMove, i32, bool)]) -> EMove {
    let mut best: Option<(EMove, i32)> = None;
    for &(mv, s, _) in scored {
        if best.is_none_or(|(_, bs)| s > bs) {
            best = Some((mv, s));
        }
    }
    best.map_or(prev_best, |(mv, _)| mv)
}

/// Generate the root moves (full evaluation) and copy them out, freeing the scratch space.
fn collect_root(eng: &mut Engine, white: bool) -> Vec<EMove> {
    let (start, count) = eng.gen_moves(white, false);
    let v = eng.arena[start..start + count].to_vec();
    eng.arena_top = 0;
    v
}

// ─── The search engine (one search) ──────────────────────────────────────────
struct Engine<'a> {
    w: [u32; 20],
    b: [u32; 20],
    undo: [UndoFrame; UNDO_CAP],
    undo_len: usize,
    quiet: i32,
    arena_top: usize,
    hash: u64,
    wit_w: i32,
    wit_b: i32,
    /// Node counter as the app keeps it (u32, wrapping; drives the clock throttle).
    nodes: u32,
    /// Nodes searched, for reporting and the node budget.
    nodes_total: u64,
    killers: [[u32; 2]; KILLER_PLIES],
    timed_out: bool,
    tt_enabled: bool,
    /// Whether a deadline or a node budget applies (the app: a finite deadline).
    limited: bool,
    deadline: Option<Instant>,
    node_limit: u64,
    weights: Weights,
    switches: Switches,
    arena: &'a mut [EMove],
    tt: &'a mut Tt,
}

impl<'a> Engine<'a> {
    fn new(
        tt: &'a mut Tt,
        arena: &'a mut [EMove],
        weights: Weights,
        switches: Switches,
        board: &Board,
        tt_enabled: bool,
    ) -> Self {
        Engine {
            w: board.w,
            b: board.b,
            undo: [UndoFrame {
                r: [0; 6],
                w: [0; 6],
                b: [0; 6],
                n: 0,
                quiet: 0,
            }; UNDO_CAP],
            undo_len: 0,
            quiet: board.quiet.clamp(0, QUIET_LIMIT),
            arena_top: 0,
            hash: compute_hash(&board.w, &board.b, false),
            wit_w: -1,
            wit_b: -1,
            nodes: 0,
            nodes_total: 0,
            killers: [[0u32; 2]; KILLER_PLIES],
            timed_out: false,
            tt_enabled,
            limited: false,
            deadline: None,
            node_limit: u64::MAX,
            weights,
            switches,
            arena,
            tt,
        }
    }

    /// Whether the clock or the node budget has run out.
    #[inline]
    fn out_of_budget(&self) -> bool {
        self.nodes_total >= self.node_limit || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    #[inline]
    fn count_node(&mut self) {
        self.nodes = self.nodes.wrapping_add(1);
        self.nodes_total += 1;
    }

    /// has_ring through a cached witness ring centre per colour, re-validated each time.
    #[inline]
    fn has_ring_wit(&mut self, white: bool) -> bool {
        let wit = if white { self.wit_w } else { self.wit_b };
        if wit >= 0 {
            let r = (wit / 20) as usize;
            let c = (wit % 20) as u32;
            let m = if white {
                row_ring_mask_w(&self.w, &self.b, r)
            } else {
                row_ring_mask_b(&self.w, &self.b, r)
            };
            if m & (1u32 << c) != 0 {
                return true;
            }
        }
        if white {
            let mut r = RING_LO;
            while r <= RING_HI {
                let m = row_ring_mask_w(&self.w, &self.b, r as usize);
                if m != 0 {
                    self.wit_w = r * 20 + m.trailing_zeros() as i32;
                    return true;
                }
                r += 1;
            }
            self.wit_w = -1;
        } else {
            let mut r = RING_HI;
            while r >= RING_LO {
                let m = row_ring_mask_b(&self.w, &self.b, r as usize);
                if m != 0 {
                    self.wit_b = r * 20 + m.trailing_zeros() as i32;
                    return true;
                }
                r -= 1;
            }
            self.wit_b = -1;
        }
        false
    }

    #[inline]
    fn eval_leaf(&self, white: bool) -> i32 {
        evaluate(&self.w, &self.b, white, self.quiet, &self.weights)
    }

    /// Apply a move; when `track`, return the Zobrist delta of the stones that changed.
    fn make_move(
        &mut self,
        from_r: i32,
        from_c: i32,
        to_r: i32,
        to_c: i32,
        white: bool,
        track: bool,
    ) -> u64 {
        let mut delta = 0u64;
        let mut frame = UndoFrame {
            r: [0; 6],
            w: [0; 6],
            b: [0; 6],
            n: 0,
            quiet: self.quiet,
        };
        let mut n = 0usize;
        let mut r = from_r - 1;
        while r <= from_r + 1 {
            if (0..SIZE).contains(&r) {
                let ri = r as usize;
                frame.r[n] = ri;
                frame.w[n] = self.w[ri];
                frame.b[n] = self.b[ri];
                n += 1;
            }
            r += 1;
        }
        let mut r = to_r - 1;
        while r <= to_r + 1 {
            if (0..SIZE).contains(&r) && !(r >= from_r - 1 && r <= from_r + 1) {
                let ri = r as usize;
                frame.r[n] = ri;
                frame.w[n] = self.w[ri];
                frame.b[n] = self.b[ri];
                n += 1;
            }
            r += 1;
        }
        frame.n = n;
        self.undo[self.undo_len] = frame;
        self.undo_len += 1;

        // The moving piece's own-stone pattern, before it is lifted.
        let mut pat = 0u32;
        for ddr in -1..=1 {
            let rr = from_r + ddr;
            if !(0..SIZE).contains(&rr) {
                continue;
            }
            for ddc in -1..=1 {
                let cc = from_c + ddc;
                if !(0..SIZE).contains(&cc) {
                    continue;
                }
                let own_row = if white {
                    self.w[rr as usize]
                } else {
                    self.b[rr as usize]
                };
                if own_row & (1u32 << cc) != 0 {
                    pat |= 1u32 << ((ddr + 1) * 3 + (ddc + 1));
                    if track {
                        delta ^= zob_cell(white, (rr * SIZE + cc) as usize);
                    }
                }
            }
        }
        // Lift it.
        let ocm = COL3[from_c as usize];
        for rr in from_r - 1..=from_r + 1 {
            if (0..SIZE).contains(&rr) {
                if white {
                    self.w[rr as usize] &= !ocm;
                } else {
                    self.b[rr as usize] &= !ocm;
                }
            }
        }
        // Clear the destination footprint (captures, both colours).
        let mut captured = false;
        let dcm = COL3[to_c as usize];
        for r in to_r - 1..=to_r + 1 {
            if (0..SIZE).contains(&r) {
                let ri = r as usize;
                if (self.w[ri] | self.b[ri]) & dcm != 0 {
                    captured = true;
                }
                if track {
                    let mut m = self.w[ri] & dcm;
                    while m != 0 {
                        let c = m.trailing_zeros() as usize;
                        delta ^= zob_cell(true, ri * 20 + c);
                        m &= m - 1;
                    }
                    let mut m = self.b[ri] & dcm;
                    while m != 0 {
                        let c = m.trailing_zeros() as usize;
                        delta ^= zob_cell(false, ri * 20 + c);
                        m &= m - 1;
                    }
                }
                self.w[ri] &= !dcm;
                self.b[ri] &= !dcm;
            }
        }
        // Put the piece down; stones landing on the border ring fall off.
        for ddr in -1..=1 {
            for ddc in -1..=1 {
                if pat & (1u32 << ((ddr + 1) * 3 + (ddc + 1))) != 0 {
                    let rr = to_r + ddr;
                    let cc = to_c + ddc;
                    if (MIN..=MAX).contains(&rr) && (MIN..=MAX).contains(&cc) {
                        if white {
                            self.w[rr as usize] |= 1u32 << cc;
                        } else {
                            self.b[rr as usize] |= 1u32 << cc;
                        }
                        if track {
                            delta ^= zob_cell(white, (rr * SIZE + cc) as usize);
                        }
                    } else {
                        captured = true;
                    }
                }
            }
        }
        self.quiet = if captured { 0 } else { self.quiet + 1 };
        delta
    }

    fn stone_counts(&self) -> (i32, i32) {
        let (mut ws, mut bs) = (0, 0);
        for r in MIN..=MAX {
            ws += (self.w[r as usize] & PLAY_MASK).count_ones() as i32;
            bs += (self.b[r as usize] & PLAY_MASK).count_ones() as i32;
        }
        (ws, bs)
    }

    fn unmake_move(&mut self) {
        self.undo_len -= 1;
        let frame = self.undo[self.undo_len];
        self.quiet = frame.quiet;
        for i in 0..frame.n {
            let ri = frame.r[i];
            self.w[ri] = frame.w[i];
            self.b[ri] = frame.b[i];
        }
    }

    /// All legal moves for the mover, sorted best first. `lazy`: key = opponent stones captured,
    /// outcome left for the search; otherwise outcome and the full evaluation of each child.
    /// Writes them at the scratch bump pointer and returns (start, count).
    fn gen_moves(&mut self, white: bool, lazy: bool) -> (usize, usize) {
        let pw = self.w;
        let pb = self.b;
        let (own, opp) = if white { (pw, pb) } else { (pb, pw) };
        let mut occ_g = [0u32; 22];
        let mut opp_g = [0u32; 22];
        for i in 0..SIZE as usize {
            occ_g[i + 1] = pw[i] | pb[i];
            opp_g[i + 1] = opp[i];
        }
        let start = self.arena_top;
        let mut count = 0usize;

        for r in 0..SIZE {
            let opp_up = if r >= 1 { opp[(r - 1) as usize] } else { 0 };
            let opp_dn = if r < SIZE - 1 {
                opp[(r + 1) as usize]
            } else {
                0
            };
            let own_up = if r >= 1 { own[(r - 1) as usize] } else { 0 };
            let own_dn = if r < SIZE - 1 {
                own[(r + 1) as usize]
            } else {
                0
            };
            let opprow = opp_up | opp[r as usize] | opp_dn;
            let ownrow = own_up | own[r as usize] | own_dn;
            let opp_block = opprow | (opprow << 1) | (opprow >> 1);
            let own_pres = ownrow | (ownrow << 1) | (ownrow >> 1);
            let mut centres = own_pres & !opp_block & 0xF_FFFF;
            while centres != 0 {
                let c = centres.trailing_zeros() as i32;
                centres &= centres - 1;
                let cm = COL3[c as usize];
                let max_steps = if own[r as usize] & (1u32 << c) != 0 {
                    SIZE
                } else {
                    3
                };
                let mut visited = [0u32; 20];
                for (dr, dc) in DIRS {
                    let sr = r + dr;
                    let sc = c + dc;
                    if !(0..SIZE).contains(&sr) || !(0..SIZE).contains(&sc) {
                        continue;
                    }
                    if own[sr as usize] & (1u32 << sc) == 0 {
                        continue;
                    }
                    let mut step = 1;
                    while step <= max_steps {
                        let nr = r + dr * step;
                        let nc = c + dc * step;
                        if !(0..SIZE).contains(&nr) || !(0..SIZE).contains(&nc) {
                            break;
                        }
                        let ncm = COL3[nc as usize];
                        let ni = (nr + 1) as usize;
                        let (hits, caps);
                        if (occ_g[ni - 1] | occ_g[ni] | occ_g[ni + 1]) & ncm == 0 {
                            hits = 0;
                            caps = 0;
                        } else {
                            let mut h = 0u32;
                            let mut cp = 0i32;
                            for row in nr - 1..=nr + 1 {
                                let gi = (row + 1) as usize;
                                let mut o = occ_g[gi] & ncm;
                                if row >= r - 1 && row <= r + 1 {
                                    o &= !cm;
                                }
                                h |= o;
                                cp += (opp_g[gi] & ncm).count_ones() as i32;
                            }
                            hits = h;
                            caps = cp;
                        }
                        if visited[nr as usize] & (1u32 << nc) == 0 {
                            visited[nr as usize] |= 1u32 << nc;
                            let mv = if lazy {
                                EMove {
                                    from_r: r,
                                    from_c: c,
                                    to_r: nr,
                                    to_c: nc,
                                    outcome: 0,
                                    key: caps,
                                }
                            } else {
                                self.make_move(r, c, nr, nc, white, false);
                                let my_has = if white {
                                    has_ring_w(&self.w, &self.b)
                                } else {
                                    has_ring_b(&self.w, &self.b)
                                };
                                let op_has = if white {
                                    has_ring_b(&self.w, &self.b)
                                } else {
                                    has_ring_w(&self.w, &self.b)
                                };
                                let at_limit = self.quiet >= QUIET_LIMIT;
                                let outcome = if !my_has {
                                    OUT_LOSS
                                } else if !op_has {
                                    OUT_WIN
                                } else if at_limit {
                                    let (ws, bs) = self.stone_counts();
                                    let (mys, ops) = if white { (ws, bs) } else { (bs, ws) };
                                    stone_outcome(mys, ops)
                                } else {
                                    OUT_CONTINUE
                                };
                                let key =
                                    evaluate(&self.w, &self.b, white, self.quiet, &self.weights);
                                self.unmake_move();
                                EMove {
                                    from_r: r,
                                    from_c: c,
                                    to_r: nr,
                                    to_c: nc,
                                    outcome,
                                    key,
                                }
                            };
                            self.arena[start + count] = mv;
                            count += 1;
                        }
                        if hits != 0 {
                            break;
                        }
                        step += 1;
                    }
                }
            }
        }
        let (moves, rest) = self.arena[start..].split_at_mut(count);
        stable_sort_moves(moves, rest);
        self.arena_top = start + count;
        (start, count)
    }

    /// Whether to stop now: the app checks the clock every 256 nodes; the node budget too.
    #[inline]
    fn check_stop(&mut self) -> bool {
        if self.nodes & CLOCK_CHECK_MASK == 0 && self.out_of_budget() {
            self.timed_out = true;
            return true;
        }
        false
    }

    /// Eager negamax (the TT-off self-check path, or `force_eager`).
    fn negamax(&mut self, white: bool, depth: i32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        if self.limited {
            self.count_node();
            if self.check_stop() {
                return 0;
            }
        }
        let mut key = 0u64;
        let mut tt_move = 0u32;
        if self.tt_enabled {
            key = self.hash ^ (if white { zobrist().side } else { 0 }) ^ zob_quiet(self.quiet);
            if let Some(h) = self.tt.probe(key) {
                if h.depth >= depth {
                    if h.flag == TF_EXACT {
                        return h.val;
                    }
                    if h.flag == TF_LOWER && h.val >= beta {
                        return h.val;
                    }
                    if h.flag == TF_UPPER && h.val <= alpha {
                        return h.val;
                    }
                }
                tt_move = h.mv;
            }
        }
        let alpha_orig = alpha;
        let save_top = self.arena_top;
        let (mstart, mcount) = self.gen_moves(white, false);
        if mcount == 0 {
            self.arena_top = save_top;
            return evaluate(&self.w, &self.b, white, self.quiet, &self.weights);
        }
        if tt_move != 0 {
            for i in 0..mcount {
                if pack_mv(&self.arena[mstart + i]) == tt_move {
                    self.arena.swap(mstart, mstart + i);
                    break;
                }
            }
        }
        let mut best = NEG_INF;
        let mut best_mv = 0u32;
        for i in 0..mcount {
            let mv = self.arena[mstart + i];
            let score = if let Some(t) = terminal_score(mv.outcome, ply) {
                t
            } else if depth == 1 {
                mv.key
            } else {
                let d = self.make_move(
                    mv.from_r,
                    mv.from_c,
                    mv.to_r,
                    mv.to_c,
                    white,
                    self.tt_enabled,
                );
                self.hash ^= d;
                let s = -self.negamax(!white, depth - 1, -beta, -alpha, ply + 1);
                self.unmake_move();
                self.hash ^= d;
                if self.timed_out {
                    self.arena_top = save_top;
                    return 0;
                }
                s
            };
            if score > best {
                best = score;
                best_mv = pack_mv(&mv);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        self.arena_top = save_top;
        if self.tt_enabled && (best.abs() < WIN_SCORE - MAX_DEPTH) {
            let flag = if best <= alpha_orig {
                TF_UPPER
            } else if best >= beta {
                TF_LOWER
            } else {
                TF_EXACT
            };
            self.tt.store(key, depth, best, flag, best_mv);
        }
        best
    }

    /// Lazy negamax (the production TT-on path): cheap capture-ordered moves, TT move and killers
    /// first, each move's outcome and evaluation only when visited; null move and LMR.
    fn negamax_lazy(
        &mut self,
        white: bool,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        can_null: bool,
    ) -> i32 {
        self.count_node();
        if self.limited && self.check_stop() {
            return 0;
        }
        let key = self.hash ^ (if white { zobrist().side } else { 0 }) ^ zob_quiet(self.quiet);
        let mut tt_move = 0u32;
        if let Some(h) = self.tt.probe(key) {
            if h.depth >= depth {
                if h.flag == TF_EXACT {
                    return h.val;
                }
                if h.flag == TF_LOWER && h.val >= beta {
                    return h.val;
                }
                if h.flag == TF_UPPER && h.val <= alpha {
                    return h.val;
                }
            }
            tt_move = h.mv;
        }
        let alpha_orig = alpha;

        if can_null
            && self.switches.null_move
            && depth >= 3
            && beta.abs() < WIN_SCORE - MAX_DEPTH
            && self.has_ring_wit(white)
        {
            let red = if depth >= 6 { 3 } else { 2 };
            let nd = (depth - 1 - red).max(1);
            let s = -self.negamax_lazy(!white, nd, -beta, -beta + 1, ply + 1, false);
            if self.timed_out {
                return 0;
            }
            if s >= beta {
                return s;
            }
        }

        let save_top = self.arena_top;
        let (mstart, mcount) = self.gen_moves(white, true);
        if mcount == 0 {
            self.arena_top = save_top;
            return self.eval_leaf(white);
        }
        {
            let moves = &mut self.arena[mstart..mstart + mcount];
            let mut front = promote(moves, 0, tt_move);
            front = promote(moves, front, self.killers[ply as usize][0]);
            let _ = promote(moves, front, self.killers[ply as usize][1]);
        }

        let track = depth > 1;
        let mut best = NEG_INF;
        let mut best_mv = 0u32;
        for i in 0..mcount {
            let mv = self.arena[mstart + i];
            let d = self.make_move(mv.from_r, mv.from_c, mv.to_r, mv.to_c, white, track);
            self.hash ^= d;
            let my_has = self.has_ring_wit(white);
            let op_has = self.has_ring_wit(!white);
            let mut score;
            if !my_has {
                score = -(WIN_SCORE - (ply + 1));
            } else if !op_has {
                score = WIN_SCORE - (ply + 1);
            } else if self.quiet >= QUIET_LIMIT {
                let (ws, bs) = self.stone_counts();
                let (mine, theirs) = if white { (ws, bs) } else { (bs, ws) };
                score = terminal_score(stone_outcome(mine, theirs), ply).unwrap();
            } else if depth == 1 {
                score = self.eval_leaf(white);
            } else {
                let reduce = self.switches.lmr && depth >= 3 && i >= 3 && mv.key == 0;
                if reduce {
                    let red = if i >= 6 { 2 } else { 1 };
                    let nd = (depth - 1 - red).max(1);
                    score = -self.negamax_lazy(!white, nd, -beta, -alpha, ply + 1, true);
                    if score > alpha && nd < depth - 1 {
                        score = -self.negamax_lazy(!white, depth - 1, -beta, -alpha, ply + 1, true);
                    }
                } else {
                    score = -self.negamax_lazy(!white, depth - 1, -beta, -alpha, ply + 1, true);
                }
            }
            self.unmake_move();
            self.hash ^= d;
            if self.timed_out {
                self.arena_top = save_top;
                return 0;
            }
            if score > best {
                best = score;
                best_mv = pack_mv(&mv);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                if mv.key == 0 {
                    let p = pack_mv(&mv);
                    let k = &mut self.killers[ply as usize];
                    if k[0] != p {
                        k[1] = k[0];
                        k[0] = p;
                    }
                }
                break;
            }
        }
        self.arena_top = save_top;
        if best.abs() < WIN_SCORE - MAX_DEPTH {
            let flag = if best <= alpha_orig {
                TF_UPPER
            } else if best >= beta {
                TF_LOWER
            } else {
                TF_EXACT
            };
            self.tt.store(key, depth, best, flag, best_mv);
        }
        best
    }

    #[inline]
    fn search(&mut self, white: bool, depth: i32, alpha: i32, beta: i32, ply: i32) -> i32 {
        if self.tt_enabled && !self.switches.force_eager {
            self.negamax_lazy(white, depth, alpha, beta, ply, true)
        } else {
            self.negamax(white, depth, alpha, beta, ply)
        }
    }
}
