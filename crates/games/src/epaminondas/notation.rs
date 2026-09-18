//! Human notation. Squares are `<col><rank>`: columns `a`..`n` left to right, ranks `1`..`12`
//! with rank 1 white's home rank, so square 0 is `a1`.
//!
//! A single-piece move is `c2-c3`. A phalanx names its rear then its lead, then where the lead
//! lands: `c1c2-c4`. A capture uses `x` in place of `-`.

use super::{Move, Position, BLACK, EMPTY, HEIGHT, MAX_QUIET_MOVES, N, WHITE, WIDTH};
use crate::Side;

const COLS: &[u8; WIDTH] = b"abcdefghijklmn";

pub fn square_label(s: usize) -> String {
    format!("{}{}", COLS[s % WIDTH] as char, s / WIDTH + 1)
}

/// Parse a square label, e.g. `a1` or `n12`.
pub fn parse_square(text: &str) -> Option<usize> {
    let (col, rank) = split_square(text)?;
    (rank.len() == text.len() - 1).then_some(())?;
    Some(col + WIDTH * (rank.parse::<usize>().ok()? - 1))
}

/// Split the leading square off `text`: its column and the digits of its rank.
fn split_square(text: &str) -> Option<(usize, &str)> {
    let first = *text.as_bytes().first()?;
    let col = COLS.iter().position(|&c| c == first)?;
    let digits = text[1..].bytes().take_while(u8::is_ascii_digit).count();
    let rank = &text[1..1 + digits];
    let n: usize = rank.parse().ok()?;
    (1..=HEIGHT).contains(&n).then_some((col, rank))
}

/// Parse `c2-c3`, `c1c2-c4` or `c1c2xc4` into a move. Purely syntactic: legality is the rules'
/// business.
pub fn parse_move(text: &str) -> Option<Move> {
    let text = text.trim().to_ascii_lowercase();
    let (from, to) = text.split_once(['-', 'x'])?;
    let dest = parse_square(to)?;
    let (col, rank) = split_square(from)?;
    let first = col + WIDTH * (rank.parse::<usize>().ok()? - 1);
    let rest = &from[1 + rank.len()..];
    if rest.is_empty() {
        return Some(Move::new(first, first, dest));
    }
    Some(Move::new(parse_square(rest)?, first, dest))
}

pub(super) fn describe(pos: &Position, mv: Move) -> String {
    let sep = if pos.board[mv.dest as usize] == EMPTY {
        '-'
    } else {
        'x'
    };
    let dest = square_label(mv.dest as usize);
    if mv.is_single() {
        format!("{}{sep}{dest}", square_label(mv.lead as usize))
    } else {
        format!(
            "{}{}{sep}{dest}",
            square_label(mv.rear as usize),
            square_label(mv.lead as usize)
        )
    }
}

/// The board with rank 12 at the top, as black sees it from across the table.
pub(super) fn render(pos: &Position) -> String {
    let mut out = String::new();
    let header: String = COLS.iter().map(|&c| format!(" {}", c as char)).collect();
    out.push_str(&format!("   {header}\n"));
    for row in (0..HEIGHT).rev() {
        out.push_str(&format!("{:>2} ", row + 1));
        for col in 0..WIDTH {
            let glyph = match pos.board[row * WIDTH + col] {
                WHITE => 'W',
                BLACK => 'B',
                _ => '.',
            };
            out.push(' ');
            out.push(glyph);
        }
        out.push_str(&format!("  {}\n", row + 1));
    }
    out.push_str(&format!("   {header}\n"));
    let mover = match pos.to_move {
        Side::First => "White",
        Side::Second => "Black",
    };
    out.push_str(&format!(
        "move {}, quiet {}/{MAX_QUIET_MOVES}, {mover} to move (W {} / B {})\n",
        pos.moves,
        pos.quiet_moves,
        pos.count(Side::First),
        pos.count(Side::Second),
    ));
    debug_assert_eq!(pos.board.len(), N);
    out
}
