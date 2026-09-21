//! Human notation, the same as the Gess app in `~/calr3gh/gess`: columns `a`..`t` left to right,
//! ranks `20`..`1` top to bottom, so `rank = 20 - row` and cell 0 is `a20`. The playing area is
//! therefore `b2`..`s19`, and the border ring is the outside of that.
//!
//! A move names the centre the piece starts on and the centre it lands on: `g14-g11`. An `x` in
//! place of the `-` means the move took at least one stone off the board — captured, self-
//! captured, or pushed over the border.

use super::{Move, Position, BLACK, EMPTY, MAX_QUIET_MOVES, MIN_IDX, N, SIZE, WHITE};
use crate::Side;

const COLS: &[u8; SIZE] = b"abcdefghijklmnopqrst";

pub fn square_label(cell: usize) -> String {
    format!("{}{}", COLS[cell % SIZE] as char, SIZE - cell / SIZE)
}

/// Parse a cell label, e.g. `a20` or `g14`.
pub fn parse_square(text: &str) -> Option<usize> {
    let (col, rank) = split_square(text)?;
    (rank.len() == text.len() - 1).then_some(())?;
    Some((SIZE - rank.parse::<usize>().ok()?) * SIZE + col)
}

/// Split the leading cell off `text`: its column and the digits of its rank.
fn split_square(text: &str) -> Option<(usize, &str)> {
    let first = *text.as_bytes().first()?;
    let col = COLS.iter().position(|&c| c == first)?;
    let digits = text[1..].bytes().take_while(u8::is_ascii_digit).count();
    let rank = &text[1..1 + digits];
    let n: usize = rank.parse().ok()?;
    (1..=SIZE).contains(&n).then_some((col, rank))
}

/// Parse `g14-g11` or `g14xg11` into a move. Purely syntactic: legality is the rules' business.
pub fn parse_move(text: &str) -> Option<Move> {
    let text = text.trim().to_ascii_lowercase();
    let (from, to) = text.split_once(['-', 'x'])?;
    Some(Move::new(parse_square(from)?, parse_square(to)?))
}

/// Stones anywhere on the grid.
fn stones(pos: &Position) -> usize {
    pos.board.iter().filter(|&&c| c != EMPTY).count()
}

pub(super) fn describe(pos: &Position, mv: Move) -> String {
    let takes = stones(&pos.apply(mv)) < stones(pos);
    format!(
        "{}{}{}",
        square_label(mv.from as usize),
        if takes { 'x' } else { '-' },
        square_label(mv.to as usize)
    )
}

/// The board with rank 20 at the top, as white sees it. The border ring is drawn as blanks: it
/// can never hold a stone, but a piece may be centred on it and stones pushed onto it are lost.
pub(super) fn render(pos: &Position) -> String {
    let mut out = String::new();
    let header: String = COLS.iter().map(|&c| format!(" {}", c as char)).collect();
    out.push_str(&format!("   {header}\n"));
    for row in 0..SIZE {
        let rank = SIZE - row;
        out.push_str(&format!("{rank:>2} "));
        for col in 0..SIZE {
            let border = !(MIN_IDX..SIZE - 1).contains(&row) || !(MIN_IDX..SIZE - 1).contains(&col);
            let glyph = match pos.board[row * SIZE + col] {
                BLACK => 'B',
                WHITE => 'W',
                _ if border => ' ',
                _ => '.',
            };
            out.push(' ');
            out.push(glyph);
        }
        out.push_str(&format!("  {rank}\n"));
    }
    out.push_str(&format!("   {header}\n"));
    let mover = match pos.to_move {
        Side::First => "Black",
        Side::Second => "White",
    };
    out.push_str(&format!(
        "move {}, quiet {}/{MAX_QUIET_MOVES}, {mover} to move (B {} / W {})\n",
        pos.moves,
        pos.quiet_moves,
        pos.count(Side::First),
        pos.count(Side::Second),
    ));
    debug_assert_eq!(pos.board.len(), N);
    out
}
