//! Breakout levels in a plaintext, diffable format.
//!
//! A level is a `key: value` header, a `---` line, then the brick grid from top to bottom, one
//! character per brick:
//!
//! ```text
//! # Comments and blank lines are allowed in the header.
//! name: Classic
//! lives: 3
//! ball_speed: 360
//! paddle_width: 112
//!
//! ---
//! SSSSSSSSSS
//! RRRRRRRRRR
//! ..BBBBBB..
//! ```
//!
//! Every header key is required exactly once. In the grid, `.` is an empty cell and the other
//! symbols are [`BrickKind`]s; every row has the same width and at least one brick is present.
//! The grid is at most 32 columns by 24 rows, so every brick stays at least 24 x 10 world units
//! and visible.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// The built-in level.
const CLASSIC: &str = include_str!("levels/classic.level");

/// The paddle widths a level may ask for, in world units; the arena is 800 wide.
const PADDLE_WIDTHS: core::ops::RangeInclusive<f32> = 16.0..=400.0;
/// The starting lives a level may ask for; the HUD has room for nine.
const LIVES: core::ops::RangeInclusive<u32> = 1..=9;
/// The supported ball speeds, in world units per second; bounds keep physics responsive and
/// prevent extreme inputs from stalling or overflowing substep counts.
const BALL_SPEEDS: core::ops::RangeInclusive<f32> = 60.0..=1200.0;
/// The largest grid; it splits the 776 x 240 brick area into cells of at least 24.25 x 10.
const MAX_COLUMNS: usize = 32;
const MAX_ROWS: usize = 24;

/// What a grid cell holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BrickKind {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    /// Takes two hits.
    Silver,
}

impl BrickKind {
    fn from_symbol(symbol: char) -> Option<Self> {
        Some(match symbol {
            'R' => Self::Red,
            'O' => Self::Orange,
            'Y' => Self::Yellow,
            'G' => Self::Green,
            'B' => Self::Blue,
            'S' => Self::Silver,
            _ => return None,
        })
    }

    fn symbol(self) -> char {
        match self {
            Self::Red => 'R',
            Self::Orange => 'O',
            Self::Yellow => 'Y',
            Self::Green => 'G',
            Self::Blue => 'B',
            Self::Silver => 'S',
        }
    }

    /// The score for destroying the brick.
    pub(crate) fn points(self) -> u32 {
        match self {
            Self::Red => 50,
            Self::Orange => 40,
            Self::Yellow => 30,
            Self::Green => 20,
            Self::Blue => 10,
            Self::Silver => 100,
        }
    }

    /// How many hits destroy the brick.
    pub(crate) fn hit_points(self) -> u8 {
        match self {
            Self::Silver => 2,
            _ => 1,
        }
    }
}

/// A parsed, valid level. Only [`Level::parse`] creates one.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Level {
    name: String,
    lives: u32,
    ball_speed: f32,
    paddle_width: f32,
    columns: usize,
    /// Row-major from the top row; `None` is an empty cell.
    cells: Vec<Option<BrickKind>>,
}

impl Level {
    /// Returns the built-in level.
    pub(crate) fn classic() -> Self {
        // `Display`, not the `Debug` that `expect` would pull into the WASM binary.
        Self::parse(CLASSIC).unwrap_or_else(|error| panic!("invalid built-in level: {error}"))
    }

    /// Parses the format described in the [module docs](self).
    pub(crate) fn parse(text: &str) -> Result<Self, LevelError> {
        let mut lines = text.lines().enumerate().map(|(i, line)| (i + 1, line));
        let error = |line, kind| LevelError { line, kind };

        let mut name = None;
        let mut lives = None;
        let mut ball_speed = None;
        let mut paddle_width = None;
        let separator_line = loop {
            let Some((number, line)) = lines.next() else {
                return Err(error(
                    text.lines().count().max(1),
                    LevelErrorKind::MissingSeparator,
                ));
            };
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "---" {
                break number;
            }
            let (key, value) = line
                .split_once(':')
                .ok_or(error(number, LevelErrorKind::ExpectedKeyValue))?;
            let (key, value) = (key.trim(), value.trim());
            let invalid = |rule| error(number, LevelErrorKind::InvalidValue(rule));
            let (slot, key) = match key {
                "name" => {
                    if value.is_empty() {
                        return Err(invalid("name must not be empty"));
                    }
                    (set(&mut name, String::from(value)), "name")
                }
                "lives" => {
                    let parsed = value.parse().ok().filter(|n| LIVES.contains(n));
                    let lives_value = parsed.ok_or(invalid("lives must be an integer in 1..=9"))?;
                    (set(&mut lives, lives_value), "lives")
                }
                "ball_speed" => {
                    let parsed = value
                        .parse::<f32>()
                        .ok()
                        .filter(|v| BALL_SPEEDS.contains(v));
                    let speed = parsed.ok_or(invalid("ball_speed must be in 60..=1200"))?;
                    (set(&mut ball_speed, speed), "ball_speed")
                }
                "paddle_width" => {
                    let parsed = value
                        .parse::<f32>()
                        .ok()
                        .filter(|v| PADDLE_WIDTHS.contains(v));
                    let width = parsed.ok_or(invalid("paddle_width must be in 16..=400"))?;
                    (set(&mut paddle_width, width), "paddle_width")
                }
                _ => return Err(error(number, LevelErrorKind::UnknownKey(key.into()))),
            };
            if !slot {
                return Err(error(number, LevelErrorKind::DuplicateKey(key)));
            }
        };
        let missing = |key| error(separator_line, LevelErrorKind::MissingKey(key));
        let name = name.ok_or(missing("name"))?;
        let lives = lives.ok_or(missing("lives"))?;
        let ball_speed = ball_speed.ok_or(missing("ball_speed"))?;
        let paddle_width = paddle_width.ok_or(missing("paddle_width"))?;

        // Blank lines around the grid are ignored; inside it they are rows of the wrong width.
        let mut rows: Vec<(usize, &str)> = lines
            .map(|(number, line)| (number, line.trim_end()))
            .skip_while(|(_, line)| line.is_empty())
            .collect();
        while rows.last().is_some_and(|(_, line)| line.is_empty()) {
            rows.pop();
        }
        let (first_line, columns) = match rows.first() {
            Some(&(number, row)) => (number, row.chars().count()),
            None => return Err(error(separator_line, LevelErrorKind::EmptyGrid)),
        };
        if columns > MAX_COLUMNS {
            return Err(error(first_line, LevelErrorKind::GridTooLarge));
        }
        if let Some(&(number, _)) = rows.get(MAX_ROWS) {
            return Err(error(number, LevelErrorKind::GridTooLarge));
        }
        for &(number, row) in &rows {
            let width = row.chars().count();
            if width != columns {
                return Err(error(
                    number,
                    LevelErrorKind::JaggedRow {
                        expected: columns,
                        actual: width,
                    },
                ));
            }
        }
        let mut cells = Vec::with_capacity(columns * rows.len());
        for &(number, row) in &rows {
            for symbol in row.chars() {
                cells.push(match symbol {
                    '.' => None,
                    _ => Some(
                        BrickKind::from_symbol(symbol)
                            .ok_or(error(number, LevelErrorKind::UnknownBrick(symbol)))?,
                    ),
                });
            }
        }
        if cells.iter().all(Option::is_none) {
            return Err(error(separator_line, LevelErrorKind::NoBricks));
        }
        Ok(Self {
            name,
            lives,
            ball_speed,
            paddle_width,
            columns,
            cells,
        })
    }

    #[cfg(test)]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn lives(&self) -> u32 {
        self.lives
    }

    /// World units per second.
    pub(crate) fn ball_speed(&self) -> f32 {
        self.ball_speed
    }

    /// World units.
    pub(crate) fn paddle_width(&self) -> f32 {
        self.paddle_width
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn rows(&self) -> usize {
        self.cells.len() / self.columns
    }

    /// Returns `(row, column, kind)` for every brick, rows numbered from the top.
    pub(crate) fn bricks(&self) -> impl Iterator<Item = (usize, usize, BrickKind)> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| cell.map(|kind| (i / self.columns, i % self.columns, kind)))
    }
}

/// Stores `value` in an empty `slot`, returning `false` if it was already set.
fn set<T>(slot: &mut Option<T>, value: T) -> bool {
    slot.replace(value).is_none()
}

impl fmt::Display for Level {
    /// Writes the canonical text, which [`Level::parse`] reads back unchanged.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "name: {}", self.name)?;
        writeln!(f, "lives: {}", self.lives)?;
        writeln!(f, "ball_speed: {}", self.ball_speed)?;
        writeln!(f, "paddle_width: {}", self.paddle_width)?;
        writeln!(f)?;
        writeln!(f, "---")?;
        for row in self.cells.chunks(self.columns) {
            for cell in row {
                let symbol = cell.map_or('.', BrickKind::symbol);
                write!(f, "{symbol}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Why a level failed to parse, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LevelError {
    /// The offending line, numbered from 1.
    pub(crate) line: usize,
    pub(crate) kind: LevelErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LevelErrorKind {
    /// A header line is not `key: value`.
    ExpectedKeyValue,
    UnknownKey(String),
    DuplicateKey(&'static str),
    /// Reported at the `---` line.
    MissingKey(&'static str),
    /// The value breaks the stated rule.
    InvalidValue(&'static str),
    /// Reported at the last line.
    MissingSeparator,
    /// Reported at the `---` line.
    EmptyGrid,
    /// More than `MAX_COLUMNS` cells across, reported at the first row, or more than `MAX_ROWS`
    /// rows, reported at the first row too many.
    GridTooLarge,
    JaggedRow {
        expected: usize,
        actual: usize,
    },
    UnknownBrick(char),
    /// Reported at the `---` line.
    NoBricks,
}

impl fmt::Display for LevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: ", self.line)?;
        match &self.kind {
            LevelErrorKind::ExpectedKeyValue => write!(f, "expected `key: value`"),
            LevelErrorKind::UnknownKey(key) => write!(f, "unknown key `{key}`"),
            LevelErrorKind::DuplicateKey(key) => write!(f, "duplicate key `{key}`"),
            LevelErrorKind::MissingKey(key) => write!(f, "missing key `{key}`"),
            LevelErrorKind::InvalidValue(rule) => write!(f, "{rule}"),
            LevelErrorKind::MissingSeparator => write!(f, "missing `---` before the grid"),
            LevelErrorKind::EmptyGrid => write!(f, "the grid has no rows"),
            LevelErrorKind::GridTooLarge => {
                write!(
                    f,
                    "the grid exceeds {MAX_COLUMNS} columns by {MAX_ROWS} rows"
                )
            }
            LevelErrorKind::JaggedRow { expected, actual } => {
                write!(f, "row has {actual} cells, expected {expected}")
            }
            LevelErrorKind::UnknownBrick(symbol) => write!(f, "unknown brick `{symbol}`"),
            LevelErrorKind::NoBricks => write!(f, "the grid has no bricks"),
        }
    }
}

impl core::error::Error for LevelError {}
