//! [`EventLog`]: recorded frames of input in a line-based text format, for replays and tests.

use crate::input::{Event, KeyCode, MouseButton};
use crate::window::SurfaceSize;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

/// One frame of input: its elapsed time and the events that arrived during it.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameInput {
    /// Seconds since the previous frame.
    pub dt: f32,
    /// The frame's events, in arrival order.
    pub events: Vec<Event>,
}

/// A sequence of input frames that replays deterministically.
///
/// The text form ([`parse`](Self::parse), and [`Display`](fmt::Display) via `to_string`) is
/// line-based and diffable. `#` starts a comment line. `frame <dt>` starts a frame, and each
/// event line below it adds an event to it:
///
/// ```text
/// # Tap Space, then click.
/// frame 0.016666668
///   key_down Space
///   pointer_move 120.5 -3
/// frame 0.016666668
///   key_up Space
///   pointer_down Left
///   pointer_up Left
///   scroll 0 -1.5
///   resize 1600 1200 2
///   close
/// ```
///
/// Keys and mouse buttons are written by their [`KeyCode::name`] and [`MouseButton::name`].
/// Numbers use Rust's shortest round-trip formatting, so `parse(log.to_string())` reproduces
/// every value exactly, as long as each one is finite.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventLog {
    frames: Vec<FrameInput>,
}

impl EventLog {
    /// Creates an empty log.
    #[must_use]
    pub const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Appends a frame.
    ///
    /// # Panics
    ///
    /// Panics unless `dt` is finite and non-negative and every event meets the rules
    /// [`parse`](Self::parse) enforces, so the log's text form always parses back.
    pub fn push_frame(&mut self, dt: f32, events: Vec<Event>) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "frame dt must be finite and non-negative"
        );
        for event in &events {
            if let Err(rule) = check_event(event) {
                panic!("invalid {event:?}: {rule}");
            }
        }
        self.frames.push(FrameInput { dt, events });
    }

    /// Returns the text form described on [`EventLog`], which [`parse`](Self::parse) reads
    /// back. The same as `to_string`.
    #[must_use]
    pub fn format(&self) -> String {
        self.to_string()
    }

    /// Returns the frames in order.
    #[must_use]
    pub fn frames(&self) -> &[FrameInput] {
        &self.frames
    }

    /// Returns the number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether the log has no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Parses the text form described on [`EventLog`].
    ///
    /// # Errors
    ///
    /// Returns the first malformed line, numbered from 1.
    pub fn parse(text: &str) -> Result<Self, EventLogError> {
        let mut log = Self::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let error = |kind| EventLogError {
                line: index + 1,
                kind,
            };
            match parse_line(line).map_err(error)? {
                Line::Frame(dt) => log.frames.push(FrameInput {
                    dt,
                    events: Vec::new(),
                }),
                Line::Event(event) => log
                    .frames
                    .last_mut()
                    .ok_or(error(EventLogErrorKind::EventBeforeFrame))?
                    .events
                    .push(event),
            }
        }
        Ok(log)
    }
}

enum Line {
    Frame(f32),
    Event(Event),
}

fn parse_line(line: &str) -> Result<Line, EventLogErrorKind> {
    let mut words = line.split_ascii_whitespace();
    let command = words.next().expect("blank lines are skipped");
    let args: Vec<&str> = words.collect();
    let arity = |expected| {
        if args.len() == expected {
            Ok(())
        } else {
            Err(EventLogErrorKind::WrongArgCount {
                expected,
                actual: args.len(),
            })
        }
    };
    let event = match command {
        "frame" => {
            arity(1)?;
            let dt = in_range(args[0], |dt| dt >= 0.0, "frame dt must be finite and >= 0")?;
            return Ok(Line::Frame(dt));
        }
        "key_down" => {
            arity(1)?;
            Event::KeyDown(key(args[0])?)
        }
        "key_up" => {
            arity(1)?;
            Event::KeyUp(key(args[0])?)
        }
        "pointer_move" => {
            arity(2)?;
            Event::PointerMove {
                x: number(args[0])?,
                y: number(args[1])?,
            }
        }
        "pointer_down" => {
            arity(1)?;
            Event::PointerDown {
                button: button(args[0])?,
            }
        }
        "pointer_up" => {
            arity(1)?;
            Event::PointerUp {
                button: button(args[0])?,
            }
        }
        "scroll" => {
            arity(2)?;
            Event::Scroll {
                dx: number(args[0])?,
                dy: number(args[1])?,
            }
        }
        "resize" => {
            arity(3)?;
            // Not `SurfaceSize::new`, which panics on the bad scale factors `check_event` reports.
            Event::Resize(SurfaceSize {
                width: number(args[0])?,
                height: number(args[1])?,
                scale_factor: number(args[2])?,
            })
        }
        "close" => {
            arity(0)?;
            Event::CloseRequested
        }
        _ => return Err(EventLogErrorKind::UnknownCommand(command.into())),
    };
    check_event(&event).map_err(EventLogErrorKind::NonFiniteOrOutOfRange)?;
    Ok(Line::Event(event))
}

/// Checks the numeric rules an event must meet for its text to parse back, returning the
/// broken rule.
fn check_event(event: &Event) -> Result<(), &'static str> {
    let finite = |values: [f32; 2], rule| {
        if values.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(rule)
        }
    };
    match *event {
        Event::PointerMove { x, y } => finite([x, y], "pointer coordinates must be finite"),
        Event::Scroll { dx, dy } => finite([dx, dy], "scroll deltas must be finite"),
        Event::Resize(size) if !(size.scale_factor.is_finite() && size.scale_factor > 0.0) => {
            Err("scale_factor must be finite and > 0")
        }
        _ => Ok(()),
    }
}

fn number<T: FromStr>(word: &str) -> Result<T, EventLogErrorKind> {
    word.parse()
        .map_err(|_| EventLogErrorKind::InvalidNumber(word.into()))
}

/// Parses a finite `f32` that satisfies `ok`, or fails stating `rule`.
fn in_range(
    word: &str,
    ok: impl Fn(f32) -> bool,
    rule: &'static str,
) -> Result<f32, EventLogErrorKind> {
    let value: f32 = number(word)?;
    if value.is_finite() && ok(value) {
        Ok(value)
    } else {
        Err(EventLogErrorKind::NonFiniteOrOutOfRange(rule))
    }
}

fn key(word: &str) -> Result<KeyCode, EventLogErrorKind> {
    KeyCode::from_name(word).ok_or_else(|| EventLogErrorKind::UnknownKey(word.into()))
}

fn button(word: &str) -> Result<MouseButton, EventLogErrorKind> {
    MouseButton::from_name(word).ok_or_else(|| EventLogErrorKind::UnknownMouseButton(word.into()))
}

impl fmt::Display for EventLog {
    /// Writes the text form that [`EventLog::parse`] reads back.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for frame in &self.frames {
            writeln!(f, "frame {}", frame.dt)?;
            for event in &frame.events {
                match *event {
                    Event::KeyDown(key) => writeln!(f, "  key_down {}", key.name()),
                    Event::KeyUp(key) => writeln!(f, "  key_up {}", key.name()),
                    Event::PointerMove { x, y } => writeln!(f, "  pointer_move {x} {y}"),
                    Event::PointerDown { button } => {
                        writeln!(f, "  pointer_down {}", button.name())
                    }
                    Event::PointerUp { button } => writeln!(f, "  pointer_up {}", button.name()),
                    Event::Scroll { dx, dy } => writeln!(f, "  scroll {dx} {dy}"),
                    Event::Resize(size) => writeln!(
                        f,
                        "  resize {} {} {}",
                        size.width, size.height, size.scale_factor
                    ),
                    Event::CloseRequested => writeln!(f, "  close"),
                }?;
            }
        }
        Ok(())
    }
}

/// A malformed line in [`EventLog`] text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventLogError {
    /// The line number, from 1.
    pub line: usize,
    /// What is wrong with it.
    pub kind: EventLogErrorKind,
}

/// What is wrong with a line of [`EventLog`] text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventLogErrorKind {
    /// An event line came before the first `frame` line.
    EventBeforeFrame,
    /// The line starts with an unknown command.
    UnknownCommand(String),
    /// The command has the wrong number of arguments.
    WrongArgCount {
        /// The number the command takes.
        expected: usize,
        /// The number given.
        actual: usize,
    },
    /// An argument is not a number of the required type.
    InvalidNumber(String),
    /// A number is infinite, NaN, or outside its allowed range, stated here.
    NonFiniteOrOutOfRange(&'static str),
    /// A key name is not a [`KeyCode::name`].
    UnknownKey(String),
    /// A button name is not a [`MouseButton::name`].
    UnknownMouseButton(String),
}

impl fmt::Display for EventLogErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventBeforeFrame => f.write_str("event before the first `frame` line"),
            Self::UnknownCommand(command) => write!(f, "unknown command `{command}`"),
            Self::WrongArgCount { expected, actual } => {
                write!(f, "expected {expected} arguments, found {actual}")
            }
            Self::InvalidNumber(word) => write!(f, "invalid number `{word}`"),
            Self::NonFiniteOrOutOfRange(rule) => f.write_str(rule),
            Self::UnknownKey(word) => write!(f, "unknown key `{word}`"),
            Self::UnknownMouseButton(word) => write!(f, "unknown mouse button `{word}`"),
        }
    }
}

impl fmt::Display for EventLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.kind)
    }
}

impl core::error::Error for EventLogError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    /// A log with every key and button pressed and released, every event variant, and awkward
    /// floats: negative, fractional, negative zero, tiny, and huge.
    fn everything() -> EventLog {
        let mut log = EventLog::new();
        log.push_frame(0.0, Vec::new());
        log.push_frame(
            1.0 / 60.0,
            KeyCode::ALL
                .iter()
                .map(|&key| Event::KeyDown(key))
                .collect(),
        );
        log.push_frame(
            1.0 / 144.0,
            KeyCode::ALL.iter().map(|&key| Event::KeyUp(key)).collect(),
        );
        log.push_frame(
            0.1,
            MouseButton::ALL
                .iter()
                .flat_map(|&button| [Event::PointerDown { button }, Event::PointerUp { button }])
                .collect(),
        );
        log.push_frame(
            0.25,
            vec![
                Event::PointerMove { x: -0.0, y: 1e-7 },
                Event::PointerMove {
                    x: -12.375,
                    y: 3.4e38,
                },
                Event::Scroll {
                    dx: -0.1,
                    dy: f32::MIN_POSITIVE,
                },
                Event::Resize(SurfaceSize::new(2560, 1600, 2.0)),
                Event::Resize(SurfaceSize::new(0, u32::MAX, 1.25)),
                Event::CloseRequested,
            ],
        );
        log
    }

    #[test]
    fn text_round_trips_exactly() {
        let log = everything();
        let text = log.to_string();
        let parsed = EventLog::parse(&text).unwrap();
        assert_eq!(parsed, log);
        // Equal floats can differ in sign (-0.0 == 0.0), so compare the text too.
        assert_eq!(parsed.to_string(), text);
        assert_eq!(log.format(), text);
        assert_eq!(parsed.len(), 5);
    }

    #[test]
    fn parses_comments_blank_lines_and_spacing() {
        let text = "\
# A recorded session.

frame 0.016666668
\tkey_down   ArrowLeft
  # Comments may be indented.
   pointer_move 1.5 -2

frame 0.5
close
";
        let parsed = EventLog::parse(text).unwrap();
        let mut expected = EventLog::new();
        expected.push_frame(
            1.0 / 60.0,
            vec![
                Event::KeyDown(KeyCode::ArrowLeft),
                Event::PointerMove { x: 1.5, y: -2.0 },
            ],
        );
        expected.push_frame(0.5, vec![Event::CloseRequested]);
        assert_eq!(parsed, expected);
        assert!(EventLog::parse("\n# only comments\n").unwrap().is_empty());
    }

    #[test]
    fn errors_name_the_line_and_problem() {
        use EventLogErrorKind::*;
        let dt_rule = NonFiniteOrOutOfRange("frame dt must be finite and >= 0");
        let scale_rule = NonFiniteOrOutOfRange("scale_factor must be finite and > 0");
        let cases = [
            ("# header\n\nkey_down A", 3, EventBeforeFrame),
            ("frame 0.1\njump", 2, UnknownCommand("jump".into())),
            (
                "frame 0.1\nkey_down A B",
                2,
                WrongArgCount {
                    expected: 1,
                    actual: 2,
                },
            ),
            (
                "frame",
                1,
                WrongArgCount {
                    expected: 1,
                    actual: 0,
                },
            ),
            (
                "frame 0\nclose now",
                2,
                WrongArgCount {
                    expected: 0,
                    actual: 1,
                },
            ),
            ("frame 0.1\nkey_down a", 2, UnknownKey("a".into())),
            (
                "frame 0.1\n\npointer_down Back",
                3,
                UnknownMouseButton("Back".into()),
            ),
            ("frame 1/60", 1, InvalidNumber("1/60".into())),
            ("frame NaN", 1, dt_rule.clone()),
            ("frame 0\nframe inf", 2, dt_rule.clone()),
            ("frame -0.016", 1, dt_rule),
            (
                "frame 0\npointer_move 1 -inf",
                2,
                NonFiniteOrOutOfRange("pointer coordinates must be finite"),
            ),
            (
                "frame 0\nscroll NaN 0",
                2,
                NonFiniteOrOutOfRange("scroll deltas must be finite"),
            ),
            ("frame 0\nresize 800 600 0", 2, scale_rule.clone()),
            ("frame 0\nresize 800 600 -2", 2, scale_rule),
            (
                "frame 0\nresize 800 -600 1",
                2,
                InvalidNumber("-600".into()),
            ),
        ];
        for (text, line, kind) in cases {
            assert_eq!(
                EventLog::parse(text),
                Err(EventLogError { line, kind }),
                "{text:?}"
            );
        }
        let error = EventLog::parse("frame 0\nkey_down Shift").unwrap_err();
        assert_eq!(error.to_string(), "line 2: unknown key `Shift`");
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn push_frame_rejects_what_parse_would() {
        EventLog::new().push_frame(f32::NAN, Vec::new());
    }

    #[test]
    #[should_panic(expected = "scale_factor must be finite and > 0")]
    fn push_frame_rejects_a_zero_scale_factor() {
        let size = SurfaceSize {
            width: 1,
            height: 1,
            scale_factor: 0.0,
        };
        EventLog::new().push_frame(0.0, vec![Event::KeyUp(KeyCode::A), Event::Resize(size)]);
    }

    #[test]
    #[should_panic(expected = "scroll deltas must be finite")]
    fn push_frame_rejects_an_infinite_scroll() {
        let scroll = Event::Scroll {
            dx: f32::INFINITY,
            dy: 0.0,
        };
        EventLog::new().push_frame(0.0, vec![scroll]);
    }

    #[test]
    #[should_panic(expected = "pointer coordinates must be finite")]
    fn push_frame_rejects_a_nan_pointer() {
        let pointer = Event::PointerMove {
            x: 0.0,
            y: f32::NAN,
        };
        EventLog::new().push_frame(0.0, vec![pointer]);
    }
}
