//! Titan's app layer: the [`App`] trait a game implements, the input it sees, and a
//! [`HeadlessRunner`] that drives it without a window.
//!
//! The crate is `no_std` (it needs only `alloc`) and depends only on `titan-core` and
//! `titan-gpu`. Platform backends own the window and event loop; this crate owns everything
//! they share, so an app behaves the same on every platform and in tests.
//!
//! - [`App`]: `init`, `resize`, fixed-step `step`, and `render` into [`FrameCommands`](titan_gpu::FrameCommands).
//! - [`Event`] and [`InputState`]: platform events folded into held keys and buttons, per-step
//!   edges, and pointer and scroll deltas, without allocating.
//! - [`EventLog`]: frames of input in a diffable text format that replays exactly.
//! - [`HeadlessRunner`]: the frame loop over a [`RecordingGpu`](titan_gpu::RecordingGpu), for
//!   deterministic tests and replays.

#![no_std]

extern crate alloc;

mod app;
mod event_log;
mod headless;
mod input;
mod window;

pub use app::App;
pub use event_log::{EventLog, EventLogError, EventLogErrorKind, FrameInput};
pub use headless::{FrameOutcome, HeadlessConfig, HeadlessRunner, ReplaySummary};
pub use input::{Event, InputState, KeyCode, MouseButton};
pub use window::{AppAction, SurfaceSize, WindowConfig};
