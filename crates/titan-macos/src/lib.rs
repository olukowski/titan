//! Titan's macOS backend: an AppKit window and a Metal [`GpuDevice`](titan_gpu::GpuDevice), in
//! pure Rust.
//!
//! The crate has no dependencies beyond Titan's own crates and compiles no C or Objective-C. It
//! links the system frameworks directly and talks to them through the Objective-C runtime's
//! `objc_msgSend`, with every message cast to its exact C signature.
//!
//! - [`run`] opens a window and drives an [`App`] with the same loop as
//!   [`HeadlessRunner`](titan_app::HeadlessRunner): apply input, run fixed steps, render, submit.
//! - `MetalGpu` implements [`GpuDevice`](titan_gpu::GpuDevice) on Metal, validating every call
//!   exactly as [`RecordingGpu`](titan_gpu::RecordingGpu) does. `MetalGpu::new_headless` and
//!   `MetalGpu::read_texture` render offscreen and read pixels back, for tests and tools.
//! - [`map_macos_keycode`] maps macOS virtual key codes to [`KeyCode`](titan_app::KeyCode)s.
//!
//! On other platforms the crate still builds: [`run`] returns an error and only the key mapping
//! is available, so workspaces that include it compile everywhere.

mod input;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
mod objc;
#[cfg(target_os = "macos")]
mod runner;
#[cfg(target_os = "macos")]
mod window;

pub use input::map_macos_keycode;
#[cfg(target_os = "macos")]
pub use metal::MetalGpu;

use titan_app::{App, WindowConfig};
use titan_gpu::GpuError;

/// Opens a window described by `config` and runs `app` in it until a step returns
/// [`AppAction::Exit`](titan_app::AppAction::Exit).
///
/// Like [`HeadlessRunner`](titan_app::HeadlessRunner), it calls [`App::init`], then
/// [`App::resize`] with the initial surface size, then runs frames: it applies window events
/// (calling [`App::resize`] for each [`Event::Resize`](titan_app::Event::Resize)), runs the fixed
/// 60 Hz steps the elapsed time adds up to, and renders and submits a frame while the window is
/// visible and has a nonzero size. Closing the window or pressing Command-Q sends
/// [`Event::CloseRequested`](titan_app::Event::CloseRequested); the app decides whether to exit.
///
/// # Errors
///
/// Returns the first error from the app or the GPU. On macOS it also fails when called off the
/// main thread or when there is no Metal device; on other platforms it always fails.
pub fn run<A: App>(app: A, config: WindowConfig) -> Result<(), GpuError> {
    #[cfg(target_os = "macos")]
    {
        runner::run(app, config)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, config);
        Err(GpuError::Backend(String::from(
            "titan-macos requires macOS",
        )))
    }
}
