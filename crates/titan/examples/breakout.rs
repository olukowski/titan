//! Breakout in a native macOS window. Left/Right or A/D (or the mouse) move the paddle, Space
//! launches the ball, Escape quits.
//!
//! ```sh
//! cargo run -p titan --example breakout
//! ```

use std::process::ExitCode;

#[cfg(target_os = "macos")]
#[path = "breakout_game/mod.rs"]
mod breakout;

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    let window = titan::app::WindowConfig::new("Titan Breakout", 800, 600);
    match titan::macos::run(breakout::BreakoutApp::new(), window) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("breakout: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("breakout: the native example runs on macOS; build `breakout_wasm` for the browser");
    ExitCode::FAILURE
}
