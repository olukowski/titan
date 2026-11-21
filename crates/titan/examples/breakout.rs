//! Breakout in a native macOS window, with sound. Left/Right or A/D (or the mouse) move the
//! paddle, Space launches the ball, Escape quits.
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
    let app = breakout::BreakoutApp::new().with_speaker(speaker());
    match titan::macos::run(app, window) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("breakout: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Starts the sound bank on the default output device. A machine without one can still play,
/// so the game reports the failure once and runs muted.
#[cfg(target_os = "macos")]
fn speaker() -> Option<breakout::Speaker> {
    let mut mixer = titan::audio::AudioMixer::new(breakout::SAMPLE_RATE).expect("48 kHz is valid");
    let sounds = breakout::Sounds::new(&mut mixer);
    match titan::macos::AudioOutput::start(mixer) {
        Ok(output) => Some(Box::new(move |event| {
            sounds.play(&mut output.mixer(), event);
        })),
        Err(error) => {
            eprintln!("breakout: audio unavailable, playing muted: {error}");
            None
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    eprintln!("breakout: the native example runs on macOS; build `breakout_wasm` for the browser");
    ExitCode::FAILURE
}
