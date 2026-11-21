//! Titan: a fast, compact game engine for desktop and WASM.
//!
//! This crate is the engine's public entry point: it re-exports every engine crate under one
//! name. It is `no_std` so it builds unchanged for native targets and `wasm32-unknown-unknown`.
//!
//! - [`core`]: deterministic math, handles, a bump arena, and a fixed-timestep clock.
//! - [`gpu`]: the GPU contract and a recording device for tests.
//! - [`app`]: the `App` trait, input, replayable event logs, and a headless runner.
//! - [`render2d`]: a batched 2D quad and sprite renderer.
//! - [`wasm`]: the browser backend and its `export_app!` macro.
//! - `macos` (macOS only): the native AppKit and Metal backend.
//!
//! The `breakout` example plays the same game in a macOS window (`breakout`), in the browser
//! (`breakout_wasm`), and in deterministic headless replays (`tests/breakout_replay.rs`).

#![no_std]

pub use titan_2d as render2d;
pub use titan_app as app;
pub use titan_core as core;
pub use titan_gpu as gpu;
#[cfg(target_os = "macos")]
pub use titan_macos as macos;
pub use titan_wasm as wasm;

/// The version of the `titan` crate, as set in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_is_major_minor_patch() {
        let mut parts = VERSION.split('.');
        for _ in 0..3 {
            let part = parts.next().expect("missing version component");
            assert!(
                part.parse::<u64>().is_ok(),
                "non-numeric component {part:?} in {VERSION}"
            );
        }
        assert_eq!(parts.next(), None, "extra version component in {VERSION}");
    }
}
