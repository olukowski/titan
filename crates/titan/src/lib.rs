//! Titan: a fast, compact game engine for desktop and WASM.
//!
//! This crate is the engine's public entry point. It is `no_std` so it builds unchanged for
//! native targets and `wasm32-unknown-unknown`.

#![no_std]

pub use titan_core as core;

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
