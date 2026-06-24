//! Titan — an agent-native game engine in `no_std`, zero-dependency, stable Rust.
//!
//! See `docs/DESIGN.md` for the thesis and principles. This crate is the
//! foundation; right now it is just the explicit [`platform`] boundary to the
//! operating system, everything else is built on top of that.
//!
//! The crate is `no_std` when shipped and links `std` only under `cfg(test)`, so
//! the test harness, `#[test]`, and assertions keep working.
#![cfg_attr(not(test), no_std)]

pub mod platform;
