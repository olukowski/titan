//! Titan's foundation: deterministic math, generational handles, a bump arena, and a
//! fixed-timestep clock.
//!
//! The crate is `no_std` (it needs only `alloc`) and has no dependencies, so it builds the same
//! for native targets and `wasm32-unknown-unknown`.
//!
//! - [`math`]: `f32` scalar functions plus [`Vec2`], [`Vec3`], [`Vec4`], [`Mat4`], [`Quat`],
//!   [`Rect`], and [`Color`], with bit-identical non-NaN results on every platform.
//! - [`handle`]: [`Pool`] storage addressed by 8-byte [`Handle`]s that detect stale use.
//! - [`arena`]: [`BumpArena`], a fixed-capacity [`Allocator`](core::alloc::Allocator) for
//!   frame-scoped `Vec`s and `Box`es.
//! - [`clock`]: [`FixedClock`], which turns frame times into fixed simulation steps.

#![no_std]

extern crate alloc;

pub mod arena;
pub mod clock;
pub mod handle;
pub mod math;

pub use arena::BumpArena;
pub use clock::FixedClock;
pub use handle::{Handle, Pool};
pub use math::{Color, Mat4, Quat, Rect, Vec2, Vec3, Vec4};
