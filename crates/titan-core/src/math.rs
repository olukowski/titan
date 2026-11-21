//! Deterministic `f32` math: scalar functions, vectors, matrices, quaternions, and colors.
//!
//! Everything here is plain Rust on IEEE 754 arithmetic, so a given input produces the same bits
//! on every platform, WASM included, for every non-NaN result (NaN bit patterns are not
//! canonicalized). Vector, matrix, and color types are `#[repr(C)]` and can be copied into GPU
//! buffers directly as packed vertex attributes; WGSL and MSL uniform and storage arrays of
//! 3-component vectors need a 16-byte stride, so use [`Vec4`] or explicit padding there. See
//! [`Mat4`] for the coordinate conventions.

mod color;
mod mat4;
mod quat;
mod rect;
mod scalar;
mod vec;

pub use color::Color;
pub use mat4::Mat4;
pub use quat::Quat;
pub use rect::Rect;
pub use scalar::{acos, sin_cos, sqrt, tan};
pub use vec::{Vec2, Vec3, Vec4};
