//! Titan's asset formats: decoded RGBA8 [`Image`]s, the QOI image codec, and sprite atlases.
//!
//! The crate is `no_std` (it needs only `alloc`), forbids `unsafe`, and depends only on
//! `titan-core`.
//!
//! - [`Image`] is a tightly packed, top-to-bottom RGBA8 pixel buffer with non-zero dimensions.
//! - [`decode_qoi`] and [`encode_qoi`] read and write the
//!   [QOI format](https://qoiformat.org/qoi-specification.pdf). Decoding rejects malformed or
//!   truncated input with a [`QoiError`] instead of panicking or over-allocating.
//! - [`Atlas`] parses a plaintext `.atlas` file of named pixel [`Region`]s and looks them up as
//!   normalized UVs for `titan_2d::Sprite::with_uv`.
//!
//! ```
//! use titan_assets::{Image, decode_qoi, encode_qoi};
//!
//! let mut image = Image::new(2, 1, vec![0; 8])?;
//! image.set_pixel(1, 0, [255, 128, 0, 255]);
//!
//! let qoi = encode_qoi(&image);
//! assert_eq!(decode_qoi(&qoi)?, image);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod atlas;
mod image;
mod qoi;

pub use atlas::{Atlas, AtlasError, AtlasErrorKind, Region};
pub use image::{Image, ImageError};
pub use qoi::{QoiError, decode_qoi, encode_qoi};
