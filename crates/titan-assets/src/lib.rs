//! Titan's asset formats: decoded RGBA8 [`Image`]s and the QOI image codec.
//!
//! The crate is `no_std` (it needs only `alloc`), forbids `unsafe`, and has no dependencies.
//!
//! - [`Image`] is a tightly packed, top-to-bottom RGBA8 pixel buffer with non-zero dimensions.
//! - [`decode_qoi`] and [`encode_qoi`] read and write the
//!   [QOI format](https://qoiformat.org/qoi-specification.pdf). Decoding rejects malformed or
//!   truncated input with a [`QoiError`] instead of panicking or over-allocating.
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

mod image;
mod qoi;

pub use image::{Image, ImageError};
pub use qoi::{QoiError, decode_qoi, encode_qoi};
