use alloc::vec::Vec;
use core::fmt;

/// A decoded image: tightly packed, top-to-bottom RGBA8 pixels.
///
/// Both dimensions are non-zero and the buffer holds exactly `width * height * 4` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// Why [`Image::new`] rejected its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// The width or height is zero.
    ZeroDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// `width * height * 4` bytes do not fit in memory (`isize::MAX`).
    DimensionsOverflow {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The pixel buffer is not exactly `width * height * 4` bytes long.
    ByteLengthMismatch {
        /// `width * height * 4`.
        expected: usize,
        /// Length of the buffer passed in.
        actual: usize,
    },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions { width, height } => {
                write!(f, "image dimensions {width}x{height} must be non-zero")
            }
            Self::DimensionsOverflow { width, height } => {
                write!(f, "image dimensions {width}x{height} are too large")
            }
            Self::ByteLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} bytes of RGBA8 pixels, got {actual}")
            }
        }
    }
}

impl core::error::Error for ImageError {}

/// Byte length of a `width` x `height` RGBA8 buffer, or `None` if it exceeds `isize::MAX`
/// (the most any allocation can hold).
pub(crate) fn rgba_len(width: u32, height: u32) -> Option<usize> {
    let len = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?
        .checked_mul(4)?;
    (isize::try_from(len).is_ok()).then_some(len)
}

impl Image {
    /// Wraps `rgba` (tightly packed, top-to-bottom RGBA8) as a `width` x `height` image.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::ZeroDimensions { width, height });
        }
        let expected =
            rgba_len(width, height).ok_or(ImageError::DimensionsOverflow { width, height })?;
        if rgba.len() != expected {
            return Err(ImageError::ByteLengthMismatch {
                expected,
                actual: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The RGBA8 pixels, row by row from the top.
    pub fn as_bytes(&self) -> &[u8] {
        &self.rgba
    }

    /// The RGBA8 pixels, row by row from the top, for in-place editing.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.rgba
    }

    /// Consumes the image and returns its RGBA8 pixels.
    pub fn into_bytes(self) -> Vec<u8> {
        self.rgba
    }

    /// The pixel at column `x` and row `y` (from the top), or `None` if it is out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        let i = self.offset(x, y)?;
        Some([
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ])
    }

    /// Sets the pixel at column `x` and row `y`. Returns `false`, changing nothing, if it is out
    /// of bounds.
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) -> bool {
        let Some(i) = self.offset(x, y) else {
            return false;
        };
        self.rgba[i..i + 4].copy_from_slice(&rgba);
        true
    }

    fn offset(&self, x: u32, y: u32) -> Option<usize> {
        // Cannot overflow: `new` checked that `width * height * 4` fits in `usize`.
        (x < self.width && y < self.height)
            .then(|| (y as usize * self.width as usize + x as usize) * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn pixels_are_row_major_and_bounds_checked() {
        let mut image = Image::new(2, 3, vec![0; 24]).unwrap();
        assert!(image.set_pixel(1, 2, [1, 2, 3, 4]));
        assert_eq!(image.pixel(1, 2), Some([1, 2, 3, 4]));
        assert_eq!(&image.as_bytes()[20..], &[1, 2, 3, 4]);
        assert_eq!(image.pixel(0, 0), Some([0; 4]));

        assert_eq!(image.pixel(2, 0), None);
        assert_eq!(image.pixel(0, 3), None);
        assert!(!image.set_pixel(2, 0, [9; 4]));
        assert!(!image.set_pixel(0, 3, [9; 4]));
        assert_eq!(image.into_bytes(), {
            let mut expected = vec![0; 24];
            expected[20..].copy_from_slice(&[1, 2, 3, 4]);
            expected
        });
    }

    #[test]
    fn new_rejects_invalid_dimensions_and_lengths() {
        assert_eq!(
            Image::new(0, 4, vec![]),
            Err(ImageError::ZeroDimensions {
                width: 0,
                height: 4
            })
        );
        assert_eq!(
            Image::new(4, 0, vec![]),
            Err(ImageError::ZeroDimensions {
                width: 4,
                height: 0
            })
        );
        assert_eq!(
            Image::new(u32::MAX, u32::MAX, vec![]),
            Err(ImageError::DimensionsOverflow {
                width: u32::MAX,
                height: u32::MAX
            })
        );
        assert_eq!(
            Image::new(2, 2, vec![0; 15]),
            Err(ImageError::ByteLengthMismatch {
                expected: 16,
                actual: 15
            })
        );
        assert_eq!(
            Image::new(2, 2, vec![0; 17]),
            Err(ImageError::ByteLengthMismatch {
                expected: 16,
                actual: 17
            })
        );
    }
}
