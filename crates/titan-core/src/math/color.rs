/// An RGBA color with `f32` channels, nominally in `[0, 1]`.
///
/// The type stores values as given and does no color-space conversion; callers decide whether
/// the channels are linear or sRGB-encoded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Color {
    /// Red.
    pub r: f32,
    /// Green.
    pub g: f32,
    /// Blue.
    pub b: f32,
    /// Alpha (opacity).
    pub a: f32,
}

impl Color {
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    /// Creates a color from all four channels.
    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates an opaque color.
    #[must_use]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::rgba(r, g, b, 1.0)
    }

    /// Creates a color from 8-bit channels, mapping `0..=255` to `0.0..=1.0`.
    #[must_use]
    pub const fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::rgba(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    /// Returns `[r, g, b, a]`.
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u8_hits_both_ends_exactly() {
        assert_eq!(Color::from_u8(0, 255, 0, 255), Color::rgb(0.0, 1.0, 0.0));
        assert_eq!(
            Color::from_u8(255, 255, 255, 0).to_array(),
            [1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(Color::from_u8(0, 0, 0, 0), Color::TRANSPARENT);
    }
}
