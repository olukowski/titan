//! Plaintext sprite atlases: named pixel rectangles inside one image.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use titan_core::{Rect, Vec2};

/// A rectangle of an atlas image in pixels, with the origin at the top-left and `y` down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width, non-zero.
    pub width: u32,
    /// Height, non-zero.
    pub height: u32,
}

/// A parsed `.atlas` file: the image size and its named [`Region`]s.
///
/// An `.atlas` file is line-based and diffable:
///
/// ```text
/// # Comment lines start with '#' after optional leading whitespace; blank lines are ignored.
/// atlas 256 128
/// player 0 0 32 32
/// enemy 32 0 32 32
/// ```
///
/// The first non-blank, non-comment line is the header `atlas <width> <height>`: the image size
/// in pixels, both non-zero. Every following line is one region `<name> <x> <y> <width>
/// <height>` in pixels, with the origin at the image's top-left and `y` pointing down. Each
/// region must have a non-zero size and lie inside the image.
///
/// Fields are separated by ASCII whitespace and each line has exactly that many fields. Numbers
/// are decimal `u32`s. A name is any token without whitespace and must be unique. There are no
/// inline comments.
///
/// ```
/// use titan_assets::Atlas;
///
/// let atlas = Atlas::parse("atlas 256 128\nplayer 64 32 32 16\n")?;
/// let uv = atlas.uv("player").unwrap();
/// assert_eq!((uv.min.x, uv.min.y, uv.max.x, uv.max.y), (0.25, 0.25, 0.375, 0.375));
/// assert_eq!(atlas.uv("missing"), None);
/// # Ok::<(), titan_assets::AtlasError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atlas {
    width: u32,
    height: u32,
    regions: BTreeMap<String, Region>,
}

/// Why [`Atlas::parse`] rejected its input, and on which line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasError {
    /// 1-based line number of the offending line. For [`AtlasErrorKind::MissingHeader`] it is
    /// one past the last line.
    pub line: usize,
    /// What is wrong with that line.
    pub kind: AtlasErrorKind,
}

/// What [`AtlasError`] found wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtlasErrorKind {
    /// The input has no non-blank, non-comment line.
    MissingHeader,
    /// The header is not `atlas <width> <height>`.
    InvalidHeader,
    /// A region line does not have exactly five fields.
    WrongFieldCount {
        /// Number of fields on the line.
        found: usize,
    },
    /// A field that should be a decimal `u32` is not one.
    InvalidNumber {
        /// The offending field.
        token: String,
    },
    /// The image or a region has a zero width or height.
    ZeroSize,
    /// A region extends past the image's right or bottom edge.
    OutOfBounds {
        /// The region's name.
        name: String,
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
    /// A region reuses an earlier region's name.
    DuplicateName {
        /// The repeated name.
        name: String,
    },
}

impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atlas line {}: ", self.line)?;
        match &self.kind {
            AtlasErrorKind::MissingHeader => f.write_str("missing `atlas <width> <height>` header"),
            AtlasErrorKind::InvalidHeader => f.write_str("expected `atlas <width> <height>`"),
            AtlasErrorKind::WrongFieldCount { found } => {
                write!(
                    f,
                    "expected `<name> <x> <y> <width> <height>`, found {found} fields"
                )
            }
            AtlasErrorKind::InvalidNumber { token } => {
                write!(f, "`{token}` is not a non-negative 32-bit integer")
            }
            AtlasErrorKind::ZeroSize => f.write_str("width and height must be non-zero"),
            AtlasErrorKind::OutOfBounds {
                name,
                width,
                height,
            } => write!(
                f,
                "region `{name}` extends outside the {width}x{height} image"
            ),
            AtlasErrorKind::DuplicateName { name } => {
                write!(f, "region `{name}` is already defined")
            }
        }
    }
}

impl core::error::Error for AtlasError {}

impl Atlas {
    /// Parses the text of an `.atlas` file.
    pub fn parse(text: &str) -> Result<Self, AtlasError> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(index, line)| (index + 1, line.trim_ascii()))
            .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'));

        let Some((header_line, header)) = lines.next() else {
            return Err(AtlasError {
                line: text.lines().count() + 1,
                kind: AtlasErrorKind::MissingHeader,
            });
        };
        let at_header = |kind| AtlasError {
            line: header_line,
            kind,
        };
        let fields: Vec<&str> = header.split_ascii_whitespace().collect();
        let ["atlas", width, height] = fields[..] else {
            return Err(at_header(AtlasErrorKind::InvalidHeader));
        };
        let (width, height) = (
            number(width).map_err(at_header)?,
            number(height).map_err(at_header)?,
        );
        if width == 0 || height == 0 {
            return Err(at_header(AtlasErrorKind::ZeroSize));
        }

        let mut regions = BTreeMap::new();
        for (line, text) in lines {
            let at_line = |kind| AtlasError { line, kind };
            let fields: Vec<&str> = text.split_ascii_whitespace().collect();
            let [name, x, y, w, h] = fields[..] else {
                let found = fields.len();
                return Err(at_line(AtlasErrorKind::WrongFieldCount { found }));
            };
            let region = Region {
                x: number(x).map_err(at_line)?,
                y: number(y).map_err(at_line)?,
                width: number(w).map_err(at_line)?,
                height: number(h).map_err(at_line)?,
            };
            if region.width == 0 || region.height == 0 {
                return Err(at_line(AtlasErrorKind::ZeroSize));
            }
            let fits = |start: u32, len: u32, end| start.checked_add(len).is_some_and(|e| e <= end);
            if !fits(region.x, region.width, width) || !fits(region.y, region.height, height) {
                let name = name.into();
                return Err(at_line(AtlasErrorKind::OutOfBounds {
                    name,
                    width,
                    height,
                }));
            }
            if regions.insert(String::from(name), region).is_some() {
                let name = name.into();
                return Err(at_line(AtlasErrorKind::DuplicateName { name }));
            }
        }

        Ok(Self {
            width,
            height,
            regions,
        })
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixel rectangle named `name`, or `None` if there is none.
    #[must_use]
    pub fn region(&self, name: &str) -> Option<Region> {
        self.regions.get(name).copied()
    }

    /// The region named `name` in normalized texture coordinates, or `None` if there is none.
    ///
    /// The result is ready for `titan_2d::Sprite::uv`: `(0, 0)` is the image's top-left, so
    /// `uv.min` is the region's top-left corner and `uv.max` its bottom-right, in the same
    /// top-to-bottom row order as [`Image`](crate::Image). A region covering the whole image
    /// maps to `(0, 0)`–`(1, 1)`.
    #[must_use]
    pub fn uv(&self, name: &str) -> Option<Rect> {
        let r = self.region(name)?;
        let (w, h) = (self.width as f32, self.height as f32);
        // `parse` checked that `x + width` and `y + height` do not overflow.
        Some(Rect::from_min_max(
            Vec2::new(r.x as f32 / w, r.y as f32 / h),
            Vec2::new((r.x + r.width) as f32 / w, (r.y + r.height) as f32 / h),
        ))
    }
}

fn number(token: &str) -> Result<u32, AtlasErrorKind> {
    token.parse().map_err(|_| AtlasErrorKind::InvalidNumber {
        token: token.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    fn rect(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Rect {
        Rect::from_min_max(Vec2::new(min_x, min_y), Vec2::new(max_x, max_y))
    }

    #[test]
    fn regions_map_to_top_left_origin_uvs() {
        let atlas = Atlas::parse(
            "# sprites\r\n\r\n  # indented comment\r\natlas 256 128\r\n\
             player 64 32 32 16\r\nall 0 0 256 128\r\n",
        )
        .unwrap();

        assert_eq!((atlas.width(), atlas.height()), (256, 128));
        assert_eq!(
            atlas.region("player"),
            Some(Region {
                x: 64,
                y: 32,
                width: 32,
                height: 16,
            })
        );
        assert_eq!(atlas.uv("player"), Some(rect(0.25, 0.25, 0.375, 0.375)));
        assert_eq!(atlas.uv("all"), Some(rect(0.0, 0.0, 1.0, 1.0)));
        assert_eq!(atlas.region("Player"), None);
        assert_eq!(atlas.uv("missing"), None);
    }

    #[test]
    fn regions_may_touch_but_not_cross_the_image_edges() {
        let edge = Atlas::parse("atlas 16 8\ncorner 12 4 4 4").unwrap();
        assert_eq!(edge.uv("corner"), Some(rect(0.75, 0.5, 1.0, 1.0)));

        let max = u32::MAX;
        for region in [
            "a 13 4 4 4".into(),
            "a 12 5 4 4".into(),
            format!("a {max} 0 1 1"),
            format!("a 0 1 1 {max}"),
        ] {
            let err = Atlas::parse(&format!("atlas 16 8\n{region}")).unwrap_err();
            assert_eq!(
                err,
                AtlasError {
                    line: 2,
                    kind: AtlasErrorKind::OutOfBounds {
                        name: "a".into(),
                        width: 16,
                        height: 8,
                    },
                },
                "{region}"
            );
        }
    }

    #[test]
    fn errors_report_kind_and_true_line() {
        use AtlasErrorKind::*;
        let invalid = |token: &str| InvalidNumber {
            token: token.into(),
        };
        let cases = [
            ("", 1, MissingHeader),
            ("# only\n\n   \n", 4, MissingHeader),
            ("\n# c\nsheet 4 4", 3, InvalidHeader),
            ("atlas 4", 1, InvalidHeader),
            ("atlas 4 4 4", 1, InvalidHeader),
            ("atlas 4 -4", 1, invalid("-4")),
            ("atlas 0 4", 1, ZeroSize),
            ("atlas 4 4\n# c\n\na 0 0 1", 4, WrongFieldCount { found: 4 }),
            ("atlas 4 4\na 0 0 1 1 # c", 2, WrongFieldCount { found: 7 }),
            ("atlas 4 4\n\na 0 0 1.5 1", 3, invalid("1.5")),
            ("atlas 4 4\na 0 0 1 99999999999", 2, invalid("99999999999")),
            ("atlas 4 4\na 0 0 0 1", 2, ZeroSize),
            (
                "atlas 4 4\na 0 0 1 1\n# c\na 1 1 1 1",
                4,
                DuplicateName { name: "a".into() },
            ),
        ];
        for (text, line, kind) in cases {
            assert_eq!(
                Atlas::parse(text),
                Err(AtlasError { line, kind }),
                "{text:?}"
            );
        }
    }

    #[test]
    fn display_names_line_and_region() {
        let err = Atlas::parse("atlas 256 128\n\n\n\n\n\nfoo 250 0 8 8").unwrap_err();
        assert_eq!(
            format!("{err}"),
            "atlas line 7: region `foo` extends outside the 256x128 image"
        );
    }
}
