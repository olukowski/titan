//! The [QOI image format](https://qoiformat.org/qoi-specification.pdf).

use crate::image::{Image, rgba_len};
use alloc::vec::Vec;
use core::fmt;

const MAGIC: [u8; 4] = *b"qoif";
const HEADER_LEN: usize = 14;
const END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];
/// The longest run one `QOI_OP_RUN` chunk encodes; also the most pixels any single byte yields.
const MAX_RUN: u8 = 62;

const OP_INDEX: u8 = 0x00;
const OP_DIFF: u8 = 0x40;
const OP_LUMA: u8 = 0x80;
const OP_RUN: u8 = 0xC0;
const OP_RGB: u8 = 0xFE;
const OP_RGBA: u8 = 0xFF;
const TAG_MASK: u8 = 0xC0;

/// Why [`decode_qoi`] rejected its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QoiError {
    /// The input ended before the header, a chunk, or the end marker was complete.
    UnexpectedEof,
    /// The input does not start with `qoif`.
    InvalidMagic,
    /// The header's width or height is zero.
    ZeroDimensions {
        /// Width from the header.
        width: u32,
        /// Height from the header.
        height: u32,
    },
    /// The decoded pixels would not fit in memory (`isize::MAX` bytes).
    DimensionsOverflow {
        /// Width from the header.
        width: u32,
        /// Height from the header.
        height: u32,
    },
    /// The header's channel count is neither 3 (RGB) nor 4 (RGBA).
    InvalidChannels {
        /// Channel count from the header.
        channels: u8,
    },
    /// The header's colorspace is neither 0 (sRGB, linear alpha) nor 1 (all linear).
    InvalidColorspace {
        /// Colorspace from the header.
        colorspace: u8,
    },
    /// A run continues past the last pixel of the image.
    RunOverflow,
    /// The 8 bytes after the last pixel are not the end marker `[0, 0, 0, 0, 0, 0, 0, 1]`.
    InvalidEndMarker,
    /// Bytes follow the end marker.
    TrailingData,
    /// The allocator refused the decoded pixel buffer.
    AllocationFailed {
        /// Size of the refused allocation.
        bytes: usize,
    },
}

impl fmt::Display for QoiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("QOI data ends unexpectedly"),
            Self::InvalidMagic => f.write_str("not QOI data: missing `qoif` magic"),
            Self::ZeroDimensions { width, height } => {
                write!(f, "QOI dimensions {width}x{height} must be non-zero")
            }
            Self::DimensionsOverflow { width, height } => {
                write!(f, "QOI dimensions {width}x{height} are too large")
            }
            Self::InvalidChannels { channels } => {
                write!(f, "invalid QOI channel count {channels} (expected 3 or 4)")
            }
            Self::InvalidColorspace { colorspace } => {
                write!(f, "invalid QOI colorspace {colorspace} (expected 0 or 1)")
            }
            Self::RunOverflow => f.write_str("QOI run extends past the last pixel"),
            Self::InvalidEndMarker => f.write_str("invalid QOI end marker"),
            Self::TrailingData => f.write_str("unexpected data after the QOI end marker"),
            Self::AllocationFailed { bytes } => {
                write!(f, "cannot allocate {bytes} bytes for QOI pixels")
            }
        }
    }
}

impl core::error::Error for QoiError {}

fn hash([r, g, b, a]: [u8; 4]) -> usize {
    (usize::from(r) * 3 + usize::from(g) * 5 + usize::from(b) * 7 + usize::from(a) * 11) % 64
}

/// Decodes a QOI file into an RGBA8 [`Image`].
///
/// Decoding starts from opaque alpha (255), so standard 3-channel RGB streams decode as opaque
/// RGBA. As the QOI spec requires, the header's channel and colorspace bytes are validated but do
/// not change how chunks decode.
pub fn decode_qoi(bytes: &[u8]) -> Result<Image, QoiError> {
    let (header, payload) = bytes
        .split_first_chunk::<HEADER_LEN>()
        .ok_or(QoiError::UnexpectedEof)?;
    if header[..4] != MAGIC {
        return Err(QoiError::InvalidMagic);
    }
    let width = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let height = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let (channels, colorspace) = (header[12], header[13]);
    if width == 0 || height == 0 {
        return Err(QoiError::ZeroDimensions { width, height });
    }
    if !matches!(channels, 3 | 4) {
        return Err(QoiError::InvalidChannels { channels });
    }
    if colorspace > 1 {
        return Err(QoiError::InvalidColorspace { colorspace });
    }
    let len = rgba_len(width, height).ok_or(QoiError::DimensionsOverflow { width, height })?;
    // A tiny file claiming huge dimensions must fail here, before the allocation below.
    let chunk_bytes = payload.len().saturating_sub(END_MARKER.len());
    if chunk_bytes.saturating_mul(usize::from(MAX_RUN)) < len / 4 {
        return Err(QoiError::UnexpectedEof);
    }

    let mut rgba = Vec::new();
    rgba.try_reserve_exact(len)
        .map_err(|_| QoiError::AllocationFailed { bytes: len })?;
    let mut index = [[0u8; 4]; 64];
    let mut px = [0, 0, 0, 255];
    let mut pos = 0;
    let mut next = || {
        let byte = payload.get(pos).copied().ok_or(QoiError::UnexpectedEof);
        pos += 1;
        byte
    };
    while rgba.len() < len {
        let b1 = next()?;
        let mut run = 1;
        match b1 {
            OP_RGB => px = [next()?, next()?, next()?, px[3]],
            OP_RGBA => px = [next()?, next()?, next()?, next()?],
            _ => match b1 & TAG_MASK {
                OP_INDEX => px = index[usize::from(b1)],
                OP_DIFF => {
                    px[0] = px[0].wrapping_add((b1 >> 4 & 3).wrapping_sub(2));
                    px[1] = px[1].wrapping_add((b1 >> 2 & 3).wrapping_sub(2));
                    px[2] = px[2].wrapping_add((b1 & 3).wrapping_sub(2));
                }
                OP_LUMA => {
                    let b2 = next()?;
                    let dg = (b1 & 0x3F).wrapping_sub(32);
                    px[0] = px[0].wrapping_add(dg.wrapping_add(b2 >> 4).wrapping_sub(8));
                    px[1] = px[1].wrapping_add(dg);
                    px[2] = px[2].wrapping_add(dg.wrapping_add(b2 & 0x0F).wrapping_sub(8));
                }
                _ => {
                    run = usize::from(b1 & 0x3F) + 1;
                    if rgba.len() + run * 4 > len {
                        return Err(QoiError::RunOverflow);
                    }
                }
            },
        }
        index[hash(px)] = px;
        for _ in 0..run {
            rgba.extend_from_slice(&px);
        }
    }

    let rest = payload.get(pos..).ok_or(QoiError::UnexpectedEof)?;
    let (marker, trailing) = rest
        .split_first_chunk::<8>()
        .ok_or(QoiError::UnexpectedEof)?;
    if *marker != END_MARKER {
        return Err(QoiError::InvalidEndMarker);
    }
    if !trailing.is_empty() {
        return Err(QoiError::TrailingData);
    }
    Ok(Image::new(width, height, rgba).expect("decoded exactly width * height pixels"))
}

/// Encodes `image` as a QOI file with 4 channels and the sRGB colorspace.
pub fn encode_qoi(image: &Image) -> Vec<u8> {
    let (pixels, _) = image.as_bytes().as_chunks::<4>();
    let mut out = Vec::with_capacity(HEADER_LEN + END_MARKER.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&image.width().to_be_bytes());
    out.extend_from_slice(&image.height().to_be_bytes());
    out.extend_from_slice(&[4, 0]);

    let mut index = [[0u8; 4]; 64];
    let mut prev = [0, 0, 0, 255];
    let mut run = 0;
    for &px in pixels {
        if px == prev {
            run += 1;
            if run == MAX_RUN {
                out.push(OP_RUN | (run - 1));
                run = 0;
            }
            continue;
        }
        if run > 0 {
            out.push(OP_RUN | (run - 1));
            run = 0;
        }
        // Like the reference `qoi.h`, only non-run pixels enter the index.
        let slot = hash(px);
        if index[slot] == px {
            out.push(OP_INDEX | slot as u8);
        } else if px[3] != prev[3] {
            index[slot] = px;
            out.push(OP_RGBA);
            out.extend_from_slice(&px);
        } else {
            index[slot] = px;
            let dr = px[0].wrapping_sub(prev[0]) as i8;
            let dg = px[1].wrapping_sub(prev[1]) as i8;
            let db = px[2].wrapping_sub(prev[2]) as i8;
            let (dr_dg, db_dg) = (dr.wrapping_sub(dg), db.wrapping_sub(dg));
            if [dr, dg, db].iter().all(|d| (-2..=1).contains(d)) {
                out.push(OP_DIFF | ((dr + 2) as u8) << 4 | ((dg + 2) as u8) << 2 | (db + 2) as u8);
            } else if (-32..=31).contains(&dg)
                && (-8..=7).contains(&dr_dg)
                && (-8..=7).contains(&db_dg)
            {
                out.push(OP_LUMA | (dg + 32) as u8);
                out.push(((dr_dg + 8) as u8) << 4 | (db_dg + 8) as u8);
            } else {
                out.extend_from_slice(&[OP_RGB, px[0], px[1], px[2]]);
            }
        }
        prev = px;
    }
    if run > 0 {
        out.push(OP_RUN | (run - 1));
    }
    out.extend_from_slice(&END_MARKER);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn image(width: u32, height: u32, pixels: &[[u8; 4]]) -> Image {
        Image::new(width, height, pixels.as_flattened().to_vec()).unwrap()
    }

    fn header(width: u32, height: u32, channels: u8, colorspace: u8) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[channels, colorspace]);
        bytes
    }

    /// A complete RGBA file around `chunks`.
    fn file(width: u32, height: u32, chunks: &[u8]) -> Vec<u8> {
        [&header(width, height, 4, 0), chunks, &END_MARKER].concat()
    }

    /// Encodes `image`, checks the chunks between header and end marker, and round-trips it.
    fn assert_encodes_to(image: &Image, chunks: &[u8]) {
        let encoded = encode_qoi(image);
        assert_eq!(encoded, file(image.width(), image.height(), chunks));
        assert_eq!(decode_qoi(&encoded).as_ref(), Ok(image));
    }

    #[test]
    fn encodes_every_opcode_with_wraparound() {
        let pixels = [
            [0, 0, 0, 255],      // equals the initial previous pixel: RUN 1
            [1, 255, 0, 255],    // DIFF (1, -1, 0): green wraps 0 -> 255
            [255, 0, 0, 255],    // DIFF (-2, 1, 0): red wraps 1 -> 255, green 255 -> 0
            [24, 20, 17, 255],   // LUMA dg 20, dr-dg 5, db-dg -3: red wraps 255 -> 24
            [100, 200, 50, 255], // RGB: dg -76 is out of LUMA range
            [100, 200, 50, 128], // RGBA: alpha changed
            [1, 255, 0, 255],    // INDEX: seen before, slot 51
        ];
        assert_encodes_to(
            &image(7, 1, &pixels),
            &[
                0xC0, 0x76, 0x4E, 0xB4, 0xD5, 0xFE, 100, 200, 50, 0xFF, 100, 200, 50, 128, 0x33,
            ],
        );
    }

    #[test]
    fn encoder_uses_exact_diff_and_luma_ranges() {
        let pixels = [
            [1, 254, 254, 255],   // DIFF (1, -2, -2)
            [24, 29, 36, 255],    // LUMA dg 31, dr-dg -8, db-dg 7
            [255, 253, 252, 255], // LUMA dg -32, dr-dg 7, db-dg -8
            [31, 29, 28, 255],    // RGB: dg 32
            [39, 29, 28, 255],    // RGB: dr-dg 8
            [41, 29, 28, 255],    // LUMA, not DIFF: dr 2
        ];
        assert_encodes_to(
            &image(6, 1, &pixels),
            &[
                0x70, 0xBF, 0x0F, 0x80, 0xF0, 0xFE, 31, 29, 28, 0xFE, 39, 29, 28, 0xA0, 0xA8,
            ],
        );
    }

    #[test]
    fn runs_split_at_62() {
        for (count, chunks) in [
            (1, &[0xC0][..]),
            (62, &[0xFD]),
            (63, &[0xFD, 0xC0]),
            (125, &[0xFD, 0xFD, 0xC0]),
        ] {
            assert_encodes_to(
                &image(count, 1, &vec![[0, 0, 0, 255]; count as usize]),
                chunks,
            );
        }
        // A run that does not start the image, then continues across rows.
        let mut pixels = vec![[9, 9, 9, 9]; 64];
        pixels[0] = [1, 2, 3, 4];
        assert_encodes_to(
            &image(8, 8, &pixels),
            &[0xFF, 1, 2, 3, 4, 0xFF, 9, 9, 9, 9, 0xFD],
        );
        // The output grows with the chunks emitted, not with a 5-bytes-per-pixel worst case.
        let solid = encode_qoi(&image(1024, 1, &vec![[0, 0, 0, 255]; 1024]));
        assert!(solid.capacity() < 1024, "capacity {}", solid.capacity());
    }

    #[test]
    fn transparent_black_hits_the_zeroed_index_slot() {
        // The index starts zeroed, so [0, 0, 0, 0] is an INDEX hit at slot 0 before it is seen.
        assert_encodes_to(&image(1, 1, &[[0, 0, 0, 0]]), &[0x00]);
    }

    #[test]
    fn colliding_index_slots_round_trip() {
        let (a, b) = ([255, 0, 0, 255], [100, 200, 50, 128]);
        assert_eq!(hash(a), hash(b));
        // `a` is a DIFF from the start pixel. `b` evicts `a`, so `a` must be re-sent in full rather than as a stale INDEX.
        assert_encodes_to(
            &image(4, 1, &[a, b, a, b]),
            &[
                0x5A, 0xFF, 100, 200, 50, 128, 0xFF, 255, 0, 0, 255, 0xFF, 100, 200, 50, 128,
            ],
        );
    }

    #[test]
    fn decodes_rgb_files_as_opaque_in_both_colorspaces() {
        for colorspace in [0, 1] {
            let bytes = [
                &header(2, 1, 3, colorspace),
                &[0xFE, 16, 32, 48, 0xC0][..],
                &END_MARKER,
            ]
            .concat();
            assert_eq!(
                decode_qoi(&bytes),
                Ok(image(2, 1, &[[16, 32, 48, 255], [16, 32, 48, 255]]))
            );
        }
    }

    #[test]
    fn decoder_indexes_run_pixels() {
        // Matching the spec and `qoi.h`: a leading run puts [0, 0, 0, 255] in slot 53.
        assert_eq!(hash([0, 0, 0, 255]), 53);
        assert_eq!(
            decode_qoi(&file(3, 1, &[0xC0, 0xFE, 1, 2, 3, 53])),
            Ok(image(
                3,
                1,
                &[[0, 0, 0, 255], [1, 2, 3, 255], [0, 0, 0, 255]]
            ))
        );
    }

    #[test]
    fn rejects_malformed_input() {
        let valid = file(1, 1, &[0xC0]);
        let with_header = |header: Vec<u8>| [&header, &[0xC0][..], &END_MARKER].concat();
        let cases: [(&str, Vec<u8>, QoiError); 21] = [
            ("empty", vec![], QoiError::UnexpectedEof),
            (
                "short header",
                valid[..10].to_vec(),
                QoiError::UnexpectedEof,
            ),
            (
                "magic",
                [b"qoix", &valid[4..]].concat(),
                QoiError::InvalidMagic,
            ),
            (
                "zero width",
                with_header(header(0, 1, 4, 0)),
                QoiError::ZeroDimensions {
                    width: 0,
                    height: 1,
                },
            ),
            (
                "zero height",
                with_header(header(1, 0, 4, 0)),
                QoiError::ZeroDimensions {
                    width: 1,
                    height: 0,
                },
            ),
            (
                "channels 0",
                with_header(header(1, 1, 0, 0)),
                QoiError::InvalidChannels { channels: 0 },
            ),
            (
                "channels 2",
                with_header(header(1, 1, 2, 0)),
                QoiError::InvalidChannels { channels: 2 },
            ),
            (
                "channels 5",
                with_header(header(1, 1, 5, 0)),
                QoiError::InvalidChannels { channels: 5 },
            ),
            (
                "colorspace 2",
                with_header(header(1, 1, 4, 2)),
                QoiError::InvalidColorspace { colorspace: 2 },
            ),
            (
                "size overflow",
                with_header(header(u32::MAX, u32::MAX, 4, 0)),
                QoiError::DimensionsOverflow {
                    width: u32::MAX,
                    height: u32::MAX,
                },
            ),
            // 1 Mi pixels (4 MiB) claimed by 1 chunk byte: rejected before allocating.
            (
                "allocation bomb",
                with_header(header(1024, 1024, 4, 0)),
                QoiError::UnexpectedEof,
            ),
            (
                "truncated RGB",
                [&header(1, 1, 4, 0)[..], &[0xFE, 1, 2]].concat(),
                QoiError::UnexpectedEof,
            ),
            (
                "truncated RGBA",
                [&header(1, 1, 4, 0)[..], &[0xFF, 1, 2, 3]].concat(),
                QoiError::UnexpectedEof,
            ),
            (
                "truncated LUMA",
                [&header(1, 1, 4, 0)[..], &[0x80]].concat(),
                QoiError::UnexpectedEof,
            ),
            (
                "no chunk bytes",
                [&header(1, 1, 4, 0)[..], &END_MARKER].concat(),
                QoiError::UnexpectedEof,
            ),
            (
                "missing pixels",
                file(64, 1, &[0xC0]),
                QoiError::UnexpectedEof,
            ),
            (
                "missing end marker",
                valid[..valid.len() - 8].to_vec(),
                QoiError::UnexpectedEof,
            ),
            (
                "short end marker",
                valid[..valid.len() - 1].to_vec(),
                QoiError::UnexpectedEof,
            ),
            ("run overflow", file(1, 1, &[0xC1]), QoiError::RunOverflow),
            (
                "corrupt end marker",
                [&header(1, 1, 4, 0)[..], &[0xC0, 0, 0, 0, 0, 0, 0, 0, 2]].concat(),
                QoiError::InvalidEndMarker,
            ),
            (
                "trailing data",
                [&valid[..], &[0]].concat(),
                QoiError::TrailingData,
            ),
        ];
        for (name, bytes, expected) in cases {
            assert_eq!(decode_qoi(&bytes), Err(expected), "{name}");
        }
        assert!(decode_qoi(&valid).is_ok());
    }
}
