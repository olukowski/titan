//! RIFF/WAVE PCM decoding.

use crate::sound::{AudioError, Channels, Sound};
use alloc::vec::Vec;
use core::fmt;

/// Why [`Sound::from_wav`] rejected its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavError {
    /// A header, chunk, or declared size runs past the end of the input.
    Truncated,
    /// The input does not start with a `RIFF` header of form type `WAVE`.
    NotRiffWave,
    /// No `fmt ` chunk precedes the `data` chunk.
    MissingFmt,
    /// The file has no `data` chunk.
    MissingData,
    /// The `fmt ` chunk is shorter than the 16 bytes of a PCM format header.
    ShortFmt,
    /// The format tag is neither PCM (`1`) nor IEEE float (`3`). This includes
    /// `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`).
    UnsupportedFormat {
        /// The format tag.
        tag: u16,
    },
    /// The bit depth is not 8 or 16 for PCM, or not 32 for IEEE float.
    UnsupportedBitDepth {
        /// The format tag.
        tag: u16,
        /// The bits per sample.
        bits: u16,
    },
    /// The channel count is neither 1 nor 2.
    UnsupportedChannels(u16),
    /// The block align is not `channels * bits / 8`.
    BlockAlignMismatch {
        /// The declared block align.
        block_align: u16,
    },
    /// The `data` chunk is not a whole number of frames.
    UnalignedData {
        /// The `data` chunk length in bytes.
        len: usize,
    },
    /// The decoded samples do not fit in memory.
    TooLarge,
    /// The decoded samples do not form a valid [`Sound`].
    Sound(AudioError),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("WAV data is truncated"),
            Self::NotRiffWave => f.write_str("not a RIFF/WAVE file"),
            Self::MissingFmt => f.write_str("no `fmt ` chunk before the `data` chunk"),
            Self::MissingData => f.write_str("no `data` chunk"),
            Self::ShortFmt => f.write_str("`fmt ` chunk is shorter than 16 bytes"),
            Self::UnsupportedFormat { tag } => write!(f, "unsupported WAV format tag {tag:#06x}"),
            Self::UnsupportedBitDepth { tag, bits } => {
                write!(
                    f,
                    "unsupported {bits}-bit samples for format tag {tag:#06x}"
                )
            }
            Self::UnsupportedChannels(channels) => {
                write!(f, "unsupported channel count {channels}")
            }
            Self::BlockAlignMismatch { block_align } => {
                write!(
                    f,
                    "block align {block_align} does not match the sample format"
                )
            }
            Self::UnalignedData { len } => {
                write!(f, "{len}-byte `data` chunk is not a whole number of frames")
            }
            Self::TooLarge => f.write_str("decoded WAV samples are too large to allocate"),
            Self::Sound(error) => write!(f, "invalid sound: {error}"),
        }
    }
}

impl core::error::Error for WavError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Sound(error) => Some(error),
            _ => None,
        }
    }
}

impl Sound {
    /// Decodes a RIFF/WAVE file holding 8-bit unsigned or 16-bit signed PCM, or 32-bit IEEE
    /// float samples, in mono or stereo.
    ///
    /// Integer samples are scaled to `[-1.0, 1.0)`. Unknown chunks are skipped; the `fmt `
    /// chunk must come before the `data` chunk, and anything after `data` is ignored.
    ///
    /// # Errors
    ///
    /// Returns a [`WavError`] if the input is truncated (including a RIFF or chunk size that
    /// runs past the end of `bytes`), is not RIFF/WAVE, lacks a `fmt ` chunk before its `data`
    /// chunk, uses an unsupported format, decodes to more samples than can be allocated, or
    /// decodes to an invalid [`Sound`] (a zero sample rate or a non-finite float sample).
    pub fn from_wav(bytes: &[u8]) -> Result<Self, WavError> {
        let (&[r0, r1, r2, r3, s0, s1, s2, s3], rest) =
            bytes.split_first_chunk::<8>().ok_or(WavError::Truncated)?;
        let riff_size = u32::from_le_bytes([s0, s1, s2, s3]);
        let body = usize::try_from(riff_size)
            .ok()
            .and_then(|len| rest.get(..len))
            .ok_or(WavError::Truncated)?;
        let (form, mut chunks) = body.split_first_chunk::<4>().ok_or(WavError::Truncated)?;
        if [r0, r1, r2, r3] != *b"RIFF" || form != b"WAVE" {
            return Err(WavError::NotRiffWave);
        }

        let mut fmt = None;
        let data = loop {
            if chunks.is_empty() {
                return Err(WavError::MissingData);
            }
            let (&[i0, i1, i2, i3, s0, s1, s2, s3], rest) =
                chunks.split_first_chunk::<8>().ok_or(WavError::Truncated)?;
            let len = usize::try_from(u32::from_le_bytes([s0, s1, s2, s3]))
                .map_err(|_| WavError::Truncated)?;
            let chunk = rest.get(..len).ok_or(WavError::Truncated)?;
            match &[i0, i1, i2, i3] {
                b"fmt " => fmt = Some(chunk),
                b"data" => break chunk,
                _ => {}
            }
            // Chunks are word-aligned: an odd-sized chunk is followed by one pad byte. `len` is
            // at most `rest.len()`, so adding the pad cannot overflow.
            chunks = rest.get(len + len % 2..).ok_or(WavError::Truncated)?;
        };

        let &[t0, t1, c0, c1, r0, r1, r2, r3, _, _, _, _, a0, a1, b0, b1] = fmt
            .ok_or(WavError::MissingFmt)?
            .first_chunk::<16>()
            .ok_or(WavError::ShortFmt)?;
        let tag = u16::from_le_bytes([t0, t1]);
        let bits = u16::from_le_bytes([b0, b1]);
        let sample_rate = u32::from_le_bytes([r0, r1, r2, r3]);
        let block_align = u16::from_le_bytes([a0, a1]);
        let bytes_per_sample = match (tag, bits) {
            (1, 8) => 1,
            (1, 16) => 2,
            (3, 32) => 4,
            (1 | 3, _) => return Err(WavError::UnsupportedBitDepth { tag, bits }),
            _ => return Err(WavError::UnsupportedFormat { tag }),
        };
        let channels = match u16::from_le_bytes([c0, c1]) {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            other => return Err(WavError::UnsupportedChannels(other)),
        };
        let frame_len = channels.count() * bytes_per_sample;
        if usize::from(block_align) != frame_len {
            return Err(WavError::BlockAlignMismatch { block_align });
        }
        if !data.len().is_multiple_of(frame_len) {
            return Err(WavError::UnalignedData { len: data.len() });
        }

        // Each decoded `f32` can be up to 4x its 8-bit source, which can exceed `isize::MAX`
        // bytes on 32-bit targets.
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(data.len() / bytes_per_sample)
            .map_err(|_| WavError::TooLarge)?;
        match bytes_per_sample {
            1 => samples.extend(data.iter().map(|&b| (f32::from(b) - 128.0) / 128.0)),
            2 => samples.extend(
                data.as_chunks()
                    .0
                    .iter()
                    .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0),
            ),
            _ => samples.extend(data.as_chunks().0.iter().map(|&b| f32::from_le_bytes(b))),
        }
        Self::new(sample_rate, channels, samples).map_err(WavError::Sound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = [id.as_slice(), &(body.len() as u32).to_le_bytes(), body].concat();
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn fmt_body(tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let byte_rate = sample_rate * u32::from(block_align);
        [
            &tag.to_le_bytes()[..],
            &channels.to_le_bytes(),
            &sample_rate.to_le_bytes(),
            &byte_rate.to_le_bytes(),
            &block_align.to_le_bytes(),
            &bits.to_le_bytes(),
        ]
        .concat()
    }

    fn riff(chunks: &[u8]) -> Vec<u8> {
        let size = (4 + chunks.len()) as u32;
        [b"RIFF".as_slice(), &size.to_le_bytes(), b"WAVE", chunks].concat()
    }

    fn wav(tag: u16, channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        riff(
            &[
                chunk(b"fmt ", &fmt_body(tag, channels, 8_000, bits)),
                chunk(b"data", data),
            ]
            .concat(),
        )
    }

    #[test]
    fn decodes_each_sample_format_exactly() {
        let sound = Sound::from_wav(&wav(1, 1, 8, &[0, 128, 255])).unwrap();
        assert_eq!(
            (sound.sample_rate(), sound.channels()),
            (8_000, Channels::Mono)
        );
        assert_eq!(sound.samples(), [-1.0, 0.0, 0.992_187_5]);

        let data = [i16::MIN, 0, i16::MAX, -1].map(i16::to_le_bytes).concat();
        let sound = Sound::from_wav(&wav(1, 2, 16, &data)).unwrap();
        assert_eq!(sound.channels(), Channels::Stereo);
        assert_eq!(
            sound.samples(),
            [-1.0, 0.0, 32767.0 / 32768.0, -1.0 / 32768.0]
        );

        let floats = [0.25_f32, -1.5, f32::MIN_POSITIVE, 0.0];
        let data = floats.map(f32::to_le_bytes).concat();
        let sound = Sound::from_wav(&wav(3, 2, 32, &data)).unwrap();
        assert_eq!(sound.samples(), floats);
    }

    #[test]
    fn skips_unknown_chunks_with_pad_bytes_and_ignores_trailing_chunks() {
        let fmt = chunk(b"fmt ", &[fmt_body(1, 1, 22_050, 8), vec![9, 9]].concat());
        let list = chunk(b"LIST", &[1, 2, 3]);
        assert_eq!(list.len(), 8 + 4, "odd chunk is padded");
        let data = chunk(b"data", &[0, 255, 128]);
        let trailer = chunk(b"junk", &[7; 5]);
        let sound = Sound::from_wav(&riff(&[fmt, list, data, trailer].concat())).unwrap();
        assert_eq!(sound.sample_rate(), 22_050);
        assert_eq!(sound.samples(), [-1.0, 0.992_187_5, 0.0]);
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let full = wav(1, 2, 16, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(Sound::from_wav(&full).is_ok());
        for len in 0..full.len() {
            assert_eq!(
                Sound::from_wav(&full[..len]),
                Err(WavError::Truncated),
                "len {len}"
            );
        }
    }

    #[test]
    fn oversized_chunk_is_truncated_without_allocating() {
        let chunks = [
            chunk(b"fmt ", &fmt_body(1, 1, 8_000, 16)),
            [*b"data", u32::MAX.to_le_bytes()].concat(),
            vec![0; 4],
        ]
        .concat();
        assert_eq!(Sound::from_wav(&riff(&chunks)), Err(WavError::Truncated));
    }

    #[test]
    fn rejects_malformed_and_unsupported_headers() {
        let cases = [
            (
                wav(1, 1, 24, &[0; 3]),
                WavError::UnsupportedBitDepth { tag: 1, bits: 24 },
            ),
            (
                wav(3, 1, 64, &[0; 8]),
                WavError::UnsupportedBitDepth { tag: 3, bits: 64 },
            ),
            (
                wav(0xFFFE, 1, 16, &[0; 2]),
                WavError::UnsupportedFormat { tag: 0xFFFE },
            ),
            (
                wav(2, 1, 4, &[0; 2]),
                WavError::UnsupportedFormat { tag: 2 },
            ),
            (wav(1, 3, 16, &[0; 6]), WavError::UnsupportedChannels(3)),
            (wav(1, 2, 16, &[0; 6]), WavError::UnalignedData { len: 6 }),
            (
                riff(&chunk(b"fmt ", &fmt_body(1, 1, 8_000, 8))),
                WavError::MissingData,
            ),
            (riff(&chunk(b"data", &[0])), WavError::MissingFmt),
            (
                riff(&[chunk(b"fmt ", &[0; 14]), chunk(b"data", &[0])].concat()),
                WavError::ShortFmt,
            ),
        ];
        for (bytes, error) in cases {
            assert_eq!(Sound::from_wav(&bytes), Err(error));
        }

        let mut bytes = wav(1, 2, 16, &[0; 4]);
        bytes[8] = b'X';
        assert_eq!(Sound::from_wav(&bytes), Err(WavError::NotRiffWave));

        // Bytes 32..34 hold the block align: claim 2 for a 4-byte stereo 16-bit frame.
        let mut bytes = wav(1, 2, 16, &[0; 4]);
        bytes[32] = 2;
        assert_eq!(
            Sound::from_wav(&bytes),
            Err(WavError::BlockAlignMismatch { block_align: 2 })
        );
    }

    #[test]
    fn invalid_sounds_surface_audio_errors() {
        let data = [0.0, f32::NAN].map(f32::to_le_bytes).concat();
        assert_eq!(
            Sound::from_wav(&wav(3, 1, 32, &data)),
            Err(WavError::Sound(AudioError::NonFiniteSample { index: 1 }))
        );
        let zero_rate =
            riff(&[chunk(b"fmt ", &fmt_body(1, 1, 0, 8)), chunk(b"data", &[0])].concat());
        assert_eq!(
            Sound::from_wav(&zero_rate),
            Err(WavError::Sound(AudioError::ZeroSampleRate))
        );
    }
}
