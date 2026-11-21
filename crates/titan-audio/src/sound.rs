//! Validated PCM clips.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU32;

/// The channel layout of a [`Sound`]'s samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channels {
    /// One sample per frame.
    Mono = 1,
    /// Two interleaved samples per frame: left, then right.
    Stereo = 2,
}

impl Channels {
    /// Returns the number of samples per frame.
    #[must_use]
    pub const fn count(self) -> usize {
        self as usize
    }
}

/// Why a [`Sound`] or [`AudioMixer`](crate::AudioMixer) could not be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioError {
    /// The sample rate was zero.
    ZeroSampleRate,
    /// The sample count is not a whole number of frames.
    UnalignedSamples {
        /// The number of samples given.
        len: usize,
        /// The channel layout the samples were meant to have.
        channels: Channels,
    },
    /// A sample was NaN or infinite.
    NonFiniteSample {
        /// The index of the first non-finite sample.
        index: usize,
    },
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleRate => f.write_str("sample rate must be non-zero"),
            Self::UnalignedSamples { len, channels } => write!(
                f,
                "{len} samples is not a whole number of {}-channel frames",
                channels.count()
            ),
            Self::NonFiniteSample { index } => write!(f, "sample {index} is not finite"),
        }
    }
}

impl core::error::Error for AudioError {}

/// An immutable PCM clip: finite `f32` samples, mono or interleaved stereo, at a fixed rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Sound {
    sample_rate: NonZeroU32,
    channels: Channels,
    samples: Box<[f32]>,
}

impl Sound {
    /// Creates a sound from interleaved samples played at `sample_rate` frames per second.
    ///
    /// # Errors
    ///
    /// Returns an [`AudioError`] if `sample_rate` is zero, `samples` is not a whole number of
    /// frames, or any sample is NaN or infinite. An empty sound (zero frames) is valid.
    pub fn new(
        sample_rate: u32,
        channels: Channels,
        samples: Vec<f32>,
    ) -> Result<Self, AudioError> {
        let sample_rate = NonZeroU32::new(sample_rate).ok_or(AudioError::ZeroSampleRate)?;
        if !samples.len().is_multiple_of(channels.count()) {
            return Err(AudioError::UnalignedSamples {
                len: samples.len(),
                channels,
            });
        }
        if let Some(index) = samples.iter().position(|s| !s.is_finite()) {
            return Err(AudioError::NonFiniteSample { index });
        }
        Ok(Self {
            sample_rate,
            channels,
            samples: samples.into_boxed_slice(),
        })
    }

    /// Returns the playback rate in frames per second.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    /// Returns the channel layout.
    #[must_use]
    pub fn channels(&self) -> Channels {
        self.channels
    }

    /// Returns the interleaved samples.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Returns the number of frames (samples per channel).
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels.count()
    }

    /// Returns the length in seconds at the sound's own sample rate.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.frame_count() as f64 / f64::from(self.sample_rate.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn new_rejects_invalid_input() {
        assert_eq!(
            Sound::new(0, Channels::Mono, vec![0.0]),
            Err(AudioError::ZeroSampleRate)
        );
        assert_eq!(
            Sound::new(44_100, Channels::Stereo, vec![0.0; 3]),
            Err(AudioError::UnalignedSamples {
                len: 3,
                channels: Channels::Stereo
            })
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                Sound::new(44_100, Channels::Mono, vec![0.0, 1.0, bad, bad]),
                Err(AudioError::NonFiniteSample { index: 2 })
            );
        }
    }

    #[test]
    fn accessors_describe_frames_not_samples() {
        let sound = Sound::new(2, Channels::Stereo, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(sound.sample_rate(), 2);
        assert_eq!(sound.channels(), Channels::Stereo);
        assert_eq!(sound.samples(), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(sound.frame_count(), 2);
        assert_eq!(sound.duration_secs(), 1.0);

        let empty = Sound::new(48_000, Channels::Stereo, Vec::new()).unwrap();
        assert_eq!((empty.frame_count(), empty.duration_secs()), (0, 0.0));
    }
}
