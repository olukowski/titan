//! Titan audio: immutable PCM [`Sound`] clips, WAV decoding, a procedural SFX synth, and a
//! deterministic stereo [`AudioMixer`].
//!
//! The crate is `no_std` (it needs only `alloc`), forbids `unsafe`, and depends only on
//! `titan-core`. Mixing is plain Rust arithmetic in slot order, so every target produces
//! bit-identical output for the same calls.
//!
//! - [`Sound`] validates and stores `f32` samples, mono or interleaved stereo.
//!   [`Sound::from_wav`] decodes 8/16-bit PCM and 32-bit float RIFF/WAVE files.
//! - [`Sfx`] renders a seeded oscillator note with an ADSR [`Envelope`] into a mono [`Sound`],
//!   bit-identically on every target.
//! - [`AudioMixer`] owns registered sounds and the voices playing them, both addressed by
//!   generational [`Handle`]s, and mixes them into an interleaved stereo buffer with linear
//!   resampling, per-voice gain and pan, looping, and a master gain.
//!
//! ```
//! use titan_audio::{AudioMixer, Channels, PlayParams, Sound};
//!
//! let mut mixer = AudioMixer::new(48_000)?;
//! let tone = mixer.add_sound(Sound::new(48_000, Channels::Mono, vec![0.5; 480])?);
//! let voice = mixer.play(tone, PlayParams::new().with_pan(-1.0)).expect("sound is registered");
//!
//! let mut out = [0.0; 2 * 256]; // 256 interleaved stereo frames
//! mixer.mix(&mut out);
//! assert_eq!(out[..2], [0.5, 0.0]); // hard left
//! assert!(mixer.is_playing(voice));
//! # Ok::<(), titan_audio::AudioError>(())
//! ```

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod mixer;
mod sound;
mod synth;
mod wav;

pub use mixer::{AudioMixer, PlayParams, Voice};
pub use sound::{AudioError, Channels, Sound};
pub use synth::{Envelope, Sfx, SynthError, Waveform};
pub use titan_core::Handle;
pub use wav::WavError;
