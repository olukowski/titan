//! Audio output: `AudioOutput` plays an [`AudioMixer`] through Web Audio. `web/titan.js` pulls
//! interleaved stereo frames through the `titan_audio_mix` export and streams them to an
//! AudioWorklet.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use titan_audio::AudioMixer;

/// The most frames one `titan_audio_mix` call may ask for. The count comes from the host, so a
/// larger one is rejected rather than allocated.
pub(crate) const MAX_MIX_FRAMES: u32 = 1 << 16;

/// The mixer an `AudioOutput` plays, and the buffer `titan_audio_mix` mixes it into.
pub(crate) struct AudioStream {
    mixer: Rc<RefCell<AudioMixer>>,
    /// Reused across calls so steady-state mixing never allocates.
    scratch: Vec<f32>,
}

impl AudioStream {
    pub(crate) fn new(mixer: Rc<RefCell<AudioMixer>>) -> Self {
        Self {
            mixer,
            scratch: Vec::new(),
        }
    }

    /// Mixes the next `frames` stereo frames, interleaved, or returns `None` if `frames` is over
    /// [`MAX_MIX_FRAMES`].
    pub(crate) fn mix(&mut self, frames: u32) -> Option<&[f32]> {
        if frames > MAX_MIX_FRAMES {
            return None;
        }
        self.scratch.resize(2 * frames as usize, 0.0);
        self.mixer.borrow_mut().mix(&mut self.scratch);
        Some(&self.scratch)
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::MIX;
#[cfg(target_arch = "wasm32")]
pub use wasm::{AudioOutput, AudioOutputError};

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::AudioStream;
    use crate::bridge::{host_error, imports};
    use alloc::boxed::Box;
    use alloc::rc::Rc;
    use alloc::string::String;
    use core::cell::{RefCell, RefMut};
    use core::fmt;
    use titan_audio::AudioMixer;

    /// Mixes the next frames of the live output and returns a pointer to them; see
    /// [`AudioStream::mix`].
    type MixFn = Box<dyn FnMut(u32) -> Option<*const f32>>;

    /// What `titan_audio_mix` calls while an [`AudioOutput`] is live. A closure that only
    /// [`AudioOutput::start`] creates, rather than an `AudioStream`, so LTO drops the mixer from
    /// apps that never start audio.
    pub(crate) struct MixSlot(pub(crate) RefCell<Option<MixFn>>);

    // SAFETY: `export::glue` rejects the `atomics` target feature, so a wasm32 module runs on one
    // thread and the cell is never accessed concurrently. The host calls one export at a time
    // and the `audio_*` imports never call back into one, so the borrows never overlap either.
    unsafe impl Sync for MixSlot {}

    pub(crate) static MIX: MixSlot = MixSlot(RefCell::new(None));

    /// Why [`AudioOutput::start`] failed.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum AudioOutputError {
        /// Another `AudioOutput` is still playing; drop it first.
        AlreadyStarted,
        /// The host could not start Web Audio, for example because the browser has none or does
        /// not support the mixer's sample rate. Holds the host's message.
        Host(String),
    }

    impl fmt::Display for AudioOutputError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::AlreadyStarted => f.write_str("an AudioOutput is already playing"),
                Self::Host(message) => write!(f, "could not start Web Audio: {message}"),
            }
        }
    }

    impl core::error::Error for AudioOutputError {}

    /// Plays an [`AudioMixer`] through Web Audio until dropped.
    ///
    /// The browser keeps audio suspended until the user first presses a key or pointer button on
    /// the page; the mixer does not advance while it is.
    #[derive(Debug)]
    pub struct AudioOutput {
        mixer: Rc<RefCell<AudioMixer>>,
    }

    impl AudioOutput {
        /// Starts playing `mixer` at its sample rate; the browser resamples to the device.
        ///
        /// # Errors
        ///
        /// Returns [`AudioOutputError::AlreadyStarted`] while another `AudioOutput` is live, or
        /// [`AudioOutputError::Host`] if the host could not start Web Audio.
        pub fn start(mixer: AudioMixer) -> Result<Self, AudioOutputError> {
            if MIX.0.borrow().is_some() {
                return Err(AudioOutputError::AlreadyStarted);
            }
            let sample_rate = mixer.sample_rate();
            if let Some(message) = host_error(imports::audio_start(sample_rate)) {
                return Err(AudioOutputError::Host(message));
            }
            let mixer = Rc::new(RefCell::new(mixer));
            let mut stream = AudioStream::new(Rc::clone(&mixer));
            *MIX.0.borrow_mut() = Some(Box::new(move |frames| {
                stream.mix(frames).map(<[f32]>::as_ptr)
            }));
            Ok(Self { mixer })
        }

        /// Borrows the mixer to play, stop, or adjust voices.
        ///
        /// # Panics
        ///
        /// Panics if the mixer is already borrowed, such as by an earlier guard still alive.
        #[must_use]
        pub fn mixer(&self) -> RefMut<'_, AudioMixer> {
            self.mixer.borrow_mut()
        }
    }

    impl Drop for AudioOutput {
        fn drop(&mut self) {
            *MIX.0.borrow_mut() = None;
            imports::audio_stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_audio::{Channels, PlayParams, Sound};

    /// A mixer playing a looping ramp at a rate it must resample, so a restarted voice or a
    /// skipped frame changes the output.
    fn playing_mixer() -> AudioMixer {
        let mut mixer = AudioMixer::new(48_000).unwrap();
        let ramp = (0..100).map(|i| i as f32 / 100.0).collect();
        let sound = mixer.add_sound(Sound::new(22_050, Channels::Mono, ramp).unwrap());
        let params = PlayParams::new().with_pan(0.3).with_looping(true);
        mixer.play(sound, params).unwrap();
        mixer
    }

    #[test]
    fn mix_matches_the_mixer_and_continues_its_voices() {
        let mut stream = AudioStream::new(Rc::new(RefCell::new(playing_mixer())));
        let mut reference = playing_mixer();

        let mut streamed = Vec::new();
        for frames in [300, 0, 57, 1000] {
            let mixed = stream.mix(frames).unwrap();
            assert_eq!(mixed.len(), 2 * frames as usize);
            streamed.extend_from_slice(mixed);
        }

        let mut expected = alloc::vec![0.0; 2 * 1357];
        reference.mix(&mut expected);
        assert!(expected.iter().any(|&s| s != 0.0));
        assert_eq!(streamed, expected);
    }

    #[test]
    fn mix_reuses_its_buffer() {
        let mut stream = AudioStream::new(Rc::new(RefCell::new(playing_mixer())));
        let first = stream.mix(512).unwrap().as_ptr();
        assert_eq!(stream.mix(512).unwrap().as_ptr(), first);
        assert_eq!(stream.mix(128).unwrap().as_ptr(), first);
    }

    #[test]
    fn mix_rejects_more_than_max_frames_without_advancing() {
        let mut stream = AudioStream::new(Rc::new(RefCell::new(playing_mixer())));
        assert_eq!(stream.mix(MAX_MIX_FRAMES + 1), None);
        assert_eq!(stream.mix(u32::MAX), None);

        let mut expected = alloc::vec![0.0; 2 * 64];
        playing_mixer().mix(&mut expected);
        assert_eq!(stream.mix(64).unwrap(), expected);
        assert_eq!(
            stream.mix(MAX_MIX_FRAMES).unwrap().len(),
            2 * MAX_MIX_FRAMES as usize
        );
    }
}
