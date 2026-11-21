//! Voices and the stereo software mixer.

use crate::sound::{AudioError, Channels, Sound};
use alloc::vec::Vec;
use core::num::NonZeroU32;
use titan_core::math::sqrt;
use titan_core::{Handle, Pool};

/// How [`AudioMixer::play`] starts a voice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayParams {
    /// Linear gain. Negative values become `0.0`; non-finite values mute the voice.
    pub gain: f32,
    /// Stereo position from `-1.0` (left) to `1.0` (right). Out-of-range values are clamped;
    /// NaN becomes `0.0` (center).
    pub pan: f32,
    /// Whether the voice restarts from the beginning instead of finishing.
    pub looping: bool,
}

impl PlayParams {
    /// Returns unit gain, centered, not looping.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gain: 1.0,
            pan: 0.0,
            looping: false,
        }
    }

    /// Returns these parameters with `gain` replaced.
    #[must_use]
    pub const fn with_gain(self, gain: f32) -> Self {
        Self { gain, ..self }
    }

    /// Returns these parameters with `pan` replaced.
    #[must_use]
    pub const fn with_pan(self, pan: f32) -> Self {
        Self { pan, ..self }
    }

    /// Returns these parameters with `looping` replaced.
    #[must_use]
    pub const fn with_looping(self, looping: bool) -> Self {
        Self { looping, ..self }
    }
}

impl Default for PlayParams {
    fn default() -> Self {
        Self::new()
    }
}

/// One playing instance of a [`Sound`], addressed by a `Handle<Voice>` from
/// [`AudioMixer::play`].
#[derive(Debug)]
pub struct Voice {
    sound: Handle<Sound>,
    /// Whole source frames played.
    frame: usize,
    /// Fraction of a source frame past `frame`, in units of `1 / dst_rate`. Integer stepping
    /// keeps durations exact; a float cursor drifts and can overshoot or undershoot the end.
    sub_frame: u32,
    gain: f32,
    pan: f32,
    looping: bool,
}

impl Voice {
    /// Adds this voice into `out` and advances it. Returns `false` once it has finished.
    fn render(&mut self, sound: &Sound, dst_rate: u32, out: &mut [f32]) -> bool {
        let frames = sound.frame_count();
        if frames == 0 {
            return false;
        }
        let (left, right) = match sound.channels() {
            // Equal power: left² + right² = 1 across the whole pan range.
            Channels::Mono => (sqrt(0.5 * (1.0 - self.pan)), sqrt(0.5 * (1.0 + self.pan))),
            // Balance: center leaves both channels untouched.
            Channels::Stereo => ((1.0 - self.pan).min(1.0), (1.0 + self.pan).min(1.0)),
        };
        let (left, right) = (left * self.gain, right * self.gain);
        let channels = sound.channels().count();
        let samples = sound.samples();
        for out_frame in out.as_chunks_mut::<2>().0 {
            let i0 = self.frame;
            let i1 = if self.looping {
                (i0 + 1) % frames
            } else {
                (i0 + 1).min(frames - 1)
            };
            let t = f64::from(self.sub_frame) / f64::from(dst_rate);
            // f64 so `b - a` cannot overflow between opposite extreme f32 samples.
            let lerp = |channel: usize| {
                let a = f64::from(samples[i0 * channels + channel]);
                let b = f64::from(samples[i1 * channels + channel]);
                (a + (b - a) * t) as f32
            };
            out_frame[0] += lerp(0) * left;
            // A mono sound's only channel feeds both sides.
            out_frame[1] += lerp(channels - 1) * right;

            let total = u64::from(self.sub_frame) + u64::from(sound.sample_rate());
            let advance = total / u64::from(dst_rate);
            // The remainder is below `dst_rate`, so it fits in u32.
            self.sub_frame = (total % u64::from(dst_rate)) as u32;
            if self.looping {
                self.frame = (self.frame + (advance % frames as u64) as usize) % frames;
            } else if advance >= (frames - self.frame) as u64 {
                self.frame = frames;
                return false;
            } else {
                // Below `frames - self.frame`, so it fits in usize.
                self.frame += advance as usize;
            }
        }
        true
    }
}

/// Owns sounds and the voices playing them, and mixes those voices into interleaved stereo.
#[derive(Debug)]
pub struct AudioMixer {
    sample_rate: NonZeroU32,
    master_gain: f32,
    sounds: Pool<Sound>,
    voices: Pool<Voice>,
    /// Scratch reused across calls so steady-state mixing never allocates.
    finished_voices: Vec<Handle<Voice>>,
}

impl AudioMixer {
    /// Creates a mixer that outputs `sample_rate` stereo frames per second.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::ZeroSampleRate`] if `sample_rate` is zero.
    pub fn new(sample_rate: u32) -> Result<Self, AudioError> {
        Ok(Self {
            sample_rate: NonZeroU32::new(sample_rate).ok_or(AudioError::ZeroSampleRate)?,
            master_gain: 1.0,
            sounds: Pool::new(),
            voices: Pool::new(),
            finished_voices: Vec::new(),
        })
    }

    /// Returns the output rate in frames per second.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    /// Returns the gain applied to the whole mix.
    #[must_use]
    pub fn master_gain(&self) -> f32 {
        self.master_gain
    }

    /// Sets the gain applied to the whole mix. Negative values become `0.0`; non-finite values
    /// mute the mixer.
    pub fn set_master_gain(&mut self, gain: f32) {
        self.master_gain = sanitize_gain(gain);
    }

    /// Registers `sound` for playback.
    pub fn add_sound(&mut self, sound: Sound) -> Handle<Sound> {
        self.sounds.insert(sound)
    }

    /// Unregisters a sound and stops every voice playing it. Returns `None` for a stale handle.
    pub fn remove_sound(&mut self, handle: Handle<Sound>) -> Option<Sound> {
        let sound = self.sounds.remove(handle)?;
        self.finished_voices.clear();
        let playing = self
            .voices
            .iter()
            .filter(|(_, voice)| voice.sound == handle);
        self.finished_voices.extend(playing.map(|(voice, _)| voice));
        for voice in self.finished_voices.drain(..) {
            self.voices.remove(voice);
        }
        Some(sound)
    }

    /// Returns a registered sound, or `None` for a stale handle.
    #[must_use]
    pub fn sound(&self, handle: Handle<Sound>) -> Option<&Sound> {
        self.sounds.get(handle)
    }

    /// Starts playing `sound` from its first frame. Returns `None` for a stale sound handle.
    pub fn play(&mut self, sound: Handle<Sound>, params: PlayParams) -> Option<Handle<Voice>> {
        self.sounds.contains(sound).then(|| {
            self.voices.insert(Voice {
                sound,
                frame: 0,
                sub_frame: 0,
                gain: sanitize_gain(params.gain),
                pan: sanitize_pan(params.pan),
                looping: params.looping,
            })
        })
    }

    /// Stops a voice. Returns `false` if it had already finished or been stopped.
    pub fn stop(&mut self, voice: Handle<Voice>) -> bool {
        self.voices.remove(voice).is_some()
    }

    /// Stops every voice.
    pub fn stop_all(&mut self) {
        self.voices.clear();
    }

    /// Sets a voice's gain, sanitized like [`PlayParams::gain`]. Returns `false` for a voice
    /// that is no longer playing.
    pub fn set_gain(&mut self, voice: Handle<Voice>, gain: f32) -> bool {
        self.voices
            .get_mut(voice)
            .map(|v| v.gain = sanitize_gain(gain))
            .is_some()
    }

    /// Sets a voice's pan, sanitized like [`PlayParams::pan`]. Returns `false` for a voice that
    /// is no longer playing.
    pub fn set_pan(&mut self, voice: Handle<Voice>, pan: f32) -> bool {
        self.voices
            .get_mut(voice)
            .map(|v| v.pan = sanitize_pan(pan))
            .is_some()
    }

    /// Returns whether a voice is still playing.
    #[must_use]
    pub fn is_playing(&self, voice: Handle<Voice>) -> bool {
        self.voices.contains(voice)
    }

    /// Returns the number of playing voices.
    #[must_use]
    pub fn active_voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Overwrites `out_stereo` (interleaved left, right) with the next frames of every voice,
    /// scaled by the master gain and clamped to `[-1.0, 1.0]`.
    ///
    /// Voices resample linearly from their sound's rate to the mixer's. Finished non-looping
    /// voices are reclaimed, so their handles stop resolving.
    ///
    /// # Panics
    ///
    /// Panics if `out_stereo.len()` is odd, since it cannot hold whole stereo frames.
    pub fn mix(&mut self, out_stereo: &mut [f32]) {
        assert!(
            out_stereo.len().is_multiple_of(2),
            "AudioMixer::mix needs interleaved stereo frames, got odd length {}",
            out_stereo.len()
        );
        out_stereo.fill(0.0);
        if out_stereo.is_empty() {
            return;
        }
        let dst_rate = self.sample_rate.get();
        self.finished_voices.clear();
        for (handle, voice) in self.voices.iter_mut() {
            let sound = self
                .sounds
                .get(voice.sound)
                .expect("remove_sound stops its voices");
            if !voice.render(sound, dst_rate, out_stereo) {
                self.finished_voices.push(handle);
            }
        }
        for voice in self.finished_voices.drain(..) {
            self.voices.remove(voice);
        }
        for sample in out_stereo {
            let scaled = *sample * self.master_gain;
            // Opposing overflowed voices sum to inf - inf = NaN; emit silence, never NaN.
            *sample = if scaled.is_nan() {
                0.0
            } else {
                scaled.clamp(-1.0, 1.0)
            };
        }
    }
}

fn sanitize_gain(gain: f32) -> f32 {
    if gain.is_finite() { gain.max(0.0) } else { 0.0 }
}

fn sanitize_pan(pan: f32) -> f32 {
    if pan.is_nan() {
        0.0
    } else {
        pan.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const LEFT: PlayParams = PlayParams::new().with_pan(-1.0);

    fn mono(rate: u32, samples: &[f32]) -> Sound {
        Sound::new(rate, Channels::Mono, samples.to_vec()).unwrap()
    }

    fn mix_frames(mixer: &mut AudioMixer, frames: usize) -> Vec<f32> {
        let mut out = vec![f32::NAN; frames * 2];
        mixer.mix(&mut out);
        out
    }

    /// Plays `sound` at its own rate and returns the left channel of the first `frames` frames.
    fn left_channel(mixer_rate: u32, sound: Sound, params: PlayParams, frames: usize) -> Vec<f32> {
        let mut mixer = AudioMixer::new(mixer_rate).unwrap();
        let handle = mixer.add_sound(sound);
        mixer.play(handle, params).unwrap();
        mix_frames(&mut mixer, frames)
            .iter()
            .step_by(2)
            .copied()
            .collect()
    }

    fn first_frame(channels: Channels, samples: &[f32], params: PlayParams) -> [f32; 2] {
        let mut mixer = AudioMixer::new(10).unwrap();
        let sound = mixer.add_sound(Sound::new(10, channels, samples.to_vec()).unwrap());
        mixer.play(sound, params).unwrap();
        let out = mix_frames(&mut mixer, 1);
        [out[0], out[1]]
    }

    #[test]
    fn new_rejects_zero_rate() {
        assert_eq!(AudioMixer::new(0).unwrap_err(), AudioError::ZeroSampleRate);
    }

    #[test]
    fn mono_pan_is_equal_power() {
        let at = |pan| first_frame(Channels::Mono, &[1.0], PlayParams::new().with_pan(pan));
        assert_eq!(at(-1.0), [1.0, 0.0]);
        assert_eq!(at(0.0), [sqrt(0.5), sqrt(0.5)]);
        assert_eq!(at(1.0), [0.0, 1.0]);
        assert_eq!(at(0.5), [0.5, sqrt(0.75)]);
        let [l, r] = at(-0.3);
        assert!((l * l + r * r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stereo_pan_is_balance_and_gain_scales() {
        let at = |gain, pan| {
            first_frame(
                Channels::Stereo,
                &[0.5, -0.25],
                PlayParams {
                    gain,
                    pan,
                    looping: false,
                },
            )
        };
        assert_eq!(at(1.0, 0.0), [0.5, -0.25]);
        assert_eq!(at(1.0, -1.0), [0.5, 0.0]);
        assert_eq!(at(1.0, 1.0), [0.0, -0.25]);
        assert_eq!(at(1.0, 0.5), [0.25, -0.25]);
        assert_eq!(at(2.0, 0.0), [1.0, -0.5]);
    }

    #[test]
    fn voices_sum_deterministically_with_master_gain() {
        let run = || {
            let mut mixer = AudioMixer::new(10).unwrap();
            let a = mixer.add_sound(mono(10, &[0.25, 0.5]));
            let b = mixer
                .add_sound(Sound::new(10, Channels::Stereo, vec![0.5, 0.25, 0.5, 0.25]).unwrap());
            mixer.play(a, LEFT).unwrap();
            mixer.play(b, PlayParams::new()).unwrap();
            mixer.set_master_gain(0.5);
            mix_frames(&mut mixer, 3)
        };
        let out = run();
        assert_eq!(out, [0.375, 0.125, 0.5, 0.125, 0.0, 0.0]);
        let bits = |v: &[f32]| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&out), bits(&run()));
    }

    #[test]
    fn non_looping_voice_is_reclaimed_when_it_reaches_the_end() {
        let mut mixer = AudioMixer::new(10).unwrap();
        let sound = mixer.add_sound(mono(10, &[0.25, 0.5, 0.75]));
        let voice = mixer.play(sound, LEFT).unwrap();
        assert_eq!(mix_frames(&mut mixer, 2), [0.25, 0.0, 0.5, 0.0]);
        assert!(mixer.is_playing(voice));
        // The last frame lands exactly at the end of the buffer: the voice still finishes now.
        assert_eq!(mix_frames(&mut mixer, 1), [0.75, 0.0]);
        assert!(!mixer.is_playing(voice));
        assert_eq!(mixer.active_voice_count(), 0);
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.0]);

        // The reclaimed slot is reused, but the old handle stays dead.
        let fresh = mixer.play(sound, LEFT).unwrap();
        assert_ne!(fresh, voice);
        assert!(!mixer.is_playing(voice));
        assert!(!mixer.stop(voice));
        assert!(!mixer.set_gain(voice, 0.5));
        assert!(!mixer.set_pan(voice, 0.5));
        assert_eq!(mix_frames(&mut mixer, 1), [0.25, 0.0]);
    }

    #[test]
    fn looping_wraps_and_interpolates_across_the_seam() {
        let looped = LEFT.with_looping(true);
        assert_eq!(
            left_channel(10, mono(10, &[0.0, 1.0]), looped, 5),
            [0.0, 1.0, 0.0, 1.0, 0.0]
        );
        // At half speed the frame between the last and first samples blends them.
        assert_eq!(
            left_channel(20, mono(10, &[0.0, 1.0]), looped, 5),
            [0.0, 0.5, 1.0, 0.5, 0.0]
        );
        // A step longer than the whole loop still wraps into range.
        assert_eq!(
            left_channel(1, mono(5, &[0.0, 0.5]), looped, 3),
            [0.0, 0.5, 0.0]
        );
    }

    #[test]
    fn empty_sound_is_silent_and_finishes_even_when_looping() {
        let mut mixer = AudioMixer::new(10).unwrap();
        let empty = mixer.add_sound(mono(10, &[]));
        let once = mixer.play(empty, PlayParams::new()).unwrap();
        let looped = mixer
            .play(empty, PlayParams::new().with_looping(true))
            .unwrap();
        mixer.mix(&mut []);
        assert_eq!(mixer.active_voice_count(), 2);
        assert_eq!(mix_frames(&mut mixer, 2), [0.0; 4]);
        assert!(!mixer.is_playing(once) && !mixer.is_playing(looped));
    }

    #[test]
    fn resamples_linearly() {
        // 2x up: in-between frames interpolate; the last frame holds instead of reading past it.
        let up = left_channel(20, mono(10, &[0.0, 1.0]), LEFT, 5);
        assert_eq!(up, [0.0, 0.5, 1.0, 1.0, 0.0]);
        // 2x down: every other source frame.
        let down = left_channel(10, mono(20, &[0.0, 0.25, 0.5, 0.75, 1.0]), LEFT, 4);
        assert_eq!(down, [0.0, 0.5, 1.0, 0.0]);
        // 3:2 steps 1.5 source frames per output frame.
        let ratio = left_channel(20, mono(30, &[0.0, 0.25, 0.5, 0.75]), LEFT, 4);
        assert_eq!(ratio, [0.0, 0.375, 0.75, 0.0]);
    }

    #[test]
    fn resampling_steps_exactly_without_drift() {
        // 160 frames at 48 kHz last exactly 147 frames at 44.1 kHz.
        let mut mixer = AudioMixer::new(44_100).unwrap();
        let clip = mixer.add_sound(mono(48_000, &[0.25; 160]));
        let voice = mixer.play(clip, LEFT).unwrap();
        let played = mix_frames(&mut mixer, 147);
        assert!(played.iter().step_by(2).all(|&s| s == 0.25));
        assert!(!mixer.is_playing(voice));
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.0]);

        // Looping, frame 147 lands exactly on the first sample, not just before it.
        let mut tail = vec![0.0; 160];
        tail[159] = 1.0;
        let looped = left_channel(44_100, mono(48_000, &tail), LEFT.with_looping(true), 148);
        assert_eq!(looped[147], 0.0);
    }

    #[test]
    fn interpolation_does_not_overflow_between_extreme_samples() {
        let extremes = mono(10, &[f32::MAX, -f32::MAX]);
        assert_eq!(left_channel(20, extremes, LEFT, 2), [1.0, 0.0]);
    }

    #[test]
    fn live_voice_controls() {
        let mut mixer = AudioMixer::new(10).unwrap();
        let sound = mixer.add_sound(mono(10, &[0.5; 8]));
        let voice = mixer.play(sound, LEFT).unwrap();
        assert_eq!(mix_frames(&mut mixer, 1), [0.5, 0.0]);
        assert!(mixer.set_pan(voice, 1.0));
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.5]);
        assert!(mixer.set_gain(voice, 0.5));
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.25]);
        assert!(mixer.set_pan(voice, f32::INFINITY) && mixer.set_gain(voice, -3.0));
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.0]);

        let others = [(); 2].map(|()| mixer.play(sound, LEFT).unwrap());
        assert!(mixer.stop(voice));
        assert!(!mixer.stop(voice));
        assert_eq!(mixer.active_voice_count(), 2);
        mixer.stop_all();
        assert_eq!(mixer.active_voice_count(), 0);
        assert!(others.iter().all(|&v| !mixer.is_playing(v)));
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.0]);
    }

    #[test]
    fn remove_sound_stops_its_voices_and_invalidates_its_handle() {
        let mut mixer = AudioMixer::new(10).unwrap();
        let a = mixer.add_sound(mono(10, &[0.25; 4]));
        let b = mixer.add_sound(mono(10, &[0.5; 4]));
        let on_a = [(); 2].map(|()| mixer.play(a, LEFT).unwrap());
        let on_b = mixer.play(b, LEFT).unwrap();

        assert_eq!(mixer.remove_sound(a), Some(mono(10, &[0.25; 4])));
        assert_eq!(mixer.remove_sound(a), None);
        assert_eq!(mixer.active_voice_count(), 1);
        assert!(on_a.iter().all(|&v| !mixer.is_playing(v)));
        assert!(mixer.is_playing(on_b));
        assert_eq!(mix_frames(&mut mixer, 1), [0.5, 0.0]);

        // A new sound reuses `a`'s slot; the old handle must not reach it.
        let c = mixer.add_sound(mono(10, &[1.0]));
        assert_ne!(a, c);
        assert_eq!(mixer.sound(a), None);
        assert_eq!(mixer.play(a, LEFT), None);
        assert_eq!(mixer.sound(c), Some(&mono(10, &[1.0])));
    }

    #[test]
    fn output_is_finite_and_clamped() {
        let mut mixer = AudioMixer::new(10).unwrap();
        let up = mixer.add_sound(mono(10, &[1.0]));
        let down = mixer.add_sound(mono(10, &[-1.0]));
        let huge = mixer.add_sound(mono(10, &[f32::MAX]));
        let neg_huge = mixer.add_sound(mono(10, &[-f32::MAX]));
        let loud = LEFT.with_gain(f32::MAX);
        // A finite sample times a finite gain overflows to +inf on the left; the right is silent.
        mixer.play(huge, loud).unwrap();
        assert_eq!(mix_frames(&mut mixer, 1), [1.0, 0.0]);
        // Opposing overflows give inf - inf = NaN, which must come out as silence.
        mixer.play(huge, loud).unwrap();
        mixer.play(neg_huge, loud).unwrap();
        assert_eq!(mix_frames(&mut mixer, 1), [0.0, 0.0]);
        // Non-finite play parameters are sanitized: infinite gain mutes, NaN pan centers.
        mixer
            .play(up, PlayParams::new().with_gain(f32::INFINITY))
            .unwrap();
        mixer
            .play(down, PlayParams::new().with_pan(f32::NAN))
            .unwrap();
        assert_eq!(mix_frames(&mut mixer, 1), [-sqrt(0.5), -sqrt(0.5)]);

        for (gain, expected) in [
            (f32::INFINITY, 0.0),
            (f32::NAN, 0.0),
            (-1.0, 0.0),
            (2.0, 2.0),
        ] {
            mixer.set_master_gain(gain);
            assert_eq!(mixer.master_gain(), expected);
        }
        mixer.set_master_gain(f32::MAX);
        mixer.play(up, LEFT.with_gain(0.5)).unwrap();
        assert_eq!(mix_frames(&mut mixer, 1), [1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "odd length 3")]
    fn mix_rejects_partial_frames() {
        AudioMixer::new(10).unwrap().mix(&mut [0.0; 3]);
    }
}
