//! Deterministic procedural sound effects.

use crate::sound::{AudioError, Channels, Sound};
use alloc::vec::Vec;
use core::f32::consts::TAU;
use core::fmt;
use titan_core::math::sin_cos;

/// The oscillator shape of an [`Sfx`], each spanning `[-1.0, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Waveform {
    /// A sine wave starting at `0.0`.
    Sine,
    /// `1.0` for the first half of each cycle, `-1.0` for the second.
    Square,
    /// A triangle wave starting at `0.0` and rising.
    Triangle,
    /// A ramp from `-1.0` up to `1.0` each cycle.
    Saw,
    /// Seeded random values, a new one at every cycle start (sample and hold), so the
    /// frequency sets the noise's pitch. A new value is drawn at most once per output frame, so
    /// at or above the sample rate this is white noise sampled at the output rate.
    Noise,
}

/// A linear ADSR envelope. Times are in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    /// Time to rise from `0.0` to `1.0`. Finite and non-negative.
    pub attack: f32,
    /// Time to fall from `1.0` to `sustain`. Finite and non-negative.
    pub decay: f32,
    /// Level held until the note ends, in `0.0..=1.0`.
    pub sustain: f32,
    /// Time to fall to `0.0` after the note ends. Finite and non-negative.
    pub release: f32,
}

/// A procedural sound effect: one oscillator note shaped by an [`Envelope`].
///
/// The envelope is piecewise linear in time `t`: `0.0` at `t = 0`, `1.0` at `attack`, `sustain`
/// at `attack + decay`, held until `duration`, then falling linearly to `0.0` over `release`.
/// A note that ends during the attack or decay releases from the level it had reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sfx {
    /// The oscillator shape.
    pub waveform: Waveform,
    /// Start frequency in Hz. Finite and positive.
    pub frequency: f32,
    /// If set, the frequency slides linearly to this value (Hz, finite and positive) over
    /// `duration`, then holds through the release.
    pub end_frequency: Option<f32>,
    /// Note length in seconds before the release starts. Finite and non-negative.
    pub duration: f32,
    /// The amplitude envelope.
    pub envelope: Envelope,
    /// Linear output gain. Finite.
    pub gain: f32,
    /// [`Waveform::Noise`] seed. `0` behaves like `1`.
    pub seed: u32,
}

/// Why [`Sfx::render`] could not produce a [`Sound`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthError {
    /// The named [`Sfx`] or [`Envelope`] field is outside its documented range.
    InvalidParam(&'static str),
    /// The sound has more frames than fit in memory.
    TooLong,
    /// The rendered samples do not form a valid [`Sound`] (a zero sample rate).
    Sound(AudioError),
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParam(name) => write!(f, "synth parameter `{name}` is out of range"),
            Self::TooLong => f.write_str("synth output is too long to allocate"),
            Self::Sound(error) => write!(f, "invalid sound: {error}"),
        }
    }
}

impl core::error::Error for SynthError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Sound(error) => Some(error),
            Self::InvalidParam(_) | Self::TooLong => None,
        }
    }
}

impl Sfx {
    /// Renders this effect as a mono [`Sound`] at `sample_rate` frames per second.
    ///
    /// Frame `n` samples time `n / sample_rate`. The sound has
    /// `ceil((duration + release) * sample_rate) + 1` frames, plus one if rounding would leave the
    /// last frame short of `duration + release`, so its last frame lies at or past the end of
    /// the release and is exactly `0.0`. The output is bit-identical on every target:
    /// rendering uses only IEEE-exact arithmetic and `titan-core`'s portable sine.
    ///
    /// # Errors
    ///
    /// Returns [`SynthError::InvalidParam`] if a field is outside its documented range,
    /// [`SynthError::TooLong`] if the samples cannot be allocated, or [`SynthError::Sound`] if
    /// `sample_rate` is zero.
    pub fn render(&self, sample_rate: u32) -> Result<Sound, SynthError> {
        let Envelope {
            attack,
            decay,
            sustain,
            release,
        } = self.envelope;
        let time = |secs: f32| secs.is_finite() && secs >= 0.0;
        let frequency = |hz: f32| hz.is_finite() && hz > 0.0;
        let checks = [
            ("frequency", frequency(self.frequency)),
            ("end_frequency", self.end_frequency.is_none_or(frequency)),
            ("duration", time(self.duration)),
            ("attack", time(attack)),
            ("decay", time(decay)),
            ("sustain", (0.0..=1.0).contains(&sustain)),
            ("release", time(release)),
            ("gain", self.gain.is_finite()),
        ];
        if let Some(&(name, _)) = checks.iter().find(|(_, ok)| !ok) {
            return Err(SynthError::InvalidParam(name));
        }
        if sample_rate == 0 {
            return Err(SynthError::Sound(AudioError::ZeroSampleRate));
        }

        let [attack, decay, sustain, release, duration, gain, start_hz] = [
            attack,
            decay,
            sustain,
            release,
            self.duration,
            self.gain,
            self.frequency,
        ]
        .map(f64::from);
        let end_hz = self.end_frequency.map_or(start_hz, f64::from);
        let rate = f64::from(sample_rate);
        let end = duration + release;
        let held_level = |t: f64| {
            if t < attack {
                t / attack
            } else if t < attack + decay {
                1.0 - (1.0 - sustain) * (t - attack) / decay
            } else {
                sustain
            }
        };
        let release_level = held_level(duration);
        let level = |t: f64| {
            if t < duration {
                held_level(t)
            } else if t < end {
                // Rounding in `end` can push the ramp a hair below zero.
                (release_level * (1.0 - (t - duration) / release)).max(0.0)
            } else {
                0.0
            }
        };

        let span = end * rate;
        let whole = span as usize;
        let mut frames = whole
            .saturating_add(usize::from((whole as f64) < span))
            .saturating_add(1);
        // `end * rate` can round down onto an integer, leaving the last frame just before `end`.
        if ((frames - 1) as f64 / rate) < end {
            frames = frames.saturating_add(1);
        }
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(frames)
            .map_err(|_| SynthError::TooLong)?;

        // xorshift32; a zero state would stay zero forever.
        let mut state = self.seed.max(1);
        let mut next_noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state) / 2_147_483_648.0 - 1.0
        };
        let mut noise = next_noise();
        let mut phase = 0.0_f64;
        samples.extend((0..frames).map(|n| {
            let t = n as f64 / rate;
            let wave = match self.waveform {
                Waveform::Sine => f64::from(sin_cos(TAU * phase as f32).0),
                Waveform::Square => {
                    if phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Waveform::Triangle => {
                    4.0 * if phase < 0.25 {
                        phase
                    } else if phase < 0.75 {
                        0.5 - phase
                    } else {
                        phase - 1.0
                    }
                }
                Waveform::Saw => 2.0 * phase - 1.0,
                Waveform::Noise => noise,
            };
            let hz = if t < duration {
                start_hz + (end_hz - start_hz) * (t / duration)
            } else {
                end_hz
            };
            let next = phase + hz / rate;
            if next >= 1.0 {
                noise = next_noise();
            }
            // IEEE remainder is exact, so this wraps identically everywhere, and unlike
            // `phase -= 1.0` it also handles steps of a whole cycle or more.
            phase = next % 1.0;
            (wave * level(t) * gain) as f32
        }));
        Sound::new(sample_rate, Channels::Mono, samples).map_err(SynthError::Sound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT: Envelope = Envelope {
        attack: 0.0,
        decay: 0.0,
        sustain: 1.0,
        release: 0.0,
    };

    fn sfx(waveform: Waveform, frequency: f32, duration: f32, envelope: Envelope) -> Sfx {
        Sfx {
            waveform,
            frequency,
            end_frequency: None,
            duration,
            envelope,
            gain: 1.0,
            seed: 7,
        }
    }

    #[test]
    fn envelope_hits_its_breakpoints_exactly() {
        // A 0.25 Hz square stays at +1 for the whole 1.5 s, exposing the envelope directly.
        // Every breakpoint is a whole number of 1/64 s frames.
        let adsr = Envelope {
            attack: 0.25,
            decay: 0.25,
            sustain: 0.5,
            release: 0.5,
        };
        let sound = sfx(Waveform::Square, 0.25, 1.0, adsr).render(64).unwrap();
        assert_eq!(sound.frame_count(), 97); // ceil(1.5 * 64) + 1
        let s = sound.samples();
        let at = [0, 8, 16, 24, 32, 64, 80, 96].map(|n| s[n]);
        assert_eq!(at, [0.0, 0.5, 1.0, 0.75, 0.5, 0.5, 0.25, 0.0]);

        // Ending mid-attack (level 0.5 at 0.125 s) releases from that level.
        let sound = sfx(Waveform::Square, 0.25, 0.125, adsr).render(64).unwrap();
        assert_eq!(sound.frame_count(), 41); // ceil(0.625 * 64) + 1
        let s = sound.samples();
        assert_eq!([s[8], s[24], s[40]], [0.5, 0.25, 0.0]);

        // Non-integral lengths round up, keeping the last frame silent.
        let sound = sfx(Waveform::Square, 1.0, 0.01, FLAT)
            .render(1_000)
            .unwrap();
        assert_eq!(sound.frame_count(), 11);
        assert_eq!(sound.samples().last(), Some(&0.0));

        // Here `end * rate` rounds down to exactly 141.0, so frame 141 lands just before the
        // end of the release and one more frame is needed to reach silence.
        let short = Envelope {
            release: 1.534_819_6e-9,
            ..FLAT
        };
        let sound = sfx(Waveform::Square, 1.0, 0.017_624_998, short)
            .render(8_000)
            .unwrap();
        assert_eq!(sound.frame_count(), 143);
        assert_eq!(sound.samples().last(), Some(&0.0));
    }

    #[test]
    fn slide_reaches_the_end_frequency() {
        // 10 Hz -> 30 Hz over 1 s covers 20 cycles and ends with ~33-frame periods.
        let mut saw = sfx(Waveform::Saw, 10.0, 1.0, FLAT);
        saw.end_frequency = Some(30.0);
        let sound = saw.render(1_000).unwrap();
        let wraps: Vec<usize> = (1..sound.frame_count())
            .filter(|&n| sound.samples()[n] < sound.samples()[n - 1] - 1.0)
            .collect();
        assert!((19..=20).contains(&wraps.len()), "{} cycles", wraps.len());
        let last_period = wraps[wraps.len() - 1] - wraps[wraps.len() - 2];
        assert!(
            (33..=35).contains(&last_period),
            "last period {last_period}"
        );
    }

    #[test]
    fn output_is_deterministic_and_seeded() {
        let noise = sfx(Waveform::Noise, 48_000.0, 0.01, FLAT);
        let a = noise.render(48_000).unwrap();
        assert_eq!(a, noise.render(48_000).unwrap());
        let reseeded = Sfx { seed: 8, ..noise }.render(48_000).unwrap();
        assert_ne!(a.samples(), reseeded.samples());
        let zero = Sfx { seed: 0, ..noise }.render(48_000).unwrap();
        assert!(zero.samples()[..480].iter().all(|&s| s != 0.0));

        // Golden bits: any arithmetic change that could break cross-target determinism fails here.
        let bits = |sound: &Sound| {
            sound.samples()[..4]
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a), [3212822078, 3209161948, 1062128515, 1054601545]);
        let sine = sfx(Waveform::Sine, 440.0, 0.01, FLAT)
            .render(48_000)
            .unwrap();
        assert_eq!(bits(&sine), [0, 1030473793, 1038836779, 1043336751]);
    }

    #[test]
    fn rejects_out_of_range_params() {
        let base = sfx(Waveform::Sine, 440.0, 0.1, FLAT);
        let env = |f: fn(&mut Envelope)| {
            let mut envelope = FLAT;
            f(&mut envelope);
            Sfx { envelope, ..base }
        };
        let cases = [
            (
                Sfx {
                    frequency: 0.0,
                    ..base
                },
                "frequency",
            ),
            (
                Sfx {
                    frequency: f32::NAN,
                    ..base
                },
                "frequency",
            ),
            (
                Sfx {
                    end_frequency: Some(-1.0),
                    ..base
                },
                "end_frequency",
            ),
            (
                Sfx {
                    duration: f32::INFINITY,
                    ..base
                },
                "duration",
            ),
            (env(|e| e.attack = -0.1), "attack"),
            (env(|e| e.decay = f32::NAN), "decay"),
            (env(|e| e.sustain = 1.5), "sustain"),
            (env(|e| e.release = -1.0), "release"),
            (
                Sfx {
                    gain: f32::INFINITY,
                    ..base
                },
                "gain",
            ),
        ];
        for (sfx, name) in cases {
            assert_eq!(sfx.render(48_000), Err(SynthError::InvalidParam(name)));
        }
        assert_eq!(
            base.render(0),
            Err(SynthError::Sound(AudioError::ZeroSampleRate))
        );
        let endless = Sfx {
            duration: f32::MAX,
            ..base
        };
        assert_eq!(endless.render(48_000), Err(SynthError::TooLong));
    }
}
