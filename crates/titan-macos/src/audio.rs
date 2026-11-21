//! Audio output: plays an [`AudioMixer`] through the default output device with an AudioUnit.
//!
//! The unit's render callback runs on CoreAudio's real-time thread and pulls frames from the
//! mixer behind a [`Mutex`]; the game thread locks the same mutex to play, stop, or adjust
//! voices.

use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex, MutexGuard};
use titan_audio::AudioMixer;

type OSStatus = i32;
/// `AudioComponentInstance`, which `AudioUnit` aliases.
type AudioUnit = *mut c_void;
type AudioComponent = *mut c_void;

/// `AURenderCallback` (`AudioToolbox/AUComponent.h`).
type RenderCallback = extern "C" fn(
    ref_con: *mut c_void,
    action_flags: *mut u32,
    time_stamp: *const AudioTimeStamp,
    bus: u32,
    frames: u32,
    data: *mut AudioBufferList,
) -> OSStatus;

const fn four_cc(code: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*code)
}

// `AudioToolbox/AUComponent.h`.
const AUDIO_UNIT_TYPE_OUTPUT: u32 = four_cc(b"auou");
const AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT: u32 = four_cc(b"def ");
#[cfg(test)]
const AUDIO_UNIT_SUB_TYPE_GENERIC_OUTPUT: u32 = four_cc(b"genr");
const AUDIO_UNIT_MANUFACTURER_APPLE: u32 = four_cc(b"appl");
const AUDIO_COMPONENT_ERR_UNSUPPORTED_TYPE: OSStatus = -66_751;
const AUDIO_UNIT_ERR_CANNOT_DO_IN_CURRENT_CONTEXT: OSStatus = -10_863;
// `AudioToolbox/AudioUnitProperties.h`.
const AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
#[cfg(test)]
const AUDIO_UNIT_SCOPE_OUTPUT: u32 = 2;
const AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
const AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;
// `CoreAudioTypes/CoreAudioBaseTypes.h`.
const AUDIO_FORMAT_LINEAR_PCM: u32 = four_cc(b"lpcm");
const AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
#[cfg(test)]
const AUDIO_TIME_STAMP_SAMPLE_TIME_VALID: u32 = 1 << 0;
// `CoreAudio/AudioHardware.h` and `AudioHardwareBase.h`.
const AUDIO_OBJECT_UNKNOWN: u32 = 0;
const AUDIO_OBJECT_SYSTEM_OBJECT: u32 = 1;
const AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL: u32 = four_cc(b"glob");
const AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN: u32 = 0;
const AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE: u32 = four_cc(b"dOut");

#[repr(C)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AuRenderCallbackStruct {
    input_proc: RenderCallback,
    input_proc_ref_con: *mut c_void,
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

/// One buffer is all interleaved audio needs; the C struct is variable length.
#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

#[repr(C)]
#[derive(Default)]
struct AudioTimeStamp {
    sample_time: f64,
    host_time: u64,
    rate_scalar: f64,
    word_clock_time: u64,
    /// `SMPTETime`, which this crate never reads or writes.
    smpte_time: [u32; 6],
    flags: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

const _: () = assert!(size_of::<AudioComponentDescription>() == 20);
const _: () = assert!(size_of::<AudioStreamBasicDescription>() == 40);
const _: () = assert!(size_of::<AuRenderCallbackStruct>() == 16);
const _: () = assert!(size_of::<AudioBufferList>() == 24);
const _: () = assert!(size_of::<AudioTimeStamp>() == 64);
const _: () = assert!(size_of::<AudioObjectPropertyAddress>() == 12);

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioComponentFindNext(
        component: AudioComponent,
        desc: *const AudioComponentDescription,
    ) -> AudioComponent;
    fn AudioComponentInstanceNew(component: AudioComponent, instance: *mut AudioUnit) -> OSStatus;
    fn AudioComponentInstanceDispose(instance: AudioUnit) -> OSStatus;
    fn AudioUnitSetProperty(
        unit: AudioUnit,
        id: u32,
        scope: u32,
        element: u32,
        data: *const c_void,
        data_size: u32,
    ) -> OSStatus;
    fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
    fn AudioUnitUninitialize(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
    #[cfg(test)]
    fn AudioUnitRender(
        unit: AudioUnit,
        action_flags: *mut u32,
        time_stamp: *const AudioTimeStamp,
        output_bus: u32,
        frames: u32,
        data: *mut AudioBufferList,
    ) -> OSStatus;
}

#[link(name = "CoreAudio", kind = "framework")]
unsafe extern "C" {
    fn AudioObjectGetPropertyData(
        object: u32,
        address: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        data_size: *mut u32,
        data: *mut c_void,
    ) -> OSStatus;
}

/// Why [`AudioOutput::start`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioOutputError {
    /// The system has no default output device.
    NoOutputDevice,
    /// A CoreAudio call returned a nonzero `OSStatus`.
    CoreAudio {
        /// The function that failed.
        call: &'static str,
        /// The `OSStatus` it returned.
        status: i32,
    },
}

impl fmt::Display for AudioOutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NoOutputDevice => f.write_str("no default audio output device"),
            Self::CoreAudio { call, status } => {
                write!(f, "{call} failed with OSStatus {status}")?;
                let code = status.to_be_bytes();
                if code.iter().all(|&b| b == b' ' || b.is_ascii_graphic()) {
                    let code: String = code.iter().map(|&b| char::from(b)).collect();
                    write!(f, " ('{code}')")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for AudioOutputError {}

fn check(call: &'static str, status: OSStatus) -> Result<(), AudioOutputError> {
    if status == 0 {
        Ok(())
    } else {
        Err(AudioOutputError::CoreAudio { call, status })
    }
}

/// Plays an [`AudioMixer`] through the default output device until dropped.
///
/// CoreAudio converts the mixer's sample rate to the device's.
#[derive(Debug)]
pub struct AudioOutput {
    unit: AudioUnit,
    /// The render callback's refcon is `Arc::as_ptr` of this. Unlike a `Box`, moving the `Arc`
    /// into `self` does not invalidate that pointer.
    mixer: Arc<Mutex<AudioMixer>>,
}

impl AudioOutput {
    /// Starts playing `mixer` through the default output device.
    ///
    /// # Errors
    ///
    /// Returns [`AudioOutputError::NoOutputDevice`] if the system has no output device, or
    /// [`AudioOutputError::CoreAudio`] naming the first CoreAudio call that failed.
    pub fn start(mixer: AudioMixer) -> Result<Self, AudioOutputError> {
        let address = AudioObjectPropertyAddress {
            selector: AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE,
            scope: AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };
        let mut device = AUDIO_OBJECT_UNKNOWN;
        let mut size = size_of::<u32>() as u32;
        // SAFETY: the property is an `AudioObjectID` (`u32`), and `size` says `device` holds one.
        let status = unsafe {
            AudioObjectGetPropertyData(
                AUDIO_OBJECT_SYSTEM_OBJECT,
                &raw const address,
                0,
                null(),
                &raw mut size,
                (&raw mut device).cast(),
            )
        };
        check("AudioObjectGetPropertyData", status)?;
        if device == AUDIO_OBJECT_UNKNOWN {
            return Err(AudioOutputError::NoOutputDevice);
        }
        Self::open(
            AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT,
            &[AUDIO_UNIT_SCOPE_INPUT],
            mixer,
        )
    }

    /// Locks the mixer so the game thread can play, stop, or adjust voices. The render thread
    /// waits while the guard is held, so hold it briefly.
    ///
    /// # Panics
    ///
    /// Panics if an earlier panic poisoned the lock while a guard was held.
    pub fn mixer(&self) -> MutexGuard<'_, AudioMixer> {
        self.mixer
            .lock()
            .expect("audio mixer lock poisoned by an earlier panic")
    }

    /// Opens, configures, initializes, and starts an Apple output unit of `sub_type` that pulls
    /// interleaved stereo `f32` from `mixer`, with that format set on bus 0 of each scope in
    /// `format_scopes`.
    fn open(
        sub_type: u32,
        format_scopes: &[u32],
        mixer: AudioMixer,
    ) -> Result<Self, AudioOutputError> {
        let desc = AudioComponentDescription {
            component_type: AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: sub_type,
            component_manufacturer: AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0,
            component_flags_mask: 0,
        };
        // SAFETY: a null component starts the search; `desc` is a valid description.
        let component = unsafe { AudioComponentFindNext(null_mut(), &raw const desc) };
        if component.is_null() {
            // No status comes back; this is the closest documented one.
            return Err(AudioOutputError::CoreAudio {
                call: "AudioComponentFindNext",
                status: AUDIO_COMPONENT_ERR_UNSUPPORTED_TYPE,
            });
        }
        let mut unit = null_mut();
        // SAFETY: `component` came from `AudioComponentFindNext`; `unit` receives the instance.
        let status = unsafe { AudioComponentInstanceNew(component, &raw mut unit) };
        check("AudioComponentInstanceNew", status)?;
        let sample_rate = mixer.sample_rate();
        let mixer = Arc::new(Mutex::new(mixer));
        // SAFETY: `unit` is a fresh instance. On success it moves into `Self` with `mixer`, whose
        // `Drop` disposes the unit before the mixer is freed.
        let configured =
            unsafe { configure(unit, format_scopes, sample_rate, Arc::as_ptr(&mixer)) };
        if let Err(error) = configured {
            // SAFETY: `configure` leaves the unit uninitialized on failure; nothing else holds it.
            if unsafe { AudioComponentInstanceDispose(unit) } != 0 {
                // A callback may be registered, and a failed dispose does not promise it stopped.
                std::mem::forget(Arc::clone(&mixer));
            }
            return Err(error);
        }
        Ok(Self { unit, mixer })
    }
}

/// Sets the stream format (at `sample_rate`) and render callback on `unit`, then initializes
/// and starts it. On failure the unit is left uninitialized.
///
/// # Safety
///
/// `unit` must be a live, uninitialized output unit, and `mixer` must stay valid as long as it.
unsafe fn configure(
    unit: AudioUnit,
    format_scopes: &[u32],
    sample_rate: u32,
    mixer: *const Mutex<AudioMixer>,
) -> Result<(), AudioOutputError> {
    let format = AudioStreamBasicDescription {
        sample_rate: f64::from(sample_rate),
        format_id: AUDIO_FORMAT_LINEAR_PCM,
        format_flags: AUDIO_FORMAT_FLAG_IS_FLOAT | AUDIO_FORMAT_FLAG_IS_PACKED,
        bytes_per_packet: 8,
        frames_per_packet: 1,
        bytes_per_frame: 8,
        channels_per_frame: 2,
        bits_per_channel: 32,
        reserved: 0,
    };
    for &scope in format_scopes {
        // SAFETY: the stream format property takes an `AudioStreamBasicDescription`.
        let status = unsafe {
            AudioUnitSetProperty(
                unit,
                AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
                scope,
                0,
                (&raw const format).cast(),
                size_of::<AudioStreamBasicDescription>() as u32,
            )
        };
        check("AudioUnitSetProperty(StreamFormat)", status)?;
    }
    let callback = AuRenderCallbackStruct {
        input_proc: render,
        input_proc_ref_con: mixer.cast_mut().cast(),
    };
    // SAFETY: the property takes an `AURenderCallbackStruct`, copied by the unit. The refcon is
    // `mixer`, which the caller keeps alive as long as the unit.
    let status = unsafe {
        AudioUnitSetProperty(
            unit,
            AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            AUDIO_UNIT_SCOPE_INPUT,
            0,
            (&raw const callback).cast(),
            size_of::<AuRenderCallbackStruct>() as u32,
        )
    };
    check("AudioUnitSetProperty(SetRenderCallback)", status)?;
    // SAFETY: `unit` is live and configured.
    check("AudioUnitInitialize", unsafe { AudioUnitInitialize(unit) })?;
    // SAFETY: `unit` is an initialized output unit.
    let started = check("AudioOutputUnitStart", unsafe {
        AudioOutputUnitStart(unit)
    });
    if started.is_err() {
        // SAFETY: `unit` is initialized and not running.
        unsafe { AudioUnitUninitialize(unit) };
    }
    started
}

/// The `AURenderCallback`: fills the unit's interleaved buffer from the mixer.
extern "C" fn render(
    ref_con: *mut c_void,
    _action_flags: *mut u32,
    _time_stamp: *const AudioTimeStamp,
    _bus: u32,
    _frames: u32,
    data: *mut AudioBufferList,
) -> OSStatus {
    // SAFETY: the refcon is the `Arc`'s `Mutex<AudioMixer>` set in `configure`, which
    // `AudioOutput` keeps alive while the unit is.
    let mixer = unsafe { &*ref_con.cast_const().cast::<Mutex<AudioMixer>>() };
    // SAFETY: the unit passes one buffer of the interleaved format set in `configure`, which
    // holds `data_byte_size` bytes of `f32`s and is ours until we return. CoreAudio does not
    // promise those bytes are initialized, so they are zeroed before the slice is formed.
    let out = unsafe {
        let buffer = &(*data).buffers[0];
        let samples = buffer.data.cast::<f32>();
        let len = buffer.data_byte_size as usize / size_of::<f32>();
        samples.write_bytes(0, len);
        std::slice::from_raw_parts_mut(samples, len)
    };
    if let Ok(mut mixer) = mixer.lock() {
        mixer.mix(out);
        0
    } else {
        AUDIO_UNIT_ERR_CANNOT_DO_IN_CURRENT_CONTEXT
    }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        // SAFETY: `self.unit` is initialized and started. A successful dispose guarantees no
        // further render callbacks, so `self.mixer`, dropped after this returns, is never used
        // after free.
        let statuses = unsafe {
            [
                AudioOutputUnitStop(self.unit),
                AudioUnitUninitialize(self.unit),
                AudioComponentInstanceDispose(self.unit),
            ]
        };
        if statuses[2] != 0 {
            // A failed dispose does not promise the callback stopped, so the mixer must outlive it.
            std::mem::forget(Arc::clone(&self.mixer));
        }
        debug_assert_eq!(statuses, [0; 3], "AudioOutput teardown failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_audio::{Channels, PlayParams, Sound};

    const RATE: u32 = 48_000;
    const BLOCK_FRAMES: usize = 256;

    /// Opens a generic output unit, which renders only when pulled, with no conversion between
    /// the mixer's format and the buffers `render_block` passes.
    fn open_generic(mixer: AudioMixer) -> AudioOutput {
        AudioOutput::open(
            AUDIO_UNIT_SUB_TYPE_GENERIC_OUTPUT,
            &[AUDIO_UNIT_SCOPE_INPUT, AUDIO_UNIT_SCOPE_OUTPUT],
            mixer,
        )
        .expect("generic output unit opens")
    }

    /// Pulls the next `out.len() / 2` frames through the unit, as the hardware would.
    fn render_block(output: &AudioOutput, sample_time: f64, out: &mut [f32]) {
        let mut list = AudioBufferList {
            number_buffers: 1,
            buffers: [AudioBuffer {
                number_channels: 2,
                data_byte_size: u32::try_from(size_of_val(out)).unwrap(),
                data: out.as_mut_ptr().cast(),
            }],
        };
        let time_stamp = AudioTimeStamp {
            sample_time,
            flags: AUDIO_TIME_STAMP_SAMPLE_TIME_VALID,
            ..AudioTimeStamp::default()
        };
        let mut flags = 0;
        // SAFETY: `list` describes `out`, which outlives the call, in the unit's output format.
        let status = unsafe {
            AudioUnitRender(
                output.unit,
                &raw mut flags,
                &raw const time_stamp,
                0,
                u32::try_from(out.len() / 2).unwrap(),
                &raw mut list,
            )
        };
        assert_eq!(status, 0, "AudioUnitRender");
    }

    /// A mixer with a mono ramp (every frame differs) playing at a non-native rate, so the
    /// output exercises resampling.
    fn ramp_mixer() -> (AudioMixer, titan_audio::Handle<titan_audio::Voice>) {
        let mut mixer = AudioMixer::new(RATE).unwrap();
        let ramp = (0..4_000).map(|i| i as f32 / 4_000.0 - 0.5).collect();
        let sound = mixer.add_sound(Sound::new(44_100, Channels::Mono, ramp).unwrap());
        let voice = mixer
            .play(sound, PlayParams::new().with_pan(0.3).with_looping(true))
            .unwrap();
        (mixer, voice)
    }

    #[test]
    fn renders_bit_exact_mixer_output_through_an_audio_unit() {
        let (mixer, voice) = ramp_mixer();
        let (mut expected_mixer, expected_voice) = ramp_mixer();
        let output = open_generic(mixer);

        let mut rendered = vec![0.0; 2 * BLOCK_FRAMES];
        let mut expected = vec![0.0; 2 * BLOCK_FRAMES];
        for block in 0..6 {
            match block {
                2 => {
                    assert!(output.mixer().set_gain(voice, 0.25));
                    assert!(expected_mixer.set_gain(expected_voice, 0.25));
                }
                4 => {
                    assert!(output.mixer().stop(voice));
                    assert!(expected_mixer.stop(expected_voice));
                }
                _ => {}
            }
            render_block(&output, (block * BLOCK_FRAMES) as f64, &mut rendered);
            expected_mixer.mix(&mut expected);
            let bits = |samples: &[f32]| samples.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&rendered), bits(&expected), "block {block}");
            if block < 4 {
                assert!(
                    rendered.windows(4).any(|w| w[0] != w[2]),
                    "block {block} is not a ramp"
                );
            } else {
                assert!(
                    rendered.iter().all(|&s| s == 0.0),
                    "block {block} after stop"
                );
            }
        }
    }

    #[test]
    fn units_tear_down_cleanly_while_voices_play() {
        for _ in 0..8 {
            let (mixer, voice) = ramp_mixer();
            let output = open_generic(mixer);
            let mut block = vec![0.0; 2 * BLOCK_FRAMES];
            render_block(&output, 0.0, &mut block);
            assert!(output.mixer().is_playing(voice));
            drop(output);
        }
    }

    #[test]
    fn error_display_names_the_call_and_four_char_codes() {
        let fmt = AudioOutputError::CoreAudio {
            call: "AudioUnitInitialize",
            status: i32::from_be_bytes(*b"fmt?"),
        };
        assert_eq!(
            fmt.to_string(),
            "AudioUnitInitialize failed with OSStatus 1718449215 ('fmt?')"
        );
        let numeric = AudioOutputError::CoreAudio {
            call: "AudioOutputUnitStart",
            status: AUDIO_UNIT_ERR_CANNOT_DO_IN_CURRENT_CONTEXT,
        };
        assert_eq!(
            numeric.to_string(),
            "AudioOutputUnitStart failed with OSStatus -10863"
        );
    }
}
