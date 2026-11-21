//! Plays through the real default output device, when the machine has one.

#![cfg(target_os = "macos")]

use std::thread;
use std::time::{Duration, Instant};
use titan_audio::{AudioMixer, Channels, PlayParams, Sound};
use titan_macos::{AudioOutput, AudioOutputError};

#[test]
fn plays_a_sound_to_completion_on_the_default_output_device() {
    let mut mixer = AudioMixer::new(48_000).unwrap();
    // 50 ms of a quiet square wave.
    let tone = (0..2_400)
        .map(|i| if i % 100 < 50 { 0.05 } else { -0.05 })
        .collect();
    let sound = mixer.add_sound(Sound::new(48_000, Channels::Mono, tone).unwrap());
    let voice = mixer.play(sound, PlayParams::new()).unwrap();
    let looping = mixer
        .play(sound, PlayParams::new().with_looping(true))
        .unwrap();

    let output = match AudioOutput::start(mixer) {
        Ok(output) => output,
        Err(error) => {
            assert_eq!(error, AudioOutputError::NoOutputDevice);
            eprintln!(
                "NO AUDIO OUTPUT DEVICE: AudioOutput::start on a real device was NOT exercised"
            );
            return;
        }
    };
    assert!(output.mixer().set_gain(looping, 0.5));

    // Only the render thread advances voices, so the one-shot finishing proves it is pulling.
    let deadline = Instant::now() + Duration::from_secs(2);
    while output.mixer().is_playing(voice) {
        assert!(
            Instant::now() < deadline,
            "the device never rendered the sound"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert!(output.mixer().stop(looping));
    drop(output);
}
