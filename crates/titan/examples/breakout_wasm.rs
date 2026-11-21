//! Breakout in the browser through WebGPU, with sound once the page gets its first key press or
//! click (browsers keep audio suspended until then).
//!
//! ```sh
//! cargo build --profile wasm --target wasm32-unknown-unknown -p titan --example breakout_wasm
//! cp target/wasm32-unknown-unknown/wasm/examples/breakout_wasm.wasm web/
//! python3 -m http.server -d web   # then open http://localhost:8000/?wasm=./breakout_wasm.wasm
//! ```

// Other targets only type-check this example, so its sound bank goes unused there.
#[cfg_attr(not(target_arch = "wasm32"), expect(dead_code))]
#[path = "breakout_game/mod.rs"]
mod breakout;

titan::wasm::export_app!(breakout::BreakoutApp::new().with_speaker(speaker()));

/// Starts the sound bank through Web Audio. A page without it can still play, so the game
/// reports the failure once and runs muted.
#[cfg(target_arch = "wasm32")]
fn speaker() -> Option<breakout::Speaker> {
    use titan::wasm::HostBridge;
    let mut mixer = titan::audio::AudioMixer::new(breakout::SAMPLE_RATE).expect("48 kHz is valid");
    let sounds = breakout::Sounds::new(&mut mixer);
    match titan::wasm::AudioOutput::start(mixer) {
        Ok(output) => Some(Box::new(move |event| {
            sounds.play(&mut output.mixer(), event);
        })),
        Err(error) => {
            let message = format!("breakout: audio unavailable, playing muted: {error}");
            titan::wasm::WasmBridge.log(titan::wasm::abi::LOG_WARN, &message);
            None
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn speaker() -> Option<breakout::Speaker> {
    None
}
