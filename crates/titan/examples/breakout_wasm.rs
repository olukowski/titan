//! Breakout in the browser through WebGPU.
//!
//! ```sh
//! cargo build --profile wasm --target wasm32-unknown-unknown -p titan --example breakout_wasm
//! cp target/wasm32-unknown-unknown/wasm/examples/breakout_wasm.wasm web/
//! python3 -m http.server -d web   # then open http://localhost:8000/?wasm=./breakout_wasm.wasm
//! ```

#[path = "breakout_game/mod.rs"]
mod breakout;

titan::wasm::export_app!(breakout::BreakoutApp::new());
