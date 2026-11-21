//! Titan's browser backend: WebGPU and Web Audio through a direct WASM/JS bridge, with no
//! bindings generator.
//!
//! The crate is `no_std` (it needs only `alloc`) and depends only on Titan's own crates. Rust
//! owns all validation and state; `web/titan.js` owns the WebGPU objects, the canvas, and the
//! `AudioContext`.
//!
//! - [`WebGpuDevice`]: a [`GpuDevice`](titan_gpu::GpuDevice) that checks every call against
//!   `titan-gpu`'s rules, then forwards it to a [`HostBridge`] as ids and byte payloads.
//! - [`abi`]: the byte layouts of those payloads, which `titan.js` decodes.
//! - [`WebRunner`]: the frame loop around an [`App`](titan_app::App), fed by browser events.
//! - [`export_app!`]: the WASM exports `titan.js` calls, for a `cdylib` crate. On `wasm32`,
//!   `WasmBridge` implements [`HostBridge`] with the `titan` import module.
//! - `AudioOutput` (`wasm32` only): plays a `titan_audio::AudioMixer` through Web Audio until
//!   dropped, like `titan-macos`'s. The host pulls mixed frames through `titan_audio_mix`.

#![no_std]

extern crate alloc;

pub mod abi;
#[cfg(any(target_arch = "wasm32", test))]
mod audio;
mod bridge;
mod export;
mod gpu;
mod runner;

#[cfg(target_arch = "wasm32")]
pub use audio::{AudioOutput, AudioOutputError};
pub use bridge::HostBridge;
#[cfg(target_arch = "wasm32")]
pub use bridge::WasmBridge;
pub use gpu::WebGpuDevice;
pub use runner::{WebConfig, WebRunner};

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub use export::glue as __private;
#[doc(hidden)]
pub use titan_app::App as __App;
