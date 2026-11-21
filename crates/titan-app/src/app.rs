//! The [`App`] trait that games implement and platform runners drive.

use crate::input::InputState;
use crate::window::{AppAction, SurfaceSize};
use titan_gpu::{FrameCommands, GpuDevice, GpuError};

/// A game or tool, driven by a runner: a platform backend or [`HeadlessRunner`](crate::HeadlessRunner).
///
/// Each frame the runner applies input, runs zero or more fixed [`step`](Self::step)s, then calls
/// [`render`](Self::render) once. Simulation lives in `step`, which sees no GPU, so the same
/// input always produces the same state on every platform. Rendering may read that state but
/// should not change it.
pub trait App {
    /// Creates GPU resources once, before the first frame.
    ///
    /// # Errors
    ///
    /// Returns any error from `gpu`; the runner passes it on to its caller.
    fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
        let _ = gpu;
        Ok(())
    }

    /// Reacts to a surface size change, for example by recreating size-dependent textures. It
    /// runs before the frame's steps.
    ///
    /// Every runner also calls it once with the initial surface size right after
    /// [`init`](Self::init), before the first frame, so size-dependent resources exist when the
    /// first frame renders.
    ///
    /// # Errors
    ///
    /// Returns any error from `gpu`; the runner passes it on to its caller.
    fn resize(&mut self, gpu: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
        let _ = (gpu, size);
        Ok(())
    }

    /// Advances the simulation by `dt` seconds, the runner's fixed step.
    fn step(&mut self, input: &InputState, dt: f32) -> AppAction;

    /// Records the frame into `commands`, which start empty. `alpha` in `[0, 1)` is how far the
    /// clock is between the last step and the next, for interpolating motion. `gpu` may be used
    /// to upload per-frame data; the runner submits `commands` afterwards.
    ///
    /// # Errors
    ///
    /// Returns any error from `gpu`; the runner passes it on to its caller.
    fn render(
        &mut self,
        gpu: &mut dyn GpuDevice,
        commands: &mut FrameCommands,
        alpha: f32,
    ) -> Result<(), GpuError>;
}
