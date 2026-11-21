//! [`WebRunner`]: the browser frame loop around an [`App`], fed by `web/titan.js` events.

use crate::abi::{decode_key_code, decode_mouse_button};
use crate::bridge::HostBridge;
use crate::gpu::WebGpuDevice;
use titan_app::{App, AppAction, Event, InputState, SurfaceSize};
use titan_core::FixedClock;
use titan_gpu::{FrameCommands, GpuDevice, GpuError};

/// Settings for a [`WebRunner`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WebConfig {
    /// The fixed simulation step in seconds.
    pub fixed_dt: f32,
    /// The longest frame time counted, which bounds the steps per frame after a hitch or a
    /// backgrounded tab.
    pub max_frame_dt: f32,
}

impl Default for WebConfig {
    /// Steps at 60 Hz, counting at most 0.25 s per frame.
    fn default() -> Self {
        Self {
            fixed_dt: 1.0 / 60.0,
            max_frame_dt: 0.25,
        }
    }
}

/// Returns a surface size, or an error instead of [`SurfaceSize::new`]'s panic when the host
/// reports a scale factor that is not finite and positive.
pub(crate) fn surface_size(
    width: u32,
    height: u32,
    scale_factor: f32,
) -> Result<SurfaceSize, GpuError> {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        Ok(SurfaceSize::new(width, height, scale_factor))
    } else {
        // The value is left out: formatting an `f32` would add ~20 KB of WASM.
        Err(GpuError::Backend(
            "the scale factor must be finite and positive".into(),
        ))
    }
}

/// Runs an [`App`] in the browser: the loop [`HeadlessRunner`](titan_app::HeadlessRunner) runs,
/// driven by the host's events and `requestAnimationFrame` instead of an event log.
///
/// Input events update the [`InputState`] as they arrive; each [`frame`](Self::frame) then runs
/// the fixed steps the elapsed time adds up to and renders once.
#[derive(Debug)]
pub struct WebRunner<A: App, B: HostBridge> {
    app: A,
    gpu: WebGpuDevice<B>,
    clock: FixedClock,
    input: InputState,
    commands: FrameCommands,
    exited: bool,
}

impl<A: App, B: HostBridge> WebRunner<A, B> {
    /// Creates a runner, calls [`App::init`], then calls [`App::resize`] with `initial_size`.
    ///
    /// # Errors
    ///
    /// Returns the error from creating the device, `init`, or `resize`.
    ///
    /// # Panics
    ///
    /// Panics if `config` is invalid for [`FixedClock::new`].
    pub fn new(
        mut app: A,
        bridge: B,
        initial_size: SurfaceSize,
        config: WebConfig,
    ) -> Result<Self, GpuError> {
        let clock = FixedClock::new(config.fixed_dt, config.max_frame_dt);
        let mut gpu = WebGpuDevice::new(bridge)?;
        app.init(&mut gpu)?;
        app.resize(&mut gpu, initial_size)?;
        Ok(Self {
            app,
            gpu,
            clock,
            input: InputState::new(initial_size),
            commands: FrameCommands::new(),
            exited: false,
        })
    }

    /// Handles a canvas resize to `width` x `height` physical pixels, calling [`App::resize`].
    /// Does nothing once the app has exited.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] if `scale_factor` is not finite and positive, or the error
    /// from `resize`.
    pub fn on_resize(
        &mut self,
        width: u32,
        height: u32,
        scale_factor: f32,
    ) -> Result<(), GpuError> {
        let size = surface_size(width, height, scale_factor)?;
        if self.exited {
            return Ok(());
        }
        self.input.apply_event(Event::Resize(size));
        self.app.resize(&mut self.gpu, size)
    }

    /// Handles a key press or release; `code` is a [`KeyCode`](titan_app::KeyCode) wire code.
    /// Unknown codes are ignored.
    pub fn on_key(&mut self, code: u32, down: bool) {
        if let Some(key) = decode_key_code(code) {
            self.input.apply_event(if down {
                Event::KeyDown(key)
            } else {
                Event::KeyUp(key)
            });
        }
    }

    /// Handles a pointer move to (`x`, `y`) logical points from the canvas's top-left corner.
    /// Non-finite positions are ignored.
    pub fn on_pointer_move(&mut self, x: f32, y: f32) {
        if x.is_finite() && y.is_finite() {
            self.input.apply_event(Event::PointerMove { x, y });
        }
    }

    /// Handles a mouse button press or release; `button` is a
    /// [`MouseButton`](titan_app::MouseButton) wire code. Unknown buttons are ignored.
    pub fn on_pointer_button(&mut self, button: u32, down: bool) {
        if let Some(button) = decode_mouse_button(button) {
            self.input.apply_event(if down {
                Event::PointerDown { button }
            } else {
                Event::PointerUp { button }
            });
        }
    }

    /// Handles a scroll of (`dx`, `dy`) logical points. Non-finite deltas are ignored.
    pub fn on_scroll(&mut self, dx: f32, dy: f32) {
        if dx.is_finite() && dy.is_finite() {
            self.input.apply_event(Event::Scroll { dx, dy });
        }
    }

    /// Runs one frame `dt_secs` after the last: the fixed steps that time adds up to, then a
    /// render and submit. Returns [`AppAction::Exit`], doing nothing, once the app has exited.
    ///
    /// A step that returns [`AppAction::Exit`] ends the run at once: no further step runs and the
    /// frame is not rendered.
    ///
    /// # Errors
    ///
    /// Returns the first error from `render` or the submit.
    pub fn frame(&mut self, dt_secs: f32) -> Result<AppAction, GpuError> {
        if self.exited {
            return Ok(AppAction::Exit);
        }
        for _ in 0..self.clock.advance(dt_secs) {
            let action = self.app.step(&self.input, self.clock.fixed_dt());
            self.input.end_step();
            if action == AppAction::Exit {
                self.exited = true;
                return Ok(AppAction::Exit);
            }
        }
        self.commands.clear();
        self.app
            .render(&mut self.gpu, &mut self.commands, self.clock.alpha())?;
        self.gpu.submit(self.commands.view())?;
        Ok(AppAction::Continue)
    }

    /// Returns the app.
    #[must_use]
    pub fn app(&self) -> &A {
        &self.app
    }

    /// Returns the GPU device.
    #[must_use]
    pub fn gpu(&self) -> &WebGpuDevice<B> {
        &self.gpu
    }

    /// Returns the input state as the next step will see it.
    #[must_use]
    pub fn input(&self) -> &InputState {
        &self.input
    }

    /// Returns whether the app has exited.
    #[must_use]
    pub fn is_exited(&self) -> bool {
        self.exited
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::decode::Reader;
    use crate::bridge::mock::{Call, MockBridge};
    use alloc::vec::Vec;
    use titan_app::{KeyCode, MouseButton};
    use titan_core::Vec2;
    use titan_gpu::{
        Buffer, BufferDesc, BufferUsage, ClearColor, Handle, RenderPassDesc, TextureFormat,
    };

    /// Counts what the runner asks of it, and exits when Escape is pressed.
    #[derive(Debug, Default)]
    struct Probe {
        uniforms: Option<Handle<Buffer>>,
        sizes: Vec<SurfaceSize>,
        steps: u32,
        renders: u32,
        keys_pressed: u32,
        clicks: u32,
        pointer: Vec2,
        scroll: Vec2,
    }

    impl App for Probe {
        fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
            self.uniforms = Some(gpu.create_buffer(&BufferDesc {
                label: Some("probe"),
                size: 16,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })?);
            Ok(())
        }

        fn resize(&mut self, _: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
            self.sizes.push(size);
            Ok(())
        }

        fn step(&mut self, input: &InputState, _: f32) -> AppAction {
            self.steps += 1;
            self.keys_pressed += u32::from(input.just_pressed(KeyCode::Space));
            self.clicks += u32::from(input.button_just_pressed(MouseButton::Right));
            self.pointer = input.pointer_position();
            self.scroll += input.scroll_delta();
            if input.just_pressed(KeyCode::Escape) {
                AppAction::Exit
            } else {
                AppAction::Continue
            }
        }

        fn render(
            &mut self,
            gpu: &mut dyn GpuDevice,
            commands: &mut FrameCommands,
            _: f32,
        ) -> Result<(), GpuError> {
            self.renders += 1;
            gpu.write_buffer(self.uniforms.expect("init ran"), 0, &[0; 16])?;
            commands.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK));
            Ok(())
        }
    }

    const DT: f32 = 1.0 / 60.0;

    fn runner() -> WebRunner<Probe, MockBridge> {
        WebRunner::new(
            Probe::default(),
            MockBridge::new(),
            SurfaceSize::new(800, 600, 2.0),
            WebConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn new_runs_init_then_the_initial_resize() {
        let runner = runner();
        assert_eq!(runner.app().sizes, [SurfaceSize::new(800, 600, 2.0)]);
        assert_eq!(
            runner.input().surface_size(),
            SurfaceSize::new(800, 600, 2.0)
        );
        assert!(matches!(
            runner.gpu().bridge().calls[..],
            [Call::CreateBuffer {
                id: 1,
                size: 16,
                ..
            }]
        ));
    }

    #[test]
    fn a_failing_init_surfaces_the_error() {
        let mut bridge = MockBridge::new();
        bridge.fail_next = Some("device lost");
        let result = WebRunner::new(
            Probe::default(),
            bridge,
            SurfaceSize::default(),
            WebConfig::default(),
        );
        assert_eq!(result.err(), Some(GpuError::Backend("device lost".into())));

        let mut bridge = MockBridge::new();
        bridge.surface_format = TextureFormat::Depth32Float;
        let result = WebRunner::new(
            Probe::default(),
            bridge,
            SurfaceSize::default(),
            WebConfig::default(),
        );
        assert!(matches!(result, Err(GpuError::Backend(_))));
    }

    #[test]
    fn frames_run_fixed_steps_then_render_once() {
        let mut runner = runner();
        // Less than one step: render only.
        assert_eq!(runner.frame(DT * 0.5), Ok(AppAction::Continue));
        assert_eq!((runner.app().steps, runner.app().renders), (0, 1));
        // A long hitch is clamped to max_frame_dt: 0.25 s is 15 steps.
        runner.frame(10.0).unwrap();
        assert_eq!((runner.app().steps, runner.app().renders), (15, 2));
        // Bad timer readings run no step but still render.
        for bad in [f32::NAN, -1.0, f32::INFINITY] {
            runner.frame(bad).unwrap();
        }
        assert_eq!((runner.app().steps, runner.app().renders), (15, 5));
        assert_eq!(runner.gpu().bridge().frames().count(), 5);
        let frame = Reader::new(runner.gpu().bridge().frames().last().unwrap()).frame();
        assert_eq!(frame.len(), 1);
    }

    #[test]
    fn resize_updates_input_and_app_and_rejects_bad_scale_factors() {
        let mut runner = runner();
        runner.on_resize(1024, 768, 1.5).unwrap();
        assert_eq!(
            runner.input().surface_size(),
            SurfaceSize::new(1024, 768, 1.5)
        );
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                runner.on_resize(10, 10, bad),
                Err(GpuError::Backend(_))
            ));
        }
        assert_eq!(runner.app().sizes.len(), 2);
        assert_eq!(
            runner.input().surface_size(),
            SurfaceSize::new(1024, 768, 1.5)
        );
    }

    #[test]
    fn input_events_reach_the_next_step() {
        let mut runner = runner();
        runner.on_key(KeyCode::Space as u32, true);
        runner.on_key(KeyCode::Space as u32, false);
        runner.on_key(65, true);
        runner.on_key(u32::MAX, true);
        runner.on_pointer_button(MouseButton::Right as u32, true);
        runner.on_pointer_button(3, true);
        runner.on_pointer_move(10.0, 20.0);
        runner.on_pointer_move(f32::NAN, 0.0);
        runner.on_scroll(1.0, -2.0);
        runner.on_scroll(f32::INFINITY, 0.0);
        runner.frame(DT).unwrap();
        let app = runner.app();
        assert_eq!((app.keys_pressed, app.clicks), (1, 1));
        assert_eq!(app.pointer, Vec2::new(10.0, 20.0));
        assert_eq!(app.scroll, Vec2::new(1.0, -2.0));
        assert!(runner.input().is_button_down(MouseButton::Right));
    }

    #[test]
    fn exit_stops_stepping_rendering_and_resizing() {
        let mut runner = runner();
        runner.on_key(KeyCode::Escape as u32, true);
        // Three steps are due, but the first exits.
        assert_eq!(runner.frame(DT * 3.0), Ok(AppAction::Exit));
        assert!(runner.is_exited());
        assert_eq!((runner.app().steps, runner.app().renders), (1, 0));
        assert_eq!(runner.frame(DT * 3.0), Ok(AppAction::Exit));
        runner.on_resize(1, 1, 1.0).unwrap();
        assert_eq!((runner.app().steps, runner.app().sizes.len()), (1, 1));
        assert_eq!(runner.gpu().bridge().frames().count(), 0);
    }

    #[test]
    fn a_failed_submit_is_reported_and_the_next_frame_still_runs() {
        let mut runner = runner();
        runner.gpu.bridge_mut().fail_next = Some("queue full");
        // The render's buffer write is the first fallible bridge call.
        assert_eq!(
            runner.frame(DT),
            Err(GpuError::Backend("queue full".into()))
        );
        assert_eq!(runner.frame(DT), Ok(AppAction::Continue));
        assert_eq!(runner.gpu().bridge().frames().count(), 1);
    }
}
