//! [`HeadlessRunner`]: drives an [`App`] without a window, against a [`RecordingGpu`].

use crate::app::App;
use crate::event_log::EventLog;
use crate::input::{Event, InputState};
use crate::window::{AppAction, SurfaceSize, WindowConfig};
use titan_core::FixedClock;
use titan_gpu::{FrameCommands, GpuDevice, GpuError, RecordingGpu, TextureFormat};

/// Settings for a [`HeadlessRunner`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadlessConfig {
    /// The simulated surface's initial size.
    pub surface_size: SurfaceSize,
    /// The fixed simulation step in seconds.
    pub fixed_dt: f32,
    /// The longest frame time counted, which bounds the steps per frame.
    pub max_frame_dt: f32,
    /// The simulated surface's format.
    pub surface_format: TextureFormat,
}

impl HeadlessConfig {
    /// The default settings with the surface sized like `window`, at a scale factor of 1.
    #[must_use]
    pub fn from_window(window: &WindowConfig) -> Self {
        Self {
            surface_size: SurfaceSize::new(window.width, window.height, 1.0),
            ..Self::default()
        }
    }
}

impl Default for HeadlessConfig {
    /// An 800 x 600 `Bgra8UnormSrgb` surface stepped at 60 Hz, counting at most 0.25 s per frame.
    fn default() -> Self {
        Self {
            surface_size: SurfaceSize::default(),
            fixed_dt: 1.0 / 60.0,
            max_frame_dt: 0.25,
            surface_format: TextureFormat::Bgra8UnormSrgb,
        }
    }
}

/// What one [`HeadlessRunner::tick_frame`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOutcome {
    /// The number of simulation steps run.
    pub steps_run: u32,
    /// Whether a frame was rendered and submitted.
    pub rendered: bool,
    /// Whether the app has exited.
    pub exited: bool,
}

/// What one [`HeadlessRunner::replay`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplaySummary {
    /// The number of log frames run, including the one the app exited in.
    pub frames_run: usize,
    /// The number of simulation steps run.
    pub total_steps: u64,
    /// Whether the app has exited.
    pub exited: bool,
}

/// Runs an [`App`] frame by frame with no window or real GPU: the same loop a platform backend
/// runs, fed by explicit frame times and events instead of a clock and an OS.
///
/// Because the frame times and events are the only inputs, a run is reproducible: replaying the
/// same [`EventLog`] always yields the same app state and the same [`RecordingGpu`] history.
#[derive(Debug)]
pub struct HeadlessRunner<A: App> {
    app: A,
    gpu: RecordingGpu,
    clock: FixedClock,
    input: InputState,
    commands: FrameCommands,
    exited: bool,
    steps_run: u64,
}

impl<A: App> HeadlessRunner<A> {
    /// Creates a runner, calls [`App::init`], then calls [`App::resize`] with the initial surface
    /// size.
    ///
    /// # Errors
    ///
    /// Returns the error from `init` or `resize`.
    ///
    /// # Panics
    ///
    /// Panics if `config` is invalid for [`FixedClock::new`] or [`RecordingGpu::with_surface_format`].
    pub fn new(mut app: A, config: HeadlessConfig) -> Result<Self, GpuError> {
        let mut gpu = RecordingGpu::with_surface_format(config.surface_format);
        let clock = FixedClock::new(config.fixed_dt, config.max_frame_dt);
        app.init(&mut gpu)?;
        app.resize(&mut gpu, config.surface_size)?;
        Ok(Self {
            app,
            gpu,
            clock,
            input: InputState::new(config.surface_size),
            commands: FrameCommands::new(),
            exited: false,
            steps_run: 0,
        })
    }

    /// Runs one frame: applies `events`, runs the fixed steps `frame_dt` seconds add up to, then
    /// renders and submits. Does nothing once the app has exited.
    ///
    /// Each [`Event::Resize`] calls [`App::resize`] as it is applied. A step that returns
    /// [`AppAction::Exit`] ends the run at once: no further step runs and the frame is not
    /// rendered. A frame with no step keeps input edges, so the next step still sees them.
    ///
    /// # Errors
    ///
    /// Returns the first error from `resize`, `render`, or the submit.
    pub fn tick_frame(
        &mut self,
        frame_dt: f32,
        events: &[Event],
    ) -> Result<FrameOutcome, GpuError> {
        if self.exited {
            return Ok(FrameOutcome {
                steps_run: 0,
                rendered: false,
                exited: true,
            });
        }
        for &event in events {
            self.input.apply_event(event);
            if let Event::Resize(size) = event {
                self.app.resize(&mut self.gpu, size)?;
            }
        }
        let steps = self.clock.advance(frame_dt);
        for step in 1..=steps {
            let action = self.app.step(&self.input, self.clock.fixed_dt());
            self.steps_run += 1;
            self.input.end_step();
            if action == AppAction::Exit {
                self.exited = true;
                return Ok(FrameOutcome {
                    steps_run: step,
                    rendered: false,
                    exited: true,
                });
            }
        }
        self.commands.clear();
        self.app
            .render(&mut self.gpu, &mut self.commands, self.clock.alpha())?;
        self.gpu.submit(self.commands.view())?;
        Ok(FrameOutcome {
            steps_run: steps,
            rendered: true,
            exited: false,
        })
    }

    /// Runs every frame of `log` in order, stopping early if the app exits.
    ///
    /// # Errors
    ///
    /// Returns the first error from [`tick_frame`](Self::tick_frame).
    pub fn replay(&mut self, log: &EventLog) -> Result<ReplaySummary, GpuError> {
        let mut frames_run = 0;
        let mut total_steps = 0;
        for frame in log.frames() {
            if self.exited {
                break;
            }
            total_steps += u64::from(self.tick_frame(frame.dt, &frame.events)?.steps_run);
            frames_run += 1;
        }
        Ok(ReplaySummary {
            frames_run,
            total_steps,
            exited: self.exited,
        })
    }

    /// Returns the app.
    #[must_use]
    pub fn app(&self) -> &A {
        &self.app
    }

    /// Returns the app mutably.
    #[must_use]
    pub fn app_mut(&mut self) -> &mut A {
        &mut self.app
    }

    /// Returns the recording GPU, with every resource, frame, and call so far.
    #[must_use]
    pub fn gpu(&self) -> &RecordingGpu {
        &self.gpu
    }

    /// Returns the recording GPU mutably.
    #[must_use]
    pub fn gpu_mut(&mut self) -> &mut RecordingGpu {
        &mut self.gpu
    }

    /// Returns the simulation clock.
    ///
    /// Its [`tick`](FixedClock::tick) counts every step [`FixedClock::advance`] scheduled, which
    /// exceeds [`steps_run`](Self::steps_run) when the app exited partway through a frame's steps.
    #[must_use]
    pub fn clock(&self) -> &FixedClock {
        &self.clock
    }

    /// Returns the number of [`App::step`] calls so far.
    #[must_use]
    pub fn steps_run(&self) -> u64 {
        self.steps_run
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

    /// Returns the app and the GPU, ending the run.
    #[must_use]
    pub fn into_parts(self) -> (A, RecordingGpu) {
        (self.app, self.gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{KeyCode, MouseButton};
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;
    use titan_core::{Handle, Vec2};
    use titan_gpu::{
        Buffer, BufferDesc, BufferUsage, ClearColor, CullMode, DepthAttachment, DepthStencilState,
        FrontFace, GpuEvent, LoadOp, Pipeline, PipelineDesc, PrimitiveTopology, RenderPassDesc,
        ShaderSource, StoreOp, Texture, TextureDesc, TextureUsage,
    };

    /// A small simulation that uses every kind of input and draws with real GPU resources.
    #[derive(Debug, Default, PartialEq)]
    struct Mover {
        position: Vec2,
        velocity: Vec2,
        scroll: Vec2,
        jumps: u32,
        clicks: u32,
        steps: u64,
        pipeline: Option<Handle<Pipeline>>,
        uniforms: Option<Handle<Buffer>>,
    }

    impl App for Mover {
        fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
            let shader = gpu.create_shader(&ShaderSource {
                label: None,
                msl: "msl",
                wgsl: "wgsl",
                glsl: None,
                vertex_entry: "vs",
                fragment_entry: "fs",
            })?;
            self.pipeline = Some(gpu.create_pipeline(&PipelineDesc {
                label: None,
                shader,
                vertex_layouts: &[],
                color_format: Some(gpu.surface_format()),
                blend: None,
                depth_stencil: None,
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
            })?);
            self.uniforms = Some(gpu.create_buffer(&BufferDesc {
                label: Some("mover"),
                size: 12,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })?);
            Ok(())
        }

        fn step(&mut self, input: &InputState, dt: f32) -> AppAction {
            let axis = |neg, pos| {
                f32::from(u8::from(input.is_down(pos))) - f32::from(u8::from(input.is_down(neg)))
            };
            let thrust = Vec2::new(
                axis(KeyCode::ArrowLeft, KeyCode::ArrowRight),
                axis(KeyCode::ArrowUp, KeyCode::ArrowDown),
            );
            self.velocity = (self.velocity + thrust * (40.0 * dt)) * 0.98;
            self.position += self.velocity * dt + input.pointer_delta() * 0.01;
            self.scroll += input.scroll_delta();
            self.jumps += u32::from(input.just_pressed(KeyCode::Space));
            self.clicks += u32::from(input.button_just_pressed(MouseButton::Left));
            self.steps += 1;
            AppAction::Continue
        }

        fn render(
            &mut self,
            gpu: &mut dyn GpuDevice,
            commands: &mut FrameCommands,
            alpha: f32,
        ) -> Result<(), GpuError> {
            let uniforms = self.uniforms.expect("init ran");
            let mut data = Vec::new();
            for value in [self.position.x, self.position.y, alpha] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            gpu.write_buffer(uniforms, 0, &data)?;
            commands
                .begin_pass(RenderPassDesc::clear_surface(ClearColor::rgba(
                    f64::from(self.scroll.y),
                    0.0,
                    0.0,
                    1.0,
                )))
                .set_pipeline(self.pipeline.expect("init ran"))
                .set_uniform_buffer(0, uniforms, 0)
                .draw(0..3, 0..1);
            Ok(())
        }
    }

    /// A pseudo-random session of `frames` frames with jittery frame times and mixed input.
    fn session(frames: usize) -> EventLog {
        let mut seed = 0x2545_f491_u32;
        let mut next = move |n: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % n
        };
        let keys = [
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::Space,
        ];
        let mut log = EventLog::new();
        for _ in 0..frames {
            let dt = [1.0 / 144.0, 1.0 / 60.0, 1.0 / 30.0, 0.4][next(4) as usize];
            let events = (0..next(4))
                .map(|_| match next(7) {
                    0 => Event::KeyDown(keys[next(5) as usize]),
                    1 => Event::KeyUp(keys[next(5) as usize]),
                    2 => Event::PointerMove {
                        x: next(800) as f32 * 0.5,
                        y: next(600) as f32 * 0.25,
                    },
                    3 => Event::PointerDown {
                        button: MouseButton::Left,
                    },
                    4 => Event::PointerUp {
                        button: MouseButton::Left,
                    },
                    5 => Event::Scroll {
                        dx: 0.0,
                        dy: next(9) as f32 * -0.125,
                    },
                    _ => Event::Resize(SurfaceSize::new(
                        640 + next(640),
                        480 + next(480),
                        [1.0, 2.0][next(2) as usize],
                    )),
                })
                .collect();
            log.push_frame(dt, events);
        }
        log
    }

    #[test]
    fn replaying_a_log_is_deterministic() {
        let log = session(600);
        let mut first = HeadlessRunner::new(Mover::default(), HeadlessConfig::default()).unwrap();
        let first_summary = first.replay(&log).unwrap();
        // The second run replays the log's text form, so the format is part of what is checked.
        let reparsed = EventLog::parse(&log.to_string()).unwrap();
        let mut second = HeadlessRunner::new(Mover::default(), HeadlessConfig::default()).unwrap();
        let second_summary = second.replay(&reparsed).unwrap();

        assert_eq!(first_summary, second_summary);
        assert_eq!(first_summary.frames_run, 600);
        assert_eq!(first_summary.total_steps, first.clock().tick());
        assert_eq!(first_summary.total_steps, first.steps_run());
        assert_eq!(second.steps_run(), first.steps_run());
        assert!(!first_summary.exited);
        let mover = first.app();
        assert!(mover.jumps > 0 && mover.clicks > 0 && mover.position != Vec2::ZERO);
        assert_eq!(mover, second.app());
        assert_eq!(mover.steps, first_summary.total_steps);
        assert_eq!(first.input().surface_size(), second.input().surface_size());
        assert_eq!(first.gpu().frames().len(), 600);
        assert_eq!(first.gpu().frames(), second.gpu().frames());
        assert_eq!(first.gpu().events(), second.gpu().events());
        let uniforms = mover.uniforms.unwrap();
        assert_eq!(
            first.gpu().buffer(uniforms).unwrap().data,
            second.gpu().buffer(uniforms).unwrap().data
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Stage {
        Init,
        Resize,
        Render,
        Submit,
    }

    /// Records what each step saw, keeps a surface-sized render target, and can fail on demand.
    #[derive(Debug, Default)]
    struct Probe {
        /// Per step: whether Space was just pressed.
        space_edges: Vec<bool>,
        renders: u32,
        resizes: Vec<SurfaceSize>,
        target: Option<Handle<Texture>>,
        fail: Option<Stage>,
    }

    impl Probe {
        fn failing(stage: Stage) -> Self {
            Self {
                fail: Some(stage),
                ..Self::default()
            }
        }

        fn check(&self, stage: Stage) -> Result<(), GpuError> {
            match self.fail {
                Some(fail) if fail == stage => Err(GpuError::Backend(String::from("injected"))),
                _ => Ok(()),
            }
        }
    }

    impl App for Probe {
        fn init(&mut self, _gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
            self.check(Stage::Init)
        }

        fn resize(&mut self, gpu: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
            self.check(Stage::Resize)?;
            if let Some(old) = self.target.take() {
                gpu.destroy_texture(old)?;
            }
            self.target = Some(gpu.create_texture(&TextureDesc {
                usage: TextureUsage::RENDER_ATTACHMENT,
                ..TextureDesc::rgba8_2d(size.width, size.height)
            })?);
            self.resizes.push(size);
            Ok(())
        }

        fn step(&mut self, input: &InputState, _dt: f32) -> AppAction {
            self.space_edges.push(input.just_pressed(KeyCode::Space));
            if input.close_requested() {
                AppAction::Exit
            } else {
                AppAction::Continue
            }
        }

        fn render(
            &mut self,
            _gpu: &mut dyn GpuDevice,
            commands: &mut FrameCommands,
            _alpha: f32,
        ) -> Result<(), GpuError> {
            self.check(Stage::Render)?;
            self.renders += 1;
            let mut pass = commands.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK));
            if self.fail == Some(Stage::Submit) {
                // Invalid: no pipeline is set.
                pass.draw(0..3, 0..1);
            }
            Ok(())
        }
    }

    /// Steps of exactly 0.25 s keep the frame arithmetic exact.
    const QUARTER: HeadlessConfig = HeadlessConfig {
        surface_size: SurfaceSize {
            width: 800,
            height: 600,
            scale_factor: 1.0,
        },
        fixed_dt: 0.25,
        max_frame_dt: 1.0,
        surface_format: TextureFormat::Bgra8UnormSrgb,
    };

    fn probe_runner() -> HeadlessRunner<Probe> {
        HeadlessRunner::new(Probe::default(), QUARTER).unwrap()
    }

    #[test]
    fn a_press_is_an_edge_for_exactly_one_step() {
        let mut runner = probe_runner();
        let press = [Event::KeyDown(KeyCode::Space)];
        let outcome = runner.tick_frame(0.75, &press).unwrap();
        assert_eq!(
            outcome,
            FrameOutcome {
                steps_run: 3,
                rendered: true,
                exited: false,
            }
        );
        assert_eq!(runner.app().space_edges, [true, false, false]);
        assert!(runner.input().is_down(KeyCode::Space));
    }

    #[test]
    fn a_frame_without_steps_keeps_edges_for_the_next_step() {
        let mut runner = probe_runner();
        let outcome = runner
            .tick_frame(0.125, &[Event::KeyDown(KeyCode::Space)])
            .unwrap();
        assert_eq!(outcome.steps_run, 0);
        // The frame still renders, interpolating halfway.
        assert!(outcome.rendered);
        assert!(runner.app().space_edges.is_empty());
        assert!(runner.input().just_pressed(KeyCode::Space));

        assert_eq!(runner.tick_frame(0.125, &[]).unwrap().steps_run, 1);
        assert_eq!(runner.tick_frame(0.25, &[]).unwrap().steps_run, 1);
        assert_eq!(runner.app().space_edges, [true, false]);
        assert_eq!(runner.app().renders, 3);
    }

    #[test]
    fn exit_stops_stepping_and_rendering_for_good() {
        let mut runner = probe_runner();
        runner.tick_frame(0.25, &[]).unwrap();
        let outcome = runner.tick_frame(1.0, &[Event::CloseRequested]).unwrap();
        // The first step of four sees the request and exits; the frame is not rendered.
        assert_eq!(
            outcome,
            FrameOutcome {
                steps_run: 1,
                rendered: false,
                exited: true,
            }
        );
        assert!(runner.is_exited());
        assert_eq!(runner.app().space_edges.len(), 2);
        // The clock scheduled all four steps, but only the first ran.
        assert_eq!((runner.steps_run(), runner.clock().tick()), (2, 5));
        assert_eq!(runner.app().renders, 1);
        assert_eq!(runner.gpu().frames().len(), 1);

        let after = runner
            .tick_frame(1.0, &[Event::KeyDown(KeyCode::Space)])
            .unwrap();
        assert_eq!(
            after,
            FrameOutcome {
                steps_run: 0,
                rendered: false,
                exited: true,
            }
        );
        assert_eq!(runner.app().space_edges.len(), 2);
        assert_eq!(runner.steps_run(), 2);
        assert!(!runner.input().is_down(KeyCode::Space));
        assert_eq!(runner.gpu().frames().len(), 1);
    }

    #[test]
    fn replay_stops_at_exit() {
        let mut log = EventLog::new();
        log.push_frame(0.5, Vec::new());
        log.push_frame(0.25, vec![Event::CloseRequested]);
        log.push_frame(0.5, Vec::new());
        let mut runner = probe_runner();
        assert_eq!(
            runner.replay(&log).unwrap(),
            ReplaySummary {
                frames_run: 2,
                total_steps: 3,
                exited: true,
            }
        );
        assert_eq!(
            runner.replay(&log).unwrap(),
            ReplaySummary {
                frames_run: 0,
                total_steps: 0,
                exited: true,
            }
        );
    }

    #[test]
    fn resize_lets_the_app_recreate_gpu_resources() {
        let mut runner = probe_runner();
        let small = SurfaceSize::new(320, 240, 1.0);
        let retina = SurfaceSize::new(2560, 1600, 2.0);
        runner.tick_frame(0.25, &[Event::Resize(small)]).unwrap();
        let first = runner.app().target.unwrap();
        runner.tick_frame(0.25, &[Event::Resize(retina)]).unwrap();
        let second = runner.app().target.unwrap();

        assert_eq!(runner.app().resizes, [QUARTER.surface_size, small, retina]);
        assert_eq!(runner.input().surface_size(), retina);
        assert!(runner.gpu().texture(first).is_none());
        let texture = runner.gpu().texture(second).unwrap();
        assert_eq!((texture.width, texture.height), (2560, 1600));
        assert_eq!(runner.gpu().live_counts().textures, 1);
        assert!(
            runner
                .gpu()
                .events()
                .contains(&GpuEvent::DestroyTexture(first))
        );
        let (app, gpu) = runner.into_parts();
        assert_eq!(app.target, Some(second));
        assert_eq!(gpu.frames().len(), 2);
    }

    #[test]
    fn app_and_gpu_errors_propagate() {
        let injected = GpuError::Backend("injected".to_string());
        assert_eq!(
            HeadlessRunner::new(Probe::failing(Stage::Init), QUARTER).err(),
            Some(injected.clone())
        );

        // The initial resize fails creation just as a failing init does.
        assert_eq!(
            HeadlessRunner::new(Probe::failing(Stage::Resize), QUARTER).err(),
            Some(injected.clone())
        );
        let mut runner = probe_runner();
        runner.app_mut().fail = Some(Stage::Resize);
        let resize = [Event::Resize(SurfaceSize::default())];
        assert_eq!(runner.tick_frame(0.25, &resize), Err(injected.clone()));
        // The failing frame stopped before stepping.
        assert!(runner.app().space_edges.is_empty());

        let mut runner = HeadlessRunner::new(Probe::failing(Stage::Render), QUARTER).unwrap();
        assert_eq!(runner.tick_frame(0.25, &[]), Err(injected));
        assert!(runner.gpu().frames().is_empty());

        let mut runner = HeadlessRunner::new(Probe::failing(Stage::Submit), QUARTER).unwrap();
        assert_eq!(
            runner.tick_frame(0.25, &[]),
            Err(GpuError::DrawWithoutPipeline)
        );
        assert_eq!(runner.app().renders, 1);
        assert!(runner.gpu().frames().is_empty());

        let mut log = EventLog::new();
        log.push_frame(0.25, Vec::new());
        assert_eq!(runner.replay(&log), Err(GpuError::DrawWithoutPipeline));
    }

    /// A 3D app that sizes its depth target in `resize`, as the surface size demands.
    #[derive(Debug, Default)]
    struct Depth3d {
        pipeline: Option<Handle<Pipeline>>,
        depth: Option<Handle<Texture>>,
    }

    impl App for Depth3d {
        fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
            let shader = gpu.create_shader(&ShaderSource {
                label: None,
                msl: "msl",
                wgsl: "wgsl",
                glsl: None,
                vertex_entry: "vs",
                fragment_entry: "fs",
            })?;
            self.pipeline = Some(gpu.create_pipeline(&PipelineDesc {
                label: None,
                shader,
                vertex_layouts: &[],
                color_format: Some(gpu.surface_format()),
                blend: None,
                depth_stencil: Some(DepthStencilState::DEFAULT_3D),
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
            })?);
            Ok(())
        }

        fn resize(&mut self, gpu: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
            if let Some(old) = self.depth.take() {
                gpu.destroy_texture(old)?;
            }
            self.depth = Some(gpu.create_texture(&TextureDesc {
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::RENDER_ATTACHMENT,
                ..TextureDesc::rgba8_2d(size.width, size.height)
            })?);
            Ok(())
        }

        fn step(&mut self, _input: &InputState, _dt: f32) -> AppAction {
            AppAction::Continue
        }

        fn render(
            &mut self,
            _gpu: &mut dyn GpuDevice,
            commands: &mut FrameCommands,
            _alpha: f32,
        ) -> Result<(), GpuError> {
            let depth = self
                .depth
                .ok_or_else(|| GpuError::Backend("no size for depth target".to_string()))?;
            commands
                .begin_pass(RenderPassDesc {
                    depth: Some(DepthAttachment {
                        target: depth,
                        depth_load_op: LoadOp::Clear(1.0),
                        depth_store_op: StoreOp::Discard,
                    }),
                    ..RenderPassDesc::clear_surface(ClearColor::BLACK)
                })
                .set_pipeline(self.pipeline.expect("init ran"))
                .draw(0..3, 0..1);
            Ok(())
        }
    }

    #[test]
    fn the_first_frame_renders_at_the_initial_size_even_without_a_step() {
        let mut runner = HeadlessRunner::new(Depth3d::default(), QUARTER).unwrap();
        let outcome = runner.tick_frame(0.125, &[]).unwrap();
        assert_eq!((outcome.steps_run, outcome.rendered), (0, true));
        let depth = runner.app().depth.unwrap();
        let texture = runner.gpu().texture(depth).unwrap();
        assert_eq!((texture.width, texture.height), (800, 600));
        let pass = &runner.gpu().last_frame().unwrap().passes[0];
        assert_eq!(pass.desc.depth.map(|d| d.target), Some(depth));
    }

    #[test]
    fn config_from_window_uses_its_size() {
        let config = HeadlessConfig::from_window(&WindowConfig::new("Demo", 1280, 720));
        assert_eq!(config.surface_size, SurfaceSize::new(1280, 720, 1.0));
        assert_eq!(config.fixed_dt, HeadlessConfig::default().fixed_dt);
    }
}
