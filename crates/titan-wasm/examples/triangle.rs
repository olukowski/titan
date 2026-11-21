//! A colorful triangle spinning in the browser; Space pauses and resumes it.
//!
//! ```sh
//! cargo build --profile wasm --target wasm32-unknown-unknown -p titan-wasm --example triangle
//! cp target/wasm32-unknown-unknown/wasm/examples/triangle.wasm web/
//! python3 -m http.server -d web   # then open http://localhost:8000
//! ```

use core::f32::consts::TAU;
use titan_app::{App, AppAction, InputState, KeyCode, SurfaceSize};
use titan_gpu::{
    Buffer, BufferDesc, BufferUsage, ClearColor, CullMode, FrameCommands, FrontFace, GpuDevice,
    GpuError, Handle, Pipeline, PipelineDesc, PrimitiveTopology, RenderPassDesc, ShaderSource,
    VertexAttribute, VertexFormat, VertexLayout, VertexStepMode,
};

const WGSL: &str = r"
// x: the surface's aspect ratio, y: the rotation in radians.
@group(0) @binding(0) var<uniform> params: vec4<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) color: vec3<f32>) -> VertexOut {
    let c = cos(params.y);
    let s = sin(params.y);
    let rotated = vec2<f32>(position.x * c - position.y * s, position.x * s + position.y * c);
    var out: VertexOut;
    out.position = vec4<f32>(rotated.x / params.x, rotated.y, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
";

/// Per vertex: position `x, y`, then color `r, g, b`.
const VERTICES: [[f32; 5]; 3] = [
    [0.0, 0.7, 1.0, 0.2, 0.3],
    [-0.62, -0.38, 0.2, 0.9, 0.4],
    [0.62, -0.38, 0.2, 0.4, 1.0],
];
const VERTEX_STRIDE: u32 = 20;

const ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute {
        format: VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    },
    VertexAttribute {
        format: VertexFormat::Float32x3,
        offset: 8,
        shader_location: 1,
    },
];

/// Radians per second.
const SPIN_SPEED: f32 = 1.2;

#[derive(Debug)]
struct Gpu {
    pipeline: Handle<Pipeline>,
    vertices: Handle<Buffer>,
    params: Handle<Buffer>,
}

#[derive(Debug, Default)]
struct TriangleApp {
    gpu: Option<Gpu>,
    aspect_ratio: f32,
    /// The rotation after the last step, and before it, for interpolating between them.
    angle: f32,
    previous_angle: f32,
    paused: bool,
}

impl App for TriangleApp {
    fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
        let shader = gpu.create_shader(&ShaderSource {
            label: Some("triangle"),
            msl: "",
            wgsl: WGSL,
            glsl: None,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
        })?;
        let pipeline = gpu.create_pipeline(&PipelineDesc {
            label: Some("triangle"),
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: VertexStepMode::Vertex,
                attributes: &ATTRIBUTES,
            }],
            color_format: Some(gpu.surface_format()),
            blend: None,
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        })?;
        // The pipeline keeps working without its shader.
        gpu.destroy_shader(shader)?;
        let vertex_bytes: Vec<u8> = VERTICES
            .as_flattened()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let vertices = gpu.create_buffer(&BufferDesc {
            label: Some("triangle vertices"),
            size: vertex_bytes.len() as u64,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        })?;
        gpu.write_buffer(vertices, 0, &vertex_bytes)?;
        let params = gpu.create_buffer(&BufferDesc {
            label: Some("triangle params"),
            size: 16,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        })?;
        self.gpu = Some(Gpu {
            pipeline,
            vertices,
            params,
        });
        Ok(())
    }

    fn resize(&mut self, _: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
        // A zero-height surface has no aspect ratio; keep the shader's division finite.
        self.aspect_ratio = Some(size.aspect_ratio())
            .filter(|&r| r > 0.0)
            .unwrap_or(1.0);
        Ok(())
    }

    fn step(&mut self, input: &InputState, dt: f32) -> AppAction {
        if input.just_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        self.previous_angle = self.angle;
        if !self.paused {
            self.angle += SPIN_SPEED * dt;
        }
        // Wrap both together so interpolation never spins backwards across the wrap.
        if self.angle >= TAU {
            self.angle -= TAU;
            self.previous_angle -= TAU;
        }
        AppAction::Continue
    }

    fn render(
        &mut self,
        gpu: &mut dyn GpuDevice,
        commands: &mut FrameCommands,
        alpha: f32,
    ) -> Result<(), GpuError> {
        let resources = self.gpu.as_ref().expect("init ran");
        let angle = self.previous_angle + (self.angle - self.previous_angle) * alpha;
        let mut params = [0; 16];
        params[..4].copy_from_slice(&self.aspect_ratio.to_le_bytes());
        params[4..8].copy_from_slice(&angle.to_le_bytes());
        gpu.write_buffer(resources.params, 0, &params)?;
        commands
            .begin_pass(RenderPassDesc::clear_surface(ClearColor::rgba(
                0.02, 0.02, 0.04, 1.0,
            )))
            .set_pipeline(resources.pipeline)
            .set_vertex_buffer(0, resources.vertices, 0)
            .set_uniform_buffer(0, resources.params, 0)
            .draw(0..3, 0..1);
        Ok(())
    }
}

titan_wasm::export_app!(TriangleApp::default());

#[cfg(test)]
mod tests {
    use super::*;
    use titan_app::{Event, HeadlessConfig, HeadlessRunner};
    use titan_gpu::{RenderCmd, TextureFormat};

    const DT: f32 = 1.0 / 60.0;

    fn params(runner: &HeadlessRunner<TriangleApp>) -> [f32; 2] {
        let handle = runner.app().gpu.as_ref().unwrap().params;
        let data = &runner.gpu().buffer(handle).unwrap().data;
        [0, 4].map(|i| f32::from_le_bytes(data[i..i + 4].try_into().unwrap()))
    }

    #[test]
    fn spins_draws_and_pauses_on_space() {
        let config = HeadlessConfig {
            surface_size: SurfaceSize::new(1600, 800, 2.0),
            surface_format: TextureFormat::Bgra8Unorm,
            ..HeadlessConfig::default()
        };
        let mut runner = HeadlessRunner::new(TriangleApp::default(), config).unwrap();
        // The shader was released after the pipeline was built.
        let live = runner.gpu().live_counts();
        assert_eq!((live.shaders, live.pipelines, live.buffers), (0, 1, 2));

        for _ in 0..30 {
            let outcome = runner.tick_frame(DT, &[]).unwrap();
            assert_eq!((outcome.steps_run, outcome.rendered), (1, true));
        }
        let [aspect, spinning] = params(&runner);
        assert_eq!(aspect, 2.0);
        // Rendering interpolates from the previous step, which alpha 0 shows exactly.
        assert!(
            (spinning - SPIN_SPEED * 29.0 * DT).abs() < 1e-4,
            "{spinning}"
        );

        let pass = &runner.gpu().last_frame().unwrap().passes[0];
        let resources = runner.app().gpu.as_ref().unwrap();
        assert!(pass.commands.contains(&RenderCmd::SetUniformBuffer {
            binding: 0,
            buffer: resources.params,
            offset: 0,
        }));
        assert!(pass.commands.contains(&RenderCmd::SetVertexBuffer {
            slot: 0,
            buffer: resources.vertices,
            offset: 0,
        }));

        runner
            .tick_frame(DT, &[Event::KeyDown(KeyCode::Space)])
            .unwrap();
        let paused = params(&runner)[1];
        runner.tick_frame(DT * 10.0, &[]).unwrap();
        assert_eq!(params(&runner)[1], paused);

        runner
            .tick_frame(
                DT,
                &[
                    Event::KeyUp(KeyCode::Space),
                    Event::KeyDown(KeyCode::Space),
                    Event::Resize(SurfaceSize::new(500, 0, 1.0)),
                ],
            )
            .unwrap();
        let [aspect, resumed] = params(&runner);
        assert_eq!(aspect, 1.0);
        assert!(resumed > paused);
    }

    #[test]
    fn interpolation_survives_the_wrap_at_a_full_turn() {
        let mut app = TriangleApp::default();
        let input = InputState::new(SurfaceSize::default());
        // 1000 steps is about three full turns.
        for _ in 0..1000 {
            app.step(&input, DT);
            let delta = app.angle - app.previous_angle;
            assert!((delta - SPIN_SPEED * DT).abs() < 1e-4, "{delta}");
            assert!((0.0..TAU).contains(&app.angle));
        }
    }
}
