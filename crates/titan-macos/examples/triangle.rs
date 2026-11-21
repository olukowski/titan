//! Opens a window and draws a red-green-blue triangle. Press Escape or close the window to quit.
//!
//! ```sh
//! cargo run -p titan-macos --example triangle
//! ```

use std::process::ExitCode;
use titan_app::{App, AppAction, InputState, KeyCode, WindowConfig};
use titan_gpu::{
    Buffer, BufferDesc, BufferUsage, ClearColor, FrameCommands, GpuDevice, GpuError, Handle,
    Pipeline, PipelineDesc, RenderPassDesc, ShaderSource, VertexAttribute, VertexFormat,
    VertexLayout, VertexStepMode,
};

const MSL: &str = r"
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
    float3 color [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut vs_main(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 fs_main(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
";

const WGSL: &str = r"
struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
";

/// x, y, then r, g, b for each corner, counter-clockwise from the bottom left.
const VERTICES: [[f32; 5]; 3] = [
    [-0.6, -0.5, 1.0, 0.0, 0.0],
    [0.6, -0.5, 0.0, 1.0, 0.0],
    [0.0, 0.6, 0.0, 0.0, 1.0],
];
const STRIDE: u32 = 5 * 4;

/// The GPU resources the triangle draws with, created in `init`.
struct Resources {
    pipeline: Handle<Pipeline>,
    vertices: Handle<Buffer>,
}

#[derive(Default)]
struct Triangle {
    resources: Option<Resources>,
}

impl App for Triangle {
    fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
        let shader = gpu.create_shader(&ShaderSource {
            label: Some("triangle"),
            msl: MSL,
            wgsl: WGSL,
            glsl: None,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
        })?;
        let attributes = [
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
        let pipeline = gpu.create_pipeline(&PipelineDesc {
            label: Some("triangle"),
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: STRIDE,
                step_mode: VertexStepMode::Vertex,
                attributes: &attributes,
            }],
            color_format: Some(gpu.surface_format()),
            blend: None,
            depth_stencil: None,
            topology: Default::default(),
            front_face: Default::default(),
            cull_mode: Default::default(),
        })?;
        gpu.destroy_shader(shader)?;
        let bytes: Vec<u8> = VERTICES
            .iter()
            .flatten()
            .flat_map(|f| f.to_ne_bytes())
            .collect();
        let vertices = gpu.create_buffer(&BufferDesc {
            label: Some("triangle vertices"),
            size: bytes.len() as u64,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        })?;
        gpu.write_buffer(vertices, 0, &bytes)?;
        self.resources = Some(Resources { pipeline, vertices });
        Ok(())
    }

    fn step(&mut self, input: &InputState, _dt: f32) -> AppAction {
        if input.close_requested() || input.just_pressed(KeyCode::Escape) {
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
        let resources = self.resources.as_ref().expect("init ran");
        commands
            .begin_pass(RenderPassDesc::clear_surface(ClearColor::rgba(
                0.05, 0.05, 0.08, 1.0,
            )))
            .set_pipeline(resources.pipeline)
            .set_vertex_buffer(0, resources.vertices, 0)
            .draw(0..3, 0..1);
        Ok(())
    }
}

fn main() -> ExitCode {
    match titan_macos::run(
        Triangle::default(),
        WindowConfig::new("Titan triangle", 800, 600),
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("triangle: {error}");
            ExitCode::FAILURE
        }
    }
}
