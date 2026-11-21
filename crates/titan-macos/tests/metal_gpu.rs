//! Renders offscreen on the real Metal GPU and checks the pixels read back.

#![cfg(target_os = "macos")]

use titan_gpu::{
    BlendState, Buffer, BufferDesc, BufferUsage, ClearColor, ColorAttachment, CompareFunction,
    CullMode, DepthAttachment, DepthStencilState, FrameCommands, FrontFace, GpuDevice, GpuError,
    Handle, IndexFormat, LoadOp, Pipeline, PipelineDesc, PrimitiveTopology, RenderPassDesc,
    ResourceKind, SamplerDesc, ScissorRect, Shader, ShaderSource, StoreOp, Texture, TextureDesc,
    TextureFormat, TextureUsage, VertexAttribute, VertexFormat, VertexLayout, VertexStepMode,
    Viewport,
};
use titan_macos::MetalGpu;

const SIZE: u32 = 64;
const BLUE: [u8; 4] = [0, 0, 255, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];

/// Vertices with a clip-space position and a color: the simplest pipeline that draws.
const COLORED_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct VertexIn {
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};
struct VertexOut {
    float4 position [[position]];
    float4 color;
};
vertex VertexOut vs_main(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 1.0);
    out.color = in.color;
    return out;
}
fragment float4 fs_main(VertexOut in [[stage_in]]) {
    return in.color;
}
";
const COLORED_ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute {
        format: VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    VertexAttribute {
        format: VertexFormat::Float32x4,
        offset: 12,
        shader_location: 1,
    },
];
const COLORED_STRIDE: u32 = 7 * 4;

fn gpu() -> MetalGpu {
    MetalGpu::new_headless().expect("this Mac has a Metal GPU")
}

fn bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

fn buffer(gpu: &mut MetalGpu, usage: BufferUsage, data: &[u8]) -> Handle<Buffer> {
    let handle = gpu
        .create_buffer(&BufferDesc {
            label: None,
            size: data.len() as u64,
            usage: usage | BufferUsage::COPY_DST,
        })
        .unwrap();
    gpu.write_buffer(handle, 0, data).unwrap();
    handle
}

fn texture(gpu: &mut MetalGpu, format: TextureFormat, usage: TextureUsage) -> Handle<Texture> {
    gpu.create_texture(&TextureDesc {
        label: None,
        width: SIZE,
        height: SIZE,
        mip_level_count: 1,
        format,
        usage,
        sampler: SamplerDesc::NEAREST_CLAMP,
    })
    .unwrap()
}

/// A readable `Rgba8Unorm` render target.
fn target(gpu: &mut MetalGpu) -> Handle<Texture> {
    texture(
        gpu,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    )
}

fn shader(gpu: &mut MetalGpu, msl: &str) -> Result<Handle<Shader>, GpuError> {
    gpu.create_shader(&ShaderSource {
        label: None,
        msl,
        wgsl: "",
        glsl: None,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
    })
}

fn colored_pipeline(
    gpu: &mut MetalGpu,
    depth: Option<TextureFormat>,
    cull_mode: CullMode,
) -> Handle<Pipeline> {
    let shader = shader(gpu, COLORED_MSL).unwrap();
    let pipeline = gpu
        .create_pipeline(&PipelineDesc {
            label: Some("colored"),
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: COLORED_STRIDE,
                step_mode: VertexStepMode::Vertex,
                attributes: &COLORED_ATTRIBUTES,
            }],
            color_format: Some(TextureFormat::Rgba8Unorm),
            blend: None,
            depth_stencil: depth.map(|format| DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode,
        })
        .unwrap();
    // Pipelines keep working after their shader is destroyed.
    gpu.destroy_shader(shader).unwrap();
    pipeline
}

/// A triangle-list rectangle from (`x0`, `y0`) to (`x1`, `y1`) in clip space at depth `z`,
/// counter-clockwise, in one color.
fn rect(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, rgba: [f32; 4]) -> Vec<f32> {
    [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
        .iter()
        .flat_map(|&[x, y]| [x, y, z].into_iter().chain(rgba))
        .collect()
}

fn clear_pass(target: Handle<Texture>, color: ClearColor) -> RenderPassDesc {
    RenderPassDesc {
        label: Some("test pass"),
        color: Some(ColorAttachment {
            target: Some(target),
            load_op: LoadOp::Clear(color),
            store_op: StoreOp::Store,
        }),
        depth: None,
    }
}

fn invalid<T>(message: &'static str) -> Result<T, GpuError> {
    Err(GpuError::InvalidDescriptor { message })
}

fn stale<T>(kind: ResourceKind) -> Result<T, GpuError> {
    Err(GpuError::InvalidHandle { kind })
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

/// Asserts each channel is within 1 of `expected`, allowing for rounding of fractions.
#[track_caller]
fn assert_pixel(pixels: &[u8], (x, y): (u32, u32), expected: [u8; 4]) {
    let actual = pixel(pixels, x, y);
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(&a, e)| a.abs_diff(e) <= 1),
        "pixel ({x}, {y}) is {actual:?}, expected {expected:?}"
    );
}

#[test]
fn renders_a_triangle_the_right_way_up_and_culls_by_winding() {
    let mut gpu = gpu();
    let target = target(&mut gpu);
    let red = [1.0, 0.0, 0.0, 1.0];
    let vertices: Vec<f32> = [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]
        .iter()
        .flat_map(|&[x, y]| [x, y, 0.0].into_iter().chain(red))
        .collect();
    let vertices = buffer(&mut gpu, BufferUsage::VERTEX, &bytes(&vertices));
    let blue = ClearColor::rgba(0.0, 0.0, 1.0, 1.0);

    // Counter-clockwise is front-facing, so culling back faces keeps the triangle.
    let keeps = colored_pipeline(&mut gpu, None, CullMode::Back);
    let mut frame = FrameCommands::new();
    frame
        .begin_pass(clear_pass(target, blue))
        .set_pipeline(keeps)
        .set_vertex_buffer(0, vertices, 0)
        // Empty draws are valid and must not reach Metal, which rejects them.
        .draw(0..0, 0..1)
        .draw(0..3, 0..0)
        .draw(0..3, 0..1);
    gpu.submit(frame.view()).unwrap();
    let pixels = gpu.read_texture(target).unwrap();
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert_pixel(&pixels, (32, 32), RED);
    // Clip-space +y is up: the apex (row 16) is near the top, the base (row 48) below.
    assert_pixel(&pixels, (32, 20), RED);
    assert_pixel(&pixels, (32, 12), BLUE);
    assert_pixel(&pixels, (32, 52), BLUE);
    assert_pixel(&pixels, (0, 0), BLUE);
    assert_pixel(&pixels, (10, 32), BLUE);

    let culls = colored_pipeline(&mut gpu, None, CullMode::Front);
    frame.clear();
    frame
        .begin_pass(clear_pass(target, blue))
        .set_pipeline(culls)
        .set_vertex_buffer(0, vertices, 0)
        .draw(0..3, 0..1);
    gpu.submit(frame.view()).unwrap();
    assert_pixel(&gpu.read_texture(target).unwrap(), (32, 32), BLUE);
}

/// Reads a vertex buffer at `[[buffer(0)]]`, a uniform at `[[buffer(8)]]` in both stages, a
/// storage buffer at `[[buffer(9)]]`, and a texture and sampler at index 0.
const BINDINGS_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct VertexIn {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
};
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};
struct Tint {
    float4 multiply;
    float4 offset;
};
vertex VertexOut vs_main(VertexIn in [[stage_in]], constant Tint& tint [[buffer(8)]]) {
    VertexOut out;
    out.position = float4(in.position + tint.offset.xy, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}
fragment float4 fs_main(
    VertexOut in [[stage_in]],
    constant Tint& tint [[buffer(8)]],
    const device float4* add [[buffer(9)]],
    texture2d<float> image [[texture(0)]],
    sampler image_sampler [[sampler(0)]]
) {
    return image.sample(image_sampler, in.uv) * tint.multiply + add[0];
}
";

#[test]
fn every_binding_reaches_its_documented_metal_slot() {
    let mut gpu = gpu();
    let target = target(&mut gpu);
    let shader = shader(&mut gpu, BINDINGS_MSL).unwrap();
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 8,
            shader_location: 1,
        },
    ];
    let pipeline = gpu
        .create_pipeline(&PipelineDesc {
            label: None,
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: 16,
                step_mode: VertexStepMode::Vertex,
                attributes: &attributes,
            }],
            color_format: Some(TextureFormat::Rgba8Unorm),
            blend: None,
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        })
        .unwrap();
    // Four junk vertices that `base_vertex` skips, then a quad over the left half of clip
    // space, which the uniform's offset moves to the middle: pixels 16..48.
    let mut quad = vec![9.0; 16];
    quad.extend([
        -1.0, -1.0, 0.0, 1.0, //
        0.0, -1.0, 1.0, 1.0, //
        0.0, 1.0, 1.0, 0.0, //
        -1.0, 1.0, 0.0, 0.0,
    ]);
    let vertices = buffer(&mut gpu, BufferUsage::VERTEX, &bytes(&quad));
    let tint = buffer(
        &mut gpu,
        BufferUsage::UNIFORM,
        &bytes(&[1.0, 0.2, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
    );
    let add = buffer(
        &mut gpu,
        BufferUsage::STORAGE,
        &bytes(&[0.0, 0.0, 1.0, 0.0]),
    );
    // A 2x2 texture: red on the left, green on the right.
    let image = gpu
        .create_texture(&TextureDesc {
            label: Some("image"),
            width: 2,
            height: 2,
            mip_level_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
            sampler: SamplerDesc::NEAREST_CLAMP,
        })
        .unwrap();
    gpu.write_texture(image, &[RED, GREEN, RED, GREEN].concat())
        .unwrap();

    // Three degenerate indices that `first_index` skips, then the quad's two triangles.
    let indices = [0, 0, 0, 0, 1, 2, 0, 2, 3, 0];
    let u16s: Vec<u8> = indices
        .iter()
        .flat_map(|&i: &u16| i.to_ne_bytes())
        .collect();
    let u32s: Vec<u8> = indices
        .iter()
        .flat_map(|&i| u32::from(i).to_ne_bytes())
        .collect();
    for (format, data) in [(IndexFormat::Uint16, u16s), (IndexFormat::Uint32, u32s)] {
        let index_buffer = buffer(&mut gpu, BufferUsage::INDEX, &data);
        let mut frame = FrameCommands::new();
        frame
            .begin_pass(clear_pass(target, ClearColor::BLACK))
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertices, 0)
            .set_index_buffer(index_buffer, format, 0)
            .set_uniform_buffer(0, tint, 0)
            .set_storage_buffer(1, add, 0)
            .set_texture(0, image)
            .draw_indexed(3..9, 4, 0..1);
        gpu.submit(frame.view()).unwrap();
        let pixels = gpu.read_texture(target).unwrap();
        // Red texel * (1, 0.2, 1, 1) + (0, 0, 1, 0), then green texel likewise.
        assert_pixel(&pixels, (20, 32), [255, 0, 255, 255]);
        assert_pixel(&pixels, (44, 32), [0, 51, 255, 255]);
        assert_pixel(&pixels, (8, 32), BLACK);
        assert_pixel(&pixels, (56, 32), BLACK);
    }
}

const UNIFORM_COLOR_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct VertexIn {
    float2 position [[attribute(0)]];
};
vertex float4 vs_main(VertexIn in [[stage_in]]) {
    return float4(in.position, 0.0, 1.0);
}
fragment float4 fs_main(constant float4& color [[buffer(8)]]) {
    return color;
}
";

const SCALAR_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct VertexIn {
    float2 position [[attribute(0)]];
};
vertex float4 vs_main(VertexIn in [[stage_in]]) {
    return float4(in.position, 0.0, 1.0);
}
fragment float4 fs_main(constant float* scale [[buffer(8)]], const device float* add [[buffer(9)]]) {
    return float4(scale[0], scale[1], scale[2], scale[3]) + float4(add[0], add[1], add[2], add[3]);
}
";

#[test]
fn scalar_uniform_and_storage_bindings_read_from_a_four_byte_offset() {
    let mut gpu = gpu();
    let target = target(&mut gpu);
    let shader = shader(&mut gpu, SCALAR_MSL).unwrap();
    let attributes = [VertexAttribute {
        format: VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    }];
    let pipeline = gpu
        .create_pipeline(&PipelineDesc {
            label: None,
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: &attributes,
            }],
            color_format: Some(TextureFormat::Rgba8Unorm),
            blend: None,
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleStrip,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        })
        .unwrap();
    let strip = bytes(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    let vertices = buffer(&mut gpu, BufferUsage::VERTEX, &strip);
    // Apple silicon uses 4-byte uniform offsets; non-Apple GPUs use the 256-byte alignment.
    let uniform_offset = if cfg!(target_arch = "aarch64") {
        4
    } else {
        256
    };
    // Sentinels fill the bytes before each offset; reading one would saturate a channel.
    let sentinels = |offset: u64| vec![99.0; usize::try_from(offset / 4).unwrap()];
    let scale = buffer(
        &mut gpu,
        BufferUsage::UNIFORM,
        &bytes(&[sentinels(uniform_offset), vec![0.0, 0.2, 0.0, 1.0]].concat()),
    );
    let add = buffer(
        &mut gpu,
        BufferUsage::STORAGE,
        &bytes(&[sentinels(4), vec![0.0, 0.0, 1.0, 0.0]].concat()),
    );
    let mut frame = FrameCommands::new();
    frame
        .begin_pass(clear_pass(target, ClearColor::BLACK))
        .set_pipeline(pipeline)
        .set_vertex_buffer(0, vertices, 0)
        .set_uniform_buffer(0, scale, uniform_offset)
        .set_storage_buffer(1, add, 4)
        .draw(0..4, 0..1);
    gpu.submit(frame.view()).unwrap();
    let pixels = gpu.read_texture(target).unwrap();
    assert_pixel(&pixels, (8, 8), [0, 51, 255, 255]);
    assert_pixel(&pixels, (56, 56), [0, 51, 255, 255]);
}

#[test]
fn blending_viewport_and_scissor_shape_what_is_drawn() {
    let mut gpu = gpu();
    let target = target(&mut gpu);
    let shader = shader(&mut gpu, UNIFORM_COLOR_MSL).unwrap();
    let attributes = [VertexAttribute {
        format: VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    }];
    let pipeline = gpu
        .create_pipeline(&PipelineDesc {
            label: None,
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: &attributes,
            }],
            color_format: Some(TextureFormat::Rgba8Unorm),
            blend: Some(BlendState::ALPHA_BLENDING),
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleStrip,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        })
        .unwrap();
    let strip = bytes(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    let vertices = buffer(&mut gpu, BufferUsage::VERTEX, &strip);
    let half_white = buffer(
        &mut gpu,
        BufferUsage::UNIFORM,
        &bytes(&[1.0, 1.0, 1.0, 0.5]),
    );
    let mut frame = FrameCommands::new();
    frame
        .begin_pass(clear_pass(target, ClearColor::BLACK))
        .set_pipeline(pipeline)
        .set_vertex_buffer(0, vertices, 0)
        .set_uniform_buffer(0, half_white, 0)
        .set_viewport(Viewport::new(0.0, 0.0, 32.0, 64.0))
        .set_scissor(ScissorRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        })
        .draw(0..4, 0..1);
    gpu.submit(frame.view()).unwrap();
    let pixels = gpu.read_texture(target).unwrap();
    // White at alpha 0.5 over opaque black: half gray, alpha 0.5 + 1 * 0.5.
    assert_pixel(&pixels, (8, 8), [128, 128, 128, 255]);
    assert_pixel(&pixels, (24, 24), [128, 128, 128, 255]);
    // Outside the viewport (right half) and outside the scissor (bottom half).
    assert_pixel(&pixels, (48, 8), BLACK);
    assert_pixel(&pixels, (8, 48), BLACK);
}

#[test]
fn depth_testing_works_with_both_depth_formats() {
    for format in [
        TextureFormat::Depth32Float,
        TextureFormat::Depth24PlusStencil8,
    ] {
        let mut gpu = gpu();
        let color = target(&mut gpu);
        let depth = texture(&mut gpu, format, TextureUsage::RENDER_ATTACHMENT);
        let pipeline = colored_pipeline(&mut gpu, Some(format), CullMode::None);
        let mut quads = rect(-1.0, -1.0, 1.0, 1.0, 0.25, [1.0, 0.0, 0.0, 1.0]);
        quads.extend(rect(-1.0, -1.0, 1.0, 1.0, 0.75, [0.0, 1.0, 0.0, 1.0]));
        quads.extend(rect(-1.0, -1.0, 0.0, 1.0, 0.1, [0.0, 0.0, 1.0, 1.0]));
        let vertices = buffer(&mut gpu, BufferUsage::VERTEX, &bytes(&quads));
        let mut frame = FrameCommands::new();
        frame
            .begin_pass(RenderPassDesc {
                depth: Some(DepthAttachment {
                    target: depth,
                    depth_load_op: LoadOp::Clear(1.0),
                    depth_store_op: StoreOp::Discard,
                }),
                ..clear_pass(color, ClearColor::BLACK)
            })
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertices, 0)
            .draw(0..6, 0..1)
            .draw(6..12, 0..1)
            .draw(12..18, 0..1);
        gpu.submit(frame.view()).unwrap();
        let pixels = gpu.read_texture(color).unwrap();
        // The far green quad loses everywhere; the nearest blue one wins on the left.
        assert_pixel(&pixels, (16, 32), BLUE);
        assert_pixel(&pixels, (48, 32), RED);
    }
}

#[test]
fn invalid_calls_fail_like_recording_gpu() {
    let mut gpu = gpu();
    let target = target(&mut gpu);

    let vertex_only = buffer(&mut gpu, BufferUsage::VERTEX, &[0; 8]);
    let readonly = gpu
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::VERTEX,
        })
        .unwrap();
    assert_eq!(
        gpu.write_buffer(readonly, 0, &[0; 4]),
        Err(GpuError::MissingUsage {
            kind: ResourceKind::Buffer,
            required: "COPY_DST",
        })
    );
    assert_eq!(
        gpu.write_buffer(vertex_only, 4, &[0; 8]),
        Err(GpuError::BufferOutOfBounds {
            offset: 4,
            write_len: 8,
            buffer_size: 8,
        })
    );
    assert_eq!(
        gpu.write_buffer(vertex_only, 2, &[0; 4]),
        invalid("a buffer write's offset and length must be multiples of 4")
    );
    assert_eq!(
        gpu.create_buffer(&BufferDesc {
            label: None,
            size: 0,
            usage: BufferUsage::VERTEX,
        }),
        invalid("buffer size must be nonzero")
    );

    let image = texture(
        &mut gpu,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED | TextureUsage::COPY_DST,
    );
    assert_eq!(
        gpu.write_texture(image, &[0; 12]),
        Err(GpuError::TextureDataSizeMismatch {
            expected: (SIZE * SIZE * 4) as usize,
            actual: 12,
        })
    );
    assert_eq!(
        gpu.create_texture(&TextureDesc {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_DST,
            format: TextureFormat::Depth32Float,
            ..TextureDesc::rgba8_2d(4, 4)
        }),
        invalid("a depth texture cannot have COPY_DST")
    );
    // Metal cannot write either depth format from a shader.
    for format in [
        TextureFormat::Depth32Float,
        TextureFormat::Depth24PlusStencil8,
    ] {
        assert_eq!(
            gpu.create_texture(&TextureDesc {
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::STORAGE,
                format,
                ..TextureDesc::rgba8_2d(4, 4)
            }),
            invalid("a depth texture cannot have STORAGE"),
            "{format:?}"
        );
    }
    assert_eq!(
        gpu.read_texture(image),
        Err(GpuError::MissingUsage {
            kind: ResourceKind::Texture,
            required: "COPY_SRC",
        })
    );
    let depth = texture(
        &mut gpu,
        TextureFormat::Depth32Float,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    assert_eq!(
        gpu.read_texture(depth),
        invalid("a depth texture cannot be read back")
    );

    // Shaders: Metal compiles the MSL, so its errors are backend errors.
    let Err(GpuError::Backend(message)) = shader(&mut gpu, "this is not MSL") else {
        panic!("bad MSL compiled");
    };
    assert!(message.contains("MSL failed to compile"), "{message}");
    let Err(GpuError::Backend(message)) = gpu.create_shader(&ShaderSource {
        label: None,
        msl: COLORED_MSL,
        wgsl: "",
        glsl: None,
        vertex_entry: "vs_main",
        fragment_entry: "missing",
    }) else {
        panic!("a missing entry point was accepted");
    };
    assert!(message.contains("no function named `missing`"), "{message}");
    assert!(matches!(
        gpu.create_shader(&ShaderSource {
            label: None,
            msl: "",
            wgsl: "@vertex fn vs_main() {}",
            glsl: None,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
        }),
        Err(GpuError::Backend(_))
    ));

    // Pipelines: Metal's attribute table and 4-byte vertex alignment.
    let shader = shader(&mut gpu, COLORED_MSL).unwrap();
    let pipeline_with = |gpu: &mut MetalGpu, stride: u32, location: u32| {
        gpu.create_pipeline(&PipelineDesc {
            label: None,
            shader,
            vertex_layouts: &[VertexLayout {
                array_stride: stride,
                step_mode: VertexStepMode::Vertex,
                attributes: &[VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 0,
                    shader_location: location,
                }],
            }],
            color_format: Some(TextureFormat::Rgba8Unorm),
            blend: None,
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        })
    };
    assert_eq!(
        pipeline_with(&mut gpu, 4, 31),
        invalid("Metal vertex attribute shader_locations must be below 31")
    );
    assert_eq!(
        pipeline_with(&mut gpu, 6, 0),
        invalid("Metal needs vertex array_strides and attribute offsets that are multiples of 4")
    );
    // A pipeline whose vertex function needs attributes it was not given fails in Metal.
    assert!(matches!(
        pipeline_with(&mut gpu, 4, 0),
        Err(GpuError::Backend(_))
    ));

    // Submits.
    let pipeline = colored_pipeline(&mut gpu, None, CullMode::None);
    let uniform = buffer(&mut gpu, BufferUsage::UNIFORM, &[0; 16]);
    let storage = buffer(&mut gpu, BufferUsage::STORAGE, &[0; 32]);
    let sampled = texture(&mut gpu, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);
    let submit = |gpu: &mut MetalGpu, record: &dyn Fn(&mut FrameCommands)| {
        let mut frame = FrameCommands::new();
        record(&mut frame);
        gpu.submit(frame.view())
    };
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK));
        }),
        invalid("a headless MetalGpu has no surface; give the color attachment a target")
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .draw(0..3, 0..1);
        }),
        Err(GpuError::DrawWithoutPipeline)
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertex_only, 0)
                .draw_indexed(0..3, 0, 0..1);
        }),
        Err(GpuError::DrawIndexedWithoutIndexBuffer)
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_vertex_buffer(0, vertex_only, 0)
                .set_pipeline(pipeline)
                .draw(0..3, 0..1);
        }),
        invalid("a draw must read only vertices inside its vertex buffers")
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_uniform_buffer(23, uniform, 0);
        }),
        invalid("Metal uniform and storage bindings must be below 23 (8 + binding < 31)")
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_texture(16, sampled);
        }),
        invalid("Metal texture bindings must be below 16")
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_scissor(ScissorRect {
                    x: 60,
                    y: 0,
                    width: 8,
                    height: 8,
                });
        }),
        invalid("a scissor rectangle must lie within the pass's attachments")
    );

    // A frame that fails validation runs no pass, not even the valid one before the error.
    let blue = ClearColor::rgba(0.0, 0.0, 1.0, 1.0);
    submit(&mut gpu, &|f| {
        f.begin_pass(clear_pass(target, blue));
    })
    .unwrap();
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::rgba(1.0, 0.0, 0.0, 1.0)));
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .draw(0..3, 0..1);
        }),
        Err(GpuError::DrawWithoutPipeline)
    );
    // Metal requires uniform and storage offsets aligned to at least 4 bytes.
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::rgba(1.0, 0.0, 0.0, 1.0)));
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_uniform_buffer(0, uniform, 1);
        }),
        invalid("Metal uniform buffer offsets must be multiples of 256 (4 on Apple GPUs)")
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::rgba(1.0, 0.0, 0.0, 1.0)));
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_storage_buffer(0, storage, 1);
        }),
        invalid("Metal storage buffer offsets must be multiples of 4")
    );
    assert_pixel(&gpu.read_texture(target).unwrap(), (32, 32), BLUE);

    // Destroyed handles of every kind stop resolving.
    gpu.destroy_buffer(readonly).unwrap();
    gpu.destroy_texture(image).unwrap();
    gpu.destroy_shader(shader).unwrap();
    gpu.destroy_pipeline(pipeline).unwrap();
    assert_eq!(
        gpu.write_buffer(readonly, 0, &[0; 4]),
        stale(ResourceKind::Buffer)
    );
    assert_eq!(gpu.destroy_buffer(readonly), stale(ResourceKind::Buffer));
    assert_eq!(gpu.write_texture(image, &[]), stale(ResourceKind::Texture));
    assert_eq!(gpu.destroy_shader(shader), stale(ResourceKind::Shader));
    assert_eq!(pipeline_with(&mut gpu, 4, 0), stale(ResourceKind::Shader));
    assert_eq!(
        gpu.destroy_pipeline(pipeline),
        stale(ResourceKind::Pipeline)
    );
    assert_eq!(
        submit(&mut gpu, &|f| {
            f.begin_pass(clear_pass(target, ClearColor::BLACK))
                .set_pipeline(pipeline);
        }),
        stale(ResourceKind::Pipeline)
    );
}

#[test]
fn writes_wait_for_in_flight_frames_and_destroys_are_safe() {
    let mut gpu = gpu();
    let first = target(&mut gpu);
    let second = target(&mut gpu);
    let scratch = target(&mut gpu);
    let pipeline = colored_pipeline(&mut gpu, None, CullMode::None);
    // One triangle that covers the whole target.
    let covering = |rgba: [f32; 4]| -> Vec<u8> {
        let corners = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
        bytes(
            &corners
                .iter()
                .flat_map(|&[x, y]| [x, y, 0.0].into_iter().chain(rgba))
                .collect::<Vec<f32>>(),
        )
    };
    let vertices = buffer(
        &mut gpu,
        BufferUsage::VERTEX,
        &covering([1.0, 0.0, 0.0, 1.0]),
    );
    let draw_to = |gpu: &mut MetalGpu, target, pipeline, vertices| {
        let mut frame = FrameCommands::new();
        frame
            .begin_pass(clear_pass(target, ClearColor::BLACK))
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertices, 0)
            .draw(0..3, 0..1);
        gpu.submit(frame.view())
    };

    // Rewriting the vertices right after a submit must not change what that frame draws.
    draw_to(&mut gpu, first, pipeline, vertices).unwrap();
    gpu.write_buffer(vertices, 0, &covering([0.0, 1.0, 0.0, 1.0]))
        .unwrap();
    draw_to(&mut gpu, second, pipeline, vertices).unwrap();

    // Destroy everything the in-flight frames use. Handles die at once; Metal keeps the objects
    // alive until the frames finish.
    gpu.destroy_buffer(vertices).unwrap();
    gpu.destroy_pipeline(pipeline).unwrap();
    assert_eq!(
        gpu.write_buffer(vertices, 0, &[0; 4]),
        Err(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        })
    );
    assert_eq!(
        draw_to(&mut gpu, first, pipeline, vertices),
        Err(GpuError::InvalidHandle {
            kind: ResourceKind::Pipeline,
        })
    );

    // More frames than may be in flight at once, into a texture destroyed straight after.
    for i in 0..10 {
        let mut frame = FrameCommands::new();
        frame.begin_pass(clear_pass(
            scratch,
            ClearColor::rgba(f64::from(i) / 10.0, 0.0, 0.0, 1.0),
        ));
        gpu.submit(frame.view()).unwrap();
    }
    gpu.destroy_texture(scratch).unwrap();

    assert_pixel(&gpu.read_texture(first).unwrap(), (32, 32), RED);
    assert_pixel(&gpu.read_texture(second).unwrap(), (32, 32), GREEN);
}
