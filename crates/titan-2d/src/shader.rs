//! The sprite shader and its vertex format.

use alloc::vec::Vec;
use core::mem::{offset_of, size_of};
use titan_gpu::{ShaderSource, VertexAttribute, VertexFormat, VertexLayout, VertexStepMode};

/// One corner of a sprite quad, as the sprite shader reads it from vertex buffer slot 0.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Vertex2D {
    /// World-space position; shader location 0.
    pub position: [f32; 2],
    /// Texture coordinate, `(0, 0)` at the texture's top-left; shader location 1.
    pub uv: [f32; 2],
    /// Tint multiplied with the texture sample; shader location 2.
    pub color: [f32; 4],
}

impl Vertex2D {
    /// The size of one vertex in bytes, which is also the vertex buffer stride.
    pub const SIZE_BYTES: u32 = 32;

    /// The attributes, matching the field offsets and the shaders' locations.
    pub const ATTRIBUTES: [VertexAttribute; 3] = [
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
        VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 16,
            shader_location: 2,
        },
    ];

    /// The layout of vertex buffer slot 0.
    pub const LAYOUT: VertexLayout<'static> = VertexLayout {
        array_stride: Self::SIZE_BYTES,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };

    /// Appends the vertex's [`SIZE_BYTES`](Self::SIZE_BYTES) bytes to `out` in the layout the
    /// GPU reads: every field in order, as little-endian `f32`s.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        for value in self.position.iter().chain(&self.uv).chain(&self.color) {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

// `ATTRIBUTES` and `write_bytes` hard-code the field layout; keep them honest.
const _: () = {
    assert!(size_of::<Vertex2D>() == Vertex2D::SIZE_BYTES as usize);
    assert!(offset_of!(Vertex2D, position) == Vertex2D::ATTRIBUTES[0].offset as usize);
    assert!(offset_of!(Vertex2D, uv) == Vertex2D::ATTRIBUTES[1].offset as usize);
    assert!(offset_of!(Vertex2D, color) == Vertex2D::ATTRIBUTES[2].offset as usize);
};

/// The sprite shader in WGSL. It binds the camera's `view_proj` matrix as uniform binding 0 and
/// the sprite texture as texture binding 0, and outputs the texture sample times the vertex
/// color.
pub const SPRITE_SHADER_WGSL: &str = r"struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(sprite_texture, sprite_sampler, f.uv) * f.color;
}
";

/// The sprite shader in Metal Shading Language, with the same bindings as
/// [`SPRITE_SHADER_WGSL`] under the [binding convention](titan_gpu::ShaderSource#binding-convention).
pub const SPRITE_SHADER_MSL: &str = r"#include <metal_stdlib>
using namespace metal;

struct CameraUniform {
    float4x4 view_proj;
};

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct VertexOut {
    float4 clip_position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut vs_main(VertexIn in [[stage_in]], constant CameraUniform& camera [[buffer(8)]]) {
    VertexOut out;
    out.clip_position = camera.view_proj * float4(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 fs_main(VertexOut in [[stage_in]], texture2d<float> sprite_tex [[texture(0)]], sampler sprite_sampler [[sampler(0)]]) {
    return sprite_tex.sample(sprite_sampler, in.uv) * in.color;
}
";

/// The sprite shader in GLSL ES 3.00, for WebGL2, with the same inputs and outputs as
/// [`SPRITE_SHADER_WGSL`].
///
/// # Source convention
///
/// One source holds both stages. It has no `#version` line: a backend compiles each stage from
/// `#version 300 es`, then `#define TITAN_VERTEX` or `#define TITAN_FRAGMENT`, then this source,
/// which wraps each stage in `#ifdef` of its macro. Each stage's entry point is `main`, so GLSL
/// ignores [`ShaderSource::vertex_entry`] and [`ShaderSource::fragment_entry`].
///
/// GLSL ES 3.00 has no `layout(binding = N)`, so a name carries its binding instead: a uniform
/// block named `<name>_binding<N>` is uniform buffer binding `N`, and a `sampler2D` uniform named
/// `<name>_binding<N>` is texture binding `N`. The backend reads those suffixes after linking and
/// calls `uniformBlockBinding` and `uniform1i` (texture unit `N`) to match. Vertex attributes keep
/// `layout(location = N)`, as in the other languages.
///
/// # Clip space and textures
///
/// - Clip-space `+Y` is up in GLSL as in WGSL and MSL, so the view-projection matrix needs no flip
///   when drawing to the canvas.
/// - GL clips `z` to `[-w, w]` and maps it to depth `[0, 1]`, where WebGPU and Metal clip to
///   `[0, w]`. [`Camera2D`](crate::Camera2D) puts sprites at clip `z = 0.5`, inside both ranges;
///   the sprite pipeline does not depth-test, so the different depth value has no effect.
/// - Texture `v` grows with the uploaded row index, so `v = 0` is the image's top row, as long as
///   textures are uploaded without `UNPACK_FLIP_Y_WEBGL`.
pub const SPRITE_SHADER_GLSL: &str = r"#ifdef TITAN_VERTEX
layout(std140) uniform CameraUniform_binding0 {
    mat4 view_proj;
} camera;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

out vec2 v_uv;
out vec4 v_color;

void main() {
    gl_Position = camera.view_proj * vec4(position, 0.0, 1.0);
    v_uv = uv;
    v_color = color;
}
#endif

#ifdef TITAN_FRAGMENT
precision highp float;

uniform highp sampler2D sprite_texture_binding0;

in vec2 v_uv;
in vec4 v_color;

layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = texture(sprite_texture_binding0, v_uv) * v_color;
}
#endif
";

/// The sprite shader, with entry points `vs_main` and `fs_main`.
pub const SPRITE_SHADER: ShaderSource<'static> = ShaderSource {
    label: Some("titan-2d sprite"),
    msl: SPRITE_SHADER_MSL,
    wgsl: SPRITE_SHADER_WGSL,
    glsl: Some(SPRITE_SHADER_GLSL),
    vertex_entry: "vs_main",
    fragment_entry: "fs_main",
};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn write_bytes_emits_fields_in_attribute_order() {
        let vertex = Vertex2D {
            position: [1.0, -2.0],
            uv: [0.25, 0.75],
            color: [0.1, 0.2, 0.3, 0.4],
        };
        let mut out = Vec::new();
        vertex.write_bytes(&mut out);
        assert_eq!(out.len(), Vertex2D::SIZE_BYTES as usize);
        let (chunks, []) = out.as_chunks::<4>() else {
            panic!("{} bytes is not a whole number of f32s", out.len());
        };
        let floats: Vec<f32> = chunks.iter().map(|&b| f32::from_le_bytes(b)).collect();
        assert_eq!(floats, [1.0, -2.0, 0.25, 0.75, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn glsl_declares_every_vertex_attribute_and_both_stages() {
        assert_eq!(SPRITE_SHADER.glsl, Some(SPRITE_SHADER_GLSL));
        // The backend prepends `#version`, which must be a shader's first line.
        assert!(!SPRITE_SHADER_GLSL.contains("#version"));
        for stage in ["#ifdef TITAN_VERTEX", "#ifdef TITAN_FRAGMENT"] {
            assert!(SPRITE_SHADER_GLSL.contains(stage), "{stage}");
        }
        for attribute in Vertex2D::ATTRIBUTES {
            let ty = match attribute.format {
                VertexFormat::Float32x2 => "vec2",
                VertexFormat::Float32x4 => "vec4",
                other => panic!("no GLSL type for {other:?}"),
            };
            let declaration = format!("layout(location = {}) in {ty} ", attribute.shader_location);
            assert!(SPRITE_SHADER_GLSL.contains(&declaration), "{declaration}");
        }
    }
}
