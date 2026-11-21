//! The binary wire protocol between Rust and `web/titan.js`.
//!
//! Descriptors and frames cross the bridge as byte payloads in WASM memory, which the host
//! decodes with `DataView`. The rules every payload follows:
//!
//! - Numbers are little-endian (`DataView` getters called with `littleEndian = true`).
//! - Enums are one `u8` tag each; [`WireEnum`] lists the tag of every variant.
//! - Strings are a `u32` byte length followed by UTF-8 bytes. An absent optional string (a
//!   label, or a shader's GLSL) is the empty string.
//! - Resources are named by `u32` ids from 1 up. Id [`SURFACE_ID`] names the canvas surface as a
//!   color target.
//!
//! The payload layouts are documented on [`encode_texture_desc`], [`encode_shader_source`],
//! [`encode_pipeline_desc`], [`begin_frame`], [`encode_pass_header`], and [`encode_cmd`].

use alloc::vec::Vec;
use titan_app::{KeyCode, MouseButton};
use titan_gpu::{
    AddressMode, BlendComponent, BlendFactor, BlendOperation, ClearColor, CompareFunction,
    CullMode, DrawCmd, FilterMode, FrontFace, IndexFormat, LoadOp, PipelineDesc, PrimitiveTopology,
    ScissorRect, ShaderSource, StoreOp, TextureDesc, TextureFormat, VertexFormat, VertexStepMode,
    Viewport,
};

/// The color target id that names the canvas surface.
pub const SURFACE_ID: u32 = 0;

/// The `color_format` byte of a pipeline without a color attachment.
pub const NO_COLOR_FORMAT: u8 = 0xFF;

/// The `log` level of an error.
pub const LOG_ERROR: u32 = 0;
/// The `log` level of a warning.
pub const LOG_WARN: u32 = 1;
/// The `log` level of an informational message.
pub const LOG_INFO: u32 = 2;

/// A fieldless enum that crosses the bridge as a one-byte tag.
pub trait WireEnum: Copy {
    /// Returns the variant's tag.
    fn tag(self) -> u8;
    /// Returns the variant with `tag`, or `None` if no variant has it.
    fn from_tag(tag: u8) -> Option<Self>;
}

/// Implements [`WireEnum`] from one list of variants and tags, so the two directions cannot
/// disagree. The `tag` match is exhaustive, so a new variant without a tag fails to compile.
macro_rules! wire_enums {
    ($($Enum:ident { $($Variant:ident = $tag:literal),+ $(,)? })+) => {$(
        impl WireEnum for $Enum {
            fn tag(self) -> u8 {
                match self {
                    $($Enum::$Variant => $tag,)+
                }
            }

            fn from_tag(tag: u8) -> Option<Self> {
                match tag {
                    $($tag => Some($Enum::$Variant),)+
                    _ => None,
                }
            }
        }
    )+};
}

wire_enums! {
    TextureFormat {
        Rgba8Unorm = 0, Rgba8UnormSrgb = 1, Bgra8Unorm = 2, Bgra8UnormSrgb = 3, R8Unorm = 4,
        Depth32Float = 5, Depth24PlusStencil8 = 6,
    }
    FilterMode { Nearest = 0, Linear = 1 }
    AddressMode { ClampToEdge = 0, Repeat = 1, MirrorRepeat = 2 }
    VertexFormat {
        Float32 = 0, Float32x2 = 1, Float32x3 = 2, Float32x4 = 3, Uint32 = 4, Sint32 = 5,
        Unorm8x4 = 6,
    }
    VertexStepMode { Vertex = 0, Instance = 1 }
    PrimitiveTopology {
        TriangleList = 0, TriangleStrip = 1, LineList = 2, LineStrip = 3, PointList = 4,
    }
    FrontFace { Ccw = 0, Cw = 1 }
    CullMode { None = 0, Front = 1, Back = 2 }
    CompareFunction {
        Never = 0, Less = 1, LessEqual = 2, Equal = 3, GreaterEqual = 4, Greater = 5,
        NotEqual = 6, Always = 7,
    }
    BlendFactor {
        Zero = 0, One = 1, SrcAlpha = 2, OneMinusSrcAlpha = 3, DstAlpha = 4,
        OneMinusDstAlpha = 5, SrcColor = 6, OneMinusSrcColor = 7, DstColor = 8,
        OneMinusDstColor = 9,
    }
    BlendOperation { Add = 0, Subtract = 1, ReverseSubtract = 2, Min = 3, Max = 4 }
    IndexFormat { Uint16 = 0, Uint32 = 1 }
    StoreOp { Store = 0, Discard = 1 }
}

const LOAD_OP_LOAD: u8 = 0;
const LOAD_OP_CLEAR: u8 = 1;

const OP_SET_PIPELINE: u8 = 0;
const OP_SET_VERTEX_BUFFER: u8 = 1;
const OP_SET_INDEX_BUFFER: u8 = 2;
const OP_SET_UNIFORM_BUFFER: u8 = 3;
const OP_SET_STORAGE_BUFFER: u8 = 4;
const OP_SET_TEXTURE: u8 = 5;
const OP_SET_VIEWPORT: u8 = 6;
const OP_SET_SCISSOR: u8 = 7;
const OP_DRAW: u8 = 8;
const OP_DRAW_INDEXED: u8 = 9;

/// Returns the key with wire code `code`: its `KeyCode as u8` value.
#[must_use]
pub fn decode_key_code(code: u32) -> Option<KeyCode> {
    KeyCode::ALL.get(usize::try_from(code).ok()?).copied()
}

/// Returns the mouse button with wire code `button`: its `MouseButton as u8` value, so 0 is left,
/// 1 right, and 2 middle. (DOM `PointerEvent.button` numbers middle and right the other way
/// round; `titan.js` maps them.)
#[must_use]
pub fn decode_mouse_button(button: u32) -> Option<MouseButton> {
    MouseButton::ALL.get(usize::try_from(button).ok()?).copied()
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_tag(out: &mut Vec<u8>, value: impl WireEnum) {
    out.push(value.tag());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(
        out,
        u32::try_from(s.len()).expect("a wire string is at most u32::MAX bytes"),
    );
    out.extend_from_slice(s.as_bytes());
}

fn put_optional_str(out: &mut Vec<u8>, s: Option<&str>) {
    put_str(out, s.unwrap_or(""));
}

/// Appends a texture descriptor: `width: u32`, `height: u32`, `mip_level_count: u32`,
/// `format: u8`, `usage: u32` ([`TextureUsage`](titan_gpu::TextureUsage) bits), `min_filter: u8`,
/// `mag_filter: u8`, `address_u: u8`, `address_v: u8`, `label: str`.
pub fn encode_texture_desc(desc: &TextureDesc<'_>, out: &mut Vec<u8>) {
    put_u32(out, desc.width);
    put_u32(out, desc.height);
    put_u32(out, desc.mip_level_count);
    put_tag(out, desc.format);
    put_u32(out, desc.usage.bits());
    put_tag(out, desc.sampler.min_filter);
    put_tag(out, desc.sampler.mag_filter);
    put_tag(out, desc.sampler.address_u);
    put_tag(out, desc.sampler.address_v);
    put_optional_str(out, desc.label);
}

/// Appends a shader source: `wgsl: str`, `glsl: str`, `vertex_entry: str`,
/// `fragment_entry: str`, `label: str`. The browser reads WGSL for WebGPU and GLSL for WebGL2,
/// so MSL stays behind.
pub fn encode_shader_source(source: &ShaderSource<'_>, out: &mut Vec<u8>) {
    put_str(out, source.wgsl);
    put_optional_str(out, source.glsl);
    put_str(out, source.vertex_entry);
    put_str(out, source.fragment_entry);
    put_optional_str(out, source.label);
}

fn put_blend_component(out: &mut Vec<u8>, component: BlendComponent) {
    put_tag(out, component.src_factor);
    put_tag(out, component.dst_factor);
    put_tag(out, component.operation);
}

/// Appends a pipeline descriptor whose shader has id `shader_id`:
///
/// - `shader_id: u32`
/// - `color_format: u8`, or [`NO_COLOR_FORMAT`]
/// - `has_blend: u8`; if 1, the color then alpha components, each `src_factor: u8`,
///   `dst_factor: u8`, `operation: u8`
/// - `has_depth: u8`; if 1, `format: u8`, `depth_write_enabled: u8`, `depth_compare: u8`
/// - `topology: u8`, `front_face: u8`, `cull_mode: u8`
/// - `vertex_layout_count: u8`, then per layout `array_stride: u32`, `step_mode: u8`,
///   `attribute_count: u8`, then per attribute `format: u8`, `offset: u32`,
///   `shader_location: u32`
/// - `label: str`
///
/// # Panics
///
/// Panics if a layout has more than 255 attributes or there are more than 255 layouts. Device
/// validation allows at most [`MAX_VERTEX_BUFFERS`](titan_gpu::MAX_VERTEX_BUFFERS) layouts, and
/// each attribute needs a distinct shader location.
pub fn encode_pipeline_desc(shader_id: u32, desc: &PipelineDesc<'_>, out: &mut Vec<u8>) {
    put_u32(out, shader_id);
    out.push(desc.color_format.map_or(NO_COLOR_FORMAT, WireEnum::tag));
    match desc.blend {
        None => out.push(0),
        Some(blend) => {
            out.push(1);
            put_blend_component(out, blend.color);
            put_blend_component(out, blend.alpha);
        }
    }
    match desc.depth_stencil {
        None => out.push(0),
        Some(depth) => {
            out.push(1);
            put_tag(out, depth.format);
            out.push(u8::from(depth.depth_write_enabled));
            put_tag(out, depth.depth_compare);
        }
    }
    put_tag(out, desc.topology);
    put_tag(out, desc.front_face);
    put_tag(out, desc.cull_mode);
    out.push(u8::try_from(desc.vertex_layouts.len()).expect("at most 255 vertex layouts"));
    for layout in desc.vertex_layouts {
        put_u32(out, layout.array_stride);
        put_tag(out, layout.step_mode);
        out.push(u8::try_from(layout.attributes.len()).expect("at most 255 vertex attributes"));
        for attribute in layout.attributes {
            put_tag(out, attribute.format);
            put_u32(out, attribute.offset);
            put_u32(out, attribute.shader_location);
        }
    }
    put_optional_str(out, desc.label);
}

/// A pass's color attachment, with its target resolved to a wire id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireColor {
    /// The target texture's id, or [`SURFACE_ID`].
    pub target: u32,
    /// What the pass starts with.
    pub load_op: LoadOp<ClearColor>,
    /// Whether the pass's output is kept.
    pub store_op: StoreOp,
}

/// A pass's depth attachment, with its target resolved to a wire id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireDepth {
    /// The target texture's id.
    pub target: u32,
    /// What the pass starts with.
    pub load_op: LoadOp<f32>,
    /// Whether the pass's depth output is kept.
    pub store_op: StoreOp,
}

/// A [`RenderCmd`](titan_gpu::RenderCmd) with its handles resolved to wire ids and its offsets
/// narrowed to `u32`, which every validated offset fits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WireCmd {
    /// Opcode 0: `pipeline: u32`.
    SetPipeline {
        /// The pipeline's id.
        pipeline: u32,
    },
    /// Opcode 1: `slot: u32`, `buffer: u32`, `offset: u32`.
    SetVertexBuffer {
        /// The vertex buffer slot.
        slot: u32,
        /// The buffer's id.
        buffer: u32,
        /// The byte offset.
        offset: u32,
    },
    /// Opcode 2: `buffer: u32`, `format: u8`, `offset: u32`.
    SetIndexBuffer {
        /// The buffer's id.
        buffer: u32,
        /// The index format.
        format: IndexFormat,
        /// The byte offset.
        offset: u32,
    },
    /// Opcode 3: `binding: u32`, `buffer: u32`, `offset: u32`.
    SetUniformBuffer {
        /// The shader binding.
        binding: u32,
        /// The buffer's id.
        buffer: u32,
        /// The byte offset.
        offset: u32,
    },
    /// Opcode 4: `binding: u32`, `buffer: u32`, `offset: u32`.
    SetStorageBuffer {
        /// The shader binding.
        binding: u32,
        /// The buffer's id.
        buffer: u32,
        /// The byte offset.
        offset: u32,
    },
    /// Opcode 5: `binding: u32`, `texture: u32`.
    SetTexture {
        /// The shader binding.
        binding: u32,
        /// The texture's id.
        texture: u32,
    },
    /// Opcode 6: `x`, `y`, `width`, `height`, `min_depth`, `max_depth`, all `f32`.
    SetViewport(Viewport),
    /// Opcode 7: `x`, `y`, `width`, `height`, all `u32`.
    SetScissor(ScissorRect),
    /// Opcode 8 for [`DrawCmd::NonIndexed`]: `vertex_count`, `instance_count`, `first_vertex`,
    /// `first_instance`, all `u32`. Opcode 9 for [`DrawCmd::Indexed`]: `index_count: u32`,
    /// `instance_count: u32`, `first_index: u32`, `base_vertex: i32`, `first_instance: u32`.
    Draw(DrawCmd),
}

/// Clears `out` and starts a frame payload with its `pass_count: u32`. Each pass follows as an
/// [`encode_pass_header`] and its commands.
pub fn begin_frame(pass_count: u32, out: &mut Vec<u8>) {
    out.clear();
    put_u32(out, pass_count);
}

/// Appends a pass header:
///
/// - `has_color: u8`; if 1, `target: u32`, `load_op: u8` (0 load, 1 clear followed by `r`, `g`,
///   `b`, `a`, all `f64`), `store_op: u8`
/// - `has_depth: u8`; if 1, `target: u32`, `load_op: u8` (0 load, 1 clear followed by
///   `value: f32`), `store_op: u8`
/// - `command_count: u32`, the number of [`encode_cmd`] commands that follow
pub fn encode_pass_header(
    color: Option<WireColor>,
    depth: Option<WireDepth>,
    command_count: u32,
    out: &mut Vec<u8>,
) {
    match color {
        None => out.push(0),
        Some(color) => {
            out.push(1);
            put_u32(out, color.target);
            match color.load_op {
                LoadOp::Load => out.push(LOAD_OP_LOAD),
                LoadOp::Clear(c) => {
                    out.push(LOAD_OP_CLEAR);
                    for channel in [c.r, c.g, c.b, c.a] {
                        put_f64(out, channel);
                    }
                }
            }
            put_tag(out, color.store_op);
        }
    }
    match depth {
        None => out.push(0),
        Some(depth) => {
            out.push(1);
            put_u32(out, depth.target);
            match depth.load_op {
                LoadOp::Load => out.push(LOAD_OP_LOAD),
                LoadOp::Clear(value) => {
                    out.push(LOAD_OP_CLEAR);
                    put_f32(out, value);
                }
            }
            put_tag(out, depth.store_op);
        }
    }
    put_u32(out, command_count);
}

/// Appends one command: its `opcode: u8`, then the operands documented on its [`WireCmd`]
/// variant.
pub fn encode_cmd(cmd: WireCmd, out: &mut Vec<u8>) {
    match cmd {
        WireCmd::SetPipeline { pipeline } => {
            out.push(OP_SET_PIPELINE);
            put_u32(out, pipeline);
        }
        WireCmd::SetVertexBuffer {
            slot,
            buffer,
            offset,
        } => {
            out.push(OP_SET_VERTEX_BUFFER);
            for value in [slot, buffer, offset] {
                put_u32(out, value);
            }
        }
        WireCmd::SetIndexBuffer {
            buffer,
            format,
            offset,
        } => {
            out.push(OP_SET_INDEX_BUFFER);
            put_u32(out, buffer);
            put_tag(out, format);
            put_u32(out, offset);
        }
        WireCmd::SetUniformBuffer {
            binding,
            buffer,
            offset,
        } => {
            out.push(OP_SET_UNIFORM_BUFFER);
            for value in [binding, buffer, offset] {
                put_u32(out, value);
            }
        }
        WireCmd::SetStorageBuffer {
            binding,
            buffer,
            offset,
        } => {
            out.push(OP_SET_STORAGE_BUFFER);
            for value in [binding, buffer, offset] {
                put_u32(out, value);
            }
        }
        WireCmd::SetTexture { binding, texture } => {
            out.push(OP_SET_TEXTURE);
            put_u32(out, binding);
            put_u32(out, texture);
        }
        WireCmd::SetViewport(v) => {
            out.push(OP_SET_VIEWPORT);
            for value in [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth] {
                put_f32(out, value);
            }
        }
        WireCmd::SetScissor(r) => {
            out.push(OP_SET_SCISSOR);
            for value in [r.x, r.y, r.width, r.height] {
                put_u32(out, value);
            }
        }
        WireCmd::Draw(DrawCmd::NonIndexed {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }) => {
            out.push(OP_DRAW);
            for value in [vertex_count, instance_count, first_vertex, first_instance] {
                put_u32(out, value);
            }
        }
        WireCmd::Draw(DrawCmd::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }) => {
            out.push(OP_DRAW_INDEXED);
            put_u32(out, index_count);
            put_u32(out, instance_count);
            put_u32(out, first_index);
            out.extend_from_slice(&base_vertex.to_le_bytes());
            put_u32(out, first_instance);
        }
    }
}

/// Test-only decoders: the inverse of every encoder above, mirroring what `titan.js` reads.
#[cfg(test)]
pub(crate) mod decode {
    use super::*;
    use alloc::vec::Vec;
    use titan_gpu::{BlendState, DepthStencilState, SamplerDesc, TextureUsage, VertexAttribute};

    /// Reads a payload front to back, panicking on truncated or malformed input.
    pub(crate) struct Reader<'a> {
        bytes: &'a [u8],
    }

    impl<'a> Reader<'a> {
        pub(crate) fn new(bytes: &'a [u8]) -> Self {
            Self { bytes }
        }

        fn take<const N: usize>(&mut self) -> [u8; N] {
            let (head, rest) = self.bytes.split_first_chunk().expect("payload truncated");
            self.bytes = rest;
            *head
        }

        pub(crate) fn u8(&mut self) -> u8 {
            self.take::<1>()[0]
        }

        pub(crate) fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take())
        }

        fn i32(&mut self) -> i32 {
            i32::from_le_bytes(self.take())
        }

        fn f32(&mut self) -> f32 {
            f32::from_le_bytes(self.take())
        }

        fn f64(&mut self) -> f64 {
            f64::from_le_bytes(self.take())
        }

        fn bool(&mut self) -> bool {
            match self.u8() {
                0 => false,
                1 => true,
                other => panic!("bad flag byte {other}"),
            }
        }

        fn tag<T: WireEnum>(&mut self) -> T {
            let tag = self.u8();
            T::from_tag(tag).unwrap_or_else(|| panic!("unknown tag {tag}"))
        }

        fn str(&mut self) -> &'a str {
            let len = self.u32() as usize;
            let (s, rest) = self.bytes.split_at(len);
            self.bytes = rest;
            core::str::from_utf8(s).expect("wire strings are UTF-8")
        }

        fn optional_str(&mut self) -> Option<&'a str> {
            Some(self.str()).filter(|s| !s.is_empty())
        }

        pub(crate) fn is_empty(&self) -> bool {
            self.bytes.is_empty()
        }

        pub(crate) fn texture_desc(&mut self) -> TextureDesc<'a> {
            let width = self.u32();
            let height = self.u32();
            let mip_level_count = self.u32();
            let format = self.tag();
            let usage = self.u32();
            let usage = [
                TextureUsage::SAMPLED,
                TextureUsage::RENDER_ATTACHMENT,
                TextureUsage::COPY_SRC,
                TextureUsage::COPY_DST,
                TextureUsage::STORAGE,
            ]
            .into_iter()
            .filter(|flag| usage & flag.bits() != 0)
            .fold(TextureUsage::EMPTY, |all, flag| all | flag);
            let sampler = SamplerDesc {
                min_filter: self.tag(),
                mag_filter: self.tag(),
                address_u: self.tag(),
                address_v: self.tag(),
            };
            TextureDesc {
                label: self.optional_str(),
                width,
                height,
                mip_level_count,
                format,
                usage,
                sampler,
            }
        }

        pub(crate) fn shader_source(&mut self) -> ShaderSource<'a> {
            ShaderSource {
                wgsl: self.str(),
                glsl: self.optional_str(),
                vertex_entry: self.str(),
                fragment_entry: self.str(),
                label: self.optional_str(),
                msl: "",
            }
        }

        fn blend_component(&mut self) -> BlendComponent {
            BlendComponent {
                src_factor: self.tag(),
                dst_factor: self.tag(),
                operation: self.tag(),
            }
        }

        pub(crate) fn pipeline_desc(&mut self) -> DecodedPipeline<'a> {
            let shader_id = self.u32();
            let color_format = match self.u8() {
                NO_COLOR_FORMAT => None,
                tag => Some(TextureFormat::from_tag(tag).expect("color format tag")),
            };
            let blend = self.bool().then(|| BlendState {
                color: self.blend_component(),
                alpha: self.blend_component(),
            });
            let depth_stencil = self.bool().then(|| DepthStencilState {
                format: self.tag(),
                depth_write_enabled: self.bool(),
                depth_compare: self.tag(),
            });
            let topology = self.tag();
            let front_face = self.tag();
            let cull_mode = self.tag();
            let vertex_layouts = (0..self.u8())
                .map(|_| {
                    let array_stride = self.u32();
                    let step_mode = self.tag();
                    let attributes = (0..self.u8())
                        .map(|_| VertexAttribute {
                            format: self.tag(),
                            offset: self.u32(),
                            shader_location: self.u32(),
                        })
                        .collect();
                    (array_stride, step_mode, attributes)
                })
                .collect();
            DecodedPipeline {
                shader_id,
                color_format,
                blend,
                depth_stencil,
                topology,
                front_face,
                cull_mode,
                vertex_layouts,
                label: self.optional_str(),
            }
        }

        pub(crate) fn frame(&mut self) -> Vec<DecodedPass> {
            (0..self.u32()).map(|_| self.pass()).collect()
        }

        fn load_op<T>(&mut self, clear: impl FnOnce(&mut Self) -> T) -> LoadOp<T> {
            match self.u8() {
                LOAD_OP_LOAD => LoadOp::Load,
                LOAD_OP_CLEAR => LoadOp::Clear(clear(self)),
                other => panic!("bad load op {other}"),
            }
        }

        fn pass(&mut self) -> DecodedPass {
            let color = self.bool().then(|| WireColor {
                target: self.u32(),
                load_op: self.load_op(|r| ClearColor::rgba(r.f64(), r.f64(), r.f64(), r.f64())),
                store_op: self.tag(),
            });
            let depth = self.bool().then(|| WireDepth {
                target: self.u32(),
                load_op: self.load_op(Self::f32),
                store_op: self.tag(),
            });
            let commands = (0..self.u32()).map(|_| self.cmd()).collect();
            DecodedPass {
                color,
                depth,
                commands,
            }
        }

        fn cmd(&mut self) -> WireCmd {
            match self.u8() {
                OP_SET_PIPELINE => WireCmd::SetPipeline {
                    pipeline: self.u32(),
                },
                OP_SET_VERTEX_BUFFER => WireCmd::SetVertexBuffer {
                    slot: self.u32(),
                    buffer: self.u32(),
                    offset: self.u32(),
                },
                OP_SET_INDEX_BUFFER => WireCmd::SetIndexBuffer {
                    buffer: self.u32(),
                    format: self.tag(),
                    offset: self.u32(),
                },
                OP_SET_UNIFORM_BUFFER => WireCmd::SetUniformBuffer {
                    binding: self.u32(),
                    buffer: self.u32(),
                    offset: self.u32(),
                },
                OP_SET_STORAGE_BUFFER => WireCmd::SetStorageBuffer {
                    binding: self.u32(),
                    buffer: self.u32(),
                    offset: self.u32(),
                },
                OP_SET_TEXTURE => WireCmd::SetTexture {
                    binding: self.u32(),
                    texture: self.u32(),
                },
                OP_SET_VIEWPORT => WireCmd::SetViewport(Viewport {
                    x: self.f32(),
                    y: self.f32(),
                    width: self.f32(),
                    height: self.f32(),
                    min_depth: self.f32(),
                    max_depth: self.f32(),
                }),
                OP_SET_SCISSOR => WireCmd::SetScissor(ScissorRect {
                    x: self.u32(),
                    y: self.u32(),
                    width: self.u32(),
                    height: self.u32(),
                }),
                OP_DRAW => WireCmd::Draw(DrawCmd::NonIndexed {
                    vertex_count: self.u32(),
                    instance_count: self.u32(),
                    first_vertex: self.u32(),
                    first_instance: self.u32(),
                }),
                OP_DRAW_INDEXED => WireCmd::Draw(DrawCmd::Indexed {
                    index_count: self.u32(),
                    instance_count: self.u32(),
                    first_index: self.u32(),
                    base_vertex: self.i32(),
                    first_instance: self.u32(),
                }),
                other => panic!("unknown opcode {other}"),
            }
        }
    }

    /// A decoded pipeline payload, owning its vertex layouts.
    #[derive(Debug, PartialEq)]
    pub(crate) struct DecodedPipeline<'a> {
        pub(crate) shader_id: u32,
        pub(crate) color_format: Option<TextureFormat>,
        pub(crate) blend: Option<BlendState>,
        pub(crate) depth_stencil: Option<DepthStencilState>,
        pub(crate) topology: PrimitiveTopology,
        pub(crate) front_face: FrontFace,
        pub(crate) cull_mode: CullMode,
        pub(crate) vertex_layouts: Vec<(u32, VertexStepMode, Vec<VertexAttribute>)>,
        pub(crate) label: Option<&'a str>,
    }

    /// A decoded pass: its header and commands.
    #[derive(Debug, PartialEq)]
    pub(crate) struct DecodedPass {
        pub(crate) color: Option<WireColor>,
        pub(crate) depth: Option<WireDepth>,
        pub(crate) commands: Vec<WireCmd>,
    }
}

#[cfg(test)]
mod tests {
    use super::decode::{DecodedPass, DecodedPipeline, Reader};
    use super::*;
    use alloc::vec;
    use titan_gpu::{
        BlendState, DepthStencilState, ResourcePool, SamplerDesc, Shader, TextureUsage,
        VertexAttribute, VertexLayout,
    };

    /// Every variant of `T`, listed by hand so a tag change shows up as a test diff.
    fn assert_tags<T: WireEnum + PartialEq + core::fmt::Debug>(expected: &[(T, u8)]) {
        for (i, &(value, tag)) in expected.iter().enumerate() {
            assert_eq!(value.tag(), tag, "{value:?}");
            assert_eq!(T::from_tag(tag), Some(value));
            assert_eq!(usize::from(tag), i, "tags are dense from 0");
        }
        let past_end = u8::try_from(expected.len()).unwrap();
        assert_eq!(T::from_tag(past_end), None);
        assert_eq!(T::from_tag(0xFF), None);
    }

    #[test]
    fn enum_tags_match_the_documented_protocol() {
        use TextureFormat as F;
        assert_tags(&[
            (F::Rgba8Unorm, 0),
            (F::Rgba8UnormSrgb, 1),
            (F::Bgra8Unorm, 2),
            (F::Bgra8UnormSrgb, 3),
            (F::R8Unorm, 4),
            (F::Depth32Float, 5),
            (F::Depth24PlusStencil8, 6),
        ]);
        assert_tags(&[(FilterMode::Nearest, 0), (FilterMode::Linear, 1)]);
        assert_tags(&[
            (AddressMode::ClampToEdge, 0),
            (AddressMode::Repeat, 1),
            (AddressMode::MirrorRepeat, 2),
        ]);
        use VertexFormat as V;
        assert_tags(&[
            (V::Float32, 0),
            (V::Float32x2, 1),
            (V::Float32x3, 2),
            (V::Float32x4, 3),
            (V::Uint32, 4),
            (V::Sint32, 5),
            (V::Unorm8x4, 6),
        ]);
        assert_tags(&[(VertexStepMode::Vertex, 0), (VertexStepMode::Instance, 1)]);
        use PrimitiveTopology as P;
        assert_tags(&[
            (P::TriangleList, 0),
            (P::TriangleStrip, 1),
            (P::LineList, 2),
            (P::LineStrip, 3),
            (P::PointList, 4),
        ]);
        assert_tags(&[(FrontFace::Ccw, 0), (FrontFace::Cw, 1)]);
        assert_tags(&[
            (CullMode::None, 0),
            (CullMode::Front, 1),
            (CullMode::Back, 2),
        ]);
        use CompareFunction as C;
        assert_tags(&[
            (C::Never, 0),
            (C::Less, 1),
            (C::LessEqual, 2),
            (C::Equal, 3),
            (C::GreaterEqual, 4),
            (C::Greater, 5),
            (C::NotEqual, 6),
            (C::Always, 7),
        ]);
        use BlendFactor as B;
        assert_tags(&[
            (B::Zero, 0),
            (B::One, 1),
            (B::SrcAlpha, 2),
            (B::OneMinusSrcAlpha, 3),
            (B::DstAlpha, 4),
            (B::OneMinusDstAlpha, 5),
            (B::SrcColor, 6),
            (B::OneMinusSrcColor, 7),
            (B::DstColor, 8),
            (B::OneMinusDstColor, 9),
        ]);
        use BlendOperation as O;
        assert_tags(&[
            (O::Add, 0),
            (O::Subtract, 1),
            (O::ReverseSubtract, 2),
            (O::Min, 3),
            (O::Max, 4),
        ]);
        assert_tags(&[(IndexFormat::Uint16, 0), (IndexFormat::Uint32, 1)]);
        assert_tags(&[(StoreOp::Store, 0), (StoreOp::Discard, 1)]);
    }

    #[test]
    fn every_key_and_button_round_trips_through_its_wire_code() {
        assert_eq!(KeyCode::ALL.len(), 65);
        for (code, &key) in KeyCode::ALL.iter().enumerate() {
            assert_eq!(key as u8 as usize, code);
            assert_eq!(decode_key_code(code as u32), Some(key));
        }
        assert_eq!(decode_key_code(65), None);
        assert_eq!(decode_key_code(u32::MAX), None);
        assert_eq!(decode_mouse_button(0), Some(MouseButton::Left));
        assert_eq!(decode_mouse_button(1), Some(MouseButton::Right));
        assert_eq!(decode_mouse_button(2), Some(MouseButton::Middle));
        assert_eq!(decode_mouse_button(3), None);
    }

    #[test]
    fn texture_desc_round_trips() {
        let descs = [
            TextureDesc {
                label: Some("albédo"),
                ..TextureDesc::rgba8_2d(640, 480)
            },
            TextureDesc {
                label: None,
                width: 8192,
                height: 1,
                mip_level_count: 14,
                format: TextureFormat::Depth24PlusStencil8,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
                sampler: SamplerDesc {
                    min_filter: FilterMode::Nearest,
                    mag_filter: FilterMode::Linear,
                    address_u: AddressMode::Repeat,
                    address_v: AddressMode::MirrorRepeat,
                },
            },
        ];
        for desc in descs {
            let mut out = Vec::new();
            encode_texture_desc(&desc, &mut out);
            let mut reader = Reader::new(&out);
            assert_eq!(reader.texture_desc(), desc);
            assert!(reader.is_empty());
        }
    }

    #[test]
    fn shader_source_round_trips_with_and_without_glsl_but_drops_msl() {
        for glsl in [Some("void main() {}"), None] {
            let source = ShaderSource {
                label: Some("sprite"),
                msl: "kernel void unused() {}",
                wgsl: "@vertex fn vs() {}",
                glsl,
                vertex_entry: "vs",
                fragment_entry: "fs",
            };
            let mut out = Vec::new();
            encode_shader_source(&source, &mut out);
            let mut reader = Reader::new(&out);
            assert_eq!(reader.shader_source(), ShaderSource { msl: "", ..source });
            assert!(reader.is_empty());
        }
    }

    /// `web/titan.test.mjs` decodes these exact bytes, so both sides agree on the layout.
    #[test]
    fn shader_source_matches_the_bytes_titan_js_decodes() {
        let encode = |glsl| {
            let mut out = Vec::new();
            let source = ShaderSource {
                label: Some("l"),
                msl: "m",
                wgsl: "w",
                glsl,
                vertex_entry: "vs",
                fragment_entry: "fs",
            };
            encode_shader_source(&source, &mut out);
            out
        };
        assert_eq!(
            encode(Some("g")),
            b"\x01\0\0\0w\x01\0\0\0g\x02\0\0\0vs\x02\0\0\0fs\x01\0\0\0l"
        );
        assert_eq!(
            encode(None),
            b"\x01\0\0\0w\0\0\0\0\x02\0\0\0vs\x02\0\0\0fs\x01\0\0\0l"
        );
    }

    #[test]
    fn pipeline_desc_round_trips_every_optional_part() {
        let instance_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Unorm8x4,
                offset: 16,
                shader_location: 3,
            },
        ];
        let vertex_attrs = [VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 4,
            shader_location: 0,
        }];
        let layouts = [
            VertexLayout {
                array_stride: 16,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            },
            VertexLayout {
                array_stride: 20,
                step_mode: VertexStepMode::Instance,
                attributes: &instance_attrs,
            },
        ];
        let base = PipelineDesc {
            label: Some("lit"),
            shader: ResourcePool::<Shader, ()>::new().insert(()),
            vertex_layouts: &layouts,
            color_format: Some(TextureFormat::Bgra8UnormSrgb),
            blend: Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::DstColor,
                    dst_factor: BlendFactor::OneMinusSrcColor,
                    operation: BlendOperation::ReverseSubtract,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Max,
                },
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
            }),
            topology: PrimitiveTopology::LineStrip,
            front_face: FrontFace::Cw,
            cull_mode: CullMode::Back,
        };
        let variants = [
            base,
            // Color only, no vertex buffers.
            PipelineDesc {
                label: None,
                vertex_layouts: &[],
                blend: None,
                depth_stencil: None,
                ..base
            },
            // Depth only.
            PipelineDesc {
                color_format: None,
                blend: None,
                depth_stencil: Some(DepthStencilState::DEFAULT_3D),
                ..base
            },
        ];
        for desc in variants {
            let mut out = Vec::new();
            encode_pipeline_desc(77, &desc, &mut out);
            let mut reader = Reader::new(&out);
            let expected = DecodedPipeline {
                shader_id: 77,
                color_format: desc.color_format,
                blend: desc.blend,
                depth_stencil: desc.depth_stencil,
                topology: desc.topology,
                front_face: desc.front_face,
                cull_mode: desc.cull_mode,
                vertex_layouts: desc
                    .vertex_layouts
                    .iter()
                    .map(|l| (l.array_stride, l.step_mode, l.attributes.to_vec()))
                    .collect(),
                label: desc.label,
            };
            assert_eq!(reader.pipeline_desc(), expected);
            assert!(reader.is_empty());
        }
    }

    #[test]
    fn frame_round_trips_every_opcode_and_attachment_kind() {
        let commands = vec![
            WireCmd::SetPipeline { pipeline: 9 },
            WireCmd::SetVertexBuffer {
                slot: 7,
                buffer: 2,
                offset: 64,
            },
            WireCmd::SetIndexBuffer {
                buffer: 3,
                format: IndexFormat::Uint16,
                offset: 2,
            },
            WireCmd::SetUniformBuffer {
                binding: 0,
                buffer: 4,
                offset: 256,
            },
            WireCmd::SetStorageBuffer {
                binding: 1,
                buffer: 5,
                offset: u32::MAX,
            },
            WireCmd::SetTexture {
                binding: 2,
                texture: 6,
            },
            WireCmd::SetViewport(Viewport {
                x: -1.5,
                y: 2.25,
                width: 640.0,
                height: 480.0,
                min_depth: 0.25,
                max_depth: 0.75,
            }),
            WireCmd::SetScissor(ScissorRect {
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            }),
            WireCmd::Draw(DrawCmd::NonIndexed {
                vertex_count: 3,
                instance_count: 2,
                first_vertex: 1,
                first_instance: 5,
            }),
            WireCmd::Draw(DrawCmd::Indexed {
                index_count: 6,
                instance_count: 1,
                first_index: 12,
                base_vertex: -4,
                first_instance: 0,
            }),
        ];
        let passes = vec![
            DecodedPass {
                color: Some(WireColor {
                    target: SURFACE_ID,
                    load_op: LoadOp::Clear(ClearColor::rgba(0.1, 0.2, 0.3, 1.0)),
                    store_op: StoreOp::Store,
                }),
                depth: Some(WireDepth {
                    target: 11,
                    load_op: LoadOp::Clear(1.0),
                    store_op: StoreOp::Discard,
                }),
                commands,
            },
            DecodedPass {
                color: Some(WireColor {
                    target: 12,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Discard,
                }),
                depth: None,
                commands: Vec::new(),
            },
            DecodedPass {
                color: None,
                depth: Some(WireDepth {
                    target: 11,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                }),
                commands: vec![WireCmd::Draw(DrawCmd::NonIndexed {
                    vertex_count: 0,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                })],
            },
        ];
        // Stale bytes from an earlier frame must not leak into the next one.
        let mut out = vec![0xAB; 5];
        begin_frame(passes.len() as u32, &mut out);
        for pass in &passes {
            encode_pass_header(pass.color, pass.depth, pass.commands.len() as u32, &mut out);
            for &cmd in &pass.commands {
                encode_cmd(cmd, &mut out);
            }
        }
        let mut reader = Reader::new(&out);
        assert_eq!(reader.frame(), passes);
        assert!(reader.is_empty());
    }
}
