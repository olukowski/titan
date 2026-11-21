//! Render pipeline descriptors: vertex layout, rasterization, blending, and depth testing.

use crate::resource::{Shader, TextureFormat};
use titan_core::Handle;

/// The type of one vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    /// One `f32`.
    Float32,
    /// Two `f32`s, such as a [`Vec2`](titan_core::Vec2).
    Float32x2,
    /// Three `f32`s, such as a [`Vec3`](titan_core::Vec3).
    Float32x3,
    /// Four `f32`s, such as a [`Vec4`](titan_core::Vec4) or [`Color`](titan_core::Color).
    Float32x4,
    /// One `u32`.
    Uint32,
    /// One `i32`.
    Sint32,
    /// Four `u8`s read as `[0, 1]` floats: a packed color.
    Unorm8x4,
}

impl VertexFormat {
    /// Returns the attribute's size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> u32 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 | Self::Unorm8x4 => 4,
            Self::Float32x2 => 8,
            Self::Float32x3 => 12,
            Self::Float32x4 => 16,
        }
    }
}

/// How often a vertex buffer advances to its next element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    /// Once per vertex.
    #[default]
    Vertex,
    /// Once per instance.
    Instance,
}

/// One attribute within a vertex buffer element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// The attribute's type.
    pub format: VertexFormat,
    /// Byte offset from the start of the element.
    pub offset: u32,
    /// The shader input location it feeds.
    pub shader_location: u32,
}

/// The layout of the elements in one vertex buffer slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayout<'a> {
    /// Bytes from one element to the next.
    pub array_stride: u32,
    /// Whether elements advance per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes in each element.
    pub attributes: &'a [VertexAttribute],
}

/// How vertices are assembled into primitives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    /// Every three vertices form a triangle.
    #[default]
    TriangleList,
    /// Each vertex after the first two forms a triangle with the previous two.
    TriangleStrip,
    /// Every two vertices form a line.
    LineList,
    /// Each vertex after the first forms a line with the previous one.
    LineStrip,
    /// Each vertex is a point.
    PointList,
}

/// The vertex winding that makes a triangle front-facing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Counter-clockwise, matching `titan-core`'s right-handed math.
    #[default]
    Ccw,
    /// Clockwise.
    Cw,
}

/// Which triangle faces are discarded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// Draw both faces.
    #[default]
    None,
    /// Discard front faces.
    Front,
    /// Discard back faces.
    Back,
}

/// A comparison between a new value and the stored one, such as a fragment's depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes if new < stored.
    Less,
    /// Passes if new <= stored.
    LessEqual,
    /// Passes if new == stored.
    Equal,
    /// Passes if new >= stored.
    GreaterEqual,
    /// Passes if new > stored.
    Greater,
    /// Passes if new != stored.
    NotEqual,
    /// Always passes.
    Always,
}

/// A multiplier applied to the source or destination in blending.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// 0.
    Zero,
    /// 1.
    One,
    /// Source alpha.
    SrcAlpha,
    /// 1 - source alpha.
    OneMinusSrcAlpha,
    /// Destination alpha.
    DstAlpha,
    /// 1 - destination alpha.
    OneMinusDstAlpha,
    /// Source color.
    SrcColor,
    /// 1 - source color.
    OneMinusSrcColor,
    /// Destination color.
    DstColor,
    /// 1 - destination color.
    OneMinusDstColor,
}

/// How the weighted source and destination are combined in blending.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    /// src + dst.
    #[default]
    Add,
    /// src - dst.
    Subtract,
    /// dst - src.
    ReverseSubtract,
    /// min(src, dst).
    Min,
    /// max(src, dst).
    Max,
}

/// Blending for the color or the alpha channel: `operation(src * src_factor, dst * dst_factor)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// The source (fragment) multiplier.
    pub src_factor: BlendFactor,
    /// The destination (attachment) multiplier.
    pub dst_factor: BlendFactor,
    /// How the weighted terms combine.
    pub operation: BlendOperation,
}

impl BlendComponent {
    const fn add(src_factor: BlendFactor, dst_factor: BlendFactor) -> Self {
        Self {
            src_factor,
            dst_factor,
            operation: BlendOperation::Add,
        }
    }
}

/// How fragments blend into the color attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// Blending for the RGB channels.
    pub color: BlendComponent,
    /// Blending for the alpha channel.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Standard "over" blending for straight (non-premultiplied) alpha.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent::add(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha),
        alpha: BlendComponent::add(BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
    };
    /// "Over" blending for colors already multiplied by their alpha.
    pub const PREMULTIPLIED_ALPHA: Self = Self {
        color: BlendComponent::add(BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
        alpha: BlendComponent::add(BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
    };
    /// Adds the source to the destination, for glows and particles.
    pub const ADDITIVE: Self = Self {
        color: BlendComponent::add(BlendFactor::One, BlendFactor::One),
        alpha: BlendComponent::add(BlendFactor::One, BlendFactor::One),
    };
}

/// Depth testing and writing for a pipeline that renders with a depth attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilState {
    /// The depth attachment's format; must be a depth format.
    pub format: TextureFormat,
    /// Whether passing fragments write their depth.
    pub depth_write_enabled: bool,
    /// The test a fragment's depth must pass against the stored depth.
    pub depth_compare: CompareFunction,
}

impl DepthStencilState {
    /// Opaque 3D geometry: 32-bit float depth, written, nearer fragments win.
    pub const DEFAULT_3D: Self = Self {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
    };
}

/// Describes a render pipeline to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineDesc<'a> {
    /// A name for debuggers and error messages.
    pub label: Option<&'a str>,
    /// The shader whose vertex and fragment entry points the pipeline runs.
    pub shader: Handle<Shader>,
    /// The layout of each vertex buffer slot, by slot index; at most
    /// [`MAX_VERTEX_BUFFERS`](crate::MAX_VERTEX_BUFFERS). Each attribute fits within its
    /// layout's `array_stride`, and no two attributes share a `shader_location`.
    pub vertex_layouts: &'a [VertexLayout<'a>],
    /// The color attachment's format, which must not be a depth format, or `None` for a
    /// depth-only pipeline. It must match the format of each pass the pipeline draws in.
    pub color_format: Option<TextureFormat>,
    /// Blending into the color attachment, or `None` to replace it.
    pub blend: Option<BlendState>,
    /// Depth testing, or `None` for passes without a depth attachment. A pipeline needs this or
    /// a `color_format`.
    pub depth_stencil: Option<DepthStencilState>,
    /// How vertices form primitives.
    pub topology: PrimitiveTopology,
    /// The winding of front faces.
    pub front_face: FrontFace,
    /// Which faces to discard.
    pub cull_mode: CullMode,
}
