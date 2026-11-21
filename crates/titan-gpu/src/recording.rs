//! [`RecordingGpu`]: a strict, CPU-only [`GpuDevice`] that records everything it is asked to do.

use crate::command::{
    DrawCmd, FrameView, IndexFormat, LoadOp, RenderCmd, RenderPass, RenderPassDesc,
};
use crate::device::{GpuDevice, GpuError, ResourceKind};
use crate::pipeline::{
    BlendState, CullMode, DepthStencilState, FrontFace, PipelineDesc, PrimitiveTopology,
    VertexAttribute, VertexStepMode,
};
use crate::resource::{
    Buffer, BufferDesc, BufferUsage, MAX_VERTEX_BUFFERS, Pipeline, ResourcePool, SamplerDesc,
    Shader, ShaderSource, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use titan_core::Handle;

/// A buffer as [`RecordingGpu`] stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedBuffer {
    /// The descriptor's label.
    pub label: Option<String>,
    /// The size in bytes.
    pub size: u64,
    /// The usage flags.
    pub usage: BufferUsage,
    /// The contents: `size` bytes, zeroed at creation and updated by every write.
    pub data: Vec<u8>,
}

/// A texture as [`RecordingGpu`] stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedTexture {
    /// The descriptor's label.
    pub label: Option<String>,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// The number of mip levels.
    pub mip_level_count: u32,
    /// The texel format.
    pub format: TextureFormat,
    /// The usage flags.
    pub usage: TextureUsage,
    /// The sampler.
    pub sampler: SamplerDesc,
    /// The last data written to mip level 0; empty until the first write.
    pub data: Vec<u8>,
}

/// A shader as [`RecordingGpu`] stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedShader {
    /// The source's label.
    pub label: Option<String>,
    /// Metal Shading Language source.
    pub msl: String,
    /// WGSL source.
    pub wgsl: String,
    /// GLSL ES source.
    pub glsl: Option<String>,
    /// The vertex entry point.
    pub vertex_entry: String,
    /// The fragment entry point.
    pub fragment_entry: String,
}

/// An owned [`VertexLayout`](crate::VertexLayout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedVertexLayout {
    /// Bytes from one element to the next.
    pub array_stride: u32,
    /// Whether elements advance per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes in each element.
    pub attributes: Vec<VertexAttribute>,
}

/// A pipeline as [`RecordingGpu`] stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedPipeline {
    /// The descriptor's label.
    pub label: Option<String>,
    /// The shader it was created from, which may since have been destroyed.
    pub shader: Handle<Shader>,
    /// The vertex buffer layouts, by slot.
    pub vertex_layouts: Vec<RecordedVertexLayout>,
    /// The color attachment format, or `None` for a depth-only pipeline.
    pub color_format: Option<TextureFormat>,
    /// Blending into the color attachment.
    pub blend: Option<BlendState>,
    /// Depth testing.
    pub depth_stencil: Option<DepthStencilState>,
    /// How vertices form primitives.
    pub topology: PrimitiveTopology,
    /// The winding of front faces.
    pub front_face: FrontFace,
    /// Which faces are discarded.
    pub cull_mode: CullMode,
}

/// An owned [`RenderPass`].
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedPass {
    /// The pass's attachments.
    pub desc: RenderPassDesc,
    /// The pass's commands.
    pub commands: Vec<RenderCmd>,
}

/// A submitted frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedFrame {
    /// The frame's passes in order.
    pub passes: Vec<RecordedPass>,
}

/// One successful [`GpuDevice`] call, in the order [`RecordingGpu`] received it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuEvent {
    /// A buffer was created.
    CreateBuffer(Handle<Buffer>),
    /// `len` bytes were written to a buffer at `offset`.
    WriteBuffer {
        /// The buffer.
        buffer: Handle<Buffer>,
        /// The byte offset.
        offset: u64,
        /// The number of bytes.
        len: usize,
    },
    /// A buffer was destroyed.
    DestroyBuffer(Handle<Buffer>),
    /// A texture was created.
    CreateTexture(Handle<Texture>),
    /// Mip level 0 of a texture was replaced with `len` bytes.
    WriteTexture {
        /// The texture.
        texture: Handle<Texture>,
        /// The number of bytes.
        len: usize,
    },
    /// A texture was destroyed.
    DestroyTexture(Handle<Texture>),
    /// A shader was created.
    CreateShader(Handle<Shader>),
    /// A shader was destroyed.
    DestroyShader(Handle<Shader>),
    /// A pipeline was created.
    CreatePipeline(Handle<Pipeline>),
    /// A pipeline was destroyed.
    DestroyPipeline(Handle<Pipeline>),
    /// A frame was submitted.
    Submit {
        /// Its index in [`RecordingGpu::frames`].
        frame: usize,
    },
}

/// The number of live resources of each kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceCounts {
    /// Live buffers.
    pub buffers: usize,
    /// Live textures.
    pub textures: usize,
    /// Live shaders.
    pub shaders: usize,
    /// Live pipelines.
    pub pipelines: usize,
}

/// A [`GpuDevice`] that draws nothing but validates and records every call, for tests and
/// headless runs.
///
/// It checks the rules this crate documents: live handles, usage flags, descriptor ranges,
/// WebGPU's default size limits and write alignment, pipeline and attachment compatibility within
/// each pass, and that draws stay inside their bound buffers. It cannot check shader source,
/// indexed vertex fetches, or a device's own limits, so a backend may still reject code that
/// runs cleanly here. A failed call changes nothing and records nothing; in particular,
/// [`submit`](GpuDevice::submit) checks the whole frame before recording any of it.
#[derive(Debug)]
pub struct RecordingGpu {
    surface_format: TextureFormat,
    buffers: ResourcePool<Buffer, RecordedBuffer>,
    textures: ResourcePool<Texture, RecordedTexture>,
    shaders: ResourcePool<Shader, RecordedShader>,
    pipelines: ResourcePool<Pipeline, RecordedPipeline>,
    frames: Vec<RecordedFrame>,
    events: Vec<GpuEvent>,
}

impl Default for RecordingGpu {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordingGpu {
    /// The largest buffer, in bytes: WebGPU's default `maxBufferSize`, so code that fits here
    /// fits everywhere.
    pub const MAX_BUFFER_SIZE: u64 = 256 << 20;
    /// The largest texture width or height: WebGPU's default `maxTextureDimension2D`.
    pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

    /// Creates a device whose surface is [`TextureFormat::Bgra8UnormSrgb`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_surface_format(TextureFormat::Bgra8UnormSrgb)
    }

    /// Creates a device whose surface has `format`.
    ///
    /// # Panics
    ///
    /// Panics if `format` is a depth format.
    #[must_use]
    pub fn with_surface_format(format: TextureFormat) -> Self {
        assert!(!format.is_depth(), "the surface needs a color format");
        Self {
            surface_format: format,
            buffers: ResourcePool::new(),
            textures: ResourcePool::new(),
            shaders: ResourcePool::new(),
            pipelines: ResourcePool::new(),
            frames: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Returns a live buffer, or `None` if `handle` is stale.
    #[must_use]
    pub fn buffer(&self, handle: Handle<Buffer>) -> Option<&RecordedBuffer> {
        self.buffers.get(handle)
    }

    /// Returns a live texture, or `None` if `handle` is stale.
    #[must_use]
    pub fn texture(&self, handle: Handle<Texture>) -> Option<&RecordedTexture> {
        self.textures.get(handle)
    }

    /// Returns a live shader, or `None` if `handle` is stale.
    #[must_use]
    pub fn shader(&self, handle: Handle<Shader>) -> Option<&RecordedShader> {
        self.shaders.get(handle)
    }

    /// Returns a live pipeline, or `None` if `handle` is stale.
    #[must_use]
    pub fn pipeline(&self, handle: Handle<Pipeline>) -> Option<&RecordedPipeline> {
        self.pipelines.get(handle)
    }

    /// Returns every frame submitted since creation or the last
    /// [`clear_history`](Self::clear_history).
    #[must_use]
    pub fn frames(&self) -> &[RecordedFrame] {
        &self.frames
    }

    /// Returns the most recently submitted frame.
    #[must_use]
    pub fn last_frame(&self) -> Option<&RecordedFrame> {
        self.frames.last()
    }

    /// Returns every successful call since creation or the last
    /// [`clear_history`](Self::clear_history).
    #[must_use]
    pub fn events(&self) -> &[GpuEvent] {
        &self.events
    }

    /// Forgets recorded frames and events, keeping live resources.
    pub fn clear_history(&mut self) {
        self.frames.clear();
        self.events.clear();
    }

    /// Returns the number of live resources of each kind.
    #[must_use]
    pub fn live_counts(&self) -> ResourceCounts {
        ResourceCounts {
            buffers: self.buffers.len(),
            textures: self.textures.len(),
            shaders: self.shaders.len(),
            pipelines: self.pipelines.len(),
        }
    }

    /// Returns a live buffer that has `usage`, whose name is `flag`.
    fn buffer_with(
        &self,
        handle: Handle<Buffer>,
        usage: BufferUsage,
        flag: &'static str,
    ) -> Result<&RecordedBuffer, GpuError> {
        let buffer = self.buffers.get(handle).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        })?;
        if !buffer.usage.contains(usage) {
            return Err(GpuError::MissingUsage {
                kind: ResourceKind::Buffer,
                required: flag,
            });
        }
        Ok(buffer)
    }

    /// Returns a live texture that has `usage`, whose name is `flag`.
    fn texture_with(
        &self,
        handle: Handle<Texture>,
        usage: TextureUsage,
        flag: &'static str,
    ) -> Result<&RecordedTexture, GpuError> {
        let texture = self.textures.get(handle).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Texture,
        })?;
        if !texture.usage.contains(usage) {
            return Err(GpuError::MissingUsage {
                kind: ResourceKind::Texture,
                required: flag,
            });
        }
        Ok(texture)
    }

    /// Checks a buffer binding: the buffer is live, has `usage`, and `offset` is within it.
    /// Returns the bytes bound: those from `offset` to the end of the buffer.
    fn check_binding(
        &self,
        handle: Handle<Buffer>,
        offset: u64,
        usage: BufferUsage,
        flag: &'static str,
    ) -> Result<u64, GpuError> {
        let buffer = self.buffer_with(handle, usage, flag)?;
        buffer
            .size
            .checked_sub(offset)
            .ok_or(GpuError::BufferOutOfBounds {
                offset,
                write_len: 0,
                buffer_size: buffer.size,
            })
    }

    fn check_pass(&self, pass: RenderPass<'_>) -> Result<(), GpuError> {
        let RenderPassDesc { color, depth, .. } = pass.desc;
        ensure(
            color.is_some() || depth.is_some(),
            "a render pass needs a color or depth attachment",
        )?;
        // What draws must match: the attachment formats, and the size when a texture is attached
        // (the surface's size is not known here).
        let mut color_format = None;
        let mut size = None;
        if let Some(color) = color {
            color_format = Some(match color.target {
                // The surface's format is a color format by construction.
                None => self.surface_format,
                Some(target) => {
                    let texture = self.texture_with(
                        target,
                        TextureUsage::RENDER_ATTACHMENT,
                        "RENDER_ATTACHMENT",
                    )?;
                    ensure(
                        !texture.format.is_depth(),
                        "a color attachment needs a color format",
                    )?;
                    size = Some((texture.width, texture.height));
                    texture.format
                }
            });
            if let LoadOp::Clear(c) = color.load_op {
                ensure(
                    [c.r, c.g, c.b, c.a].iter().all(|v| v.is_finite()),
                    "a clear color must be finite",
                )?;
            }
        }
        let mut depth_format = None;
        if let Some(depth) = depth {
            let texture = self.texture_with(
                depth.target,
                TextureUsage::RENDER_ATTACHMENT,
                "RENDER_ATTACHMENT",
            )?;
            ensure(
                texture.format.is_depth(),
                "a depth attachment needs a depth format",
            )?;
            let depth_size = (texture.width, texture.height);
            ensure(
                size.is_none_or(|color_size| color_size == depth_size),
                "a pass's color and depth targets must be the same size",
            )?;
            size = Some(depth_size);
            depth_format = Some(texture.format);
            if let LoadOp::Clear(value) = depth.depth_load_op {
                ensure(
                    (0.0..=1.0).contains(&value),
                    "a depth clear value must be in [0, 1]",
                )?;
            }
        }
        let attachments = [color.and_then(|c| c.target), depth.map(|d| d.target)];

        // Bindings do not carry over between passes, matching Metal and WebGPU encoders.
        let mut pipeline = None;
        // Per slot, the bytes the bound vertex buffer holds from its offset on.
        let mut vertex_buffers = [None; MAX_VERTEX_BUFFERS];
        // The bound index buffer's format and the bytes it holds from its offset on.
        let mut index_buffer = None;
        for &command in pass.commands {
            match command {
                RenderCmd::SetPipeline(handle) => {
                    let recorded = self.pipelines.get(handle).ok_or(GpuError::InvalidHandle {
                        kind: ResourceKind::Pipeline,
                    })?;
                    ensure(
                        recorded.color_format == color_format
                            && recorded.depth_stencil.map(|ds| ds.format) == depth_format,
                        "a pipeline's color_format and depth_stencil format must match its \
                         pass's attachments",
                    )?;
                    pipeline = Some(recorded);
                }
                RenderCmd::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    let slot = usize::try_from(slot)
                        .ok()
                        .filter(|&slot| slot < MAX_VERTEX_BUFFERS)
                        .ok_or(GpuError::InvalidDescriptor {
                            message: "a vertex buffer slot must be below MAX_VERTEX_BUFFERS",
                        })?;
                    vertex_buffers[slot] =
                        Some(self.check_binding(buffer, offset, BufferUsage::VERTEX, "VERTEX")?);
                }
                RenderCmd::SetIndexBuffer {
                    buffer,
                    format,
                    offset,
                } => {
                    let bound = self.check_binding(buffer, offset, BufferUsage::INDEX, "INDEX")?;
                    ensure(
                        offset.is_multiple_of(u64::from(format.size_bytes())),
                        "an index buffer offset must be a multiple of the index size",
                    )?;
                    index_buffer = Some((format, bound));
                }
                RenderCmd::SetUniformBuffer { buffer, offset, .. } => {
                    self.check_binding(buffer, offset, BufferUsage::UNIFORM, "UNIFORM")?;
                }
                RenderCmd::SetStorageBuffer { buffer, offset, .. } => {
                    self.check_binding(buffer, offset, BufferUsage::STORAGE, "STORAGE")?;
                }
                RenderCmd::SetTexture { texture, .. } => {
                    self.texture_with(texture, TextureUsage::SAMPLED, "SAMPLED")?;
                    ensure(
                        !attachments.contains(&Some(texture)),
                        "a pass cannot sample its own color or depth attachment",
                    )?;
                }
                RenderCmd::SetViewport(v) => ensure(
                    [v.x, v.y, v.width, v.height].iter().all(|c| c.is_finite())
                        && v.width > 0.0
                        && v.height > 0.0
                        && 0.0 <= v.min_depth
                        && v.min_depth <= v.max_depth
                        && v.max_depth <= 1.0,
                    "a viewport must be finite, have a positive size, and have \
                     0 <= min_depth <= max_depth <= 1",
                )?,
                RenderCmd::SetScissor(r) => ensure(
                    size.is_none_or(|(width, height)| {
                        u64::from(r.x) + u64::from(r.width) <= u64::from(width)
                            && u64::from(r.y) + u64::from(r.height) <= u64::from(height)
                    }),
                    "a scissor rectangle must lie within the pass's attachments",
                )?,
                RenderCmd::Draw(draw) => check_draw(
                    pipeline.ok_or(GpuError::DrawWithoutPipeline)?,
                    &vertex_buffers,
                    index_buffer,
                    draw,
                )?,
            }
        }
        Ok(())
    }
}

/// Checks that `draw` has the buffers `pipeline` reads and stays inside them. `vertex_buffers`
/// and `index_buffer` hold the bytes bound from each buffer's offset on.
fn check_draw(
    pipeline: &RecordedPipeline,
    vertex_buffers: &[Option<u64>; MAX_VERTEX_BUFFERS],
    index_buffer: Option<(IndexFormat, u64)>,
    draw: DrawCmd,
) -> Result<(), GpuError> {
    // One past the last element a draw reads per vertex and per instance. Vertices an index
    // buffer picks are only known on the GPU, so indexed draws leave per-vertex reads unchecked.
    // Like WebGPU, each range is checked on its own, even when another count makes the draw empty.
    let (vertex_end, instance_end) = match draw {
        DrawCmd::NonIndexed {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => (
            Some(range_end(first_vertex, vertex_count)),
            range_end(first_instance, instance_count),
        ),
        DrawCmd::Indexed {
            index_count,
            instance_count,
            first_index,
            first_instance,
            ..
        } => {
            let (format, bound) = index_buffer.ok_or(GpuError::DrawIndexedWithoutIndexBuffer)?;
            ensure(
                range_end(first_index, index_count)
                    .checked_mul(u64::from(format.size_bytes()))
                    .is_some_and(|bytes| bytes <= bound),
                "an indexed draw must read only indices inside its index buffer",
            )?;
            (None, range_end(first_instance, instance_count))
        }
    };
    for (layout, bound) in pipeline.vertex_layouts.iter().zip(vertex_buffers) {
        let bound = bound.ok_or(GpuError::InvalidDescriptor {
            message: "a draw needs a vertex buffer in every slot its pipeline has a layout for",
        })?;
        let elements = match layout.step_mode {
            VertexStepMode::Vertex => vertex_end,
            VertexStepMode::Instance => Some(instance_end),
        };
        if let Some(elements @ 1..) = elements {
            // The last element ends at its last attribute, not at a full stride.
            let last_element_bytes = layout
                .attributes
                .iter()
                .map(|a| u64::from(a.offset) + u64::from(a.format.size_bytes()))
                .max()
                .unwrap_or(0);
            ensure(
                (elements - 1)
                    .checked_mul(u64::from(layout.array_stride))
                    .and_then(|bytes| bytes.checked_add(last_element_bytes))
                    .is_some_and(|bytes| bytes <= bound),
                "a draw must read only vertices inside its vertex buffers",
            )?;
        }
    }
    Ok(())
}

/// Returns one past the last of `count` elements starting at `first`, without overflow.
fn range_end(first: u32, count: u32) -> u64 {
    u64::from(first) + u64::from(count)
}

/// Returns an [`InvalidDescriptor`](GpuError::InvalidDescriptor) error stating `rule` unless
/// `ok`.
fn ensure(ok: bool, rule: &'static str) -> Result<(), GpuError> {
    if ok {
        Ok(())
    } else {
        Err(GpuError::InvalidDescriptor { message: rule })
    }
}

fn owned(label: Option<&str>) -> Option<String> {
    label.map(String::from)
}

impl GpuDevice for RecordingGpu {
    fn surface_format(&self) -> TextureFormat {
        self.surface_format
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<Handle<Buffer>, GpuError> {
        ensure(desc.size > 0, "buffer size must be nonzero")?;
        ensure(
            desc.size <= Self::MAX_BUFFER_SIZE,
            "buffer size exceeds MAX_BUFFER_SIZE",
        )?;
        ensure(!desc.usage.is_empty(), "buffer usage must not be empty")?;
        let handle = self.buffers.insert(RecordedBuffer {
            label: owned(desc.label),
            size: desc.size,
            usage: desc.usage,
            // Lossless: the size is at most `MAX_BUFFER_SIZE`.
            data: vec![0; desc.size as usize],
        });
        self.events.push(GpuEvent::CreateBuffer(handle));
        Ok(handle)
    }

    fn write_buffer(
        &mut self,
        buffer: Handle<Buffer>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GpuError> {
        let size = self
            .buffer_with(buffer, BufferUsage::COPY_DST, "COPY_DST")?
            .size;
        let write_len = data.len() as u64;
        let end = offset
            .checked_add(write_len)
            .filter(|&end| end <= size)
            .ok_or(GpuError::BufferOutOfBounds {
                offset,
                write_len,
                buffer_size: size,
            })?;
        ensure(
            offset.is_multiple_of(4) && write_len.is_multiple_of(4),
            "a buffer write's offset and length must be multiples of 4",
        )?;
        let target = &mut self.buffers.get_mut(buffer).expect("checked above").data;
        // Lossless: both are at most the buffer size, which is at most `MAX_BUFFER_SIZE`.
        target[offset as usize..end as usize].copy_from_slice(data);
        self.events.push(GpuEvent::WriteBuffer {
            buffer,
            offset,
            len: data.len(),
        });
        Ok(())
    }

    fn destroy_buffer(&mut self, buffer: Handle<Buffer>) -> Result<(), GpuError> {
        self.buffers.remove(buffer).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        })?;
        self.events.push(GpuEvent::DestroyBuffer(buffer));
        Ok(())
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> Result<Handle<Texture>, GpuError> {
        ensure(
            desc.width > 0 && desc.height > 0,
            "texture width and height must be nonzero",
        )?;
        ensure(
            desc.width <= Self::MAX_TEXTURE_DIMENSION && desc.height <= Self::MAX_TEXTURE_DIMENSION,
            "texture width and height must be at most MAX_TEXTURE_DIMENSION",
        )?;
        let max_mips = u32::BITS - desc.width.max(desc.height).leading_zeros();
        ensure(
            (1..=max_mips).contains(&desc.mip_level_count),
            "mip_level_count must be in 1..=floor(log2(max(width, height))) + 1",
        )?;
        ensure(!desc.usage.is_empty(), "texture usage must not be empty")?;
        ensure(
            !(desc.usage.contains(TextureUsage::COPY_DST) && desc.format.is_depth()),
            "a depth texture cannot have COPY_DST",
        )?;
        let handle = self.textures.insert(RecordedTexture {
            label: owned(desc.label),
            width: desc.width,
            height: desc.height,
            mip_level_count: desc.mip_level_count,
            format: desc.format,
            usage: desc.usage,
            sampler: desc.sampler,
            data: Vec::new(),
        });
        self.events.push(GpuEvent::CreateTexture(handle));
        Ok(handle)
    }

    fn write_texture(&mut self, texture: Handle<Texture>, data: &[u8]) -> Result<(), GpuError> {
        let recorded = self.texture_with(texture, TextureUsage::COPY_DST, "COPY_DST")?;
        let bytes_per_pixel = recorded
            .format
            .bytes_per_pixel()
            .expect("create_texture rejects COPY_DST on depth formats; color formats have a size");
        // Cannot overflow: at most 8192 * 8192 * 4 bytes.
        let expected =
            recorded.width as usize * recorded.height as usize * bytes_per_pixel as usize;
        if data.len() != expected {
            return Err(GpuError::TextureDataSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let recorded = self.textures.get_mut(texture).expect("checked above");
        recorded.data.clear();
        recorded.data.extend_from_slice(data);
        self.events.push(GpuEvent::WriteTexture {
            texture,
            len: data.len(),
        });
        Ok(())
    }

    fn destroy_texture(&mut self, texture: Handle<Texture>) -> Result<(), GpuError> {
        self.textures
            .remove(texture)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Texture,
            })?;
        self.events.push(GpuEvent::DestroyTexture(texture));
        Ok(())
    }

    fn create_shader(&mut self, source: &ShaderSource<'_>) -> Result<Handle<Shader>, GpuError> {
        ensure(
            !source.vertex_entry.is_empty() && !source.fragment_entry.is_empty(),
            "shader entry point names must not be empty",
        )?;
        ensure(
            !source.msl.is_empty() || !source.wgsl.is_empty(),
            "a shader needs MSL or WGSL source",
        )?;
        let handle = self.shaders.insert(RecordedShader {
            label: owned(source.label),
            msl: source.msl.into(),
            wgsl: source.wgsl.into(),
            glsl: owned(source.glsl),
            vertex_entry: source.vertex_entry.into(),
            fragment_entry: source.fragment_entry.into(),
        });
        self.events.push(GpuEvent::CreateShader(handle));
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: Handle<Shader>) -> Result<(), GpuError> {
        self.shaders.remove(shader).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Shader,
        })?;
        self.events.push(GpuEvent::DestroyShader(shader));
        Ok(())
    }

    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<Handle<Pipeline>, GpuError> {
        if !self.shaders.contains(desc.shader) {
            return Err(GpuError::InvalidHandle {
                kind: ResourceKind::Shader,
            });
        }
        ensure(
            desc.color_format.is_some() || desc.depth_stencil.is_some(),
            "a pipeline needs a color_format or depth_stencil",
        )?;
        ensure(
            desc.color_format.is_none_or(|format| !format.is_depth()),
            "a pipeline's color_format must be a color format",
        )?;
        ensure(
            desc.depth_stencil.is_none_or(|ds| ds.format.is_depth()),
            "a pipeline's depth_stencil format must be a depth format",
        )?;
        ensure(
            desc.vertex_layouts.len() <= MAX_VERTEX_BUFFERS,
            "a pipeline has at most MAX_VERTEX_BUFFERS vertex layouts",
        )?;
        ensure(
            desc.vertex_layouts.iter().all(|layout| {
                layout.attributes.iter().all(|attribute| {
                    u64::from(attribute.offset) + u64::from(attribute.format.size_bytes())
                        <= u64::from(layout.array_stride)
                })
            }),
            "a vertex attribute must fit within its layout's array_stride",
        )?;
        let mut locations: Vec<u32> = desc
            .vertex_layouts
            .iter()
            .flat_map(|layout| layout.attributes)
            .map(|attribute| attribute.shader_location)
            .collect();
        locations.sort_unstable();
        ensure(
            locations.windows(2).all(|pair| pair[0] != pair[1]),
            "a pipeline's vertex attributes need distinct shader_locations",
        )?;
        let handle = self.pipelines.insert(RecordedPipeline {
            label: owned(desc.label),
            shader: desc.shader,
            vertex_layouts: desc
                .vertex_layouts
                .iter()
                .map(|layout| RecordedVertexLayout {
                    array_stride: layout.array_stride,
                    step_mode: layout.step_mode,
                    attributes: layout.attributes.to_vec(),
                })
                .collect(),
            color_format: desc.color_format,
            blend: desc.blend,
            depth_stencil: desc.depth_stencil,
            topology: desc.topology,
            front_face: desc.front_face,
            cull_mode: desc.cull_mode,
        });
        self.events.push(GpuEvent::CreatePipeline(handle));
        Ok(handle)
    }

    fn destroy_pipeline(&mut self, pipeline: Handle<Pipeline>) -> Result<(), GpuError> {
        self.pipelines
            .remove(pipeline)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Pipeline,
            })?;
        self.events.push(GpuEvent::DestroyPipeline(pipeline));
        Ok(())
    }

    fn submit(&mut self, commands: FrameView<'_>) -> Result<(), GpuError> {
        for pass in commands.passes() {
            self.check_pass(pass)?;
        }
        self.events.push(GpuEvent::Submit {
            frame: self.frames.len(),
        });
        self.frames.push(RecordedFrame {
            passes: commands
                .passes()
                .map(|pass| RecordedPass {
                    desc: pass.desc,
                    commands: pass.commands.to_vec(),
                })
                .collect(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{
        ClearColor, ColorAttachment, DepthAttachment, FrameCommands, ScissorRect, StoreOp, Viewport,
    };
    use crate::pipeline::{VertexFormat, VertexLayout};

    /// A textured, tinted quad following the binding convention on [`ShaderSource`]: vertex
    /// slot 0 per [`LAYOUTS`], uniform binding 0, and texture binding 1.
    const MSL: &str = r"
#include <metal_stdlib>
using namespace metal;

struct Globals {
    float4x4 transform;
};

struct VertexIn {
    float2 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut vs_main(VertexIn v [[stage_in]], constant Globals& globals [[buffer(8)]]) {
    VertexOut out;
    out.position = globals.transform * float4(v.position, 0.0, 1.0);
    out.uv = v.position;
    out.color = v.color;
    return out;
}

fragment float4 fs_main(
    VertexOut f [[stage_in]],
    texture2d<float> sprite_texture [[texture(1)]],
    sampler sprite_sampler [[sampler(1)]]
) {
    return sprite_texture.sample(sprite_sampler, f.uv) * f.color;
}
";

    const WGSL: &str = r"
struct Globals {
    transform: mat4x4<f32>,
}

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(2) var sprite_texture: texture_2d<f32>;
@group(1) @binding(3) var sprite_sampler: sampler;

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = globals.transform * vec4<f32>(v.position, 0.0, 1.0);
    out.uv = v.position;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(f: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(sprite_texture, sprite_sampler, f.uv) * f.color;
}
";

    const SHADER: ShaderSource<'static> = ShaderSource {
        label: Some("sprite"),
        msl: MSL,
        wgsl: WGSL,
        glsl: None,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
    };

    const ATTRIBUTES: [VertexAttribute; 2] = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Unorm8x4,
            offset: 8,
            shader_location: 1,
        },
    ];

    const LAYOUTS: [VertexLayout<'static>; 1] = [VertexLayout {
        array_stride: 12,
        step_mode: VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }];

    fn pipeline_desc(shader: Handle<Shader>) -> PipelineDesc<'static> {
        PipelineDesc {
            label: Some("sprites"),
            shader,
            vertex_layouts: &LAYOUTS,
            color_format: Some(TextureFormat::Bgra8UnormSrgb),
            blend: Some(BlendState::ALPHA_BLENDING),
            depth_stencil: None,
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
        }
    }

    fn buffer(gpu: &mut RecordingGpu, size: u64, usage: BufferUsage) -> Handle<Buffer> {
        gpu.create_buffer(&BufferDesc {
            label: None,
            size,
            usage,
        })
        .unwrap()
    }

    fn texture(
        gpu: &mut RecordingGpu,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Handle<Texture> {
        gpu.create_texture(&TextureDesc {
            format,
            usage,
            ..TextureDesc::rgba8_2d(4, 4)
        })
        .unwrap()
    }

    /// One live resource of every kind, each with every usage the tests bind it with.
    struct Scene {
        shader: Handle<Shader>,
        pipeline: Handle<Pipeline>,
        buffer: Handle<Buffer>,
        texture: Handle<Texture>,
        depth: Handle<Texture>,
    }

    fn scene(gpu: &mut RecordingGpu) -> Scene {
        let shader = gpu.create_shader(&SHADER).unwrap();
        let all_buffer_usages = BufferUsage::VERTEX
            | BufferUsage::INDEX
            | BufferUsage::UNIFORM
            | BufferUsage::STORAGE
            | BufferUsage::COPY_DST;
        Scene {
            shader,
            pipeline: gpu.create_pipeline(&pipeline_desc(shader)).unwrap(),
            buffer: buffer(gpu, 64, all_buffer_usages),
            texture: texture(
                gpu,
                TextureFormat::Rgba8Unorm,
                TextureUsage::SAMPLED | TextureUsage::RENDER_ATTACHMENT,
            ),
            depth: texture(
                gpu,
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ),
        }
    }

    /// A scene whose resources are all destroyed again, leaving only stale handles.
    fn stale_scene(gpu: &mut RecordingGpu) -> Scene {
        let scene = scene(gpu);
        gpu.destroy_pipeline(scene.pipeline).unwrap();
        gpu.destroy_shader(scene.shader).unwrap();
        gpu.destroy_buffer(scene.buffer).unwrap();
        gpu.destroy_texture(scene.texture).unwrap();
        gpu.destroy_texture(scene.depth).unwrap();
        scene
    }

    fn submit_pass(
        gpu: &mut RecordingGpu,
        desc: RenderPassDesc,
        commands: &[RenderCmd],
    ) -> Result<(), GpuError> {
        let mut frame = FrameCommands::new();
        let mut pass = frame.begin_pass(desc);
        for &command in commands {
            pass.push(command);
        }
        gpu.submit(frame.view())
    }

    fn surface() -> RenderPassDesc {
        RenderPassDesc::clear_surface(ClearColor::BLACK)
    }

    fn invalid<T>(message: &'static str) -> Result<T, GpuError> {
        Err(GpuError::InvalidDescriptor { message })
    }

    #[test]
    fn textured_quad_lifecycle_is_recorded() {
        let mut gpu = RecordingGpu::new();
        let shader = gpu.create_shader(&SHADER).unwrap();
        let pipeline = gpu.create_pipeline(&pipeline_desc(shader)).unwrap();
        let vertices = buffer(&mut gpu, 48, BufferUsage::VERTEX | BufferUsage::COPY_DST);
        let indices = buffer(&mut gpu, 12, BufferUsage::INDEX | BufferUsage::COPY_DST);
        let uniforms = buffer(&mut gpu, 64, BufferUsage::UNIFORM | BufferUsage::COPY_DST);
        let image = gpu
            .create_texture(&TextureDesc {
                label: Some("checker"),
                ..TextureDesc::rgba8_2d(2, 2)
            })
            .unwrap();

        gpu.write_buffer(vertices, 12, &[7; 24]).unwrap();
        gpu.write_buffer(indices, 0, &[0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0])
            .unwrap();
        let texels: Vec<u8> = (0..16).collect();
        gpu.write_texture(image, &texels).unwrap();

        let mut frame = FrameCommands::new();
        frame
            .begin_pass(RenderPassDesc::clear_surface(ClearColor::rgba(
                0.1, 0.2, 0.3, 1.0,
            )))
            .set_viewport(Viewport::new(0.0, 0.0, 800.0, 600.0))
            .set_pipeline(pipeline)
            .set_vertex_buffer(0, vertices, 0)
            .set_index_buffer(indices, IndexFormat::Uint16, 0)
            .set_uniform_buffer(0, uniforms, 0)
            .set_texture(1, image)
            .draw_indexed(0..6, 0, 0..1);
        gpu.submit(frame.view()).unwrap();

        let recorded = &gpu.buffer(vertices).unwrap().data;
        assert_eq!(recorded.len(), 48);
        assert!(recorded[..12].iter().all(|&b| b == 0));
        assert!(recorded[12..36].iter().all(|&b| b == 7));
        assert!(recorded[36..].iter().all(|&b| b == 0));
        let image_record = gpu.texture(image).unwrap();
        assert_eq!(image_record.label.as_deref(), Some("checker"));
        assert_eq!(image_record.data, texels);
        let pipeline_record = gpu.pipeline(pipeline).unwrap();
        assert_eq!(pipeline_record.shader, shader);
        assert_eq!(pipeline_record.vertex_layouts[0].attributes, ATTRIBUTES);
        assert_eq!(gpu.shader(shader).unwrap().vertex_entry, "vs_main");

        let last = gpu.last_frame().unwrap();
        assert_eq!(last.passes.len(), 1);
        assert_eq!(last.passes[0].desc, frame.pass(0).unwrap().desc);
        assert_eq!(last.passes[0].commands, frame.pass(0).unwrap().commands);
        assert_eq!(
            gpu.live_counts(),
            ResourceCounts {
                buffers: 3,
                textures: 1,
                shaders: 1,
                pipelines: 1,
            }
        );
        assert_eq!(
            gpu.events(),
            [
                GpuEvent::CreateShader(shader),
                GpuEvent::CreatePipeline(pipeline),
                GpuEvent::CreateBuffer(vertices),
                GpuEvent::CreateBuffer(indices),
                GpuEvent::CreateBuffer(uniforms),
                GpuEvent::CreateTexture(image),
                GpuEvent::WriteBuffer {
                    buffer: vertices,
                    offset: 12,
                    len: 24,
                },
                GpuEvent::WriteBuffer {
                    buffer: indices,
                    offset: 0,
                    len: 12,
                },
                GpuEvent::WriteTexture {
                    texture: image,
                    len: 16,
                },
                GpuEvent::Submit { frame: 0 },
            ]
        );

        // The shader may go while its pipeline lives on.
        gpu.destroy_shader(shader).unwrap();
        gpu.destroy_buffer(uniforms).unwrap();
        assert!(gpu.buffer(uniforms).is_none());
        gpu.clear_history();
        assert!(gpu.frames().is_empty() && gpu.events().is_empty());
        let counts = gpu.live_counts();
        assert_eq!(
            (counts.buffers, counts.shaders, counts.pipelines),
            (2, 0, 1)
        );
    }

    #[test]
    fn stale_handles_are_rejected_by_every_call() {
        let mut gpu = RecordingGpu::new();
        let stale = stale_scene(&mut gpu);
        // The live resources reuse the stale ones' slots with newer generations.
        let live = scene(&mut gpu);
        assert_eq!(gpu.live_counts().buffers, 1);
        let invalid = |kind| Err(GpuError::InvalidHandle { kind });

        assert_eq!(
            gpu.write_buffer(stale.buffer, 0, &[1]),
            invalid(ResourceKind::Buffer)
        );
        assert_eq!(
            gpu.destroy_buffer(stale.buffer),
            invalid(ResourceKind::Buffer)
        );
        assert_eq!(
            gpu.write_texture(stale.texture, &[]),
            invalid(ResourceKind::Texture)
        );
        assert_eq!(
            gpu.destroy_texture(stale.texture),
            invalid(ResourceKind::Texture)
        );
        assert_eq!(
            gpu.destroy_shader(stale.shader),
            invalid(ResourceKind::Shader)
        );
        assert_eq!(
            gpu.create_pipeline(&pipeline_desc(stale.shader)),
            Err(GpuError::InvalidHandle {
                kind: ResourceKind::Shader
            })
        );
        assert_eq!(
            gpu.destroy_pipeline(stale.pipeline),
            invalid(ResourceKind::Pipeline)
        );

        let color_target = |target| RenderPassDesc {
            label: None,
            color: Some(ColorAttachment {
                target: Some(target),
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
            }),
            depth: None,
        };
        let depth_target = |target| RenderPassDesc {
            label: None,
            color: None,
            depth: Some(DepthAttachment {
                target,
                depth_load_op: LoadOp::Clear(1.0),
                depth_store_op: StoreOp::Discard,
            }),
        };
        let cases = [
            (
                surface(),
                RenderCmd::SetPipeline(stale.pipeline),
                ResourceKind::Pipeline,
            ),
            (
                surface(),
                RenderCmd::SetVertexBuffer {
                    slot: 0,
                    buffer: stale.buffer,
                    offset: 0,
                },
                ResourceKind::Buffer,
            ),
            (
                surface(),
                RenderCmd::SetIndexBuffer {
                    buffer: stale.buffer,
                    format: IndexFormat::Uint32,
                    offset: 0,
                },
                ResourceKind::Buffer,
            ),
            (
                surface(),
                RenderCmd::SetUniformBuffer {
                    binding: 0,
                    buffer: stale.buffer,
                    offset: 0,
                },
                ResourceKind::Buffer,
            ),
            (
                surface(),
                RenderCmd::SetStorageBuffer {
                    binding: 0,
                    buffer: stale.buffer,
                    offset: 0,
                },
                ResourceKind::Buffer,
            ),
            (
                surface(),
                RenderCmd::SetTexture {
                    binding: 0,
                    texture: stale.texture,
                },
                ResourceKind::Texture,
            ),
            (
                color_target(stale.texture),
                RenderCmd::SetPipeline(live.pipeline),
                ResourceKind::Texture,
            ),
            (
                depth_target(stale.depth),
                RenderCmd::SetPipeline(live.pipeline),
                ResourceKind::Texture,
            ),
        ];
        for (desc, command, kind) in cases {
            assert_eq!(
                submit_pass(&mut gpu, desc, &[command]),
                invalid(kind),
                "{command:?}"
            );
        }
        assert!(gpu.frames().is_empty());

        // The same commands with live handles pass, so each case failed only on its stale handle.
        submit_pass(&mut gpu, color_target(live.texture), &[]).unwrap();
        submit_pass(&mut gpu, depth_target(live.depth), &[]).unwrap();
        assert_eq!(gpu.frames().len(), 2);
    }

    #[test]
    fn failed_submit_records_nothing() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        submit_pass(&mut gpu, surface(), &[]).unwrap();
        let frames = gpu.frames().to_vec();
        let events = gpu.events().to_vec();

        let mut frame = FrameCommands::new();
        frame
            .begin_pass(surface())
            .set_pipeline(s.pipeline)
            .set_vertex_buffer(0, s.buffer, 0)
            .draw(0..3, 0..1);
        frame
            .begin_pass(RenderPassDesc::load_surface())
            .set_texture(0, s.texture)
            .draw(0..3, 0..1);
        assert_eq!(gpu.submit(frame.view()), Err(GpuError::DrawWithoutPipeline));
        assert_eq!(gpu.frames(), frames);
        assert_eq!(gpu.events(), events);
    }

    #[test]
    fn buffer_writes_stay_in_bounds() {
        let mut gpu = RecordingGpu::new();
        let target = buffer(&mut gpu, 16, BufferUsage::COPY_DST);
        let out_of_bounds = |offset, write_len| {
            Err(GpuError::BufferOutOfBounds {
                offset,
                write_len,
                buffer_size: 16,
            })
        };
        assert_eq!(gpu.write_buffer(target, 12, &[1; 5]), out_of_bounds(12, 5));
        assert_eq!(gpu.write_buffer(target, 17, &[]), out_of_bounds(17, 0));
        assert_eq!(
            gpu.write_buffer(target, u64::MAX, &[1; 2]),
            out_of_bounds(u64::MAX, 2)
        );
        assert!(gpu.buffer(target).unwrap().data.iter().all(|&b| b == 0));
        let unaligned = invalid("a buffer write's offset and length must be multiples of 4");
        assert_eq!(gpu.write_buffer(target, 1, &[1; 4]), unaligned);
        assert_eq!(gpu.write_buffer(target, 0, &[1; 3]), unaligned);
        assert!(gpu.buffer(target).unwrap().data.iter().all(|&b| b == 0));
        // Exactly filling the tail and empty writes at the end are fine.
        gpu.write_buffer(target, 12, &[9; 4]).unwrap();
        gpu.write_buffer(target, 16, &[]).unwrap();
        assert_eq!(gpu.buffer(target).unwrap().data[12..], [9; 4]);
    }

    #[test]
    fn texture_writes_must_be_one_full_level() {
        let mut gpu = RecordingGpu::new();
        let rgba = gpu.create_texture(&TextureDesc::rgba8_2d(3, 2)).unwrap();
        let mismatch = |actual| {
            Err(GpuError::TextureDataSizeMismatch {
                expected: 24,
                actual,
            })
        };
        assert_eq!(gpu.write_texture(rgba, &[0; 23]), mismatch(23));
        assert_eq!(gpu.write_texture(rgba, &[0; 25]), mismatch(25));
        assert!(gpu.texture(rgba).unwrap().data.is_empty());
        gpu.write_texture(rgba, &[0; 24]).unwrap();

        let r8 = texture(&mut gpu, TextureFormat::R8Unorm, TextureUsage::COPY_DST);
        assert_eq!(
            gpu.write_texture(r8, &[0; 64]),
            Err(GpuError::TextureDataSizeMismatch {
                expected: 16,
                actual: 64,
            })
        );
        gpu.write_texture(r8, &[0; 16]).unwrap();

        // WebGPU cannot copy CPU data into any depth texture, even one with a fixed texel size.
        for format in [
            TextureFormat::Depth32Float,
            TextureFormat::Depth24PlusStencil8,
        ] {
            assert_eq!(
                gpu.create_texture(&TextureDesc {
                    format,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_DST,
                    ..TextureDesc::rgba8_2d(4, 4)
                }),
                invalid("a depth texture cannot have COPY_DST"),
                "{format:?}"
            );
        }
    }

    #[test]
    fn missing_usage_is_rejected() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        let missing = |kind, required| Err(GpuError::MissingUsage { kind, required });
        let vertex_only = buffer(&mut gpu, 16, BufferUsage::VERTEX);
        let sampled_only = texture(&mut gpu, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);
        let target_only = texture(
            &mut gpu,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT,
        );

        assert_eq!(
            gpu.write_buffer(vertex_only, 0, &[1]),
            missing(ResourceKind::Buffer, "COPY_DST")
        );
        assert_eq!(
            gpu.write_texture(sampled_only, &[0; 64]),
            missing(ResourceKind::Texture, "COPY_DST")
        );
        let uniform_only = buffer(&mut gpu, 16, BufferUsage::UNIFORM);
        let cases = [
            (
                RenderCmd::SetVertexBuffer {
                    slot: 0,
                    buffer: uniform_only,
                    offset: 0,
                },
                "VERTEX",
            ),
            (
                RenderCmd::SetIndexBuffer {
                    buffer: vertex_only,
                    format: IndexFormat::Uint16,
                    offset: 0,
                },
                "INDEX",
            ),
            (
                RenderCmd::SetUniformBuffer {
                    binding: 0,
                    buffer: vertex_only,
                    offset: 0,
                },
                "UNIFORM",
            ),
            (
                RenderCmd::SetStorageBuffer {
                    binding: 0,
                    buffer: vertex_only,
                    offset: 0,
                },
                "STORAGE",
            ),
        ];
        for (command, required) in cases {
            assert_eq!(
                submit_pass(&mut gpu, surface(), &[command]),
                missing(ResourceKind::Buffer, required)
            );
        }
        assert_eq!(
            submit_pass(
                &mut gpu,
                surface(),
                &[RenderCmd::SetTexture {
                    binding: 0,
                    texture: target_only,
                }]
            ),
            missing(ResourceKind::Texture, "SAMPLED")
        );
        let offscreen = |target| RenderPassDesc {
            color: Some(ColorAttachment {
                target: Some(target),
                load_op: LoadOp::Clear(ClearColor::TRANSPARENT),
                store_op: StoreOp::Store,
            }),
            ..surface()
        };
        assert_eq!(
            submit_pass(&mut gpu, offscreen(sampled_only), &[]),
            missing(ResourceKind::Texture, "RENDER_ATTACHMENT")
        );
        submit_pass(&mut gpu, offscreen(s.texture), &[]).unwrap();
    }

    #[test]
    fn draws_need_bindings_from_their_own_pass() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        let draw = RenderCmd::Draw(DrawCmd::NonIndexed {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });
        let draw_indexed = RenderCmd::Draw(DrawCmd::Indexed {
            index_count: 3,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        });
        let set_index = RenderCmd::SetIndexBuffer {
            buffer: s.buffer,
            format: IndexFormat::Uint16,
            offset: 0,
        };
        let set_pipeline = RenderCmd::SetPipeline(s.pipeline);
        let set_vertex = RenderCmd::SetVertexBuffer {
            slot: 0,
            buffer: s.buffer,
            offset: 0,
        };

        assert_eq!(
            submit_pass(&mut gpu, surface(), &[draw]),
            Err(GpuError::DrawWithoutPipeline)
        );
        assert_eq!(
            submit_pass(&mut gpu, surface(), &[set_index, draw_indexed]),
            Err(GpuError::DrawWithoutPipeline)
        );
        assert_eq!(
            submit_pass(&mut gpu, surface(), &[set_pipeline, draw_indexed]),
            Err(GpuError::DrawIndexedWithoutIndexBuffer)
        );
        submit_pass(
            &mut gpu,
            surface(),
            &[set_pipeline, set_vertex, set_index, draw_indexed],
        )
        .unwrap();

        // A pipeline and index buffer bound in one pass do not carry over to the next.
        let mut frame = FrameCommands::new();
        frame
            .begin_pass(surface())
            .push(set_pipeline)
            .push(set_vertex)
            .push(set_index);
        frame
            .begin_pass(RenderPassDesc::load_surface())
            .push(draw_indexed);
        assert_eq!(gpu.submit(frame.view()), Err(GpuError::DrawWithoutPipeline));
        assert_eq!(gpu.frames().len(), 1);
    }

    #[test]
    fn invalid_resource_descriptors_are_rejected() {
        let mut gpu = RecordingGpu::new();
        let buffer_desc = |size, usage| BufferDesc {
            label: None,
            size,
            usage,
        };
        assert_eq!(
            gpu.create_buffer(&buffer_desc(0, BufferUsage::VERTEX)),
            invalid("buffer size must be nonzero")
        );
        assert_eq!(
            gpu.create_buffer(&buffer_desc(
                RecordingGpu::MAX_BUFFER_SIZE + 1,
                BufferUsage::VERTEX
            )),
            invalid("buffer size exceeds MAX_BUFFER_SIZE")
        );
        assert_eq!(
            gpu.create_buffer(&buffer_desc(4, BufferUsage::EMPTY)),
            invalid("buffer usage must not be empty")
        );

        let mips = "mip_level_count must be in 1..=floor(log2(max(width, height))) + 1";
        let texture_cases = [
            (0, 4, 1, "texture width and height must be nonzero"),
            (4, 0, 1, "texture width and height must be nonzero"),
            (
                8193,
                1,
                1,
                "texture width and height must be at most MAX_TEXTURE_DIMENSION",
            ),
            (4, 4, 0, mips),
            // 5x3 has levels 5x3, 2x1, 1x1.
            (5, 3, 4, mips),
        ];
        for (width, height, mip_level_count, message) in texture_cases {
            assert_eq!(
                gpu.create_texture(&TextureDesc {
                    width,
                    height,
                    mip_level_count,
                    ..TextureDesc::rgba8_2d(1, 1)
                }),
                invalid(message),
                "{width}x{height} with {mip_level_count} mips"
            );
        }
        gpu.create_texture(&TextureDesc {
            mip_level_count: 3,
            ..TextureDesc::rgba8_2d(5, 3)
        })
        .unwrap();
        assert_eq!(
            gpu.create_texture(&TextureDesc {
                usage: TextureUsage::EMPTY,
                ..TextureDesc::rgba8_2d(1, 1)
            }),
            invalid("texture usage must not be empty")
        );

        assert_eq!(
            gpu.create_shader(&ShaderSource {
                fragment_entry: "",
                ..SHADER
            }),
            invalid("shader entry point names must not be empty")
        );
        assert_eq!(
            gpu.create_shader(&ShaderSource {
                msl: "",
                wgsl: "",
                ..SHADER
            }),
            invalid("a shader needs MSL or WGSL source")
        );
        let shader = gpu
            .create_shader(&ShaderSource { msl: "", ..SHADER })
            .unwrap();
        assert_eq!(
            gpu.create_pipeline(&PipelineDesc {
                color_format: Some(TextureFormat::Depth32Float),
                ..pipeline_desc(shader)
            }),
            invalid("a pipeline's color_format must be a color format")
        );
        assert_eq!(
            gpu.create_pipeline(&PipelineDesc {
                color_format: None,
                ..pipeline_desc(shader)
            }),
            invalid("a pipeline needs a color_format or depth_stencil")
        );
        assert_eq!(
            gpu.create_pipeline(&PipelineDesc {
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Rgba8Unorm,
                    ..DepthStencilState::DEFAULT_3D
                }),
                ..pipeline_desc(shader)
            }),
            invalid("a pipeline's depth_stencil format must be a depth format")
        );
        assert_eq!(gpu.live_counts().pipelines, 0);
    }

    #[test]
    fn invalid_passes_are_rejected() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        let color = |target| {
            Some(ColorAttachment {
                target,
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
            })
        };
        let depth = |target, clear| {
            Some(DepthAttachment {
                target,
                depth_load_op: LoadOp::Clear(clear),
                depth_store_op: StoreOp::Store,
            })
        };
        let pass = |color, depth| RenderPassDesc {
            label: None,
            color,
            depth,
        };
        let viewport = |viewport| [RenderCmd::SetViewport(viewport)];

        assert_eq!(
            submit_pass(&mut gpu, pass(None, None), &[]),
            invalid("a render pass needs a color or depth attachment")
        );
        assert_eq!(
            submit_pass(&mut gpu, pass(color(Some(s.depth)), None), &[]),
            invalid("a color attachment needs a color format")
        );
        assert_eq!(
            submit_pass(&mut gpu, pass(None, depth(s.texture, 1.0)), &[]),
            invalid("a depth attachment needs a depth format")
        );
        assert_eq!(
            submit_pass(&mut gpu, pass(None, depth(s.depth, 1.5)), &[]),
            invalid("a depth clear value must be in [0, 1]")
        );
        assert_eq!(
            submit_pass(&mut gpu, pass(None, depth(s.depth, f32::NAN)), &[]),
            invalid("a depth clear value must be in [0, 1]")
        );
        assert_eq!(
            submit_pass(
                &mut gpu,
                RenderPassDesc::clear_surface(ClearColor::rgba(f64::INFINITY, 0.0, 0.0, 1.0)),
                &[]
            ),
            invalid("a clear color must be finite")
        );
        let bad_viewport = invalid(
            "a viewport must be finite, have a positive size, and have \
             0 <= min_depth <= max_depth <= 1",
        );
        for v in [
            Viewport::new(0.0, 0.0, 0.0, 600.0),
            Viewport::new(0.0, 0.0, 800.0, -1.0),
            Viewport::new(f32::NAN, 0.0, 800.0, 600.0),
            Viewport::new(0.0, 0.0, f32::INFINITY, 600.0),
            Viewport {
                min_depth: 0.6,
                max_depth: 0.4,
                ..Viewport::new(0.0, 0.0, 800.0, 600.0)
            },
            Viewport {
                max_depth: 1.5,
                ..Viewport::new(0.0, 0.0, 800.0, 600.0)
            },
        ] {
            assert_eq!(
                submit_pass(&mut gpu, surface(), &viewport(v)),
                bad_viewport,
                "{v:?}"
            );
        }
        assert_eq!(
            submit_pass(
                &mut gpu,
                surface(),
                &[RenderCmd::SetUniformBuffer {
                    binding: 0,
                    buffer: s.buffer,
                    offset: 65,
                }]
            ),
            Err(GpuError::BufferOutOfBounds {
                offset: 65,
                write_len: 0,
                buffer_size: 64,
            })
        );
        assert!(gpu.frames().is_empty());

        // Color plus depth, a depth-only pass, and a full-buffer offset are all valid.
        submit_pass(
            &mut gpu,
            pass(color(Some(s.texture)), depth(s.depth, 0.0)),
            &viewport(Viewport::new(0.0, 0.0, 4.0, 4.0)),
        )
        .unwrap();
        submit_pass(&mut gpu, pass(None, depth(s.depth, 1.0)), &[]).unwrap();
        submit_pass(
            &mut gpu,
            surface(),
            &[RenderCmd::SetUniformBuffer {
                binding: 0,
                buffer: s.buffer,
                offset: 64,
            }],
        )
        .unwrap();
        assert_eq!(gpu.frames().len(), 3);
    }

    #[test]
    fn passes_reject_mismatched_pipelines_targets_scissors_and_feedback() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        let sampled_depth = texture(
            &mut gpu,
            TextureFormat::Depth32Float,
            TextureUsage::SAMPLED | TextureUsage::RENDER_ATTACHMENT,
        );
        let big_depth = gpu
            .create_texture(&TextureDesc {
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::RENDER_ATTACHMENT,
                ..TextureDesc::rgba8_2d(8, 8)
            })
            .unwrap();
        let mut pipeline = |color_format, depth_stencil| {
            gpu.create_pipeline(&PipelineDesc {
                vertex_layouts: &[],
                color_format,
                depth_stencil,
                ..pipeline_desc(s.shader)
            })
            .unwrap()
        };
        let surface_pipeline = pipeline(Some(TextureFormat::Bgra8UnormSrgb), None);
        let r8_pipeline = pipeline(Some(TextureFormat::R8Unorm), None);
        let rgba_depth_pipeline = pipeline(
            Some(TextureFormat::Rgba8Unorm),
            Some(DepthStencilState::DEFAULT_3D),
        );
        let depth_only_pipeline = pipeline(None, Some(DepthStencilState::DEFAULT_3D));
        let pass = |color: Option<Option<Handle<Texture>>>, depth: Option<Handle<Texture>>| {
            RenderPassDesc {
                label: None,
                color: color.map(|target| ColorAttachment {
                    target,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                }),
                depth: depth.map(|target| DepthAttachment {
                    target,
                    depth_load_op: LoadOp::Load,
                    depth_store_op: StoreOp::Store,
                }),
            }
        };
        let draw = |pipeline| {
            [
                RenderCmd::SetPipeline(pipeline),
                RenderCmd::Draw(DrawCmd::NonIndexed {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                }),
            ]
        };
        let scissor = |x, y, width, height| {
            [RenderCmd::SetScissor(ScissorRect {
                x,
                y,
                width,
                height,
            })]
        };
        let sample = |texture| {
            [RenderCmd::SetTexture {
                binding: 0,
                texture,
            }]
        };
        let offscreen = pass(Some(Some(s.texture)), Some(s.depth));

        let mismatch = invalid(
            "a pipeline's color_format and depth_stencil format must match its pass's attachments",
        );
        let cases = [
            (pass(Some(None), None), r8_pipeline),
            (pass(Some(None), Some(s.depth)), surface_pipeline),
            (offscreen, surface_pipeline),
            (pass(Some(Some(s.texture)), None), rgba_depth_pipeline),
            (pass(None, Some(s.depth)), rgba_depth_pipeline),
            (
                pass(Some(Some(s.texture)), Some(s.depth)),
                depth_only_pipeline,
            ),
        ];
        for (desc, pipeline) in cases {
            assert_eq!(submit_pass(&mut gpu, desc, &draw(pipeline)), mismatch);
        }
        assert_eq!(
            submit_pass(&mut gpu, pass(Some(Some(s.texture)), Some(big_depth)), &[]),
            invalid("a pass's color and depth targets must be the same size")
        );
        let outside = invalid("a scissor rectangle must lie within the pass's attachments");
        for rect in [
            scissor(4, 0, 1, 1),
            scissor(0, 2, 4, 3),
            scissor(u32::MAX, 0, 1, 1),
        ] {
            assert_eq!(submit_pass(&mut gpu, offscreen, &rect), outside);
        }
        // A depth-only pass bounds the scissor by its depth target.
        assert_eq!(
            submit_pass(&mut gpu, pass(None, Some(s.depth)), &scissor(0, 0, 5, 4)),
            outside
        );
        let feedback = invalid("a pass cannot sample its own color or depth attachment");
        assert_eq!(
            submit_pass(&mut gpu, offscreen, &sample(s.texture)),
            feedback
        );
        assert_eq!(
            submit_pass(
                &mut gpu,
                pass(None, Some(sampled_depth)),
                &sample(sampled_depth)
            ),
            feedback
        );
        assert!(gpu.frames().is_empty());

        // Matching pipelines, in-bounds scissors, and sampling other textures all pass, as do
        // depth-only pipelines and any scissor on the surface, whose size is unknown here.
        submit_pass(&mut gpu, pass(Some(None), None), &draw(surface_pipeline)).unwrap();
        submit_pass(&mut gpu, offscreen, &draw(rgba_depth_pipeline)).unwrap();
        submit_pass(
            &mut gpu,
            pass(None, Some(s.depth)),
            &draw(depth_only_pipeline),
        )
        .unwrap();
        submit_pass(&mut gpu, offscreen, &scissor(0, 0, 4, 4)).unwrap();
        submit_pass(&mut gpu, offscreen, &scissor(3, 3, 1, 1)).unwrap();
        submit_pass(&mut gpu, surface(), &scissor(4000, 0, 100, 100)).unwrap();
        submit_pass(&mut gpu, offscreen, &sample(sampled_depth)).unwrap();
        submit_pass(&mut gpu, pass(None, Some(s.depth)), &sample(s.texture)).unwrap();
        assert_eq!(gpu.frames().len(), 8);
    }

    #[test]
    fn vertex_layouts_are_validated_at_pipeline_creation() {
        let mut gpu = RecordingGpu::new();
        let shader = gpu.create_shader(&SHADER).unwrap();
        let attribute = |format, offset, shader_location| VertexAttribute {
            format,
            offset,
            shader_location,
        };
        fn layout(array_stride: u32, attributes: &[VertexAttribute]) -> VertexLayout<'_> {
            VertexLayout {
                array_stride,
                step_mode: VertexStepMode::Vertex,
                attributes,
            }
        }
        let mut create = |vertex_layouts: &[VertexLayout<'_>]| {
            gpu.create_pipeline(&PipelineDesc {
                vertex_layouts,
                ..pipeline_desc(shader)
            })
        };

        let wide = [attribute(VertexFormat::Float32x4, 0, 0)];
        assert_eq!(
            create(&[layout(4, &wide)]),
            invalid("a vertex attribute must fit within its layout's array_stride")
        );
        let past_end = [attribute(VertexFormat::Float32, 13, 0)];
        assert_eq!(
            create(&[layout(16, &past_end)]),
            invalid("a vertex attribute must fit within its layout's array_stride")
        );
        let first = [attribute(VertexFormat::Float32, 0, 3)];
        let second = [attribute(VertexFormat::Float32, 0, 3)];
        assert_eq!(
            create(&[layout(4, &first), layout(4, &second)]),
            invalid("a pipeline's vertex attributes need distinct shader_locations")
        );
        assert_eq!(
            create(&[layout(4, &[]); MAX_VERTEX_BUFFERS + 1]),
            invalid("a pipeline has at most MAX_VERTEX_BUFFERS vertex layouts")
        );

        let fits_exactly = [attribute(VertexFormat::Float32, 12, 0)];
        let other = [attribute(VertexFormat::Float32, 0, 1)];
        create(&[layout(16, &fits_exactly), layout(4, &other)]).unwrap();
        create(&[layout(4, &[]); MAX_VERTEX_BUFFERS]).unwrap();
        // Only the two valid pipelines exist.
        assert_eq!(gpu.live_counts().pipelines, 2);
    }

    #[test]
    fn draws_stay_inside_their_bound_buffers() {
        let mut gpu = RecordingGpu::new();
        let s = scene(&mut gpu);
        let instance_attributes = [VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 0,
            shader_location: 2,
        }];
        let instanced = gpu
            .create_pipeline(&PipelineDesc {
                vertex_layouts: &[
                    LAYOUTS[0],
                    VertexLayout {
                        array_stride: 16,
                        step_mode: VertexStepMode::Instance,
                        attributes: &instance_attributes,
                    },
                ],
                ..pipeline_desc(s.shader)
            })
            .unwrap();
        let fullscreen = gpu
            .create_pipeline(&PipelineDesc {
                vertex_layouts: &[],
                ..pipeline_desc(s.shader)
            })
            .unwrap();
        let four_bytes = buffer(&mut gpu, 4, BufferUsage::INDEX);
        // `s.buffer` is 64 bytes: five 12-byte vertices, or four 16-byte instances.
        let vertices = |offset| RenderCmd::SetVertexBuffer {
            slot: 0,
            buffer: s.buffer,
            offset,
        };
        let instances = RenderCmd::SetVertexBuffer {
            slot: 1,
            buffer: s.buffer,
            offset: 0,
        };
        let indices = |buffer, format, offset| RenderCmd::SetIndexBuffer {
            buffer,
            format,
            offset,
        };
        let draw = |first_vertex, vertex_count, first_instance, instance_count| {
            RenderCmd::Draw(DrawCmd::NonIndexed {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            })
        };
        let draw_indexed = |first_index, index_count, first_instance, instance_count| {
            RenderCmd::Draw(DrawCmd::Indexed {
                index_count,
                instance_count,
                first_index,
                base_vertex: 0,
                first_instance,
            })
        };
        let sprites = RenderCmd::SetPipeline(s.pipeline);
        let instanced = RenderCmd::SetPipeline(instanced);
        let fullscreen = RenderCmd::SetPipeline(fullscreen);
        let index_u16 = indices(s.buffer, IndexFormat::Uint16, 0);

        let missing =
            invalid("a draw needs a vertex buffer in every slot its pipeline has a layout for");
        let vertex_overrun = invalid("a draw must read only vertices inside its vertex buffers");
        let index_overrun =
            invalid("an indexed draw must read only indices inside its index buffer");
        let misaligned = invalid("an index buffer offset must be a multiple of the index size");
        let cases: [(&[RenderCmd], _); 15] = [
            (&[sprites, draw(0, 3, 0, 1)], missing.clone()),
            (&[instanced, vertices(0), draw(0, 3, 0, 1)], missing.clone()),
            (
                &[sprites, vertices(0), draw(0, 6, 0, 1)],
                vertex_overrun.clone(),
            ),
            (
                &[sprites, vertices(8), draw(0, 5, 0, 1)],
                vertex_overrun.clone(),
            ),
            (
                &[sprites, vertices(0), draw(u32::MAX, u32::MAX, 0, 1)],
                vertex_overrun.clone(),
            ),
            // A zero count elsewhere in the draw does not excuse an overrun.
            (
                &[sprites, vertices(28), draw(0, 4, 0, 0)],
                vertex_overrun.clone(),
            ),
            (
                &[instanced, vertices(0), instances, draw(0, 0, 0, 5)],
                vertex_overrun.clone(),
            ),
            (
                &[sprites, vertices(64), draw(u32::MAX, 0, 0, 1)],
                vertex_overrun.clone(),
            ),
            (
                &[instanced, vertices(0), instances, draw(0, 3, 2, 3)],
                vertex_overrun.clone(),
            ),
            (
                &[
                    instanced,
                    vertices(0),
                    instances,
                    index_u16,
                    draw_indexed(0, 3, 0, 5),
                ],
                vertex_overrun,
            ),
            (
                &[
                    sprites,
                    vertices(0),
                    indices(four_bytes, IndexFormat::Uint16, 0),
                    draw_indexed(0, 3, 0, 1),
                ],
                index_overrun.clone(),
            ),
            (
                &[
                    sprites,
                    vertices(0),
                    indices(four_bytes, IndexFormat::Uint16, 0),
                    draw_indexed(0, 3, 0, 0),
                ],
                index_overrun.clone(),
            ),
            (
                &[
                    sprites,
                    vertices(0),
                    indices(four_bytes, IndexFormat::Uint32, 0),
                    draw_indexed(u32::MAX, 1, 0, 1),
                ],
                index_overrun,
            ),
            (
                &[indices(s.buffer, IndexFormat::Uint16, 1)],
                misaligned.clone(),
            ),
            (&[indices(s.buffer, IndexFormat::Uint32, 2)], misaligned),
        ];
        for (commands, error) in cases {
            assert_eq!(
                submit_pass(&mut gpu, surface(), commands),
                error,
                "{commands:?}"
            );
        }
        assert_eq!(
            submit_pass(
                &mut gpu,
                surface(),
                &[RenderCmd::SetVertexBuffer {
                    slot: 8,
                    buffer: s.buffer,
                    offset: 0,
                }]
            ),
            invalid("a vertex buffer slot must be below MAX_VERTEX_BUFFERS")
        );
        assert!(gpu.frames().is_empty());

        // Exactly filling each buffer passes, as do empty draws at a buffer's end, indexed draws
        // whose vertices only the GPU knows, and draws whose pipeline reads no vertex buffer.
        let valid: [&[RenderCmd]; 7] = [
            &[sprites, vertices(0), draw(0, 5, 0, 1)],
            &[sprites, vertices(4), draw(1, 4, 0, 1)],
            &[sprites, vertices(64), draw(0, 0, 0, 0)],
            &[instanced, vertices(0), instances, draw(0, 5, 1, 3)],
            &[
                instanced,
                vertices(0),
                instances,
                indices(four_bytes, IndexFormat::Uint16, 2),
                draw_indexed(0, 1, 0, 4),
            ],
            &[
                sprites,
                vertices(0),
                indices(four_bytes, IndexFormat::Uint32, 0),
                draw_indexed(0, 1, 0, 1),
            ],
            &[fullscreen, draw(0, 3, 0, 1)],
        ];
        for commands in valid {
            submit_pass(&mut gpu, surface(), commands).unwrap();
        }
        assert_eq!(gpu.frames().len(), 7);
    }

    #[test]
    fn the_last_element_needs_no_stride_padding() {
        let mut gpu = RecordingGpu::new();
        let shader = gpu.create_shader(&SHADER).unwrap();
        let float = [VertexAttribute {
            format: VertexFormat::Float32,
            offset: 0,
            shader_location: 0,
        }];
        let mut pipeline = |step_mode, attributes| {
            let pipeline = gpu
                .create_pipeline(&PipelineDesc {
                    vertex_layouts: &[VertexLayout {
                        array_stride: 16,
                        step_mode,
                        attributes,
                    }],
                    ..pipeline_desc(shader)
                })
                .unwrap();
            RenderCmd::SetPipeline(pipeline)
        };
        let per_vertex = pipeline(VertexStepMode::Vertex, &float);
        let per_instance = pipeline(VertexStepMode::Instance, &float);
        let no_attributes = pipeline(VertexStepMode::Vertex, &[]);
        // Three 16-byte elements whose last four bytes hold the final float.
        let buffer = buffer(&mut gpu, 36, BufferUsage::VERTEX);
        let bind = |offset| RenderCmd::SetVertexBuffer {
            slot: 0,
            buffer,
            offset,
        };
        let draw = |first_vertex, vertex_count, first_instance, instance_count| {
            RenderCmd::Draw(DrawCmd::NonIndexed {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            })
        };

        let valid: [&[RenderCmd]; 4] = [
            &[per_vertex, bind(0), draw(0, 3, 0, 1)],
            &[per_instance, bind(0), draw(0, 3, 0, 3)],
            &[no_attributes, bind(4), draw(0, 3, 0, 1)],
            &[no_attributes, bind(36), draw(0, 1, 0, 1)],
        ];
        for commands in valid {
            submit_pass(&mut gpu, surface(), commands).unwrap();
        }
        let overrun = invalid("a draw must read only vertices inside its vertex buffers");
        let cases: [&[RenderCmd]; 4] = [
            &[per_vertex, bind(0), draw(0, 4, 0, 1)],
            &[per_instance, bind(0), draw(0, 3, 0, 4)],
            &[per_instance, bind(0), draw(0, 3, 1, 3)],
            &[no_attributes, bind(4), draw(0, 4, 0, 1)],
        ];
        for commands in cases {
            assert_eq!(
                submit_pass(&mut gpu, surface(), commands),
                overrun,
                "{commands:?}"
            );
        }
        assert_eq!(gpu.frames().len(), 4);
    }

    #[test]
    #[should_panic(expected = "color format")]
    fn depth_surface_format_panics() {
        let _ = RecordingGpu::with_surface_format(TextureFormat::Depth32Float);
    }
}
