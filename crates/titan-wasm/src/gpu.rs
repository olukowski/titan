//! [`WebGpuDevice`]: a [`GpuDevice`] that validates in Rust and draws through a [`HostBridge`].

use crate::abi::{self, SURFACE_ID, WireCmd, WireColor, WireDepth};
use crate::bridge::HostBridge;
use alloc::vec::Vec;
use core::mem;
use titan_gpu::{
    BlendFactor, BlendOperation, Buffer, BufferDesc, BufferUsage, DrawCmd, FrameView, GpuDevice,
    GpuError, Handle, IndexFormat, LoadOp, MAX_VERTEX_BUFFERS, Pipeline, PipelineDesc,
    RecordingGpu, RenderCmd, RenderPass, RenderPassDesc, ResourceKind, ResourcePool, Shader,
    ShaderSource, Texture, TextureDesc, TextureFormat, TextureUsage, VertexStepMode,
};

#[derive(Debug)]
struct BufferMeta {
    id: u32,
    size: u64,
    usage: BufferUsage,
}

#[derive(Debug)]
struct TextureMeta {
    id: u32,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: TextureUsage,
}

#[derive(Debug)]
struct ShaderMeta {
    id: u32,
}

/// What draw validation needs from a vertex layout.
#[derive(Debug)]
struct VertexLayoutMeta {
    array_stride: u32,
    step_mode: VertexStepMode,
    /// Where the last element's last attribute ends: the bytes a draw needs past the last full
    /// stride.
    last_element_bytes: u64,
}

#[derive(Debug)]
struct PipelineMeta {
    id: u32,
    color_format: Option<TextureFormat>,
    depth_format: Option<TextureFormat>,
    vertex_layouts: Vec<VertexLayoutMeta>,
}

/// A validated buffer binding.
struct Binding {
    id: u32,
    /// Lossless: offsets are at most a buffer's size, which is at most `MAX_BUFFER_SIZE`.
    offset: u32,
    /// The bytes from the offset to the end of the buffer.
    bound: u64,
}

/// A [`GpuDevice`] for browsers: it checks every call against the rules `titan-gpu` documents,
/// the same ones [`RecordingGpu`] enforces, then forwards it to a [`HostBridge`] that drives
/// WebGPU.
///
/// Resources are named across the bridge by `u32` ids from 1 up, never reused. A call that fails
/// validation never reaches the bridge, and a create the bridge rejects issues no handle.
/// Descriptor and frame payloads are encoded into one reused buffer, so steady-state frames
/// allocate nothing.
#[derive(Debug)]
pub struct WebGpuDevice<B: HostBridge> {
    bridge: B,
    surface_format: TextureFormat,
    buffers: ResourcePool<Buffer, BufferMeta>,
    textures: ResourcePool<Texture, TextureMeta>,
    shaders: ResourcePool<Shader, ShaderMeta>,
    pipelines: ResourcePool<Pipeline, PipelineMeta>,
    next_id: u32,
    scratch: Vec<u8>,
}

impl<B: HostBridge> WebGpuDevice<B> {
    /// Creates a device that draws through `bridge`, reading its surface format once.
    ///
    /// # Errors
    ///
    /// Returns the bridge's error, or [`GpuError::Backend`] if the surface has a depth format.
    pub fn new(mut bridge: B) -> Result<Self, GpuError> {
        let surface_format = bridge.surface_format()?;
        if surface_format.is_depth() {
            return Err(GpuError::Backend(alloc::format!(
                "the surface needs a color format, not {surface_format:?}"
            )));
        }
        Ok(Self {
            bridge,
            surface_format,
            buffers: ResourcePool::new(),
            textures: ResourcePool::new(),
            shaders: ResourcePool::new(),
            pipelines: ResourcePool::new(),
            next_id: 1,
            scratch: Vec::new(),
        })
    }

    /// Returns the bridge.
    #[must_use]
    pub fn bridge(&self) -> &B {
        &self.bridge
    }

    /// Returns the bridge mutably.
    #[must_use]
    pub fn bridge_mut(&mut self) -> &mut B {
        &mut self.bridge
    }

    /// Issues the next resource id. Ids never wrap, so a stale id can never name a new resource.
    fn take_id(&mut self) -> Result<u32, GpuError> {
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| GpuError::Backend("resource ids exhausted".into()))?;
        Ok(id)
    }

    fn buffer_with(
        &self,
        handle: Handle<Buffer>,
        usage: BufferUsage,
        flag: &'static str,
    ) -> Result<&BufferMeta, GpuError> {
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

    fn texture_with(
        &self,
        handle: Handle<Texture>,
        usage: TextureUsage,
        flag: &'static str,
    ) -> Result<&TextureMeta, GpuError> {
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
    fn binding(
        &self,
        handle: Handle<Buffer>,
        offset: u64,
        usage: BufferUsage,
        flag: &'static str,
    ) -> Result<Binding, GpuError> {
        let buffer = self.buffer_with(handle, usage, flag)?;
        let bound = buffer
            .size
            .checked_sub(offset)
            .ok_or(GpuError::BufferOutOfBounds {
                offset,
                write_len: 0,
                buffer_size: buffer.size,
            })?;
        Ok(Binding {
            id: buffer.id,
            offset: offset as u32,
            bound,
        })
    }

    /// Validates every pass of `frame`, encoding it into `out` as it goes. Nothing reaches the
    /// bridge until the whole frame is valid.
    fn encode_frame(&self, frame: FrameView<'_>, out: &mut Vec<u8>) -> Result<(), GpuError> {
        abi::begin_frame(wire_count(frame.pass_count())?, out);
        for pass in frame.passes() {
            self.encode_pass(pass, out)?;
        }
        Ok(())
    }

    fn encode_pass(&self, pass: RenderPass<'_>, out: &mut Vec<u8>) -> Result<(), GpuError> {
        let RenderPassDesc { color, depth, .. } = pass.desc;
        ensure(
            color.is_some() || depth.is_some(),
            "a render pass needs a color or depth attachment",
        )?;
        // What draws must match: the attachment formats, and the size when a texture is attached
        // (the surface's size is not known here).
        let mut color_format = None;
        let mut size = None;
        let mut wire_color = None;
        if let Some(color) = color {
            let target = match color.target {
                None => {
                    color_format = Some(self.surface_format);
                    SURFACE_ID
                }
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
                    color_format = Some(texture.format);
                    size = Some((texture.width, texture.height));
                    texture.id
                }
            };
            if let LoadOp::Clear(c) = color.load_op {
                ensure(
                    [c.r, c.g, c.b, c.a].iter().all(|v| v.is_finite()),
                    "a clear color must be finite",
                )?;
            }
            wire_color = Some(WireColor {
                target,
                load_op: color.load_op,
                store_op: color.store_op,
            });
        }
        let mut depth_format = None;
        let mut wire_depth = None;
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
            wire_depth = Some(WireDepth {
                target: texture.id,
                load_op: depth.depth_load_op,
                store_op: depth.depth_store_op,
            });
        }
        let attachments = [color.and_then(|c| c.target), depth.map(|d| d.target)];
        abi::encode_pass_header(
            wire_color,
            wire_depth,
            wire_count(pass.commands.len())?,
            out,
        );

        // Bindings do not carry over between passes, matching Metal and WebGPU encoders.
        let mut pipeline = None;
        // Per slot, the bytes the bound vertex buffer holds from its offset on.
        let mut vertex_buffers = [None; MAX_VERTEX_BUFFERS];
        // The bound index buffer's format and the bytes it holds from its offset on.
        let mut index_buffer = None;
        for &command in pass.commands {
            let wire = match command {
                RenderCmd::SetPipeline(handle) => {
                    let meta = self.pipelines.get(handle).ok_or(GpuError::InvalidHandle {
                        kind: ResourceKind::Pipeline,
                    })?;
                    ensure(
                        meta.color_format == color_format && meta.depth_format == depth_format,
                        "a pipeline's color_format and depth_stencil format must match its \
                         pass's attachments",
                    )?;
                    pipeline = Some(meta);
                    WireCmd::SetPipeline { pipeline: meta.id }
                }
                RenderCmd::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    let index = usize::try_from(slot)
                        .ok()
                        .filter(|&slot| slot < MAX_VERTEX_BUFFERS)
                        .ok_or(GpuError::InvalidDescriptor {
                            message: "a vertex buffer slot must be below MAX_VERTEX_BUFFERS",
                        })?;
                    let b = self.binding(buffer, offset, BufferUsage::VERTEX, "VERTEX")?;
                    vertex_buffers[index] = Some(b.bound);
                    WireCmd::SetVertexBuffer {
                        slot,
                        buffer: b.id,
                        offset: b.offset,
                    }
                }
                RenderCmd::SetIndexBuffer {
                    buffer,
                    format,
                    offset,
                } => {
                    let b = self.binding(buffer, offset, BufferUsage::INDEX, "INDEX")?;
                    ensure(
                        offset.is_multiple_of(u64::from(format.size_bytes())),
                        "an index buffer offset must be a multiple of the index size",
                    )?;
                    index_buffer = Some((format, b.bound));
                    WireCmd::SetIndexBuffer {
                        buffer: b.id,
                        format,
                        offset: b.offset,
                    }
                }
                RenderCmd::SetUniformBuffer {
                    binding,
                    buffer,
                    offset,
                } => {
                    let b = self.binding(buffer, offset, BufferUsage::UNIFORM, "UNIFORM")?;
                    WireCmd::SetUniformBuffer {
                        binding,
                        buffer: b.id,
                        offset: b.offset,
                    }
                }
                RenderCmd::SetStorageBuffer {
                    binding,
                    buffer,
                    offset,
                } => {
                    let b = self.binding(buffer, offset, BufferUsage::STORAGE, "STORAGE")?;
                    WireCmd::SetStorageBuffer {
                        binding,
                        buffer: b.id,
                        offset: b.offset,
                    }
                }
                RenderCmd::SetTexture { binding, texture } => {
                    let meta = self.texture_with(texture, TextureUsage::SAMPLED, "SAMPLED")?;
                    ensure(
                        !attachments.contains(&Some(texture)),
                        "a pass cannot sample its own color or depth attachment",
                    )?;
                    WireCmd::SetTexture {
                        binding,
                        texture: meta.id,
                    }
                }
                RenderCmd::SetViewport(v) => {
                    ensure(
                        [v.x, v.y, v.width, v.height].iter().all(|c| c.is_finite())
                            && v.width > 0.0
                            && v.height > 0.0
                            && 0.0 <= v.min_depth
                            && v.min_depth <= v.max_depth
                            && v.max_depth <= 1.0,
                        "a viewport must be finite, have a positive size, and have \
                         0 <= min_depth <= max_depth <= 1",
                    )?;
                    WireCmd::SetViewport(v)
                }
                RenderCmd::SetScissor(r) => {
                    ensure(
                        size.is_none_or(|(width, height)| {
                            u64::from(r.x) + u64::from(r.width) <= u64::from(width)
                                && u64::from(r.y) + u64::from(r.height) <= u64::from(height)
                        }),
                        "a scissor rectangle must lie within the pass's attachments",
                    )?;
                    WireCmd::SetScissor(r)
                }
                RenderCmd::Draw(draw) => {
                    check_draw(
                        pipeline.ok_or(GpuError::DrawWithoutPipeline)?,
                        &vertex_buffers,
                        index_buffer,
                        draw,
                    )?;
                    WireCmd::Draw(draw)
                }
            };
            abi::encode_cmd(wire, out);
        }
        Ok(())
    }
}

/// Checks that `draw` has the buffers `pipeline` reads and stays inside them, as
/// [`RecordingGpu`] does. `vertex_buffers` and `index_buffer` hold the bytes bound from each
/// buffer's offset on.
fn check_draw(
    pipeline: &PipelineMeta,
    vertex_buffers: &[Option<u64>; MAX_VERTEX_BUFFERS],
    index_buffer: Option<(IndexFormat, u64)>,
    draw: DrawCmd,
) -> Result<(), GpuError> {
    // One past the last element a draw reads per vertex and per instance. Vertices an index
    // buffer picks are only known on the GPU, so indexed draws leave per-vertex reads unchecked.
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
            ensure(
                (elements - 1)
                    .checked_mul(u64::from(layout.array_stride))
                    .and_then(|bytes| bytes.checked_add(layout.last_element_bytes))
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

/// Narrows a pass or command count to the wire's `u32`.
fn wire_count(count: usize) -> Result<u32, GpuError> {
    u32::try_from(count).map_err(|_| GpuError::InvalidDescriptor {
        message: "a frame has at most u32::MAX passes and a pass at most u32::MAX commands",
    })
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

impl<B: HostBridge> GpuDevice for WebGpuDevice<B> {
    fn surface_format(&self) -> TextureFormat {
        self.surface_format
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<Handle<Buffer>, GpuError> {
        ensure(desc.size > 0, "buffer size must be nonzero")?;
        ensure(
            desc.size <= RecordingGpu::MAX_BUFFER_SIZE,
            "buffer size exceeds MAX_BUFFER_SIZE",
        )?;
        ensure(!desc.usage.is_empty(), "buffer usage must not be empty")?;
        let id = self.take_id()?;
        // Lossless: the size is at most `MAX_BUFFER_SIZE`.
        self.bridge
            .create_buffer(id, desc.size as u32, desc.usage.bits(), desc.label)?;
        Ok(self.buffers.insert(BufferMeta {
            id,
            size: desc.size,
            usage: desc.usage,
        }))
    }

    fn write_buffer(
        &mut self,
        buffer: Handle<Buffer>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GpuError> {
        let meta = self.buffer_with(buffer, BufferUsage::COPY_DST, "COPY_DST")?;
        let write_len = data.len() as u64;
        offset
            .checked_add(write_len)
            .filter(|&end| end <= meta.size)
            .ok_or(GpuError::BufferOutOfBounds {
                offset,
                write_len,
                buffer_size: meta.size,
            })?;
        ensure(
            offset.is_multiple_of(4) && write_len.is_multiple_of(4),
            "a buffer write's offset and length must be multiples of 4",
        )?;
        // Lossless: the offset is below the buffer size, which is at most `MAX_BUFFER_SIZE`.
        self.bridge.write_buffer(meta.id, offset as u32, data)
    }

    fn destroy_buffer(&mut self, buffer: Handle<Buffer>) -> Result<(), GpuError> {
        let meta = self.buffers.remove(buffer).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        })?;
        self.bridge.destroy_buffer(meta.id);
        Ok(())
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> Result<Handle<Texture>, GpuError> {
        ensure(
            desc.width > 0 && desc.height > 0,
            "texture width and height must be nonzero",
        )?;
        ensure(
            desc.width <= RecordingGpu::MAX_TEXTURE_DIMENSION
                && desc.height <= RecordingGpu::MAX_TEXTURE_DIMENSION,
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
        let id = self.take_id()?;
        self.scratch.clear();
        abi::encode_texture_desc(desc, &mut self.scratch);
        self.bridge.create_texture(id, &self.scratch)?;
        Ok(self.textures.insert(TextureMeta {
            id,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            usage: desc.usage,
        }))
    }

    fn write_texture(&mut self, texture: Handle<Texture>, data: &[u8]) -> Result<(), GpuError> {
        let meta = self.texture_with(texture, TextureUsage::COPY_DST, "COPY_DST")?;
        let bytes_per_pixel = meta
            .format
            .bytes_per_pixel()
            .expect("create_texture rejects COPY_DST on depth formats; color formats have a size");
        // Cannot overflow: at most 8192 * 8192 * 4 bytes.
        let expected = meta.width as usize * meta.height as usize * bytes_per_pixel as usize;
        if data.len() != expected {
            return Err(GpuError::TextureDataSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        self.bridge.write_texture(meta.id, data)
    }

    fn destroy_texture(&mut self, texture: Handle<Texture>) -> Result<(), GpuError> {
        let meta = self
            .textures
            .remove(texture)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Texture,
            })?;
        self.bridge.destroy_texture(meta.id);
        Ok(())
    }

    fn create_shader(&mut self, source: &ShaderSource<'_>) -> Result<Handle<Shader>, GpuError> {
        ensure(
            !source.vertex_entry.is_empty() && !source.fragment_entry.is_empty(),
            "shader entry point names must not be empty",
        )?;
        ensure(!source.wgsl.is_empty(), "a WebGPU shader needs WGSL source")?;
        let id = self.take_id()?;
        self.scratch.clear();
        abi::encode_shader_source(source, &mut self.scratch);
        self.bridge.create_shader(id, &self.scratch)?;
        Ok(self.shaders.insert(ShaderMeta { id }))
    }

    fn destroy_shader(&mut self, shader: Handle<Shader>) -> Result<(), GpuError> {
        let meta = self.shaders.remove(shader).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Shader,
        })?;
        self.bridge.destroy_shader(meta.id);
        Ok(())
    }

    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<Handle<Pipeline>, GpuError> {
        let shader_id = self
            .shaders
            .get(desc.shader)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Shader,
            })?
            .id;
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
            desc.blend.is_none_or(|blend| {
                [blend.color, blend.alpha].iter().all(|c| {
                    !matches!(c.operation, BlendOperation::Min | BlendOperation::Max)
                        || (c.src_factor == BlendFactor::One && c.dst_factor == BlendFactor::One)
                })
            }),
            "min and max blending need a src_factor and dst_factor of One",
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
        let locations = || {
            desc.vertex_layouts
                .iter()
                .flat_map(|layout| layout.attributes)
                .map(|attribute| attribute.shader_location)
        };
        // The wire counts attributes in a `u8`.
        ensure(
            locations().count() <= usize::from(u8::MAX),
            "a pipeline has at most 255 vertex attributes",
        )?;
        // Quadratic, but over a handful of attributes, and smaller in WASM than sorting.
        ensure(
            locations()
                .enumerate()
                .all(|(i, location)| !locations().take(i).any(|earlier| earlier == location)),
            "a pipeline's vertex attributes need distinct shader_locations",
        )?;
        let id = self.take_id()?;
        self.scratch.clear();
        abi::encode_pipeline_desc(shader_id, desc, &mut self.scratch);
        self.bridge.create_pipeline(id, &self.scratch)?;
        Ok(self.pipelines.insert(PipelineMeta {
            id,
            color_format: desc.color_format,
            depth_format: desc.depth_stencil.map(|ds| ds.format),
            vertex_layouts: desc
                .vertex_layouts
                .iter()
                .map(|layout| VertexLayoutMeta {
                    array_stride: layout.array_stride,
                    step_mode: layout.step_mode,
                    last_element_bytes: layout
                        .attributes
                        .iter()
                        .map(|a| u64::from(a.offset) + u64::from(a.format.size_bytes()))
                        .max()
                        .unwrap_or(0),
                })
                .collect(),
        }))
    }

    fn destroy_pipeline(&mut self, pipeline: Handle<Pipeline>) -> Result<(), GpuError> {
        let meta = self
            .pipelines
            .remove(pipeline)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Pipeline,
            })?;
        self.bridge.destroy_pipeline(meta.id);
        Ok(())
    }

    fn submit(&mut self, commands: FrameView<'_>) -> Result<(), GpuError> {
        // Taken out so `encode_frame` can read the pools while writing it; put back either way so
        // its capacity carries over to the next frame.
        let mut out = mem::take(&mut self.scratch);
        let result = self
            .encode_frame(commands, &mut out)
            .and_then(|()| self.bridge.submit_frame(&out));
        self.scratch = out;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::decode::Reader;
    use crate::bridge::mock::{Call, MockBridge};
    use alloc::vec;
    use titan_gpu::{
        BlendComponent, BlendState, ClearColor, ColorAttachment, CullMode, DepthAttachment,
        DepthStencilState, FrameCommands, FrontFace, PrimitiveTopology, ScissorRect, StoreOp,
        VertexAttribute, VertexFormat, VertexLayout, Viewport,
    };

    const SHADER: ShaderSource<'static> = ShaderSource {
        label: Some("sprite"),
        msl: "",
        wgsl: "@vertex fn vs_main() {}",
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

    fn device() -> WebGpuDevice<MockBridge> {
        WebGpuDevice::new(MockBridge::new()).unwrap()
    }

    fn buffer(gpu: &mut WebGpuDevice<MockBridge>, size: u64, usage: BufferUsage) -> Handle<Buffer> {
        gpu.create_buffer(&BufferDesc {
            label: None,
            size,
            usage,
        })
        .unwrap()
    }

    fn texture(
        gpu: &mut WebGpuDevice<MockBridge>,
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

    struct Scene {
        pipeline: Handle<Pipeline>,
        buffer: Handle<Buffer>,
        texture: Handle<Texture>,
    }

    fn scene(gpu: &mut WebGpuDevice<MockBridge>) -> Scene {
        let shader = gpu.create_shader(&SHADER).unwrap();
        let pipeline = gpu.create_pipeline(&pipeline_desc(shader)).unwrap();
        let buffer = buffer(
            gpu,
            240,
            BufferUsage::VERTEX
                | BufferUsage::INDEX
                | BufferUsage::UNIFORM
                | BufferUsage::STORAGE
                | BufferUsage::COPY_DST,
        );
        let texture = texture(
            gpu,
            TextureFormat::Rgba8UnormSrgb,
            TextureUsage::SAMPLED | TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_DST,
        );
        Scene {
            pipeline,
            buffer,
            texture,
        }
    }

    fn submitted(gpu: &WebGpuDevice<MockBridge>) -> usize {
        gpu.bridge().frames().count()
    }

    #[test]
    fn creates_send_ids_from_one_and_payloads_the_host_can_decode() {
        let mut gpu = device();
        let shader = gpu.create_shader(&SHADER).unwrap();
        gpu.create_pipeline(&pipeline_desc(shader)).unwrap();
        let tex_desc = TextureDesc::rgba8_2d(2, 2);
        gpu.create_texture(&tex_desc).unwrap();
        let [
            Call::CreateShader(1, shader_bytes),
            Call::CreatePipeline(2, pipeline_bytes),
            Call::CreateTexture(3, texture_bytes),
        ] = gpu.bridge().calls.as_slice()
        else {
            panic!("unexpected calls: {:?}", gpu.bridge().calls);
        };
        assert_eq!(
            Reader::new(shader_bytes).shader_source().wgsl,
            "@vertex fn vs_main() {}"
        );
        let pipeline = Reader::new(pipeline_bytes).pipeline_desc();
        assert_eq!(pipeline.shader_id, 1);
        assert_eq!(pipeline.vertex_layouts[0].2, ATTRIBUTES);
        assert_eq!(Reader::new(texture_bytes).texture_desc(), tex_desc);
    }

    #[test]
    fn stale_handles_are_rejected_after_destroy_and_slot_reuse() {
        let mut gpu = device();
        let old = buffer(&mut gpu, 16, BufferUsage::COPY_DST);
        gpu.destroy_buffer(old).unwrap();
        let new = buffer(&mut gpu, 16, BufferUsage::COPY_DST);
        let stale = Err(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        });
        assert_eq!(gpu.write_buffer(old, 0, &[0; 4]), stale);
        assert_eq!(gpu.destroy_buffer(old), stale);
        gpu.write_buffer(new, 0, &[1, 2, 3, 4]).unwrap();
        // The new buffer got a fresh id, so the host can never confuse the two.
        assert_eq!(
            gpu.bridge().calls[1..],
            [
                Call::DestroyBuffer(1),
                Call::CreateBuffer {
                    id: 2,
                    size: 16,
                    usage: BufferUsage::COPY_DST.bits(),
                    label: None,
                },
                Call::WriteBuffer {
                    id: 2,
                    offset: 0,
                    data: vec![1, 2, 3, 4],
                },
            ]
        );

        let shader = gpu.create_shader(&SHADER).unwrap();
        gpu.destroy_shader(shader).unwrap();
        assert_eq!(
            gpu.create_pipeline(&pipeline_desc(shader)),
            Err(GpuError::InvalidHandle {
                kind: ResourceKind::Shader,
            })
        );
    }

    #[test]
    fn buffer_writes_check_usage_bounds_and_alignment() {
        let mut gpu = device();
        let readonly = buffer(&mut gpu, 16, BufferUsage::VERTEX);
        assert_eq!(
            gpu.write_buffer(readonly, 0, &[0; 4]),
            Err(GpuError::MissingUsage {
                kind: ResourceKind::Buffer,
                required: "COPY_DST",
            })
        );
        let target = buffer(&mut gpu, 16, BufferUsage::COPY_DST);
        assert_eq!(
            gpu.write_buffer(target, 12, &[0; 8]),
            Err(GpuError::BufferOutOfBounds {
                offset: 12,
                write_len: 8,
                buffer_size: 16,
            })
        );
        assert!(matches!(
            gpu.write_buffer(target, u64::MAX, &[0; 4]),
            Err(GpuError::BufferOutOfBounds { .. })
        ));
        assert!(gpu.write_buffer(target, 2, &[0; 4]).is_err());
        assert!(gpu.write_buffer(target, 0, &[0; 3]).is_err());
        let calls_before = gpu.bridge().calls.len();
        gpu.write_buffer(target, 12, &[0; 4]).unwrap();
        assert_eq!(gpu.bridge().calls.len(), calls_before + 1);
        for size in [0, RecordingGpu::MAX_BUFFER_SIZE + 1] {
            assert!(
                gpu.create_buffer(&BufferDesc {
                    label: None,
                    size,
                    usage: BufferUsage::VERTEX,
                })
                .is_err()
            );
        }
    }

    #[test]
    fn textures_check_mips_data_size_and_depth_copies() {
        let mut gpu = device();
        let too_many_mips = TextureDesc {
            mip_level_count: 4,
            ..TextureDesc::rgba8_2d(4, 4)
        };
        assert!(gpu.create_texture(&too_many_mips).is_err());
        let depth_upload = TextureDesc {
            format: TextureFormat::Depth32Float,
            ..TextureDesc::rgba8_2d(4, 4)
        };
        assert_eq!(
            gpu.create_texture(&depth_upload),
            Err(GpuError::InvalidDescriptor {
                message: "a depth texture cannot have COPY_DST",
            })
        );
        let texture = gpu.create_texture(&TextureDesc::rgba8_2d(4, 2)).unwrap();
        assert_eq!(
            gpu.write_texture(texture, &[0; 31]),
            Err(GpuError::TextureDataSizeMismatch {
                expected: 32,
                actual: 31,
            })
        );
        gpu.write_texture(texture, &[7; 32]).unwrap();
        assert_eq!(
            gpu.bridge().calls.last(),
            Some(&Call::WriteTexture(1, vec![7; 32]))
        );
    }

    #[test]
    fn failed_creates_issue_no_handle() {
        let mut gpu = device();
        gpu.bridge_mut().fail_next = Some("WGSL parse error");
        assert_eq!(
            gpu.create_shader(&SHADER),
            Err(GpuError::Backend("WGSL parse error".into()))
        );
        assert!(gpu.shaders.is_empty());
        let shader = gpu.create_shader(&SHADER).unwrap();

        gpu.bridge_mut().fail_next = Some("out of memory");
        assert!(gpu.create_pipeline(&pipeline_desc(shader)).is_err());
        gpu.bridge_mut().fail_next = Some("out of memory");
        assert!(
            gpu.create_buffer(&BufferDesc {
                label: Some("big"),
                size: 64,
                usage: BufferUsage::VERTEX,
            })
            .is_err()
        );
        gpu.bridge_mut().fail_next = Some("out of memory");
        assert!(gpu.create_texture(&TextureDesc::rgba8_2d(1, 1)).is_err());
        assert!(gpu.pipelines.is_empty() && gpu.buffers.is_empty() && gpu.textures.is_empty());
        assert_eq!(gpu.bridge().calls.len(), 1);
    }

    #[test]
    fn invalid_pipelines_never_reach_the_host() {
        let mut gpu = device();
        let shader = gpu.create_shader(&SHADER).unwrap();
        let overlapping = [VertexAttribute {
            format: VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        }];
        let short_stride = [VertexLayout {
            array_stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: &overlapping,
        }];
        let duplicate_location = [LAYOUTS[0], LAYOUTS[0]];
        let weighted_max = BlendState {
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Max,
            },
            ..BlendState::ALPHA_BLENDING
        };
        let weighted_min = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Min,
            },
            ..BlendState::ALPHA_BLENDING
        };
        let invalid = [
            PipelineDesc {
                color_format: None,
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                color_format: Some(TextureFormat::Depth32Float),
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Rgba8Unorm,
                    ..DepthStencilState::DEFAULT_3D
                }),
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                vertex_layouts: &short_stride,
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                vertex_layouts: &duplicate_location,
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                vertex_layouts: &[LAYOUTS[0]; MAX_VERTEX_BUFFERS + 1],
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                blend: Some(weighted_max),
                ..pipeline_desc(shader)
            },
            PipelineDesc {
                blend: Some(weighted_min),
                ..pipeline_desc(shader)
            },
        ];
        for desc in invalid {
            assert!(
                matches!(
                    gpu.create_pipeline(&desc),
                    Err(GpuError::InvalidDescriptor { .. })
                ),
                "{desc:?}"
            );
        }
        assert_eq!(gpu.bridge().calls.len(), 1);
    }

    #[test]
    fn valid_frame_encodes_every_command_with_host_ids() {
        let mut gpu = device();
        let s = scene(&mut gpu);
        let depth = texture(
            &mut gpu,
            TextureFormat::Depth32Float,
            TextureUsage::RENDER_ATTACHMENT,
        );
        // WebGPU storage offsets must be multiples of 256.
        let storage = buffer(&mut gpu, 512, BufferUsage::STORAGE);
        let mut frame = FrameCommands::new();
        frame
            .begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
            .set_pipeline(s.pipeline)
            .set_vertex_buffer(0, s.buffer, 12)
            .set_index_buffer(s.buffer, IndexFormat::Uint16, 2)
            .set_uniform_buffer(0, s.buffer, 0)
            .set_storage_buffer(1, storage, 256)
            .set_texture(1, s.texture)
            .set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0))
            .set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 9000,
                height: 9000,
            })
            .draw(0..19, 0..1)
            .draw_indexed(0..6, -1, 0..1);
        frame.begin_pass(RenderPassDesc {
            label: None,
            color: None,
            depth: Some(DepthAttachment {
                target: depth,
                depth_load_op: LoadOp::Clear(1.0),
                depth_store_op: StoreOp::Discard,
            }),
        });
        gpu.submit(frame.view()).unwrap();

        let passes = Reader::new(gpu.bridge().frames().next().unwrap()).frame();
        // Ids: shader 1, pipeline 2, buffer 3, texture 4, depth 5, storage 6.
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].color.unwrap().target, SURFACE_ID);
        assert_eq!(
            passes[0].commands[..6],
            [
                WireCmd::SetPipeline { pipeline: 2 },
                WireCmd::SetVertexBuffer {
                    slot: 0,
                    buffer: 3,
                    offset: 12,
                },
                WireCmd::SetIndexBuffer {
                    buffer: 3,
                    format: IndexFormat::Uint16,
                    offset: 2,
                },
                WireCmd::SetUniformBuffer {
                    binding: 0,
                    buffer: 3,
                    offset: 0,
                },
                WireCmd::SetStorageBuffer {
                    binding: 1,
                    buffer: 6,
                    offset: 256,
                },
                WireCmd::SetTexture {
                    binding: 1,
                    texture: 4,
                },
            ]
        );
        assert_eq!(passes[0].commands.len(), 10);
        assert_eq!(passes[1].depth.unwrap().target, 5);
        assert_eq!(passes[1].depth.unwrap().load_op, LoadOp::Clear(1.0));
    }

    #[test]
    fn a_frame_with_any_invalid_command_sends_nothing() {
        let mut gpu = device();
        let s = scene(&mut gpu);
        let other_size = gpu
            .create_texture(&TextureDesc {
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::RENDER_ATTACHMENT,
                ..TextureDesc::rgba8_2d(8, 8)
            })
            .unwrap();
        type Case = (fn(&Scene, &mut FrameCommands), GpuError);
        let cases: [Case; 7] = [
            (
                |_, f| {
                    f.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                        .draw(0..3, 0..1);
                },
                GpuError::DrawWithoutPipeline,
            ),
            (
                |s, f| {
                    f.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                        .set_pipeline(s.pipeline)
                        .set_vertex_buffer(0, s.buffer, 0)
                        .draw_indexed(0..3, 0, 0..1);
                },
                GpuError::DrawIndexedWithoutIndexBuffer,
            ),
            (
                |s, f| {
                    // 21 vertices of 12 bytes end at byte 248, past the 240-byte buffer.
                    f.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                        .set_pipeline(s.pipeline)
                        .set_vertex_buffer(0, s.buffer, 0)
                        .draw(0..21, 0..1);
                },
                GpuError::InvalidDescriptor {
                    message: "a draw must read only vertices inside its vertex buffers",
                },
            ),
            (
                |s, f| {
                    f.begin_pass(RenderPassDesc {
                        label: None,
                        color: Some(ColorAttachment {
                            target: Some(s.texture),
                            load_op: LoadOp::Load,
                            store_op: StoreOp::Store,
                        }),
                        depth: None,
                    })
                    .set_texture(0, s.texture);
                },
                GpuError::InvalidDescriptor {
                    message: "a pass cannot sample its own color or depth attachment",
                },
            ),
            (
                |s, f| {
                    // The pipeline targets the Bgra8UnormSrgb surface, not an Rgba8UnormSrgb
                    // texture.
                    f.begin_pass(RenderPassDesc {
                        label: None,
                        color: Some(ColorAttachment {
                            target: Some(s.texture),
                            load_op: LoadOp::Load,
                            store_op: StoreOp::Store,
                        }),
                        depth: None,
                    })
                    .set_pipeline(s.pipeline);
                },
                GpuError::InvalidDescriptor {
                    message: "a pipeline's color_format and depth_stencil format must match its \
                              pass's attachments",
                },
            ),
            (
                |s, f| {
                    f.begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                        .set_uniform_buffer(0, s.buffer, 241);
                },
                GpuError::BufferOutOfBounds {
                    offset: 241,
                    write_len: 0,
                    buffer_size: 240,
                },
            ),
            (
                |_, f| {
                    f.begin_pass(RenderPassDesc::clear_surface(ClearColor::rgba(
                        f64::NAN,
                        0.0,
                        0.0,
                        1.0,
                    )));
                },
                GpuError::InvalidDescriptor {
                    message: "a clear color must be finite",
                },
            ),
        ];
        for (build, expected) in cases {
            let mut frame = FrameCommands::new();
            // A valid pass first: a failure later in the frame must still send nothing.
            frame
                .begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                .set_pipeline(s.pipeline);
            build(&s, &mut frame);
            assert_eq!(gpu.submit(frame.view()), Err(expected));
        }
        let mut mismatched = FrameCommands::new();
        mismatched.begin_pass(RenderPassDesc {
            label: None,
            color: Some(ColorAttachment {
                target: Some(s.texture),
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
            }),
            depth: Some(DepthAttachment {
                target: other_size,
                depth_load_op: LoadOp::Load,
                depth_store_op: StoreOp::Store,
            }),
        });
        assert!(gpu.submit(mismatched.view()).is_err());
        assert_eq!(submitted(&gpu), 0);
    }

    /// Counts heap allocations on the current thread, so parallel tests do not interfere.
    mod counting {
        extern crate std;
        use core::alloc::{GlobalAlloc, Layout};
        use core::cell::Cell;
        use std::alloc::System;

        std::thread_local! {
            static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
        }

        struct Counting;

        // SAFETY: every call forwards to `System`, which upholds the `GlobalAlloc` contract.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
                // SAFETY: the caller's contract for `alloc` is passed on unchanged.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: the caller's contract for `dealloc` is passed on unchanged.
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
                // SAFETY: the caller's contract for `realloc` is passed on unchanged.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        #[global_allocator]
        static GLOBAL: Counting = Counting;

        pub(super) fn allocations() -> usize {
            ALLOCATIONS.with(Cell::get)
        }
    }

    /// A bridge that accepts everything and keeps nothing, so only the device's own
    /// allocations count.
    #[derive(Debug)]
    struct NullBridge;

    impl HostBridge for NullBridge {
        fn surface_format(&mut self) -> Result<TextureFormat, GpuError> {
            Ok(TextureFormat::Bgra8UnormSrgb)
        }
        fn create_buffer(
            &mut self,
            _: u32,
            _: u32,
            _: u32,
            _: Option<&str>,
        ) -> Result<(), GpuError> {
            Ok(())
        }
        fn write_buffer(&mut self, _: u32, _: u32, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
        fn destroy_buffer(&mut self, _: u32) {}
        fn create_texture(&mut self, _: u32, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
        fn write_texture(&mut self, _: u32, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
        fn destroy_texture(&mut self, _: u32) {}
        fn create_shader(&mut self, _: u32, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
        fn destroy_shader(&mut self, _: u32) {}
        fn create_pipeline(&mut self, _: u32, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
        fn destroy_pipeline(&mut self, _: u32) {}
        fn submit_frame(&mut self, _: &[u8]) -> Result<(), GpuError> {
            Ok(())
        }
    }

    #[test]
    fn steady_state_frames_do_not_allocate() {
        let mut gpu = WebGpuDevice::new(NullBridge).unwrap();
        let shader = gpu.create_shader(&SHADER).unwrap();
        let pipeline = gpu.create_pipeline(&pipeline_desc(shader)).unwrap();
        let vertices = gpu
            .create_buffer(&BufferDesc {
                label: None,
                size: 36,
                usage: BufferUsage::VERTEX | BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })
            .unwrap();
        let mut frame = FrameCommands::new();
        let mut run_frame = |gpu: &mut WebGpuDevice<NullBridge>| {
            frame.clear();
            frame
                .begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertices, 0)
                .set_uniform_buffer(0, vertices, 0)
                .draw(0..3, 0..1);
            gpu.write_buffer(vertices, 0, &[0; 16]).unwrap();
            gpu.submit(frame.view()).unwrap();
        };
        run_frame(&mut gpu);
        let before = counting::allocations();
        for _ in 0..10 {
            run_frame(&mut gpu);
        }
        assert_eq!(counting::allocations(), before);
    }
}
