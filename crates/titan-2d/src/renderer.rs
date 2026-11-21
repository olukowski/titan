//! [`Renderer2D`]: uploads sprites and records batched draws.

use crate::camera::Camera2D;
use crate::draw_list::{DrawList2D, Sprite};
use crate::shader::{SPRITE_SHADER, Vertex2D};
use alloc::vec::Vec;
use core::alloc::Allocator;
use core::mem;
use titan_core::math::sin_cos;
use titan_core::{Color, Handle, Rect, Vec2};
use titan_gpu::{
    BlendState, Buffer, BufferDesc, BufferUsage, CullMode, FrameCommands, FrontFace, GpuDevice,
    GpuError, IndexFormat, Pipeline, PipelineDesc, PrimitiveTopology, RenderPassDesc,
    RenderPassEncoder, ResourceKind, SamplerDesc, Shader, Texture, TextureDesc, TextureFormat,
};

/// The number of quads a new [`Renderer2D`]'s buffers hold. They grow to the next power of two
/// when a frame needs more.
pub const INITIAL_QUAD_CAPACITY: usize = 64;

const VERTICES_PER_QUAD: usize = 4;
const INDICES_PER_QUAD: usize = 6;
/// Two counter-clockwise triangles over a quad's corners: bottom-left, bottom-right, top-right,
/// top-left.
const QUAD_INDICES: [u32; INDICES_PER_QUAD] = [0, 1, 2, 0, 2, 3];
/// The uniform buffer holds the camera's `view_proj` matrix.
const UNIFORM_SIZE: u64 = 64;

/// The GPU objects a live renderer owns.
#[derive(Debug)]
struct GpuResources {
    pipeline: Handle<Pipeline>,
    shader: Handle<Shader>,
    /// Bound for untextured sprites, so they batch with each other and with sprites that name
    /// it explicitly.
    white_texture: Handle<Texture>,
    uniform_buffer: Handle<Buffer>,
    vertex_buffer: Handle<Buffer>,
    index_buffer: Handle<Buffer>,
    /// The number of quads the vertex and index buffers hold.
    quad_capacity: usize,
}

/// A run of consecutive quads, after sorting, that share a texture: one draw call.
#[derive(Clone, Copy, Debug)]
struct Batch {
    texture: Handle<Texture>,
    first_quad: u32,
    quad_count: u32,
}

/// A handle this crate created, for cleaning up after a failed creation or on destroy.
#[derive(Clone, Copy, Debug)]
enum Owned {
    Buffer(Handle<Buffer>),
    Texture(Handle<Texture>),
    Shader(Handle<Shader>),
    Pipeline(Handle<Pipeline>),
}

/// Draws [`Sprite`]s in batches through a [`GpuDevice`].
///
/// Queue sprites with [`draw_sprite`](Self::draw_sprite) and friends, then
/// [`flush`](Self::flush) them into a frame. A flush sorts sprites by
/// [`layer`](Sprite::layer), keeping submission order within a layer, and records one indexed
/// draw per run of sprites that share a texture; untextured sprites share the renderer's
/// [`white_texture`](Self::white_texture). To draw a [`DrawList2D`] you own, use
/// [`flush_list`](Self::flush_list).
///
/// Every flush writes the camera and the sprites to the same GPU buffers from offset 0, so
/// [`submit`](GpuDevice::submit) a frame before flushing again with the same renderer: a frame
/// holding two flushes of one renderer would draw the second flush's sprites twice.
///
/// Call [`destroy`](Self::destroy) to free the GPU objects; dropping the renderer leaks them.
#[derive(Debug)]
pub struct Renderer2D {
    /// `None` once destroyed.
    resources: Option<GpuResources>,
    /// Objects the device failed to destroy with [`GpuError::Backend`], retried by
    /// [`destroy`](Self::destroy).
    pending_destroy: Vec<Owned>,
    draw_list: DrawList2D,
    // Scratch reused by every flush, so steady-state frames do not allocate.
    vertex_bytes: Vec<u8>,
    batches: Vec<Batch>,
    /// `(layer, submission index)` of each sprite, sorted.
    sort_order: Vec<(i32, u32)>,
}

impl Renderer2D {
    /// Creates a renderer that draws into passes whose color attachment is the window surface.
    ///
    /// # Errors
    ///
    /// Returns the device's error if creating a GPU object fails, after attempting to destroy
    /// the objects already created.
    pub fn new(gpu: &mut dyn GpuDevice) -> Result<Self, GpuError> {
        let color_format = gpu.surface_format();
        Self::with_color_format(gpu, color_format)
    }

    /// Creates a renderer that draws into passes whose color attachment has `color_format` and
    /// that have no depth attachment.
    ///
    /// # Errors
    ///
    /// Returns the device's error if creating a GPU object fails, after attempting to destroy
    /// the objects already created.
    pub fn with_color_format(
        gpu: &mut dyn GpuDevice,
        color_format: TextureFormat,
    ) -> Result<Self, GpuError> {
        // There is no renderer to retry objects the device fails to destroy, so drop them.
        let resources = destroy_on_error(gpu, &mut Vec::new(), |gpu, created| {
            let white_texture = gpu.create_texture(&TextureDesc {
                label: Some("titan-2d white"),
                format: TextureFormat::Rgba8Unorm,
                sampler: SamplerDesc::NEAREST_CLAMP,
                ..TextureDesc::rgba8_2d(1, 1)
            })?;
            created.push(Owned::Texture(white_texture));
            gpu.write_texture(white_texture, &[255; 4])?;

            let shader = gpu.create_shader(&SPRITE_SHADER)?;
            created.push(Owned::Shader(shader));
            let pipeline = gpu.create_pipeline(&PipelineDesc {
                label: Some("titan-2d sprites"),
                shader,
                vertex_layouts: &[Vertex2D::LAYOUT],
                color_format: Some(color_format),
                blend: Some(BlendState::ALPHA_BLENDING),
                depth_stencil: None,
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
            })?;
            created.push(Owned::Pipeline(pipeline));

            let uniform_buffer = gpu.create_buffer(&BufferDesc {
                label: Some("titan-2d camera"),
                size: UNIFORM_SIZE,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })?;
            created.push(Owned::Buffer(uniform_buffer));
            let (vertex_buffer, index_buffer) =
                create_quad_buffers(gpu, created, INITIAL_QUAD_CAPACITY)?;
            Ok(GpuResources {
                pipeline,
                shader,
                white_texture,
                uniform_buffer,
                vertex_buffer,
                index_buffer,
                quad_capacity: INITIAL_QUAD_CAPACITY,
            })
        })?;
        Ok(Self {
            resources: Some(resources),
            pending_destroy: Vec::new(),
            draw_list: DrawList2D::new(),
            vertex_bytes: Vec::new(),
            batches: Vec::new(),
            sort_order: Vec::new(),
        })
    }

    /// Queues `sprite` for the next [`flush`](Self::flush).
    pub fn draw_sprite(&mut self, sprite: Sprite) {
        self.draw_list.draw_sprite(sprite);
    }

    /// Queues a solid quad of `color` covering `rect`.
    pub fn draw_rect(&mut self, rect: Rect, color: Color) {
        self.draw_list.draw_rect(rect, color);
    }

    /// Queues a solid quad of `color` whose bottom-left corner is `position` and whose size is
    /// `size`.
    pub fn draw_quad(&mut self, position: Vec2, size: Vec2, color: Color) {
        self.draw_list.draw_quad(position, size, color);
    }

    /// Discards the queued sprites.
    pub fn clear(&mut self) {
        self.draw_list.clear();
    }

    /// Returns the number of queued sprites.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.draw_list.len()
    }

    /// Returns whether no sprite is queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.draw_list.is_empty()
    }

    /// Returns the queued sprites.
    #[must_use]
    pub fn draw_list(&self) -> &DrawList2D {
        &self.draw_list
    }

    /// Returns the queued sprites mutably.
    pub fn draw_list_mut(&mut self) -> &mut DrawList2D {
        &mut self.draw_list
    }

    /// Returns the 1×1 opaque white texture untextured sprites draw with, or `None` once
    /// destroyed. Sprites that name it batch with untextured ones.
    #[must_use]
    pub fn white_texture(&self) -> Option<Handle<Texture>> {
        self.resources.as_ref().map(|r| r.white_texture)
    }

    /// Returns the sprite pipeline, or `None` once destroyed.
    #[must_use]
    pub fn pipeline(&self) -> Option<Handle<Pipeline>> {
        self.resources.as_ref().map(|r| r.pipeline)
    }

    /// Uploads the queued sprites, records them as a new render pass `pass` in `commands`, and
    /// clears the queue. With no sprites queued it uploads nothing and records `pass` with no
    /// commands, so a clear still happens.
    ///
    /// `pass` must have a color attachment of the renderer's color format and no depth
    /// attachment. See [`Renderer2D`] for when to submit.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidHandle`] for [`ResourceKind::Pipeline`] after
    /// [`destroy`](Self::destroy), or the device's error if an upload or freeing outgrown
    /// buffers fails. On error nothing is recorded and the queue is kept.
    ///
    /// # Panics
    ///
    /// Panics if `camera` is invalid; see [`Camera2D`].
    pub fn flush<A: Allocator>(
        &mut self,
        gpu: &mut dyn GpuDevice,
        commands: &mut FrameCommands<A>,
        pass: RenderPassDesc,
        camera: &Camera2D,
    ) -> Result<(), GpuError> {
        let list = mem::take(&mut self.draw_list);
        let result = self.flush_list(gpu, commands, pass, camera, &list);
        self.restore_draw_list(list, result.is_ok());
        result
    }

    /// Like [`flush`](Self::flush), but records into a pass the caller has already begun, after
    /// its existing commands. With no sprites queued it records nothing.
    ///
    /// # Errors
    ///
    /// As for [`flush`](Self::flush).
    ///
    /// # Panics
    ///
    /// Panics if `camera` is invalid; see [`Camera2D`].
    pub fn flush_into_pass<A: Allocator>(
        &mut self,
        gpu: &mut dyn GpuDevice,
        pass: &mut RenderPassEncoder<'_, A>,
        camera: &Camera2D,
    ) -> Result<(), GpuError> {
        let list = mem::take(&mut self.draw_list);
        let result = self.flush_list_into_pass(gpu, pass, camera, &list);
        self.restore_draw_list(list, result.is_ok());
        result
    }

    /// Like [`flush`](Self::flush), but draws `list` instead of the queue, leaving both
    /// unchanged.
    ///
    /// # Errors
    ///
    /// As for [`flush`](Self::flush).
    ///
    /// # Panics
    ///
    /// Panics if `camera` is invalid; see [`Camera2D`].
    pub fn flush_list<A: Allocator>(
        &mut self,
        gpu: &mut dyn GpuDevice,
        commands: &mut FrameCommands<A>,
        pass: RenderPassDesc,
        camera: &Camera2D,
        list: &DrawList2D,
    ) -> Result<(), GpuError> {
        // Upload before beginning the pass, so a failed upload records nothing.
        self.upload(gpu, camera, list.sprites())?;
        self.record(&mut commands.begin_pass(pass));
        Ok(())
    }

    /// Like [`flush_into_pass`](Self::flush_into_pass), but draws `list` instead of the queue,
    /// leaving both unchanged.
    ///
    /// # Errors
    ///
    /// As for [`flush`](Self::flush).
    ///
    /// # Panics
    ///
    /// Panics if `camera` is invalid; see [`Camera2D`].
    pub fn flush_list_into_pass<A: Allocator>(
        &mut self,
        gpu: &mut dyn GpuDevice,
        pass: &mut RenderPassEncoder<'_, A>,
        camera: &Camera2D,
        list: &DrawList2D,
    ) -> Result<(), GpuError> {
        self.upload(gpu, camera, list.sprites())?;
        self.record(pass);
        Ok(())
    }

    /// Destroys the renderer's GPU objects. Later flushes fail; destroying again only retries
    /// objects the device failed to destroy, so after a successful destroy it does nothing.
    ///
    /// # Errors
    ///
    /// Returns the first error from the device. Every object is attempted even if an earlier
    /// one fails, and the renderer counts as destroyed either way. Objects that failed with
    /// [`GpuError::Backend`] are kept for the next destroy to retry.
    pub fn destroy(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
        let mut owned = mem::take(&mut self.pending_destroy);
        if let Some(r) = self.resources.take() {
            owned.extend([
                Owned::Pipeline(r.pipeline),
                Owned::Shader(r.shader),
                Owned::Texture(r.white_texture),
                Owned::Buffer(r.uniform_buffer),
                Owned::Buffer(r.vertex_buffer),
                Owned::Buffer(r.index_buffer),
            ]);
        }
        destroy_all(gpu, &owned, &mut self.pending_destroy)
    }

    /// Puts the queue back after a flush took it, emptied if the flush succeeded.
    fn restore_draw_list(&mut self, mut list: DrawList2D, flushed: bool) {
        if flushed {
            list.clear();
        }
        self.draw_list = list;
    }

    /// Sorts `sprites` into batches and uploads the camera and their vertices, growing the
    /// buffers if needed. Leaves `batches` describing what to record. If growing fails, the
    /// old buffers are kept.
    fn upload(
        &mut self,
        gpu: &mut dyn GpuDevice,
        camera: &Camera2D,
        sprites: &[Sprite],
    ) -> Result<(), GpuError> {
        let resources = self.resources.as_mut().ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Pipeline,
        })?;
        let view_proj = camera.view_projection();
        self.batches.clear();
        if sprites.is_empty() {
            return Ok(());
        }
        if u32::try_from(sprites.len()).is_err() {
            return Err(GpuError::InvalidDescriptor {
                message: "titan-2d: quad count exceeds u32",
            });
        }

        self.sort_order.clear();
        self.sort_order.extend(
            (0..)
                .zip(sprites)
                .map(|(index, sprite)| (sprite.layer, index)),
        );
        // Submission indices are unique, so an unstable sort still keeps sprites within a layer
        // in submission order, and it does not allocate.
        self.sort_order.sort_unstable();

        self.vertex_bytes.clear();
        for (quad, &(_, index)) in (0..).zip(&self.sort_order) {
            let sprite = &sprites[index as usize];
            let texture = sprite.texture.unwrap_or(resources.white_texture);
            match self.batches.last_mut() {
                Some(batch) if batch.texture == texture => batch.quad_count += 1,
                _ => self.batches.push(Batch {
                    texture,
                    first_quad: quad,
                    quad_count: 1,
                }),
            }
            write_quad(sprite, &mut self.vertex_bytes);
        }

        let mut uniform = [0; UNIFORM_SIZE as usize];
        let floats = view_proj.cols.iter().flat_map(|c| [c.x, c.y, c.z, c.w]);
        for (bytes, value) in uniform.as_chunks_mut::<4>().0.iter_mut().zip(floats) {
            *bytes = value.to_le_bytes();
        }
        if sprites.len() <= resources.quad_capacity {
            gpu.write_buffer(resources.uniform_buffer, 0, &uniform)?;
            return gpu.write_buffer(resources.vertex_buffer, 0, &self.vertex_bytes);
        }

        // Grow to the next power of two. Fill the new buffers before swapping them in, so a
        // failed upload leaves the old ones in use.
        let capacity =
            sprites
                .len()
                .checked_next_power_of_two()
                .ok_or(GpuError::InvalidDescriptor {
                    message: "titan-2d: quad count overflows usize",
                })?;
        let (vertex_buffer, index_buffer) =
            destroy_on_error(gpu, &mut self.pending_destroy, |gpu, created| {
                let (vertex_buffer, index_buffer) = create_quad_buffers(gpu, created, capacity)?;
                gpu.write_buffer(resources.uniform_buffer, 0, &uniform)?;
                gpu.write_buffer(vertex_buffer, 0, &self.vertex_bytes)?;
                Ok((vertex_buffer, index_buffer))
            })?;
        let old_vertex = mem::replace(&mut resources.vertex_buffer, vertex_buffer);
        let old_index = mem::replace(&mut resources.index_buffer, index_buffer);
        resources.quad_capacity = capacity;
        destroy_all(
            gpu,
            &[Owned::Buffer(old_vertex), Owned::Buffer(old_index)],
            &mut self.pending_destroy,
        )
    }

    /// Records the draws [`upload`](Self::upload) prepared; nothing if there are none.
    fn record<A: Allocator>(&self, pass: &mut RenderPassEncoder<'_, A>) {
        if self.batches.is_empty() {
            return;
        }
        let resources = self
            .resources
            .as_ref()
            .expect("upload found the renderer live");
        pass.set_pipeline(resources.pipeline)
            .set_uniform_buffer(0, resources.uniform_buffer, 0)
            .set_vertex_buffer(0, resources.vertex_buffer, 0)
            .set_index_buffer(resources.index_buffer, IndexFormat::Uint32, 0);
        // No overflow: `create_quad_buffers` checks that every index count fits `u32`.
        let indices_per_quad = INDICES_PER_QUAD as u32;
        for batch in &self.batches {
            let first = batch.first_quad * indices_per_quad;
            pass.set_texture(0, batch.texture).draw_indexed(
                first..first + batch.quad_count * indices_per_quad,
                0,
                0..1,
            );
        }
    }
}

/// Appends the four vertices of `sprite`: bottom-left, bottom-right, top-right, top-left.
fn write_quad(sprite: &Sprite, out: &mut Vec<u8>) {
    let Rect { min, max } = sprite.rect;
    let mut corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
    if sprite.rotation != 0.0 {
        let (sin, cos) = sin_cos(sprite.rotation);
        let center = sprite.rect.center();
        for corner in &mut corners {
            let d = *corner - center;
            *corner = center + Vec2::new(d.x * cos - d.y * sin, d.x * sin + d.y * cos);
        }
    }
    // Texture rows run top to bottom, so the quad's top edge samples `uv.min.y`.
    let Rect {
        min: uv_min,
        max: uv_max,
    } = sprite.uv;
    let uvs = [
        [uv_min.x, uv_max.y],
        [uv_max.x, uv_max.y],
        [uv_max.x, uv_min.y],
        [uv_min.x, uv_min.y],
    ];
    for (corner, uv) in corners.into_iter().zip(uvs) {
        Vertex2D {
            position: [corner.x, corner.y],
            uv,
            color: sprite.color.to_array(),
        }
        .write_bytes(out);
    }
}

/// Creates a vertex buffer and a filled index buffer for `capacity` quads, recording each in
/// `created`.
fn create_quad_buffers(
    gpu: &mut dyn GpuDevice,
    created: &mut Vec<Owned>,
    capacity: usize,
) -> Result<(Handle<Buffer>, Handle<Buffer>), GpuError> {
    let too_many = GpuError::InvalidDescriptor {
        message: "titan-2d: quad capacity exceeds u32 indices or u64 buffer sizes",
    };
    // Vertex numbers and index counts must fit the `u32` indices and draw ranges.
    let index_count = capacity
        .checked_mul(INDICES_PER_QUAD)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(too_many.clone())?;
    let vertex_count = capacity
        .checked_mul(VERTICES_PER_QUAD)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(too_many.clone())?;
    let vertex_size = u64::from(vertex_count) * u64::from(Vertex2D::SIZE_BYTES);
    let index_size = u64::from(index_count) * u64::from(IndexFormat::Uint32.size_bytes());

    let vertex_buffer = gpu.create_buffer(&BufferDesc {
        label: Some("titan-2d vertices"),
        size: vertex_size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
    })?;
    created.push(Owned::Buffer(vertex_buffer));
    let index_buffer = gpu.create_buffer(&BufferDesc {
        label: Some("titan-2d indices"),
        size: index_size,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
    })?;
    created.push(Owned::Buffer(index_buffer));

    // Lossless: `index_size` came from a `usize` count times 4 and fits the device's buffer.
    let mut indices = Vec::with_capacity(index_size as usize);
    for quad in 0..vertex_count / 4 {
        for index in QUAD_INDICES {
            indices.extend_from_slice(&(quad * 4 + index).to_le_bytes());
        }
    }
    gpu.write_buffer(index_buffer, 0, &indices)?;
    Ok((vertex_buffer, index_buffer))
}

/// Runs `build`, which records every object it creates in its third argument. If `build`
/// fails, destroys them as [`destroy_all`] does and returns its error.
fn destroy_on_error<T>(
    gpu: &mut dyn GpuDevice,
    failed: &mut Vec<Owned>,
    build: impl FnOnce(&mut dyn GpuDevice, &mut Vec<Owned>) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    let mut created = Vec::new();
    build(gpu, &mut created).inspect_err(|_| {
        // The build error is the one to report.
        let _ = destroy_all(gpu, &created, failed);
    })
}

/// Destroys every object in `owned`, newest first, and returns the first error. Appends each
/// object the device failed to destroy with [`GpuError::Backend`] to `failed`; other errors
/// mean the device no longer has the object.
fn destroy_all(
    gpu: &mut dyn GpuDevice,
    owned: &[Owned],
    failed: &mut Vec<Owned>,
) -> Result<(), GpuError> {
    let mut result = Ok(());
    for &object in owned.iter().rev() {
        let destroyed = match object {
            Owned::Buffer(h) => gpu.destroy_buffer(h),
            Owned::Texture(h) => gpu.destroy_texture(h),
            Owned::Shader(h) => gpu.destroy_shader(h),
            Owned::Pipeline(h) => gpu.destroy_pipeline(h),
        };
        if let Err(GpuError::Backend(_)) = destroyed {
            failed.push(object);
        }
        if result.is_ok() {
            result = destroyed;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw_list::UV_FULL;
    use alloc::string::String;
    use core::f32::consts::FRAC_PI_2;
    use titan_core::BumpArena;
    use titan_gpu::{
        ClearColor, DrawCmd, FrameView, GpuEvent, RecordedFrame, RecordingGpu, RenderCmd,
        ResourceCounts, ShaderSource,
    };

    const CAMERA: Camera2D = Camera2D {
        position: Vec2::ZERO,
        viewport: Vec2::new(800.0, 600.0),
        zoom: 1.0,
        rotation: 0.0,
    };
    const PASS: RenderPassDesc = RenderPassDesc::clear_surface(ClearColor::BLACK);
    const RED: Color = Color::rgb(1.0, 0.0, 0.0);
    const BLUE: Color = Color::rgb(0.0, 0.0, 1.0);

    fn square(x: f32, y: f32) -> Rect {
        Rect::from_pos_size(Vec2::new(x, y), Vec2::ONE)
    }

    fn user_texture(gpu: &mut RecordingGpu) -> Handle<Texture> {
        gpu.create_texture(&TextureDesc::rgba8_2d(2, 2)).unwrap()
    }

    /// Flushes the queue into a new frame, submits it, and returns what the device recorded.
    fn flush_and_submit(gpu: &mut RecordingGpu, renderer: &mut Renderer2D) -> RecordedFrame {
        let mut frame = FrameCommands::new();
        renderer.flush(gpu, &mut frame, PASS, &CAMERA).unwrap();
        gpu.submit(frame.view()).unwrap();
        gpu.last_frame().unwrap().clone()
    }

    /// The `(texture, first_index, index_count)` of every draw in `commands`, in order.
    fn draws(commands: &[RenderCmd]) -> Vec<(Handle<Texture>, u32, u32)> {
        let mut texture = None;
        let mut out = Vec::new();
        for command in commands {
            match *command {
                RenderCmd::SetTexture {
                    binding: 0,
                    texture: t,
                } => texture = Some(t),
                RenderCmd::Draw(DrawCmd::Indexed {
                    index_count,
                    first_index,
                    base_vertex: 0,
                    instance_count: 1,
                    first_instance: 0,
                }) => out.push((texture.unwrap(), first_index, index_count)),
                RenderCmd::Draw(other) => panic!("unexpected draw {other:?}"),
                _ => {}
            }
        }
        out
    }

    fn floats(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
    }

    /// Reads back the first `quads * 4` vertices of the renderer's vertex buffer.
    fn vertices(gpu: &RecordingGpu, renderer: &Renderer2D, quads: usize) -> Vec<Vertex2D> {
        let buffer = renderer.resources.as_ref().unwrap().vertex_buffer;
        let data = &gpu.buffer(buffer).unwrap().data;
        let f: Vec<f32> = floats(&data[..quads * 4 * Vertex2D::SIZE_BYTES as usize]).collect();
        f.as_chunks::<8>()
            .0
            .iter()
            .map(|&[x, y, u, v, r, g, b, a]| Vertex2D {
                position: [x, y],
                uv: [u, v],
                color: [r, g, b, a],
            })
            .collect()
    }

    fn indices(gpu: &RecordingGpu, renderer: &Renderer2D) -> Vec<u32> {
        let buffer = renderer.resources.as_ref().unwrap().index_buffer;
        gpu.buffer(buffer)
            .unwrap()
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect()
    }

    fn expected_indices(quads: u32) -> Vec<u32> {
        (0..quads)
            .flat_map(|q| QUAD_INDICES.map(|i| q * 4 + i))
            .collect()
    }

    #[test]
    fn creates_the_pipeline_texture_and_buffers_it_needs() {
        let mut gpu = RecordingGpu::with_surface_format(TextureFormat::Rgba8UnormSrgb);
        let renderer = Renderer2D::new(&mut gpu).unwrap();
        let resources = renderer.resources.as_ref().unwrap();

        let pipeline = gpu.pipeline(renderer.pipeline().unwrap()).unwrap();
        assert_eq!(pipeline.color_format, Some(TextureFormat::Rgba8UnormSrgb));
        assert_eq!(pipeline.blend, Some(BlendState::ALPHA_BLENDING));
        assert_eq!(pipeline.depth_stencil, None);
        assert_eq!(pipeline.vertex_layouts.len(), 1);
        assert_eq!(pipeline.vertex_layouts[0].array_stride, 32);
        assert_eq!(pipeline.vertex_layouts[0].attributes, Vertex2D::ATTRIBUTES);
        let shader = gpu.shader(resources.shader).unwrap();
        assert_eq!(shader.wgsl, crate::SPRITE_SHADER_WGSL);
        assert_eq!(shader.msl, crate::SPRITE_SHADER_MSL);

        let white = gpu.texture(renderer.white_texture().unwrap()).unwrap();
        assert_eq!((white.width, white.height), (1, 1));
        assert_eq!(white.format, TextureFormat::Rgba8Unorm);
        assert_eq!(white.data, [255; 4]);

        assert_eq!(gpu.buffer(resources.uniform_buffer).unwrap().size, 64);
        assert_eq!(
            gpu.buffer(resources.vertex_buffer).unwrap().size,
            (INITIAL_QUAD_CAPACITY * 4 * 32) as u64
        );
        assert_eq!(
            indices(&gpu, &renderer),
            expected_indices(INITIAL_QUAD_CAPACITY as u32)
        );
        assert_eq!(
            gpu.live_counts(),
            ResourceCounts {
                buffers: 3,
                textures: 1,
                shaders: 1,
                pipelines: 1
            }
        );
    }

    #[test]
    fn untextured_and_white_textured_quads_share_one_draw() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let white = renderer.white_texture().unwrap();
        renderer.draw_rect(square(0.0, 0.0), RED);
        renderer.draw_sprite(Sprite::textured(square(1.0, 0.0), white));
        renderer.draw_quad(Vec2::new(2.0, 0.0), Vec2::ONE, BLUE);
        assert_eq!(renderer.quad_count(), 3);

        let frame = flush_and_submit(&mut gpu, &mut renderer);
        assert!(renderer.is_empty(), "a successful flush clears the queue");
        assert_eq!(frame.passes.len(), 1);
        let pass = &frame.passes[0];
        assert_eq!(pass.desc, PASS);
        let r = renderer.resources.as_ref().unwrap();
        assert_eq!(
            pass.commands[..4],
            [
                RenderCmd::SetPipeline(r.pipeline),
                RenderCmd::SetUniformBuffer {
                    binding: 0,
                    buffer: r.uniform_buffer,
                    offset: 0
                },
                RenderCmd::SetVertexBuffer {
                    slot: 0,
                    buffer: r.vertex_buffer,
                    offset: 0
                },
                RenderCmd::SetIndexBuffer {
                    buffer: r.index_buffer,
                    format: IndexFormat::Uint32,
                    offset: 0
                },
            ]
        );
        assert_eq!(draws(&pass.commands), [(white, 0, 18)]);
        assert_eq!(pass.commands.len(), 6);
    }

    #[test]
    fn alternating_textures_get_the_fewest_draws() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let white = renderer.white_texture().unwrap();
        let a = user_texture(&mut gpu);
        let b = user_texture(&mut gpu);
        for (x, texture) in [Some(a), Some(a), Some(b), None, None, Some(a), Some(b)]
            .into_iter()
            .enumerate()
        {
            let sprite = Sprite::solid(square(x as f32, 0.0), RED);
            renderer.draw_sprite(Sprite { texture, ..sprite });
        }
        let frame = flush_and_submit(&mut gpu, &mut renderer);
        assert_eq!(
            draws(&frame.passes[0].commands),
            [
                (a, 0, 12),
                (b, 12, 6),
                (white, 18, 12),
                (a, 30, 6),
                (b, 36, 6)
            ]
        );
    }

    #[test]
    fn layers_sort_ascending_and_keep_submission_order_within_a_layer() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let texture = user_texture(&mut gpu);
        let white = renderer.white_texture().unwrap();
        // (x, layer, textured): submission order.
        let sprites = [
            (0.0, 2, false),
            (1.0, -1, true),
            (2.0, 2, false),
            (3.0, 0, false),
            (4.0, -1, true),
            (5.0, 0, false),
        ];
        for (i, &(x, layer, textured)) in sprites.iter().enumerate() {
            let color = Color::rgba(i as f32 / 8.0, 0.5, 0.25, 1.0);
            let sprite = Sprite::solid(square(x, 0.0), color).with_layer(layer);
            renderer.draw_sprite(if textured {
                Sprite {
                    texture: Some(texture),
                    ..sprite
                }
            } else {
                sprite
            });
        }
        let frame = flush_and_submit(&mut gpu, &mut renderer);

        // Layer -1 (submitted 1, 4), then 0 (3, 5), then 2 (0, 2).
        let order = [1, 4, 3, 5, 0, 2];
        let drawn = vertices(&gpu, &renderer, order.len());
        for (quad, &submitted) in order.iter().enumerate() {
            let v = drawn[quad * 4];
            assert_eq!(v.position, [sprites[submitted].0, 0.0], "quad {quad}");
            assert_eq!(v.color[0], submitted as f32 / 8.0, "quad {quad}");
        }
        assert_eq!(
            draws(&frame.passes[0].commands),
            [(texture, 0, 12), (white, 12, 24)]
        );
    }

    #[test]
    fn empty_frame_uploads_nothing_but_still_clears() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        gpu.clear_history();

        let frame = flush_and_submit(&mut gpu, &mut renderer);
        assert_eq!(frame.passes.len(), 1);
        assert_eq!(frame.passes[0].desc, PASS);
        assert!(frame.passes[0].commands.is_empty());
        assert_eq!(gpu.events(), [GpuEvent::Submit { frame: 0 }]);

        let mut frame = FrameCommands::new();
        renderer
            .flush_into_pass(&mut gpu, &mut frame.begin_pass(PASS), &CAMERA)
            .unwrap();
        assert_eq!(frame.command_count(), 0);
        assert_eq!(gpu.events(), [GpuEvent::Submit { frame: 0 }]);
    }

    #[test]
    fn buffers_grow_across_frames_and_free_the_old_ones() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let live = gpu.live_counts();
        let old = renderer
            .resources
            .as_ref()
            .map(|r| (r.vertex_buffer, r.index_buffer));

        // Frame 1 fits the initial capacity.
        renderer.draw_rect(square(0.0, 0.0), RED);
        flush_and_submit(&mut gpu, &mut renderer);

        // Frame 2 needs 100 quads: the buffers grow to 128.
        let quads = 100;
        let uv = Rect::from_min_max(Vec2::new(0.25, 0.5), Vec2::new(0.75, 1.0));
        let rotated = Sprite::solid(
            Rect::from_min_max(Vec2::new(10.0, 20.0), Vec2::new(14.0, 22.0)),
            BLUE,
        )
        .with_uv(uv)
        .with_rotation(FRAC_PI_2);
        renderer.draw_sprite(rotated);
        for i in 1..quads {
            renderer.draw_rect(square(i as f32, -(i as f32)), RED);
        }
        let frame = flush_and_submit(&mut gpu, &mut renderer);

        let r = renderer.resources.as_ref().unwrap();
        assert_eq!(r.quad_capacity, 128);
        let (old_vertex, old_index) = old.unwrap();
        assert!(gpu.buffer(old_vertex).is_none() && gpu.buffer(old_index).is_none());
        assert_eq!(gpu.live_counts(), live, "the old buffers were freed");
        assert_eq!(gpu.buffer(r.vertex_buffer).unwrap().size, 128 * 4 * 32);
        assert_eq!(indices(&gpu, &renderer), expected_indices(128));
        let white = r.white_texture;
        assert_eq!(draws(&frame.passes[0].commands), [(white, 0, quads * 6)]);

        let drawn = vertices(&gpu, &renderer, quads as usize);
        // The 4×2 rect turned a quarter turn about its center (12, 21) is 2×4, with each corner
        // moved one place counter-clockwise.
        let expected_rotated = [[13.0, 19.0], [13.0, 23.0], [11.0, 23.0], [11.0, 19.0]];
        let expected_uvs = [[0.25, 1.0], [0.75, 1.0], [0.75, 0.5], [0.25, 0.5]];
        for ((v, position), uv) in drawn[..4].iter().zip(expected_rotated).zip(expected_uvs) {
            let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
            assert!(
                close(v.position[0], position[0]) && close(v.position[1], position[1]),
                "{v:?} != {position:?}"
            );
            assert_eq!(v.uv, uv);
            assert_eq!(v.color, BLUE.to_array());
        }
        let last = (quads - 1) as f32;
        assert_eq!(
            drawn[drawn.len() - 4..],
            [
                [last, -last],
                [last + 1.0, -last],
                [last + 1.0, 1.0 - last],
                [last, 1.0 - last]
            ]
            .into_iter()
            .zip([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
            .map(|(position, uv)| Vertex2D {
                position,
                uv,
                color: RED.to_array()
            })
            .collect::<Vec<_>>()
        );

        // Frame 3 is back under capacity: the grown buffers are reused.
        let vertex_buffer = r.vertex_buffer;
        renderer.draw_rect(square(0.0, 0.0), RED);
        flush_and_submit(&mut gpu, &mut renderer);
        assert_eq!(
            renderer.resources.as_ref().unwrap().vertex_buffer,
            vertex_buffer
        );
        renderer.destroy(&mut gpu).unwrap();
        assert_eq!(gpu.live_counts(), ResourceCounts::default());
    }

    #[test]
    fn uploads_the_camera_matrix() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let camera = Camera2D {
            position: Vec2::new(3.0, -4.0),
            zoom: 2.5,
            rotation: 0.3,
            ..CAMERA
        };
        renderer.draw_rect(square(0.0, 0.0), RED);
        let mut frame = FrameCommands::new();
        renderer.flush(&mut gpu, &mut frame, PASS, &camera).unwrap();
        let uniform = gpu
            .buffer(renderer.resources.as_ref().unwrap().uniform_buffer)
            .unwrap();
        let expected = camera.view_projection().cols;
        let expected = expected.iter().flat_map(|c| [c.x, c.y, c.z, c.w]);
        assert!(floats(&uniform.data).eq(expected));
    }

    #[test]
    fn flush_list_draws_a_caller_list_into_an_arena_frame() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        renderer.draw_rect(square(9.0, 9.0), BLUE);
        let mut list = DrawList2D::with_capacity(2);
        list.draw_rect(square(0.0, 0.0), RED);
        list.draw_sprite(Sprite::solid(square(1.0, 0.0), RED).with_color(BLUE));
        let before = list.clone();

        let arena = BumpArena::new(16 * 1024);
        let mut frame = FrameCommands::new_in(&arena);
        renderer
            .flush_list(&mut gpu, &mut frame, PASS, &CAMERA, &list)
            .unwrap();
        gpu.submit(frame.view()).unwrap();

        assert_eq!(
            list.sprites(),
            before.sprites(),
            "the caller's list is kept"
        );
        assert_eq!(renderer.quad_count(), 1, "the queue is untouched");
        let white = renderer.white_texture().unwrap();
        let pass = &gpu.last_frame().unwrap().passes[0];
        assert_eq!(draws(&pass.commands), [(white, 0, 12)]);
        let drawn = vertices(&gpu, &renderer, 2);
        assert_eq!(
            (drawn[0].color, drawn[4].color),
            (RED.to_array(), BLUE.to_array())
        );
    }

    #[test]
    fn flush_into_pass_appends_to_the_callers_pass() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let texture = user_texture(&mut gpu);
        renderer.draw_sprite(Sprite::textured(square(0.0, 0.0), texture).with_uv(UV_FULL));

        let mut frame = FrameCommands::new();
        {
            let mut pass = frame.begin_pass(PASS);
            pass.set_viewport(titan_gpu::Viewport::new(0.0, 0.0, 800.0, 600.0));
            renderer
                .flush_into_pass(&mut gpu, &mut pass, &CAMERA)
                .unwrap();
        }
        assert!(renderer.is_empty());
        gpu.submit(FrameView::from(&frame)).unwrap();
        let commands = &gpu.last_frame().unwrap().passes[0].commands;
        assert!(matches!(commands[0], RenderCmd::SetViewport(_)));
        assert_eq!(draws(commands), [(texture, 0, 6)]);
    }

    #[test]
    fn destroy_frees_everything_and_later_flushes_fail() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        renderer.destroy(&mut gpu).unwrap();
        assert_eq!(gpu.live_counts(), ResourceCounts::default());
        assert_eq!(
            (renderer.pipeline(), renderer.white_texture()),
            (None, None)
        );
        renderer.destroy(&mut gpu).unwrap();

        renderer.draw_rect(square(0.0, 0.0), RED);
        let destroyed = Err(GpuError::InvalidHandle {
            kind: ResourceKind::Pipeline,
        });
        let mut frame = FrameCommands::new();
        assert_eq!(
            renderer.flush(&mut gpu, &mut frame, PASS, &CAMERA),
            destroyed
        );
        assert!(frame.is_empty(), "a failed flush records no pass");
        assert_eq!(renderer.quad_count(), 1, "a failed flush keeps the queue");
        let empty = DrawList2D::new();
        assert_eq!(
            renderer.flush_list(&mut gpu, &mut frame, PASS, &CAMERA, &empty),
            destroyed
        );
        assert_eq!(
            renderer.flush_into_pass(&mut gpu, &mut frame.begin_pass(PASS), &CAMERA),
            destroyed
        );
    }

    /// A [`RecordingGpu`] that fails one call of a kind once that many calls of the kind have
    /// gone through, without forwarding it. `None` lets every call through.
    struct FailingGpu {
        inner: RecordingGpu,
        creates_before_failure: Option<usize>,
        buffer_writes_before_failure: Option<usize>,
        destroys_before_failure: Option<usize>,
    }

    impl FailingGpu {
        fn new() -> Self {
            Self {
                inner: RecordingGpu::new(),
                creates_before_failure: None,
                buffer_writes_before_failure: None,
                destroys_before_failure: None,
            }
        }
    }

    /// Counts a call down; fails it, once, when the count reaches zero.
    fn countdown(calls_before_failure: &mut Option<usize>) -> Result<(), GpuError> {
        match calls_before_failure {
            Some(0) => {
                *calls_before_failure = None;
                Err(GpuError::Backend(String::from("injected")))
            }
            Some(n) => {
                *n -= 1;
                Ok(())
            }
            None => Ok(()),
        }
    }

    impl GpuDevice for FailingGpu {
        fn surface_format(&self) -> TextureFormat {
            self.inner.surface_format()
        }
        fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<Handle<Buffer>, GpuError> {
            countdown(&mut self.creates_before_failure)?;
            self.inner.create_buffer(desc)
        }
        fn write_buffer(&mut self, b: Handle<Buffer>, o: u64, d: &[u8]) -> Result<(), GpuError> {
            countdown(&mut self.buffer_writes_before_failure)?;
            self.inner.write_buffer(b, o, d)
        }
        fn destroy_buffer(&mut self, buffer: Handle<Buffer>) -> Result<(), GpuError> {
            countdown(&mut self.destroys_before_failure)?;
            self.inner.destroy_buffer(buffer)
        }
        fn create_texture(&mut self, desc: &TextureDesc<'_>) -> Result<Handle<Texture>, GpuError> {
            countdown(&mut self.creates_before_failure)?;
            self.inner.create_texture(desc)
        }
        fn write_texture(&mut self, texture: Handle<Texture>, data: &[u8]) -> Result<(), GpuError> {
            self.inner.write_texture(texture, data)
        }
        fn destroy_texture(&mut self, texture: Handle<Texture>) -> Result<(), GpuError> {
            countdown(&mut self.destroys_before_failure)?;
            self.inner.destroy_texture(texture)
        }
        fn create_shader(&mut self, source: &ShaderSource<'_>) -> Result<Handle<Shader>, GpuError> {
            countdown(&mut self.creates_before_failure)?;
            self.inner.create_shader(source)
        }
        fn destroy_shader(&mut self, shader: Handle<Shader>) -> Result<(), GpuError> {
            countdown(&mut self.destroys_before_failure)?;
            self.inner.destroy_shader(shader)
        }
        fn create_pipeline(
            &mut self,
            desc: &PipelineDesc<'_>,
        ) -> Result<Handle<Pipeline>, GpuError> {
            countdown(&mut self.creates_before_failure)?;
            self.inner.create_pipeline(desc)
        }
        fn destroy_pipeline(&mut self, pipeline: Handle<Pipeline>) -> Result<(), GpuError> {
            countdown(&mut self.destroys_before_failure)?;
            self.inner.destroy_pipeline(pipeline)
        }
        fn submit(&mut self, commands: FrameView<'_>) -> Result<(), GpuError> {
            self.inner.submit(commands)
        }
    }

    #[test]
    fn failed_creation_leaves_nothing_allocated() {
        let injected = Err(GpuError::Backend(String::from("injected")));
        // Texture, shader, pipeline, then the uniform, vertex, and index buffers.
        for creates in 0..6 {
            let mut gpu = FailingGpu::new();
            gpu.creates_before_failure = Some(creates);
            assert_eq!(
                Renderer2D::new(&mut gpu).map(|_| ()),
                injected,
                "failing create {creates}"
            );
            assert_eq!(gpu.inner.live_counts(), ResourceCounts::default());
        }
    }

    #[test]
    fn failed_growth_keeps_the_old_buffers_and_records_nothing() {
        let mut gpu = FailingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let live = gpu.inner.live_counts();
        let r = renderer.resources.as_ref().unwrap();
        let old = (r.vertex_buffer, r.index_buffer);
        for quad in 0..=INITIAL_QUAD_CAPACITY {
            renderer.draw_rect(square(quad as f32, 0.0), RED);
        }
        // Growth creates the vertex then the index buffer, then writes the indices, the
        // camera, and the vertices.
        let failures = [
            (Some(0), None),
            (Some(1), None),
            (None, Some(0)),
            (None, Some(1)),
            (None, Some(2)),
        ];
        let mut one_quad = DrawList2D::new();
        one_quad.draw_rect(square(0.0, 0.0), BLUE);
        for (creates, writes) in failures {
            gpu.creates_before_failure = creates;
            gpu.buffer_writes_before_failure = writes;
            let mut frame = FrameCommands::new();
            assert!(renderer.flush(&mut gpu, &mut frame, PASS, &CAMERA).is_err());
            assert!(frame.is_empty());
            assert_eq!(gpu.inner.live_counts(), live, "{creates:?} {writes:?}");
            let r = renderer.resources.as_ref().unwrap();
            assert_eq!((r.vertex_buffer, r.index_buffer), old);
            assert_eq!(r.quad_capacity, INITIAL_QUAD_CAPACITY);
            assert_eq!(renderer.quad_count(), INITIAL_QUAD_CAPACITY + 1);

            // The old buffers still draw a frame that fits them.
            gpu.creates_before_failure = None;
            gpu.buffer_writes_before_failure = None;
            renderer
                .flush_list(&mut gpu, &mut frame, PASS, &CAMERA, &one_quad)
                .unwrap();
            gpu.submit(frame.view()).unwrap();
            assert_eq!(vertices(&gpu.inner, &renderer, 1)[0].color, BLUE.to_array());
            assert_eq!(
                indices(&gpu.inner, &renderer),
                expected_indices(INITIAL_QUAD_CAPACITY as u32)
            );
        }

        // With the device healthy again, the same queue grows the buffers and draws.
        let mut frame = FrameCommands::new();
        renderer.flush(&mut gpu, &mut frame, PASS, &CAMERA).unwrap();
        gpu.submit(frame.view()).unwrap();
        assert_eq!(renderer.resources.as_ref().unwrap().quad_capacity, 128);
        assert_eq!(gpu.inner.live_counts(), live);
        renderer.destroy(&mut gpu).unwrap();
        assert_eq!(gpu.inner.live_counts(), ResourceCounts::default());
    }

    #[test]
    fn destroy_retries_objects_a_transient_failure_kept() {
        for during_growth in [true, false] {
            let mut gpu = FailingGpu::new();
            let mut renderer = Renderer2D::new(&mut gpu).unwrap();
            gpu.destroys_before_failure = Some(0);
            let injected = Err(GpuError::Backend(String::from("injected")));
            if during_growth {
                for quad in 0..=INITIAL_QUAD_CAPACITY {
                    renderer.draw_rect(square(quad as f32, 0.0), RED);
                }
                let mut frame = FrameCommands::new();
                assert_eq!(
                    renderer.flush(&mut gpu, &mut frame, PASS, &CAMERA),
                    injected
                );
                assert!(frame.is_empty());
                assert_eq!(renderer.resources.as_ref().unwrap().quad_capacity, 128);
            } else {
                assert_eq!(renderer.destroy(&mut gpu), injected);
            }
            // One buffer the device failed to destroy stays live: an outgrown one during growth,
            // beside the three in use.
            let expected = if during_growth { 4 } else { 1 };
            assert_eq!(gpu.inner.live_counts().buffers, expected, "{during_growth}");

            renderer.destroy(&mut gpu).unwrap();
            assert_eq!(gpu.inner.live_counts(), ResourceCounts::default());
            renderer.destroy(&mut gpu).unwrap();
        }
    }
}
