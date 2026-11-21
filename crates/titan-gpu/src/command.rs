//! Frame command lists: render passes and the draw commands inside them.

use crate::resource::{Buffer, Pipeline, Texture};
use alloc::alloc::Global;
use alloc::vec::Vec;
use core::alloc::Allocator;
use core::ops::Range;
use titan_core::{Color, Handle};

/// The color a pass clears its color attachment to. Channels are `f64` because that is what
/// Metal and WebGPU take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor {
    /// Red.
    pub r: f64,
    /// Green.
    pub g: f64,
    /// Blue.
    pub b: f64,
    /// Alpha.
    pub a: f64,
}

impl ClearColor {
    /// Opaque black.
    pub const BLACK: Self = Self::rgba(0.0, 0.0, 0.0, 1.0);
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    /// Creates a clear color from all four channels.
    #[must_use]
    pub const fn rgba(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    /// Converts a [`Color`] exactly.
    #[must_use]
    pub const fn from_color(color: Color) -> Self {
        Self::rgba(
            color.r as f64,
            color.g as f64,
            color.b as f64,
            color.a as f64,
        )
    }
}

impl From<Color> for ClearColor {
    fn from(color: Color) -> Self {
        Self::from_color(color)
    }
}

/// What an attachment holds when its pass begins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Every texel is set to this value.
    Clear(T),
    /// The previous contents are kept.
    Load,
}

/// What happens to an attachment's contents when its pass ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// The rendered contents are kept.
    Store,
    /// The contents may be thrown away, which saves bandwidth on tiled GPUs.
    Discard,
}

/// A pass's color attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachment {
    /// The texture to render to, or `None` for the window's surface.
    pub target: Option<Handle<Texture>>,
    /// What the attachment holds at the start of the pass.
    pub load_op: LoadOp<ClearColor>,
    /// What happens to it at the end.
    pub store_op: StoreOp,
}

/// A pass's depth attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthAttachment {
    /// The depth texture to test and write.
    pub target: Handle<Texture>,
    /// What the depth holds at the start of the pass; clear values are in `[0, 1]`.
    pub depth_load_op: LoadOp<f32>,
    /// What happens to the depth at the end.
    pub depth_store_op: StoreOp,
}

/// The attachments of one render pass. A pass needs at least one attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPassDesc {
    /// A name for debuggers and error messages. `'static` keeps recorded frames self-contained.
    pub label: Option<&'static str>,
    /// The color attachment, if any.
    pub color: Option<ColorAttachment>,
    /// The depth attachment, if any.
    pub depth: Option<DepthAttachment>,
}

impl RenderPassDesc {
    /// A pass that clears the surface to `color` and draws over it.
    #[must_use]
    pub const fn clear_surface(color: ClearColor) -> Self {
        Self::surface(LoadOp::Clear(color))
    }

    /// A pass that draws over the surface's current contents.
    #[must_use]
    pub const fn load_surface() -> Self {
        Self::surface(LoadOp::Load)
    }

    const fn surface(load_op: LoadOp<ClearColor>) -> Self {
        Self {
            label: None,
            color: Some(ColorAttachment {
                target: None,
                load_op,
                store_op: StoreOp::Store,
            }),
            depth: None,
        }
    }
}

/// The region of the attachments that clip space maps to, in pixels from the top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width; must be positive.
    pub width: f32,
    /// Height; must be positive.
    pub height: f32,
    /// The depth that clip-space z = 0 maps to, in `[0, 1]`.
    pub min_depth: f32,
    /// The depth that clip-space z = 1 maps to, in `[min_depth, 1]`.
    pub max_depth: f32,
}

impl Viewport {
    /// Creates a viewport over the full `[0, 1]` depth range.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

/// The rectangle outside which fragments are discarded, in pixels from the top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// The type of the indices in an index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    /// `u16` indices.
    Uint16,
    /// `u32` indices.
    Uint32,
}

impl IndexFormat {
    /// Returns the size of one index in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> u32 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// A draw call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawCmd {
    /// Draws vertices in order.
    NonIndexed {
        /// The number of vertices.
        vertex_count: u32,
        /// The number of instances.
        instance_count: u32,
        /// The first vertex.
        first_vertex: u32,
        /// The first instance.
        first_instance: u32,
    },
    /// Draws the vertices named by the bound index buffer.
    Indexed {
        /// The number of indices.
        index_count: u32,
        /// The number of instances.
        instance_count: u32,
        /// The first index.
        first_index: u32,
        /// Added to each index before it fetches a vertex.
        base_vertex: i32,
        /// The first instance.
        first_instance: u32,
    },
}

/// One command inside a render pass. Bindings persist until the pass ends or are replaced.
///
/// Slots and bindings map to shader declarations by the
/// [binding convention](crate::ShaderSource#binding-convention).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderCmd {
    /// Selects the pipeline for later draws.
    SetPipeline(Handle<Pipeline>),
    /// Binds a vertex buffer to a slot of the pipeline's vertex layouts.
    SetVertexBuffer {
        /// The vertex layout slot, below [`MAX_VERTEX_BUFFERS`](crate::MAX_VERTEX_BUFFERS).
        slot: u32,
        /// A buffer with [`BufferUsage::VERTEX`](crate::BufferUsage::VERTEX).
        buffer: Handle<Buffer>,
        /// Byte offset of the first element.
        offset: u64,
    },
    /// Binds the index buffer for indexed draws.
    SetIndexBuffer {
        /// A buffer with [`BufferUsage::INDEX`](crate::BufferUsage::INDEX).
        buffer: Handle<Buffer>,
        /// The index type.
        format: IndexFormat,
        /// Byte offset of the first index; a multiple of the index size.
        offset: u64,
    },
    /// Binds a uniform buffer to a shader binding.
    SetUniformBuffer {
        /// The shader binding.
        binding: u32,
        /// A buffer with [`BufferUsage::UNIFORM`](crate::BufferUsage::UNIFORM).
        buffer: Handle<Buffer>,
        /// Byte offset of the uniform data.
        offset: u64,
    },
    /// Binds a storage buffer to a shader binding.
    SetStorageBuffer {
        /// The shader binding.
        binding: u32,
        /// A buffer with [`BufferUsage::STORAGE`](crate::BufferUsage::STORAGE).
        buffer: Handle<Buffer>,
        /// Byte offset of the storage data.
        offset: u64,
    },
    /// Binds a texture and its sampler to a shader binding. A pass cannot sample its own color
    /// or depth attachment.
    SetTexture {
        /// The shader binding.
        binding: u32,
        /// A texture with [`TextureUsage::SAMPLED`](crate::TextureUsage::SAMPLED).
        texture: Handle<Texture>,
    },
    /// Sets the viewport.
    SetViewport(Viewport),
    /// Sets the scissor rectangle, which must lie within the pass's attachments.
    SetScissor(ScissorRect),
    /// Draws with the current bindings.
    Draw(DrawCmd),
}

/// One recorded render pass: its attachments and its commands in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPass<'a> {
    /// The pass's attachments.
    pub desc: RenderPassDesc,
    /// The pass's commands.
    pub commands: &'a [RenderCmd],
}

/// A pass's attachments and where its commands start in the shared command list; it ends where
/// the next pass starts.
#[derive(Clone, Copy, Debug)]
struct PassSpan {
    desc: RenderPassDesc,
    start: usize,
}

/// A frame's render passes, recorded into two flat lists that are reused frame after frame.
///
/// Record each pass through the encoder that [`begin_pass`](Self::begin_pass) returns, then
/// hand [`view`](Self::view) to [`GpuDevice::submit`](crate::GpuDevice::submit). Any
/// [`Allocator`] works, so a frame can live in a [`BumpArena`](titan_core::BumpArena):
///
/// ```
/// use titan_core::BumpArena;
/// use titan_gpu::{ClearColor, FrameCommands, RenderPassDesc, Viewport};
///
/// let arena = BumpArena::new(64 * 1024);
/// let mut frame = FrameCommands::new_in(&arena);
/// frame
///     .begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
///     .set_viewport(Viewport::new(0.0, 0.0, 800.0, 600.0));
/// assert_eq!((frame.pass_count(), frame.command_count()), (1, 1));
/// ```
#[derive(Debug)]
pub struct FrameCommands<A: Allocator = Global> {
    passes: Vec<PassSpan, A>,
    commands: Vec<RenderCmd, A>,
}

impl FrameCommands {
    /// Creates an empty frame on the global heap without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            passes: Vec::new(),
            commands: Vec::new(),
        }
    }
}

impl Default for FrameCommands {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator> FrameCommands<A> {
    /// Creates an empty frame that allocates from `alloc`, such as `&BumpArena`.
    #[must_use]
    pub fn new_in(alloc: A) -> Self
    where
        A: Clone,
    {
        Self {
            passes: Vec::new_in(alloc.clone()),
            commands: Vec::new_in(alloc),
        }
    }

    /// Starts a render pass. Commands recorded through the returned encoder belong to this pass;
    /// the pass ends when the encoder is dropped or [`finish`](RenderPassEncoder::finish)ed.
    pub fn begin_pass(&mut self, desc: RenderPassDesc) -> RenderPassEncoder<'_, A> {
        self.passes.push(PassSpan {
            desc,
            start: self.commands.len(),
        });
        RenderPassEncoder {
            commands: &mut self.commands,
        }
    }

    /// Removes every pass and command, keeping capacity for the next frame.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.commands.clear();
    }

    /// Returns whether no pass has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Returns the number of passes.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Returns the number of commands across all passes.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Returns pass `index`, or `None` if out of range.
    #[must_use]
    pub fn pass(&self, index: usize) -> Option<RenderPass<'_>> {
        self.view().pass(index)
    }

    /// Iterates over the passes in recording order.
    pub fn passes(&self) -> impl ExactSizeIterator<Item = RenderPass<'_>> {
        self.view().passes()
    }

    /// Returns a borrowed, allocator-independent view of the frame for submission.
    #[must_use]
    pub fn view(&self) -> FrameView<'_> {
        FrameView {
            passes: &self.passes,
            commands: &self.commands,
        }
    }
}

/// Records commands into one pass of a [`FrameCommands`]. Methods return `&mut Self` so calls
/// chain.
#[derive(Debug)]
pub struct RenderPassEncoder<'a, A: Allocator = Global> {
    commands: &'a mut Vec<RenderCmd, A>,
}

impl<A: Allocator> RenderPassEncoder<'_, A> {
    /// Appends `command` to the pass.
    pub fn push(&mut self, command: RenderCmd) -> &mut Self {
        self.commands.push(command);
        self
    }

    /// Selects the pipeline for later draws.
    pub fn set_pipeline(&mut self, pipeline: Handle<Pipeline>) -> &mut Self {
        self.push(RenderCmd::SetPipeline(pipeline))
    }

    /// Binds a vertex buffer to vertex layout `slot`, starting `offset` bytes in.
    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Handle<Buffer>,
        offset: u64,
    ) -> &mut Self {
        self.push(RenderCmd::SetVertexBuffer {
            slot,
            buffer,
            offset,
        })
    }

    /// Binds the index buffer, starting `offset` bytes in.
    pub fn set_index_buffer(
        &mut self,
        buffer: Handle<Buffer>,
        format: IndexFormat,
        offset: u64,
    ) -> &mut Self {
        self.push(RenderCmd::SetIndexBuffer {
            buffer,
            format,
            offset,
        })
    }

    /// Binds a uniform buffer to shader `binding`, starting `offset` bytes in.
    pub fn set_uniform_buffer(
        &mut self,
        binding: u32,
        buffer: Handle<Buffer>,
        offset: u64,
    ) -> &mut Self {
        self.push(RenderCmd::SetUniformBuffer {
            binding,
            buffer,
            offset,
        })
    }

    /// Binds a storage buffer to shader `binding`, starting `offset` bytes in.
    pub fn set_storage_buffer(
        &mut self,
        binding: u32,
        buffer: Handle<Buffer>,
        offset: u64,
    ) -> &mut Self {
        self.push(RenderCmd::SetStorageBuffer {
            binding,
            buffer,
            offset,
        })
    }

    /// Binds a texture and its sampler to shader `binding`.
    pub fn set_texture(&mut self, binding: u32, texture: Handle<Texture>) -> &mut Self {
        self.push(RenderCmd::SetTexture { binding, texture })
    }

    /// Sets the viewport.
    pub fn set_viewport(&mut self, viewport: Viewport) -> &mut Self {
        self.push(RenderCmd::SetViewport(viewport))
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&mut self, rect: ScissorRect) -> &mut Self {
        self.push(RenderCmd::SetScissor(rect))
    }

    /// Draws `vertices` of each of `instances`.
    ///
    /// # Panics
    ///
    /// Panics if either range ends before it starts.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> &mut Self {
        self.push(RenderCmd::Draw(DrawCmd::NonIndexed {
            vertex_count: count(&vertices),
            instance_count: count(&instances),
            first_vertex: vertices.start,
            first_instance: instances.start,
        }))
    }

    /// Draws the vertices named by `indices` of the bound index buffer, offset by `base_vertex`,
    /// for each of `instances`.
    ///
    /// # Panics
    ///
    /// Panics if either range ends before it starts.
    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> &mut Self {
        self.push(RenderCmd::Draw(DrawCmd::Indexed {
            index_count: count(&indices),
            instance_count: count(&instances),
            first_index: indices.start,
            base_vertex,
            first_instance: instances.start,
        }))
    }

    /// Ends the pass. Dropping the encoder does the same; this only makes the end explicit.
    pub fn finish(self) {}
}

fn count(range: &Range<u32>) -> u32 {
    range
        .end
        .checked_sub(range.start)
        .expect("draw range ends before it starts")
}

/// A borrowed frame of render passes, as [`GpuDevice::submit`](crate::GpuDevice::submit) takes
/// it. It does not depend on the allocator the [`FrameCommands`] uses.
#[derive(Clone, Copy, Debug)]
pub struct FrameView<'a> {
    passes: &'a [PassSpan],
    commands: &'a [RenderCmd],
}

impl<'a> FrameView<'a> {
    /// Returns pass `index`, or `None` if out of range.
    #[must_use]
    pub fn pass(self, index: usize) -> Option<RenderPass<'a>> {
        let span = self.passes.get(index)?;
        let end = self
            .passes
            .get(index + 1)
            .map_or(self.commands.len(), |next| next.start);
        Some(RenderPass {
            desc: span.desc,
            commands: &self.commands[span.start..end],
        })
    }

    /// Iterates over the passes in recording order.
    pub fn passes(self) -> impl ExactSizeIterator<Item = RenderPass<'a>> {
        (0..self.passes.len())
            .map(move |index| self.pass(index).expect("index is below the pass count"))
    }

    /// Returns the number of passes.
    #[must_use]
    pub fn pass_count(self) -> usize {
        self.passes.len()
    }

    /// Returns the number of commands across all passes.
    #[must_use]
    pub fn command_count(self) -> usize {
        self.commands.len()
    }

    /// Returns whether the frame has no passes.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.passes.is_empty()
    }
}

impl<'a, A: Allocator> From<&'a FrameCommands<A>> for FrameView<'a> {
    fn from(frame: &'a FrameCommands<A>) -> Self {
        frame.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourcePool;
    use titan_core::BumpArena;

    struct Handles {
        pipeline: Handle<Pipeline>,
        vertices: Handle<Buffer>,
        indices: Handle<Buffer>,
        texture: Handle<Texture>,
    }

    fn handles() -> Handles {
        Handles {
            pipeline: ResourcePool::<Pipeline, ()>::new().insert(()),
            vertices: ResourcePool::<Buffer, ()>::new().insert(()),
            indices: ResourcePool::<Buffer, ()>::new().insert(()),
            texture: ResourcePool::<Texture, ()>::new().insert(()),
        }
    }

    /// Records a clear pass with an indexed draw, an empty load pass, and a plain draw pass.
    fn record<A: Allocator>(frame: &mut FrameCommands<A>, h: &Handles) {
        frame
            .begin_pass(RenderPassDesc::clear_surface(ClearColor::BLACK))
            .set_pipeline(h.pipeline)
            .set_vertex_buffer(0, h.vertices, 0)
            .set_index_buffer(h.indices, IndexFormat::Uint16, 0)
            .set_texture(0, h.texture)
            .draw_indexed(0..6, 0, 0..1);
        frame.begin_pass(RenderPassDesc::load_surface()).finish();
        let mut pass = frame.begin_pass(RenderPassDesc::load_surface());
        pass.set_pipeline(h.pipeline);
        pass.draw(3..9, 2..4);
    }

    fn assert_recorded<A: Allocator>(frame: &FrameCommands<A>, h: &Handles) {
        assert_eq!((frame.pass_count(), frame.command_count()), (3, 7));
        let passes: Vec<_> = frame.passes().collect();
        assert_eq!(passes.len(), 3);
        assert_eq!(
            passes[0].desc,
            RenderPassDesc::clear_surface(ClearColor::BLACK)
        );
        assert_eq!(passes[0].commands.len(), 5);
        assert_eq!(
            passes[0].commands[4],
            RenderCmd::Draw(DrawCmd::Indexed {
                index_count: 6,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            })
        );
        assert!(passes[1].commands.is_empty());
        assert_eq!(
            passes[2].commands,
            [
                RenderCmd::SetPipeline(h.pipeline),
                RenderCmd::Draw(DrawCmd::NonIndexed {
                    vertex_count: 6,
                    instance_count: 2,
                    first_vertex: 3,
                    first_instance: 2,
                }),
            ]
        );
        assert_eq!(frame.pass(2), Some(passes[2]));
        assert_eq!(frame.pass(3), None);
        let view = FrameView::from(frame);
        assert_eq!(view.passes().collect::<Vec<_>>(), passes);
    }

    #[test]
    fn records_passes_on_the_global_heap_and_reuses_them() {
        let h = handles();
        let mut frame = FrameCommands::new();
        assert!(frame.is_empty());
        record(&mut frame, &h);
        assert_recorded(&frame, &h);
        frame.clear();
        assert!(frame.is_empty() && frame.view().is_empty());
        assert_eq!(frame.command_count(), 0);
        assert_eq!(frame.pass(0), None);
        record(&mut frame, &h);
        assert_recorded(&frame, &h);
    }

    #[test]
    fn records_passes_in_a_bump_arena() {
        let h = handles();
        let mut arena = BumpArena::new(4096);
        {
            let mut frame = FrameCommands::new_in(&arena);
            record(&mut frame, &h);
            assert_recorded(&frame, &h);
            assert!(arena.used() > 0);
            frame.clear();
            record(&mut frame, &h);
            assert_recorded(&frame, &h);
        }
        arena.reset();
        let mut frame = FrameCommands::new_in(&arena);
        record(&mut frame, &h);
        assert_recorded(&frame, &h);
    }

    #[test]
    #[should_panic(expected = "ends before it starts")]
    fn inverted_draw_range_panics() {
        FrameCommands::new()
            .begin_pass(RenderPassDesc::load_surface())
            .draw(Range { start: 5, end: 2 }, 0..1);
    }
}
