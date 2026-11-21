//! [`MetalGpu`]: the [`GpuDevice`] backend on Metal.

use crate::objc::{
    Id, NO, NSInteger, NSSize, NSUInteger, ObjcBool, Retained, Sel, YES, autoreleasepool, class,
    msg_send, ns_error_description, ns_string, sel,
};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, null_mut};
use titan_gpu::{
    AddressMode, BlendFactor, BlendOperation, Buffer, BufferDesc, BufferUsage, CompareFunction,
    CullMode, DrawCmd, FilterMode, FrameView, FrontFace, GpuDevice, GpuError, Handle, IndexFormat,
    LoadOp, MAX_VERTEX_BUFFERS, METAL_BUFFER_BINDING_OFFSET, Pipeline, PipelineDesc,
    PrimitiveTopology, RecordingGpu, RenderCmd, RenderPass, RenderPassDesc, ResourceKind,
    ResourcePool, SamplerDesc, Shader, ShaderSource, StoreOp, Texture, TextureDesc, TextureFormat,
    TextureUsage, VertexFormat, VertexStepMode,
};

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

// Every value below is copied from the macOS SDK header named beside it.

// `MTLPixelFormat` (Metal/MTLPixelFormat.h).
const MTL_PIXEL_FORMAT_R8_UNORM: NSUInteger = 10;
const MTL_PIXEL_FORMAT_RGBA8_UNORM: NSUInteger = 70;
const MTL_PIXEL_FORMAT_RGBA8_UNORM_SRGB: NSUInteger = 71;
const MTL_PIXEL_FORMAT_BGRA8_UNORM: NSUInteger = 80;
const MTL_PIXEL_FORMAT_BGRA8_UNORM_SRGB: NSUInteger = 81;
const MTL_PIXEL_FORMAT_DEPTH32_FLOAT: NSUInteger = 252;
const MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8: NSUInteger = 260;

// `MTLStorageMode` and `MTLResourceStorageModeShift` (Metal/MTLResource.h).
const MTL_STORAGE_MODE_SHARED: NSUInteger = 0;
const MTL_STORAGE_MODE_MANAGED: NSUInteger = 1;
const MTL_STORAGE_MODE_PRIVATE: NSUInteger = 2;
const MTL_RESOURCE_STORAGE_MODE_SHIFT: NSUInteger = 4;

// `MTLTextureType` and `MTLTextureUsage` (Metal/MTLTexture.h).
const MTL_TEXTURE_TYPE_2D: NSUInteger = 2;
const MTL_TEXTURE_USAGE_SHADER_READ: NSUInteger = 0x1;
const MTL_TEXTURE_USAGE_SHADER_WRITE: NSUInteger = 0x2;
const MTL_TEXTURE_USAGE_RENDER_TARGET: NSUInteger = 0x4;

// `MTLVertexFormat` and `MTLVertexStepFunction` (Metal/MTLVertexDescriptor.h).
const MTL_VERTEX_FORMAT_UCHAR4_NORMALIZED: NSUInteger = 9;
const MTL_VERTEX_FORMAT_FLOAT: NSUInteger = 28;
const MTL_VERTEX_FORMAT_FLOAT2: NSUInteger = 29;
const MTL_VERTEX_FORMAT_FLOAT3: NSUInteger = 30;
const MTL_VERTEX_FORMAT_FLOAT4: NSUInteger = 31;
const MTL_VERTEX_FORMAT_INT: NSUInteger = 32;
const MTL_VERTEX_FORMAT_UINT: NSUInteger = 36;
const MTL_VERTEX_STEP_FUNCTION_PER_VERTEX: NSUInteger = 1;
const MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE: NSUInteger = 2;

// `MTLPrimitiveType`, `MTLCullMode`, and `MTLWinding` (Metal/MTLRenderCommandEncoder.h).
const MTL_PRIMITIVE_TYPE_POINT: NSUInteger = 0;
const MTL_PRIMITIVE_TYPE_LINE: NSUInteger = 1;
const MTL_PRIMITIVE_TYPE_LINE_STRIP: NSUInteger = 2;
const MTL_PRIMITIVE_TYPE_TRIANGLE: NSUInteger = 3;
const MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP: NSUInteger = 4;
const MTL_CULL_MODE_NONE: NSUInteger = 0;
const MTL_CULL_MODE_FRONT: NSUInteger = 1;
const MTL_CULL_MODE_BACK: NSUInteger = 2;
const MTL_WINDING_CLOCKWISE: NSUInteger = 0;
const MTL_WINDING_COUNTER_CLOCKWISE: NSUInteger = 1;

// `MTLIndexType` (Metal/MTLArgument.h).
const MTL_INDEX_TYPE_UINT16: NSUInteger = 0;
const MTL_INDEX_TYPE_UINT32: NSUInteger = 1;

// `MTLCompareFunction` (Metal/MTLDepthStencil.h).
const MTL_COMPARE_FUNCTION_NEVER: NSUInteger = 0;
const MTL_COMPARE_FUNCTION_LESS: NSUInteger = 1;
const MTL_COMPARE_FUNCTION_EQUAL: NSUInteger = 2;
const MTL_COMPARE_FUNCTION_LESS_EQUAL: NSUInteger = 3;
const MTL_COMPARE_FUNCTION_GREATER: NSUInteger = 4;
const MTL_COMPARE_FUNCTION_NOT_EQUAL: NSUInteger = 5;
const MTL_COMPARE_FUNCTION_GREATER_EQUAL: NSUInteger = 6;
const MTL_COMPARE_FUNCTION_ALWAYS: NSUInteger = 7;

// `MTLBlendFactor` and `MTLBlendOperation` (Metal/MTLRenderPipeline.h).
const MTL_BLEND_FACTOR_ZERO: NSUInteger = 0;
const MTL_BLEND_FACTOR_ONE: NSUInteger = 1;
const MTL_BLEND_FACTOR_SOURCE_COLOR: NSUInteger = 2;
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR: NSUInteger = 3;
const MTL_BLEND_FACTOR_SOURCE_ALPHA: NSUInteger = 4;
const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA: NSUInteger = 5;
const MTL_BLEND_FACTOR_DESTINATION_COLOR: NSUInteger = 6;
const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR: NSUInteger = 7;
const MTL_BLEND_FACTOR_DESTINATION_ALPHA: NSUInteger = 8;
const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA: NSUInteger = 9;
const MTL_BLEND_OPERATION_ADD: NSUInteger = 0;
const MTL_BLEND_OPERATION_SUBTRACT: NSUInteger = 1;
const MTL_BLEND_OPERATION_REVERSE_SUBTRACT: NSUInteger = 2;
const MTL_BLEND_OPERATION_MIN: NSUInteger = 3;
const MTL_BLEND_OPERATION_MAX: NSUInteger = 4;

// `MTLLoadAction` and `MTLStoreAction` (Metal/MTLRenderPass.h).
const MTL_LOAD_ACTION_LOAD: NSUInteger = 1;
const MTL_LOAD_ACTION_CLEAR: NSUInteger = 2;
const MTL_STORE_ACTION_DONT_CARE: NSUInteger = 0;
const MTL_STORE_ACTION_STORE: NSUInteger = 1;

// `MTLSamplerMinMagFilter` and `MTLSamplerAddressMode` (Metal/MTLSampler.h).
const MTL_SAMPLER_MIN_MAG_FILTER_NEAREST: NSUInteger = 0;
const MTL_SAMPLER_MIN_MAG_FILTER_LINEAR: NSUInteger = 1;
const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: NSUInteger = 0;
const MTL_SAMPLER_ADDRESS_MODE_REPEAT: NSUInteger = 2;
const MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT: NSUInteger = 3;

// `MTLCommandBufferStatus` (Metal/MTLCommandBuffer.h).
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: NSUInteger = 4;
const MTL_COMMAND_BUFFER_STATUS_ERROR: NSUInteger = 5;

// `MTLGPUFamily` (Metal/MTLDevice.h).
const MTL_GPU_FAMILY_APPLE1: NSInteger = 1001;

/// `MTLClearColor` (Metal/MTLRenderPass.h).
#[repr(C)]
struct MTLClearColor {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

/// `MTLViewport` (Metal/MTLRenderCommandEncoder.h).
#[repr(C)]
struct MTLViewport {
    origin_x: f64,
    origin_y: f64,
    width: f64,
    height: f64,
    znear: f64,
    zfar: f64,
}

/// `MTLScissorRect` (Metal/MTLRenderCommandEncoder.h).
#[repr(C)]
struct MTLScissorRect {
    x: NSUInteger,
    y: NSUInteger,
    width: NSUInteger,
    height: NSUInteger,
}

/// `MTLRegion`, an `MTLOrigin` followed by an `MTLSize` (Metal/MTLTypes.h).
#[repr(C)]
struct MTLRegion {
    x: NSUInteger,
    y: NSUInteger,
    z: NSUInteger,
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,
}

impl MTLRegion {
    fn whole_2d(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            z: 0,
            width: width as NSUInteger,
            height: height as NSUInteger,
            depth: 1,
        }
    }
}

/// The most command buffers the CPU runs ahead of the GPU; encoding another one first waits for
/// the oldest.
const MAX_FRAMES_IN_FLIGHT: usize = 3;
/// Metal's buffer argument table has 31 entries per stage, `[[buffer(0)]]` to `[[buffer(30)]]`.
const METAL_BUFFER_SLOTS: u32 = 31;
/// Metal's sampler table has 16 entries per stage, which bounds texture bindings too.
const METAL_SAMPLER_SLOTS: u32 = 16;
/// `MTLVertexDescriptor` has 31 attributes, `[[attribute(0)]]` to `[[attribute(30)]]`.
const METAL_VERTEX_ATTRIBUTES: u32 = 31;
/// The window surface's format: `CAMetalLayer` is set to `MTLPixelFormatBGRA8Unorm_sRGB`.
const SURFACE_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;

#[derive(Debug)]
struct MetalBuffer {
    raw: Retained,
    size: u64,
    usage: BufferUsage,
    /// The newest submission that reads the buffer; CPU writes wait for it.
    last_used_submission: u64,
}

#[derive(Debug)]
struct MetalTexture {
    raw: Retained,
    sampler: Retained,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: TextureUsage,
    /// Whether the texture uses `MTLStorageModeManaged`, whose GPU writes need a blit
    /// `synchronizeResource:` before the CPU reads them.
    managed: bool,
    /// The newest submission that reads or renders to the texture; CPU access waits for it.
    last_used_submission: u64,
}

#[derive(Debug)]
struct MetalShader {
    vertex: Retained,
    fragment: Retained,
}

/// What a draw needs to know about one of its pipeline's vertex layouts.
#[derive(Debug)]
struct LayoutBounds {
    array_stride: u32,
    step_mode: VertexStepMode,
    /// Where the last element's last attribute ends: a draw's final element need not hold a
    /// full stride.
    last_element_bytes: u64,
}

#[derive(Debug)]
struct MetalPipeline {
    state: Retained,
    depth_stencil_state: Option<Retained>,
    color_format: Option<TextureFormat>,
    depth_format: Option<TextureFormat>,
    vertex_layouts: Vec<LayoutBounds>,
    primitive_type: NSUInteger,
    cull_mode: NSUInteger,
    winding: NSUInteger,
}

/// The window's `CAMetalLayer` and its drawable size in pixels.
#[derive(Debug)]
struct Surface {
    layer: Retained,
    width: u32,
    height: u32,
}

/// A [`GpuDevice`] that renders with Metal, to a window's `CAMetalLayer` or offscreen.
///
/// It validates every call exactly as [`RecordingGpu`] does, then checks Metal's own binding
/// limits, so a frame that fails validation runs no pass at all. Shaders are compiled from
/// [`ShaderSource::msl`] and bound by the
/// [binding convention](titan_gpu::ShaderSource#binding-convention).
///
/// Up to three frames run on the GPU while the CPU records the next. A CPU write to a buffer or
/// texture that an unfinished frame uses first waits for that frame, and destroying a resource
/// invalidates its handle at once while Metal keeps the object alive until frames using it
/// finish.
#[derive(Debug)]
pub struct MetalGpu {
    device: Retained,
    queue: Retained,
    /// Whether the GPU is in an Apple family. Only then can color textures use
    /// `MTLStorageModeShared` (others use `MTLStorageModeManaged`), and uniform buffer offsets
    /// need not be multiples of 256.
    apple_gpu: bool,
    surface: Option<Surface>,
    buffers: ResourcePool<Buffer, MetalBuffer>,
    textures: ResourcePool<Texture, MetalTexture>,
    shaders: ResourcePool<Shader, MetalShader>,
    pipelines: ResourcePool<Pipeline, MetalPipeline>,
    /// The number the next committed command buffer gets; numbers start at 1.
    next_submission: u64,
    /// Every submission up to this number has finished on the GPU.
    completed_submission: u64,
    /// Committed command buffers that may still run, oldest first.
    in_flight: VecDeque<(u64, Retained)>,
}

impl MetalGpu {
    /// Creates a device on the system's default GPU with no window surface, for offscreen
    /// rendering. Passes must give their color attachment a texture target.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] if the system has no Metal device.
    pub fn new_headless() -> Result<Self, GpuError> {
        autoreleasepool(|| {
            // SAFETY: `MTLCreateSystemDefaultDevice` returns a +1 device or nil.
            let device = unsafe { Retained::from_retained(MTLCreateSystemDefaultDevice()) }
                .ok_or_else(|| backend("this Mac has no Metal device"))?;
            // SAFETY: `device` is a live `MTLDevice`; `newCommandQueue` returns +1 or nil,
            // `respondsToSelector:` takes a `SEL`, and `supportsFamily:` (sent only when it
            // exists) takes an `MTLGPUFamily` (`NSInteger`). Both return `BOOL`.
            let (queue, apple_gpu) = unsafe {
                let queue = Retained::from_retained(msg_send![Id, &device, c"newCommandQueue"]);
                // `supportsFamily:` is new in macOS 10.15; older Macs have only Intel and AMD GPUs.
                let has_supports_family: ObjcBool = msg_send![
                    ObjcBool,
                    &device,
                    c"respondsToSelector:",
                    sel!(c"supportsFamily:") => Sel,
                ];
                let apple = has_supports_family != NO && {
                    let apple: ObjcBool = msg_send![
                        ObjcBool,
                        &device,
                        c"supportsFamily:",
                        MTL_GPU_FAMILY_APPLE1 => NSInteger,
                    ];
                    apple != NO
                };
                (queue, apple)
            };
            Ok(Self {
                queue: queue.ok_or_else(|| backend("Metal could not create a command queue"))?,
                device,
                apple_gpu,
                surface: None,
                buffers: ResourcePool::new(),
                textures: ResourcePool::new(),
                shaders: ResourcePool::new(),
                pipelines: ResourcePool::new(),
                next_submission: 1,
                completed_submission: 0,
                in_flight: VecDeque::new(),
            })
        })
    }

    /// Creates a device that presents to `layer`, a `CAMetalLayer`, whose drawable is `width` x
    /// `height` pixels.
    pub(crate) fn with_layer(layer: Retained, width: u32, height: u32) -> Result<Self, GpuError> {
        let mut gpu = Self::new_headless()?;
        // SAFETY: `layer` is a live `CAMetalLayer`; these are its property setters with their
        // declared types (`id<MTLDevice>`, `MTLPixelFormat`, `BOOL`).
        unsafe {
            msg_send![(), &layer, c"setDevice:", gpu.device.as_ptr() => Id];
            msg_send![
                (),
                &layer,
                c"setPixelFormat:",
                MTL_PIXEL_FORMAT_BGRA8_UNORM_SRGB => NSUInteger,
            ];
            msg_send![(), &layer, c"setFramebufferOnly:", YES => ObjcBool];
        }
        gpu.surface = Some(Surface {
            layer,
            width: 0,
            height: 0,
        });
        gpu.resize(width, height);
        Ok(gpu)
    }

    /// Sets the surface's size in pixels. A zero size keeps the old drawables, and passes that
    /// draw to the surface fail validation until it is nonzero again.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let surface = self
            .surface
            .as_mut()
            .expect("only a windowed MetalGpu is resized");
        surface.width = width;
        surface.height = height;
        if width > 0 && height > 0 {
            let size = NSSize {
                width: f64::from(width),
                height: f64::from(height),
            };
            // SAFETY: `drawableSize` is a `CGSize` property of the live layer.
            unsafe { msg_send![(), &surface.layer, c"setDrawableSize:", size => NSSize] };
        }
    }

    /// [`GpuDevice::submit`], also returning whether the frame presented a drawable. Only a
    /// presented frame waits for the display, so the runner paces the others itself.
    pub(crate) fn submit_frame(&mut self, commands: FrameView<'_>) -> Result<bool, GpuError> {
        for pass in commands.passes() {
            self.check_pass(pass)?;
        }
        if commands.is_empty() {
            return Ok(false);
        }
        autoreleasepool(|| self.encode_and_commit(commands))
    }

    /// Waits for every submitted frame, then returns mip level 0 of `texture`: tightly packed
    /// rows in the texture's format, top row first.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidHandle`] for a destroyed texture,
    /// [`GpuError::MissingUsage`] without [`TextureUsage::COPY_SRC`],
    /// [`GpuError::InvalidDescriptor`] for a depth texture, whose layout the CPU cannot read,
    /// and [`GpuError::Backend`] if a frame failed on the GPU.
    pub fn read_texture(&mut self, texture: Handle<Texture>) -> Result<Vec<u8>, GpuError> {
        let tex = self.texture_with(texture, TextureUsage::COPY_SRC, "COPY_SRC")?;
        ensure(
            !tex.format.is_depth(),
            "a depth texture cannot be read back",
        )?;
        let bytes_per_pixel = tex
            .format
            .bytes_per_pixel()
            .expect("color formats have a size");
        let (width, height, managed) = (tex.width, tex.height, tex.managed);
        autoreleasepool(|| {
            if managed {
                let raw = self
                    .textures
                    .get(texture)
                    .expect("checked above")
                    .raw
                    .clone();
                // SAFETY: the queue is live; the command buffer (retained here) and blit encoder
                // are autoreleased and kept by this pool, and checked for nil before use;
                // `synchronizeResource:` takes the managed texture.
                let command_buffer = unsafe {
                    let command_buffer =
                        Retained::retain(msg_send![Id, &self.queue, c"commandBuffer"])
                            .ok_or_else(|| backend("Metal returned no command buffer"))?;
                    let blit: Id = msg_send![Id, &command_buffer, c"blitCommandEncoder"];
                    if blit.is_null() {
                        return Err(backend("Metal returned no blit command encoder"));
                    }
                    msg_send![(), blit, c"synchronizeResource:", raw.as_ptr() => Id];
                    msg_send![(), blit, c"endEncoding"];
                    msg_send![(), &command_buffer, c"commit"];
                    command_buffer
                };
                let submission = self.next_submission;
                self.next_submission += 1;
                self.in_flight.push_back((submission, command_buffer));
            }
            self.wait_idle()
        })?;
        let bytes_per_row = width as usize * bytes_per_pixel as usize;
        let mut pixels = vec![0; bytes_per_row * height as usize];
        let raw = &self.textures.get(texture).expect("checked above").raw;
        // SAFETY: the texture is a live 2D color texture of `width` x `height` with a CPU-
        // readable storage mode, every frame that wrote it has finished, and `pixels` holds
        // exactly `bytes_per_row * height` bytes.
        unsafe {
            msg_send![
                (),
                raw,
                c"getBytes:bytesPerRow:fromRegion:mipmapLevel:",
                pixels.as_mut_ptr().cast::<c_void>() => *mut c_void,
                bytes_per_row => NSUInteger,
                MTLRegion::whole_2d(width, height) => MTLRegion,
                0 => NSUInteger,
            ];
        }
        Ok(pixels)
    }

    /// Returns a live buffer that has `usage`, whose name is `flag`.
    fn buffer_with(
        &self,
        handle: Handle<Buffer>,
        usage: BufferUsage,
        flag: &'static str,
    ) -> Result<&MetalBuffer, GpuError> {
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
    ) -> Result<&MetalTexture, GpuError> {
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

    /// Checks one pass by `RecordingGpu`'s rules, plus the surface's known size and Metal's
    /// binding limits.
    fn check_pass(&self, pass: RenderPass<'_>) -> Result<(), GpuError> {
        let RenderPassDesc { color, depth, .. } = pass.desc;
        ensure(
            color.is_some() || depth.is_some(),
            "a render pass needs a color or depth attachment",
        )?;
        let mut color_format = None;
        let mut size = None;
        if let Some(color) = color {
            color_format = Some(match color.target {
                None => {
                    let surface = self.surface.as_ref().ok_or(GpuError::InvalidDescriptor {
                        message: "a headless MetalGpu has no surface; give the color \
                                  attachment a target",
                    })?;
                    ensure(
                        surface.width > 0 && surface.height > 0,
                        "the surface has zero size; skip frames until it is resized",
                    )?;
                    size = Some((surface.width, surface.height));
                    SURFACE_FORMAT
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

        let mut pipeline = None;
        let mut vertex_buffers = [None; MAX_VERTEX_BUFFERS];
        let mut index_buffer = None;
        for &command in pass.commands {
            match command {
                RenderCmd::SetPipeline(handle) => {
                    let p = self.pipelines.get(handle).ok_or(GpuError::InvalidHandle {
                        kind: ResourceKind::Pipeline,
                    })?;
                    ensure(
                        p.color_format == color_format && p.depth_format == depth_format,
                        "a pipeline's color_format and depth_stencil format must match its \
                         pass's attachments",
                    )?;
                    pipeline = Some(p);
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
                RenderCmd::SetUniformBuffer {
                    binding,
                    buffer,
                    offset,
                } => {
                    self.check_binding(buffer, offset, BufferUsage::UNIFORM, "UNIFORM")?;
                    check_buffer_binding(binding)?;
                    // On Apple GPUs enforce Metal's 4-byte minimum; elsewhere enforce the
                    // backend's conservative 256-byte non-Apple alignment. A stricter per-type
                    // alignment (16 for `float4`) depends on the shader's argument type, which
                    // this command does not carry.
                    ensure(
                        offset.is_multiple_of(if self.apple_gpu { 4 } else { 256 }),
                        "Metal uniform buffer offsets must be multiples of 256 (4 on Apple GPUs)",
                    )?;
                }
                RenderCmd::SetStorageBuffer {
                    binding,
                    buffer,
                    offset,
                } => {
                    self.check_binding(buffer, offset, BufferUsage::STORAGE, "STORAGE")?;
                    check_buffer_binding(binding)?;
                    ensure(
                        offset.is_multiple_of(4),
                        "Metal storage buffer offsets must be multiples of 4",
                    )?;
                }
                RenderCmd::SetTexture { binding, texture } => {
                    self.texture_with(texture, TextureUsage::SAMPLED, "SAMPLED")?;
                    ensure(
                        !attachments.contains(&Some(texture)),
                        "a pass cannot sample its own color or depth attachment",
                    )?;
                    ensure(
                        binding < METAL_SAMPLER_SLOTS,
                        "Metal texture bindings must be below 16",
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

    /// Retires command buffers that have already finished, without blocking.
    fn drain_completed(&mut self) -> Result<(), GpuError> {
        while let Some((_, command_buffer)) = self.in_flight.front() {
            // SAFETY: `status` is a readonly `MTLCommandBufferStatus` property of a live
            // command buffer.
            let status: NSUInteger = unsafe { msg_send![NSUInteger, command_buffer, c"status"] };
            if status < MTL_COMMAND_BUFFER_STATUS_COMPLETED {
                break;
            }
            let (submission, command_buffer) = self.in_flight.pop_front().expect("front exists");
            self.completed_submission = submission;
            check_completed(&command_buffer, status)?;
        }
        Ok(())
    }

    /// Blocks until every submission up to `submission` has finished on the GPU.
    fn wait_for(&mut self, submission: u64) -> Result<(), GpuError> {
        while self
            .in_flight
            .front()
            .is_some_and(|&(id, _)| id <= submission)
        {
            let (id, command_buffer) = self.in_flight.pop_front().expect("front exists");
            // SAFETY: `waitUntilCompleted` blocks until the committed command buffer finishes;
            // `status` is then readable.
            let status = unsafe {
                msg_send![(), &command_buffer, c"waitUntilCompleted"];
                msg_send![NSUInteger, &command_buffer, c"status"]
            };
            self.completed_submission = id;
            check_completed(&command_buffer, status)?;
        }
        // Submissions up to here that were never committed have nothing left to wait for.
        self.completed_submission = self.completed_submission.max(submission);
        Ok(())
    }

    /// Blocks until every submitted frame has finished on the GPU.
    pub(crate) fn wait_idle(&mut self) -> Result<(), GpuError> {
        self.wait_for(self.next_submission - 1)
    }

    /// Encodes and commits a validated, nonempty frame. Returns whether it presents a drawable.
    fn encode_and_commit(&mut self, frame: FrameView<'_>) -> Result<bool, GpuError> {
        self.drain_completed()?;
        while self.in_flight.len() >= MAX_FRAMES_IN_FLIGHT {
            let oldest = self.in_flight[0].0;
            self.wait_for(oldest)?;
        }
        let uses_surface = frame
            .passes()
            .any(|pass| pass.desc.color.is_some_and(|color| color.target.is_none()));
        let drawable: Id = if uses_surface {
            let layer = &self.surface.as_ref().expect("validated").layer;
            // SAFETY: `nextDrawable` returns an autoreleased drawable, kept alive by the
            // caller's pool until after the commit retains it, or nil on timeout.
            let drawable: Id = unsafe { msg_send![Id, layer, c"nextDrawable"] };
            if drawable.is_null() {
                return Err(backend("CAMetalLayer returned no drawable"));
            }
            drawable
        } else {
            null_mut()
        };
        // SAFETY: `drawable` is nil or a live `CAMetalDrawable`; `texture` is its readonly
        // `id<MTLTexture>`. `commandBuffer` returns an autoreleased command buffer, retained
        // here.
        let (surface_texture, command_buffer) = unsafe {
            let surface_texture: Id = if drawable.is_null() {
                null_mut()
            } else {
                msg_send![Id, drawable, c"texture"]
            };
            let command_buffer = Retained::retain(msg_send![Id, &self.queue, c"commandBuffer"])
                .ok_or_else(|| backend("Metal returned no command buffer"))?;
            (surface_texture, command_buffer)
        };
        for pass in frame.passes() {
            self.encode_pass(&command_buffer, pass, surface_texture)?;
        }
        let submission = self.next_submission;
        self.next_submission += 1;
        self.mark_used(frame, submission);
        // SAFETY: the command buffer is live and every encoder on it has ended;
        // `presentDrawable:` takes the drawable acquired above.
        unsafe {
            if !drawable.is_null() {
                msg_send![(), &command_buffer, c"presentDrawable:", drawable => Id];
            }
            msg_send![(), &command_buffer, c"commit"];
        }
        self.in_flight.push_back((submission, command_buffer));
        Ok(uses_surface)
    }

    /// Encodes one validated pass. `surface_texture` is the drawable's texture, or nil when no
    /// pass draws to the surface.
    fn encode_pass(
        &self,
        command_buffer: &Retained,
        pass: RenderPass<'_>,
        surface_texture: Id,
    ) -> Result<(), GpuError> {
        let texture = |handle| &self.textures.get(handle).expect("validated").raw;
        let buffer = |handle| self.buffers.get(handle).expect("validated").raw.as_ptr();
        // SAFETY: every object is live (validated handles, the drawable texture, autoreleased
        // descriptors and encoders kept by the caller's pool), every selector is sent with its
        // declared C types, and validation guarantees the arguments are in Metal's ranges:
        // binding indices within the argument tables, scissors within the attachments, and
        // draws within their bound buffers. Zero-count draws, which Metal rejects, are skipped.
        unsafe {
            let descriptor: Id = msg_send![
                Id,
                class!(c"MTLRenderPassDescriptor"),
                c"renderPassDescriptor"
            ];
            if let Some(color) = pass.desc.color {
                let target = color
                    .target
                    .map_or(surface_texture, |t| texture(t).as_ptr());
                let attachments: Id = msg_send![Id, descriptor, c"colorAttachments"];
                let attachment: Id = msg_send![
                    Id,
                    attachments,
                    c"objectAtIndexedSubscript:",
                    0 => NSUInteger,
                ];
                msg_send![(), attachment, c"setTexture:", target => Id];
                let load_action = match color.load_op {
                    LoadOp::Clear(c) => {
                        let clear = MTLClearColor {
                            red: c.r,
                            green: c.g,
                            blue: c.b,
                            alpha: c.a,
                        };
                        msg_send![(), attachment, c"setClearColor:", clear => MTLClearColor];
                        MTL_LOAD_ACTION_CLEAR
                    }
                    LoadOp::Load => MTL_LOAD_ACTION_LOAD,
                };
                msg_send![(), attachment, c"setLoadAction:", load_action => NSUInteger];
                msg_send![
                    (),
                    attachment,
                    c"setStoreAction:",
                    store_action(color.store_op) => NSUInteger,
                ];
            }
            if let Some(depth) = pass.desc.depth {
                let target = self.textures.get(depth.target).expect("validated");
                let attachment: Id = msg_send![Id, descriptor, c"depthAttachment"];
                msg_send![(), attachment, c"setTexture:", target.raw.as_ptr() => Id];
                let load_action = match depth.depth_load_op {
                    LoadOp::Clear(value) => {
                        msg_send![(), attachment, c"setClearDepth:", f64::from(value) => f64];
                        MTL_LOAD_ACTION_CLEAR
                    }
                    LoadOp::Load => MTL_LOAD_ACTION_LOAD,
                };
                msg_send![(), attachment, c"setLoadAction:", load_action => NSUInteger];
                msg_send![
                    (),
                    attachment,
                    c"setStoreAction:",
                    store_action(depth.depth_store_op) => NSUInteger,
                ];
                if target.format == TextureFormat::Depth24PlusStencil8 {
                    // Titan has no stencil API, but a depth-stencil texture must be bound as
                    // both. Its stencil starts cleared and is never kept.
                    let stencil: Id = msg_send![Id, descriptor, c"stencilAttachment"];
                    msg_send![(), stencil, c"setTexture:", target.raw.as_ptr() => Id];
                    msg_send![
                        (),
                        stencil,
                        c"setLoadAction:",
                        MTL_LOAD_ACTION_CLEAR => NSUInteger,
                    ];
                    msg_send![
                        (),
                        stencil,
                        c"setStoreAction:",
                        MTL_STORE_ACTION_DONT_CARE => NSUInteger,
                    ];
                }
            }
            let encoder: Id = msg_send![
                Id,
                command_buffer,
                c"renderCommandEncoderWithDescriptor:",
                descriptor => Id,
            ];
            if encoder.is_null() {
                return Err(backend("Metal returned no render command encoder"));
            }
            if let Some(label) = pass.desc.label {
                msg_send![(), encoder, c"setLabel:", ns_string(label).as_ptr() => Id];
            }
            let mut primitive_type = MTL_PRIMITIVE_TYPE_TRIANGLE;
            // The bound index buffer, its `MTLIndexType`, its offset, and its index size.
            let mut index_buffer: Option<(Id, NSUInteger, u64, u64)> = None;
            for &command in pass.commands {
                match command {
                    RenderCmd::SetPipeline(handle) => {
                        let p = self.pipelines.get(handle).expect("validated");
                        msg_send![
                            (),
                            encoder,
                            c"setRenderPipelineState:",
                            p.state.as_ptr() => Id,
                        ];
                        if let Some(state) = &p.depth_stencil_state {
                            msg_send![(), encoder, c"setDepthStencilState:", state.as_ptr() => Id];
                        }
                        msg_send![(), encoder, c"setCullMode:", p.cull_mode => NSUInteger];
                        msg_send![(), encoder, c"setFrontFacingWinding:", p.winding => NSUInteger];
                        primitive_type = p.primitive_type;
                    }
                    RenderCmd::SetVertexBuffer {
                        slot,
                        buffer: handle,
                        offset,
                    } => msg_send![
                        (),
                        encoder,
                        c"setVertexBuffer:offset:atIndex:",
                        buffer(handle) => Id,
                        offset as NSUInteger => NSUInteger,
                        slot as NSUInteger => NSUInteger,
                    ],
                    RenderCmd::SetIndexBuffer {
                        buffer: handle,
                        format,
                        offset,
                    } => {
                        let index_type = match format {
                            IndexFormat::Uint16 => MTL_INDEX_TYPE_UINT16,
                            IndexFormat::Uint32 => MTL_INDEX_TYPE_UINT32,
                        };
                        index_buffer = Some((
                            buffer(handle),
                            index_type,
                            offset,
                            u64::from(format.size_bytes()),
                        ));
                    }
                    RenderCmd::SetUniformBuffer {
                        binding,
                        buffer: handle,
                        offset,
                    }
                    | RenderCmd::SetStorageBuffer {
                        binding,
                        buffer: handle,
                        offset,
                    } => {
                        let index = (METAL_BUFFER_BINDING_OFFSET + binding) as NSUInteger;
                        msg_send![
                            (),
                            encoder,
                            c"setVertexBuffer:offset:atIndex:",
                            buffer(handle) => Id,
                            offset as NSUInteger => NSUInteger,
                            index => NSUInteger,
                        ];
                        msg_send![
                            (),
                            encoder,
                            c"setFragmentBuffer:offset:atIndex:",
                            buffer(handle) => Id,
                            offset as NSUInteger => NSUInteger,
                            index => NSUInteger,
                        ];
                    }
                    RenderCmd::SetTexture {
                        binding,
                        texture: handle,
                    } => {
                        let t = self.textures.get(handle).expect("validated");
                        msg_send![
                            (),
                            encoder,
                            c"setFragmentTexture:atIndex:",
                            t.raw.as_ptr() => Id,
                            binding as NSUInteger => NSUInteger,
                        ];
                        msg_send![
                            (),
                            encoder,
                            c"setFragmentSamplerState:atIndex:",
                            t.sampler.as_ptr() => Id,
                            binding as NSUInteger => NSUInteger,
                        ];
                    }
                    RenderCmd::SetViewport(v) => {
                        let viewport = MTLViewport {
                            origin_x: f64::from(v.x),
                            origin_y: f64::from(v.y),
                            width: f64::from(v.width),
                            height: f64::from(v.height),
                            znear: f64::from(v.min_depth),
                            zfar: f64::from(v.max_depth),
                        };
                        msg_send![(), encoder, c"setViewport:", viewport => MTLViewport];
                    }
                    RenderCmd::SetScissor(r) => {
                        let rect = MTLScissorRect {
                            x: r.x as NSUInteger,
                            y: r.y as NSUInteger,
                            width: r.width as NSUInteger,
                            height: r.height as NSUInteger,
                        };
                        msg_send![(), encoder, c"setScissorRect:", rect => MTLScissorRect];
                    }
                    RenderCmd::Draw(DrawCmd::NonIndexed {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    }) => {
                        if vertex_count == 0 || instance_count == 0 {
                            continue;
                        }
                        msg_send![
                            (),
                            encoder,
                            c"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                            primitive_type => NSUInteger,
                            first_vertex as NSUInteger => NSUInteger,
                            vertex_count as NSUInteger => NSUInteger,
                            instance_count as NSUInteger => NSUInteger,
                            first_instance as NSUInteger => NSUInteger,
                        ];
                    }
                    RenderCmd::Draw(DrawCmd::Indexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    }) => {
                        if index_count == 0 || instance_count == 0 {
                            continue;
                        }
                        let (raw, index_type, offset, index_size) =
                            index_buffer.expect("validated");
                        let index_offset = offset + u64::from(first_index) * index_size;
                        msg_send![
                            (),
                            encoder,
                            c"drawIndexedPrimitives:indexCount:indexType:indexBuffer:\
                              indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                            primitive_type => NSUInteger,
                            index_count as NSUInteger => NSUInteger,
                            index_type => NSUInteger,
                            raw => Id,
                            index_offset as NSUInteger => NSUInteger,
                            instance_count as NSUInteger => NSUInteger,
                            base_vertex as NSInteger => NSInteger,
                            first_instance as NSUInteger => NSUInteger,
                        ];
                    }
                }
            }
            msg_send![(), encoder, c"endEncoding"];
        }
        Ok(())
    }

    /// Records that `submission` uses every buffer and texture `frame` names, so CPU writes to
    /// them wait for it.
    fn mark_used(&mut self, frame: FrameView<'_>, submission: u64) {
        for pass in frame.passes() {
            let attachments = [
                pass.desc.color.and_then(|color| color.target),
                pass.desc.depth.map(|depth| depth.target),
            ];
            for handle in attachments.into_iter().flatten() {
                self.textures
                    .get_mut(handle)
                    .expect("validated")
                    .last_used_submission = submission;
            }
            for command in pass.commands {
                match *command {
                    RenderCmd::SetVertexBuffer { buffer, .. }
                    | RenderCmd::SetIndexBuffer { buffer, .. }
                    | RenderCmd::SetUniformBuffer { buffer, .. }
                    | RenderCmd::SetStorageBuffer { buffer, .. } => {
                        self.buffers
                            .get_mut(buffer)
                            .expect("validated")
                            .last_used_submission = submission;
                    }
                    RenderCmd::SetTexture { texture, .. } => {
                        self.textures
                            .get_mut(texture)
                            .expect("validated")
                            .last_used_submission = submission;
                    }
                    RenderCmd::SetPipeline(_)
                    | RenderCmd::SetViewport(_)
                    | RenderCmd::SetScissor(_)
                    | RenderCmd::Draw(_) => {}
                }
            }
        }
    }

    /// Creates the Metal objects for a pipeline whose descriptor passed validation.
    fn new_pipeline(
        &self,
        desc: &PipelineDesc<'_>,
        shader: &MetalShader,
    ) -> Result<MetalPipeline, GpuError> {
        // SAFETY: every object is live (the shader's functions, the +1 descriptors owned by
        // `Retained`, and autoreleased sub-descriptors kept by the caller's pool), and every
        // setter is sent with its declared C type. Vertex attribute indices are below 31 and
        // layout indices below 8, as validated.
        unsafe {
            let descriptor = Retained::from_retained(msg_send![
                Id,
                msg_send![Id, class!(c"MTLRenderPipelineDescriptor"), c"alloc"],
                c"init"
            ])
            .expect("MTLRenderPipelineDescriptor init failed");
            if let Some(label) = desc.label {
                msg_send![(), &descriptor, c"setLabel:", ns_string(label).as_ptr() => Id];
            }
            msg_send![(), &descriptor, c"setVertexFunction:", shader.vertex.as_ptr() => Id];
            msg_send![(), &descriptor, c"setFragmentFunction:", shader.fragment.as_ptr() => Id];
            if let Some(format) = desc.color_format {
                let attachments: Id = msg_send![Id, &descriptor, c"colorAttachments"];
                let attachment: Id = msg_send![
                    Id,
                    attachments,
                    c"objectAtIndexedSubscript:",
                    0 => NSUInteger,
                ];
                msg_send![(), attachment, c"setPixelFormat:", pixel_format(format) => NSUInteger];
                if let Some(blend) = desc.blend {
                    msg_send![(), attachment, c"setBlendingEnabled:", YES => ObjcBool];
                    let a = attachment;
                    let (color, alpha) = (blend.color, blend.alpha);
                    let src_rgb = blend_factor(color.src_factor);
                    let dst_rgb = blend_factor(color.dst_factor);
                    let src_alpha = blend_factor(alpha.src_factor);
                    let dst_alpha = blend_factor(alpha.dst_factor);
                    let rgb_op = blend_operation(color.operation);
                    let alpha_op = blend_operation(alpha.operation);
                    msg_send![(), a, c"setSourceRGBBlendFactor:", src_rgb => NSUInteger];
                    msg_send![(), a, c"setDestinationRGBBlendFactor:", dst_rgb => NSUInteger];
                    msg_send![(), a, c"setRgbBlendOperation:", rgb_op => NSUInteger];
                    msg_send![(), a, c"setSourceAlphaBlendFactor:", src_alpha => NSUInteger];
                    msg_send![(), a, c"setDestinationAlphaBlendFactor:", dst_alpha => NSUInteger];
                    msg_send![(), a, c"setAlphaBlendOperation:", alpha_op => NSUInteger];
                }
            }
            if let Some(depth) = desc.depth_stencil {
                let format = pixel_format(depth.format);
                msg_send![
                    (),
                    &descriptor,
                    c"setDepthAttachmentPixelFormat:",
                    format => NSUInteger,
                ];
                if depth.format == TextureFormat::Depth24PlusStencil8 {
                    msg_send![
                        (),
                        &descriptor,
                        c"setStencilAttachmentPixelFormat:",
                        format => NSUInteger,
                    ];
                }
            }
            if !desc.vertex_layouts.is_empty() {
                let vertex: Id = msg_send![Id, class!(c"MTLVertexDescriptor"), c"vertexDescriptor"];
                let layouts: Id = msg_send![Id, vertex, c"layouts"];
                let attributes: Id = msg_send![Id, vertex, c"attributes"];
                for (slot, layout) in desc.vertex_layouts.iter().enumerate() {
                    // Metal needs a layout only for slots that feed attributes; an empty one
                    // with a zero stride would be rejected.
                    if layout.attributes.is_empty() {
                        continue;
                    }
                    let l: Id = msg_send![
                        Id,
                        layouts,
                        c"objectAtIndexedSubscript:",
                        slot => NSUInteger,
                    ];
                    let step_function = match layout.step_mode {
                        VertexStepMode::Vertex => MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
                        VertexStepMode::Instance => MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
                    };
                    msg_send![
                        (),
                        l,
                        c"setStride:",
                        layout.array_stride as NSUInteger => NSUInteger,
                    ];
                    msg_send![(), l, c"setStepFunction:", step_function => NSUInteger];
                    msg_send![(), l, c"setStepRate:", 1 => NSUInteger];
                    for attribute in layout.attributes {
                        let a: Id = msg_send![
                            Id,
                            attributes,
                            c"objectAtIndexedSubscript:",
                            attribute.shader_location as NSUInteger => NSUInteger,
                        ];
                        msg_send![
                            (),
                            a,
                            c"setFormat:",
                            vertex_format(attribute.format) => NSUInteger,
                        ];
                        msg_send![
                            (),
                            a,
                            c"setOffset:",
                            attribute.offset as NSUInteger => NSUInteger,
                        ];
                        msg_send![(), a, c"setBufferIndex:", slot => NSUInteger];
                    }
                }
                msg_send![(), &descriptor, c"setVertexDescriptor:", vertex => Id];
            }
            let mut error: Id = null_mut();
            let state = Retained::from_retained(msg_send![
                Id,
                &self.device,
                c"newRenderPipelineStateWithDescriptor:error:",
                descriptor.as_ptr() => Id,
                &raw mut error => *mut Id,
            ])
            .ok_or_else(|| {
                backend(format!(
                    "Metal rejected the pipeline: {}",
                    ns_error_description(error)
                ))
            })?;
            let depth_stencil_state = match desc.depth_stencil {
                None => None,
                Some(depth) => {
                    let ds = Retained::from_retained(msg_send![
                        Id,
                        msg_send![Id, class!(c"MTLDepthStencilDescriptor"), c"alloc"],
                        c"init"
                    ])
                    .expect("MTLDepthStencilDescriptor init failed");
                    msg_send![
                        (),
                        &ds,
                        c"setDepthCompareFunction:",
                        compare_function(depth.depth_compare) => NSUInteger,
                    ];
                    let write = if depth.depth_write_enabled { YES } else { NO };
                    msg_send![(), &ds, c"setDepthWriteEnabled:", write => ObjcBool];
                    Some(
                        Retained::from_retained(msg_send![
                            Id,
                            &self.device,
                            c"newDepthStencilStateWithDescriptor:",
                            ds.as_ptr() => Id,
                        ])
                        .ok_or_else(|| backend("Metal could not create a depth-stencil state"))?,
                    )
                }
            };
            Ok(MetalPipeline {
                state,
                depth_stencil_state,
                color_format: desc.color_format,
                depth_format: desc.depth_stencil.map(|ds| ds.format),
                vertex_layouts: desc
                    .vertex_layouts
                    .iter()
                    .map(|layout| LayoutBounds {
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
                primitive_type: match desc.topology {
                    PrimitiveTopology::TriangleList => MTL_PRIMITIVE_TYPE_TRIANGLE,
                    PrimitiveTopology::TriangleStrip => MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP,
                    PrimitiveTopology::LineList => MTL_PRIMITIVE_TYPE_LINE,
                    PrimitiveTopology::LineStrip => MTL_PRIMITIVE_TYPE_LINE_STRIP,
                    PrimitiveTopology::PointList => MTL_PRIMITIVE_TYPE_POINT,
                },
                cull_mode: match desc.cull_mode {
                    CullMode::None => MTL_CULL_MODE_NONE,
                    CullMode::Front => MTL_CULL_MODE_FRONT,
                    CullMode::Back => MTL_CULL_MODE_BACK,
                },
                winding: match desc.front_face {
                    FrontFace::Ccw => MTL_WINDING_COUNTER_CLOCKWISE,
                    FrontFace::Cw => MTL_WINDING_CLOCKWISE,
                },
            })
        }
    }
}

/// Checks that a uniform or storage `binding` lands inside Metal's buffer argument table.
fn check_buffer_binding(binding: u32) -> Result<(), GpuError> {
    ensure(
        binding < METAL_BUFFER_SLOTS - METAL_BUFFER_BINDING_OFFSET,
        "Metal uniform and storage bindings must be below 23 (8 + binding < 31)",
    )
}

/// Checks that `draw` has the buffers `pipeline` reads and stays inside them, as
/// `RecordingGpu` does. `vertex_buffers` and `index_buffer` hold the bytes bound from each
/// buffer's offset on.
fn check_draw(
    pipeline: &MetalPipeline,
    vertex_buffers: &[Option<u64>; MAX_VERTEX_BUFFERS],
    index_buffer: Option<(IndexFormat, u64)>,
    draw: DrawCmd,
) -> Result<(), GpuError> {
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

/// Returns an [`InvalidDescriptor`](GpuError::InvalidDescriptor) error stating `rule` unless
/// `ok`.
fn ensure(ok: bool, rule: &'static str) -> Result<(), GpuError> {
    if ok {
        Ok(())
    } else {
        Err(GpuError::InvalidDescriptor { message: rule })
    }
}

fn backend(message: impl Into<String>) -> GpuError {
    GpuError::Backend(message.into())
}

/// Turns a finished command buffer's error status into a [`GpuError::Backend`].
fn check_completed(command_buffer: &Retained, status: NSUInteger) -> Result<(), GpuError> {
    if status != MTL_COMMAND_BUFFER_STATUS_ERROR {
        return Ok(());
    }
    // SAFETY: `error` is a readonly `NSError` property of a live command buffer, nil or live.
    let description = unsafe { ns_error_description(msg_send![Id, command_buffer, c"error"]) };
    Err(backend(format!("a frame failed on the GPU: {description}")))
}

fn pixel_format(format: TextureFormat) -> NSUInteger {
    match format {
        TextureFormat::Rgba8Unorm => MTL_PIXEL_FORMAT_RGBA8_UNORM,
        TextureFormat::Rgba8UnormSrgb => MTL_PIXEL_FORMAT_RGBA8_UNORM_SRGB,
        TextureFormat::Bgra8Unorm => MTL_PIXEL_FORMAT_BGRA8_UNORM,
        TextureFormat::Bgra8UnormSrgb => MTL_PIXEL_FORMAT_BGRA8_UNORM_SRGB,
        TextureFormat::R8Unorm => MTL_PIXEL_FORMAT_R8_UNORM,
        TextureFormat::Depth32Float => MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
        // Every Mac GPU supports Depth32Float_Stencil8; Depth24Unorm_Stencil8 is Intel-only
        // and deprecated.
        TextureFormat::Depth24PlusStencil8 => MTL_PIXEL_FORMAT_DEPTH32_FLOAT_STENCIL8,
    }
}

fn vertex_format(format: VertexFormat) -> NSUInteger {
    match format {
        VertexFormat::Float32 => MTL_VERTEX_FORMAT_FLOAT,
        VertexFormat::Float32x2 => MTL_VERTEX_FORMAT_FLOAT2,
        VertexFormat::Float32x3 => MTL_VERTEX_FORMAT_FLOAT3,
        VertexFormat::Float32x4 => MTL_VERTEX_FORMAT_FLOAT4,
        VertexFormat::Uint32 => MTL_VERTEX_FORMAT_UINT,
        VertexFormat::Sint32 => MTL_VERTEX_FORMAT_INT,
        VertexFormat::Unorm8x4 => MTL_VERTEX_FORMAT_UCHAR4_NORMALIZED,
    }
}

fn compare_function(function: CompareFunction) -> NSUInteger {
    match function {
        CompareFunction::Never => MTL_COMPARE_FUNCTION_NEVER,
        CompareFunction::Less => MTL_COMPARE_FUNCTION_LESS,
        CompareFunction::LessEqual => MTL_COMPARE_FUNCTION_LESS_EQUAL,
        CompareFunction::Equal => MTL_COMPARE_FUNCTION_EQUAL,
        CompareFunction::GreaterEqual => MTL_COMPARE_FUNCTION_GREATER_EQUAL,
        CompareFunction::Greater => MTL_COMPARE_FUNCTION_GREATER,
        CompareFunction::NotEqual => MTL_COMPARE_FUNCTION_NOT_EQUAL,
        CompareFunction::Always => MTL_COMPARE_FUNCTION_ALWAYS,
    }
}

fn blend_factor(factor: BlendFactor) -> NSUInteger {
    match factor {
        BlendFactor::Zero => MTL_BLEND_FACTOR_ZERO,
        BlendFactor::One => MTL_BLEND_FACTOR_ONE,
        BlendFactor::SrcAlpha => MTL_BLEND_FACTOR_SOURCE_ALPHA,
        BlendFactor::OneMinusSrcAlpha => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        BlendFactor::DstAlpha => MTL_BLEND_FACTOR_DESTINATION_ALPHA,
        BlendFactor::OneMinusDstAlpha => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA,
        BlendFactor::SrcColor => MTL_BLEND_FACTOR_SOURCE_COLOR,
        BlendFactor::OneMinusSrcColor => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR,
        BlendFactor::DstColor => MTL_BLEND_FACTOR_DESTINATION_COLOR,
        BlendFactor::OneMinusDstColor => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR,
    }
}

fn blend_operation(operation: BlendOperation) -> NSUInteger {
    match operation {
        BlendOperation::Add => MTL_BLEND_OPERATION_ADD,
        BlendOperation::Subtract => MTL_BLEND_OPERATION_SUBTRACT,
        BlendOperation::ReverseSubtract => MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
        BlendOperation::Min => MTL_BLEND_OPERATION_MIN,
        BlendOperation::Max => MTL_BLEND_OPERATION_MAX,
    }
}

fn store_action(op: StoreOp) -> NSUInteger {
    match op {
        StoreOp::Store => MTL_STORE_ACTION_STORE,
        StoreOp::Discard => MTL_STORE_ACTION_DONT_CARE,
    }
}

fn min_mag_filter(filter: FilterMode) -> NSUInteger {
    match filter {
        FilterMode::Nearest => MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        FilterMode::Linear => MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
    }
}

fn address_mode(mode: AddressMode) -> NSUInteger {
    match mode {
        AddressMode::ClampToEdge => MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        AddressMode::Repeat => MTL_SAMPLER_ADDRESS_MODE_REPEAT,
        AddressMode::MirrorRepeat => MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
    }
}

fn texture_usage(usage: TextureUsage) -> NSUInteger {
    let mut bits = 0;
    if usage.contains(TextureUsage::SAMPLED) {
        bits |= MTL_TEXTURE_USAGE_SHADER_READ;
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        bits |= MTL_TEXTURE_USAGE_RENDER_TARGET;
    }
    if usage.contains(TextureUsage::STORAGE) {
        bits |= MTL_TEXTURE_USAGE_SHADER_READ | MTL_TEXTURE_USAGE_SHADER_WRITE;
    }
    bits
}

/// Creates a sampler state for `desc`.
///
/// # Safety
///
/// `device` is a live `MTLDevice`.
unsafe fn new_sampler(device: &Retained, desc: SamplerDesc) -> Result<Retained, GpuError> {
    // SAFETY: `new` returns a +1 sampler descriptor; its setters take `NSUInteger` enums; the
    // device (caller contract) returns a +1 sampler state or nil.
    unsafe {
        let descriptor =
            Retained::from_retained(msg_send![Id, class!(c"MTLSamplerDescriptor"), c"new"])
                .expect("MTLSamplerDescriptor new failed");
        let d = &descriptor;
        msg_send![(), d, c"setMinFilter:", min_mag_filter(desc.min_filter) => NSUInteger];
        msg_send![(), d, c"setMagFilter:", min_mag_filter(desc.mag_filter) => NSUInteger];
        msg_send![(), d, c"setSAddressMode:", address_mode(desc.address_u) => NSUInteger];
        msg_send![(), d, c"setTAddressMode:", address_mode(desc.address_v) => NSUInteger];
        Retained::from_retained(msg_send![
            Id,
            device,
            c"newSamplerStateWithDescriptor:",
            d.as_ptr() => Id,
        ])
        .ok_or_else(|| backend("Metal could not create a sampler state"))
    }
}

/// Sets an object's `label` for GPU debuggers.
///
/// # Safety
///
/// `object` is a live Metal object with a `label` property.
unsafe fn set_label(object: &Retained, label: Option<&str>) {
    if let Some(label) = label {
        // SAFETY: `label` is an `NSString` property of a live Metal object (caller contract).
        unsafe { msg_send![(), object, c"setLabel:", ns_string(label).as_ptr() => Id] };
    }
}

impl GpuDevice for MetalGpu {
    fn surface_format(&self) -> TextureFormat {
        SURFACE_FORMAT
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<Handle<Buffer>, GpuError> {
        ensure(desc.size > 0, "buffer size must be nonzero")?;
        ensure(
            desc.size <= RecordingGpu::MAX_BUFFER_SIZE,
            "buffer size exceeds MAX_BUFFER_SIZE",
        )?;
        ensure(!desc.usage.is_empty(), "buffer usage must not be empty")?;
        let raw = autoreleasepool(|| {
            // SAFETY: the device is live; `newBufferWithLength:options:` returns a +1 buffer,
            // zero-filled, or nil. The size fits `NSUInteger` (at most MAX_BUFFER_SIZE).
            let raw = unsafe {
                Retained::from_retained(msg_send![
                    Id,
                    &self.device,
                    c"newBufferWithLength:options:",
                    desc.size as NSUInteger => NSUInteger,
                    MTL_STORAGE_MODE_SHARED << MTL_RESOURCE_STORAGE_MODE_SHIFT => NSUInteger,
                ])
            }
            .ok_or_else(|| backend("Metal could not allocate the buffer"))?;
            // SAFETY: `raw` is a live `MTLBuffer`.
            unsafe { set_label(&raw, desc.label) };
            Ok(raw)
        })?;
        Ok(self.buffers.insert(MetalBuffer {
            raw,
            size: desc.size,
            usage: desc.usage,
            last_used_submission: 0,
        }))
    }

    fn write_buffer(
        &mut self,
        buffer: Handle<Buffer>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GpuError> {
        let target = self.buffer_with(buffer, BufferUsage::COPY_DST, "COPY_DST")?;
        let size = target.size;
        let write_len = data.len() as u64;
        offset
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
        let last_used = target.last_used_submission;
        if last_used > self.completed_submission {
            self.wait_for(last_used)?;
        }
        let raw = &self.buffers.get(buffer).expect("checked above").raw;
        // SAFETY: `contents` is the CPU address of a live shared-storage buffer of `size`
        // bytes; `offset + data.len() <= size` was checked, and no unfinished frame uses the
        // buffer, so nothing reads the bytes while they are copied.
        unsafe {
            let contents: *mut u8 = msg_send![*mut u8, raw, c"contents"];
            ptr::copy_nonoverlapping(data.as_ptr(), contents.add(offset as usize), data.len());
        }
        Ok(())
    }

    fn destroy_buffer(&mut self, buffer: Handle<Buffer>) -> Result<(), GpuError> {
        // In-flight command buffers retain the Metal buffer until they finish.
        self.buffers.remove(buffer).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Buffer,
        })?;
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
        ensure(
            !(desc.usage.contains(TextureUsage::STORAGE) && desc.format.is_depth()),
            "a depth texture cannot have STORAGE",
        )?;
        let storage_mode = if desc.format.is_depth() {
            MTL_STORAGE_MODE_PRIVATE
        } else if self.apple_gpu {
            MTL_STORAGE_MODE_SHARED
        } else {
            MTL_STORAGE_MODE_MANAGED
        };
        let (raw, sampler) = autoreleasepool(|| {
            // SAFETY: `new` returns a +1 texture descriptor whose setters take `NSUInteger`
            // values; the device returns a +1 texture or nil. Width, height, and mip count are
            // in Metal's ranges (validated above), and the storage mode suits the format.
            unsafe {
                let descriptor =
                    Retained::from_retained(msg_send![Id, class!(c"MTLTextureDescriptor"), c"new"])
                        .expect("MTLTextureDescriptor new failed");
                let d = &descriptor;
                msg_send![(), d, c"setTextureType:", MTL_TEXTURE_TYPE_2D => NSUInteger];
                msg_send![(), d, c"setPixelFormat:", pixel_format(desc.format) => NSUInteger];
                msg_send![(), d, c"setWidth:", desc.width as NSUInteger => NSUInteger];
                msg_send![(), d, c"setHeight:", desc.height as NSUInteger => NSUInteger];
                msg_send![
                    (),
                    d,
                    c"setMipmapLevelCount:",
                    desc.mip_level_count as NSUInteger => NSUInteger,
                ];
                msg_send![(), d, c"setUsage:", texture_usage(desc.usage) => NSUInteger];
                msg_send![(), d, c"setStorageMode:", storage_mode => NSUInteger];
                let raw = Retained::from_retained(msg_send![
                    Id,
                    &self.device,
                    c"newTextureWithDescriptor:",
                    d.as_ptr() => Id,
                ])
                .ok_or_else(|| backend("Metal could not allocate the texture"))?;
                set_label(&raw, desc.label);
                Ok((raw, new_sampler(&self.device, desc.sampler)?))
            }
        })?;
        Ok(self.textures.insert(MetalTexture {
            raw,
            sampler,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            usage: desc.usage,
            managed: storage_mode == MTL_STORAGE_MODE_MANAGED,
            last_used_submission: 0,
        }))
    }

    fn write_texture(&mut self, texture: Handle<Texture>, data: &[u8]) -> Result<(), GpuError> {
        let target = self.texture_with(texture, TextureUsage::COPY_DST, "COPY_DST")?;
        let bytes_per_pixel = target
            .format
            .bytes_per_pixel()
            .expect("create_texture rejects COPY_DST on depth formats; color formats have a size");
        // Cannot overflow: at most 8192 * 8192 * 4 bytes.
        let bytes_per_row = target.width as usize * bytes_per_pixel as usize;
        let expected = bytes_per_row * target.height as usize;
        if data.len() != expected {
            return Err(GpuError::TextureDataSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let (width, height, last_used) = (target.width, target.height, target.last_used_submission);
        if last_used > self.completed_submission {
            self.wait_for(last_used)?;
        }
        let raw = &self.textures.get(texture).expect("checked above").raw;
        // SAFETY: the texture is a live, CPU-writable (shared or managed) 2D color texture of
        // `width` x `height`; `data` holds exactly `bytes_per_row * height` bytes, and no
        // unfinished frame uses the texture.
        unsafe {
            msg_send![
                (),
                raw,
                c"replaceRegion:mipmapLevel:withBytes:bytesPerRow:",
                MTLRegion::whole_2d(width, height) => MTLRegion,
                0 => NSUInteger,
                data.as_ptr().cast::<c_void>() => *const c_void,
                bytes_per_row => NSUInteger,
            ];
        }
        Ok(())
    }

    fn destroy_texture(&mut self, texture: Handle<Texture>) -> Result<(), GpuError> {
        // In-flight command buffers retain the Metal texture until they finish.
        self.textures
            .remove(texture)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Texture,
            })?;
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
        if source.msl.is_empty() {
            return Err(backend("the Metal backend needs MSL source"));
        }
        let shader = autoreleasepool(|| {
            let mut error: Id = null_mut();
            // SAFETY: the device is live; `newLibraryWithSource:options:error:` takes an
            // `NSString`, nil options, and an `NSError **` that it fills (autoreleased) on
            // failure, and returns a +1 library or nil.
            let library = unsafe {
                Retained::from_retained(msg_send![
                    Id,
                    &self.device,
                    c"newLibraryWithSource:options:error:",
                    ns_string(source.msl).as_ptr() => Id,
                    null_mut() => Id,
                    &raw mut error => *mut Id,
                ])
            }
            .ok_or_else(|| {
                // SAFETY: `error` is nil or the autoreleased `NSError` Metal returned.
                let description = unsafe { ns_error_description(error) };
                backend(format!("MSL failed to compile: {description}"))
            })?;
            // SAFETY: `library` is a live `MTLLibrary`.
            unsafe { set_label(&library, source.label) };
            let function = |name: &str| {
                // SAFETY: `newFunctionWithName:` takes an `NSString` and returns a +1 function
                // or nil.
                unsafe {
                    Retained::from_retained(msg_send![
                        Id,
                        &library,
                        c"newFunctionWithName:",
                        ns_string(name).as_ptr() => Id,
                    ])
                }
                .ok_or_else(|| backend(format!("the MSL has no function named `{name}`")))
            };
            Ok(MetalShader {
                vertex: function(source.vertex_entry)?,
                fragment: function(source.fragment_entry)?,
            })
        })?;
        Ok(self.shaders.insert(shader))
    }

    fn destroy_shader(&mut self, shader: Handle<Shader>) -> Result<(), GpuError> {
        // Pipelines hold their own retains on the shader's functions.
        self.shaders.remove(shader).ok_or(GpuError::InvalidHandle {
            kind: ResourceKind::Shader,
        })?;
        Ok(())
    }

    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<Handle<Pipeline>, GpuError> {
        let shader = self
            .shaders
            .get(desc.shader)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Shader,
            })?;
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
        ensure(
            locations
                .last()
                .is_none_or(|&last| last < METAL_VERTEX_ATTRIBUTES),
            "Metal vertex attribute shader_locations must be below 31",
        )?;
        ensure(
            desc.vertex_layouts.iter().all(|layout| {
                layout.array_stride.is_multiple_of(4)
                    && layout.attributes.iter().all(|a| a.offset.is_multiple_of(4))
            }),
            "Metal needs vertex array_strides and attribute offsets that are multiples of 4",
        )?;
        let pipeline = autoreleasepool(|| self.new_pipeline(desc, shader))?;
        Ok(self.pipelines.insert(pipeline))
    }

    fn destroy_pipeline(&mut self, pipeline: Handle<Pipeline>) -> Result<(), GpuError> {
        self.pipelines
            .remove(pipeline)
            .ok_or(GpuError::InvalidHandle {
                kind: ResourceKind::Pipeline,
            })?;
        Ok(())
    }

    fn submit(&mut self, commands: FrameView<'_>) -> Result<(), GpuError> {
        self.submit_frame(commands).map(|_presented| ())
    }
}

impl Drop for MetalGpu {
    fn drop(&mut self) {
        // Errors from earlier frames have nowhere to go now; the wait itself is what matters,
        // so no command buffer outlives the resources it was encoded against.
        let _ = self.wait_idle();
    }
}
