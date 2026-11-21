//! The [`GpuDevice`] trait every backend implements, and its errors.

use crate::command::FrameView;
use crate::pipeline::PipelineDesc;
use crate::resource::{
    Buffer, BufferDesc, Pipeline, Shader, ShaderSource, Texture, TextureDesc, TextureFormat,
};
use alloc::string::String;
use core::fmt;
use titan_core::Handle;

/// A kind of GPU resource, for error reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// A [`Buffer`].
    Buffer,
    /// A [`Texture`].
    Texture,
    /// A [`Shader`].
    Shader,
    /// A [`Pipeline`].
    Pipeline,
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::Shader => "shader",
            Self::Pipeline => "pipeline",
        })
    }
}

/// Why a [`GpuDevice`] call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    /// A handle names a destroyed resource or was not issued by this device.
    InvalidHandle {
        /// The handle's resource kind.
        kind: ResourceKind,
    },
    /// A descriptor or command argument breaks a documented rule.
    InvalidDescriptor {
        /// The rule that was broken.
        message: &'static str,
    },
    /// A buffer write or binding extends past the end of the buffer.
    BufferOutOfBounds {
        /// The byte offset of the write or binding.
        offset: u64,
        /// The number of bytes written; 0 for a binding.
        write_len: u64,
        /// The buffer's size in bytes.
        buffer_size: u64,
    },
    /// Texture data is not exactly one full mip level 0.
    TextureDataSizeMismatch {
        /// The required number of bytes: `width * height * bytes_per_pixel`.
        expected: usize,
        /// The number of bytes given.
        actual: usize,
    },
    /// A resource was used in a way its usage flags do not allow.
    MissingUsage {
        /// The resource's kind.
        kind: ResourceKind,
        /// The name of the missing usage flag, such as `"COPY_DST"`.
        required: &'static str,
    },
    /// A draw was recorded before any `SetPipeline` in its pass.
    DrawWithoutPipeline,
    /// An indexed draw was recorded before any `SetIndexBuffer` in its pass.
    DrawIndexedWithoutIndexBuffer,
    /// The native graphics API reported an error.
    Backend(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle { kind } => write!(f, "invalid or destroyed {kind} handle"),
            Self::InvalidDescriptor { message } => write!(f, "invalid descriptor: {message}"),
            Self::BufferOutOfBounds {
                offset,
                write_len,
                buffer_size,
            } => write!(
                f,
                "{write_len} bytes at offset {offset} exceed the buffer size of {buffer_size}"
            ),
            Self::TextureDataSizeMismatch { expected, actual } => write!(
                f,
                "texture data is {actual} bytes but mip level 0 is {expected} bytes"
            ),
            Self::MissingUsage { kind, required } => {
                write!(f, "{kind} is missing the {required} usage")
            }
            Self::DrawWithoutPipeline => f.write_str("draw without a pipeline set in the pass"),
            Self::DrawIndexedWithoutIndexBuffer => {
                f.write_str("indexed draw without an index buffer set in the pass")
            }
            Self::Backend(message) => write!(f, "backend error: {message}"),
        }
    }
}

impl core::error::Error for GpuError {}

/// A GPU that creates resources and runs frames of render passes.
///
/// Every backend (Metal, WebGPU, WebGL, and the test-only [`RecordingGpu`](crate::RecordingGpu))
/// implements this trait, and apps take it as `&mut dyn GpuDevice`. Handles are only valid on
/// the device that created them. Destroying a resource invalidates its handle immediately; the
/// backend keeps the native object alive until in-flight frames are done with it.
pub trait GpuDevice {
    /// Returns the format of the window surface that passes without a color target draw to.
    fn surface_format(&self) -> TextureFormat;

    /// Creates a buffer. Its contents start zeroed.
    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<Handle<Buffer>, GpuError>;

    /// Copies `data` into `buffer` at byte `offset`. Needs
    /// [`BufferUsage::COPY_DST`](crate::BufferUsage::COPY_DST). `offset` and `data.len()` must be
    /// multiples of 4, as WebGPU's `queue.writeBuffer` requires.
    fn write_buffer(
        &mut self,
        buffer: Handle<Buffer>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GpuError>;

    /// Destroys a buffer.
    fn destroy_buffer(&mut self, buffer: Handle<Buffer>) -> Result<(), GpuError>;

    /// Creates a texture. Its contents are undefined until written or rendered to. A depth
    /// format cannot have [`TextureUsage::COPY_DST`](crate::TextureUsage::COPY_DST), because
    /// WebGPU cannot copy CPU data into depth textures.
    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> Result<Handle<Texture>, GpuError>;

    /// Replaces mip level 0 of `texture` with `data`: tightly packed rows, top row first. Needs
    /// [`TextureUsage::COPY_DST`](crate::TextureUsage::COPY_DST), so the texture has a color
    /// format.
    fn write_texture(&mut self, texture: Handle<Texture>, data: &[u8]) -> Result<(), GpuError>;

    /// Destroys a texture.
    fn destroy_texture(&mut self, texture: Handle<Texture>) -> Result<(), GpuError>;

    /// Compiles a shader from the source language this backend reads.
    fn create_shader(&mut self, source: &ShaderSource<'_>) -> Result<Handle<Shader>, GpuError>;

    /// Destroys a shader. Pipelines created from it keep working.
    fn destroy_shader(&mut self, shader: Handle<Shader>) -> Result<(), GpuError>;

    /// Creates a render pipeline.
    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<Handle<Pipeline>, GpuError>;

    /// Destroys a render pipeline.
    fn destroy_pipeline(&mut self, pipeline: Handle<Pipeline>) -> Result<(), GpuError>;

    /// Runs a frame's passes in order. A frame that fails validation runs no pass at all.
    fn submit(&mut self, commands: FrameView<'_>) -> Result<(), GpuError>;
}
