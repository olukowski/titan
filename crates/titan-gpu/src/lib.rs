//! Titan's GPU abstraction: what the engine asks of Metal, WebGPU, and WebGL.
//!
//! The crate is `no_std` (it needs only `alloc`) and has no dependencies beyond `titan-core`. It
//! defines the interface and a test device; native backends live in their own crates.
//!
//! - Resources: [`Buffer`], [`Texture`], [`Shader`], and [`Pipeline`] are named by 8-byte
//!   generational [`Handle`]s that a backend issues from a [`ResourcePool`], so a destroyed
//!   resource's handle never resolves again. They are created from plain descriptors such as
//!   [`BufferDesc`], [`TextureDesc`], [`ShaderSource`], and [`PipelineDesc`].
//! - Commands: a frame is a [`FrameCommands`] list of render passes, recorded through a
//!   [`RenderPassEncoder`] into any [`Allocator`](core::alloc::Allocator), such as a per-frame
//!   [`BumpArena`](titan_core::BumpArena).
//! - Devices: [`GpuDevice`] is the object-safe trait backends implement. [`RecordingGpu`]
//!   implements it on the CPU, validating every call strictly and recording it for inspection
//!   in tests and headless runs.

#![no_std]

extern crate alloc;

mod command;
mod device;
mod pipeline;
mod recording;
mod resource;

pub use command::{
    ClearColor, ColorAttachment, DepthAttachment, DrawCmd, FrameCommands, FrameView, IndexFormat,
    LoadOp, RenderCmd, RenderPass, RenderPassDesc, RenderPassEncoder, ScissorRect, StoreOp,
    Viewport,
};
pub use device::{GpuDevice, GpuError, ResourceKind};
pub use pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, CompareFunction, CullMode,
    DepthStencilState, FrontFace, PipelineDesc, PrimitiveTopology, VertexAttribute, VertexFormat,
    VertexLayout, VertexStepMode,
};
pub use recording::{
    GpuEvent, RecordedBuffer, RecordedFrame, RecordedPass, RecordedPipeline, RecordedShader,
    RecordedTexture, RecordedVertexLayout, RecordingGpu, ResourceCounts,
};
pub use resource::{
    AddressMode, Buffer, BufferDesc, BufferUsage, FilterMode, MAX_VERTEX_BUFFERS,
    METAL_BUFFER_BINDING_OFFSET, Pipeline, Resource, ResourcePool, SamplerDesc, Shader,
    ShaderSource, Texture, TextureDesc, TextureFormat, TextureUsage,
};
pub use titan_core::Handle;
