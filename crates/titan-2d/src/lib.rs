//! Titan's batched 2D renderer: colored and textured quads drawn through any
//! [`GpuDevice`](titan_gpu::GpuDevice).
//!
//! The crate is `no_std` (it needs only `alloc`), forbids `unsafe`, and depends only on
//! `titan-core` and `titan-gpu`.
//!
//! - [`Camera2D`] maps a `+Y`-up world onto a `+Y`-down screen of pixels, with pan, zoom, and
//!   rotation, and converts points between the two.
//! - [`Sprite`] is one quad: a world-space [`Rect`](titan_core::Rect), a texture region, a tint,
//!   a rotation, and a layer. A [`DrawList2D`] collects a frame's sprites.
//! - [`Renderer2D`] owns the GPU pipeline and buffers. Each flush sorts sprites by layer,
//!   uploads them, and records one indexed draw per run of sprites that share a texture.
//! - [`SPRITE_SHADER`] and [`Vertex2D`] are the shader and vertex format it draws with.
//! - [`create_font_texture`] uploads an embedded 8×8 bitmap font, and
//!   [`DrawList2D::draw_text`] lays out text in it as ordinary sprites, styled by a
//!   [`TextStyle`] and sized by [`measure_text`].
//!
//! ```
//! use titan_2d::{Camera2D, Renderer2D};
//! use titan_core::{Color, Rect, Vec2};
//! use titan_gpu::{ClearColor, FrameCommands, GpuDevice, RecordingGpu, RenderPassDesc};
//!
//! let mut gpu = RecordingGpu::new();
//! let mut renderer = Renderer2D::new(&mut gpu)?;
//! let camera = Camera2D::bottom_left(Vec2::new(800.0, 600.0));
//!
//! renderer.draw_rect(Rect::from_pos_size(Vec2::ZERO, Vec2::splat(32.0)), Color::WHITE);
//! let mut frame = FrameCommands::new();
//! let pass = RenderPassDesc::clear_surface(ClearColor::BLACK);
//! renderer.flush(&mut gpu, &mut frame, pass, &camera)?;
//! gpu.submit(frame.view())?;
//!
//! renderer.destroy(&mut gpu)?;
//! # Ok::<(), titan_gpu::GpuError>(())
//! ```

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod camera;
mod draw_list;
mod renderer;
mod shader;
mod text;

pub use camera::Camera2D;
pub use draw_list::{DrawList2D, Sprite, UV_FULL};
pub use renderer::{INITIAL_QUAD_CAPACITY, Renderer2D};
pub use shader::{
    SPRITE_SHADER, SPRITE_SHADER_GLSL, SPRITE_SHADER_MSL, SPRITE_SHADER_WGSL, Vertex2D,
};
pub use text::{
    FONT_ATLAS_HEIGHT, FONT_ATLAS_WIDTH, GLYPH_SIZE, TextStyle, create_font_texture,
    font_atlas_rgba, glyph_uv, measure_text,
};
