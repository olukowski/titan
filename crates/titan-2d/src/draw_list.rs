//! Sprites and the per-frame list that collects them.

use alloc::vec::Vec;
use titan_core::{Color, Handle, Rect, Vec2};
use titan_gpu::Texture;

/// The texture region covering the whole texture.
pub const UV_FULL: Rect = Rect::from_min_max(Vec2::ZERO, Vec2::ONE);

/// One quad to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprite {
    /// The world-space area the quad covers before rotation, with `+Y` up.
    pub rect: Rect,
    /// The texture region drawn into `rect`, in texture coordinates with `(0, 0)` at the
    /// texture's top-left: `uv.min` maps to the quad's top-left corner and `uv.max` to its
    /// bottom-right.
    pub uv: Rect,
    /// The tint multiplied with each texel; the vertex color for untextured quads.
    pub color: Color,
    /// The texture, or `None` for a solid quad of `color`.
    pub texture: Option<Handle<Texture>>,
    /// Counter-clockwise rotation in radians about `rect`'s center.
    pub rotation: f32,
    /// Draw order: lower layers are drawn first, so higher layers cover them. Sprites in the
    /// same layer are drawn in submission order.
    pub layer: i32,
}

impl Sprite {
    /// A solid quad of `color` covering `rect`.
    #[must_use]
    pub const fn solid(rect: Rect, color: Color) -> Self {
        Self {
            rect,
            uv: UV_FULL,
            color,
            texture: None,
            rotation: 0.0,
            layer: 0,
        }
    }

    /// The whole of `texture`, untinted, covering `rect`.
    #[must_use]
    pub const fn textured(rect: Rect, texture: Handle<Texture>) -> Self {
        Self {
            texture: Some(texture),
            ..Self::solid(rect, Color::WHITE)
        }
    }

    /// Returns the sprite drawing the texture region `uv`.
    #[must_use]
    pub const fn with_uv(mut self, uv: Rect) -> Self {
        self.uv = uv;
        self
    }

    /// Returns the sprite tinted with `color`.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Returns the sprite rotated `rotation` radians counter-clockwise about its center.
    #[must_use]
    pub const fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns the sprite in `layer`.
    #[must_use]
    pub const fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }
}

/// A frame's sprites, in submission order. Reuse one across frames with
/// [`clear`](Self::clear) to keep its capacity.
#[derive(Clone, Debug, Default)]
pub struct DrawList2D {
    sprites: Vec<Sprite>,
}

impl DrawList2D {
    /// Creates an empty list without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sprites: Vec::new(),
        }
    }

    /// Creates an empty list with room for `capacity` sprites.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sprites: Vec::with_capacity(capacity),
        }
    }

    /// Adds `sprite`.
    pub fn draw_sprite(&mut self, sprite: Sprite) {
        self.sprites.push(sprite);
    }

    /// Adds a solid quad of `color` covering `rect`.
    pub fn draw_rect(&mut self, rect: Rect, color: Color) {
        self.draw_sprite(Sprite::solid(rect, color));
    }

    /// Adds a solid quad of `color` whose bottom-left corner is `position` and whose size is
    /// `size`.
    pub fn draw_quad(&mut self, position: Vec2, size: Vec2, color: Color) {
        self.draw_rect(Rect::from_pos_size(position, size), color);
    }

    /// Removes every sprite, keeping capacity.
    pub fn clear(&mut self) {
        self.sprites.clear();
    }

    /// Returns the number of sprites.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// Returns whether the list has no sprites.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    /// Returns the sprites in submission order.
    #[must_use]
    pub fn sprites(&self) -> &[Sprite] {
        &self.sprites
    }
}
