//! Bitmap text: an embedded 8×8 monospace font, its atlas texture, and layout that emits
//! one [`Sprite`] per glyph into a [`DrawList2D`], so text batches with other sprites that
//! use the font texture.
//!
//! Text is anchored at the top-left corner of its first glyph cell. The world is `+Y` up (see
//! [`Camera2D`](crate::Camera2D)), so successive lines move toward `-Y`.

use crate::draw_list::{DrawList2D, Sprite};
use alloc::vec::Vec;
use titan_core::{Color, Handle, Rect, Vec2};
use titan_gpu::{GpuDevice, GpuError, SamplerDesc, Texture, TextureDesc, TextureFormat};

/// The side of one glyph cell in atlas texels.
pub const GLYPH_SIZE: u32 = 8;
/// The width of the font atlas in texels: 16 glyph cells.
pub const FONT_ATLAS_WIDTH: u32 = 128;
/// The height of the font atlas in texels: 6 glyph cells.
pub const FONT_ATLAS_HEIGHT: u32 = 48;

const ATLAS_COLUMNS: usize = (FONT_ATLAS_WIDTH / GLYPH_SIZE) as usize;
/// The glyph drawn for any char outside printable ASCII.
const FALLBACK_GLYPH: usize = 95;

/// Glyph `i` is char `0x20 + i` for printable ASCII (`' '..='~'`); the last is the fallback
/// box. Each glyph is 8 rows, top first, with bit 7 as the leftmost pixel. The right column
/// and bottom row stay blank (descenders aside), which spaces letters and lines apart.
///
/// The glyphs are an original design drawn for Titan, licensed with the rest of the
/// repository under MIT OR Apache-2.0.
#[rustfmt::skip]
const GLYPHS: [[u8; 8]; 96] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ' '
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x00, 0x10, 0x00], // '!'
    [0x28, 0x28, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00], // '"'
    [0x28, 0x28, 0x7C, 0x28, 0x7C, 0x28, 0x28, 0x00], // '#'
    [0x10, 0x3C, 0x50, 0x38, 0x14, 0x78, 0x10, 0x00], // '$'
    [0x60, 0x64, 0x08, 0x10, 0x20, 0x4C, 0x0C, 0x00], // '%'
    [0x30, 0x48, 0x50, 0x20, 0x54, 0x48, 0x34, 0x00], // '&'
    [0x10, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00], // '\''
    [0x08, 0x10, 0x20, 0x20, 0x20, 0x10, 0x08, 0x00], // '('
    [0x20, 0x10, 0x08, 0x08, 0x08, 0x10, 0x20, 0x00], // ')'
    [0x00, 0x10, 0x54, 0x38, 0x54, 0x10, 0x00, 0x00], // '*'
    [0x00, 0x10, 0x10, 0x7C, 0x10, 0x10, 0x00, 0x00], // '+'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x10, 0x20], // ','
    [0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00], // '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30, 0x00], // '.'
    [0x00, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00], // '/'
    [0x38, 0x44, 0x4C, 0x54, 0x64, 0x44, 0x38, 0x00], // '0'
    [0x10, 0x30, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00], // '1'
    [0x38, 0x44, 0x04, 0x08, 0x10, 0x20, 0x7C, 0x00], // '2'
    [0x7C, 0x08, 0x10, 0x08, 0x04, 0x44, 0x38, 0x00], // '3'
    [0x08, 0x18, 0x28, 0x48, 0x7C, 0x08, 0x08, 0x00], // '4'
    [0x7C, 0x40, 0x78, 0x04, 0x04, 0x44, 0x38, 0x00], // '5'
    [0x18, 0x20, 0x40, 0x78, 0x44, 0x44, 0x38, 0x00], // '6'
    [0x7C, 0x04, 0x08, 0x10, 0x20, 0x20, 0x20, 0x00], // '7'
    [0x38, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38, 0x00], // '8'
    [0x38, 0x44, 0x44, 0x3C, 0x04, 0x08, 0x30, 0x00], // '9'
    [0x00, 0x30, 0x30, 0x00, 0x30, 0x30, 0x00, 0x00], // ':'
    [0x00, 0x30, 0x30, 0x00, 0x30, 0x10, 0x20, 0x00], // ';'
    [0x08, 0x10, 0x20, 0x40, 0x20, 0x10, 0x08, 0x00], // '<'
    [0x00, 0x00, 0x7C, 0x00, 0x7C, 0x00, 0x00, 0x00], // '='
    [0x20, 0x10, 0x08, 0x04, 0x08, 0x10, 0x20, 0x00], // '>'
    [0x38, 0x44, 0x04, 0x08, 0x10, 0x00, 0x10, 0x00], // '?'
    [0x38, 0x44, 0x5C, 0x54, 0x5C, 0x40, 0x38, 0x00], // '@'
    [0x38, 0x44, 0x44, 0x7C, 0x44, 0x44, 0x44, 0x00], // 'A'
    [0x78, 0x44, 0x44, 0x78, 0x44, 0x44, 0x78, 0x00], // 'B'
    [0x38, 0x44, 0x40, 0x40, 0x40, 0x44, 0x38, 0x00], // 'C'
    [0x70, 0x48, 0x44, 0x44, 0x44, 0x48, 0x70, 0x00], // 'D'
    [0x7C, 0x40, 0x40, 0x78, 0x40, 0x40, 0x7C, 0x00], // 'E'
    [0x7C, 0x40, 0x40, 0x78, 0x40, 0x40, 0x40, 0x00], // 'F'
    [0x38, 0x44, 0x40, 0x5C, 0x44, 0x44, 0x3C, 0x00], // 'G'
    [0x44, 0x44, 0x44, 0x7C, 0x44, 0x44, 0x44, 0x00], // 'H'
    [0x38, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00], // 'I'
    [0x1C, 0x08, 0x08, 0x08, 0x08, 0x48, 0x30, 0x00], // 'J'
    [0x44, 0x48, 0x50, 0x60, 0x50, 0x48, 0x44, 0x00], // 'K'
    [0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7C, 0x00], // 'L'
    [0x44, 0x6C, 0x54, 0x54, 0x44, 0x44, 0x44, 0x00], // 'M'
    [0x44, 0x44, 0x64, 0x54, 0x4C, 0x44, 0x44, 0x00], // 'N'
    [0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00], // 'O'
    [0x78, 0x44, 0x44, 0x78, 0x40, 0x40, 0x40, 0x00], // 'P'
    [0x38, 0x44, 0x44, 0x44, 0x54, 0x48, 0x34, 0x00], // 'Q'
    [0x78, 0x44, 0x44, 0x78, 0x50, 0x48, 0x44, 0x00], // 'R'
    [0x3C, 0x40, 0x40, 0x38, 0x04, 0x04, 0x78, 0x00], // 'S'
    [0x7C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00], // 'T'
    [0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00], // 'U'
    [0x44, 0x44, 0x44, 0x44, 0x44, 0x28, 0x10, 0x00], // 'V'
    [0x44, 0x44, 0x44, 0x54, 0x54, 0x54, 0x28, 0x00], // 'W'
    [0x44, 0x44, 0x28, 0x10, 0x28, 0x44, 0x44, 0x00], // 'X'
    [0x44, 0x44, 0x28, 0x10, 0x10, 0x10, 0x10, 0x00], // 'Y'
    [0x7C, 0x04, 0x08, 0x10, 0x20, 0x40, 0x7C, 0x00], // 'Z'
    [0x38, 0x20, 0x20, 0x20, 0x20, 0x20, 0x38, 0x00], // '['
    [0x00, 0x40, 0x20, 0x10, 0x08, 0x04, 0x00, 0x00], // '\\'
    [0x38, 0x08, 0x08, 0x08, 0x08, 0x08, 0x38, 0x00], // ']'
    [0x10, 0x28, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00], // '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7C, 0x00], // '_'
    [0x20, 0x10, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00], // '`'
    [0x00, 0x00, 0x38, 0x04, 0x3C, 0x44, 0x3C, 0x00], // 'a'
    [0x40, 0x40, 0x58, 0x64, 0x44, 0x44, 0x78, 0x00], // 'b'
    [0x00, 0x00, 0x38, 0x40, 0x40, 0x44, 0x38, 0x00], // 'c'
    [0x04, 0x04, 0x34, 0x4C, 0x44, 0x44, 0x3C, 0x00], // 'd'
    [0x00, 0x00, 0x38, 0x44, 0x7C, 0x40, 0x38, 0x00], // 'e'
    [0x18, 0x24, 0x20, 0x70, 0x20, 0x20, 0x20, 0x00], // 'f'
    [0x00, 0x00, 0x3C, 0x44, 0x44, 0x3C, 0x04, 0x38], // 'g'
    [0x40, 0x40, 0x58, 0x64, 0x44, 0x44, 0x44, 0x00], // 'h'
    [0x10, 0x00, 0x30, 0x10, 0x10, 0x10, 0x38, 0x00], // 'i'
    [0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x48, 0x30], // 'j'
    [0x40, 0x40, 0x48, 0x50, 0x60, 0x50, 0x48, 0x00], // 'k'
    [0x30, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00], // 'l'
    [0x00, 0x00, 0x68, 0x54, 0x54, 0x54, 0x54, 0x00], // 'm'
    [0x00, 0x00, 0x58, 0x64, 0x44, 0x44, 0x44, 0x00], // 'n'
    [0x00, 0x00, 0x38, 0x44, 0x44, 0x44, 0x38, 0x00], // 'o'
    [0x00, 0x00, 0x78, 0x44, 0x44, 0x78, 0x40, 0x40], // 'p'
    [0x00, 0x00, 0x3C, 0x44, 0x44, 0x3C, 0x04, 0x04], // 'q'
    [0x00, 0x00, 0x58, 0x64, 0x40, 0x40, 0x40, 0x00], // 'r'
    [0x00, 0x00, 0x3C, 0x40, 0x38, 0x04, 0x78, 0x00], // 's'
    [0x20, 0x20, 0x70, 0x20, 0x20, 0x24, 0x18, 0x00], // 't'
    [0x00, 0x00, 0x44, 0x44, 0x44, 0x4C, 0x34, 0x00], // 'u'
    [0x00, 0x00, 0x44, 0x44, 0x44, 0x28, 0x10, 0x00], // 'v'
    [0x00, 0x00, 0x44, 0x44, 0x54, 0x54, 0x28, 0x00], // 'w'
    [0x00, 0x00, 0x44, 0x28, 0x10, 0x28, 0x44, 0x00], // 'x'
    [0x00, 0x00, 0x44, 0x44, 0x44, 0x3C, 0x04, 0x38], // 'y'
    [0x00, 0x00, 0x7C, 0x08, 0x10, 0x20, 0x7C, 0x00], // 'z'
    [0x08, 0x10, 0x10, 0x20, 0x10, 0x10, 0x08, 0x00], // '{'
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00], // '|'
    [0x20, 0x10, 0x10, 0x08, 0x10, 0x10, 0x20, 0x00], // '}'
    [0x00, 0x00, 0x20, 0x54, 0x08, 0x00, 0x00, 0x00], // '~'
    [0xFE, 0x82, 0x82, 0x82, 0x82, 0x82, 0xFE, 0x00], // fallback
];

/// Returns the font atlas as RGBA8 texels, row-major from the top-left:
/// [`FONT_ATLAS_WIDTH`] × [`FONT_ATLAS_HEIGHT`] texels, 4 bytes each. Every texel is white;
/// alpha is 255 where a glyph pixel is set and 0 elsewhere, so [`Sprite::color`] tints text.
/// Glyph `i` sits in cell column `i % 16`, row `i / 16`.
#[must_use]
pub fn font_atlas_rgba() -> Vec<u8> {
    let width = FONT_ATLAS_WIDTH as usize;
    let mut rgba = [255, 255, 255, 0].repeat(width * FONT_ATLAS_HEIGHT as usize);
    for (index, glyph) in GLYPHS.iter().enumerate() {
        let left = index % ATLAS_COLUMNS * GLYPH_SIZE as usize;
        let top = index / ATLAS_COLUMNS * GLYPH_SIZE as usize;
        for (y, bits) in glyph.iter().enumerate() {
            for x in 0..8 {
                if bits & (0x80 >> x) != 0 {
                    rgba[((top + y) * width + left + x) * 4 + 3] = 255;
                }
            }
        }
    }
    rgba
}

/// Creates the font atlas texture: [`font_atlas_rgba`] with nearest-neighbour sampling, so
/// scaled text stays crisp. Draw text with it through [`DrawList2D::draw_text`].
///
/// # Errors
///
/// Returns the device's error if creating or writing the texture fails, after attempting to
/// destroy a texture whose write failed.
pub fn create_font_texture(gpu: &mut dyn GpuDevice) -> Result<Handle<Texture>, GpuError> {
    let texture = gpu.create_texture(&TextureDesc {
        label: Some("titan-2d font"),
        format: TextureFormat::Rgba8Unorm,
        sampler: SamplerDesc::NEAREST_CLAMP,
        ..TextureDesc::rgba8_2d(FONT_ATLAS_WIDTH, FONT_ATLAS_HEIGHT)
    })?;
    if let Err(error) = gpu.write_texture(texture, &font_atlas_rgba()) {
        // The write error is the one to report.
        let _ = gpu.destroy_texture(texture);
        return Err(error);
    }
    Ok(texture)
}

/// Returns the atlas region of `c`'s glyph, in texture coordinates with `(0, 0)` at the
/// top-left as [`Sprite::uv`] expects. Chars outside printable ASCII (`' '..='~'`) get the
/// fallback box glyph.
#[must_use]
pub fn glyph_uv(c: char) -> Rect {
    let index = match c {
        ' '..='~' => c as usize - 0x20,
        _ => FALLBACK_GLYPH,
    };
    let (column, row) = (
        (index % ATLAS_COLUMNS) as u32,
        (index / ATLAS_COLUMNS) as u32,
    );
    let u = |column: u32| (column * GLYPH_SIZE) as f32 / FONT_ATLAS_WIDTH as f32;
    let v = |row: u32| (row * GLYPH_SIZE) as f32 / FONT_ATLAS_HEIGHT as f32;
    Rect::from_min_max(
        Vec2::new(u(column), v(row)),
        Vec2::new(u(column + 1), v(row + 1)),
    )
}

/// How [`DrawList2D::draw_text`] and [`measure_text`] lay out text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    /// World units per atlas texel. A glyph cell is `GLYPH_SIZE * scale` on each side, which
    /// must be positive and finite, and is both the advance and the line height.
    pub scale: f32,
    /// The tint of every glyph.
    pub color: Color,
    /// The [`Sprite::layer`] of every glyph.
    pub layer: i32,
    /// If set, words wrap onto a new line rather than extend past this width in world units.
    /// At least one glyph fits on each line.
    pub wrap_width: Option<f32>,
}

impl TextStyle {
    /// White text at `scale` in layer 0, without wrapping.
    #[must_use]
    pub const fn new(scale: f32) -> Self {
        Self {
            scale,
            color: Color::WHITE,
            layer: 0,
            wrap_width: None,
        }
    }

    /// Returns the style tinted with `color`.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Returns the style drawing into `layer`.
    #[must_use]
    pub const fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    /// Returns the style wrapping words at `wrap_width` world units.
    #[must_use]
    pub const fn with_wrap_width(mut self, wrap_width: f32) -> Self {
        self.wrap_width = Some(wrap_width);
        self
    }

    /// The side of one glyph cell in world units.
    fn cell(&self) -> f32 {
        GLYPH_SIZE as f32 * self.scale
    }
}

/// Returns the size of the box [`DrawList2D::draw_text`] fills with `text`: the widest line's
/// glyph count by the line count, in glyph cells. Spaces count; spaces dropped at a wrap do
/// not. Empty text measures [`Vec2::ZERO`]; `"\n"` is two empty lines.
#[must_use]
pub fn measure_text(text: &str, style: &TextStyle) -> Vec2 {
    let (columns, lines) = layout(text, style, |_, _, _| {});
    Vec2::new(columns as f32, lines as f32) * style.cell()
}

impl DrawList2D {
    /// Adds one textured sprite per visible glyph of `text`, drawn from `font`, the texture
    /// [`create_font_texture`] made.
    ///
    /// `top_left` is the world position of the top-left corner of the first glyph cell. With
    /// the world `+Y` up, the glyph in column `c` of line `l` covers `(x + c * cell,
    /// y - (l + 1) * cell)` to `(x + (c + 1) * cell, y - l * cell)`, where `cell` is
    /// `GLYPH_SIZE * style.scale`.
    ///
    /// `'\n'` starts a new line and `' '` advances one column without a sprite. Every other
    /// char is one glyph; chars outside printable ASCII (including `'\t'` and `'\r'`) draw the
    /// fallback box, so they stay visible. With [`TextStyle::wrap_width`] set, a word that does
    /// not fit on the current line moves to the next one, the space before it dropped, and a
    /// word longer than a whole line breaks at the line's end.
    pub fn draw_text(
        &mut self,
        font: Handle<Texture>,
        text: &str,
        top_left: Vec2,
        style: &TextStyle,
    ) {
        let cell = style.cell();
        layout(text, style, |c, column, line| {
            if c == ' ' {
                return;
            }
            let (column, line) = (column as f32, line as f32);
            let rect = Rect::from_min_max(
                Vec2::new(top_left.x + column * cell, top_left.y - (line + 1.0) * cell),
                Vec2::new(top_left.x + (column + 1.0) * cell, top_left.y - line * cell),
            );
            self.draw_sprite(
                Sprite::textured(rect, font)
                    .with_uv(glyph_uv(c))
                    .with_color(style.color)
                    .with_layer(style.layer),
            );
        });
    }
}

/// Calls `place(c, column, line)` for each char of `text` that occupies a cell, spaces
/// included, and returns `(widest line's column count, line count)`.
fn layout(
    text: &str,
    style: &TextStyle,
    mut place: impl FnMut(char, usize, usize),
) -> (usize, usize) {
    // Whether a line of `columns` cells fits, its extent computed as `measure_text` computes
    // it so a measured width wraps nothing. One cell always fits, even a negative or NaN width.
    let cell = style.cell();
    let fits = |columns: usize| {
        style
            .wrap_width
            .is_none_or(|width| columns <= 1 || columns as f32 * cell <= width)
    };
    let (mut widest, mut line) = (0, 0);
    for (index, hard_line) in text.split('\n').enumerate() {
        if index > 0 {
            line += 1;
        }
        let mut column = 0;
        for (index, word) in hard_line.split(' ').enumerate() {
            let len = word.chars().count();
            if index > 0 {
                if fits(column + 1 + len) {
                    place(' ', column, line);
                    column += 1;
                } else if len > 0 && column > 0 {
                    // The space and the word do not both fit: drop the space, and move the word
                    // to a new line unless the line is still empty. With no word to carry over
                    // (trailing or repeated spaces), start no line, so no empty line appears.
                    line += 1;
                    column = 0;
                }
            }
            for c in word.chars() {
                if !fits(column + 1) {
                    widest = widest.max(column);
                    line += 1;
                    column = 0;
                }
                place(c, column, line);
                column += 1;
            }
            widest = widest.max(column);
        }
    }
    let lines = if text.is_empty() { 0 } else { line + 1 };
    (widest, lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera2D, Renderer2D, Vertex2D};
    use alloc::string::String;
    use alloc::vec;
    use titan_gpu::{ClearColor, DrawCmd, FrameCommands, RecordingGpu, RenderCmd, RenderPassDesc};

    fn uv(min: (f32, f32), max: (f32, f32)) -> Rect {
        Rect::from_min_max(Vec2::new(min.0, min.1), Vec2::new(max.0, max.1))
    }

    /// Lays out `text` at scale 1 wrapping to `columns` glyphs (none if 0), and returns the
    /// lines as strings with every placed char, spaces included, at its column.
    fn lines(text: &str, columns: usize) -> Vec<String> {
        let mut style = TextStyle::new(1.0);
        if columns > 0 {
            style = style.with_wrap_width(columns as f32 * 8.0);
        }
        let mut out: Vec<Vec<char>> = Vec::new();
        let (_, count) = layout(text, &style, |c, column, line| {
            if out.len() <= line {
                out.resize(line + 1, Vec::new());
            }
            assert_eq!(out[line].len(), column, "{c:?} placed out of order");
            out[line].push(c);
        });
        out.resize(count, Vec::new());
        out.into_iter().map(|l| l.into_iter().collect()).collect()
    }

    #[test]
    fn glyph_uvs_index_the_atlas_grid() {
        // 'A' is glyph 33: column 1, row 2.
        assert_eq!(
            glyph_uv('A'),
            uv((8.0 / 128.0, 16.0 / 48.0), (16.0 / 128.0, 24.0 / 48.0))
        );
        // '~' is glyph 94: column 14, row 5.
        assert_eq!(
            glyph_uv('~'),
            uv((112.0 / 128.0, 40.0 / 48.0), (120.0 / 128.0, 1.0))
        );
        let fallback = uv((120.0 / 128.0, 40.0 / 48.0), (1.0, 1.0));
        for c in ['é', '\t', '\r', '\u{7F}', '\u{1F}'] {
            assert_eq!(glyph_uv(c), fallback, "{c:?}");
        }
        assert_ne!(glyph_uv(' '), fallback);
    }

    #[test]
    fn atlas_holds_white_glyphs_in_their_cells() {
        let rgba = font_atlas_rgba();
        assert_eq!(rgba.len(), 128 * 48 * 4);
        assert!(
            rgba.as_chunks::<4>()
                .0
                .iter()
                .all(|&[r, g, b, a]| { [r, g, b] == [255; 3] && (a == 0 || a == 255) })
        );
        let alpha = |x: usize, y: usize| rgba[(y * 128 + x) * 4 + 3];
        // The whole 8×8 cell of 'A' (column 1, row 2), so nothing spills past its blank right
        // column and bottom row into the neighbours.
        let a = [
            "..###...", ".#...#..", ".#...#..", ".#####..", ".#...#..", ".#...#..", ".#...#..",
            "........",
        ];
        for (y, row) in a.iter().enumerate() {
            let texels: String = (0..8)
                .map(|x| {
                    if alpha(8 + x, 16 + y) == 255 {
                        '#'
                    } else {
                        '.'
                    }
                })
                .collect();
            assert_eq!(texels, *row, "row {y}");
        }
        // ' ' (cell 0) is blank; the fallback box's corners reach the cell's corners.
        assert!((0..8).all(|y| (0..8).all(|x| alpha(x, y) == 0)));
        assert_eq!(
            [alpha(120, 40), alpha(126, 46), alpha(127, 47)],
            [255, 255, 0]
        );
    }

    #[test]
    fn layout_breaks_lines_and_wraps_words() {
        assert_eq!(lines("ab\ncd", 0), ["ab", "cd"]);
        assert_eq!(lines("a  b\n\nc", 0), ["a  b", "", "c"]);
        assert_eq!(lines("\n", 0), ["", ""]);
        assert!(lines("", 0).is_empty());
        // "hello big" fits 9 columns exactly; one more char breaks before the word.
        assert_eq!(lines("hello big world", 9), ["hello big", "world"]);
        assert_eq!(lines("hello bigg world", 9), ["hello", "bigg", "world"]);
        // Words longer than a line break at its end, after moving to a fresh line.
        assert_eq!(lines("abcdefghijkl", 5), ["abcde", "fghij", "kl"]);
        assert_eq!(lines("ab abcdefgh", 5), ["ab", "abcde", "fgh"]);
        // Spaces at a break vanish without adding an empty line; explicit newlines still work.
        assert_eq!(lines("abc   ", 3), ["abc"]);
        assert_eq!(lines("abc   d", 3), ["abc", "d"]);
        assert_eq!(lines("a\nb c", 3), ["a", "b c"]);
        // A leading space that does not fit with its word is dropped, not the word split.
        assert_eq!(lines(" abc", 3), ["abc"]);
        assert_eq!(lines("a\n abc", 3), ["a", "abc"]);
        assert_eq!(lines(" ab", 3), [" ab"]);
        // Unknown chars take a cell like any glyph.
        assert_eq!(lines("é\té", 2), ["é\t", "é"]);
    }

    #[test]
    fn wrap_width_fits_whole_cells_with_at_least_one() {
        // Just under 9 cells floors to 8 columns, so "hello big" no longer fits.
        let style = TextStyle::new(2.0).with_wrap_width(9.0 * 16.0 - 0.5);
        assert_eq!(
            measure_text("hello big world", &style),
            Vec2::new(5.0, 3.0) * 16.0
        );
        let narrow = TextStyle::new(1.0).with_wrap_width(3.0);
        assert_eq!(measure_text("ab", &narrow), Vec2::new(8.0, 16.0));
        // A measured width wraps nothing, and one representable width less wraps the last
        // glyph, even when `width / cell` rounds across a whole cell.
        for (scale, text) in [(1.1, "abc"), (1.2, "abcdefghijklmno")] {
            let width = measure_text(text, &TextStyle::new(scale)).x;
            let at = |width: f32| measure_text(text, &TextStyle::new(scale).with_wrap_width(width));
            let cell = GLYPH_SIZE as f32 * scale;
            assert_eq!(at(width), Vec2::new(width, cell), "scale {scale}");
            assert_eq!(at(width.next_down()).y, 2.0 * cell, "scale {scale}");
        }
        // Near this width, adjacent column counts round to one extent; "ab" still fits.
        let wide = TextStyle::new(0.1).with_wrap_width(107_374_192.0);
        assert_eq!(measure_text("ab", &wide), Vec2::new(1.6, 0.8));
    }

    #[test]
    fn measure_counts_widest_line_and_lines() {
        let style = TextStyle::new(2.0);
        assert_eq!(measure_text("", &style), Vec2::ZERO);
        assert_eq!(measure_text("\n", &style), Vec2::new(0.0, 32.0));
        assert_eq!(measure_text("ab\ncde \n", &style), Vec2::new(64.0, 48.0));
        let wrapped = TextStyle::new(1.0).with_wrap_width(72.0);
        assert_eq!(
            measure_text("hello big world", &wrapped),
            Vec2::new(72.0, 16.0)
        );
        // A word broken across lines measures its widest piece, not its last.
        let narrow = TextStyle::new(1.0).with_wrap_width(40.0);
        assert_eq!(measure_text("abcdef", &narrow), Vec2::new(40.0, 16.0));
    }

    #[test]
    fn draw_text_places_glyph_sprites_down_from_top_left() {
        let font = create_font_texture(&mut RecordingGpu::new()).unwrap();
        let red = Color::rgb(1.0, 0.0, 0.0);
        let style = TextStyle::new(2.0).with_color(red).with_layer(3);
        let mut list = DrawList2D::new();
        list.draw_text(font, "ab\nc é", Vec2::new(10.0, 100.0), &style);

        let cell = |column: f32, line: f32| {
            Rect::from_min_max(
                Vec2::new(10.0 + column * 16.0, 100.0 - (line + 1.0) * 16.0),
                Vec2::new(26.0 + column * 16.0, 100.0 - line * 16.0),
            )
        };
        let expected = [
            ('a', 0.0, 0.0),
            ('b', 1.0, 0.0),
            ('c', 0.0, 1.0),
            ('é', 2.0, 1.0),
        ]
        .map(|(c, column, line)| {
            Sprite::textured(cell(column, line), font)
                .with_uv(glyph_uv(c))
                .with_color(red)
                .with_layer(3)
        });
        assert_eq!(list.sprites(), expected);
    }

    #[test]
    fn font_texture_text_batches_with_other_font_sprites() {
        let mut gpu = RecordingGpu::new();
        let mut renderer = Renderer2D::new(&mut gpu).unwrap();
        let font = create_font_texture(&mut gpu).unwrap();
        let texture = gpu.texture(font).unwrap();
        assert_eq!(texture.label.as_deref(), Some("titan-2d font"));
        assert_eq!((texture.width, texture.height), (128, 48));
        assert_eq!(texture.format, TextureFormat::Rgba8Unorm);
        assert_eq!(texture.sampler, SamplerDesc::NEAREST_CLAMP);
        assert_eq!(texture.data, font_atlas_rgba());

        let top_left = Vec2::new(0.0, 100.0);
        renderer
            .draw_list_mut()
            .draw_text(font, "Hi !", top_left, &TextStyle::new(1.0));
        renderer.draw_sprite(Sprite::textured(
            Rect::from_min_max(Vec2::ZERO, Vec2::ONE),
            font,
        ));
        let mut frame = FrameCommands::new();
        let pass = RenderPassDesc::clear_surface(ClearColor::BLACK);
        let camera = Camera2D::bottom_left(Vec2::new(800.0, 600.0));
        renderer.flush(&mut gpu, &mut frame, pass, &camera).unwrap();
        gpu.submit(frame.view()).unwrap();

        let commands = &gpu.last_frame().unwrap().passes[0].commands;
        let mut vertex_buffer = None;
        let mut draws = vec![];
        let mut bound = None;
        for command in commands {
            match *command {
                RenderCmd::SetVertexBuffer { buffer, .. } => vertex_buffer = Some(buffer),
                RenderCmd::SetTexture { texture, .. } => bound = Some(texture),
                RenderCmd::Draw(DrawCmd::Indexed {
                    index_count,
                    first_index,
                    ..
                }) => draws.push((bound.unwrap(), first_index, index_count)),
                _ => {}
            }
        }
        // Three glyphs (the space draws nothing) plus the extra sprite: one draw.
        assert_eq!(draws, [(font, 0, 4 * 6)]);

        // The first quad is 'H'; its corners run bottom-left, bottom-right, top-right, top-left.
        let data = &gpu.buffer(vertex_buffer.unwrap()).unwrap().data;
        let uv_at = |vertex: usize| {
            let offset = vertex * Vertex2D::SIZE_BYTES as usize + 8;
            let f = |at: usize| f32::from_le_bytes(data[at..at + 4].try_into().unwrap());
            [f(offset), f(offset + 4)]
        };
        let Rect { min, max } = glyph_uv('H');
        assert_eq!(
            [uv_at(0), uv_at(1), uv_at(2), uv_at(3)],
            [
                [min.x, max.y],
                [max.x, max.y],
                [max.x, min.y],
                [min.x, min.y]
            ]
        );
    }
}
