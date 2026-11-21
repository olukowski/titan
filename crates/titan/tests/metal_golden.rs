//! Golden frames: draws sprites with `Renderer2D` on the real Metal GPU and checks the pixels
//! read back, so camera, texturing, blending, sorting, and rotation regressions show up as
//! wrong colors.

#![cfg(target_os = "macos")]

use core::f32::consts::FRAC_PI_2;
use titan::core::{Color, Rect, Vec2};
use titan::gpu::{
    ClearColor, ColorAttachment, FrameCommands, GpuDevice, Handle, LoadOp, RenderPassDesc,
    SamplerDesc, StoreOp, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use titan::macos::MetalGpu;
use titan::render2d::{Camera2D, Renderer2D, Sprite};

type Rgba = [u8; 4];
const RED: Rgba = [255, 0, 0, 255];
const GREEN: Rgba = [0, 255, 0, 255];
const BLUE: Rgba = [0, 0, 255, 255];
const WHITE: Rgba = [255, 255, 255, 255];
const BLACK: Rgba = [0, 0, 0, 255];

/// The 2×2 checker texture's texels, top row first: red, green / blue, white. Every texel and
/// every 90° turn of it is distinct, so a flipped or rotated sample shows as a wrong color.
const CHECKER: [Rgba; 4] = [RED, GREEN, BLUE, WHITE];

/// One test's GPU and renderer, drawing into `Rgba8Unorm` targets.
struct Scene {
    gpu: MetalGpu,
    renderer: Renderer2D,
}

/// A frame read back from the GPU: tightly packed `Rgba8Unorm` rows, top row first.
struct Pixels {
    width: u32,
    bytes: Vec<u8>,
}

impl Scene {
    fn new() -> Self {
        let mut gpu = MetalGpu::new_headless().expect("this Mac has a Metal GPU");
        let renderer = Renderer2D::with_color_format(&mut gpu, TextureFormat::Rgba8Unorm).unwrap();
        Self { gpu, renderer }
    }

    /// The 2×2 [`CHECKER`] texture, sampled nearest so each texel is a crisp quadrant.
    fn checker(&mut self) -> Handle<Texture> {
        let texture = self
            .gpu
            .create_texture(&TextureDesc {
                format: TextureFormat::Rgba8Unorm,
                sampler: SamplerDesc::NEAREST_CLAMP,
                ..TextureDesc::rgba8_2d(2, 2)
            })
            .unwrap();
        self.gpu
            .write_texture(texture, CHECKER.as_flattened())
            .unwrap();
        texture
    }

    /// Draws `sprites` through `camera` into a fresh `width`×`height` target cleared to
    /// `clear`, and reads it back.
    fn render(
        &mut self,
        (width, height): (u32, u32),
        clear: ClearColor,
        camera: &Camera2D,
        sprites: &[Sprite],
    ) -> Pixels {
        let target = self
            .gpu
            .create_texture(&TextureDesc {
                label: Some("golden target"),
                width,
                height,
                mip_level_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                sampler: SamplerDesc::NEAREST_CLAMP,
            })
            .unwrap();
        for &sprite in sprites {
            self.renderer.draw_sprite(sprite);
        }
        let pass = RenderPassDesc {
            label: Some("golden pass"),
            color: Some(ColorAttachment {
                target: Some(target),
                load_op: LoadOp::Clear(clear),
                store_op: StoreOp::Store,
            }),
            depth: None,
        };
        let mut frame = FrameCommands::new();
        self.renderer
            .flush(&mut self.gpu, &mut frame, pass, camera)
            .unwrap();
        self.gpu.submit(frame.view()).unwrap();
        let bytes = self.gpu.read_texture(target).unwrap();
        self.gpu.destroy_texture(target).unwrap();
        assert_eq!(bytes.len(), (width * height * 4) as usize);
        Pixels { width, bytes }
    }

    fn destroy(mut self, textures: &[Handle<Texture>]) {
        for &texture in textures {
            self.gpu.destroy_texture(texture).unwrap();
        }
        self.renderer.destroy(&mut self.gpu).unwrap();
    }
}

impl Pixels {
    fn at(&self, x: u32, y: u32) -> Rgba {
        let i = ((y * self.width + x) * 4) as usize;
        self.bytes[i..i + 4].try_into().unwrap()
    }

    /// Asserts each pixel at `(x, y)` is exactly `expected`.
    #[track_caller]
    fn assert(&self, expected: &[((u32, u32), Rgba)]) {
        for &((x, y), rgba) in expected {
            assert_eq!(self.at(x, y), rgba, "pixel ({x}, {y})");
        }
    }
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect::from_min_max(Vec2::new(x0, y0), Vec2::new(x1, y1))
}

fn color(rgba: Rgba) -> Color {
    Color::from_u8(rgba[0], rgba[1], rgba[2], rgba[3])
}

/// A camera with one pixel per world unit and the world origin at the bottom-left pixel.
fn bottom_left(width: u32, height: u32) -> Camera2D {
    Camera2D::bottom_left(Vec2::new(width as f32, height as f32))
}

#[test]
fn camera_maps_y_up_world_to_top_left_pixels_and_pans_and_zooms() {
    let mut scene = Scene::new();
    let corners = [
        Sprite::solid(rect(0.0, 0.0, 16.0, 16.0), color(RED)),
        Sprite::solid(rect(48.0, 48.0, 64.0, 64.0), color(GREEN)),
    ];
    let pixels = scene.render((64, 64), ClearColor::BLACK, &bottom_left(64, 64), &corners);
    // World +Y is up, texture rows run down: the world origin is the bottom-left pixel.
    pixels.assert(&[
        ((8, 56), RED),
        ((56, 8), GREEN),
        ((8, 8), BLACK),
        ((56, 56), BLACK),
        ((32, 32), BLACK),
    ]);

    // Centered on (10, 10) at 2 pixels per unit, world (x, y) lands on pixel
    // (32 + 2(x - 10), 32 - 2(y - 10)).
    let camera = Camera2D {
        position: Vec2::new(10.0, 10.0),
        zoom: 2.0,
        ..Camera2D::new(Vec2::new(64.0, 64.0))
    };
    let sprites = [
        // Pixels x 32..48, y 24..32: just up and right of center.
        Sprite::solid(rect(10.0, 10.0, 18.0, 14.0), color(RED)),
        // Pixels x 12..20, y 44..52.
        Sprite::solid(rect(0.0, 0.0, 4.0, 4.0), color(GREEN)),
    ];
    let pixels = scene.render((64, 64), ClearColor::BLACK, &camera, &sprites);
    pixels.assert(&[
        ((34, 26), RED),
        ((46, 30), RED),
        ((40, 22), BLACK),
        ((40, 34), BLACK),
        ((30, 28), BLACK),
        ((50, 28), BLACK),
        ((16, 48), GREEN),
        ((10, 48), BLACK),
        ((22, 48), BLACK),
        ((16, 42), BLACK),
        ((16, 54), BLACK),
    ]);
    scene.destroy(&[]);
}

#[test]
fn textured_quads_sample_upright_texels_regions_and_tints() {
    let mut scene = Scene::new();
    let checker = scene.checker();
    let tint = [255, 128, 0, 255];
    let sprites = [
        // Top-left 32×32 pixels: the whole texture, the right way up.
        Sprite::textured(rect(0.0, 32.0, 32.0, 64.0), checker),
        // Top-right: only the bottom-right (white) texel, stretched.
        Sprite::textured(rect(32.0, 32.0, 64.0, 64.0), checker).with_uv(rect(0.5, 0.5, 1.0, 1.0)),
        // Bottom-left: the whole texture times the tint.
        Sprite::textured(rect(0.0, 0.0, 32.0, 32.0), checker).with_color(color(tint)),
    ];
    let pixels = scene.render((64, 64), ClearColor::BLACK, &bottom_left(64, 64), &sprites);
    pixels.assert(&[
        ((8, 8), RED),
        ((24, 8), GREEN),
        ((8, 24), BLUE),
        ((24, 24), WHITE),
        ((40, 8), WHITE),
        ((56, 8), WHITE),
        ((40, 24), WHITE),
        ((56, 24), WHITE),
        // Red, green, blue, and white texels, each multiplied by (1, 128/255, 0).
        ((8, 40), RED),
        ((24, 40), [0, 128, 0, 255]),
        ((8, 56), BLACK),
        ((24, 56), tint),
        ((56, 56), BLACK),
    ]);
    scene.destroy(&[checker]);
}

#[test]
fn translucent_quads_blend_color_and_alpha_over_what_is_below() {
    let mut scene = Scene::new();
    let half_blue = [0, 0, 255, 128];
    let sprites = [
        Sprite::solid(rect(0.0, 0.0, 32.0, 64.0), color(half_blue)),
        // Straddles the half-transparent blue quad and the transparent clear.
        Sprite::solid(rect(16.0, 16.0, 48.0, 48.0), color([255, 0, 0, 64])),
    ];
    let pixels = scene.render(
        (64, 64),
        ClearColor::TRANSPARENT,
        &bottom_left(64, 64),
        &sprites,
    );

    // `BlendState::ALPHA_BLENDING`: color = src * src.a + dst * (1 - src.a),
    // alpha = src.a + dst.a * (1 - src.a). The blue quad lands on the clear as
    // (0, 0, 128, 128); the red quad (a = 64/255) over it gives (64, 0, 96, 160), and over
    // the clear (64, 0, 0, 64).
    for ((x, y), expected) in [((24, 32), [64, 0, 96, 160]), ((40, 32), [64, 0, 0, 64])] {
        let actual = pixels.at(x, y);
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(&a, e)| a.abs_diff(e) <= 1),
            "pixel ({x}, {y}) is {actual:?}, expected {expected:?}"
        );
    }
    pixels.assert(&[((8, 32), [0, 0, 128, 128]), ((56, 32), [0; 4])]);
    scene.destroy(&[]);
}

#[test]
fn layers_sort_first_then_submission_order_across_texture_batches() {
    let mut scene = Scene::new();
    let checker = scene.checker();
    let red_texel = rect(0.0, 0.0, 0.5, 0.5);
    let green_texel = rect(0.5, 0.0, 1.0, 0.5);
    let sprites = [
        // Top half (pixel rows 0..32): layers submitted out of order.
        Sprite::solid(rect(4.0, 32.0, 36.0, 64.0), color(RED)).with_layer(3),
        Sprite::solid(rect(28.0, 32.0, 60.0, 64.0), color(GREEN)).with_layer(1),
        Sprite::solid(rect(0.0, 32.0, 64.0, 64.0), color(WHITE)).with_layer(-1),
        // Bottom half (pixel rows 32..64), all layer 0: textured, solid, textured, so each
        // overlap crosses a texture batch boundary and the later sprite must still win.
        Sprite::textured(rect(0.0, 0.0, 32.0, 32.0), checker).with_uv(red_texel),
        Sprite::solid(rect(16.0, 0.0, 48.0, 32.0), color(BLUE)),
        Sprite::textured(rect(40.0, 0.0, 64.0, 32.0), checker).with_uv(green_texel),
    ];
    let pixels = scene.render((64, 64), ClearColor::BLACK, &bottom_left(64, 64), &sprites);
    pixels.assert(&[
        ((2, 16), WHITE),
        ((16, 16), RED),
        // Red (layer 3) covers green (layer 1) though green was submitted later.
        ((32, 16), RED),
        ((48, 16), GREEN),
        // White (layer -1) was submitted last but stays beneath.
        ((62, 16), WHITE),
        ((8, 48), RED),
        ((24, 48), BLUE),
        ((36, 48), BLUE),
        ((44, 48), GREEN),
        ((56, 48), GREEN),
    ]);
    scene.destroy(&[checker]);
}

#[test]
fn rotation_turns_quads_counter_clockwise_about_their_center() {
    let mut scene = Scene::new();
    let checker = scene.checker();
    let sprites = [
        // A 32×8 bar centered on (32, 32), turned upright to 8×32.
        Sprite::solid(rect(16.0, 28.0, 48.0, 36.0), color(RED)).with_rotation(FRAC_PI_2),
        // The checker centered on (96, 32), turned a quarter counter-clockwise: its top-left
        // red texel moves to the bottom-left, green to the top-left, and so on.
        Sprite::textured(rect(80.0, 16.0, 112.0, 48.0), checker).with_rotation(FRAC_PI_2),
    ];
    let pixels = scene.render(
        (128, 64),
        ClearColor::BLACK,
        &bottom_left(128, 64),
        &sprites,
    );
    pixels.assert(&[
        ((32, 20), RED),
        ((32, 44), RED),
        ((20, 32), BLACK),
        ((44, 32), BLACK),
        ((88, 24), GREEN),
        ((104, 24), WHITE),
        ((88, 40), RED),
        ((104, 40), BLUE),
    ]);
    scene.destroy(&[checker]);
}
