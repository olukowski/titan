//! Breakout on Titan, shared by the macOS and browser examples and the replay tests.
//!
//! It uses only `core` and `alloc`. The arena is a fixed 800 x 600 world, `+Y` up, letterboxed
//! into any surface. Simulation runs in fixed steps on `titan::core` math, so the same input
//! gives the same game on every platform.
//!
//! Controls: Left/Right or A/D (or the pointer) move the paddle; Space, Enter, or a click
//! launches the ball and restarts after a win or loss; Escape quits.

extern crate alloc;

pub(crate) mod level;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use core::fmt;
use level::{BrickKind, Level};
use titan::app::{App, AppAction, InputState, KeyCode, MouseButton, SurfaceSize};
use titan::assets::{Atlas, decode_qoi};
use titan::audio::{AudioMixer, Envelope, PlayParams, Sfx, Sound, Waveform};
use titan::core::math::sin_cos;
use titan::core::{Color, Handle, Rect, Vec2};
use titan::gpu::{
    ClearColor, FrameCommands, GpuDevice, GpuError, RenderPassDesc, SamplerDesc, Texture,
    TextureDesc, TextureFormat,
};
use titan::render2d::{Camera2D, Renderer2D, Sprite, TextStyle, create_font_texture, measure_text};

/// The world the camera always shows in full.
const ARENA: Vec2 = Vec2::new(800.0, 600.0);
const WALL: f32 = 12.0;
/// The HUD bar spans from here to the top of the arena.
const HUD_BOTTOM: f32 = 560.0;
/// The ball moves inside `[PLAY_LEFT, PLAY_RIGHT] x (-inf, PLAY_TOP]`.
const PLAY_LEFT: f32 = WALL;
const PLAY_RIGHT: f32 = ARENA.x - WALL;
const PLAY_TOP: f32 = HUD_BOTTOM - WALL;
/// The brick grid hangs from here, leaving a gap under the top wall to bounce in.
const BRICKS_TOP: f32 = 500.0;
const BRICK_AREA_HEIGHT: f32 = 240.0;
const MAX_ROW_PITCH: f32 = 24.0;

const PADDLE_Y: f32 = 40.0;
const PADDLE_HEIGHT: f32 = 14.0;
const PADDLE_TOP: f32 = PADDLE_Y + PADDLE_HEIGHT;
/// World units per second under keyboard control.
const PADDLE_SPEED: f32 = 620.0;

const BALL_SIZE: f32 = 12.0;
const BALL_HALF: f32 = BALL_SIZE / 2.0;
/// Radians from vertical when hitting a paddle edge; at most 60° keeps the ball from skimming
/// sideways for ever.
const MAX_BOUNCE_ANGLE: f32 = 1.05;
const LAUNCH_ANGLE: f32 = 0.35;
/// The ball moves at most this far between collision checks, half its size, so it cannot pass
/// through a wall, brick, or paddle at any speed.
const MAX_SUBSTEP_TRAVEL: f32 = BALL_HALF;

const CLEAR: ClearColor = ClearColor::rgba(0.01, 0.01, 0.02, 1.0);
const ARENA_COLOR: Color = Color::rgb(0.035, 0.04, 0.07);
const WALL_COLOR: Color = Color::rgb(0.2, 0.22, 0.3);
const HUD_COLOR: Color = Color::rgb(0.07, 0.08, 0.13);
const TEXT_COLOR: Color = Color::rgb(0.93, 0.94, 1.0);
const HINT_COLOR: Color = Color::rgb(0.55, 0.58, 0.7);
const PADDLE_COLOR: Color = Color::rgb(0.82, 0.86, 0.95);
const BALL_COLOR: Color = Color::WHITE;
const PANEL_COLOR: Color = Color::rgba(0.02, 0.02, 0.05, 0.88);
const WON_COLOR: Color = Color::rgb(0.4, 0.9, 0.5);
const LOST_COLOR: Color = Color::rgb(0.95, 0.35, 0.3);

// Layers, drawn lowest first; solid quads stay in layer 0. `Renderer2D` merges adjacent sprites
// that share a texture into one draw, so keeping each texture's layers together draws a frame
// with one call per texture: solid quads, then the sheet (sprites, then the banner panel over
// them), then text.
const SPRITE_LAYER: i32 = 1;
const PANEL_LAYER: i32 = 2;
const TEXT_LAYER: i32 = 3;

const SCORE_STYLE: TextStyle = TextStyle::new(3.0)
    .with_color(TEXT_COLOR)
    .with_layer(TEXT_LAYER);
const HINT_STYLE: TextStyle = TextStyle::new(2.0)
    .with_color(HINT_COLOR)
    .with_layer(TEXT_LAYER);

/// Where the game is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The ball rests on the paddle until launched.
    Ready,
    Playing,
    /// Every brick is gone.
    Won,
    /// The last ball fell.
    Lost,
}

/// Something the simulation did that makes a sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SoundEvent {
    Paddle,
    /// A side wall or the ceiling.
    Wall,
    /// A brick took a hit and survived.
    Brick,
    /// A brick was destroyed.
    BrickBreak,
    /// The ball fell with lives to spare.
    Miss,
    Won,
    Lost,
}

impl SoundEvent {
    /// Every event in declaration order, so `ALL[event as usize] == event`.
    pub(crate) const ALL: [Self; 7] = [
        Self::Paddle,
        Self::Wall,
        Self::Brick,
        Self::BrickBreak,
        Self::Miss,
        Self::Won,
        Self::Lost,
    ];
}

/// Plays each [`SoundEvent`] a step produces, after the step.
pub(crate) type Speaker = Box<dyn FnMut(SoundEvent)>;

/// The rate the sound bank renders and mixes at.
pub(crate) const SAMPLE_RATE: u32 = 48_000;

/// One rendered sound per [`SoundEvent`], indexed by `event as usize`.
#[derive(Debug)]
pub(crate) struct Sounds([Handle<Sound>; SoundEvent::ALL.len()]);

impl Sounds {
    /// Renders every effect once and registers it with `mixer`. The effects are part of the
    /// program, so a bad one panics rather than returning an error.
    pub(crate) fn new(mixer: &mut AudioMixer) -> Self {
        let rate = mixer.sample_rate();
        Self(SoundEvent::ALL.map(|event| {
            let sound = sfx(event)
                .render(rate)
                .unwrap_or_else(|e| panic!("breakout {event:?} sfx: {e}"));
            mixer.add_sound(sound)
        }))
    }

    pub(crate) fn play(&self, mixer: &mut AudioMixer, event: SoundEvent) {
        mixer.play(self.0[event as usize], PlayParams::new());
    }
}

/// Short chiptune effects; gains stay low so a burst of overlapping voices does not clip.
fn sfx(event: SoundEvent) -> Sfx {
    const BLIP: Sfx = Sfx {
        waveform: Waveform::Square,
        frequency: 440.0,
        end_frequency: None,
        duration: 0.05,
        envelope: Envelope {
            attack: 0.002,
            decay: 0.04,
            sustain: 0.5,
            release: 0.05,
        },
        gain: 0.22,
        seed: 0,
    };
    let swell = |decay, sustain, release| Envelope {
        decay,
        sustain,
        release,
        ..BLIP.envelope
    };
    match event {
        SoundEvent::Paddle => BLIP,
        SoundEvent::Wall => Sfx {
            waveform: Waveform::Triangle,
            frequency: 330.0,
            duration: 0.03,
            gain: 0.3,
            ..BLIP
        },
        SoundEvent::Brick => Sfx {
            frequency: 660.0,
            duration: 0.04,
            gain: 0.2,
            ..BLIP
        },
        SoundEvent::BrickBreak => Sfx {
            frequency: 660.0,
            end_frequency: Some(1320.0),
            duration: 0.08,
            ..BLIP
        },
        SoundEvent::Miss => Sfx {
            waveform: Waveform::Saw,
            frequency: 330.0,
            end_frequency: Some(110.0),
            duration: 0.35,
            envelope: swell(0.1, 0.6, 0.1),
            gain: 0.2,
            ..BLIP
        },
        SoundEvent::Won => Sfx {
            waveform: Waveform::Triangle,
            frequency: 440.0,
            end_frequency: Some(1320.0),
            duration: 0.6,
            envelope: swell(0.2, 0.7, 0.3),
            gain: 0.3,
            ..BLIP
        },
        SoundEvent::Lost => Sfx {
            waveform: Waveform::Saw,
            frequency: 220.0,
            end_frequency: Some(55.0),
            duration: 0.9,
            envelope: swell(0.3, 0.6, 0.4),
            gain: 0.2,
            ..BLIP
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct Brick {
    rect: Rect,
    kind: BrickKind,
    hits_left: u8,
}

/// The simulation: everything a step changes.
#[derive(Clone, Debug)]
struct Game {
    phase: Phase,
    score: u32,
    lives: u32,
    ball_speed: f32,
    paddle_half_width: f32,
    /// The paddle's center, and the ball's, after the last step and before it, for interpolating.
    paddle_x: f32,
    previous_paddle_x: f32,
    ball: Vec2,
    previous_ball: Vec2,
    ball_velocity: Vec2,
    bricks: Vec<Brick>,
    /// What happened this step, in order, for the app to play.
    sounds: Vec<SoundEvent>,
}

impl Game {
    fn new(level: &Level) -> Self {
        let pitch = Vec2::new(
            (PLAY_RIGHT - PLAY_LEFT) / level.columns() as f32,
            (BRICK_AREA_HEIGHT / level.rows() as f32).min(MAX_ROW_PITCH),
        );
        let bricks = level
            .bricks()
            .map(|(row, column, kind)| {
                let min = Vec2::new(
                    PLAY_LEFT + column as f32 * pitch.x,
                    BRICKS_TOP - (row + 1) as f32 * pitch.y,
                );
                Brick {
                    rect: Rect::from_pos_size(min, pitch),
                    kind,
                    hits_left: kind.hit_points(),
                }
            })
            .collect();
        let mut game = Self {
            phase: Phase::Ready,
            score: 0,
            lives: level.lives(),
            ball_speed: level.ball_speed(),
            paddle_half_width: level.paddle_width() / 2.0,
            paddle_x: ARENA.x / 2.0,
            previous_paddle_x: ARENA.x / 2.0,
            ball: Vec2::ZERO,
            previous_ball: Vec2::ZERO,
            ball_velocity: Vec2::ZERO,
            bricks,
            sounds: Vec::new(),
        };
        game.serve();
        game
    }

    /// Puts the ball back on the paddle, waiting for a launch.
    fn serve(&mut self) {
        self.phase = Phase::Ready;
        self.ball = self.resting_ball();
        self.previous_ball = self.ball;
        self.ball_velocity = Vec2::ZERO;
    }

    fn resting_ball(&self) -> Vec2 {
        Vec2::new(self.paddle_x, PADDLE_TOP + BALL_HALF)
    }

    fn paddle_rect(&self, x: f32) -> Rect {
        Rect::from_pos_size(
            Vec2::new(x - self.paddle_half_width, PADDLE_Y),
            Vec2::new(2.0 * self.paddle_half_width, PADDLE_HEIGHT),
        )
    }

    fn move_paddle(&mut self, input: &InputState, dt: f32) {
        let held = |a, b| f32::from(u8::from(input.is_down(a) || input.is_down(b)));
        let direction =
            held(KeyCode::ArrowRight, KeyCode::D) - held(KeyCode::ArrowLeft, KeyCode::A);
        let mut x = self.paddle_x + direction * PADDLE_SPEED * dt;
        // Follow the pointer only when it moves, so a resting pointer never fights the keys.
        let surface = input.surface_size();
        if input.pointer_delta() != Vec2::ZERO
            && let Some(camera) = fit_camera(surface)
        {
            let screen = input.pointer_position() * surface.scale_factor;
            x = camera.screen_to_world(screen).x;
        }
        let half = self.paddle_half_width;
        self.paddle_x = x.clamp(PLAY_LEFT + half, PLAY_RIGHT - half);
    }

    fn launch(&mut self) {
        let (sin, cos) = sin_cos(LAUNCH_ANGLE);
        self.ball_velocity = Vec2::new(sin, cos) * self.ball_speed;
        self.phase = Phase::Playing;
    }

    /// Moves the ball `dt` seconds, in substeps short enough that it never tunnels, resolving
    /// the X and Y axes separately so each collision reflects exactly one velocity component.
    fn advance_ball(&mut self, dt: f32) {
        let substeps = ((self.ball_velocity * dt).length() / MAX_SUBSTEP_TRAVEL) as u32 + 1;
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.ball.x += self.ball_velocity.x * h;
            self.collide_x();
            self.ball.y += self.ball_velocity.y * h;
            self.collide_y();
            if self.bricks.is_empty() {
                self.sounds.push(SoundEvent::Won);
                self.phase = Phase::Won;
                self.ball_velocity = Vec2::ZERO;
                return;
            }
            if self.ball.y + BALL_HALF < 0.0 {
                self.lives -= 1;
                if self.lives == 0 {
                    self.sounds.push(SoundEvent::Lost);
                    self.phase = Phase::Lost;
                    self.ball_velocity = Vec2::ZERO;
                } else {
                    self.sounds.push(SoundEvent::Miss);
                    self.serve();
                }
                return;
            }
        }
    }

    fn collide_x(&mut self) {
        let ball = ball_rect(self.ball);
        if ball.min.x < PLAY_LEFT {
            self.ball.x = PLAY_LEFT + BALL_HALF;
            self.ball_velocity.x = self.ball_velocity.x.abs();
            self.sounds.push(SoundEvent::Wall);
        } else if ball.max.x > PLAY_RIGHT {
            self.ball.x = PLAY_RIGHT - BALL_HALF;
            self.ball_velocity.x = -self.ball_velocity.x.abs();
            self.sounds.push(SoundEvent::Wall);
        } else if let Some(brick) = self.hit_brick(ball) {
            self.ball.x = if self.ball_velocity.x > 0.0 {
                brick.min.x - BALL_HALF
            } else {
                brick.max.x + BALL_HALF
            };
            self.ball_velocity.x = -self.ball_velocity.x;
        }
    }

    fn collide_y(&mut self) {
        let ball = ball_rect(self.ball);
        if ball.max.y > PLAY_TOP {
            self.ball.y = PLAY_TOP - BALL_HALF;
            self.ball_velocity.y = -self.ball_velocity.y.abs();
            self.sounds.push(SoundEvent::Wall);
        } else if let Some(brick) = self.hit_brick(ball) {
            self.ball.y = if self.ball_velocity.y > 0.0 {
                brick.min.y - BALL_HALF
            } else {
                brick.max.y + BALL_HALF
            };
            self.ball_velocity.y = -self.ball_velocity.y;
        } else if self.ball_velocity.y < 0.0
            && self.ball.y >= PADDLE_Y + PADDLE_HEIGHT / 2.0
            && ball.intersects(self.paddle_rect(self.paddle_x))
        {
            // The further from the paddle's center, the steeper the bounce to that side.
            let reach = self.paddle_half_width + BALL_HALF;
            let offset = ((self.ball.x - self.paddle_x) / reach).clamp(-1.0, 1.0);
            let (sin, cos) = sin_cos(offset * MAX_BOUNCE_ANGLE);
            self.ball_velocity = Vec2::new(sin, cos) * self.ball_speed;
            self.ball.y = PADDLE_TOP + BALL_HALF;
            self.sounds.push(SoundEvent::Paddle);
        }
    }

    /// Damages the first brick `ball` overlaps, scoring and removing it once destroyed, and
    /// returns its rectangle.
    fn hit_brick(&mut self, ball: Rect) -> Option<Rect> {
        let index = self.bricks.iter().position(|b| b.rect.intersects(ball))?;
        let brick = &mut self.bricks[index];
        let rect = brick.rect;
        brick.hits_left -= 1;
        if brick.hits_left == 0 {
            self.score += brick.kind.points();
            self.bricks.remove(index);
            self.sounds.push(SoundEvent::BrickBreak);
        } else {
            self.sounds.push(SoundEvent::Brick);
        }
        Some(rect)
    }
}

fn ball_rect(center: Vec2) -> Rect {
    Rect::from_pos_size(center - Vec2::splat(BALL_HALF), Vec2::splat(BALL_SIZE))
}

/// Returns a camera that shows the whole arena centered and as large as the surface allows, or
/// `None` for a zero-size (minimized) surface, which has nothing to show.
fn fit_camera(surface: SurfaceSize) -> Option<Camera2D> {
    if surface.width == 0 || surface.height == 0 {
        return None;
    }
    let viewport = Vec2::new(surface.width as f32, surface.height as f32);
    Some(Camera2D {
        viewport,
        zoom: (viewport.x / ARENA.x).min(viewport.y / ARENA.y),
        ..Camera2D::bottom_left(ARENA)
    })
}

/// The renderer and everything it draws from, made once in [`App::init`].
#[derive(Debug)]
struct Graphics {
    renderer: Renderer2D,
    font: Handle<Texture>,
    sheet: Handle<Texture>,
    /// UVs of the `sheet.atlas` regions. All but the panel are gray bevels tinted when drawn.
    brick: Rect,
    paddle: Rect,
    ball: Rect,
    /// Solid white, so the banner panel draws from the sheet and joins its draw call.
    panel: Rect,
}

impl Graphics {
    /// Decodes the embedded sprite sheet and uploads it and the font. The sheet is part of the
    /// program, so a bad one panics rather than returning an error.
    fn new(gpu: &mut dyn GpuDevice) -> Result<Self, GpuError> {
        let image = decode_qoi(include_bytes!("sheet.qoi"))
            .unwrap_or_else(|e| panic!("breakout sheet.qoi: {e}"));
        let atlas = Atlas::parse(include_str!("sheet.atlas"))
            .unwrap_or_else(|e| panic!("breakout sheet.atlas: {e}"));
        assert_eq!(
            (atlas.width(), atlas.height()),
            (image.width(), image.height()),
            "breakout sheet.atlas: size differs from sheet.qoi"
        );
        let uv = |name| {
            atlas
                .uv(name)
                .unwrap_or_else(|| panic!("breakout sheet.atlas: no `{name}` region"))
        };
        let renderer = Renderer2D::new(gpu)?;
        let font = create_font_texture(gpu)?;
        let sheet = gpu.create_texture(&TextureDesc {
            label: Some("breakout sheet"),
            // Linear like the font, so a texel's gray scales its sprite's tint as is.
            format: TextureFormat::Rgba8Unorm,
            sampler: SamplerDesc::NEAREST_CLAMP,
            ..TextureDesc::rgba8_2d(image.width(), image.height())
        })?;
        gpu.write_texture(sheet, image.as_bytes())?;
        Ok(Self {
            renderer,
            font,
            sheet,
            brick: uv("brick"),
            paddle: uv("paddle"),
            ball: uv("ball"),
            panel: uv("panel"),
        })
    }

    fn draw_sheet(&mut self, rect: Rect, uv: Rect, color: Color, layer: i32) {
        self.renderer.draw_sprite(
            Sprite::textured(rect, self.sheet)
                .with_uv(uv)
                .with_color(color)
                .with_layer(layer),
        );
    }

    /// Draws `text` centered across the arena with its top at `top`.
    fn draw_centered(&mut self, text: &str, top: f32, style: &TextStyle) {
        let left = (ARENA.x - measure_text(text, style).x) / 2.0;
        let list = self.renderer.draw_list_mut();
        list.draw_text(self.font, text, Vec2::new(left, top), style);
    }
}

/// Breakout as a Titan [`App`].
pub(crate) struct BreakoutApp {
    level: Level,
    game: Game,
    surface: SurfaceSize,
    graphics: Option<Graphics>,
    speaker: Option<Speaker>,
}

impl fmt::Debug for BreakoutApp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BreakoutApp")
            .field("level", &self.level)
            .field("game", &self.game)
            .finish_non_exhaustive()
    }
}

impl BreakoutApp {
    /// A muted game of the built-in level.
    pub(crate) fn new() -> Self {
        Self::with_level(Level::classic())
    }

    pub(crate) fn with_level(level: Level) -> Self {
        Self {
            game: Game::new(&level),
            level,
            surface: SurfaceSize::default(),
            graphics: None,
            speaker: None,
        }
    }

    /// Plays this game's sounds through `speaker`, or mutes it with `None`.
    pub(crate) fn with_speaker(self, speaker: Option<Speaker>) -> Self {
        Self { speaker, ..self }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Snapshot {
        let game = &self.game;
        Snapshot {
            phase: game.phase,
            score: game.score,
            lives: game.lives,
            bricks_remaining: game.bricks.len(),
            ball_position: game.ball,
            ball_velocity: game.ball_velocity,
            paddle_x: game.paddle_x,
        }
    }
}

/// The game state tests inspect.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Snapshot {
    pub(crate) phase: Phase,
    pub(crate) score: u32,
    pub(crate) lives: u32,
    pub(crate) bricks_remaining: usize,
    /// The ball's center.
    pub(crate) ball_position: Vec2,
    pub(crate) ball_velocity: Vec2,
    pub(crate) paddle_x: f32,
}

impl App for BreakoutApp {
    fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
        self.graphics = Some(Graphics::new(gpu)?);
        Ok(())
    }

    fn resize(&mut self, _: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
        self.surface = size;
        Ok(())
    }

    fn step(&mut self, input: &InputState, dt: f32) -> AppAction {
        if input.close_requested() || input.just_pressed(KeyCode::Escape) {
            return AppAction::Exit;
        }
        let pressed = input.just_pressed(KeyCode::Space)
            || input.just_pressed(KeyCode::Enter)
            || input.button_just_pressed(MouseButton::Left);
        let game = &mut self.game;
        game.previous_ball = game.ball;
        game.previous_paddle_x = game.paddle_x;
        match game.phase {
            Phase::Ready => {
                game.move_paddle(input, dt);
                game.ball = game.resting_ball();
                if pressed {
                    game.launch();
                }
            }
            Phase::Playing => {
                game.move_paddle(input, dt);
                game.advance_ball(dt);
            }
            Phase::Won | Phase::Lost => {
                if pressed {
                    self.game = Game::new(&self.level);
                }
            }
        }
        match &mut self.speaker {
            Some(speaker) => self.game.sounds.drain(..).for_each(speaker),
            None => self.game.sounds.clear(),
        }
        AppAction::Continue
    }

    fn render(
        &mut self,
        gpu: &mut dyn GpuDevice,
        commands: &mut FrameCommands,
        alpha: f32,
    ) -> Result<(), GpuError> {
        let Some(camera) = fit_camera(self.surface) else {
            return Ok(());
        };
        let graphics = self.graphics.as_mut().expect("init runs before render");
        draw_game(graphics, &self.game, alpha.clamp(0.0, 1.0));
        let pass = RenderPassDesc::clear_surface(CLEAR);
        graphics.renderer.flush(gpu, commands, pass, &camera)
    }
}

/// Queues the whole frame: one draw call each for solid quads, the sheet, and the font.
fn draw_game(g: &mut Graphics, game: &Game, alpha: f32) {
    let r = &mut g.renderer;
    r.draw_rect(rect(0.0, 0.0, ARENA.x, HUD_BOTTOM), ARENA_COLOR);
    r.draw_rect(rect(0.0, 0.0, PLAY_LEFT, HUD_BOTTOM), WALL_COLOR);
    r.draw_rect(rect(PLAY_RIGHT, 0.0, ARENA.x, HUD_BOTTOM), WALL_COLOR);
    r.draw_rect(rect(0.0, PLAY_TOP, ARENA.x, HUD_BOTTOM), WALL_COLOR);
    r.draw_rect(rect(0.0, HUD_BOTTOM, ARENA.x, ARENA.y), HUD_COLOR);

    for brick in &game.bricks {
        let color = match (brick.kind, brick.hits_left) {
            (BrickKind::Red, _) => Color::rgb(0.91, 0.3, 0.24),
            (BrickKind::Orange, _) => Color::rgb(0.95, 0.56, 0.2),
            (BrickKind::Yellow, _) => Color::rgb(0.96, 0.83, 0.27),
            (BrickKind::Green, _) => Color::rgb(0.36, 0.78, 0.4),
            (BrickKind::Blue, _) => Color::rgb(0.3, 0.55, 0.95),
            (BrickKind::Silver, 2..) => Color::rgb(0.8, 0.82, 0.88),
            (BrickKind::Silver, _) => Color::rgb(0.48, 0.5, 0.57),
        };
        g.draw_sheet(inset(brick.rect, 1.0), g.brick, color, SPRITE_LAYER);
    }

    let paddle_x = lerp(game.previous_paddle_x, game.paddle_x, alpha);
    g.draw_sheet(
        game.paddle_rect(paddle_x),
        g.paddle,
        PADDLE_COLOR,
        SPRITE_LAYER,
    );
    if game.phase != Phase::Lost {
        let ball = game.previous_ball.lerp(game.ball, alpha);
        g.draw_sheet(ball_rect(ball), g.ball, BALL_COLOR, SPRITE_LAYER);
    }
    for i in 0..game.lives {
        let left = PLAY_RIGHT - BALL_SIZE - i as f32 * 22.0;
        let icon = rect(left, 574.0, left + BALL_SIZE, 574.0 + BALL_SIZE);
        g.draw_sheet(icon, g.ball, BALL_COLOR, SPRITE_LAYER);
    }

    match game.phase {
        Phase::Ready => g.draw_centered("PRESS SPACE", 200.0, &HINT_STYLE),
        Phase::Playing => {}
        Phase::Won => draw_banner(g, "YOU WIN", WON_COLOR),
        Phase::Lost => draw_banner(g, "GAME OVER", LOST_COLOR),
    }

    let score = format!("{}", game.score);
    let list = g.renderer.draw_list_mut();
    list.draw_text(g.font, &score, Vec2::new(24.0, 590.0), &SCORE_STYLE);
}

fn draw_banner(g: &mut Graphics, title: &str, color: Color) {
    g.draw_sheet(
        rect(180.0, 220.0, 620.0, 340.0),
        g.panel,
        PANEL_COLOR,
        PANEL_LAYER,
    );
    let style = TextStyle::new(5.0).with_color(color).with_layer(TEXT_LAYER);
    g.draw_centered(title, 320.0, &style);
    g.draw_centered("PRESS SPACE", 262.0, &HINT_STYLE);
}

fn rect(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Rect {
    Rect::from_min_max(Vec2::new(min_x, min_y), Vec2::new(max_x, max_y))
}

/// Shrinks `area` by `by` on every side, but never past its center.
fn inset(area: Rect, by: f32) -> Rect {
    let half = area.size() * 0.5;
    let by = Vec2::splat(by.min(half.x).min(half.y));
    Rect::from_min_max(area.min + by, area.max - by)
}

fn lerp(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}
