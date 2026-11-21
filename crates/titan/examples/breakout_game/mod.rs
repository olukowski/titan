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

use alloc::vec::Vec;
use level::{BrickKind, Level};
use titan::app::{App, AppAction, InputState, KeyCode, MouseButton, SurfaceSize};
use titan::core::math::sin_cos;
use titan::core::{Color, Rect, Vec2};
use titan::gpu::{ClearColor, FrameCommands, GpuDevice, GpuError, RenderPassDesc};
use titan::render2d::{Camera2D, Renderer2D};

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
                self.phase = Phase::Won;
                self.ball_velocity = Vec2::ZERO;
                return;
            }
            if self.ball.y + BALL_HALF < 0.0 {
                self.lives -= 1;
                if self.lives == 0 {
                    self.phase = Phase::Lost;
                    self.ball_velocity = Vec2::ZERO;
                } else {
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
        } else if ball.max.x > PLAY_RIGHT {
            self.ball.x = PLAY_RIGHT - BALL_HALF;
            self.ball_velocity.x = -self.ball_velocity.x.abs();
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

/// Breakout as a Titan [`App`].
#[derive(Debug)]
pub(crate) struct BreakoutApp {
    level: Level,
    game: Game,
    surface: SurfaceSize,
    renderer: Option<Renderer2D>,
}

impl BreakoutApp {
    /// A game of the built-in level.
    pub(crate) fn new() -> Self {
        Self::with_level(Level::classic())
    }

    pub(crate) fn with_level(level: Level) -> Self {
        Self {
            game: Game::new(&level),
            level,
            surface: SurfaceSize::default(),
            renderer: None,
        }
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
        self.renderer = Some(Renderer2D::new(gpu)?);
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
        let renderer = self.renderer.as_mut().expect("init runs before render");
        draw_game(renderer, &self.game, alpha.clamp(0.0, 1.0));
        renderer.flush(gpu, commands, RenderPassDesc::clear_surface(CLEAR), &camera)
    }
}

/// Queues the whole frame. Every quad is solid, so it flushes as one draw call.
fn draw_game(r: &mut Renderer2D, game: &Game, alpha: f32) {
    r.draw_rect(rect(0.0, 0.0, ARENA.x, HUD_BOTTOM), ARENA_COLOR);
    r.draw_rect(rect(0.0, 0.0, PLAY_LEFT, HUD_BOTTOM), WALL_COLOR);
    r.draw_rect(rect(PLAY_RIGHT, 0.0, ARENA.x, HUD_BOTTOM), WALL_COLOR);
    r.draw_rect(rect(0.0, PLAY_TOP, ARENA.x, HUD_BOTTOM), WALL_COLOR);

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
        draw_block(r, inset(brick.rect, 1.0), color);
    }

    let paddle_x = lerp(game.previous_paddle_x, game.paddle_x, alpha);
    draw_block(r, game.paddle_rect(paddle_x), PADDLE_COLOR);

    match game.phase {
        Phase::Ready => draw_centered(r, b"PRESS SPACE", 200.0, 3.0, HINT_COLOR),
        Phase::Playing => {}
        Phase::Won => draw_banner(r, b"YOU WIN", WON_COLOR),
        Phase::Lost => draw_banner(r, b"GAME OVER", LOST_COLOR),
    }
    if game.phase != Phase::Lost {
        let ball = game.previous_ball.lerp(game.ball, alpha);
        r.draw_rect(ball_rect(ball), BALL_COLOR);
    }

    r.draw_rect(rect(0.0, HUD_BOTTOM, ARENA.x, ARENA.y), HUD_COLOR);
    draw_number(r, game.score, Vec2::new(24.0, 590.0), 4.0, TEXT_COLOR);
    for i in 0..game.lives {
        let left = PLAY_RIGHT - 14.0 - i as f32 * 22.0;
        r.draw_rect(rect(left, 574.0, left + 14.0, 586.0), PADDLE_COLOR);
    }
}

/// A rectangle with a dark border and a light top edge.
fn draw_block(r: &mut Renderer2D, area: Rect, color: Color) {
    r.draw_rect(area, mix(color, Color::BLACK, 0.45));
    let face = inset(area, 2.0);
    r.draw_rect(face, color);
    let shine_bottom = (face.max.y - 3.0).max(face.min.y);
    let shine = Rect::from_min_max(Vec2::new(face.min.x, shine_bottom), face.max);
    r.draw_rect(shine, mix(color, Color::WHITE, 0.35));
}

fn draw_banner(r: &mut Renderer2D, title: &[u8], color: Color) {
    r.draw_rect(rect(180.0, 220.0, 620.0, 340.0), PANEL_COLOR);
    draw_centered(r, title, 320.0, 8.0, color);
    draw_centered(r, b"PRESS SPACE", 262.0, 3.0, HINT_COLOR);
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

fn mix(from: Color, to: Color, t: f32) -> Color {
    Color::rgba(
        lerp(from.r, to.r, t),
        lerp(from.g, to.g, t),
        lerp(from.b, to.b, t),
        from.a,
    )
}

// A 3 x 5 pixel font with just the characters the game writes.

/// Glyph columns plus one column of spacing.
const GLYPH_ADVANCE: f32 = 4.0;

/// Returns the glyph's 15 pixels, row by row from the top, the leftmost pixel in the highest
/// bit. Unknown characters, including space, are blank.
fn glyph(c: u8) -> u16 {
    match c {
        b'0' => 0b111_101_101_101_111,
        b'1' => 0b010_110_010_010_111,
        b'2' => 0b111_001_111_100_111,
        b'3' => 0b111_001_111_001_111,
        b'4' => 0b101_101_111_001_001,
        b'5' => 0b111_100_111_001_111,
        b'6' => 0b111_100_111_101_111,
        b'7' => 0b111_001_001_001_001,
        b'8' => 0b111_101_111_101_111,
        b'9' => 0b111_101_111_001_111,
        b'A' => 0b010_101_111_101_101,
        b'C' => 0b011_100_100_100_011,
        b'E' => 0b111_100_110_100_111,
        b'G' => 0b011_100_101_101_011,
        b'I' => 0b111_010_010_010_111,
        b'M' => 0b101_111_111_101_101,
        b'N' => 0b110_101_101_101_101,
        b'O' => 0b010_101_101_101_010,
        b'P' => 0b110_101_110_100_100,
        b'R' => 0b110_101_110_101_101,
        b'S' => 0b011_100_010_001_110,
        b'U' => 0b101_101_101_101_111,
        b'V' => 0b101_101_101_101_010,
        b'W' => 0b101_101_111_111_101,
        b'Y' => 0b101_101_010_010_010,
        _ => 0,
    }
}

/// Draws `text` with its top-left corner at `top_left`, each font pixel `pixel` world units.
fn draw_text(r: &mut Renderer2D, text: &[u8], top_left: Vec2, pixel: f32, color: Color) {
    for (i, &c) in text.iter().enumerate() {
        let bits = glyph(c);
        let left = top_left.x + i as f32 * GLYPH_ADVANCE * pixel;
        for cell in 0..15 {
            if bits & (1 << (14 - cell)) != 0 {
                let (row, column) = ((cell / 3) as f32, (cell % 3) as f32);
                let min = Vec2::new(left + column * pixel, top_left.y - (row + 1.0) * pixel);
                r.draw_quad(min, Vec2::splat(pixel), color);
            }
        }
    }
}

/// Draws `text` centered across the arena with its top at `top`.
fn draw_centered(r: &mut Renderer2D, text: &[u8], top: f32, pixel: f32, color: Color) {
    let width = (text.len() as f32 * GLYPH_ADVANCE - 1.0) * pixel;
    let top_left = Vec2::new((ARENA.x - width) / 2.0, top);
    draw_text(r, text, top_left, pixel, color);
}

fn draw_number(r: &mut Renderer2D, value: u32, top_left: Vec2, pixel: f32, color: Color) {
    // `u32::MAX` has ten digits.
    let mut digits = [0; 10];
    let mut start = digits.len();
    let mut rest = value;
    loop {
        start -= 1;
        digits[start] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    draw_text(r, &digits[start..], top_left, pixel, color);
}
