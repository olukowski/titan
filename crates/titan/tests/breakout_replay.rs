//! Breakout's level format, its gameplay rules, its generated sprite sheet, and a 1,260-frame
//! headless replay that must reproduce the same game and the same GPU frames every time.

#[path = "../examples/breakout_game/mod.rs"]
mod breakout;

use breakout::level::{Level, LevelError, LevelErrorKind};
use breakout::{BreakoutApp, Phase, Snapshot};
use titan::app::{
    Event, EventLog, HeadlessConfig, HeadlessRunner, KeyCode, MouseButton, SurfaceSize,
};
use titan::assets::{Atlas, Image, encode_qoi};
use titan::core::Vec2;
use titan::gpu::{DrawCmd, RecordedFrame, RenderCmd};

const DT: f32 = 1.0 / 60.0;

fn level(grid: &str) -> Level {
    let text = format!("name: Test\nlives: 2\nball_speed: 360\npaddle_width: 112\n---\n{grid}");
    Level::parse(&text).unwrap()
}

fn run(app: BreakoutApp) -> HeadlessRunner<BreakoutApp> {
    HeadlessRunner::new(app, HeadlessConfig::default()).unwrap()
}

fn tick(runner: &mut HeadlessRunner<BreakoutApp>, events: &[Event]) -> Snapshot {
    runner.tick_frame(DT, events).unwrap();
    runner.app().snapshot()
}

/// Presses and releases `key` over two frames.
fn tap(runner: &mut HeadlessRunner<BreakoutApp>, key: KeyCode) -> Snapshot {
    tick(runner, &[Event::KeyDown(key)]);
    tick(runner, &[Event::KeyUp(key)])
}

/// Ticks idle frames until `done` holds, failing after `limit` frames.
fn tick_until(
    runner: &mut HeadlessRunner<BreakoutApp>,
    limit: usize,
    done: impl Fn(&Snapshot) -> bool,
) -> Snapshot {
    for _ in 0..limit {
        let state = tick(runner, &[]);
        if done(&state) {
            return state;
        }
    }
    panic!(
        "not reached in {limit} frames: {:?}",
        runner.app().snapshot()
    );
}

/// Asserts the frame is one `Renderer2D` pass: the sprite pipeline and indexed draws of whole
/// quads, at most one per texture (solid quads, the sprite sheet, and the font).
fn assert_one_draw_per_texture(frame: &RecordedFrame) {
    assert_eq!(frame.passes.len(), 1);
    let commands = &frame.passes[0].commands;
    assert!(
        matches!(commands[0], RenderCmd::SetPipeline(_)),
        "{commands:?}"
    );
    let mut bound = None;
    let mut textures = Vec::new();
    for command in commands {
        match *command {
            RenderCmd::SetTexture { texture, .. } => bound = Some(texture),
            RenderCmd::Draw(DrawCmd::Indexed { index_count, .. }) => {
                assert!(index_count > 0 && index_count % 6 == 0, "{index_count}");
                let texture = bound.expect("a draw before any texture is bound");
                assert!(!textures.contains(&texture), "{texture:?} drawn twice");
                textures.push(texture);
            }
            RenderCmd::Draw(draw) => panic!("expected indexed draws, got {draw:?}"),
            _ => {}
        }
    }
    assert!((1..=3).contains(&textures.len()), "{commands:?}");
}

// Level format.

#[test]
fn classic_level_parses_and_round_trips() {
    let level = Level::classic();
    assert_eq!(level.name(), "Classic");
    assert_eq!(
        (level.lives(), level.ball_speed(), level.paddle_width()),
        (3, 360.0, 112.0)
    );
    assert_eq!((level.rows(), level.columns()), (6, 10));
    assert_eq!(level.bricks().count(), 60);
    assert_eq!(Level::parse(&level.to_string()), Ok(level));
}

#[test]
fn comments_blank_lines_and_custom_grids_parse_and_round_trip() {
    let text = "\
# A custom level.

  name :  Two words
# Keys may come in any order.
paddle_width: 64.5
ball_speed:420.25
lives: 9

---

S..S
.RB.
....

";
    let level = Level::parse(text).unwrap();
    assert_eq!(level.name(), "Two words");
    assert_eq!(
        (level.lives(), level.ball_speed(), level.paddle_width()),
        (9, 420.25, 64.5)
    );
    assert_eq!((level.rows(), level.columns()), (3, 4));
    let bricks: Vec<_> = level
        .bricks()
        .map(|(row, column, _)| (row, column))
        .collect();
    assert_eq!(bricks, [(0, 0), (0, 3), (1, 1), (1, 2)]);
    let canonical = level.to_string();
    assert_eq!(
        canonical,
        "name: Two words\nlives: 9\nball_speed: 420.25\npaddle_width: 64.5\n\n---\nS..S\n.RB.\n....\n"
    );
    assert_eq!(Level::parse(&canonical), Ok(level));
}

#[test]
fn the_largest_grid_parses_round_trips_and_renders() {
    let grid = "SROYGB..".repeat(4) + "\n";
    let level = level(&grid.repeat(24));
    assert_eq!((level.rows(), level.columns()), (24, 32));
    assert_eq!(level.bricks().count(), 24 * 24);
    assert_eq!(Level::parse(&level.to_string()), Ok(level.clone()));

    let mut runner = run(BreakoutApp::with_level(level));
    tick(&mut runner, &[]);
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
    tap(&mut runner, KeyCode::Space);
    tick_until(&mut runner, 120, |s| s.score > 0);
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
}

#[test]
fn every_level_error_names_its_line() {
    const BALL_SPEED_RULE: &str = "ball_speed must be in 60..=1200";
    const HEADER: &str = "name: X\nlives: 3\nball_speed: 360\npaddle_width: 112\n";
    let with_grid = |grid: &str| format!("{HEADER}---\n{grid}");
    let cases = [
        (
            format!("# c\n\nname X\n{HEADER}---\nR"),
            3,
            LevelErrorKind::ExpectedKeyValue,
        ),
        (
            format!("{HEADER}speed: 1\n---\nR"),
            5,
            LevelErrorKind::UnknownKey("speed".into()),
        ),
        (
            format!("{HEADER}lives: 2\n---\nR"),
            5,
            LevelErrorKind::DuplicateKey("lives"),
        ),
        (
            "name: X\nlives: 3\nball_speed: 360\n\n---\nR".into(),
            5,
            LevelErrorKind::MissingKey("paddle_width"),
        ),
        (
            "lives: 0".into(),
            1,
            LevelErrorKind::InvalidValue("lives must be an integer in 1..=9"),
        ),
        (
            "lives: 10".into(),
            1,
            LevelErrorKind::InvalidValue("lives must be an integer in 1..=9"),
        ),
        (
            "name:   ".into(),
            1,
            LevelErrorKind::InvalidValue("name must not be empty"),
        ),
        (
            "\nball_speed: -1".into(),
            2,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "ball_speed: inf".into(),
            1,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "ball_speed: 1e-7".into(),
            1,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "name: X\nball_speed: 59".into(),
            2,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "ball_speed: 1201".into(),
            1,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "ball_speed: 2e12".into(),
            1,
            LevelErrorKind::InvalidValue(BALL_SPEED_RULE),
        ),
        (
            "paddle_width: NaN".into(),
            1,
            LevelErrorKind::InvalidValue("paddle_width must be in 16..=400"),
        ),
        (
            "paddle_width: 401".into(),
            1,
            LevelErrorKind::InvalidValue("paddle_width must be in 16..=400"),
        ),
        (HEADER.into(), 4, LevelErrorKind::MissingSeparator),
        (String::new(), 1, LevelErrorKind::MissingSeparator),
        (with_grid("\n\n"), 5, LevelErrorKind::EmptyGrid),
        // Grid rows start on line 6.
        (with_grid(&"R".repeat(33)), 6, LevelErrorKind::GridTooLarge),
        (
            with_grid(&"R\n".repeat(25)),
            30,
            LevelErrorKind::GridTooLarge,
        ),
        // A 5,000-wide first row over 4,999 narrow rows is rejected before any grid is allocated.
        (
            with_grid(&format!("{}\n{}", "R".repeat(5000), "R\n".repeat(4999))),
            6,
            LevelErrorKind::GridTooLarge,
        ),
        (
            with_grid("RR\nR\n"),
            7,
            LevelErrorKind::JaggedRow {
                expected: 2,
                actual: 1,
            },
        ),
        (
            with_grid("RR\n\nRR"),
            7,
            LevelErrorKind::JaggedRow {
                expected: 2,
                actual: 0,
            },
        ),
        (
            with_grid(&format!("RR\nRR\n{}", "R".repeat(5000))),
            8,
            LevelErrorKind::JaggedRow {
                expected: 2,
                actual: 5000,
            },
        ),
        (with_grid("R.\n.X"), 7, LevelErrorKind::UnknownBrick('X')),
        (with_grid("..\n.."), 5, LevelErrorKind::NoBricks),
    ];
    for (text, line, kind) in cases {
        let expected = LevelError { line, kind };
        assert_eq!(Level::parse(&text), Err(expected.clone()), "{text:?}");
        assert!(expected.to_string().starts_with(&format!("line {line}: ")));
    }
}

// Sprite sheet.

/// How the generator draws each `sheet.atlas` region.
enum Shape {
    /// A gray block with a dark border, a light top shine two texels deep, and corners rounded
    /// to `radius` texels.
    Bevel {
        radius: i64,
    },
    Solid,
}

const SHEET_SIZE: (u32, u32) = (48, 22);
const SHAPES: [(&str, Shape); 4] = [
    ("brick", Shape::Bevel { radius: 2 }),
    ("ball", Shape::Bevel { radius: 6 }),
    ("paddle", Shape::Bevel { radius: 4 }),
    ("panel", Shape::Solid),
];

/// Draws every region of `atlas` into a transparent sheet, texel by texel.
fn sheet_image(atlas: &Atlas) -> Image {
    let (width, height) = SHEET_SIZE;
    let mut image = Image::new(width, height, vec![0; (width * height * 4) as usize]).unwrap();
    for (name, shape) in SHAPES {
        let region = atlas
            .region(name)
            .unwrap_or_else(|| panic!("no `{name}` region"));
        let (w, h) = (i64::from(region.width), i64::from(region.height));
        // Doubled coordinates keep texel centers, and so the shape, in exact integers.
        let inside = |x: i64, y: i64| {
            let Shape::Bevel { radius } = shape else {
                return true;
            };
            let (cx, cy) = (2 * x + 1, 2 * y + 1);
            let dx = cx - cx.clamp(2 * radius, 2 * (w - radius));
            let dy = cy - cy.clamp(2 * radius, 2 * (h - radius));
            (0..w).contains(&x) && (0..h).contains(&y) && dx * dx + dy * dy <= 4 * radius * radius
        };
        for y in 0..h {
            for x in 0..w {
                let gray = |v| [v, v, v, 255];
                let texel = match shape {
                    Shape::Solid => gray(255),
                    Shape::Bevel { .. } if !inside(x, y) => [0; 4],
                    _ if [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .any(|(dx, dy)| !inside(x + dx, y + dy)) =>
                    {
                        gray(110)
                    }
                    _ if !inside(x, y - 3) => gray(255),
                    _ => gray(200),
                };
                let (x, y) = (region.x + x as u32, region.y + y as u32);
                assert!(
                    image.set_pixel(x, y, texel),
                    "`{name}` is outside the sheet"
                );
            }
        }
    }
    image
}

#[test]
fn the_sprite_sheet_is_its_generators_output() {
    let atlas = Atlas::parse(include_str!("../examples/breakout_game/sheet.atlas")).unwrap();
    assert_eq!((atlas.width(), atlas.height()), SHEET_SIZE);
    let qoi = encode_qoi(&sheet_image(&atlas));
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/breakout_game/sheet.qoi"
    );
    if std::env::var_os("TITAN_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, &qoi).unwrap();
    }
    let committed = std::fs::read(path).unwrap();
    assert!(
        committed == qoi,
        "sheet.qoi differs from the generator; rerun with TITAN_BLESS=1"
    );
}

// Gameplay.

#[test]
fn keys_move_the_paddle_and_walls_stop_it() {
    let mut runner = run(BreakoutApp::new());
    assert_eq!(runner.app().snapshot().paddle_x, 400.0);
    let start = tick(&mut runner, &[Event::KeyDown(KeyCode::ArrowRight)]);
    assert!(start.paddle_x > 400.0);
    // The resting ball rides along.
    assert_eq!(start.ball_position.x, start.paddle_x);
    // Half a 112-wide paddle from the 12-wide right wall.
    let right = tick_until(&mut runner, 120, |s| s.paddle_x == 732.0);
    tick(&mut runner, &[Event::KeyUp(KeyCode::ArrowRight)]);
    assert_eq!(tick(&mut runner, &[]).paddle_x, right.paddle_x);

    tick(&mut runner, &[Event::KeyDown(KeyCode::A)]);
    tick_until(&mut runner, 120, |s| s.paddle_x == 68.0);
    let both = tick(
        &mut runner,
        &[
            Event::KeyUp(KeyCode::A),
            Event::KeyDown(KeyCode::D),
            Event::KeyDown(KeyCode::ArrowLeft),
        ],
    );
    assert_eq!(both.paddle_x, 68.0, "opposite keys cancel");
}

#[test]
fn pointer_moves_the_paddle_through_the_letterboxed_camera() {
    let mut runner = run(BreakoutApp::new());
    // A Retina surface: logical points are half the pixels and the arena fills it at zoom 2.
    tick(
        &mut runner,
        &[Event::Resize(SurfaceSize::new(1600, 1200, 2.0))],
    );
    let placed = tick(&mut runner, &[Event::PointerMove { x: 100.0, y: 300.0 }]);
    assert_eq!(placed.paddle_x, 400.0, "placing the pointer is not a move");
    let moved = tick(&mut runner, &[Event::PointerMove { x: 200.0, y: 300.0 }]);
    assert_eq!(moved.paddle_x, 200.0);

    // A 1600 x 600 surface shows the arena at zoom 1 with 400-pixel bars on each side.
    tick(
        &mut runner,
        &[Event::Resize(SurfaceSize::new(1600, 600, 1.0))],
    );
    let centered = tick(&mut runner, &[Event::PointerMove { x: 700.0, y: 300.0 }]);
    assert_eq!(centered.paddle_x, 300.0);
    let clamped = tick(&mut runner, &[Event::PointerMove { x: 0.0, y: 300.0 }]);
    assert_eq!(clamped.paddle_x, 68.0);

    // A resting pointer does not override the keys.
    tick(&mut runner, &[Event::KeyDown(KeyCode::ArrowRight)]);
    assert!(tick(&mut runner, &[]).paddle_x > 68.0);
}

#[test]
fn space_enter_and_click_each_launch_the_ball() {
    let launches: [&[Event]; 3] = [
        &[Event::KeyDown(KeyCode::Space)],
        &[Event::KeyDown(KeyCode::Enter)],
        &[Event::PointerDown {
            button: MouseButton::Left,
        }],
    ];
    for events in launches {
        let mut runner = run(BreakoutApp::new());
        assert_eq!(tick(&mut runner, &[]).phase, Phase::Ready);
        let launched = tick(&mut runner, events);
        assert_eq!(launched.phase, Phase::Playing, "{events:?}");
        assert!(launched.ball_velocity.y > 0.0);
        assert!((launched.ball_velocity.length() - 360.0).abs() < 1e-3);
        let flying = tick(&mut runner, &[]);
        assert!(flying.ball_position.y > launched.ball_position.y);
    }
}

#[test]
fn the_slowest_and_fastest_ball_speeds_play() {
    for speed in [60.0, 1200.0] {
        let text = format!("name: T\nlives: 2\nball_speed: {speed}\npaddle_width: 112\n---\nR");
        let mut runner = run(BreakoutApp::with_level(Level::parse(&text).unwrap()));
        let launched = tap(&mut runner, KeyCode::Space);
        assert_eq!(launched.phase, Phase::Playing);
        assert!((launched.ball_velocity.length() - speed).abs() < 1e-3);
        // Free flight covers one 60 Hz step of travel per frame.
        let flying = tick(&mut runner, &[]);
        let travel = (flying.ball_position - launched.ball_position).length();
        assert!(
            (travel - speed * DT).abs() < 1e-3,
            "{speed}: moved {travel}"
        );
        // The ball reaches the brick 400 units up.
        let won = tick_until(&mut runner, 600, |s| s.phase == Phase::Won);
        assert_eq!(won.score, 50, "{speed}");
    }
}

#[test]
fn bricks_score_when_destroyed_and_silver_takes_two_hits() {
    let mut runner = run(BreakoutApp::new());
    tap(&mut runner, KeyCode::Space);
    let hit = tick_until(&mut runner, 120, |s| s.score > 0);
    // The ball reaches the bottom blue row first.
    assert_eq!((hit.score, hit.bricks_remaining), (10, 59));
    assert!(hit.ball_velocity.y < 0.0, "it bounces back down");

    // One full-width silver brick: whatever the ball does, it hits that brick on the way up.
    let mut runner = run(BreakoutApp::with_level(level("S")));
    tap(&mut runner, KeyCode::Space);
    let damaged = tick_until(&mut runner, 120, |s| s.ball_velocity.y < 0.0);
    assert_eq!((damaged.score, damaged.bricks_remaining), (0, 1));
    // The untouched paddle misses the rebound; the second serve finishes the brick.
    let fallen = tick_until(&mut runner, 600, |s| s.phase != Phase::Playing);
    assert_eq!(
        (fallen.phase, fallen.lives, fallen.bricks_remaining),
        (Phase::Ready, 1, 1)
    );
    tap(&mut runner, KeyCode::Space);
    let won = tick_until(&mut runner, 120, |s| s.phase != Phase::Playing);
    assert_eq!(
        (won.phase, won.score, won.bricks_remaining),
        (Phase::Won, 100, 0)
    );
    assert_eq!(won.ball_velocity, Vec2::ZERO);
}

#[test]
fn clearing_the_level_wins_and_space_restarts() {
    let mut runner = run(BreakoutApp::with_level(level("B")));
    tap(&mut runner, KeyCode::Space);
    let won = tick_until(&mut runner, 120, |s| s.phase == Phase::Won);
    assert_eq!((won.score, won.bricks_remaining, won.lives), (10, 0, 2));
    // The banner panel and its text still draw once per texture: solid, sheet, font.
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
    // Winning freezes the game until a restart.
    assert_eq!(
        tick(&mut runner, &[Event::KeyDown(KeyCode::ArrowLeft)]),
        won
    );
    let restarted = tap(&mut runner, KeyCode::Space);
    assert_eq!(
        (restarted.phase, restarted.score, restarted.bricks_remaining),
        (Phase::Ready, 0, 1)
    );
}

#[test]
fn falling_balls_cost_lives_until_the_game_is_lost_and_enter_restarts() {
    // The paddle runs to the far left after each launch, so the ball falls past it.
    let mut runner = run(BreakoutApp::with_level(level("B.........")));
    for lives_left in [1, 0] {
        tap(&mut runner, KeyCode::Space);
        tick(&mut runner, &[Event::KeyDown(KeyCode::ArrowLeft)]);
        let fallen = tick_until(&mut runner, 600, |s| s.phase != Phase::Playing);
        tick(&mut runner, &[Event::KeyUp(KeyCode::ArrowLeft)]);
        assert_eq!(fallen.lives, lives_left);
        let expected = if lives_left == 0 {
            Phase::Lost
        } else {
            Phase::Ready
        };
        assert_eq!(fallen.phase, expected);
    }
    let lost = runner.app().snapshot();
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
    // A lost game ignores the paddle keys.
    assert_eq!(tick(&mut runner, &[Event::KeyDown(KeyCode::D)]), lost);
    let restarted = tap(&mut runner, KeyCode::Enter);
    assert_eq!(
        (restarted.phase, restarted.lives, restarted.score),
        (Phase::Ready, 2, 0)
    );
}

#[test]
fn escape_or_closing_the_window_exits() {
    for event in [Event::KeyDown(KeyCode::Escape), Event::CloseRequested] {
        let mut runner = run(BreakoutApp::new());
        let outcome = runner.tick_frame(DT, &[event]).unwrap();
        assert!(outcome.exited && !outcome.rendered, "{event:?}");
    }
}

#[test]
fn zero_size_surface_renders_nothing_without_panicking() {
    let mut runner = run(BreakoutApp::new());
    tick(&mut runner, &[]);
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
    for size in [SurfaceSize::new(0, 0, 1.0), SurfaceSize::new(640, 0, 2.0)] {
        tick(
            &mut runner,
            &[Event::Resize(size), Event::KeyDown(KeyCode::Space)],
        );
        assert!(runner.gpu().last_frame().unwrap().passes.is_empty());
    }
    tick(
        &mut runner,
        &[Event::Resize(SurfaceSize::new(320, 200, 1.0))],
    );
    assert_one_draw_per_texture(runner.gpu().last_frame().unwrap());
    assert_eq!(runner.app().snapshot().phase, Phase::Playing);
}

// Deterministic replay.

const REPLAY_FRAMES: usize = 1260;

/// Plays the classic level for `REPLAY_FRAMES` frames with a simple autopilot that chases the
/// ball with the arrow keys, then A/D, serves with Space, and also moves the pointer and resizes
/// the surface. Returns the recorded input, the finished run, and how often the ball bounced off
/// the paddle and the side walls.
fn record_autopilot() -> (EventLog, HeadlessRunner<BreakoutApp>, u32, u32) {
    let mut log = EventLog::new();
    let mut runner = run(BreakoutApp::new());
    let (mut paddle_bounces, mut side_bounces) = (0, 0);
    let mut held: Option<KeyCode> = None;
    for frame in 0..REPLAY_FRAMES {
        let before = runner.app().snapshot();
        let mut events = Vec::new();
        // Every seventh frame is long, so the clock interpolates between steps.
        let dt = if frame % 7 == 6 { 1.0 / 45.0 } else { DT };
        let (left, right) = if frame < REPLAY_FRAMES / 2 {
            (KeyCode::ArrowLeft, KeyCode::ArrowRight)
        } else {
            (KeyCode::A, KeyCode::D)
        };
        let target = before.ball_position.x;
        let want = if before.paddle_x < target - 20.0 {
            Some(right)
        } else if before.paddle_x > target + 20.0 {
            Some(left)
        } else {
            None
        };
        if want != held {
            events.extend(held.map(Event::KeyUp));
            events.extend(want.map(Event::KeyDown));
            held = want;
        }
        match frame {
            // Serve once the ball has rested a moment, and again after every lost ball.
            _ if before.phase != Phase::Playing && frame % 30 == 0 => {
                events.push(Event::KeyDown(KeyCode::Space));
            }
            _ if frame % 30 == 1 => events.push(Event::KeyUp(KeyCode::Space)),
            // Drag the pointer to the paddle's current place: exercised but harmless.
            310 => events.push(Event::PointerMove {
                x: before.paddle_x,
                y: 500.0,
            }),
            311 => events.push(Event::PointerMove {
                x: before.paddle_x + 0.5,
                y: 500.0,
            }),
            // A Retina surface of the same logical size, then a wide letterboxed one.
            500 => events.push(Event::Resize(SurfaceSize::new(1600, 1200, 2.0))),
            905 => events.push(Event::Resize(SurfaceSize::new(1280, 600, 1.0))),
            _ => {}
        }
        runner.tick_frame(dt, &events).unwrap();
        log.push_frame(dt, events);

        let after = runner.app().snapshot();
        if before.ball_velocity.y < 0.0
            && after.ball_velocity.y > 0.0
            && after.ball_position.y < 100.0
        {
            paddle_bounces += 1;
        }
        if before.ball_velocity.x * after.ball_velocity.x < 0.0
            && !(30.0..770.0).contains(&after.ball_position.x)
        {
            side_bounces += 1;
        }
    }
    (log, runner, paddle_bounces, side_bounces)
}

#[test]
fn a_recorded_game_replays_identically_from_its_text_log() {
    let (recorded_log, recorded, paddle_bounces, side_bounces) = record_autopilot();
    let text = recorded_log.to_string();
    let log = EventLog::parse(&text).unwrap();
    assert_eq!(log, recorded_log);
    assert_eq!(log.len(), REPLAY_FRAMES);
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    for needle in [
        "key_down Space",
        "key_down ArrowLeft",
        "key_down ArrowRight",
        "key_down A",
        "key_down D",
        "resize 1600 1200 2",
    ] {
        assert!(lines.contains(&needle), "the log has no `{needle}` line");
    }
    assert!(lines.iter().any(|l| l.starts_with("pointer_move ")));

    let replay = || {
        let mut runner = run(BreakoutApp::new());
        let summary = runner.replay(&log).unwrap();
        (summary, runner)
    };
    let (first_summary, first) = replay();
    let (second_summary, second) = replay();

    assert_eq!(first_summary, second_summary);
    assert_eq!(first_summary.frames_run, REPLAY_FRAMES);
    assert!(first_summary.total_steps >= 1200, "{first_summary:?}");
    assert!(!first_summary.exited);
    let state = first.app().snapshot();
    assert_eq!(state, second.app().snapshot());
    assert_eq!(
        state,
        recorded.app().snapshot(),
        "the replay matches the live run"
    );
    assert_eq!(first.gpu().frames(), second.gpu().frames());
    assert_eq!(first.gpu().events(), second.gpu().events());
    assert_eq!(first.gpu().frames(), recorded.gpu().frames());

    // Real gameplay happened.
    assert!(paddle_bounces >= 3, "{paddle_bounces} paddle bounces");
    assert!(side_bounces >= 1, "{side_bounces} side wall bounces");
    assert!(state.score > 0 && state.bricks_remaining < 60, "{state:?}");
    assert_eq!(first.gpu().frames().len(), REPLAY_FRAMES);
    first
        .gpu()
        .frames()
        .iter()
        .for_each(assert_one_draw_per_texture);
}
