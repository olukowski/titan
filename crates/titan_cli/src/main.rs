use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{CommandFactory, Parser, Subcommand, error::ErrorKind};
use serde::Serialize;
use titan_core::{
    DEFAULT_FIXED_DT, DEFAULT_RUN_SEED, FixedStepContext, Schedule, World,
    velocity_integration_system,
};
use titan_render::{
    CameraSelection, CaptureMode, OutputSize, RenderRequest, RenderService, error as render_error,
};
use titan_scene::{
    Diagnostic, DiagnosticSpan, Document, Position, Span, TsfError, load_world, parse,
    phase2_component_registry,
};

const APP_NAME: &str = "titan";
const EXIT_FAILURE: u8 = 1;
const EXIT_USAGE_OR_IO: u8 = 2;

#[derive(Debug, Parser)]
#[command(name = APP_NAME, about = "Agent-native game engine tooling")]
struct Cli {
    /// Emit structured JSON output.
    #[arg(long, global = true)]
    json: bool,

    /// Print version information and exit.
    #[arg(long)]
    version: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a scene in the deterministic headless runtime.
    Run(RunArgs),
    /// Render one frame of a scene without opening a window.
    Render(RenderArgs),
    /// Print version information.
    Version,
    /// Validate, query, edit, and format Titan Scene Format files.
    Scene {
        #[command(subcommand)]
        command: SceneCommand,
    },
}

#[derive(Debug, Subcommand)]
enum SceneCommand {
    /// Parse and validate a TSF scene.
    Validate {
        /// Scene file to validate.
        path: PathBuf,
    },
    /// Print a JSON value selected by JSON Pointer.
    Query {
        /// Scene file to inspect.
        path: PathBuf,
        /// JSON Pointer path. Entity arrays also accept entity:<slug> ids.
        pointer: String,
    },
    /// Replace a JSON value and rewrite the file canonically.
    Edit {
        /// Scene file to edit.
        path: PathBuf,
        /// JSON Pointer path. Entity arrays also accept entity:<slug> ids.
        pointer: String,
        /// Replacement value parsed as TSF/JSON5.
        #[arg(allow_hyphen_values = true)]
        value: String,
    },
    /// Rewrite a TSF scene in canonical form.
    Fmt {
        /// Scene file to format.
        path: PathBuf,
        /// Check whether the file is already canonical without writing.
        #[arg(long)]
        check: bool,
    },
}

#[derive(Debug, Parser)]
struct RunArgs {
    /// Scene file to load.
    scene: PathBuf,

    /// Run headless. This is the only runtime mode in Phase 1.
    #[arg(long)]
    headless: bool,

    /// Number of fixed-timestep frames to simulate.
    #[arg(long)]
    frames: u64,

    /// Deterministic run seed.
    #[arg(long, default_value_t = DEFAULT_RUN_SEED)]
    seed: u64,

    /// Fixed timestep in seconds.
    #[arg(long, default_value_t = DEFAULT_FIXED_DT)]
    dt: f32,

    /// Write the final state dump JSON to this file.
    #[arg(long)]
    dump_state: Option<PathBuf>,

    /// Write the event log as JSONL to this file.
    #[arg(long)]
    event_log: Option<PathBuf>,

    /// Capture a frame after every N fixed-step updates.
    #[arg(long)]
    capture_every: Option<u64>,

    /// Capture the initial frame before the first update.
    #[arg(long)]
    capture_initial: bool,

    /// Directory for PNGs and the authoritative captures.jsonl manifest; stale PNGs are retained.
    #[arg(long, default_value = ".titan/cache/captures")]
    capture_dir: PathBuf,

    /// Camera name or serialized entity ID.
    #[arg(long)]
    camera: Option<String>,

    /// Captured image width (requires --height).
    #[arg(long, requires = "height")]
    width: Option<u32>,

    /// Captured image height (requires --width).
    #[arg(long, requires = "width")]
    height: Option<u32>,
}

#[derive(Debug, Parser)]
struct RenderArgs {
    /// Scene file to render.
    scene: PathBuf,

    /// Camera name or serialized entity ID (for example entity:main_camera).
    #[arg(long)]
    camera: Option<String>,

    /// PNG output file.
    #[arg(long)]
    out: PathBuf,

    /// Write render statistics as JSON to a file.
    #[arg(long)]
    stats_json: Option<PathBuf>,

    /// Output image width (requires --height).
    #[arg(long, requires = "height")]
    width: Option<u32>,

    /// Output image height (requires --width).
    #[arg(long, requires = "width")]
    height: Option<u32>,
}

#[derive(Debug)]
struct TitanError {
    code: &'static str,
    message: String,
    exit_code: u8,
    diagnostics: Option<Vec<Diagnostic>>,
    location: Option<String>,
}

enum SceneLoadError {
    Io(TitanError),
    Tsf(TsfError),
}

impl TitanError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self::with_exit_code(code, message, EXIT_USAGE_OR_IO)
    }

    fn with_exit_code(code: &'static str, message: impl Into<String>, exit_code: u8) -> Self {
        Self {
            code,
            message: message.into(),
            exit_code,
            diagnostics: None,
            location: None,
        }
    }

    fn from_tsf(message: impl Into<String>, error: titan_scene::TsfError) -> Self {
        Self {
            code: "TITAN_TSF_ERROR",
            message: message.into(),
            exit_code: EXIT_FAILURE,
            diagnostics: Some(error.errors),
            location: None,
        }
    }

    fn from_render(error: titan_render::RenderError) -> Self {
        Self {
            code: error.code,
            message: error.message,
            exit_code: EXIT_FAILURE,
            diagnostics: None,
            location: error.path,
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<&'a [Diagnostic]>,
}

#[derive(Serialize)]
struct VersionOutput<'a> {
    name: &'a str,
    version: &'a str,
}

#[derive(Serialize)]
struct SceneValidateOutput {
    ok: bool,
    path: String,
}

#[derive(Serialize)]
struct SceneQueryOutput {
    ok: bool,
    path: String,
    resolved_pointer: String,
    span: DiagnosticSpan,
    value: serde_json::Value,
}

#[derive(Serialize)]
struct SceneEditOutput {
    ok: bool,
    path: String,
    changed: bool,
    changed_lines: usize,
    diff: Vec<LineDiff>,
}

#[derive(Serialize)]
struct SceneFmtOutput {
    ok: bool,
    path: String,
    canonical: bool,
    written: bool,
}

#[derive(Serialize)]
struct LineDiff {
    line: usize,
    old: String,
    new: String,
}

#[derive(Serialize)]
struct SceneErrorOutput<'a> {
    ok: bool,
    errors: &'a [Diagnostic],
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            report_error(&error);
            ExitCode::from(error.exit_code)
        }
    }
}

fn run() -> Result<ExitCode, TitanError> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            error.print().map_err(|source| {
                TitanError::new(
                    "TITAN_CLI_HELP_RENDER",
                    format!("failed to render help: {source}"),
                )
            })?;
            return Ok(ExitCode::SUCCESS);
        }
        Err(error) => return Err(TitanError::from_clap(error)),
    };

    if cli.version {
        print_version(cli.json)?;
        return Ok(ExitCode::SUCCESS);
    }

    match cli.command {
        Some(Command::Run(args)) => run_runtime(args, cli.json),
        Some(Command::Render(args)) => run_render(args, cli.json),
        Some(Command::Version) => print_version(cli.json),
        Some(Command::Scene { command }) => return run_scene(command, cli.json),
        None => {
            Cli::command().print_help().map_err(|source| {
                TitanError::new(
                    "TITAN_CLI_HELP_RENDER",
                    format!("failed to render help: {source}"),
                )
            })?;
            println!();
            Ok(())
        }
    }?;

    Ok(ExitCode::SUCCESS)
}

fn run_runtime(args: RunArgs, json: bool) -> Result<(), TitanError> {
    if !args.dt.is_finite() || args.dt <= 0.0 {
        return Err(TitanError::new(
            "TITAN_CLI_ARGUMENT_ERROR",
            "--dt must be finite and positive",
        ));
    }
    if args.capture_every == Some(0) {
        return Err(TitanError::new(
            "TITAN_CLI_ARGUMENT_ERROR",
            "--capture-every must be greater than zero",
        ));
    }
    let capture_requested = args.capture_initial || args.capture_every.is_some();
    if !capture_requested
        && (args.camera.is_some() || args.width.is_some() || args.height.is_some())
    {
        return Err(TitanError::new(
            "TITAN_CLI_ARGUMENT_ERROR",
            "--camera, --width, and --height require --capture-initial or --capture-every",
        ));
    }
    let output_size = paired_output_size(args.width, args.height)?;
    let capture_scheduled =
        args.capture_initial || args.capture_every.is_some_and(|every| every <= args.frames);

    let source = fs::read_to_string(&args.scene).map_err(|source| {
        TitanError::new(
            "TITAN_SCENE_READ",
            format!("failed to read {}: {source}", args.scene.display()),
        )
    })?;
    let file = args.scene.to_string_lossy();
    let document = parse(Some(&file), &source)
        .map_err(|error| TitanError::from_tsf("failed to parse scene", error))?;
    let registry = phase2_component_registry().map_err(|source| {
        TitanError::new(
            "TITAN_COMPONENT_REGISTRY",
            format!("failed to build component registry: {source}"),
        )
    })?;
    let mut world = load_world(&document, registry)
        .map_err(|error| TitanError::from_tsf("failed to load scene", error))?;
    world.set_runtime_metadata(0, args.seed);

    let mut capture = if capture_scheduled {
        Some(CaptureSession::new(
            &args.capture_dir,
            &world,
            args.camera.as_deref(),
            output_size,
            json,
            CaptureSchedule {
                initial: args.capture_initial,
                every: args.capture_every,
                frames: args.frames,
            },
        )?)
    } else {
        if capture_requested {
            // An empty schedule still validates the camera and scene-facing
            // render inputs before the stale manifest is replaced, so a
            // mistyped --camera or --capture-every never exits 0 silently.
            let camera = select_camera(&world, args.camera.as_deref())?;
            RenderService::cpu_only()
                .render(
                    &world,
                    RenderRequest {
                        camera,
                        output_size,
                        capture: CaptureMode::StatsOnly,
                        ..RenderRequest::default()
                    },
                )
                .map_err(TitanError::from_render)?;
            create_empty_capture_manifest(&args.capture_dir)?;
        }
        None
    };
    if args.capture_initial {
        capture
            .as_mut()
            .expect("capture session")
            .capture(&world, 0, args.seed)?;
    }

    let mut schedule = Schedule::new();
    schedule.add_system(
        "titan.core.velocity_integration",
        velocity_integration_system,
    );
    for frame in 1..=args.frames {
        schedule
            .run_fixed_step(&mut world, FixedStepContext::new(args.dt, frame, args.seed))
            .map_err(|source| {
                TitanError::new(
                    "TITAN_RUNTIME_SYSTEM",
                    format!("runtime system failed: {source}"),
                )
            })?;
        if args.capture_every.is_some_and(|every| frame % every == 0) {
            capture
                .as_mut()
                .expect("capture session")
                .capture(&world, frame, args.seed)?;
        }
    }

    let dump = world.dump_state().map_err(|source| {
        TitanError::new(
            "TITAN_STATE_DUMP",
            format!("failed to create state dump: {source}"),
        )
    })?;
    let dump_json = serde_json::to_string(&dump).map_err(|source| {
        TitanError::new(
            "TITAN_OUTPUT_SERIALIZE",
            format!("failed to encode state dump JSON: {source}"),
        )
    })?;
    if let Some(path) = args.dump_state {
        fs::write(&path, dump_json).map_err(|source| {
            TitanError::new(
                "TITAN_STATE_DUMP_WRITE",
                format!("failed to write {}: {source}", path.display()),
            )
        })?;
    } else if json && !capture_requested {
        println!("{dump_json}");
    }

    if let Some(path) = args.event_log {
        let jsonl = world.event_log().to_jsonl().map_err(|source| {
            TitanError::new(
                "TITAN_EVENT_LOG",
                format!("failed to encode event log: {source}"),
            )
        })?;
        fs::write(&path, jsonl).map_err(|source| {
            TitanError::new(
                "TITAN_EVENT_LOG_WRITE",
                format!("failed to write {}: {source}", path.display()),
            )
        })?;
    }

    Ok(())
}

#[derive(Serialize)]
struct RenderOutput {
    ok: bool,
    frame: u64,
    camera: String,
    output: String,
    width: u32,
    height: u32,
    draw_calls: u32,
    triangles: u64,
    visible_meshes: u32,
    active_directional_lights: u32,
    backend: &'static str,
    adapter: String,
    shader_version: u32,
    material_models: std::collections::BTreeMap<String, u32>,
}

struct CaptureSession {
    directory: PathBuf,
    stats: fs::File,
    service: RenderService,
    camera: CameraSelection,
    output_size: Option<OutputSize>,
    mirror_stdout: bool,
}

#[derive(Clone, Copy)]
struct CaptureSchedule {
    initial: bool,
    every: Option<u64>,
    frames: u64,
}

impl CaptureSession {
    fn new(
        directory: &Path,
        world: &World,
        camera: Option<&str>,
        output_size: Option<OutputSize>,
        mirror_stdout: bool,
        schedule: CaptureSchedule,
    ) -> Result<Self, TitanError> {
        let camera = select_camera(world, camera)?;
        let request = RenderRequest {
            camera: camera.clone(),
            output_size,
            capture: CaptureMode::Image,
            ..RenderRequest::default()
        };
        // CPU validation cannot request pixels, so validate the scene-facing
        // inputs without capture before initializing the image-capable backend.
        RenderService::cpu_only()
            .render(
                world,
                RenderRequest {
                    capture: CaptureMode::StatsOnly,
                    ..request.clone()
                },
            )
            .map_err(TitanError::from_render)?;
        // Validate filesystem paths before initializing the image-capable
        // backend. Existing PNGs are deliberately retained; only records in
        // captures.jsonl belong to this run.
        prepare_capture_directory(directory)?;
        validate_scheduled_capture_paths(directory, schedule)?;
        validate_capture_manifest_path(directory)?;

        // Validate the complete render path before replacing the authoritative
        // manifest.
        let service = RenderService::new().map_err(TitanError::from_render)?;
        let preflight = service
            .render(world, request)
            .map_err(TitanError::from_render)?;
        require_capture_pixels(&preflight)?;
        let stats = open_capture_manifest(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
            stats,
            service,
            camera,
            output_size,
            mirror_stdout,
        })
    }

    fn capture(&mut self, world: &World, frame: u64, seed: u64) -> Result<(), TitanError> {
        let path = self.directory.join(format!("frame-{frame:06}.png"));
        validate_capture_output_path(&path)?;
        let result = self
            .service
            .render(
                world,
                RenderRequest {
                    camera: self.camera.clone(),
                    output_size: self.output_size,
                    capture: CaptureMode::Image,
                    ..RenderRequest::default()
                },
            )
            .map_err(TitanError::from_render)?;
        let pixels = require_capture_pixels(&result)?;
        write_png(
            &path,
            result.output_size.width,
            result.output_size.height,
            pixels,
        )?;
        // The manifest is authoritative: write the PNG before appending its
        // record. A later stats failure may leave an unreferenced PNG, but a
        // record never claims a PNG that this run failed to write.
        let mut output = render_output(world, &self.service, result, frame, &path)?;
        let record = CaptureOutput::from_render(seed, &mut output);
        serde_json::to_writer(&mut self.stats, &record).map_err(|source| {
            TitanError::new(
                "TITAN_OUTPUT_SERIALIZE",
                format!("failed to encode capture stats: {source}"),
            )
        })?;
        self.stats.write_all(b"\n").map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to write capture stats: {source}"),
            )
        })?;
        self.stats.flush().map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to flush capture stats: {source}"),
            )
        })?;
        if self.mirror_stdout {
            print_json(&record)?;
        }
        Ok(())
    }
}

fn validate_scheduled_capture_paths(
    directory: &Path,
    schedule: CaptureSchedule,
) -> Result<(), TitanError> {
    if schedule.initial {
        validate_capture_output_path(&directory.join(format!("frame-{:06}.png", 0)))?;
    }
    if let Some(every) = schedule.every {
        debug_assert!(every > 0);
        let mut frame = every;
        while frame <= schedule.frames {
            validate_capture_output_path(&directory.join(format!("frame-{frame:06}.png")))?;
            let Some(next) = frame.checked_add(every) else {
                break;
            };
            frame = next;
        }
    }
    Ok(())
}

fn validate_capture_output_path(path: &Path) -> Result<(), TitanError> {
    reject_symlink(path, "capture PNG output")?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() => Err(TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!(
                "capture PNG output {} must be a regular file or not exist",
                path.display()
            ),
        )),
        Ok(_) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!(
                "failed to inspect capture PNG output {}: {source}",
                path.display()
            ),
        )),
    }
}

fn require_capture_pixels(result: &titan_render::RenderResult) -> Result<&[u8], TitanError> {
    result.rgba8.as_deref().ok_or_else(|| {
        TitanError::from_render(titan_render::RenderError {
            code: render_error::CAPTURE_UNAVAILABLE,
            message: "renderer did not return captured pixels".to_owned(),
            path: None,
        })
    })
}

fn open_capture_manifest(directory: &Path) -> Result<fs::File, TitanError> {
    let stats_path = directory.join("captures.jsonl");
    validate_capture_manifest_path(directory)?;
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&stats_path)
        .map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to write {}: {source}", stats_path.display()),
            )
        })
}

fn validate_capture_manifest_path(directory: &Path) -> Result<(), TitanError> {
    let stats_path = directory.join("captures.jsonl");
    reject_symlink(&stats_path, "capture stats output")?;
    match fs::symlink_metadata(&stats_path) {
        Ok(metadata) if !metadata.is_file() => Err(TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!(
                "capture stats output {} must be a regular file or not exist",
                stats_path.display()
            ),
        )),
        Ok(_) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!(
                "failed to inspect capture stats output {}: {source}",
                stats_path.display()
            ),
        )),
    }
}

fn create_empty_capture_manifest(directory: &Path) -> Result<(), TitanError> {
    prepare_capture_directory(directory)?;
    open_capture_manifest(directory)?;
    Ok(())
}

#[derive(Serialize)]
struct CaptureOutput {
    ok: bool,
    frame: u64,
    seed: u64,
    camera: String,
    output: String,
    width: u32,
    height: u32,
    draw_calls: u32,
    triangles: u64,
    visible_meshes: u32,
    active_directional_lights: u32,
    backend: &'static str,
    adapter: String,
    shader_version: u32,
    material_models: std::collections::BTreeMap<String, u32>,
}

impl CaptureOutput {
    fn from_render(seed: u64, output: &mut RenderOutput) -> Self {
        Self {
            ok: output.ok,
            frame: output.frame,
            seed,
            camera: std::mem::take(&mut output.camera),
            output: std::mem::take(&mut output.output),
            width: output.width,
            height: output.height,
            draw_calls: output.draw_calls,
            triangles: output.triangles,
            visible_meshes: output.visible_meshes,
            active_directional_lights: output.active_directional_lights,
            backend: output.backend,
            adapter: std::mem::take(&mut output.adapter),
            shader_version: output.shader_version,
            material_models: std::mem::take(&mut output.material_models),
        }
    }
}

fn paired_output_size(
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Option<OutputSize>, TitanError> {
    match (width, height) {
        (None, None) => Ok(None),
        (Some(width), Some(height)) => Ok(Some(OutputSize::new(width, height))),
        _ => Err(TitanError::new(
            "TITAN_CLI_ARGUMENT_ERROR",
            "--width and --height must be provided together",
        )),
    }
}

fn prepare_capture_directory(path: &Path) -> Result<(), TitanError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(TitanError::new(
                "TITAN_OUTPUT_PATH",
                format!(
                    "capture directory {} must be a directory, not a symlink",
                    path.display()
                ),
            ));
        }
        Ok(_) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|source| {
                TitanError::new(
                    "TITAN_OUTPUT_PATH",
                    format!(
                        "failed to create capture directory {}: {source}",
                        path.display()
                    ),
                )
            })?;
        }
        Err(source) => {
            return Err(TitanError::new(
                "TITAN_OUTPUT_PATH",
                format!(
                    "failed to inspect capture directory {}: {source}",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn select_camera(world: &World, camera: Option<&str>) -> Result<CameraSelection, TitanError> {
    match camera {
        None => Ok(CameraSelection::Default),
        Some(value) if value.starts_with("entity:") => {
            let state = world.dump_state().map_err(|source| {
                TitanError::new(
                    "TITAN_STATE_DUMP",
                    format!("failed to resolve scene entity IDs: {source}"),
                )
            })?;
            let raw = state.entity_ids.get(value).copied().ok_or_else(|| {
                TitanError::from_render(titan_render::RenderError {
                    code: render_error::CAMERA_UNAVAILABLE,
                    message: format!("camera entity '{value}' was not found"),
                    path: Some(format!("camera:{value}")),
                })
            })?;
            Ok(CameraSelection::Entity(titan_core::EntityId::from_raw(raw)))
        }
        Some(value) => Ok(CameraSelection::Name(value.to_owned())),
    }
}

fn render_output(
    world: &World,
    service: &RenderService,
    result: titan_render::RenderResult,
    frame: u64,
    path: &Path,
) -> Result<RenderOutput, TitanError> {
    let state = world.dump_state().map_err(|source| {
        TitanError::new(
            "TITAN_STATE_DUMP",
            format!("failed to resolve scene entity IDs: {source}"),
        )
    })?;
    let camera = result
        .camera
        .and_then(|id| {
            state
                .entity_ids
                .iter()
                .find_map(|(name, raw)| (*raw == id.raw()).then(|| name.clone()))
        })
        .unwrap_or_else(|| "entity:unknown".to_owned());
    let adapter = service
        .adapter_info()
        .map(|info| info.name.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    Ok(RenderOutput {
        ok: true,
        frame,
        camera,
        output: path.display().to_string(),
        width: result.output_size.width,
        height: result.output_size.height,
        draw_calls: result.stats.draw_calls,
        triangles: result.stats.triangles,
        visible_meshes: result.stats.visible_meshes,
        active_directional_lights: result.stats.active_directional_lights,
        backend: "wgpu",
        adapter,
        shader_version: result.stats.shader_version,
        material_models: result.stats.material_models,
    })
}

fn run_render(args: RenderArgs, json: bool) -> Result<(), TitanError> {
    let output_size = paired_output_size(args.width, args.height)?;
    if let Some(stats_path) = args.stats_json.as_ref()
        && normalized_output_path(&args.out)? == normalized_output_path(stats_path)?
    {
        return Err(TitanError::new(
            "TITAN_OUTPUT_COLLISION",
            format!(
                "PNG output {} and stats output {} must be different files",
                args.out.display(),
                stats_path.display()
            ),
        ));
    }
    reject_symlink(&args.out, "PNG output")?;
    let document = read_scene(&args.scene).map_err(|error| match error {
        SceneLoadError::Io(error) => error,
        SceneLoadError::Tsf(error) => TitanError::from_tsf("failed to parse scene", error),
    })?;
    let registry = phase2_component_registry().map_err(|source| {
        TitanError::new(
            "TITAN_COMPONENT_REGISTRY",
            format!("failed to build component registry: {source}"),
        )
    })?;
    let world = load_world(&document, registry)
        .map_err(|error| TitanError::from_tsf("failed to load scene", error))?;
    let selection = select_camera(&world, args.camera.as_deref())?;
    // Validate camera selection and all scene-facing render inputs before asking
    // wgpu for an adapter, so diagnostics are useful on adapter-less hosts.
    RenderService::cpu_only()
        .render(
            &world,
            RenderRequest {
                camera: selection.clone(),
                output_size,
                ..RenderRequest::default()
            },
        )
        .map_err(TitanError::from_render)?;
    let service = RenderService::new().map_err(TitanError::from_render)?;
    let result = service
        .render(
            &world,
            RenderRequest {
                camera: selection,
                output_size,
                capture: CaptureMode::Image,
                ..RenderRequest::default()
            },
        )
        .map_err(TitanError::from_render)?;
    let pixels = result.rgba8.as_deref().ok_or_else(|| {
        TitanError::from_render(titan_render::RenderError {
            code: render_error::CAPTURE_UNAVAILABLE,
            message: "renderer did not return captured pixels".to_owned(),
            path: None,
        })
    })?;
    write_png(
        &args.out,
        result.output_size.width,
        result.output_size.height,
        pixels,
    )?;

    let output = render_output(&world, &service, result, 0, &args.out)?;
    if let Some(path) = args.stats_json {
        let body = serde_json::to_vec(&output).map_err(|source| {
            TitanError::new(
                "TITAN_OUTPUT_SERIALIZE",
                format!("failed to encode render stats: {source}"),
            )
        })?;
        let mut file = open_stats_file(&path)?;
        file.write_all(&body).map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to write {}: {source}", path.display()),
            )
        })?;
    }
    if json {
        print_json(&output)?;
    } else {
        println!("rendered {}", args.out.display());
    }
    Ok(())
}

fn open_stats_file(path: &Path) -> Result<fs::File, TitanError> {
    reject_symlink(path, "stats output")?;

    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to write {}: {source}", path.display()),
            )
        })
}

fn reject_symlink(path: &Path, label: &str) -> Result<(), TitanError> {
    // Pathname- and symlink-level aliasing is rejected for both outputs; hard-link aliases are intentionally not detected.
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(TitanError::new(
                "TITAN_OUTPUT_PATH",
                format!("{label} {} must not be a symlink", path.display()),
            ));
        }
        Ok(_) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(TitanError::new(
                "TITAN_OUTPUT_PATH",
                format!("failed to inspect {label} {}: {source}", path.display()),
            ));
        }
    }
    Ok(())
}

fn normalized_output_path(path: &Path) -> Result<std::path::PathBuf, TitanError> {
    let file_name = path.file_name().ok_or_else(|| {
        TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!("output path {} has no file name", path.display()),
        )
    })?;

    match fs::canonicalize(path) {
        Ok(path) => return Ok(path),
        Err(source) if source.kind() != std::io::ErrorKind::NotFound => {
            return Err(TitanError::new(
                "TITAN_OUTPUT_PATH",
                format!("failed to resolve output path {}: {source}", path.display()),
            ));
        }
        Err(_) => {}
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|source| {
        TitanError::new(
            "TITAN_OUTPUT_PATH",
            format!(
                "failed to resolve output directory {}: {source}",
                parent.display()
            ),
        )
    })?;
    Ok(parent.join(file_name))
}

fn write_png(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), TitanError> {
    let file = fs::File::create(path).map_err(|source| {
        TitanError::new(
            "TITAN_PNG_WRITE",
            format!("failed to create {}: {source}", path.display()),
        )
    })?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|source| {
        TitanError::new(
            "TITAN_PNG_WRITE",
            format!("failed to write PNG header: {source}"),
        )
    })?;
    writer.write_image_data(pixels).map_err(|source| {
        TitanError::new(
            "TITAN_PNG_WRITE",
            format!("failed to write PNG pixels: {source}"),
        )
    })?;
    Ok(())
}

fn print_version(json: bool) -> Result<(), TitanError> {
    let output = VersionOutput {
        name: APP_NAME,
        version: env!("CARGO_PKG_VERSION"),
    };

    if json {
        let body = serde_json::to_string(&output).map_err(|source| {
            TitanError::new(
                "TITAN_OUTPUT_SERIALIZE",
                format!("failed to encode JSON: {source}"),
            )
        })?;
        println!("{body}");
    } else {
        println!("{} {}", output.name, output.version);
    }

    Ok(())
}

fn report_error(error: &TitanError) {
    let envelope = ErrorEnvelope {
        error: ErrorBody {
            code: error.code,
            message: &error.message,
            location: error.location.as_deref(),
            diagnostics: error.diagnostics.as_deref(),
        },
    };

    match serde_json::to_string(&envelope) {
        Ok(body) => eprintln!("{body}"),
        Err(_) => eprintln!("{}: {}", error.code, error.message),
    }
}

fn run_scene(command: SceneCommand, json: bool) -> Result<ExitCode, TitanError> {
    match command {
        SceneCommand::Validate { path } => {
            let document = match read_scene(&path) {
                Ok(document) => document,
                Err(SceneLoadError::Tsf(error)) => {
                    report_scene_error(&error, json);
                    return Ok(ExitCode::from(EXIT_FAILURE));
                }
                Err(SceneLoadError::Io(error)) => return Err(error),
            };
            match titan_scene::validate(&document) {
                Ok(()) => {
                    if json {
                        print_json(&SceneValidateOutput {
                            ok: true,
                            path: path.display().to_string(),
                        })?;
                    } else {
                        println!("ok: {} is valid", path.display());
                    }
                    Ok(ExitCode::SUCCESS)
                }
                Err(error) => {
                    report_scene_error(&error, json);
                    Ok(ExitCode::from(EXIT_FAILURE))
                }
            }
        }
        SceneCommand::Query { path, pointer } => {
            let document = match read_scene(&path) {
                Ok(document) => document,
                Err(SceneLoadError::Tsf(error)) => {
                    report_scene_error(&error, json);
                    return Ok(ExitCode::from(EXIT_FAILURE));
                }
                Err(SceneLoadError::Io(error)) => return Err(error),
            };
            match titan_scene::query(&document, &pointer) {
                Ok(result) => {
                    if json {
                        print_json(&SceneQueryOutput {
                            ok: true,
                            path: pointer,
                            resolved_pointer: result.resolved_pointer,
                            span: diagnostic_span(document.file.as_deref(), result.span),
                            value: result.value,
                        })?;
                    } else {
                        println!("path: {pointer}");
                        println!("resolved: {}", result.resolved_pointer);
                        println!(
                            "span: {}",
                            format_span(document.file.as_deref(), result.span)
                        );
                        println!("{}", result.value);
                    }
                    Ok(ExitCode::SUCCESS)
                }
                Err(error) => {
                    report_scene_error(&error, json);
                    Ok(ExitCode::from(EXIT_FAILURE))
                }
            }
        }
        SceneCommand::Edit {
            path,
            pointer,
            value,
        } => {
            let document = match read_scene(&path) {
                Ok(document) => document,
                Err(SceneLoadError::Tsf(error)) => {
                    report_scene_error(&error, json);
                    return Ok(ExitCode::from(EXIT_FAILURE));
                }
                Err(SceneLoadError::Io(error)) => return Err(error),
            };
            let old_source = fs::read_to_string(&path).map_err(|source| {
                TitanError::new(
                    "TITAN_IO_READ",
                    format!("failed to read {}: {source}", path.display()),
                )
            })?;
            match titan_scene::edit(&document, &pointer, &value) {
                Ok(new_source) => {
                    write_scene_atomic(&path, &new_source)?;
                    let diff = line_diff(&old_source, &new_source);
                    if json {
                        print_json(&SceneEditOutput {
                            ok: true,
                            path: path.display().to_string(),
                            changed: old_source != new_source,
                            changed_lines: diff.len(),
                            diff,
                        })?;
                    } else if old_source == new_source {
                        println!("unchanged: {}", path.display());
                    } else {
                        println!("updated: {}", path.display());
                        for entry in &diff {
                            println!("-{}: {}", entry.line, entry.old);
                            println!("+{}: {}", entry.line, entry.new);
                        }
                    }
                    Ok(ExitCode::SUCCESS)
                }
                Err(error) => {
                    report_scene_error(&error, json);
                    Ok(ExitCode::from(EXIT_FAILURE))
                }
            }
        }
        SceneCommand::Fmt { path, check } => {
            let document = match read_scene(&path) {
                Ok(document) => document,
                Err(SceneLoadError::Tsf(error)) => {
                    report_scene_error(&error, json);
                    return Ok(ExitCode::from(EXIT_FAILURE));
                }
                Err(SceneLoadError::Io(error)) => return Err(error),
            };
            let old_source = fs::read_to_string(&path).map_err(|source| {
                TitanError::new(
                    "TITAN_IO_READ",
                    format!("failed to read {}: {source}", path.display()),
                )
            })?;
            let new_source = titan_scene::fmt(&document);
            let canonical = old_source == new_source;
            if check {
                if canonical {
                    if json {
                        print_json(&SceneFmtOutput {
                            ok: true,
                            path: path.display().to_string(),
                            canonical: true,
                            written: false,
                        })?;
                    } else {
                        println!("ok: {} is canonical", path.display());
                    }
                    Ok(ExitCode::SUCCESS)
                } else {
                    if json {
                        let error = not_canonical_error(&path);
                        report_scene_error(&error, json);
                    } else {
                        eprintln!("not canonical: {}", path.display());
                    }
                    Ok(ExitCode::from(EXIT_FAILURE))
                }
            } else {
                if !canonical {
                    write_scene_atomic(&path, &new_source)?;
                }
                if json {
                    print_json(&SceneFmtOutput {
                        ok: true,
                        path: path.display().to_string(),
                        canonical: true,
                        written: !canonical,
                    })?;
                } else if canonical {
                    println!("unchanged: {}", path.display());
                } else {
                    println!("formatted: {}", path.display());
                }
                Ok(ExitCode::SUCCESS)
            }
        }
    }
}

fn read_scene(path: &Path) -> Result<Document, SceneLoadError> {
    let source = fs::read_to_string(path).map_err(|error| {
        SceneLoadError::Io(TitanError::new(
            "TITAN_IO_READ",
            format!("failed to read {}: {error}", path.display()),
        ))
    })?;
    titan_scene::parse(Some(&path.display().to_string()), &source).map_err(SceneLoadError::Tsf)
}

fn write_scene_atomic(path: &Path, source: &str) -> Result<(), TitanError> {
    let target = fs::canonicalize(path).map_err(|source| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!("failed to resolve {}: {source}", path.display()),
        )
    })?;
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let permissions = fs::metadata(&target)
        .map_err(|source| {
            TitanError::new(
                "TITAN_IO_WRITE",
                format!("failed to read metadata for {}: {source}", path.display()),
            )
        })?
        .permissions();
    let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|source| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!(
                "failed to create temporary file for {}: {source}",
                path.display()
            ),
        )
    })?;
    file.write_all(source.as_bytes()).map_err(|source| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!("failed to write {}: {source}", path.display()),
        )
    })?;
    file.flush().map_err(|source| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!("failed to flush {}: {source}", path.display()),
        )
    })?;
    file.as_file()
        .set_permissions(permissions)
        .map_err(|source| {
            TitanError::new(
                "TITAN_IO_WRITE",
                format!("failed to set permissions for {}: {source}", path.display()),
            )
        })?;
    file.as_file().sync_all().map_err(|source| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!("failed to sync {}: {source}", path.display()),
        )
    })?;
    file.persist(&target).map_err(|error| {
        TitanError::new(
            "TITAN_IO_WRITE",
            format!("failed to replace {}: {}", path.display(), error.error),
        )
    })?;
    Ok(())
}

fn report_scene_error(error: &TsfError, json: bool) {
    if json {
        let output = SceneErrorOutput {
            ok: false,
            errors: &error.errors,
        };
        match serde_json::to_string(&output) {
            Ok(body) => eprintln!("{body}"),
            Err(_) => eprintln!("{error}"),
        }
    } else {
        for diagnostic in &error.errors {
            eprintln!(
                "{}: {} at {}",
                diagnostic.code,
                diagnostic.message,
                format_diagnostic_span(&diagnostic.span)
            );
            if !diagnostic.path.is_empty() {
                eprintln!("  path: {}", diagnostic.path);
            }
        }
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<(), TitanError> {
    let body = serde_json::to_string(value).map_err(|source| {
        TitanError::new(
            "TITAN_OUTPUT_SERIALIZE",
            format!("failed to encode JSON: {source}"),
        )
    })?;
    println!("{body}");
    Ok(())
}

fn diagnostic_span(file: Option<&str>, span: Span) -> DiagnosticSpan {
    DiagnosticSpan {
        file: file.map(str::to_owned),
        start: span.start,
        end: span.end,
    }
}

fn format_diagnostic_span(span: &DiagnosticSpan) -> String {
    let location = format!(
        "{}:{}-{}:{}",
        span.start.line, span.start.column, span.end.line, span.end.column
    );
    match &span.file {
        Some(file) => format!("{file}:{location}"),
        None => location,
    }
}

fn format_span(file: Option<&str>, span: Span) -> String {
    format_diagnostic_span(&diagnostic_span(file, span))
}

fn line_diff(old_source: &str, new_source: &str) -> Vec<LineDiff> {
    let old_lines: Vec<_> = old_source.lines().collect();
    let new_lines: Vec<_> = new_source.lines().collect();
    let max_len = old_lines.len().max(new_lines.len());
    (0..max_len)
        .filter_map(|index| {
            let old = old_lines.get(index).copied().unwrap_or("");
            let new = new_lines.get(index).copied().unwrap_or("");
            (old != new).then(|| LineDiff {
                line: index + 1,
                old: old.to_owned(),
                new: new.to_owned(),
            })
        })
        .collect()
}

fn not_canonical_error(path: &Path) -> TsfError {
    TsfError::one(
        Some(&path.display().to_string()),
        "TSF_NOT_CANONICAL",
        "file is not in canonical TSF format",
        "",
        Span {
            start: Position { line: 1, column: 1 },
            end: Position { line: 1, column: 1 },
        },
    )
}

impl TitanError {
    fn from_clap(error: clap::Error) -> Self {
        let code = match error.kind() {
            ErrorKind::UnknownArgument => "TITAN_CLI_UNKNOWN_ARGUMENT",
            ErrorKind::InvalidSubcommand => "TITAN_CLI_UNKNOWN_COMMAND",
            ErrorKind::MissingRequiredArgument => "TITAN_CLI_MISSING_ARGUMENT",
            _ => "TITAN_CLI_ARGUMENT_ERROR",
        };

        Self::new(code, normalize_clap_message(error))
    }
}

fn normalize_clap_message(error: clap::Error) -> String {
    let message = error.to_string();
    message
        .strip_prefix("error: ")
        .unwrap_or(&message)
        .trim()
        .to_owned()
}
