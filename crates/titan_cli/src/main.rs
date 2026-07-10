use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{CommandFactory, Parser, Subcommand, error::ErrorKind};
use serde::Serialize;
use titan_core::{
    DEFAULT_FIXED_DT, DEFAULT_RUN_SEED, FixedStepContext, Schedule, phase1_component_registry,
    velocity_integration_system,
};
use titan_render::{
    CameraSelection, CaptureMode, RenderRequest, RenderService, error as render_error,
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

    let source = fs::read_to_string(&args.scene).map_err(|source| {
        TitanError::new(
            "TITAN_SCENE_READ",
            format!("failed to read {}: {source}", args.scene.display()),
        )
    })?;
    let file = args.scene.to_string_lossy();
    let document = parse(Some(&file), &source)
        .map_err(|error| TitanError::from_tsf("failed to parse scene", error))?;
    let registry = phase1_component_registry().map_err(|source| {
        TitanError::new(
            "TITAN_COMPONENT_REGISTRY",
            format!("failed to build component registry: {source}"),
        )
    })?;
    let mut world = load_world(&document, registry)
        .map_err(|error| TitanError::from_tsf("failed to load scene", error))?;
    world.set_runtime_metadata(0, args.seed);

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
    } else if json {
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

fn run_render(args: RenderArgs, json: bool) -> Result<(), TitanError> {
    let has_stats_file = args.stats_json.is_some();
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
    let state = world.dump_state().map_err(|source| {
        TitanError::new(
            "TITAN_STATE_DUMP",
            format!("failed to resolve scene entity IDs: {source}"),
        )
    })?;
    let selection = match args.camera.as_deref() {
        None => CameraSelection::Default,
        Some(value) if value.starts_with("entity:") => {
            let raw = state.entity_ids.get(value).copied().ok_or_else(|| {
                TitanError::from_render(titan_render::RenderError {
                    code: render_error::CAMERA_UNAVAILABLE,
                    message: format!("camera entity '{value}' was not found"),
                    path: Some(format!("camera:{value}")),
                })
            })?;
            CameraSelection::Entity(titan_core::EntityId::from_raw(raw))
        }
        Some(value) => CameraSelection::Name(value.to_owned()),
    };
    // Validate camera selection and all scene-facing render inputs before asking
    // wgpu for an adapter, so diagnostics are useful on adapter-less hosts.
    RenderService::cpu_only()
        .render(
            &world,
            RenderRequest {
                camera: selection.clone(),
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
    let output = RenderOutput {
        ok: true,
        frame: 0,
        camera,
        output: args.out.display().to_string(),
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
    };
    if let Some(path) = args.stats_json {
        let body = serde_json::to_vec(&output).map_err(|source| {
            TitanError::new(
                "TITAN_OUTPUT_SERIALIZE",
                format!("failed to encode render stats: {source}"),
            )
        })?;
        fs::write(&path, body).map_err(|source| {
            TitanError::new(
                "TITAN_STATS_WRITE",
                format!("failed to write {}: {source}", path.display()),
            )
        })?;
    }
    if json && !has_stats_file {
        print_json(&output)?;
    } else {
        println!("rendered {}", args.out.display());
    }
    Ok(())
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
