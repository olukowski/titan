use std::{fs, path::PathBuf, process::ExitCode};

use clap::{CommandFactory, Parser, Subcommand, error::ErrorKind};
use serde::Serialize;
use titan_core::{
    DEFAULT_FIXED_DT, DEFAULT_RUN_SEED, FixedStepContext, Schedule, phase1_component_registry,
    velocity_integration_system,
};
use titan_scene::{Diagnostic, load_world, parse};

const APP_NAME: &str = "titan";

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
    /// Print version information.
    Version,
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

#[derive(Debug)]
struct TitanError {
    code: &'static str,
    message: String,
    diagnostics: Option<Vec<Diagnostic>>,
}

impl TitanError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            diagnostics: None,
        }
    }

    fn from_tsf(message: impl Into<String>, error: titan_scene::TsfError) -> Self {
        Self {
            code: "TITAN_TSF_ERROR",
            message: message.into(),
            diagnostics: Some(error.errors),
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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            report_error(&error);
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), TitanError> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            error.print().map_err(|source| {
                TitanError::new(
                    "TITAN_CLI_HELP_RENDER",
                    format!("failed to render help: {source}"),
                )
            })?;
            return Ok(());
        }
        Err(error) => return Err(TitanError::from_clap(error)),
    };

    if cli.version {
        return print_version(cli.json);
    }

    match cli.command {
        Some(Command::Run(args)) => run_scene(args, cli.json),
        Some(Command::Version) => print_version(cli.json),
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
    }
}

fn run_scene(args: RunArgs, json: bool) -> Result<(), TitanError> {
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
            location: None,
            diagnostics: error.diagnostics.as_deref(),
        },
    };

    match serde_json::to_string(&envelope) {
        Ok(body) => eprintln!("{body}"),
        Err(_) => eprintln!("{}: {}", error.code, error.message),
    }
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
