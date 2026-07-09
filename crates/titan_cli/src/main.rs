use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand, error::ErrorKind};
use serde::Serialize;

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
    /// Print version information.
    Version,
}

#[derive(Debug)]
struct TitanError {
    code: &'static str,
    message: String,
}

impl TitanError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
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
