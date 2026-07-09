mod format;
mod parse;
mod path;
mod validate;

use serde::Serialize;

pub use format::{fmt, fmt_with_registry};
pub use parse::parse;
pub use path::{QueryResult, edit, query};
pub use validate::{validate, validate_with_registry};
pub(crate) use validate::{
    validate_camera_binding, validate_directional_light_binding, validate_material_binding,
    validate_mesh_binding, validate_transform_binding, validate_velocity_binding,
};

pub(crate) const QUATERNION_TOLERANCE: f32 = 1e-5;

pub(crate) fn fits_f32_without_underflow(value: f64) -> bool {
    let converted = value as f32;
    converted.is_finite() && (value == 0.0 || converted != 0.0)
}

/// Returns a deterministic f32 quaternion normalization after validating its f32 norm.
pub(crate) fn normalized_quaternion(rotation: [f32; 4]) -> Option<[f32; 4]> {
    let norm = rotation
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() || (norm - 1.0).abs() > QUATERNION_TOLERANCE {
        return None;
    }

    // Normalize in f64, then round once to f32 for the loader's runtime value.
    let norm = rotation
        .iter()
        .map(|component| f64::from(*component) * f64::from(*component))
        .sum::<f64>()
        .sqrt();
    Some(rotation.map(|component| (f64::from(component) / norm) as f32))
}

pub type TsfResult<T> = Result<T, TsfError>;

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub file: Option<String>,
    pub root: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub kind: ValueKind,
    pub span: Span,
    pub comments: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValueKind {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Vec<Member>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Number {
    pub value: f64,
    pub had_fraction: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub key: String,
    pub key_span: Span,
    pub value: Value,
    pub comments: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DiagnosticSpan {
    pub file: Option<String>,
    pub start: Position,
    pub end: Position,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub path: String,
    pub span: DiagnosticSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TsfError {
    pub errors: Vec<Diagnostic>,
}

impl TsfError {
    pub fn one(
        file: Option<&str>,
        code: impl Into<String>,
        message: impl Into<String>,
        path: impl Into<String>,
        span: Span,
    ) -> Self {
        Self {
            errors: vec![Diagnostic {
                code: code.into(),
                message: message.into(),
                path: path.into(),
                span: DiagnosticSpan {
                    file: file.map(str::to_owned),
                    start: span.start,
                    end: span.end,
                },
            }],
        }
    }

    pub fn many(errors: Vec<Diagnostic>) -> Self {
        Self { errors }
    }
}

impl std::fmt::Display for TsfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(first) = self.errors.first() {
            write!(f, "{}: {}", first.code, first.message)
        } else {
            write!(f, "TSF error")
        }
    }
}

impl std::error::Error for TsfError {}

impl Span {
    fn merge(start: Span, end: Span) -> Self {
        Self {
            start: start.start,
            end: end.end,
        }
    }
}

impl Value {
    pub fn to_json(&self) -> serde_json::Value {
        match &self.kind {
            ValueKind::Null => serde_json::Value::Null,
            ValueKind::Bool(value) => serde_json::Value::Bool(*value),
            ValueKind::Number(number) if !number.had_fraction => {
                let number = if number.value >= 0.0 {
                    serde_json::Number::from(number.value as u64)
                } else {
                    serde_json::Number::from(number.value as i64)
                };
                serde_json::Value::Number(number)
            }
            ValueKind::Number(number) => serde_json::Number::from_f64(number.value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            ValueKind::String(value) => serde_json::Value::String(value.clone()),
            ValueKind::Array(values) => {
                serde_json::Value::Array(values.iter().map(Value::to_json).collect())
            }
            ValueKind::Object(members) => {
                let mut map = serde_json::Map::new();
                for member in members {
                    map.insert(member.key.clone(), member.value.to_json());
                }
                serde_json::Value::Object(map)
            }
        }
    }
}

pub(crate) fn diagnostic(
    file: Option<&str>,
    code: &str,
    message: impl Into<String>,
    path: impl Into<String>,
    span: Span,
) -> Diagnostic {
    Diagnostic {
        code: code.to_owned(),
        message: message.into(),
        path: path.into(),
        span: DiagnosticSpan {
            file: file.map(str::to_owned),
            start: span.start,
            end: span.end,
        },
    }
}

fn json_pointer_escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

pub(crate) fn json_pointer_join(parent: &str, segment: &str) -> String {
    if parent.is_empty() {
        format!("/{}", json_pointer_escape(segment))
    } else {
        format!("{}/{}", parent, json_pointer_escape(segment))
    }
}
