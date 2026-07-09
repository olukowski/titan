use super::{
    Document, Span, TsfError, TsfResult, Value, ValueKind, fmt, json_pointer_escape, parse,
    validate,
};

#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    pub value: serde_json::Value,
    pub span: Span,
    pub resolved_pointer: String,
}

pub fn query(document: &Document, path: &str) -> TsfResult<QueryResult> {
    let (value, resolved_pointer) = resolve(document, path)?;
    Ok(QueryResult {
        value: value.to_json(),
        span: value.span,
        resolved_pointer,
    })
}

pub fn edit(document: &Document, path: &str, new_json5_value: &str) -> TsfResult<String> {
    let replacement = parse(Some("<replacement>"), new_json5_value)?.root;
    let mut edited = document.clone();
    let resolved_pointer = {
        let (_, resolved_pointer) = resolve(&edited, path)?;
        resolved_pointer
    };
    let segments = split_pointer(&resolved_pointer)?;
    let target = resolve_mut(&mut edited.root, &segments, "")?;
    *target = replacement;
    validate(&edited)?;
    Ok(fmt(&edited))
}

fn resolve<'a>(document: &'a Document, path: &str) -> TsfResult<(&'a Value, String)> {
    let segments = split_pointer(path)?;
    let mut value = &document.root;
    let mut resolved = String::new();
    for segment in segments {
        match &value.kind {
            ValueKind::Object(members) => {
                let member = members
                    .iter()
                    .find(|member| member.key == segment)
                    .ok_or_else(|| missing(document.file.as_deref(), path, value.span, &segment))?;
                value = &member.value;
                resolved.push('/');
                resolved.push_str(&json_pointer_escape(&member.key));
            }
            ValueKind::Array(values) => {
                let index = if resolved == "/entities" && segment.starts_with("entity:") {
                    values
                        .iter()
                        .position(|candidate| entity_id(candidate).as_deref() == Some(&segment))
                        .ok_or_else(|| {
                            missing(document.file.as_deref(), path, value.span, &segment)
                        })?
                } else {
                    segment.parse::<usize>().map_err(|_| {
                        missing(document.file.as_deref(), path, value.span, &segment)
                    })?
                };
                value = values
                    .get(index)
                    .ok_or_else(|| missing(document.file.as_deref(), path, value.span, &segment))?;
                resolved.push('/');
                resolved.push_str(&index.to_string());
            }
            _ => {
                return Err(missing(
                    document.file.as_deref(),
                    path,
                    value.span,
                    &segment,
                ));
            }
        }
    }
    Ok((value, resolved))
}

fn resolve_mut<'a>(
    value: &'a mut Value,
    segments: &[String],
    current: &str,
) -> TsfResult<&'a mut Value> {
    if segments.is_empty() {
        return Ok(value);
    }
    let segment = &segments[0];
    match &mut value.kind {
        ValueKind::Object(members) => {
            let member = members
                .iter_mut()
                .find(|member| member.key == *segment)
                .ok_or_else(|| missing(None, current, value.span, segment))?;
            resolve_mut(
                &mut member.value,
                &segments[1..],
                &format!("{current}/{}", json_pointer_escape(segment)),
            )
        }
        ValueKind::Array(values) => {
            let index = segment
                .parse::<usize>()
                .map_err(|_| missing(None, current, value.span, segment))?;
            let next = values
                .get_mut(index)
                .ok_or_else(|| missing(None, current, value.span, segment))?;
            resolve_mut(next, &segments[1..], &format!("{current}/{index}"))
        }
        _ => Err(missing(None, current, value.span, segment)),
    }
}

fn split_pointer(path: &str) -> TsfResult<Vec<String>> {
    if path.is_empty() {
        return Ok(Vec::new());
    }
    if !path.starts_with('/') {
        return Err(TsfError::one(
            None,
            "TSF_INVALID_PATH",
            "JSON Pointer paths must start with '/'",
            path,
            Span::default(),
        ));
    }
    path.split('/')
        .skip(1)
        .map(|segment| {
            let mut out = String::new();
            let mut chars = segment.chars();
            while let Some(ch) = chars.next() {
                if ch == '~' {
                    match chars.next() {
                        Some('0') => out.push('~'),
                        Some('1') => out.push('/'),
                        _ => {
                            return Err(TsfError::one(
                                None,
                                "TSF_INVALID_PATH",
                                "invalid JSON Pointer escape",
                                path,
                                Span::default(),
                            ));
                        }
                    }
                } else {
                    out.push(ch);
                }
            }
            Ok(out)
        })
        .collect()
}

fn entity_id(value: &Value) -> Option<String> {
    let ValueKind::Object(members) = &value.kind else {
        return None;
    };
    members.iter().find_map(|member| {
        if member.key == "id"
            && let ValueKind::String(id) = &member.value.kind
        {
            Some(id.clone())
        } else {
            None
        }
    })
}

fn missing(file: Option<&str>, path: &str, span: Span, segment: &str) -> TsfError {
    TsfError::one(
        file,
        "TSF_PATH_NOT_FOUND",
        format!("path segment '{segment}' was not found"),
        path,
        span,
    )
}
