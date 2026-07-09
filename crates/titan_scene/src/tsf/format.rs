use super::{
    Document, Member, Number, Value, ValueKind, fits_f32_without_underflow, normalized_quaternion,
};
use crate::registry::TsfComponentRegistry;

pub fn fmt(document: &Document) -> String {
    let registry = crate::registry::phase2_component_registry()
        .expect("built-in TSF registry must be constructible");
    fmt_with_registry(document, &registry)
}

pub fn fmt_with_registry(document: &Document, registry: &TsfComponentRegistry) -> String {
    let mut out = String::new();
    write_value(&mut out, &document.root, 0, Context::Top, registry);
    out.push('\n');
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context<'a> {
    Top,
    Scene,
    Assets,
    Asset,
    Entities,
    Entity,
    Components,
    Component(&'a str),
    Other,
}

fn write_value(
    out: &mut String,
    value: &Value,
    indent: usize,
    context: Context<'_>,
    registry: &TsfComponentRegistry,
) {
    write_comments(out, &value.comments, indent);
    match &value.kind {
        ValueKind::Null => out.push_str("null"),
        ValueKind::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        ValueKind::Number(number) => out.push_str(&format_number(number)),
        ValueKind::String(value) => write_string(out, value),
        ValueKind::Array(values) => write_array(out, values, indent, context, registry),
        ValueKind::Object(members) => write_object(out, members, indent, context, registry),
    }
}

fn write_array(
    out: &mut String,
    values: &[Value],
    indent: usize,
    context: Context<'_>,
    registry: &TsfComponentRegistry,
) {
    if values.is_empty() {
        out.push_str("[]");
        return;
    }
    if values.len() <= 4 && values.iter().all(is_scalar_without_comments) {
        out.push('[');
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            write_value(out, value, indent, Context::Other, registry);
        }
        out.push(']');
        return;
    }
    out.push_str("[\n");
    let child_context = if context == Context::Entities {
        Context::Entity
    } else {
        Context::Other
    };
    for value in values {
        push_indent(out, indent + 2);
        write_value(out, value, indent + 2, child_context, registry);
        out.push_str(",\n");
    }
    push_indent(out, indent);
    out.push(']');
}

fn write_object(
    out: &mut String,
    members: &[Member],
    indent: usize,
    context: Context<'_>,
    registry: &TsfComponentRegistry,
) {
    if members.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push_str("{\n");
    for member in ordered_members(members, context, registry) {
        write_comments(out, &member.comments, indent + 2);
        push_indent(out, indent + 2);
        write_key(out, &member.key);
        out.push_str(": ");
        if context == Context::Component("transform") && member.key == "rotation" {
            write_value(out, &member.value, indent + 2, Context::Other, registry);
        } else {
            write_value(
                out,
                &member.value,
                indent + 2,
                child_context(context, &member.key, registry),
                registry,
            );
        }
        out.push_str(",\n");
    }
    push_indent(out, indent);
    out.push('}');
}

fn child_context<'a>(
    context: Context<'a>,
    key: &'a str,
    registry: &TsfComponentRegistry,
) -> Context<'a> {
    match context {
        Context::Top => match key {
            "scene" => Context::Scene,
            "assets" => Context::Assets,
            "entities" => Context::Entities,
            _ => Context::Other,
        },
        Context::Assets => Context::Asset,
        Context::Entity if key == "components" => Context::Components,
        Context::Components if registry.binding(key).is_some() => Context::Component(key),
        _ => Context::Other,
    }
}

fn ordered_members<'a>(
    members: &'a [Member],
    context: Context<'_>,
    registry: &TsfComponentRegistry,
) -> Vec<&'a Member> {
    let mut ordered: Vec<_> = members
        .iter()
        .filter(|member| {
            !(context == Context::Component("transform")
                && member.key == "rotation"
                && rotation_is_identity(&member.value))
        })
        .collect();
    ordered.sort_by(|a, b| {
        let ar = rank(context, &a.key, registry);
        let br = rank(context, &b.key, registry);
        ar.cmp(&br).then_with(|| a.key.cmp(&b.key))
    });
    ordered
}

fn rotation_is_identity(value: &Value) -> bool {
    let Some(rotation) = normalized_rotation(value) else {
        return false;
    };
    rotation == [0.0, 0.0, 0.0, 1.0]
}

fn normalized_rotation(value: &Value) -> Option<[f32; 4]> {
    let ValueKind::Array(values) = &value.kind else {
        return None;
    };
    if values.len() != 4 {
        return None;
    }
    let mut rotation = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        let ValueKind::Number(number) = &value.kind else {
            return None;
        };
        if !fits_f32_without_underflow(number.value) {
            return None;
        }
        rotation[index] = number.value as f32;
        if !rotation[index].is_finite() {
            return None;
        }
    }
    normalized_quaternion(rotation)
}

fn rank(context: Context<'_>, key: &str, registry: &TsfComponentRegistry) -> usize {
    match context {
        Context::Top => ["tsf", "scene", "assets", "entities"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Scene => ["id", "name", "metadata"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Entity => ["id", "name", "parent", "components"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Components => registry
            .component_order()
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Component("transform") => ["translation", "rotation"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Component("velocity") => ["linear"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Component("camera") => [
            "projection",
            "vertical_fov_degrees",
            "height",
            "near",
            "far",
            "viewport",
        ]
        .iter()
        .position(|candidate| *candidate == key)
        .unwrap_or(1000),
        Context::Component("directional_light") => ["color", "illuminance", "ambient"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Component("mesh") => ["geometry", "submeshes"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        Context::Component("material") => ["model", "base_color", "metallic", "roughness"]
            .iter()
            .position(|candidate| *candidate == key)
            .unwrap_or(1000),
        _ => 1000,
    }
}

fn is_scalar_without_comments(value: &Value) -> bool {
    value.comments.is_empty()
        && matches!(
            value.kind,
            ValueKind::Null | ValueKind::Bool(_) | ValueKind::Number(_) | ValueKind::String(_)
        )
}

fn write_comments(out: &mut String, comments: &[String], indent: usize) {
    for comment in comments {
        for line in comment.lines() {
            push_indent(out, indent);
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }
}

fn write_key(out: &mut String, key: &str) {
    if is_simple_key(key) {
        out.push_str(key);
    } else {
        write_string(out, key);
    }
}

fn write_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn format_number(number: &Number) -> String {
    let value = if number.value == 0.0 {
        0.0
    } else {
        number.value
    };
    if value.fract() == 0.0 {
        if number.had_fraction {
            format!("{value:.1}")
        } else {
            format!("{value:.0}")
        }
    } else if value.abs() < 0.000001 {
        format!("{value:e}")
    } else {
        let mut text = value.to_string();
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

fn is_simple_key(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(ch) if ch == '_' || ch == '$' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
}
