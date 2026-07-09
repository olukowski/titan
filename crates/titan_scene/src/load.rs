use titan_core::{Component, ComponentRegistry, EntityId, Transform, Velocity, World};

use crate::tsf::{
    Diagnostic, Document, Member, Span, TsfError, TsfResult, Value, ValueKind, json_pointer_join,
    validate,
};

/// Loads a validated TSF document into a deterministic Titan world.
pub fn load_world(document: &Document, registry: ComponentRegistry) -> TsfResult<World> {
    validate(document)?;

    let mut world = World::new(registry);
    let entities = required_array(document, &document.root, "entities", "/entities")?;

    for (index, entity) in entities.iter().enumerate() {
        let path = format!("/entities/{index}");
        let members = object_members(entity).ok_or_else(|| {
            one(
                document,
                "TSF_SCHEMA",
                "entity must be an object",
                &path,
                entity.span,
            )
        })?;
        let scene_id = required_string(document, members, "id", &format!("{path}/id"))?;
        let entity_id = EntityId::from_raw(index as u64 + 1);
        world.spawn_with_id(entity_id).map_err(|error| {
            one(
                document,
                "TSF_LOAD_WORLD",
                error.to_string(),
                &path,
                entity.span,
            )
        })?;
        world
            .bind_scene_entity_id(scene_id.to_owned(), entity_id)
            .map_err(|error| {
                one(
                    document,
                    "TSF_LOAD_WORLD",
                    error.to_string(),
                    format!("{path}/id"),
                    entity.span,
                )
            })?;

        if let Some(components) = member(members, "components") {
            load_components(
                document,
                &mut world,
                entity_id,
                &components.value,
                &format!("{path}/components"),
            )?;
        }
    }

    Ok(world)
}

fn load_components(
    document: &Document,
    world: &mut World,
    entity_id: EntityId,
    value: &Value,
    path: &str,
) -> TsfResult<()> {
    let members = object_members(value).ok_or_else(|| {
        one(
            document,
            "TSF_SCHEMA",
            "components must be an object",
            path,
            value.span,
        )
    })?;
    for component in members {
        let component_path = json_pointer_join(path, &component.key);
        let (registered_name, payload) = match component.key.as_str() {
            "transform" => (
                Transform::NAME,
                transform_payload(document, &component.value, &component_path)?,
            ),
            "velocity" => (
                Velocity::NAME,
                velocity_payload(document, &component.value, &component_path)?,
            ),
            _ => {
                return Err(one(
                    document,
                    "TSF_UNKNOWN_COMPONENT",
                    format!("unknown component '{}'", component.key),
                    component_path,
                    component.key_span,
                ));
            }
        };
        world
            .insert_serialized(entity_id, registered_name, payload)
            .map_err(|error| {
                one(
                    document,
                    "TSF_COMPONENT_DESERIALIZE",
                    error.to_string(),
                    component_path,
                    component.value.span,
                )
            })?;
    }
    Ok(())
}

fn transform_payload(
    document: &Document,
    value: &Value,
    path: &str,
) -> TsfResult<serde_json::Value> {
    let members = object_members(value).ok_or_else(|| {
        one(
            document,
            "TSF_SCHEMA",
            "transform payload must be an object",
            path,
            value.span,
        )
    })?;
    let translation = required_vec3(document, members, "translation", path)?;
    Ok(serde_json::json!({ "translation": translation }))
}

fn velocity_payload(
    document: &Document,
    value: &Value,
    path: &str,
) -> TsfResult<serde_json::Value> {
    let members = object_members(value).ok_or_else(|| {
        one(
            document,
            "TSF_SCHEMA",
            "velocity payload must be an object",
            path,
            value.span,
        )
    })?;
    let linear = required_vec3(document, members, "linear", path)?;
    Ok(serde_json::json!({ "linear": linear }))
}

fn required_vec3(
    document: &Document,
    members: &[Member],
    key: &str,
    parent_path: &str,
) -> TsfResult<serde_json::Value> {
    let path = json_pointer_join(parent_path, key);
    let member = member(members, key).ok_or_else(|| {
        one(
            document,
            "TSF_MISSING_KEY",
            format!("component missing '{key}'"),
            parent_path,
            Span::default(),
        )
    })?;
    let ValueKind::Array(values) = &member.value.kind else {
        return Err(one(
            document,
            "TSF_SCHEMA",
            format!("{key} must be an array"),
            path,
            member.value.span,
        ));
    };
    if values.len() != 3 {
        return Err(one(
            document,
            "TSF_SCHEMA",
            format!("{key} must contain 3 numbers"),
            path,
            member.value.span,
        ));
    }
    let mut out = Vec::with_capacity(3);
    for (index, value) in values.iter().enumerate() {
        let ValueKind::Number(number) = &value.kind else {
            return Err(one(
                document,
                "TSF_SCHEMA",
                format!("{key}[{index}] must be a number"),
                format!("{path}/{index}"),
                value.span,
            ));
        };
        out.push(number.value as f32);
    }
    Ok(serde_json::json!({ "x": out[0], "y": out[1], "z": out[2] }))
}

fn required_array<'a>(
    document: &Document,
    root: &'a Value,
    key: &str,
    path: &str,
) -> TsfResult<&'a [Value]> {
    let root_members = object_members(root).ok_or_else(|| {
        one(
            document,
            "TSF_SCHEMA",
            "top-level TSF document must be an object",
            "",
            root.span,
        )
    })?;
    let entry = member(root_members, key).ok_or_else(|| {
        one(
            document,
            "TSF_MISSING_KEY",
            format!("missing required top-level key '{key}'"),
            "",
            root.span,
        )
    })?;
    match &entry.value.kind {
        ValueKind::Array(values) => Ok(values),
        _ => Err(one(
            document,
            "TSF_SCHEMA",
            format!("{key} must be an array"),
            path,
            entry.value.span,
        )),
    }
}

fn required_string<'a>(
    document: &Document,
    members: &'a [Member],
    key: &str,
    path: &str,
) -> TsfResult<&'a str> {
    let entry = member(members, key).ok_or_else(|| {
        one(
            document,
            "TSF_MISSING_KEY",
            format!("{key} is required"),
            path,
            Span::default(),
        )
    })?;
    match &entry.value.kind {
        ValueKind::String(value) => Ok(value),
        _ => Err(one(
            document,
            "TSF_SCHEMA",
            format!("{key} must be a string"),
            path,
            entry.value.span,
        )),
    }
}

fn object_members(value: &Value) -> Option<&[Member]> {
    match &value.kind {
        ValueKind::Object(members) => Some(members),
        _ => None,
    }
}

fn member<'a>(members: &'a [Member], key: &str) -> Option<&'a Member> {
    members.iter().find(|member| member.key == key)
}

fn one(
    document: &Document,
    code: impl Into<String>,
    message: impl Into<String>,
    path: impl Into<String>,
    span: Span,
) -> TsfError {
    TsfError::many(vec![Diagnostic {
        code: code.into(),
        message: message.into(),
        path: path.into(),
        span: crate::tsf::DiagnosticSpan {
            file: document.file.clone(),
            start: span.start,
            end: span.end,
        },
    }])
}
