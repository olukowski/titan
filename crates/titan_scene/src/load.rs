use std::collections::BTreeMap;

use titan_core::{ComponentRegistry, EntityId, World};

use crate::registry::registry_for_core;
use crate::tsf::{
    Diagnostic, Document, Member, Span, TsfError, TsfResult, Value, ValueKind,
    fits_f32_without_underflow, json_pointer_join, normalized_quaternion, validate,
};

/// Loads a validated TSF document into a deterministic Titan world.
pub fn load_world(document: &Document, registry: ComponentRegistry) -> TsfResult<World> {
    validate(document)?;

    let tsf_registry = registry_for_core(registry);
    let mut world = World::new(tsf_registry.component_registry().clone());
    let entities = required_array(document, &document.root, "entities", "/entities")?;
    let entity_ids = scene_entity_ids(document, entities)?;

    let mut ordered_entities = Vec::new();
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
        let entity_id = *entity_ids.get(scene_id).ok_or_else(|| {
            one(
                document,
                "TSF_LOAD_WORLD",
                format!("missing runtime entity id for '{scene_id}'"),
                format!("{path}/id"),
                entity.span,
            )
        })?;
        ordered_entities.push((scene_id.to_owned(), entity_id, path, members, entity.span));
    }

    ordered_entities.sort_by(|left, right| left.0.cmp(&right.0));
    for (scene_id, entity_id, path, members, entity_span) in ordered_entities {
        world.spawn_with_id(entity_id).map_err(|error| {
            one(
                document,
                "TSF_LOAD_WORLD",
                error.to_string(),
                &path,
                entity_span,
            )
        })?;
        world
            .bind_scene_entity_id(scene_id, entity_id)
            .map_err(|error| {
                one(
                    document,
                    "TSF_LOAD_WORLD",
                    error.to_string(),
                    format!("{path}/id"),
                    entity_span,
                )
            })?;

        if let Some(components) = member(members, "components") {
            load_components(
                document,
                &mut world,
                entity_id,
                &components.value,
                &format!("{path}/components"),
                &tsf_registry,
            )?;
        }
    }

    Ok(world)
}

fn scene_entity_ids(
    document: &Document,
    entities: &[Value],
) -> TsfResult<BTreeMap<String, EntityId>> {
    let mut scene_ids = BTreeMap::new();
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
        scene_ids.insert(scene_id.to_owned(), EntityId::from_raw(0));
    }

    for (index, entity_id) in scene_ids.values_mut().enumerate() {
        *entity_id = EntityId::from_raw(index as u64 + 1);
    }
    Ok(scene_ids)
}

fn load_components(
    document: &Document,
    world: &mut World,
    entity_id: EntityId,
    value: &Value,
    path: &str,
    tsf_registry: &crate::registry::TsfComponentRegistry,
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
    let mut loaded = Vec::new();
    for component in members {
        let component_path = json_pointer_join(path, &component.key);
        let Some(binding) = tsf_registry.binding(&component.key) else {
            return Err(one(
                document,
                "TSF_UNKNOWN_COMPONENT",
                format!("unknown component '{}'", component.key),
                component_path,
                component.key_span,
            ));
        };
        if tsf_registry
            .component_registry()
            .meta_by_name(binding.registered_name)
            .is_err()
        {
            return Err(one(
                document,
                "TSF_COMPONENT_NOT_REGISTERED",
                format!(
                    "component alias '{}' maps to unregistered type '{}'",
                    component.key, binding.registered_name
                ),
                component_path,
                component.key_span,
            ));
        }
        let payload = match component.key.as_str() {
            "transform" => transform_payload(document, &component.value, &component_path)?,
            "velocity" => velocity_payload(document, &component.value, &component_path)?,
            _ => component.value.to_json(),
        };
        loaded.push((
            binding.registered_name,
            payload,
            component_path,
            component.value.span,
        ));
    }

    loaded.sort_by_key(|(_, _, path, _)| {
        let alias = path.rsplit('/').next().unwrap_or_default();
        crate::registry::BUILTIN_COMPONENT_ORDER
            .iter()
            .position(|item| *item == alias)
            .unwrap_or(usize::MAX)
    });
    for (registered_name, payload, component_path, value_span) in loaded {
        world
            .insert_serialized(entity_id, registered_name, payload)
            .map_err(|error| {
                one(
                    document,
                    "TSF_COMPONENT_DESERIALIZE",
                    error.to_string(),
                    component_path,
                    value_span,
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
    let rotation = optional_quaternion(document, members, path)?;
    Ok(serde_json::json!({
        "translation": translation,
        "rotation": rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]),
    }))
}

fn optional_quaternion(
    document: &Document,
    members: &[Member],
    parent_path: &str,
) -> TsfResult<Option<[f32; 4]>> {
    let Some(entry) = member(members, "rotation") else {
        return Ok(None);
    };
    let path = json_pointer_join(parent_path, "rotation");
    let ValueKind::Array(values) = &entry.value.kind else {
        return Err(one(
            document,
            "TSF_SCHEMA",
            "rotation must be an array",
            path,
            entry.value.span,
        ));
    };
    if values.len() != 4 {
        return Err(one(
            document,
            "TSF_SCHEMA",
            "rotation must contain 4 numbers",
            path,
            entry.value.span,
        ));
    }
    let mut rotation = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        let ValueKind::Number(number) = &value.kind else {
            return Err(one(
                document,
                "TSF_SCHEMA",
                format!("rotation[{index}] must be a number"),
                format!("{path}/{index}"),
                value.span,
            ));
        };
        let converted = number.value as f32;
        if !number.value.is_finite() || !fits_f32_without_underflow(number.value) {
            return Err(one(
                document,
                "TSF_INVALID_NUMBER",
                if !number.value.is_finite() {
                    format!("rotation[{index}] must be finite")
                } else {
                    format!("rotation[{index}] must fit in f32 without underflow")
                },
                format!("{path}/{index}"),
                value.span,
            ));
        }
        rotation[index] = converted;
    }
    let Some(rotation) = normalized_quaternion(rotation) else {
        return Err(one(
            document,
            "TSF_INVALID_QUATERNION",
            "rotation quaternion must have unit norm within 1e-5",
            path,
            entry.value.span,
        ));
    };
    Ok(Some(rotation))
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
        let converted = number.value as f32;
        if !converted.is_finite() || (number.value != 0.0 && converted == 0.0) {
            return Err(one(
                document,
                "TSF_INVALID_NUMBER",
                format!("{key}[{index}] must fit in f32 without underflow"),
                format!("{path}/{index}"),
                value.span,
            ));
        }
        out.push(converted);
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
