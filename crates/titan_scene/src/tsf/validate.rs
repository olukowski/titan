use std::collections::HashSet;
use std::path::{Component, Path};

use super::{
    Diagnostic, Document, Member, Span, TsfError, TsfResult, Value, ValueKind, diagnostic,
    json_pointer_join,
};

pub fn validate(document: &Document) -> TsfResult<()> {
    let mut validator = Validator {
        file: document.file.as_deref(),
        errors: Vec::new(),
        asset_aliases: HashSet::new(),
        entity_ids: HashSet::new(),
    };
    validator.validate_document(document);
    if validator.errors.is_empty() {
        Ok(())
    } else {
        Err(TsfError::many(validator.errors))
    }
}

struct Validator<'a> {
    file: Option<&'a str>,
    errors: Vec<Diagnostic>,
    asset_aliases: HashSet<String>,
    entity_ids: HashSet<String>,
}

impl Validator<'_> {
    fn validate_document(&mut self, document: &Document) {
        let root = match object_members(&document.root) {
            Some(members) => members,
            None => {
                self.push(
                    "TSF_SCHEMA",
                    "top-level TSF document must be an object",
                    "",
                    document.root.span,
                );
                return;
            }
        };

        for key in ["tsf", "scene", "assets", "entities"] {
            if member(root, key).is_none() {
                self.push(
                    "TSF_MISSING_KEY",
                    format!("missing required top-level key '{key}'"),
                    "",
                    document.root.span,
                );
            }
        }

        if let Some(tsf) = member(root, "tsf") {
            match &tsf.value.kind {
                ValueKind::Number(number) if number.value == 1.0 && number.value.fract() == 0.0 => {
                }
                _ => self.push(
                    "TSF_SCHEMA",
                    "tsf must be integer version 1",
                    "/tsf",
                    tsf.value.span,
                ),
            }
        }

        if let Some(scene) = member(root, "scene") {
            self.validate_scene(&scene.value);
        }
        if let Some(assets) = member(root, "assets") {
            self.collect_assets(&assets.value);
        }
        if let Some(entities) = member(root, "entities") {
            self.collect_entities(&entities.value);
        }
        if let Some(assets) = member(root, "assets") {
            self.validate_assets(&assets.value);
        }
        if let Some(entities) = member(root, "entities") {
            self.validate_entities(&entities.value);
        }
        self.validate_numbers(&document.root, "");
        self.validate_refs(&document.root, "");
    }

    fn validate_scene(&mut self, value: &Value) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "scene must be an object",
                "/scene",
                value.span,
            );
            return;
        };
        match member(members, "id") {
            Some(id) if string_value(&id.value).is_some_and(|id| id.starts_with("scene:")) => {}
            Some(id) => self.push(
                "TSF_INVALID_ID",
                "scene.id must be a scene: id string",
                "/scene/id",
                id.value.span,
            ),
            None => self.push(
                "TSF_MISSING_KEY",
                "scene.id is required",
                "/scene",
                value.span,
            ),
        }
    }

    fn collect_assets(&mut self, value: &Value) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "assets must be an object",
                "/assets",
                value.span,
            );
            return;
        };
        for asset in members {
            self.asset_aliases.insert(asset.key.clone());
        }
    }

    fn validate_assets(&mut self, value: &Value) {
        let Some(members) = object_members(value) else {
            return;
        };
        for asset in members {
            let path = json_pointer_join("/assets", &asset.key);
            let Some(asset_members) = object_members(&asset.value) else {
                self.push(
                    "TSF_SCHEMA",
                    "asset entry must be an object",
                    path,
                    asset.value.span,
                );
                continue;
            };
            for required in ["path", "kind"] {
                match member(asset_members, required) {
                    Some(entry) if matches!(entry.value.kind, ValueKind::String(_)) => {}
                    Some(entry) => self.push(
                        "TSF_SCHEMA",
                        format!("asset.{required} must be a string"),
                        json_pointer_join(&path, required),
                        entry.value.span,
                    ),
                    None => self.push(
                        "TSF_MISSING_KEY",
                        format!("asset entry missing '{required}'"),
                        path.clone(),
                        asset.value.span,
                    ),
                }
            }
            if let Some(entry) = member(asset_members, "path")
                && let Some(path_value) = string_value(&entry.value)
                && !valid_relative_path(path_value)
            {
                self.push(
                    "TSF_SCHEMA",
                    "asset.path must be a relative normalized path",
                    json_pointer_join(&path, "path"),
                    entry.value.span,
                );
            }
        }
    }

    fn collect_entities(&mut self, value: &Value) {
        let ValueKind::Array(values) = &value.kind else {
            self.push(
                "TSF_SCHEMA",
                "entities must be an array",
                "/entities",
                value.span,
            );
            return;
        };
        for entity in values {
            let Some(members) = object_members(entity) else {
                self.push(
                    "TSF_SCHEMA",
                    "entity must be an object",
                    "/entities",
                    entity.span,
                );
                continue;
            };
            if let Some(id) = member(members, "id") {
                if let Some(id_value) = string_value(&id.value) {
                    if !valid_entity_id(id_value) {
                        self.push(
                            "TSF_INVALID_ID",
                            "entity id must be in entity:<slug> form",
                            "/entities",
                            id.value.span,
                        );
                    } else if !self.entity_ids.insert(id_value.to_owned()) {
                        self.push(
                            "TSF_DUPLICATE_ENTITY",
                            format!("duplicate entity id '{id_value}'"),
                            format!("/entities/{id_value}"),
                            id.value.span,
                        );
                    }
                } else {
                    self.push(
                        "TSF_SCHEMA",
                        "entity.id must be a string",
                        "/entities",
                        id.value.span,
                    );
                }
            } else {
                self.push(
                    "TSF_MISSING_KEY",
                    "entity.id is required",
                    "/entities",
                    entity.span,
                );
            }
        }
    }

    fn validate_entities(&mut self, value: &Value) {
        let ValueKind::Array(values) = &value.kind else {
            return;
        };
        for (index, entity) in values.iter().enumerate() {
            let entity_path = entity_path(entity, index);
            let Some(members) = object_members(entity) else {
                continue;
            };
            if let Some(parent) = member(members, "parent") {
                self.validate_parent(&parent.value, &format!("{entity_path}/parent"));
            }
            if let Some(components) = member(members, "components") {
                self.validate_components(&components.value, &format!("{entity_path}/components"));
            }
        }
    }

    fn validate_parent(&mut self, value: &Value, path: &str) {
        let Some(parent) = string_value(value) else {
            self.push(
                "TSF_SCHEMA",
                "entity.parent must be a string",
                path,
                value.span,
            );
            return;
        };
        if !valid_entity_id(parent) {
            self.push(
                "TSF_INVALID_ID",
                "entity.parent must be an entity:<slug> id string",
                path,
                value.span,
            );
        } else if !self.entity_ids.contains(parent) {
            self.push(
                "TSF_BROKEN_REF",
                format!("entity parent '{parent}' does not point at an entity in this file"),
                path,
                value.span,
            );
        }
    }

    fn validate_components(&mut self, value: &Value, path: &str) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "components must be an object",
                path,
                value.span,
            );
            return;
        };
        for component in members {
            let component_path = json_pointer_join(path, &component.key);
            match component.key.as_str() {
                "transform" => self.validate_transform(&component.value, &component_path),
                "velocity" => self.validate_velocity(&component.value, &component_path),
                "mesh" | "camera" | "light" => {
                    if object_members(&component.value).is_none() {
                        self.push(
                            "TSF_SCHEMA",
                            "component payload must be an object",
                            component_path,
                            component.value.span,
                        );
                    }
                }
                _ => self.push(
                    "TSF_UNKNOWN_COMPONENT",
                    format!("unknown component '{}'", component.key),
                    component_path,
                    component.key_span,
                ),
            }
        }
    }

    fn validate_transform(&mut self, value: &Value, path: &str) {
        self.validate_payload_vectors(
            value,
            path,
            &[("translation", 3), ("rotation", 4), ("scale", 3)],
            "transform",
        );
    }

    fn validate_velocity(&mut self, value: &Value, path: &str) {
        self.validate_payload_vectors(value, path, &[("linear", 3), ("angular", 3)], "velocity");
    }

    fn validate_payload_vectors(
        &mut self,
        value: &Value,
        path: &str,
        fields: &[(&str, usize)],
        component: &str,
    ) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                format!("{component} payload must be an object"),
                path,
                value.span,
            );
            return;
        };
        for (field, len) in fields {
            match member(members, field) {
                Some(entry) => self.validate_number_array(
                    &entry.value,
                    &json_pointer_join(path, field),
                    *len,
                    &format!("{component}.{field}"),
                ),
                None => self.push(
                    "TSF_MISSING_KEY",
                    format!("{component} missing '{field}'"),
                    path,
                    value.span,
                ),
            }
        }
    }

    fn validate_number_array(&mut self, value: &Value, path: &str, len: usize, label: &str) {
        let ValueKind::Array(values) = &value.kind else {
            self.push(
                "TSF_SCHEMA",
                format!("{label} must be an array"),
                path,
                value.span,
            );
            return;
        };
        if values.len() != len {
            self.push(
                "TSF_SCHEMA",
                format!("{label} must contain {len} numbers"),
                path,
                value.span,
            );
        }
        for (index, value) in values.iter().enumerate() {
            if !matches!(value.kind, ValueKind::Number(_)) {
                self.push(
                    "TSF_SCHEMA",
                    format!("{label}[{index}] must be a number"),
                    format!("{path}/{index}"),
                    value.span,
                );
            }
        }
    }

    fn validate_refs(&mut self, value: &Value, path: &str) {
        match &value.kind {
            ValueKind::Object(members) => {
                if path != "/assets"
                    && let Some(ref_member) = member(members, "ref")
                {
                    if let Some(target) = string_value(&ref_member.value) {
                        self.validate_ref(target, path, ref_member.value.span);
                    } else {
                        self.push(
                            "TSF_SCHEMA",
                            "ref must be a string",
                            json_pointer_join(path, "ref"),
                            ref_member.value.span,
                        );
                    }
                    if members.len() != 1 {
                        self.push(
                            "TSF_SCHEMA",
                            "ref object may not contain other keys",
                            path,
                            value.span,
                        );
                    }
                    return;
                }
                for member in members {
                    self.validate_refs(&member.value, &json_pointer_join(path, &member.key));
                }
            }
            ValueKind::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    self.validate_refs(value, &format!("{path}/{index}"));
                }
            }
            _ => {}
        }
    }

    fn validate_ref(&mut self, target: &str, path: &str, span: Span) {
        if let Some(alias) = target.strip_prefix("asset:") {
            if !self.asset_aliases.contains(alias) {
                self.push(
                    "TSF_BROKEN_REF",
                    format!("asset reference '{target}' does not point at a declared asset"),
                    format!("{path}/ref"),
                    span,
                );
            }
        } else if valid_external_ref(target) {
        } else if target.starts_with("entity:") {
            if !self.entity_ids.contains(target) {
                self.push(
                    "TSF_BROKEN_REF",
                    format!("entity reference '{target}' does not point at an entity in this file"),
                    format!("{path}/ref"),
                    span,
                );
            }
        } else {
            self.push(
                "TSF_BROKEN_REF",
                format!("reference '{target}' has an invalid prefix"),
                format!("{path}/ref"),
                span,
            );
        }
    }

    fn validate_numbers(&mut self, value: &Value, path: &str) {
        match &value.kind {
            ValueKind::Number(number) if !number.value.is_finite() => self.push(
                "TSF_INVALID_NUMBER",
                "number must be finite",
                path,
                value.span,
            ),
            ValueKind::Object(members) => {
                for member in members {
                    self.validate_numbers(&member.value, &json_pointer_join(path, &member.key));
                }
            }
            ValueKind::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    self.validate_numbers(value, &format!("{path}/{index}"));
                }
            }
            _ => {}
        }
    }

    fn push(
        &mut self,
        code: &str,
        message: impl Into<String>,
        path: impl Into<String>,
        span: Span,
    ) {
        self.errors
            .push(diagnostic(self.file, code, message, path, span));
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

fn string_value(value: &Value) -> Option<&str> {
    match &value.kind {
        ValueKind::String(value) => Some(value),
        _ => None,
    }
}

fn valid_entity_id(value: &str) -> bool {
    let Some(slug) = value.strip_prefix("entity:") else {
        return false;
    };
    valid_entity_slug(slug)
}

fn valid_entity_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn valid_external_ref(value: &str) -> bool {
    if let Some(path) = value.strip_prefix("file:") {
        return valid_relative_path(path);
    }
    let Some(target) = value.strip_prefix("scene:") else {
        return false;
    };
    let Some((path, slug)) = target.split_once("#entity:") else {
        return false;
    };
    valid_relative_path(path) && valid_entity_slug(slug)
}

fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.contains(':')
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn entity_path(entity: &Value, index: usize) -> String {
    if let Some(members) = object_members(entity)
        && let Some(id) = member(members, "id")
        && let Some(id) = string_value(&id.value)
    {
        return format!("/entities/{id}");
    }
    format!("/entities/{index}")
}
