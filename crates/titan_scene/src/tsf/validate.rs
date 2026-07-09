use std::collections::HashSet;
use std::path::{Component, Path};

use super::{
    Diagnostic, Document, Member, Span, TsfError, TsfResult, Value, ValueKind, diagnostic,
    fits_f32_without_underflow, json_pointer_join, normalized_quaternion,
};
use crate::registry::{Diagnostics, TsfComponentRegistry};

pub fn validate(document: &Document) -> TsfResult<()> {
    let registry = crate::registry::phase2_component_registry()
        .expect("built-in TSF registry must be constructible");
    validate_with_registry(document, &registry)
}

pub fn validate_with_registry(
    document: &Document,
    registry: &TsfComponentRegistry,
) -> TsfResult<()> {
    let mut validator = Validator {
        file: document.file.as_deref(),
        errors: Vec::new(),
        asset_aliases: HashSet::new(),
        entity_ids: HashSet::new(),
        registry: Some(registry),
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
    registry: Option<&'a TsfComponentRegistry>,
}

impl Validator<'_> {
    fn validate_document(&mut self, document: &Document) {
        self.validate_unique_keys(&document.root, "");

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

    fn validate_unique_keys(&mut self, value: &Value, path: &str) {
        match &value.kind {
            ValueKind::Object(members) => {
                let mut keys = HashSet::new();
                for member in members {
                    if !keys.insert(member.key.as_str()) {
                        self.push(
                            "TSF_DUPLICATE_KEY",
                            format!("duplicate object key '{}'", member.key),
                            path,
                            member.key_span,
                        );
                    }
                    self.validate_unique_keys(&member.value, &json_pointer_join(path, &member.key));
                }
            }
            ValueKind::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    self.validate_unique_keys(value, &format!("{path}/{index}"));
                }
            }
            _ => {}
        }
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
            let Some(binding) = self
                .registry
                .expect("document validator has a registry")
                .binding(&component.key)
            else {
                self.push(
                    "TSF_UNKNOWN_COMPONENT",
                    format!("unknown component '{}'", component.key),
                    component_path,
                    component.key_span,
                );
                continue;
            };
            let diagnostics_start = self.errors.len();
            (binding.validate)(&component.value, &component_path, &mut self.errors);
            for diagnostic in &mut self.errors[diagnostics_start..] {
                if diagnostic.span.file.is_none() {
                    diagnostic.span.file = self.file.map(str::to_owned);
                }
            }
        }
    }

    fn validate_transform(&mut self, value: &Value, path: &str) {
        self.validate_payload_vectors(
            value,
            path,
            &[("translation", 3)],
            &["rotation"],
            "transform",
        );
        if let Some(members) = object_members(value)
            && let Some(rotation) = member(members, "rotation")
        {
            self.validate_quaternion(&rotation.value, &json_pointer_join(path, "rotation"));
        }
    }

    fn validate_velocity(&mut self, value: &Value, path: &str) {
        self.validate_payload_vectors(value, path, &[("linear", 3)], &[], "velocity");
    }

    fn validate_camera(&mut self, value: &Value, path: &str) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "camera payload must be an object",
                path,
                value.span,
            );
            return;
        };
        let Some(projection) = member(members, "projection") else {
            self.push(
                "TSF_MISSING_KEY",
                "camera missing 'projection'",
                path,
                value.span,
            );
            return;
        };
        let Some(kind) = string_value(&projection.value) else {
            self.push(
                "TSF_SCHEMA",
                "camera.projection must be a string",
                json_pointer_join(path, "projection"),
                projection.value.span,
            );
            return;
        };
        let required = match kind {
            "perspective" => &["vertical_fov_degrees", "near", "far"][..],
            "orthographic" => &["height", "near", "far"][..],
            _ => {
                self.push(
                    "TSF_SCHEMA",
                    "camera.projection must be perspective or orthographic",
                    json_pointer_join(path, "projection"),
                    projection.value.span,
                );
                return;
            }
        };
        let allowed = match kind {
            "perspective" => &[
                "projection",
                "vertical_fov_degrees",
                "near",
                "far",
                "viewport",
            ][..],
            "orthographic" => &["projection", "height", "near", "far", "viewport"][..],
            _ => unreachable!(),
        };
        for field in required {
            self.validate_scalar(members, field, path, "camera");
        }
        for field in required
            .iter()
            .filter(|field| **field != "near" && **field != "far")
        {
            self.validate_positive_scalar(members, field, path, "camera");
        }
        if let (Some(near), Some(far)) = (
            number_member(members, "near"),
            number_member(members, "far"),
        ) && far <= near
        {
            self.push(
                "TSF_SCHEMA",
                "camera.far must be greater than camera.near",
                json_pointer_join(path, "far"),
                member(members, "far")
                    .expect("number member exists")
                    .value
                    .span,
            );
        }
        self.validate_optional_viewport(members, path);
        self.reject_unknown(members, allowed, path, "camera");
    }

    fn validate_directional_light(&mut self, value: &Value, path: &str) {
        self.validate_payload_vectors(
            value,
            path,
            &[("color", 3)],
            &["illuminance", "ambient"],
            "directional_light",
        );
        if let Some(members) = object_members(value) {
            self.validate_scalar(members, "illuminance", path, "directional_light");
            self.validate_scalar(members, "ambient", path, "directional_light");
            self.validate_nonnegative_scalar(members, "illuminance", path, "directional_light");
            self.validate_nonnegative_scalar(members, "ambient", path, "directional_light");
            self.validate_range_array_member(members, "color", path, "directional_light.color");
            self.reject_unknown(
                members,
                &["color", "illuminance", "ambient"],
                path,
                "directional_light",
            );
        }
    }

    fn validate_mesh(&mut self, value: &Value, path: &str) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "mesh payload must be an object",
                path,
                value.span,
            );
            return;
        };
        let Some(geometry) = member(members, "geometry") else {
            self.push(
                "TSF_MISSING_KEY",
                "mesh missing 'geometry'",
                path,
                value.span,
            );
            return;
        };
        let valid_reference = object_members(&geometry.value).is_some_and(|members| {
            members.len() == 1
                && member(members, "ref")
                    .and_then(|entry| string_value(&entry.value))
                    .is_some()
        });
        if !valid_reference {
            self.push(
                "TSF_SCHEMA",
                "mesh.geometry must be a reference object",
                json_pointer_join(path, "geometry"),
                geometry.value.span,
            );
        }
        if let Some(submeshes) = member(members, "submeshes") {
            self.validate_integer_array(&submeshes.value, &json_pointer_join(path, "submeshes"));
        }
        self.reject_unknown(members, &["geometry", "submeshes"], path, "mesh");
    }

    fn validate_material(&mut self, value: &Value, path: &str) {
        let Some(members) = object_members(value) else {
            self.push(
                "TSF_SCHEMA",
                "material payload must be an object",
                path,
                value.span,
            );
            return;
        };
        let Some(model_entry) = member(members, "model") else {
            self.push(
                "TSF_MISSING_KEY",
                "material missing 'model'",
                path,
                value.span,
            );
            return;
        };
        let Some(model) = string_value(&model_entry.value) else {
            self.push(
                "TSF_SCHEMA",
                "material.model must be a string",
                json_pointer_join(path, "model"),
                model_entry.value.span,
            );
            return;
        };
        if !matches!(model, "unlit" | "pbr") {
            self.push(
                "TSF_SCHEMA",
                "material.model must be unlit or pbr",
                json_pointer_join(path, "model"),
                model_entry.value.span,
            );
        }
        self.validate_number_array_member(members, "base_color", path, 4, "material.base_color");
        self.validate_range_array_member(members, "base_color", path, "material.base_color");
        if model == "pbr" {
            self.validate_scalar(members, "metallic", path, "material");
            self.validate_scalar(members, "roughness", path, "material");
        }
        if model == "unlit" {
            for field in ["metallic", "roughness"] {
                if let Some(entry) = member(members, field) {
                    self.push(
                        "TSF_SCHEMA",
                        format!("material.{field} is only supported for pbr"),
                        json_pointer_join(path, field),
                        entry.value.span,
                    );
                }
            }
        }
        self.reject_unknown(
            members,
            &["model", "base_color", "metallic", "roughness"],
            path,
            "material",
        );
        for field in ["metallic", "roughness"] {
            if let Some(entry) = member(members, field)
                && let Some(number) = number_value(&entry.value)
                && !(0.0..=1.0).contains(&number)
            {
                self.push(
                    "TSF_SCHEMA",
                    format!("material.{field} must be between 0 and 1"),
                    json_pointer_join(path, field),
                    entry.value.span,
                );
            }
        }
    }

    fn validate_scalar(&mut self, members: &[Member], field: &str, path: &str, label: &str) {
        let Some(entry) = member(members, field) else {
            self.push(
                "TSF_MISSING_KEY",
                format!("{label} missing '{field}'"),
                path,
                Span::default(),
            );
            return;
        };
        match &entry.value.kind {
            ValueKind::Number(number) if fits_f32_without_underflow(number.value) => {}
            ValueKind::Number(_) => self.push(
                "TSF_INVALID_NUMBER",
                format!("{label}.{field} must be finite and fit in f32"),
                json_pointer_join(path, field),
                entry.value.span,
            ),
            _ => self.push(
                "TSF_SCHEMA",
                format!("{label}.{field} must be a number"),
                json_pointer_join(path, field),
                entry.value.span,
            ),
        }
    }

    fn validate_positive_scalar(
        &mut self,
        members: &[Member],
        field: &str,
        path: &str,
        label: &str,
    ) {
        if let Some(entry) = member(members, field)
            && let Some(value) = number_value(&entry.value)
            && value <= 0.0
        {
            self.push(
                "TSF_SCHEMA",
                format!("{label}.{field} must be positive"),
                json_pointer_join(path, field),
                entry.value.span,
            );
        }
    }

    fn validate_nonnegative_scalar(
        &mut self,
        members: &[Member],
        field: &str,
        path: &str,
        label: &str,
    ) {
        if let Some(entry) = member(members, field)
            && let Some(value) = number_value(&entry.value)
            && value < 0.0
        {
            self.push(
                "TSF_SCHEMA",
                format!("{label}.{field} must be non-negative"),
                json_pointer_join(path, field),
                entry.value.span,
            );
        }
    }

    fn validate_range_array_member(
        &mut self,
        members: &[Member],
        field: &str,
        path: &str,
        label: &str,
    ) {
        if let Some(entry) = member(members, field)
            && let ValueKind::Array(values) = &entry.value.kind
        {
            for (index, value) in values.iter().enumerate() {
                if let Some(number) = number_value(value)
                    && !(0.0..=1.0).contains(&number)
                {
                    self.push(
                        "TSF_SCHEMA",
                        format!("{label}[{index}] must be between 0 and 1"),
                        format!("{path}/{field}/{index}"),
                        value.span,
                    );
                }
            }
        }
    }

    fn validate_optional_viewport(&mut self, members: &[Member], path: &str) {
        if let Some(viewport) = member(members, "viewport") {
            let Some(fields) = object_members(&viewport.value) else {
                self.push(
                    "TSF_SCHEMA",
                    "camera.viewport must be an object",
                    json_pointer_join(path, "viewport"),
                    viewport.value.span,
                );
                return;
            };
            for field in ["width", "height"] {
                let field_path = json_pointer_join(&json_pointer_join(path, "viewport"), field);
                let Some(entry) = member(fields, field) else {
                    self.push(
                        "TSF_MISSING_KEY",
                        format!("camera.viewport missing '{field}'"),
                        json_pointer_join(path, "viewport"),
                        viewport.value.span,
                    );
                    continue;
                };
                if !matches!(&entry.value.kind, ValueKind::Number(number)
                    if number.value.is_finite() && number.value.fract() == 0.0
                        && (1.0..=u32::MAX as f64).contains(&number.value))
                {
                    self.push(
                        "TSF_SCHEMA",
                        "camera viewport dimension must be a positive u32 integer",
                        field_path,
                        entry.value.span,
                    );
                }
            }
            self.reject_unknown(
                fields,
                &["width", "height"],
                &json_pointer_join(path, "viewport"),
                "camera.viewport",
            );
        }
    }

    fn reject_unknown(
        &mut self,
        members: &[Member],
        allowed: &[&str],
        path: &str,
        component: &str,
    ) {
        for entry in members {
            if !allowed.contains(&entry.key.as_str()) {
                self.push(
                    "TSF_SCHEMA",
                    format!("{component} field '{}' is not supported", entry.key),
                    json_pointer_join(path, &entry.key),
                    entry.key_span,
                );
            }
        }
    }

    fn validate_number_array_member(
        &mut self,
        members: &[Member],
        field: &str,
        path: &str,
        len: usize,
        label: &str,
    ) {
        match member(members, field) {
            Some(entry) => self.validate_number_array(
                &entry.value,
                &json_pointer_join(path, field),
                len,
                label,
            ),
            None => self.push(
                "TSF_MISSING_KEY",
                format!("{label} is required"),
                path,
                Span::default(),
            ),
        }
    }

    fn validate_integer_array(&mut self, value: &Value, path: &str) {
        let ValueKind::Array(values) = &value.kind else {
            self.push(
                "TSF_SCHEMA",
                "mesh.submeshes must be an array",
                path,
                value.span,
            );
            return;
        };
        for (index, value) in values.iter().enumerate() {
            if !matches!(&value.kind, ValueKind::Number(number) if number.value.is_finite() && number.value.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&number.value))
            {
                self.push(
                    "TSF_SCHEMA",
                    "mesh.submeshes must contain non-negative integers",
                    format!("{path}/{index}"),
                    value.span,
                );
            }
        }
    }

    fn validate_quaternion(&mut self, value: &Value, path: &str) {
        let ValueKind::Array(values) = &value.kind else {
            self.push(
                "TSF_SCHEMA",
                "transform.rotation must be an array",
                path,
                value.span,
            );
            return;
        };
        if values.len() != 4 {
            self.push(
                "TSF_SCHEMA",
                "transform.rotation must contain 4 numbers",
                path,
                value.span,
            );
            return;
        }
        let mut rotation = [0.0_f32; 4];
        for (index, value) in values.iter().enumerate() {
            match &value.kind {
                ValueKind::Number(number) if !number.value.is_finite() => {}
                ValueKind::Number(number) if fits_f32_without_underflow(number.value) => {
                    rotation[index] = number.value as f32;
                }
                ValueKind::Number(_) => self.push(
                    "TSF_INVALID_NUMBER",
                    format!("transform.rotation[{index}] must fit in f32 without underflow"),
                    format!("{path}/{index}"),
                    value.span,
                ),
                _ => self.push(
                    "TSF_SCHEMA",
                    format!("transform.rotation[{index}] must be a number"),
                    format!("{path}/{index}"),
                    value.span,
                ),
            }
        }
        if values.iter().all(|value| matches!(&value.kind, ValueKind::Number(number) if number.value.is_finite() && fits_f32_without_underflow(number.value)))
            && normalized_quaternion(rotation).is_none()
        {
            self.push(
                "TSF_INVALID_QUATERNION",
                "transform.rotation quaternion must have unit norm within 1e-5",
                path,
                value.span,
            );
        }
    }

    fn validate_payload_vectors(
        &mut self,
        value: &Value,
        path: &str,
        fields: &[(&str, usize)],
        optional_fields: &[&str],
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
        for member in members {
            if !fields.iter().any(|(field, _)| *field == member.key)
                && !optional_fields.contains(&member.key.as_str())
            {
                self.push(
                    "TSF_UNKNOWN_COMPONENT_FIELD",
                    format!("{component} field '{}' is not supported", member.key),
                    json_pointer_join(path, &member.key),
                    member.key_span,
                );
            }
        }
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
            match &value.kind {
                ValueKind::Number(number) if !fits_f32_without_underflow(number.value) => {
                    self.push(
                        "TSF_INVALID_NUMBER",
                        format!("{label}[{index}] must fit in f32 without underflow"),
                        format!("{path}/{index}"),
                        value.span,
                    );
                }
                ValueKind::Number(_) => {}
                _ => {
                    self.push(
                        "TSF_SCHEMA",
                        format!("{label}[{index}] must be a number"),
                        format!("{path}/{index}"),
                        value.span,
                    );
                }
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

fn run_component_validator(
    diagnostics: &mut Diagnostics,
    validate: impl FnOnce(&mut Validator<'_>),
) {
    let mut validator = Validator {
        file: None,
        errors: Vec::new(),
        asset_aliases: HashSet::new(),
        entity_ids: HashSet::new(),
        registry: None,
    };
    validate(&mut validator);
    diagnostics.extend(validator.errors);
}

pub(crate) fn validate_transform_binding(value: &Value, path: &str, diagnostics: &mut Diagnostics) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_transform(value, path)
    });
}

pub(crate) fn validate_velocity_binding(value: &Value, path: &str, diagnostics: &mut Diagnostics) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_velocity(value, path)
    });
}

pub(crate) fn validate_camera_binding(value: &Value, path: &str, diagnostics: &mut Diagnostics) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_camera(value, path)
    });
}

pub(crate) fn validate_directional_light_binding(
    value: &Value,
    path: &str,
    diagnostics: &mut Diagnostics,
) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_directional_light(value, path)
    });
}

pub(crate) fn validate_mesh_binding(value: &Value, path: &str, diagnostics: &mut Diagnostics) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_mesh(value, path)
    });
}

pub(crate) fn validate_material_binding(value: &Value, path: &str, diagnostics: &mut Diagnostics) {
    run_component_validator(diagnostics, |validator| {
        validator.validate_material(value, path)
    });
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

fn number_value(value: &Value) -> Option<f64> {
    match &value.kind {
        ValueKind::Number(number) => Some(number.value),
        _ => None,
    }
}

fn number_member(members: &[Member], key: &str) -> Option<f64> {
    member(members, key).and_then(|entry| number_value(&entry.value))
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
            .all(|component| matches!(component, Component::Normal(_) | Component::ParentDir))
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
