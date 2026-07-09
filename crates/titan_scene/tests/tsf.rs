use rand::{Rng, SeedableRng, rngs::StdRng};
use titan_core::{
    Camera, CameraProjection, Component, DirectionalLight, EventKind, Material, Mesh, Transform,
    Velocity, phase1_component_registry, phase2_component_registry,
};
use titan_scene::{
    Diagnostic, DiagnosticSpan, TsfComponentBinding, TsfComponentRegistry,
    TsfComponentRegistryError, validate_with_registry,
};
use titan_scene::{
    Number, ValueKind, edit, fmt, fmt_with_registry, load_world, load_world_with_runtime_registry,
    parse, query, validate,
};

const MOVING_ENTITY: &str = include_str!("fixtures/moving_entity.tsf");

fn phase2_source(component: &str) -> String {
    format!(
        "{{ tsf: 1, scene: {{ id: \"scene:test\" }}, assets: {{ cube: {{ path: \"cube.mesh\", kind: \"geometry\" }} }}, entities: [{{ id: \"entity:test\", components: {{ {component} }} }}] }}"
    )
}

fn assert_schema(component: &str, path: &str) {
    let document = parse(Some("schema.tsf"), &phase2_source(component)).expect("parse");
    let error = validate(&document).expect_err("schema must be rejected");
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.code == "TSF_SCHEMA" && d.path == path),
        "diagnostics: {:?}",
        error.errors
    );
}

#[test]
fn registry_builders_and_binding_accessors_are_complete() {
    let phase1 = titan_scene::phase1_component_registry().unwrap();
    assert_eq!(
        phase1.registered_name("transform"),
        Some("titan.core.Transform")
    );
    assert!(phase1.binding("camera").is_none());
    assert!(
        titan_scene::phase2_component_registry()
            .unwrap()
            .binding("material")
            .is_some()
    );
}

#[test]
fn registry_rejects_duplicates_schema_mismatch_and_missing_types() {
    let core = phase2_component_registry().unwrap();
    let binding = TsfComponentBinding {
        alias: "x",
        registered_name: "titan.core.Transform",
        schema_version: 2,
        validate: |_v, _p, _d| {},
    };
    assert!(matches!(
        TsfComponentRegistry::new(core.clone(), [binding, binding]),
        Err(TsfComponentRegistryError::DuplicateAlias("x"))
    ));
    let duplicate_name = [
        binding,
        TsfComponentBinding {
            alias: "y",
            ..binding
        },
    ];
    assert!(matches!(
        TsfComponentRegistry::new(core.clone(), duplicate_name),
        Err(TsfComponentRegistryError::DuplicateRegisteredName(
            "titan.core.Transform"
        ))
    ));
    let mismatch = TsfComponentBinding {
        schema_version: 1,
        ..binding
    };
    assert!(matches!(
        TsfComponentRegistry::new(core, [mismatch]),
        Err(TsfComponentRegistryError::SchemaVersionMismatch { .. })
    ));
    assert!(
        matches!(TsfComponentRegistry::new(titan_core::ComponentRegistry::new(), [binding]),
        Err(TsfComponentRegistryError::ComponentNotRegistered(name)) if name == "titan.core.Transform")
    );
}

fn custom_validator(_: &titan_scene::Value, path: &str, diagnostics: &mut Vec<Diagnostic>) {
    diagnostics.push(Diagnostic {
        code: "CUSTOM_VALIDATOR".into(),
        message: "custom validator was called".into(),
        path: path.into(),
        span: DiagnosticSpan {
            file: None,
            start: Default::default(),
            end: Default::default(),
        },
    });
}

#[test]
fn custom_registry_drives_validation_and_missing_runtime_type_diagnostic() {
    let core = phase2_component_registry().unwrap();
    let registry = TsfComponentRegistry::new(
        core.clone(),
        [TsfComponentBinding {
            alias: "custom_transform",
            registered_name: "titan.core.Camera",
            schema_version: 1,
            validate: custom_validator,
        }],
    )
    .unwrap();
    let source = phase2_source(
        "custom_transform: { projection: \"perspective\", vertical_fov_degrees: 60, near: 0.1, far: 10 }",
    );
    let document = parse(Some("custom.tsf"), &source).unwrap();
    let error = validate_with_registry(&document, &registry).unwrap_err();
    assert_eq!(error.errors[0].code, "CUSTOM_VALIDATOR");
    let runtime_registry = TsfComponentRegistry::new(
        core,
        [TsfComponentBinding {
            alias: "custom_transform",
            registered_name: "titan.core.Camera",
            schema_version: 1,
            validate: |_value, _path, _diagnostics| {},
        }],
    )
    .unwrap();
    let error = match load_world_with_runtime_registry(
        &document,
        &runtime_registry,
        titan_core::phase1_component_registry().unwrap(),
    ) {
        Ok(_) => panic!("mapped component should be absent from runtime registry"),
        Err(error) => error,
    };
    assert_eq!(error.errors[0].code, "TSF_COMPONENT_NOT_REGISTERED");
    assert!(error.errors[0].message.contains("custom_transform"));
    assert!(error.errors[0].message.contains("titan.core.Camera"));
    assert_eq!(
        error.errors[0].path,
        "/entities/0/components/custom_transform"
    );
}

#[test]
fn custom_camera_alias_preserves_integer_json_values() {
    let registry = TsfComponentRegistry::new(
        titan_core::phase2_component_registry().unwrap(),
        [TsfComponentBinding {
            alias: "lens",
            registered_name: "titan.core.Camera",
            schema_version: 1,
            validate: |_value, _path, _diagnostics| {},
        }],
    )
    .unwrap();
    let document = parse(
        Some("custom-camera.tsf"),
        &phase2_source(
            "lens: { projection: \"perspective\", vertical_fov_degrees: 60, near: 0.1, far: 10, viewport: { width: 640, height: 480 } }",
        ),
    )
    .unwrap();

    let world = load_world(&document, registry).expect("custom camera alias should load");
    let camera = world
        .get::<Camera>(titan_core::EntityId::from_raw(1))
        .unwrap()
        .expect("camera should be inserted");
    match camera.projection {
        CameraProjection::Perspective {
            viewport: Some(viewport),
            ..
        } => {
            assert_eq!(viewport.width, 640);
            assert_eq!(viewport.height, 480);
        }
        projection => panic!("unexpected projection: {projection:?}"),
    }
}

#[test]
fn projection_material_mesh_viewport_and_submesh_schema_rules_are_enforced() {
    assert_schema(
        "camera: { projection: \"perspective\", vertical_fov_degrees: 60, near: 0.1, far: 10, height: 2 }",
        "/entities/entity:test/components/camera/height",
    );
    assert_schema(
        "camera: { projection: \"orthographic\", height: 2, near: 0.1, far: 10, vertical_fov_degrees: 60 }",
        "/entities/entity:test/components/camera/vertical_fov_degrees",
    );
    assert_schema(
        "camera: { projection: \"perspective\", vertical_fov_degrees: 60, near: 0.1, far: 10, viewport: { width: 1.5, height: 480 } }",
        "/entities/entity:test/components/camera/viewport/width",
    );
    assert_schema(
        "mesh: { geometry: { ref: 4 } }",
        "/entities/entity:test/components/mesh/geometry",
    );
    assert_schema(
        "mesh: { geometry: { ref: \"asset:cube\" }, submeshes: [4294967296] }",
        "/entities/entity:test/components/mesh/submeshes/0",
    );
    assert_schema(
        "material: { model: \"unlit\", base_color: [1, 1, 1, 1], metallic: 0.5 }",
        "/entities/entity:test/components/material/metallic",
    );
}

#[test]
fn binding_validator_diagnostic_keeps_document_filename() {
    let document = parse(
        Some("binding-diagnostic.tsf"),
        &phase2_source("transform: { translation: [1, 2] }"),
    )
    .expect("parse");
    let error = validate(&document).expect_err("invalid transform should be rejected");

    let diagnostic = error
        .errors
        .iter()
        .find(|diagnostic| {
            diagnostic.code == "TSF_SCHEMA"
                && diagnostic.path == "/entities/entity:test/components/transform/translation"
        })
        .expect("transform binding diagnostic");
    assert_eq!(
        diagnostic.span.file.as_deref(),
        Some("binding-diagnostic.tsf")
    );
}

#[test]
fn moving_entity_round_trips_and_format_is_idempotent() {
    let document = parse(Some("moving_entity.tsf"), MOVING_ENTITY).expect("parse fixture");
    validate(&document).expect("validate fixture");

    let formatted = fmt(&document);
    assert_eq!(formatted, MOVING_ENTITY);

    let reparsed = parse(Some("moving_entity.tsf"), &formatted).expect("parse formatted fixture");
    let reformatted = fmt(&reparsed);
    assert_eq!(reformatted, formatted);
}

#[test]
fn phase2_components_validate_format_and_insert_as_typed_values() {
    let source = r#"{
  tsf: 1,
  scene: { id: "scene:red_cube" },
  assets: { cube_mesh: { path: "cube.mesh", kind: "geometry" } },
  entities: [{
    id: "entity:cube",
    components: {
      material: { model: "unlit", base_color: [1.0, 0.0, 0.0, 1.0] },
      mesh: { geometry: { ref: "asset:cube_mesh" }, submeshes: [0, 3] },
      directional_light: { color: [1.0, 1.0, 1.0], illuminance: 1.0, ambient: 0.05 },
      camera: { projection: "perspective", vertical_fov_degrees: 60.0, near: 0.1, far: 100.0, viewport: { width: 640, height: 480 } },
      velocity: { linear: [0.0, 0.0, 0.0] },
      transform: { translation: [0.0, 0.0, 0.0] },
    },
  }],
}
"#;
    let document = parse(Some("phase2.tsf"), source).expect("parse");
    validate(&document).expect("validate");
    let formatted = fmt(&document);
    assert_eq!(fmt(&parse(None, &formatted).expect("reparse")), formatted);
    let world = load_world(&document, phase2_component_registry().unwrap()).expect("load");
    let entity = titan_core::EntityId::from_raw(1);
    assert!(world.get::<Camera>(entity).unwrap().is_some());
    assert!(world.get::<DirectionalLight>(entity).unwrap().is_some());
    assert!(world.get::<Mesh>(entity).unwrap().is_some());
    assert!(world.get::<Material>(entity).unwrap().is_some());
    assert!(world.get::<Transform>(entity).unwrap().is_some());
    assert!(world.get::<Velocity>(entity).unwrap().is_some());
    assert_eq!(
        world.get::<Mesh>(entity).unwrap().unwrap().submeshes,
        Some(vec![0, 3])
    );
    assert_eq!(
        world.get::<Camera>(entity).unwrap().unwrap().projection,
        CameraProjection::Perspective {
            vertical_fov_degrees: 60.0,
            near: 0.1,
            far: 100.0,
            viewport: Some(titan_core::Viewport {
                width: 640,
                height: 480
            }),
        }
    );
}

#[test]
fn formatter_uses_supplied_registry_component_order() {
    let core = titan_core::phase2_component_registry().unwrap();
    let registry = TsfComponentRegistry::new(
        core,
        [
            titan_scene::TsfComponentBinding {
                alias: "mesh",
                registered_name: "titan.core.Mesh",
                schema_version: 1,
                validate: |_value, _path, _diagnostics| {},
            },
            titan_scene::TsfComponentBinding {
                alias: "transform",
                registered_name: "titan.core.Transform",
                schema_version: 2,
                validate: |_value, _path, _diagnostics| {},
            },
        ],
    )
    .unwrap();
    let document = parse(
        None,
        "{ tsf: 1, scene: { id: \"scene:test\" }, assets: { cube: { path: \"cube.mesh\", kind: \"geometry\" } }, entities: [{ id: \"entity:test\", components: { transform: { translation: [0, 0, 0] }, mesh: { geometry: { ref: \"asset:cube\" } } } }] }",
    )
    .unwrap();
    let formatted = fmt_with_registry(&document, &registry);
    assert!(formatted.find("mesh:").unwrap() < formatted.find("transform:").unwrap());
    let world = load_world(&document, registry).expect("custom registry should load");
    let inserted: Vec<_> = world
        .event_log()
        .records()
        .iter()
        .filter_map(|record| match &record.kind {
            EventKind::ComponentInserted { component, .. } => Some(component.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(inserted, ["titan.core.Mesh", "titan.core.Transform"]);
}

#[test]
fn formatter_uses_registered_component_identity_for_custom_aliases() {
    let registry = TsfComponentRegistry::new(
        titan_core::phase2_component_registry().unwrap(),
        [
            titan_scene::TsfComponentBinding {
                alias: "lens",
                registered_name: "titan.core.Camera",
                schema_version: 1,
                validate: |_value, _path, _diagnostics| {},
            },
            titan_scene::TsfComponentBinding {
                alias: "pose",
                registered_name: "titan.core.Transform",
                schema_version: 2,
                validate: |_value, _path, _diagnostics| {},
            },
        ],
    )
    .unwrap();
    let document = parse(
        None,
        "{ tsf: 1, scene: { id: \"scene:test\" }, assets: {}, entities: [{ id: \"entity:test\", components: { lens: { viewport: { height: 480, width: 640 }, far: 10, near: 0.1, vertical_fov_degrees: 60, projection: \"perspective\" }, pose: { rotation: [0, 0, 0, 1], translation: [0, 0, 0] } } }] }",
    )
    .unwrap();

    let formatted = fmt_with_registry(&document, &registry);
    let lens = formatted.find("lens:").unwrap();
    assert!(
        formatted[lens..].find("projection:").unwrap()
            < formatted[lens..].find("viewport:").unwrap()
    );
    let pose = formatted.find("pose:").unwrap();
    assert!(formatted[pose..].find("translation:").is_some());
    assert!(!formatted.contains("rotation:"));
}

#[test]
fn load_world_assigns_scene_entity_ids_independent_of_array_order() {
    let first = parse(
        Some("ordered.tsf"),
        r#"{
  tsf: 1,
  scene: { id: "scene:test" },
  assets: {},
  entities: [
    {
      id: "entity:b",
      components: {
        transform: { translation: [2.0, 0.0, 0.0] },
      },
    },
    {
      id: "entity:a",
      components: {
        transform: { translation: [1.0, 0.0, 0.0] },
      },
    },
  ],
}
"#,
    )
    .expect("parse first scene");
    let second = parse(
        Some("reordered.tsf"),
        r#"{
  tsf: 1,
  scene: { id: "scene:test" },
  assets: {},
  entities: [
    {
      id: "entity:a",
      components: {
        transform: { translation: [1.0, 0.0, 0.0] },
      },
    },
    {
      id: "entity:b",
      components: {
        transform: { translation: [2.0, 0.0, 0.0] },
      },
    },
  ],
}
"#,
    )
    .expect("parse reordered scene");

    let first = load_world(&first, phase1_component_registry().unwrap())
        .unwrap()
        .dump_state()
        .unwrap();
    let second = load_world(&second, phase1_component_registry().unwrap())
        .unwrap()
        .dump_state()
        .unwrap();

    assert_eq!(first.entity_ids, second.entity_ids);
    assert_eq!(first.entity_ids.get("entity:a"), Some(&1));
    assert_eq!(first.entity_ids.get("entity:b"), Some(&2));
}

#[test]
fn edit_scalar_changes_one_line() {
    let document = parse(Some("moving_entity.tsf"), MOVING_ENTITY).expect("parse fixture");
    let edited = edit(
        &document,
        "/entities/entity:mover/components/velocity/linear/0",
        "0.2",
    )
    .expect("edit scalar");

    let old_lines: Vec<_> = MOVING_ENTITY.lines().collect();
    let new_lines: Vec<_> = edited.lines().collect();
    let changed: Vec<_> = old_lines
        .iter()
        .zip(new_lines.iter())
        .filter(|(old, new)| old != new)
        .collect();

    assert_eq!(old_lines.len(), new_lines.len());
    assert_eq!(changed.len(), 1);
    assert_eq!(*changed[0].1, "          linear: [0.2, 0.0, 0.0],");
}

#[test]
fn edit_reports_semantic_replacement_errors_at_replacement() {
    let document = parse(Some("moving_entity.tsf"), MOVING_ENTITY).expect("parse fixture");
    let error = edit(
        &document,
        "/entities/0/components/velocity/linear/0",
        "\"bad\"",
    )
    .expect_err("invalid replacement should fail validation");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_SCHEMA"
            && diagnostic.path == "/entities/entity:mover/components/velocity/linear/0"
            && diagnostic.span.file.as_deref() == Some("<replacement>")
    }));
}

#[test]
fn duplicate_key_fails_before_validation() {
    let error = parse(None, "{ tsf: 1, tsf: 2 }").expect_err("duplicate key should fail");
    assert_eq!(error.errors[0].code, "TSF_DUPLICATE_KEY");
    assert_eq!(error.errors[0].path, "");
}

#[test]
fn direct_duplicate_keys_fail_validation() {
    let mut document = parse(Some("duplicate-public.tsf"), MOVING_ENTITY).expect("parse");
    let ValueKind::Object(root) = &mut document.root.kind else {
        unreachable!("fixture root is an object");
    };
    let duplicate = root
        .iter()
        .find(|member| member.key == "scene")
        .expect("scene")
        .clone();
    root.push(duplicate);

    let error = validate(&document).expect_err("duplicate public keys should fail");
    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_DUPLICATE_KEY" && diagnostic.path.is_empty()
    }));
}

#[test]
fn broken_asset_reference_is_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: {},\n  broken: { ref: \"asset:missing\" },",
    );
    let document = parse(Some("broken.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("broken reference should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/broken/ref"
    }));
}

#[test]
fn malformed_reference_objects_are_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: {},\n  bad: { ref: 123, note: \"extra\" },",
    );
    let document = parse(Some("bad-ref.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("malformed reference should fail");

    assert!(
        error
            .errors
            .iter()
            .any(|diagnostic| { diagnostic.code == "TSF_SCHEMA" && diagnostic.path == "/bad/ref" })
    );
    assert!(
        error
            .errors
            .iter()
            .any(|diagnostic| { diagnostic.code == "TSF_SCHEMA" && diagnostic.path == "/bad" })
    );
}

#[test]
fn malformed_external_references_are_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: {},\n  bad_file: { ref: \"file:\" },\n  bad_scene: { ref: \"scene:#entity:\" },",
    );
    let document = parse(Some("bad-external-ref.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("malformed external references should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/bad_file/ref"
    }));
    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/bad_scene/ref"
    }));
}

#[test]
fn non_relative_external_references_are_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: {},\n  absolute_file: { ref: \"file:/tmp/mesh.tgeo\" },\n  current_scene: { ref: \"scene:./scene.tsf#entity:mover\" },",
    );
    let document = parse(Some("bad-external-ref.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("non-relative external references should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/absolute_file/ref"
    }));
    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/current_scene/ref"
    }));
}

#[test]
fn parent_relative_paths_are_valid() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: { mesh: { path: \"../assets/cube.tgeo\", kind: \"geometry\" } },\n  external_mesh: { ref: \"file:../assets/cube.tgeo\" },\n  external_scene: { ref: \"scene:../scenes/other.tsf#entity:mover\" },",
    );
    let document = parse(Some("parent-relative.tsf"), &source).expect("parse");

    validate(&document).expect("parent-relative paths should validate");
}

#[test]
fn asset_alias_named_ref_is_not_a_reference_object() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: { ref: { path: \"meshes/cube.tgeo\", kind: \"geometry\" } },",
    );
    let document = parse(Some("asset-ref-alias.tsf"), &source).expect("parse");

    validate(&document).expect("asset alias named ref should validate");
}

#[test]
fn absolute_asset_paths_are_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "assets: {},",
        "assets: { mesh: { path: \"/tmp/mesh.tgeo\", kind: \"geometry\" } },",
    );
    let document = parse(Some("absolute-asset-path.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("absolute asset path should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_SCHEMA" && diagnostic.path == "/assets/mesh/path"
    }));
}

#[test]
fn invalid_entity_parent_is_diagnostic() {
    let source = MOVING_ENTITY.replace(
        "      id: \"entity:mover\",\n",
        "      id: \"entity:mover\",\n      parent: \"entity:missing\",\n",
    );
    let document = parse(Some("bad-parent.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("missing parent should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_BROKEN_REF" && diagnostic.path == "/entities/entity:mover/parent"
    }));

    let source = MOVING_ENTITY.replace(
        "      id: \"entity:mover\",\n",
        "      id: \"entity:mover\",\n      parent: 7,\n",
    );
    let document = parse(Some("bad-parent.tsf"), &source).expect("parse");
    let error = validate(&document).expect_err("non-string parent should fail");

    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_SCHEMA" && diagnostic.path == "/entities/entity:mover/parent"
    }));
}

#[test]
fn unsupported_runtime_components_are_diagnostic() {
    for component in ["light", "unknown"] {
        let source = MOVING_ENTITY.replace(
            "        velocity: {",
            &format!("        {component}: {{}},\n        velocity: {{"),
        );
        let document = parse(Some("unsupported-component.tsf"), &source).expect("parse");
        let error = validate(&document).expect_err("unsupported component should fail");

        assert!(
            error.errors.iter().any(|diagnostic| {
                diagnostic.code == "TSF_UNKNOWN_COMPONENT"
                    && diagnostic.path == format!("/entities/entity:mover/components/{component}")
            }),
            "missing diagnostic for {component}: {:?}",
            error.errors
        );
    }
}

#[test]
fn unsupported_runtime_component_fields_are_diagnostic() {
    for (component, field, payload) in [
        ("transform", "scale", "[1.0, 1.0, 1.0]"),
        ("velocity", "angular", "[0.0, 0.0, 0.0]"),
    ] {
        let source = MOVING_ENTITY.replace(
            &format!("        {component}: {{"),
            &format!("        {component}: {{\n          {field}: {payload},"),
        );
        let document = parse(Some("unsupported-field.tsf"), &source).expect("parse");
        let error = validate(&document).expect_err("unsupported field should fail");

        assert!(
            error.errors.iter().any(|diagnostic| {
                diagnostic.code == "TSF_UNKNOWN_COMPONENT_FIELD"
                    && diagnostic.path
                        == format!("/entities/entity:mover/components/{component}/{field}")
            }),
            "missing diagnostic for {component}.{field}: {:?}",
            error.errors
        );
    }
}

fn transform_scene(rotation: &str) -> String {
    format!(
        r#"{{
  tsf: 1,
  scene: {{ id: "scene:test" }},
  assets: {{}},
  entities: [{{
    id: "entity:test",
        components: {{ transform: {{ translation: [1.0, 2.0, 3.0]{} }} }},
  }}],
}}
"#,
        if rotation.is_empty() {
            String::new()
        } else {
            format!(", rotation: {rotation}")
        }
    )
}

#[test]
fn transform_v1_shape_loads_with_identity_and_dump_is_v2() {
    let document = parse(None, &transform_scene("")).expect("parse");
    validate(&document).expect("validate");
    let world = load_world(&document, phase1_component_registry().unwrap()).expect("load");
    let transform = world
        .get::<Transform>(titan_core::EntityId::from_raw(1))
        .unwrap()
        .unwrap();
    assert_eq!(transform.rotation, [0.0, 0.0, 0.0, 1.0]);
    let dump = world.dump_state().unwrap();
    assert_eq!(
        dump.entities[0].components[Transform::NAME].schema_version,
        2
    );
    assert_eq!(
        dump.entities[0].components[Transform::NAME].value["rotation"],
        serde_json::json!([0.0, 0.0, 0.0, 1.0])
    );
}

#[test]
fn quaternion_validation_accepts_tolerance_and_rejects_just_beyond_and_zero_length() {
    for rotation in ["[0.0, 0.0, 0.0, 1.000009]", "[0.0, 0.0, 0.0, 1.00002]"] {
        let document = parse(None, &transform_scene(rotation)).expect("parse");
        let result = validate(&document);
        if rotation.ends_with("1.000009]") {
            assert!(result.is_ok(), "boundary should be accepted: {rotation}");
        } else {
            let error = result.expect_err("out-of-tolerance quaternion should fail");
            assert!(
                error
                    .errors
                    .iter()
                    .any(|d| d.code == "TSF_INVALID_QUATERNION" && d.path.ends_with("/rotation"))
            );
        }
    }
    {
        let rotation = "[0.0, 0.0, 0.0, 0.0]";
        let document = parse(None, &transform_scene(rotation)).expect("parse");
        let error = validate(&document).expect_err("invalid quaternion should fail");
        assert!(
            error
                .errors
                .iter()
                .any(|d| d.code == "TSF_INVALID_QUATERNION" && d.path.contains("/rotation"))
        );
    }
}

#[test]
fn non_finite_transform_rotation_numbers_are_diagnostic() {
    let mut document = parse(None, &transform_scene("[0.0, 0.0, 0.0, 1.0]")).expect("parse");
    let ValueKind::Object(root) = &mut document.root.kind else {
        unreachable!("scene root is an object");
    };
    let entities = root
        .iter_mut()
        .find(|member| member.key == "entities")
        .expect("entities");
    let ValueKind::Array(entities) = &mut entities.value.kind else {
        unreachable!("entities is an array");
    };
    let ValueKind::Object(entity) = &mut entities[0].kind else {
        unreachable!("entity is an object");
    };
    let components = entity
        .iter_mut()
        .find(|member| member.key == "components")
        .expect("components");
    let ValueKind::Object(components) = &mut components.value.kind else {
        unreachable!("components is an object");
    };
    let transform = components
        .iter_mut()
        .find(|member| member.key == "transform")
        .expect("transform");
    let ValueKind::Object(transform) = &mut transform.value.kind else {
        unreachable!("transform is an object");
    };
    let rotation = transform
        .iter_mut()
        .find(|member| member.key == "rotation")
        .expect("rotation");
    let ValueKind::Array(rotation) = &mut rotation.value.kind else {
        unreachable!("rotation is an array");
    };
    for value in rotation {
        let ValueKind::Number(number) = &mut value.kind else {
            unreachable!("rotation component is a number");
        };
        number.value = f64::NAN;
    }

    let error = validate(&document).expect_err("non-finite rotation should fail");
    for index in 0..4 {
        let path = format!("/entities/0/components/transform/rotation/{index}");
        let diagnostics: Vec<_> = error.errors.iter().filter(|d| d.path == path).collect();
        assert_eq!(
            diagnostics.len(),
            1,
            "unexpected diagnostics at {path}: {diagnostics:?}"
        );
        assert_eq!(diagnostics[0].code, "TSF_INVALID_NUMBER");
        assert_eq!(diagnostics[0].message, "number must be finite");
    }
}

#[test]
fn formatter_orders_and_canonicalizes_transform_rotation() {
    let document = parse(None, &transform_scene("[0.0, 0.0, 0.0, 1.000005]")).expect("parse");
    validate(&document).expect("validate");
    let formatted = fmt(&document);
    assert!(formatted.contains("translation: [1.0, 2.0, 3.0],\n"));
    assert!(
        !formatted.contains("rotation:"),
        "identity rotation is a default and is omitted"
    );

    let document = parse(None, &transform_scene("[0.0, 0.0, 0.707111, 0.707111]")).expect("parse");
    validate(&document).expect("validate");
    let formatted = fmt(&document);
    assert!(formatted.find("translation:").unwrap() < formatted.find("rotation:").unwrap());
    assert!(formatted.contains("rotation: [0.0, 0.0, 0.707111, 0.707111]"));
}

#[test]
fn formatter_is_idempotent_for_near_unit_rotations() {
    for rotation in [
        "[0.0, 0.0, 0.707111, 0.707111]",
        "[0.000001, -0.000002, 0.707111, 0.707111]",
        "[0.0, 0.70710678, 0.70710678, 0.0]",
        "[0.0, 0.999995, 0.0, 0.0]",
    ] {
        let document = parse(None, &transform_scene(rotation)).expect("parse");
        if let Err(error) = validate(&document) {
            panic!("validate {rotation}: {error:?}");
        }
        let first = fmt(&document);
        let reparsed = parse(None, &first).expect("parse formatted scene");
        let second = fmt(&reparsed);
        assert_eq!(second, first, "formatter was not idempotent for {rotation}");
    }
}

#[test]
fn formatter_is_idempotent_for_seeded_random_near_unit_rotations() {
    let mut rng = StdRng::seed_from_u64(0x5eed_cafe);

    for case in 0..2_000 {
        let mut components = [
            rng.random_range(-1.0_f64..=1.0),
            rng.random_range(-1.0_f64..=1.0),
            rng.random_range(-1.0_f64..=1.0),
            rng.random_range(-1.0_f64..=1.0),
        ];
        let norm = components
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            components[3] = 1.0;
        } else {
            for component in &mut components {
                *component /= norm;
            }
        }
        let rotation = format!(
            "[{:.8}, {:.8}, {:.8}, {:.8}]",
            components[0], components[1], components[2], components[3]
        );
        let document = parse(None, &transform_scene(&rotation)).expect("parse generated scene");
        validate(&document).unwrap_or_else(|error| panic!("validate case {case}: {error:?}"));
        let first = fmt(&document);
        let reparsed = parse(None, &first).expect("parse formatted generated scene");
        validate(&reparsed)
            .unwrap_or_else(|error| panic!("validate formatted case {case}: {error:?}"));
        let second = fmt(&reparsed);
        assert_eq!(
            second, first,
            "formatter was not idempotent for case {case}: {rotation}"
        );
    }
}

#[test]
fn explicit_identity_rotation_loads_and_dump_round_trips() {
    let document = parse(None, &transform_scene("[0.0, 0.0, 0.0, 1.0]")).expect("parse");
    let world = load_world(&document, phase1_component_registry().unwrap()).expect("load");
    let value = &world.dump_state().unwrap().entities[0].components[Transform::NAME].value;
    assert_eq!(value["rotation"], serde_json::json!([0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn direct_non_finite_numbers_are_diagnostic() {
    let mut document = parse(Some("nonfinite.tsf"), MOVING_ENTITY).expect("parse");
    let ValueKind::Object(root) = &mut document.root.kind else {
        unreachable!("fixture root is an object");
    };
    let entities = root
        .iter_mut()
        .find(|member| member.key == "entities")
        .expect("entities");
    let ValueKind::Array(entities) = &mut entities.value.kind else {
        unreachable!("entities is an array");
    };
    let ValueKind::Object(entity) = &mut entities[0].kind else {
        unreachable!("entity is an object");
    };
    let components = entity
        .iter_mut()
        .find(|member| member.key == "components")
        .expect("components");
    let ValueKind::Object(components) = &mut components.value.kind else {
        unreachable!("components is an object");
    };
    let velocity = components
        .iter_mut()
        .find(|member| member.key == "velocity")
        .expect("velocity");
    let ValueKind::Object(velocity) = &mut velocity.value.kind else {
        unreachable!("velocity is an object");
    };
    let linear = velocity
        .iter_mut()
        .find(|member| member.key == "linear")
        .expect("linear");
    let ValueKind::Array(linear) = &mut linear.value.kind else {
        unreachable!("linear is an array");
    };
    let ValueKind::Number(number) = &mut linear[0].kind else {
        unreachable!("linear[0] is a number");
    };
    *number = Number {
        value: f64::NAN,
        had_fraction: true,
    };

    let error = validate(&document).expect_err("non-finite number should fail");
    assert!(error.errors.iter().any(|diagnostic| {
        diagnostic.code == "TSF_INVALID_NUMBER"
            && diagnostic.path == "/entities/0/components/velocity/linear/0"
    }));
}

#[test]
fn formatter_preserves_nonzero_exponent_numbers() {
    let document = parse(None, "{ value: 1e-100 }").expect("parse exponent");
    let formatted = fmt(&document);

    assert!(formatted.contains("value: 1e-100"));
}

#[test]
fn json_conversion_uses_mathematical_integrality() {
    let document = parse(
        None,
        "{ decimal: 1e-1, exponent_integer: 1e2, trailing_fraction: 100.0 }",
    )
    .expect("parse numbers");

    assert_eq!(
        query(&document, "/decimal").unwrap().value,
        serde_json::json!(0.1)
    );
    assert_eq!(
        query(&document, "/exponent_integer").unwrap().value,
        serde_json::json!(100)
    );
    assert_eq!(
        query(&document, "/trailing_fraction").unwrap().value,
        serde_json::json!(100)
    );
}

#[test]
fn exponent_and_trailing_fraction_spellings_load_typed_camera_and_mesh_values() {
    let document = parse(
        None,
        &phase2_source(
            "camera: { projection: \"perspective\", vertical_fov_degrees: 6e1, near: 1e-1, far: 1e2, viewport: { width: 6.4e2, height: 4.8e2 } }, mesh: { geometry: { ref: \"asset:cube\" }, submeshes: [0e0, 3.0] }",
        ),
    )
    .unwrap();
    let world = load_world(&document, phase2_component_registry().unwrap()).expect("load");
    let entity = titan_core::EntityId::from_raw(1);
    assert_eq!(
        world.get::<Camera>(entity).unwrap().unwrap().projection,
        CameraProjection::Perspective {
            vertical_fov_degrees: 60.0,
            near: 0.1,
            far: 100.0,
            viewport: Some(titan_core::Viewport {
                width: 640,
                height: 480,
            }),
        }
    );
    assert_eq!(
        world.get::<Mesh>(entity).unwrap().unwrap().submeshes,
        Some(vec![0, 3])
    );
}

#[test]
fn forbidden_number_values_fail() {
    for (source, message) in [
        ("{ value: Infinity }", "Infinity"),
        ("{ value: NaN }", "NaN"),
        ("{ value: 0x10 }", "hexadecimal"),
        ("{ value: +1 }", "leading plus"),
        ("{ value: 1e-324 }", "underflows"),
    ] {
        let error = parse(None, source).expect_err("forbidden number should fail");
        assert_eq!(error.errors[0].code, "TSF_INVALID_NUMBER", "{source}");
        assert!(
            error.errors[0].message.contains(message),
            "unexpected message for {source}: {}",
            error.errors[0].message
        );
    }
}

#[test]
fn string_escapes_follow_json5() {
    let document = parse(None, r#"{ value: "\x41\v\0", escaped: "\z" }"#).expect_err("bad escape");
    assert_eq!(document.errors[0].code, "TSF_PARSE_ERROR");

    let document = parse(None, r#"{ value: "\x41\v\0" }"#).expect("parse JSON5 escapes");
    assert_eq!(
        query(&document, "/value").expect("query value").value,
        "A\u{b}\0"
    );

    let document = parse(None, r#"{ value: "\uD83D\uDE00" }"#).expect("parse surrogate pair");
    assert_eq!(
        query(&document, "/value").expect("query value").value,
        "\u{1f600}"
    );

    let error = parse(None, r#"{ value: "\uD83D" }"#).expect_err("reject lone surrogate");
    assert_eq!(error.errors[0].code, "TSF_PARSE_ERROR");
}

#[test]
fn overly_nested_input_fails_before_stack_overflow() {
    let mut source = "[".repeat(130);
    source.push_str("null");
    source.push_str(&"]".repeat(130));

    let error = parse(None, &source).expect_err("deep nesting should fail");
    assert_eq!(error.errors[0].code, "TSF_PARSE_ERROR");
    assert!(error.errors[0].message.contains("nesting depth"));
}

#[test]
fn formatter_quotes_hyphenated_keys() {
    let document = parse(None, "{ asset-name: { nested_key: true } }").expect("parse");
    let formatted = fmt(&document);

    assert!(formatted.contains("\"asset-name\": {"));
    assert!(formatted.contains("nested_key: true"));
}

#[test]
fn crate_root_reexports_tsf_model_types() {
    let _kind = titan_scene::ValueKind::Number(titan_scene::Number {
        value: 1.0,
        had_fraction: false,
    });
    let _members: Vec<titan_scene::Member> = Vec::new();
}

#[test]
fn query_and_edit_resolve_entity_id_path() {
    let document = parse(Some("moving_entity.tsf"), MOVING_ENTITY).expect("parse fixture");

    let result = query(
        &document,
        "/entities/entity:mover/components/velocity/linear/0",
    )
    .expect("query by entity id");

    assert_eq!(result.value, serde_json::json!(0.1));
    assert_eq!(
        result.resolved_pointer,
        "/entities/0/components/velocity/linear/0"
    );

    let edited = edit(
        &document,
        "/entities/entity:mover/components/velocity/linear/0",
        "0.2",
    )
    .expect("edit by entity id");
    assert!(edited.contains("linear: [0.2, 0.0, 0.0],"));
}
