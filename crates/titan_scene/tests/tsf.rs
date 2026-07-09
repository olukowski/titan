use titan_core::phase1_component_registry;
use titan_scene::{Number, ValueKind, edit, fmt, load_world, parse, query, validate};

const MOVING_ENTITY: &str = include_str!("fixtures/moving_entity.tsf");

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
    for component in ["mesh", "camera", "light"] {
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
        ("transform", "rotation", "[0.0, 0.0, 0.0, 1.0]"),
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
