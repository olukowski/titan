use titan_scene::{edit, fmt, parse, query, validate};

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
fn duplicate_key_fails_before_validation() {
    let error = parse(None, "{ tsf: 1, tsf: 2 }").expect_err("duplicate key should fail");
    assert_eq!(error.errors[0].code, "TSF_DUPLICATE_KEY");
    assert_eq!(error.errors[0].path, "");
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
