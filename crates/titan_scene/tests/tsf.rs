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
fn forbidden_number_values_fail() {
    for (source, message) in [
        ("{ value: Infinity }", "Infinity"),
        ("{ value: NaN }", "NaN"),
        ("{ value: 0x10 }", "hexadecimal"),
        ("{ value: +1 }", "leading plus"),
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
