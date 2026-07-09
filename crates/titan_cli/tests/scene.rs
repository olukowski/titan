use std::{fs, path::Path};

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

const MOVING_ENTITY: &str = include_str!("../../titan_scene/tests/fixtures/moving_entity.tsf");

fn write_scene(dir: &TempDir, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, source).expect("write scene fixture");
    path
}

fn titan() -> Command {
    Command::cargo_bin("titan").expect("titan binary should build")
}

fn stdout_json(assert: assert_cmd::assert::Assert) -> Value {
    let output = assert.get_output();
    serde_json::from_slice(&output.stdout).expect("stdout should be JSON")
}

fn stderr_json(assert: assert_cmd::assert::Assert) -> Value {
    let output = assert.get_output();
    serde_json::from_slice(&output.stderr).expect("stderr should be JSON")
}

#[test]
fn validate_accepts_valid_scene() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_scene(&dir, "moving.tsf", MOVING_ENTITY);
    let path = path_string(&path);

    titan()
        .args(["scene", "validate", path.as_str()])
        .assert()
        .success()
        .stdout(predicate::str::contains("is valid"))
        .stderr("");

    let json = stdout_json(
        titan()
            .args(["--json", "scene", "validate", path.as_str()])
            .assert()
            .success()
            .stderr(""),
    );
    assert_eq!(json["ok"], true);
    assert_eq!(json["path"], path);
}

#[test]
fn validate_reports_structured_diagnostics_for_invalid_scene() {
    let dir = TempDir::new().expect("tempdir");
    let invalid = MOVING_ENTITY.replace("velocity:", "velocty:");
    let path = write_scene(&dir, "invalid.tsf", &invalid);
    let path = path_string(&path);

    titan()
        .args(["scene", "validate", path.as_str()])
        .assert()
        .failure()
        .code(1)
        .stdout("")
        .stderr(predicate::str::contains("TSF_UNKNOWN_COMPONENT"))
        .stderr(predicate::str::contains(":18:"));

    let json = stderr_json(
        titan()
            .args(["--json", "scene", "validate", path.as_str()])
            .assert()
            .failure()
            .code(1)
            .stdout(""),
    );
    assert_eq!(json["ok"], false);
    assert_eq!(json["errors"][0]["code"], "TSF_UNKNOWN_COMPONENT");
    assert_eq!(
        json["errors"][0]["path"],
        "/entities/entity:mover/components/velocty"
    );
    assert_eq!(json["errors"][0]["span"]["start"]["line"], 18);
}

#[test]
fn query_by_entity_id_returns_value_span_and_resolved_pointer() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_scene(&dir, "moving.tsf", MOVING_ENTITY);
    let path = path_string(&path);

    let json = stdout_json(
        titan()
            .args([
                "--json",
                "scene",
                "query",
                path.as_str(),
                "/entities/entity:mover/components/velocity",
            ])
            .assert()
            .success()
            .stderr(""),
    );

    assert_eq!(json["ok"], true);
    assert_eq!(json["path"], "/entities/entity:mover/components/velocity");
    assert_eq!(json["resolved_pointer"], "/entities/0/components/velocity");
    assert_eq!(json["value"]["linear"][0], 0.1);
    assert_eq!(json["span"]["start"]["line"], 18);
}

#[test]
fn edit_updates_one_line_and_leaves_file_canonical() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_scene(&dir, "moving.tsf", MOVING_ENTITY);
    let path_str = path_string(&path);

    let json = stdout_json(
        titan()
            .args([
                "--json",
                "scene",
                "edit",
                path_str.as_str(),
                "/entities/entity:mover/components/velocity/linear/0",
                "0.2",
            ])
            .assert()
            .success()
            .stderr(""),
    );

    assert_eq!(json["ok"], true);
    assert_eq!(json["changed"], true);
    assert_eq!(json["changed_lines"], 1);
    assert_eq!(json["diff"][0]["new"], "          linear: [0.2, 0.0, 0.0],");

    let source = fs::read_to_string(&path).expect("read edited scene");
    assert!(source.contains("linear: [0.2, 0.0, 0.0],"));

    titan()
        .args(["scene", "fmt", path_str.as_str(), "--check"])
        .assert()
        .success();
}

#[test]
fn edit_can_repair_parseable_invalid_scene() {
    let dir = TempDir::new().expect("tempdir");
    let invalid = MOVING_ENTITY.replace("linear: [0.1, 0.0, 0.0]", "linear: [0.1, 'bad', 0.0]");
    let path = write_scene(&dir, "repairable.tsf", &invalid);
    let path_str = path_string(&path);

    titan()
        .args(["scene", "validate", path_str.as_str()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("TSF_SCHEMA"));

    titan()
        .args([
            "scene",
            "edit",
            path_str.as_str(),
            "/entities/entity:mover/components/velocity/linear/1",
            "0.0",
        ])
        .assert()
        .success()
        .stderr("");

    titan()
        .args(["scene", "validate", path_str.as_str()])
        .assert()
        .success();
}

#[test]
fn edit_accepts_negative_replacement_value() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_scene(&dir, "negative.tsf", MOVING_ENTITY);
    let path_str = path_string(&path);

    titan()
        .args([
            "scene",
            "edit",
            path_str.as_str(),
            "/entities/entity:mover/components/velocity/linear/0",
            "-0.2",
        ])
        .assert()
        .success()
        .stderr("");

    let source = fs::read_to_string(&path).expect("read edited scene");
    assert!(source.contains("linear: [-0.2, 0.0, 0.0],"));
}

#[cfg(unix)]
#[test]
fn fmt_preserves_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().expect("tempdir");
    let path = write_scene(
        &dir,
        "permissions.tsf",
        "{ entities: [], assets: {}, scene: { name: 'Demo', id: 'scene:demo' }, tsf: 1 }\n",
    );
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).expect("set fixture mode");
    let path_str = path_string(&path);

    titan()
        .args(["scene", "fmt", path_str.as_str()])
        .assert()
        .success();

    let mode = fs::metadata(&path)
        .expect("read formatted scene metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o640);
}

#[cfg(unix)]
#[test]
fn fmt_rewrites_symlink_target_without_replacing_link() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let target = write_scene(
        &dir,
        "target.tsf",
        "{ entities: [], assets: {}, scene: { name: 'Demo', id: 'scene:demo' }, tsf: 1 }\n",
    );
    let link = dir.path().join("linked.tsf");
    symlink(&target, &link).expect("create symlink fixture");
    let link_str = path_string(&link);
    let target_str = path_string(&target);

    titan()
        .args(["scene", "fmt", link_str.as_str()])
        .assert()
        .success();

    assert!(
        fs::symlink_metadata(&link)
            .expect("read link metadata")
            .file_type()
            .is_symlink()
    );
    titan()
        .args(["scene", "fmt", target_str.as_str(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_check_reports_canonical_and_noncanonical_files() {
    let dir = TempDir::new().expect("tempdir");
    let canonical = write_scene(&dir, "canonical.tsf", MOVING_ENTITY);
    let noncanonical = write_scene(
        &dir,
        "noncanonical.tsf",
        "{ entities: [], assets: {}, scene: { name: 'Demo', id: 'scene:demo' }, tsf: 1 }\n",
    );
    let canonical = path_string(&canonical);
    let noncanonical = path_string(&noncanonical);

    let json = stdout_json(
        titan()
            .args(["--json", "scene", "fmt", canonical.as_str(), "--check"])
            .assert()
            .success()
            .stderr(""),
    );
    assert_eq!(json["ok"], true);
    assert_eq!(json["canonical"], true);
    assert_eq!(json["written"], false);

    let json = stderr_json(
        titan()
            .args(["--json", "scene", "fmt", noncanonical.as_str(), "--check"])
            .assert()
            .failure()
            .code(1)
            .stdout(""),
    );
    assert_eq!(json["ok"], false);
    assert_eq!(json["errors"][0]["code"], "TSF_NOT_CANONICAL");

    titan()
        .args(["scene", "fmt", noncanonical.as_str()])
        .assert()
        .success();
    titan()
        .args(["scene", "fmt", noncanonical.as_str(), "--check"])
        .assert()
        .success();
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}
