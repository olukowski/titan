use std::{fs, path::PathBuf};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;

const RED_CUBE: &str = include_str!("../../titan_render/tests/fixtures/red_cube.tsf");

fn titan() -> Command {
    Command::cargo_bin("titan").expect("titan binary should build")
}

fn scene(dir: &TempDir, source: &str) -> PathBuf {
    let path = dir.path().join("scene.tsf");
    fs::write(&path, source).expect("write scene");
    path
}

fn render(dir: &TempDir, source: &str, camera: &str) -> std::process::Output {
    let path = scene(dir, source);
    titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", camera, "--out"])
        .arg(dir.path().join("frame.png"))
        .output()
        .expect("run titan render")
}

fn successful_render(dir: &TempDir, source: &str, camera: &str) -> Option<Value> {
    let output = render(dir, source, camera);
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return None;
        }
        panic!("render failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Some(serde_json::from_slice(&output.stdout).expect("structured render output"))
}

#[test]
fn render_writes_png_and_reports_red_cube_stats() {
    let dir = TempDir::new().expect("tempdir");
    let result = successful_render(&dir, RED_CUBE, "main");
    let Some(result) = result else { return };
    assert!(dir.path().join("frame.png").is_file());
    assert_eq!(result["ok"], true);
    assert_eq!(result["draw_calls"], 1);
    assert_eq!(result["triangles"], 12);
    assert_eq!(result["visible_meshes"], 1);
    assert_eq!(result["material_models"]["unlit"], 1);
}

#[test]
fn render_selects_camera_by_name_and_serialized_id() {
    let dir = TempDir::new().expect("tempdir");
    let by_name = successful_render(&dir, RED_CUBE, "main");
    let Some(by_name) = by_name else { return };
    let by_id = successful_render(&dir, RED_CUBE, "entity:main_camera").expect("adapter state");
    assert_eq!(by_name["camera"], "entity:main_camera");
    assert_eq!(by_id["camera"], "entity:main_camera");
}

#[test]
fn render_unknown_camera_is_structured_error() {
    let dir = TempDir::new().expect("tempdir");
    let output = render(&dir, RED_CUBE, "missing");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "RENDER_CAMERA_UNAVAILABLE");
}

#[test]
fn render_ambiguous_camera_is_structured_error() {
    let source = RED_CUBE.replace(
        "    {\n      id: \"entity:red_cube\"",
        "    {\n      id: \"entity:second_camera\",\n      name: \"main\",\n      components: {\n        transform: { translation: [0.0, 0.0, 3.0] },\n        camera: { projection: \"perspective\", vertical_fov_degrees: 60.0, near: 0.1, far: 100.0 },\n      },\n    },\n    {\n      id: \"entity:red_cube\"",
    );
    let dir = TempDir::new().expect("tempdir");
    let output = render(&dir, &source, "main");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "RENDER_CAMERA_UNAVAILABLE");
}

#[test]
fn render_invalid_scene_preserves_tsf_diagnostics() {
    let dir = TempDir::new().expect("tempdir");
    let output = render(&dir, &RED_CUBE.replace("camera:", "camerx:"), "main");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_TSF_ERROR");
    assert_eq!(
        error["error"]["diagnostics"][0]["code"],
        "TSF_UNKNOWN_COMPONENT"
    );
    assert!(
        error["error"]["diagnostics"][0]["path"]
            .as_str()
            .expect("diagnostic path")
            .contains("camera")
    );
}
