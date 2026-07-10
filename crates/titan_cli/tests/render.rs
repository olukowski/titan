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

fn render_with_stats(dir: &TempDir, source: &str, camera: &str) -> std::process::Output {
    let path = scene(dir, source);
    titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", camera, "--out"])
        .arg(dir.path().join("frame.png"))
        .args(["--stats-json"])
        .arg(dir.path().join("stats.json"))
        .output()
        .expect("run titan render")
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
    let mut keys: Vec<_> = result
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "active_directional_lights",
            "adapter",
            "backend",
            "camera",
            "draw_calls",
            "frame",
            "height",
            "material_models",
            "ok",
            "output",
            "shader_version",
            "triangles",
            "visible_meshes",
            "width",
        ]
    );

    let decoder = png::Decoder::new(fs::File::open(dir.path().join("frame.png")).expect("PNG"));
    let mut reader = decoder.read_info().expect("PNG header");
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).expect("PNG frame");
    assert_eq!(info.width, result["width"].as_u64().expect("width") as u32);
    assert_eq!(
        info.height,
        result["height"].as_u64().expect("height") as u32
    );
}

#[test]
fn render_applies_paired_output_dimensions() {
    let dir = TempDir::new().expect("tempdir");
    let path = scene(&dir, RED_CUBE);
    let output = titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", "main", "--out"])
        .arg(dir.path().join("sized.png"))
        .args(["--width", "37", "--height", "23"])
        .output()
        .expect("run titan render");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!("render failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let result: Value = serde_json::from_slice(&output.stdout).expect("render output");
    assert_eq!(result["width"], 37);
    assert_eq!(result["height"], 23);
}

#[test]
fn render_rejects_unpaired_output_dimensions() {
    let dir = TempDir::new().expect("tempdir");
    let path = scene(&dir, RED_CUBE);
    titan()
        .args(["render"])
        .arg(path)
        .args(["--out"])
        .arg(dir.path().join("frame.png"))
        .args(["--width", "37"])
        .assert()
        .failure();
}

#[test]
fn render_stats_file_does_not_change_json_stdout() {
    let dir = TempDir::new().expect("tempdir");
    let output = render_with_stats(&dir, RED_CUBE, "main");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!("render failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let stdout: Value = serde_json::from_slice(&output.stdout).expect("structured render output");
    let stats: Value =
        serde_json::from_slice(&fs::read(dir.path().join("stats.json")).expect("stats file"))
            .expect("structured stats output");
    assert_eq!(stdout, stats);
    assert!(dir.path().join("frame.png").is_file());
}

#[test]
fn render_rejects_colliding_png_and_stats_paths() {
    let dir = TempDir::new().expect("tempdir");
    let output_path = dir.path().join("same.output");
    let path = scene(&dir, RED_CUBE);
    let output = titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", "main", "--out"])
        .arg(&output_path)
        .args(["--stats-json"])
        .arg(&output_path)
        .output()
        .expect("run titan render");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_OUTPUT_COLLISION");
    assert!(!output_path.exists());
}

#[cfg(unix)]
#[test]
fn render_rejects_stats_symlink_to_png_output() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let output_path = dir.path().join("frame.png");
    let stats_path = dir.path().join("stats.json");
    let original_png = b"existing png contents";
    fs::write(&output_path, original_png).expect("write PNG fixture");
    symlink(&output_path, &stats_path).expect("create stats symlink");
    let path = scene(&dir, RED_CUBE);

    let output = titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", "main", "--out"])
        .arg(&output_path)
        .args(["--stats-json"])
        .arg(&stats_path)
        .output()
        .expect("run titan render");

    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_OUTPUT_COLLISION");
    assert_eq!(
        fs::read(&output_path).expect("read PNG fixture"),
        original_png
    );
    assert!(
        fs::symlink_metadata(&stats_path)
            .expect("read stats symlink metadata")
            .file_type()
            .is_symlink()
    );
}

#[cfg(unix)]
#[test]
fn render_rejects_dangling_stats_symlink() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let output_path = dir.path().join("frame.png");
    let stats_path = dir.path().join("stats.json");
    symlink(&output_path, &stats_path).expect("create dangling stats symlink");
    let path = scene(&dir, RED_CUBE);

    let output = titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", "main", "--out"])
        .arg(&output_path)
        .args(["--stats-json"])
        .arg(&stats_path)
        .output()
        .expect("run titan render");

    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
    } else {
        panic!("render unexpectedly succeeded");
    }

    if output_path.exists() {
        let decoder = png::Decoder::new(fs::File::open(&output_path).expect("PNG output"));
        let mut reader = decoder.read_info().expect("valid PNG output");
        let mut pixels = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut pixels).expect("valid PNG frame");
    }
}

#[cfg(unix)]
#[test]
fn render_rejects_dangling_png_symlink() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let output_path = dir.path().join("frame.png");
    let stats_path = dir.path().join("stats.json");
    symlink(&stats_path, &output_path).expect("create dangling PNG symlink");
    let path = scene(&dir, RED_CUBE);

    let output = titan()
        .args(["--json", "render"])
        .arg(path)
        .args(["--camera", "main", "--out"])
        .arg(&output_path)
        .args(["--stats-json"])
        .arg(&stats_path)
        .output()
        .expect("run titan render");

    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
    assert!(
        fs::symlink_metadata(&output_path)
            .expect("read PNG symlink metadata")
            .file_type()
            .is_symlink()
    );
    assert!(!stats_path.exists());
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
