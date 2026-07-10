use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use assert_cmd::Command;
use serde_json::Value;
use titan_core::DEFAULT_FIXED_DT;

const MOVING_ENTITY: &str = "tests/fixtures/moving_entity.tsf";
const RED_CUBE: &str = "../titan_render/tests/fixtures/red_cube.tsf";

#[test]
fn phase1_exit_criterion_runs_moving_entity_for_100_frames() {
    let dir = temp_dir("phase1_exit");
    let dump = dir.join("state.json");

    titan()
        .args([
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "100",
            "--dump-state",
        ])
        .arg(&dump)
        .assert()
        .success();

    let state = read_json(&dump);
    let translation =
        &state["entities"][0]["components"]["titan.core.Transform"]["value"]["translation"];
    let mut expected_x = 0.0_f32;
    for _ in 0..100 {
        expected_x += 0.1_f32 * DEFAULT_FIXED_DT;
    }

    assert_eq!(state["frame"], 100);
    assert_eq!(state["entity_ids"]["entity:mover"], 1);
    assert_eq!(translation["x"].as_f64().unwrap(), f64::from(expected_x));
    assert_eq!(translation["y"], 0.0);
    assert_eq!(translation["z"], 0.0);
}

#[test]
fn same_scene_seed_and_frames_produce_byte_identical_dump_files() {
    let dir = temp_dir("determinism_same_seed");
    let first = dir.join("first.json");
    let second = dir.join("second.json");

    run_to_dump(&first, "1234");
    run_to_dump(&second, "1234");

    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}

#[test]
// The CLI currently records scene-loading events only; runtime RNG behavior is
// covered by the core test-local seeded system.
fn same_scene_seed_and_frames_produce_byte_identical_scene_loading_event_logs() {
    let dir = temp_dir("determinism_same_seed_events");
    let first = dir.join("first.jsonl");
    let second = dir.join("second.jsonl");

    run_to_event_log(&first, "1234");
    run_to_event_log(&second, "1234");

    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}

#[test]
fn different_seed_is_recorded_in_dump_metadata() {
    let dir = temp_dir("determinism_different_seed");
    let first = dir.join("first.json");
    let second = dir.join("second.json");

    run_to_dump(&first, "1");
    run_to_dump(&second, "2");

    let first = read_json(&first);
    let second = read_json(&second);

    assert_eq!(first["seed"], 1);
    assert_eq!(second["seed"], 2);
}

#[test]
fn capture_sequence_has_exact_frames_names_sizes_and_json_shape() {
    let dir = temp_dir("capture_sequence");
    let captures = dir.join("nested/captures");
    let output = titan()
        .args(["--json", "run", RED_CUBE, "--headless", "--frames", "5"])
        .args(["--seed", "42", "--capture-initial", "--capture-every", "2"])
        .args(["--capture-dir"])
        .arg(&captures)
        .args(["--camera", "main", "--width", "31", "--height", "19"])
        .output()
        .expect("run captures");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!(
            "capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for frame in [0, 2, 4] {
        assert!(captures.join(format!("frame-{frame:06}.png")).is_file());
    }
    assert!(!captures.join("frame-000001.png").exists());
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let file = fs::read_to_string(captures.join("captures.jsonl")).expect("capture stats");
    assert_eq!(stdout, file);
    let records: Vec<Value> = file
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records
            .iter()
            .map(|r| r["frame"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [0, 2, 4]
    );
    assert!(
        records
            .iter()
            .all(|r| r["seed"] == 42 && r["width"] == 31 && r["height"] == 19)
    );
    let mut keys: Vec<_> = records[0]
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
            "seed",
            "shader_version",
            "triangles",
            "visible_meshes",
            "width"
        ]
    );
}

#[test]
fn capture_without_initial_starts_after_requested_update_interval() {
    let dir = temp_dir("capture_no_initial");
    let output = titan()
        .args([
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "2",
            "--capture-every",
            "2",
            "--capture-dir",
        ])
        .arg(&dir)
        .output()
        .expect("run captures");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!(
            "capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(!dir.join("frame-000000.png").exists());
    assert!(dir.join("frame-000002.png").is_file());
    assert!(
        output.stdout.is_empty(),
        "non-JSON capture run should not emit records"
    );
}

#[test]
fn capture_occurs_after_fixed_step_updates_visible_world_state() {
    let dir = temp_dir("capture_post_update_pixels");
    let source = fs::read_to_string(RED_CUBE)
        .expect("red cube fixture")
        .replace(
            "transform: { translation: [0.0, 0.0, 0.0] },\n        mesh:",
            "transform: { translation: [0.0, 0.0, 0.0] },\n        velocity: { linear: [100.0, 0.0, 0.0] },\n        mesh:",
        );
    assert!(source.contains("velocity: { linear: [100.0"));
    let scene = dir.join("moving-cube.tsf");
    fs::write(&scene, source).expect("moving cube scene");
    let captures = dir.join("captures");
    let output = titan()
        .args(["--json", "run"])
        .arg(scene)
        .args([
            "--headless",
            "--frames",
            "1",
            "--dt",
            "1",
            "--capture-initial",
            "--capture-every",
            "1",
            "--capture-dir",
        ])
        .arg(&captures)
        .args(["--camera", "main", "--width", "64", "--height", "64"])
        .output()
        .expect("run moving cube captures");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!(
            "capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let initial = fs::read(captures.join("frame-000000.png")).expect("initial PNG");
    let updated = fs::read(captures.join("frame-000001.png")).expect("updated PNG");
    assert_ne!(
        initial, updated,
        "post-update capture must reflect moved cube"
    );
}

#[test]
fn capture_rejects_zero_interval_before_rendering() {
    let output = titan()
        .args([
            "--json",
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "1",
            "--capture-every",
            "0",
        ])
        .output()
        .expect("run capture");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_CLI_ARGUMENT_ERROR");
}

#[test]
fn capture_bad_camera_fails_with_render_error_before_adapter_lookup() {
    let dir = temp_dir("capture_bad_camera");
    fs::write(dir.join("captures.jsonl"), "prior manifest\n").unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(&dir)
        .args(["--camera", "missing"])
        .output()
        .expect("run capture");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "RENDER_CAMERA_UNAVAILABLE");
    assert_eq!(
        fs::read_to_string(dir.join("captures.jsonl")).unwrap(),
        "prior manifest\n"
    );
}

#[test]
fn empty_capture_schedules_write_empty_manifest_without_gpu_or_state_stdout() {
    for (name, frames, every) in [("zero", "0", "1"), ("beyond", "5", "10")] {
        let dir = temp_dir(&format!("empty_capture_{name}"));
        fs::write(dir.join("captures.jsonl"), "stale\n").unwrap();
        fs::write(dir.join("frame-000001.png"), "stale png").unwrap();
        let output = titan()
            .env("WGPU_ADAPTER_NAME", "titan-test-no-such-adapter")
            .args([
                "--json",
                "run",
                RED_CUBE,
                "--headless",
                "--frames",
                frames,
                "--capture-every",
                every,
                "--capture-dir",
            ])
            .arg(&dir)
            .output()
            .expect("run empty capture schedule");
        assert!(
            output.status.success(),
            "empty schedule unexpectedly needed GPU: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(dir.join("captures.jsonl")).unwrap(), b"");
        assert_eq!(
            fs::read_to_string(dir.join("frame-000001.png")).unwrap(),
            "stale png"
        );
    }
}

#[test]
fn empty_capture_schedule_still_validates_camera_and_keeps_manifest() {
    let dir = temp_dir("empty_capture_bad_camera");
    fs::write(dir.join("captures.jsonl"), "stale\n").unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "5",
            "--capture-every",
            "10",
            "--camera",
            "no-such-camera",
            "--capture-dir",
        ])
        .arg(&dir)
        .output()
        .expect("run empty capture schedule with bad camera");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "RENDER_CAMERA_UNAVAILABLE");
    assert_eq!(
        fs::read_to_string(dir.join("captures.jsonl")).unwrap(),
        "stale\n"
    );
}

#[test]
fn capture_only_options_require_a_capture_trigger() {
    for args in [
        vec!["--camera", "main"],
        vec!["--width", "32", "--height", "24"],
    ] {
        let mut command = titan();
        command.args([
            "--json",
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "0",
        ]);
        let output = command.args(args).output().expect("run scene");
        assert!(!output.status.success());
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        assert_eq!(error["error"]["code"], "TITAN_CLI_ARGUMENT_ERROR");
    }
}

#[test]
fn run_without_capture_remains_compatible_with_implicit_json_state_dump() {
    let output = titan()
        .args([
            "--json",
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "1",
        ])
        .output()
        .expect("run scene");
    assert!(output.status.success());
    let state: Value = serde_json::from_slice(&output.stdout).expect("single state JSON");
    assert_eq!(state["frame"], 1);
}

#[test]
fn capture_keeps_explicit_state_dump_while_stdout_is_jsonl_only() {
    let dir = temp_dir("capture_explicit_dump");
    let dump = dir.join("state.json");
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(dir.join("captures"))
        .args(["--dump-state"])
        .arg(&dump)
        .output()
        .expect("run capture");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        panic!(
            "capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(read_json(&dump)["frame"], 0);
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 1);
}

#[cfg(unix)]
#[test]
fn capture_rejects_symlink_directory() {
    use std::os::unix::fs::symlink;
    let dir = temp_dir("capture_symlink_dir");
    let real = dir.join("real");
    fs::create_dir(&real).unwrap();
    let link = dir.join("captures");
    symlink(&real, &link).unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(link)
        .output()
        .expect("run capture");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
}

#[cfg(unix)]
#[test]
fn capture_rejects_symlink_stats_file() {
    use std::os::unix::fs::symlink;
    let dir = temp_dir("capture_symlink_stats");
    let target = dir.join("target");
    fs::write(&target, "keep").unwrap();
    symlink(&target, dir.join("captures.jsonl")).unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(&dir)
        .output()
        .expect("run capture");
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
    assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
    assert_eq!(fs::read_to_string(target).unwrap(), "keep");
}

#[cfg(unix)]
#[test]
fn capture_rejects_generated_png_symlink() {
    use std::os::unix::fs::symlink;
    let dir = temp_dir("capture_symlink_png");
    let target = dir.join("target.png");
    fs::write(&target, "keep").unwrap();
    fs::write(dir.join("captures.jsonl"), "prior manifest\n").unwrap();
    symlink(&target, dir.join("frame-000000.png")).unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(&dir)
        .output()
        .expect("run capture");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
    } else {
        panic!("capture unexpectedly accepted PNG symlink");
    }
    assert_eq!(fs::read_to_string(target).unwrap(), "keep");
    assert_eq!(
        fs::read_to_string(dir.join("captures.jsonl")).unwrap(),
        "prior manifest\n"
    );
}

#[test]
fn capture_rejects_directory_at_generated_png_path_without_replacing_manifest() {
    let dir = temp_dir("capture_png_directory");
    fs::create_dir(dir.join("frame-000000.png")).unwrap();
    fs::write(dir.join("captures.jsonl"), "prior manifest\n").unwrap();
    let output = titan()
        .args([
            "--json",
            "run",
            RED_CUBE,
            "--headless",
            "--frames",
            "0",
            "--capture-initial",
            "--capture-dir",
        ])
        .arg(&dir)
        .output()
        .expect("run capture");
    if !output.status.success() {
        let error: Value = serde_json::from_slice(&output.stderr).expect("structured error");
        if error["error"]["code"] == "RENDER_NO_ADAPTER" {
            return;
        }
        assert_eq!(error["error"]["code"], "TITAN_OUTPUT_PATH");
    } else {
        panic!("capture unexpectedly accepted directory as PNG output");
    }
    assert!(dir.join("frame-000000.png").is_dir());
    assert_eq!(
        fs::read_to_string(dir.join("captures.jsonl")).unwrap(),
        "prior manifest\n"
    );
}

#[test]
fn loader_reports_unknown_component_with_path_and_span() {
    let dir = temp_dir("unknown_component");
    let scene = dir.join("unknown.tsf");
    fs::write(
        &scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/unknown" },
  assets: {},
  entities: [
    {
      id: "entity:mover",
      components: {
        mystery: {},
      },
    },
  ],
}
"#,
    )
    .unwrap();

    let output = titan()
        .args(["run"])
        .arg(&scene)
        .args(["--headless", "--frames", "1"])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let error: Value = serde_json::from_slice(&output).unwrap();
    let diagnostic = &error["error"]["diagnostics"][0];

    assert_eq!(diagnostic["code"], "TSF_UNKNOWN_COMPONENT");
    assert_eq!(
        diagnostic["path"],
        "/entities/entity:mover/components/mystery"
    );
    assert!(diagnostic["span"]["start"]["line"].as_u64().unwrap() > 0);
}

#[test]
fn loader_reports_bad_component_payload_with_path_and_span() {
    let dir = temp_dir("bad_payload");
    let scene = dir.join("bad_payload.tsf");
    fs::write(
        &scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/bad_payload" },
  assets: {},
  entities: [
    {
      id: "entity:mover",
      components: {
        velocity: {
          linear: ["fast", 0.0, 0.0],
        },
      },
    },
  ],
}
"#,
    )
    .unwrap();

    let output = titan()
        .args(["run"])
        .arg(&scene)
        .args(["--headless", "--frames", "1"])
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let error: Value = serde_json::from_slice(&output).unwrap();
    let diagnostic = &error["error"]["diagnostics"][0];

    assert_eq!(diagnostic["code"], "TSF_SCHEMA");
    assert_eq!(
        diagnostic["path"],
        "/entities/entity:mover/components/velocity/linear/0"
    );
    assert!(diagnostic["span"]["start"]["line"].as_u64().unwrap() > 0);
}

#[test]
fn loader_rejects_vector_values_that_do_not_fit_f32() {
    for (name, value) in [("overflow", "1e39"), ("underflow", "1e-46")] {
        let dir = temp_dir(name);
        let scene = dir.join(format!("{name}.tsf"));
        fs::write(
            &scene,
            format!(
                r#"{{
  tsf: 1,
  scene: {{ id: "scene:tests/{name}" }},
  assets: {{}},
  entities: [
    {{
      id: "entity:mover",
      components: {{
        transform: {{
          translation: [{value}, 0.0, 0.0],
        }},
        velocity: {{
          linear: [0.0, 0.0, 0.0],
        }},
      }},
    }},
  ],
}}
"#
            ),
        )
        .unwrap();

        let output = titan()
            .args(["run"])
            .arg(&scene)
            .args(["--headless", "--frames", "1"])
            .assert()
            .failure()
            .get_output()
            .stderr
            .clone();
        let error: Value = serde_json::from_slice(&output).unwrap();
        let diagnostic = &error["error"]["diagnostics"][0];

        assert_eq!(diagnostic["code"], "TSF_INVALID_NUMBER");
        assert_eq!(
            diagnostic["path"],
            "/entities/entity:mover/components/transform/translation/0"
        );
    }
}

#[test]
fn run_rejects_non_finite_or_non_positive_dt() {
    for dt in ["NaN", "inf", "0", "-0.01"] {
        let output = titan()
            .args([
                "run",
                MOVING_ENTITY,
                "--headless",
                "--frames",
                "1",
                &format!("--dt={dt}"),
            ])
            .assert()
            .failure()
            .get_output()
            .stderr
            .clone();
        let error: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(error["error"]["code"], "TITAN_CLI_ARGUMENT_ERROR");
        assert_eq!(
            error["error"]["message"],
            "--dt must be finite and positive"
        );
    }
}

#[test]
// This is scene-loading log stability coverage, not seeded runtime event coverage.
fn scene_loading_event_log_jsonl_records_events_in_stable_order() {
    let dir = temp_dir("event_log");
    let log = dir.join("events.jsonl");

    titan()
        .args([
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "0",
            "--event-log",
        ])
        .arg(&log)
        .assert()
        .success();

    let lines: Vec<Value> = fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["sequence"], 0);
    assert_eq!(lines[0]["event"], "entity_spawned");
    assert_eq!(lines[0]["entity"], 1);
    assert_eq!(lines[1]["sequence"], 1);
    assert_eq!(lines[1]["event"], "component_inserted");
    assert_eq!(lines[1]["component"], "titan.core.Transform");
    assert_eq!(lines[2]["sequence"], 2);
    assert_eq!(lines[2]["event"], "component_inserted");
    assert_eq!(lines[2]["component"], "titan.core.Velocity");
}

#[test]
fn scene_loading_event_log_is_stable_across_scene_component_order() {
    let dir = temp_dir("event_log_component_order");
    let first_scene = dir.join("first.tsf");
    let second_scene = dir.join("second.tsf");
    let first_log = dir.join("first.jsonl");
    let second_log = dir.join("second.jsonl");

    fs::write(
        &first_scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/component_order" },
  assets: {},
  entities: [
    {
      id: "entity:mover",
      components: {
        transform: { translation: [0.0, 0.0, 0.0] },
        velocity: { linear: [0.1, 0.0, 0.0] },
      },
    },
  ],
}
"#,
    )
    .unwrap();
    fs::write(
        &second_scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/component_order" },
  assets: {},
  entities: [
    {
      id: "entity:mover",
      components: {
        velocity: { linear: [0.1, 0.0, 0.0] },
        transform: { translation: [0.0, 0.0, 0.0] },
      },
    },
  ],
}
"#,
    )
    .unwrap();

    for (scene, log) in [(&first_scene, &first_log), (&second_scene, &second_log)] {
        titan()
            .args(["run"])
            .arg(scene)
            .args(["--headless", "--frames", "0", "--event-log"])
            .arg(log)
            .assert()
            .success();
    }

    assert_eq!(
        fs::read(&first_log).unwrap(),
        fs::read(&second_log).unwrap()
    );
}

#[test]
fn scene_loading_event_log_is_stable_across_scene_entity_order() {
    let dir = temp_dir("event_log_entity_order");
    let first_scene = dir.join("first.tsf");
    let second_scene = dir.join("second.tsf");
    let first_log = dir.join("first.jsonl");
    let second_log = dir.join("second.jsonl");

    fs::write(
        &first_scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/entity_order" },
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
    .unwrap();
    fs::write(
        &second_scene,
        r#"{
  tsf: 1,
  scene: { id: "scene:tests/entity_order" },
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
    .unwrap();

    for (scene, log) in [(&first_scene, &first_log), (&second_scene, &second_log)] {
        titan()
            .args(["run"])
            .arg(scene)
            .args(["--headless", "--frames", "0", "--event-log"])
            .arg(log)
            .assert()
            .success();
    }

    assert_eq!(
        fs::read(&first_log).unwrap(),
        fs::read(&second_log).unwrap()
    );
}

fn run_to_dump(path: &Path, seed: &str) {
    titan()
        .args([
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "100",
            "--seed",
            seed,
            "--dump-state",
        ])
        .arg(path)
        .assert()
        .success();
}

fn run_to_event_log(path: &Path, seed: &str) {
    titan()
        .args([
            "run",
            MOVING_ENTITY,
            "--headless",
            "--frames",
            "100",
            "--seed",
            seed,
            "--event-log",
        ])
        .arg(path)
        .assert()
        .success();
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn titan() -> Command {
    Command::cargo_bin("titan").unwrap()
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("titan_cli_{name}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
