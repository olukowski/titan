use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use assert_cmd::Command;
use serde_json::Value;
use titan_core::DEFAULT_FIXED_DT;

const MOVING_ENTITY: &str = "tests/fixtures/moving_entity.tsf";

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
fn different_seed_keeps_velocity_result_but_records_seed() {
    let dir = temp_dir("determinism_different_seed");
    let first = dir.join("first.json");
    let second = dir.join("second.json");

    run_to_dump(&first, "1");
    run_to_dump(&second, "2");

    let first = read_json(&first);
    let second = read_json(&second);

    assert_eq!(first["seed"], 1);
    assert_eq!(second["seed"], 2);
    assert_eq!(
        first["entities"][0]["components"]["titan.core.Transform"]["value"],
        second["entities"][0]["components"]["titan.core.Transform"]["value"]
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
          angular: [0.0, 0.0, 0.0],
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
fn event_log_jsonl_records_loading_events_in_stable_order() {
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
