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
