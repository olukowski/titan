use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn version_outputs_human_readable_text() {
    let mut cmd = Command::cargo_bin("titan").expect("titan binary should build");

    cmd.arg("--version")
        .assert()
        .success()
        .stdout("titan 0.1.0\n")
        .stderr("");
}

#[test]
fn version_outputs_json_when_requested() {
    let mut cmd = Command::cargo_bin("titan").expect("titan binary should build");

    cmd.args(["--json", "--version"])
        .assert()
        .success()
        .stdout(predicate::str::is_match(r#"^\{"name":"titan","version":"0\.1\.0"\}\n$"#).unwrap())
        .stderr("");
}

#[test]
fn json_errors_are_structured() {
    let mut cmd = Command::cargo_bin("titan").expect("titan binary should build");

    cmd.args(["--json", "bogus"])
        .assert()
        .failure()
        .stdout("")
        .stderr(predicate::str::contains(
            r#""error":{"code":"TITAN_CLI_UNKNOWN_COMMAND""#,
        ))
        .stderr(predicate::str::contains("unrecognized subcommand 'bogus'"));
}
