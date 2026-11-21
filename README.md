# Titan

Titan (`titan-engine/titan`) is an open-source game engine written in Rust, built to be fast,
small, and easy to work on for both people and AI agents.

> Titan is in early development. There are no stability or backwards-compatibility guarantees:
> APIs, formats, and behavior change whenever a better design appears. Pin an exact version or
> commit if you depend on it.

## Values

- **Performance.** Fast at runtime, lean in memory, and quick to compile.
- **Compact output.** Small binaries across all first-class targets (Linux, macOS, Windows, and
  WASM), and small `target/` directories so parallel git worktrees do not fill the disk.
- **No backwards compatibility.** Breaking changes migrate every caller; nothing keeps
  deprecated shims around.
- **Readable code.** Plain, obvious code over clever code.
- **AI-friendly.** Engines and games are built from code and plaintext formats, not GUI-only
  configuration, so agents can read, write, and diff everything.
- **Fast CI.** Every check stays quick enough to run on every push.

## Requirements

- Rust `1.100.0`, edition 2024, pinned in [`rust-toolchain.toml`](rust-toolchain.toml).
  `rustup` installs it, plus the `wasm32-unknown-unknown` target, on first use.

## Local checks

These are the same checks CI runs:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --workspace --target wasm32-unknown-unknown
```

## Contributing

All changes land through pull requests from a fork to `main`. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first; AI agents should also read [`AGENTS.md`](AGENTS.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option (`MIT OR Apache-2.0`).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
Titan by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
