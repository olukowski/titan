# AGENTS.md

Engineering guide for AI agents working on Titan. Read it before editing code.

## Principles

- **Performance and compile time are features.** Measure before and after performance work.
  Avoid needless generics, macros, and dependencies that slow builds.
- **Small binaries, lean `target/`.** Every dependency, monomorphization, and debuginfo byte
  costs WASM size and disk space in every worktree. Justify each new dependency in the PR.
- **No backwards-compatibility shims.** When you change an API, format, or behavior, migrate
  every caller and delete the old path in the same change. No deprecated aliases.
- **Readable over clever.** Minimize what a reader must hold in their head: obvious control
  flow, few layers, no one-caller wrappers. Comment only the non-obvious *why*.
- **Code and plaintext over GUI.** Engine and game data live in code or diffable text formats
  that agents can read and write.

## Rust conventions

- Rust `1.100.0`, edition 2024, pinned in `rust-toolchain.toml`. Use stable features only.
- The never type `!` and `core::alloc::Allocator` are stable in 1.100; use them where they
  express intent (for example `Result<T, !>`, or allocator-aware containers).
- Prefer `#![no_std]` (with `alloc` where needed) for core crates when it is natural. It keeps
  them portable to `wasm32-unknown-unknown` and small.
- Crates live in `crates/*` and inherit metadata and lints from the workspace:
  `version.workspace = true`, …, and `[lints] workspace = true`.
- Every `unsafe` block needs a `// SAFETY:` comment; `unsafe_op_in_unsafe_fn` is denied.
- Workspace lints are high-signal. Fix warnings instead of adding `#[allow]`; if an `allow` is
  truly needed, scope it narrowly and say why.

## Verify before you finish

Run exactly what CI runs, and make sure every command passes:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo check --workspace --target wasm32-unknown-unknown --locked
```

`Cargo.lock` is tracked. Commit it whenever dependencies change.

## Parallel worktrees and disk space

Agents often build many git worktrees at once, each with its own `target/`. The workspace
profiles in `Cargo.toml` keep those small:

- `[profile.dev] debug = "line-tables-only"`: backtraces keep `file:line` while artifacts and
  link times shrink.
- `[profile.dev.package."*"] debug = false`: no debuginfo for dependencies, which is most of a
  typical `target/`.
- `[profile.wasm]`: size-optimized release build (`opt-level = "z"`, fat LTO, `panic = "abort"`):
  `cargo build --profile wasm --target wasm32-unknown-unknown`.

Also:

- Keep the dependency count low. Every crate is compiled once per worktree.
- Optionally share compilation across worktrees with
  [`sccache`](https://github.com/mozilla/sccache) (`RUSTC_WRAPPER=sccache`). Do not point
  several worktrees at one shared `CARGO_TARGET_DIR`; concurrent builds block on its lock.
- Delete `target/` in worktrees you are done with (`cargo clean`).

## Workflow

- All changes go through a pull request from a fork to `titan-engine/titan` `main`. Nobody
  pushes to `main` directly; the only merge method is squash.
- Keep each PR focused on one change with a minimal, readable diff, and list the commands you
  ran under "Verification" in the PR description.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for issue types, labels, and licensing.
