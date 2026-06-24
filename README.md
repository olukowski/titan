# Titan

An **agent-native game engine** in `no_std`, zero-dependency, stable Rust.

A Titan game is a **deterministic pure function**:
`(initial state, seed, action sequence) → (frames, observations)`. It is designed
first to be driven by AI agents — with humans and graphical tools as first-class
but optional participants — and determinism is what makes that practical:
reproducible runs, record/replay, exact bug reports, and tests that assert on
outcomes rather than vibes.

## What makes it different

- **`no_std`, zero external dependencies.** Titan depends on `core` only — no
  `std`, no libc, no crates.io. Every layer is ours, legible top to bottom.
- **Stable Rust.** No nightly. Raw Linux syscalls via stable `asm!`; macOS via
  thin documented `libSystem` symbols; an explicit, swappable platform `trait`
  for everything else.
- **No hidden global state.** Allocation, the platform, time, and randomness are
  values passed explicitly into the code that needs them.
- **Deterministic, not immutable.** State is mutated in place through exclusive
  references for speed, while staying a pure function of its inputs — a property
  Rust *enforces*.
- **Two front-ends, one core.** A programmatic action/observation API for agents
  and an interactive viewer for humans, both clients of the same deterministic
  core.

## Status

Early foundation. This repository is a clean restart on a new `no_std`
foundation; the original `std` prototype — which proved the thesis (deterministic
agent loop, CPU software renderer, CSG geometry kernel) — is archived at
[`titan-engine/titan-prototype`](https://github.com/titan-engine/titan-prototype)
and is being ported here layer by layer.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full thesis, principles, and
roadmap.

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT)
at your option.
