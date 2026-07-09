# Titan Engine

Titan is a game engine written in Rust with one defining constraint: **it must be equally usable by AI agents and human programmers**. The agent-friendly pipeline is the primary path; human-friendly interfaces (GUI editors, visualizers) are built *on top of* that pipeline, never beside it.

## The prime directive

> If a human can do it in an editor, an agent can do it through a CLI command or API call — with structured input and structured output.

Any feature that only works through a GUI is a bug in the architecture. Agents must never need computer use, screenshot-clicking, or other workarounds to tweak a setting, edit a scene, or inspect state.

## Design tenets

1. **Text-first, diffable everything.** Scenes, materials, entity definitions, project config — all stored in deterministic, schema-validated, human-readable text formats. Stable key ordering, stable formatting, so diffs are minimal and merges are sane.
2. **Every tool speaks JSON.** Every CLI command supports `--json` for output. Errors are structured (error code, message, span/location) — never just a panic string.
3. **Headless is the default.** Rendering, simulation, and asset processing all run headless. The windowed app is a thin shell over the headless core.
4. **Closed feedback loops for agents.** An agent must be able to: make a change → run the game N frames deterministically → capture a screenshot / event log / scene dump → verify the change. Determinism (seeded RNG, fixed timestep) is a hard requirement of the core loop.
5. **Assets as code (parametric-first).** Geometry is defined procedurally — CSG as the starting point, growing into a serialized node-graph (SDFs, mesh operators, modifiers). Traditional mesh import (glTF) is supported but is the escape hatch, not the main path.
6. **Introspection is a feature.** The engine can dump its full state (scene graph, component values, asset dependency graph, render stats) as structured data on demand.
7. **The GUI editor is a client.** The human editor consumes the exact same command/query API that agents use. Building the editor is how we prove the agent API is complete.

## Repository conventions

- Rust workspace; crates live under `crates/` and are prefixed `titan_` (e.g. `titan_core`, `titan_render`, `titan_cli`). The user-facing binary is `titan`.
- `docs/ROADMAP.md` holds the phase plan. Significant design decisions get a short design doc in `docs/design/` **before** implementation (agents should write one and get it reviewed in a PR first).
- No binary blobs in source paths. Generated/imported artifacts go to `.titan/cache/` (gitignored).
- Standard checks before any PR: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --workspace`.

## Workflow

- Upstream is `titan-engine/titan`. Work happens in the fork `olukowski/titan`; changes land via PRs to upstream, which are reviewed automatically by AI agents.
- One branch per unit of work, named `<area>/<short-description>` (e.g. `render/headless-capture`).
- Keep PRs small and single-purpose — they are reviewed by agents, and small diffs get better reviews.
- PR descriptions must state: what changed, why, and how it was verified (test names, commands run).

## Hard problems to keep in mind

- **Agent-friendly 3D assets**: CSG is the seed idea but is expressively limited. Likely direction: a text-serialized procedural geometry graph (CSG + SDF + mesh operators) with headless render previews so agents can *see* what they built. Treat this as an open design area — propose, prototype, measure.
- **Scene format ergonomics**: must be pleasant for humans to hand-edit *and* trivially machine-editable (schema, stable ordering, references between files).
- **Determinism vs. performance**: the deterministic fixed-timestep mode is non-negotiable for agent feedback loops; a non-deterministic fast path may exist alongside it, never instead of it.
