# Titan Roadmap

Titan's phases are ordered so that the **agent feedback loop closes as early as possible**: an agent should be able to create, run, observe, and verify a game long before the engine is feature-rich. Human-facing polish comes after — and is built on — the agent-facing layer.

## Phase 0 — Foundation & workflow (now)

Goal: a repo where parallel agents can contribute safely.

- [ ] Cargo workspace skeleton: `titan_core`, `titan_math`, `titan_scene`, `titan_cli` (binary: `titan`)
- [ ] CI: fmt, clippy (`-D warnings`), tests on every PR
- [ ] PR template requiring what/why/verification
- [ ] `docs/design/` directory + design-doc template (design docs precede implementation for significant decisions)
- [ ] `titan --version` / `titan --json` plumbing: global structured-output and structured-error conventions established from the first command

## Phase 1 — The agent loop (minimum closed loop)

Goal: an agent can author a scene as text, run it headless, and observe the result as data. **No graphics yet.**

- [ ] **Titan Scene Format (TSF)**: text-based, schema-validated scene/entity/component format with deterministic serialization (stable ordering & formatting → clean diffs). Design doc first; evaluate RON/KDL/custom.
- [ ] ECS core in `titan_core` (decide: build minimal vs. adopt e.g. `hecs` — design doc)
- [ ] Deterministic fixed-timestep runtime with seeded RNG
- [ ] `titan run --headless --frames N --dump-state out.json` — run a scene and dump full world state
- [ ] `titan scene` subcommands: `validate`, `query`, `edit` (path-addressed get/set of components), `fmt`
- [ ] Structured event log output (entity spawned, component changed, system errors) as JSONL

Exit criterion: an agent, using only the CLI, creates a scene with a moving entity, simulates 100 frames, and asserts the entity's final position from the state dump.

## Phase 2 — Seeing (headless rendering)

Goal: agents (and humans) can look at what they built.

- [ ] wgpu-based renderer in `titan_render`; forward pipeline, cameras, directional light, unlit + basic PBR materials
- [ ] **Headless render-to-image**: `titan render scene.tsf --camera main --out frame.png` — this is the agent's eyes and the most important deliverable of the phase
- [ ] `titan run --capture-every N` for frame sequences
- [ ] Windowed viewer (`titan view`) as a thin shell over the headless renderer
- [ ] Render stats as structured output (draw calls, triangle counts) for verification without pixels

Exit criterion: an agent adds a red cube to a scene, renders headlessly, and can verify the cube is visible (via the image + render stats).

## Phase 3 — Making things (procedural assets)

Goal: geometry that agents can author, diff, and reason about. This is the hardest open design area.

- [ ] **TitanGeo v1**: CSG primitives (box, sphere, cylinder, …) + boolean ops, serialized as text in the scene format
- [ ] Design doc for TitanGeo v2 direction: procedural node graph (SDFs, extrude/revolve/loft, modifiers, materials-per-node) — CSG alone is too limited, this is the growth path
- [ ] `titan geo preview asset.tgeo --out preview.png` (multi-angle turntable option) — closes the see-what-you-made loop for assets
- [ ] Mesh statistics / bounding info as structured output
- [ ] glTF import as the escape hatch for traditional art pipelines
- [ ] Hot reload of scenes and assets in `titan view`

Exit criterion: an agent authors a recognizable compound object (e.g. a table, a simple house) purely in text, previews it, and iterates on it.

## Phase 4 — Playing (interactivity)

Goal: actual games.

- [ ] Input handling (with a *replayable* input recording format — agents test gameplay by replaying inputs deterministically)
- [ ] Scripting/behavior layer (design doc: Rust-first with hot reload vs. embedded language vs. WASM)
- [ ] 2D/3D physics integration (likely `rapier`), deterministic mode required
- [ ] Audio (data-driven trigger model)
- [ ] `titan test` — scenario-based gameplay tests: load scene, inject recorded input, assert on world state

Exit criterion: a small complete game (e.g. Breakout or a 3D collect-the-items demo) built by an agent end-to-end, with gameplay assertions in CI.

## Phase 5 — The human layer

Goal: prove the agent API is complete by building the human GUI on top of it.

- [ ] GUI editor (scene tree, inspector, viewport) that performs **every mutation through the same command API agents use** — no privileged backdoors
- [ ] Command palette that exposes CLI commands 1:1
- [ ] Undo/redo derived from the command log (which agents also get for free)
- [ ] Collaboration story: human edits in GUI and agent edits via CLI interleave cleanly (text format + command log make this tractable)

## Ongoing tracks (all phases)

- **Agent tooling**: MCP server exposing engine commands/queries; keep CLI, API, and MCP surface in lockstep
- **Docs**: every crate and command documented; examples runnable headlessly in CI
- **Benchmarks & determinism tests**: same seed + same inputs ⇒ bit-identical state dumps, enforced in CI
