# Titan

**An agent-native game engine in Rust.**

Titan is built on one idea: AI agents should be able to help humans build games *effectively* — no screenshot-clicking, no GUI automation, no hoops. Everything the engine can do is exposed as text formats, CLI commands, and structured APIs. The human-friendly interfaces (viewers, editors) are clients of that same layer.

## What that means in practice

- **Scenes are text.** Diffable, schema-validated, deterministically formatted. Humans hand-edit them; agents machine-edit them; git merges them.
- **Headless first.** `titan run --headless --frames 100 --dump-state out.json` — simulate deterministically and inspect the result as data. `titan render scene.tsf --out frame.png` — the agent's eyes.
- **Assets as code.** Geometry starts as CSG and grows into a procedural node graph, all serialized as text with headless previews.
- **Every command speaks JSON.** Structured output, structured errors, everywhere.
- **The editor is a client.** The GUI performs every mutation through the same command API agents use — building it proves the API is complete.

## Status

Pre-alpha; design and foundation phase. See [docs/ROADMAP.md](docs/ROADMAP.md) for the phase plan and [CLAUDE.md](CLAUDE.md) for design tenets and contribution conventions.

## Contributing

Work happens in the fork `olukowski/titan`; PRs target upstream `titan-engine/titan` and are reviewed automatically by AI agents. Keep PRs small and single-purpose; significant designs get a doc in `docs/design/` first.
