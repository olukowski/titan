# ECS core for titan_core

Status: draft

## Context

Phase 1 needs the ECS core for headless simulation, scene loading, state dumps,
event logs, and later GUI/editor clients. The decision is whether `titan_core`
should build a minimal ECS or adopt an existing crate.

The hard constraints come from Titan's agent-first architecture:

- Deterministic iteration order. With the same scene, seed, inputs, and frame
  count, a fixed-timestep run must produce bit-identical state dumps.
- Full introspection. Titan must dump every entity and every registered
  component as structured data, including dynamic component sets.
- Stable entity IDs. Text scene files need references that survive save/load,
  diffs, and command logs.
- Headless-first operation. The ECS cannot depend on rendering, windows,
  editor-only state, or non-deterministic parallel execution.

External versions and maintenance status were checked on 2026-07-09 with crates.io metadata, `cargo search`, docs.rs, and GitHub metadata:

- [`hecs` 0.11.0](https://crates.io/crates/hecs), updated 2026-01-10; repository pushed 2026-06-23. Docs describe a minimalist ECS with a `World` queried from ordinary code, not a framework.
- [`bevy_ecs` 0.19.0](https://crates.io/crates/bevy_ecs), updated 2026-06-19; Bevy repository pushed 2026-07-09 and released `v0.19.0`. Docs state it is standalone-usable and includes `World`, `Schedule`, events, reflection hooks, and a large feature surface.
- [`flecs_ecs` 0.2.2](https://crates.io/crates/flecs_ecs), updated 2025-11-17; repository pushed 2026-06-07. Docs advertise Flecs C/C++ bindings, relationships, prefabs, a query language, and WIP Rust reflection/JSON support.
- [`shipyard` 0.11.4](https://crates.io/crates/shipyard), updated 2026-06-24; repository pushed 2026-06-24.
- [`legion` 0.4.0](https://crates.io/crates/legion), updated 2021-02-25; repository pushed 2021-12-30.
- [`specs` 0.20.0](https://crates.io/crates/specs), updated 2023-09-25; repository pushed 2024-06-07.
- [`edict` 1.0.0-rc8](https://crates.io/crates/edict), updated 2025-06-26; repository pushed 2025-11-21.

## Goals & non-goals

Goals:

- Provide the Phase 1 ECS API for scene load, simulation, state dump, and CLI editing.
- Make stable entity IDs the primary identity exposed outside `titan_core`.
- Make query iteration deterministic by construction.
- Require all serializable/introspectable components to register structured metadata.
- Keep the implementation small enough to review and replace if later phases prove that a larger ECS is needed.

Non-goals:

- A high-performance archetype ECS in Phase 1.
- Parallel system execution in deterministic mode.
- A visual editor object model.
- Runtime scripting, prefabs, hierarchy semantics, or relationship queries.
- Long-term commitment to the internal storage layout.

## Proposed design

Build a minimal deterministic ECS inside `titan_core` for Phase 1. Do not adopt
an external ECS crate as the core dependency yet.

The ECS is optimized for correctness, dumpability, and stable command/query behavior. It can use slower ordered storage initially; later implementation PRs may swap storage behind the same public API if determinism and introspection tests keep passing.

### Public API surface

Expose the ECS through Titan-owned types. Do not expose storage internals or third-party ECS handles from `titan_core`.

```rust
pub struct World;
pub struct Schedule;
pub struct CommandBuffer;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EntityId(u64);

pub trait Component: 'static + Send + Sync + Serialize + DeserializeOwned {
    const NAME: &'static str;
    const SCHEMA_VERSION: u32;
}

pub struct ComponentRegistry;
pub struct ComponentMeta;
pub struct StateDump;
pub struct EventLog;
```

Required `World` operations:

```rust
impl World {
    pub fn new(registry: ComponentRegistry) -> Self;
    pub fn spawn(&mut self) -> EntityId;
    pub fn spawn_with_id(&mut self, id: EntityId) -> Result<EntityId>;
    pub fn despawn(&mut self, id: EntityId) -> Result<()>;

    pub fn insert<T: Component>(&mut self, id: EntityId, component: T) -> Result<()>;
    pub fn remove<T: Component>(&mut self, id: EntityId) -> Result<Option<T>>;
    pub fn get<T: Component>(&self, id: EntityId) -> Result<Option<&T>>;
    pub fn get_mut<T: Component>(&mut self, id: EntityId) -> Result<Option<&mut T>>;
    pub fn set<T: Component>(&mut self, id: EntityId, component: T) -> Result<()>;

    pub fn query<Q: Query>(&self) -> QueryIter<'_, Q>;
    pub fn query_mut<Q: QueryMut>(&mut self) -> QueryMutIter<'_, Q>;
    pub fn dump_state(&self) -> Result<StateDump>;
    pub fn event_log(&self) -> &EventLog;
}
```

Required scheduling operations:

```rust
pub type SystemFn = fn(&mut World, &mut CommandBuffer, FixedStepContext) -> Result<()>;

impl Schedule {
    pub fn new() -> Self;
    pub fn add_system(&mut self, name: &'static str, system: SystemFn);
    pub fn run_fixed_step(&mut self, world: &mut World, ctx: FixedStepContext) -> Result<()>;
}
```

The scheduler runs systems in insertion order in deterministic mode. Systems record structural changes into `CommandBuffer`; the scheduler applies one buffer after each system, in command insertion order. A later parallel scheduler may exist only as a separate non-default mode and must never be used by determinism CI.

### Entity identity

`EntityId` is the only identity serialized to TSF, JSON state dumps, event logs, and CLI output. IDs are unsigned 64-bit integers.

Rules:

- Scene loading may call `spawn_with_id` for IDs declared in TSF.
- Runtime spawning allocates the next unused monotonic ID above the current maximum scene/runtime ID.
- IDs are never reused during a `World` lifetime.
- Despawn records a tombstone in the event log but removes the entity from
  normal queries and dumps.
- Text references use the canonical form selected by the TSF design doc; until then docs and tests should treat `EntityId(42)` as the semantic reference.

If the implementation needs a generation counter internally, it stays private. External references do not include generations.

### Storage and query order

Initial storage should favor ordered data structures over raw speed:

- Entity table ordered by `EntityId`.
- Component stores keyed by registered component name and then `EntityId`.
- Queries compute the matching entity set and always iterate by ascending `EntityId`.
- Dumping iterates entities by ascending `EntityId` and components by registered component name.

This avoids depending on archetype insertion order, hash iteration order, or system-local allocation accidents. It also makes state dumps stable and easy to diff.

### Component registration and introspection

Every component that can appear in a scene, state dump, CLI edit, or event log must be registered before loading a world:

```rust
registry.register::<Transform>();
registry.register::<Velocity>();
```

`ComponentRegistry` stores:

- Stable component name, e.g. `titan.core.Transform`.
- Schema version.
- Rust `TypeId`.
- JSON schema or schema descriptor for CLI validation.
- Type-erased serialize and deserialize functions.
- Type-erased clone/equality/debug helpers needed for state dumps and tests.

Unregistered components may exist only for private runtime caches and must be excluded from TSF and state dumps by type. Phase 1 should avoid private runtime components unless there is a concrete need.

`StateDump` is structured JSON with stable ordering:

```json
{
  "frame": 100,
  "seed": 1234,
  "entities": [
    {
      "id": 1,
      "components": {
        "titan.core.Transform": { "schema_version": 1, "value": {} }
      }
    }
  ]
}
```

### Determinism enforcement

Implementation must include deterministic tests before the ECS is considered
done:

- Same scene + same seed + same fixed-step inputs + same frame count produces
  byte-identical JSON state dumps.
- Query iteration is ascending `EntityId` after mixed spawn, despawn, and component insertion/removal.
- State dump entity order and component order are stable.
- Event log order is stable for spawn, despawn, component insert, component set,
  component remove, and system error events.

Runtime code used by deterministic systems must not read wall-clock time, process-global randomness, pointer addresses, thread IDs, unordered map iteration, or non-canonical floating-point formatting. Seeded RNG and fixed delta time are passed through `FixedStepContext`.

## Alternatives considered

### Adopt `hecs`

`hecs` is active, small, and close to Titan's desired API shape. It has `World::spawn`, insert/remove/get/query APIs and supports `serde` for entity handles. It intentionally has no framework-level system scheduler.

Why not choose it now:

- Titan would still need a wrapper for stable `EntityId`, deterministic sorted query iteration, component registry, state dump, event log, and schedule.
- Query order in an archetype ECS is not the external contract Titan needs; all public queries would need sorting by Titan ID.
- Full dynamic introspection is not the core feature of `hecs`; Titan would own most serialization logic anyway.

`hecs` remains the best later replacement candidate if Phase 2+ needs faster
storage behind the Titan-owned facade.

### Adopt standalone `bevy_ecs`

`bevy_ecs` is the most maintained and featureful Rust ECS candidate. It has a large ecosystem, mature schedule machinery, reflection hooks through Bevy's type registry, and standalone usage.

Why not choose it now:

- It is much larger than Phase 1 needs and would import Bevy-specific concepts before Titan has decided its own command/query surface.
- Bevy's `Entity` docs warn that direct serialized entity values make no long-term wire-format guarantee; Titan would still need separate stable IDs.
- Deterministic sorted public queries and byte-identical full dumps would still require a Titan-owned facade and registry discipline.
- Parallel scheduling is valuable later but is extra surface area for the deterministic default loop.

### Adopt `flecs_ecs`

Flecs is serious technology with relationships, prefabs, a powerful query language, and built-in metadata/JSON ambitions. The Rust binding is active but still comparatively young; docs describe Rust reflection as WIP.

Why not choose it now:

- It adds a C ECS core and binding layer to the Phase 1 foundation.
- Relationships/prefabs/query language are not needed for the minimum closed agent loop.
- Titan would still need stable text IDs, deterministic sorted dumps, and a Rust-first component registry contract.

### Adopt `shipyard`

`shipyard` is active and Rust-native, with good recent maintenance. It is a serious candidate for game ECS workloads.

Why not choose it now:

- Its strengths are ECS ergonomics and storage access, while Titan's Phase 1 risk is identity, serialization, introspection, and deterministic dumps.
- It would still need the same Titan facade as `hecs`, with a larger API shape to hide.

### Adopt `edict`, `legion`, `specs`, or other candidates

`edict` is interesting but was still on a release-candidate version in the checked crate metadata. `legion` and `specs` have historical importance but show substantially older maintenance activity than `hecs`, `bevy_ecs`, and `shipyard`. Smaller crates such as `flax`, `sparsey`, and `apecs` do not reduce Titan's core Phase 1 risk enough to justify adding a dependency.

### Build a high-performance in-house archetype ECS

Rejected for Phase 1. Titan needs a minimal deterministic core, not a new general-purpose ECS project. The first implementation should be deliberately small and benchmarked only enough to avoid obvious pathologies in the Phase 1 agent loop.

## Impact on the agent pipeline

The ECS decision directly defines what agents can observe and verify.

- `titan run --headless --frames N --dump-state out.json` will call `World::dump_state()` after running the deterministic fixed-step schedule.
- State dumps are stable JSON: entities sorted by `EntityId`, components sorted by stable component name, and component payloads serialized through the registry.
- Event logs are JSONL records emitted by ECS operations and systems:
  `entity_spawned`, `entity_despawned`, `component_inserted`,
  `component_removed`, `component_set`, and `system_error`.
- CLI scene editing can validate component names and payloads through the same
  `ComponentRegistry` used by runtime loading.
- Determinism CI must include: load fixture scene, run fixed inputs for N frames twice with the same seed, compare state dump bytes, and fail on any difference.

This keeps the GUI editor a client of the same command/query/state-dump path
used by agents.

## Open questions

- Whether TSF chooses decimal entity IDs, symbolic names mapped to IDs, or both.
- Whether component schemas should be generated from `serde`, `schemars`, a Titan derive macro, or a small custom schema trait.
- Whether Phase 2 performance requires swapping ordered storage for `hecs` behind the Titan-owned API.
