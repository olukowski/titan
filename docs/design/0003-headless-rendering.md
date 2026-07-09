# Headless rendering

Status: draft

## Context

Phase 2 closes the first visual feedback loop. An agent must be able to take a
text scene, render it without opening a window, inspect the resulting image,
and verify useful facts without image interpretation. The renderer is also the
foundation for the later `titan view` application; a second rendering path for
the viewer would violate Titan's headless-first and agent-first constraints.

The implementation has four constraints:

- Rendering must be usable in a process with no display server or window.
- Scene data must remain TSF components and references, not renderer-specific
  sidecar state.
- The same fixed-timestep run must be able to produce a sequence of captures.
- CI and agents need machine-readable render statistics even when pixels are
  unavailable or differ slightly between GPUs.

This document chooses `wgpu` for the rendering abstraction and a small
forward renderer for Phase 2. It does not design a full material graph,
deferred renderer, lighting model, or asset pipeline.

## Goals & non-goals

Goals:

- Add a `titan_render` crate that renders a loaded ECS world through a stable,
  renderer-owned interface.
- Support a forward pipeline with cameras, directional lights, unlit materials,
  and basic PBR materials.
- Make `titan render scene.tsf --camera main --out frame.png` the primary Phase
  2 deliverable.
- Capture frame sequences from deterministic runs with
  `titan run --capture-every N`.
- Emit structured render statistics, including draw calls and triangle count,
  for pixel-free checks.
- Define the TSF component payloads and asset references needed by the first
  renderer while preserving 0001's canonical formatting and reference rules.
- Leave `titan view` as a thin later shell over the same render service.

Non-goals:

- A deferred renderer, clustered lighting, shadows, post-processing, skinning,
  animation, particles, or transparency sorting beyond a documented basic
  policy.
- A general-purpose material or shader graph.
- Defining TitanGeo v1; procedural geometry remains the Phase 3 design.
- Promising bit-identical pixels across different GPU vendors and drivers.
- A privileged GUI/editor rendering API.

## Proposed design

### Crate and renderer boundary

Create `crates/titan_render` with `wgpu` as its graphics dependency. The crate
owns device/queue creation, shader modules, render targets, GPU buffers,
resource caches, and render statistics. It must not own the ECS or the TSF
parser. `titan_core` remains responsible for world state and fixed-timestep
execution; `titan_scene` remains responsible for loading and validating TSF.

The central operation is conceptually:

```text
RenderService::render(world, RenderRequest) -> RenderResult
```

`RenderRequest` selects a camera entity, output dimensions, clear color, and
capture mode. The service extracts render components from the world in stable
entity order, resolves their asset references, uploads or reuses GPU resources,
submits one forward pass, and returns statistics plus the image when requested.
The public API should expose Titan-owned request/result types rather than
`wgpu` handles so the CLI and a future viewer share the same contract.

The initial pipeline is:

1. Create a `wgpu::Instance`, request an adapter, device, and queue, and create
   an offscreen texture for headless rendering.
2. Select exactly one active camera for a render request.
3. Build view and projection matrices from the camera and entity transform.
4. Iterate visible mesh entities in ascending stable entity ID. Bind each
   mesh, material, transform, and the selected directional-light data.
5. Draw into the offscreen color target, then copy it to a CPU-readable buffer
   and encode PNG when requested.

The first pass uses depth testing, back-face culling, a color target in an
sRGB format, and one draw per mesh/material grouping. Blending is disabled
for opaque materials. A material marked transparent is out of scope for the
first implementation and must be rejected with a structured diagnostic
rather than silently rendered incorrectly.

### Cameras and lighting

`Camera` is an entity component. The camera's entity `transform` defines its
world position and orientation; the camera payload defines projection. Phase 2
supports perspective and orthographic projections:

```json5
camera: {
  projection: "perspective",
  vertical_fov_degrees: 60.0,
  near: 0.1,
  far: 1000.0,
  viewport: { width: 640, height: 480 },
}
```

`vertical_fov_degrees`, `near`, and `far` are required for perspective.
Orthographic cameras use `height`, `near`, and `far` instead. The optional
viewport is a camera default; `--width` and `--height` on a render command
override it. A scene may contain multiple cameras, but a command must select
one by stable entity ID or entity name. `--camera main` resolves the serialized
entity name `main`, and ambiguity is an error.

`DirectionalLight` is also an entity component. Its entity transform's forward
direction is the light direction; the payload contains color, illuminance, and
whether the light participates in ambient fill:

```json5
directional_light: {
  color: [1.0, 1.0, 1.0],
  illuminance: 1.0,
  ambient: 0.05,
}
```

Phase 2 uses the first directional light in ascending entity ID as the active
light. Additional directional lights are reported in stats as ignored rather
than changing the result implicitly. With no light, the renderer uses a fixed
black-to-gray ambient fallback suitable for unlit previews and reports
`active_directional_lights: 0`.

### Meshes and materials

`Mesh` is the scene-facing component and contains a reference to a resolved
geometry asset. It does not inline vertex data in the scene:

```json5
mesh: {
  geometry: { ref: "asset:cube_mesh" },
  submeshes: [0],
}
```

The geometry asset is an entry in the TSF `assets` object with `kind:
"geometry"`. `titan_render` consumes a small resolved `MeshAsset` interface
(positions, normals, UVs when present, indices, and submesh ranges). Phase 2
may ship a minimal fixture geometry loader and built-in test meshes so the
renderer can prove the red-cube path; it must not encode a second scene syntax.
The Phase 3 TitanGeo loader will produce the same resolved interface, and glTF
import remains an escape hatch rather than a Phase 2 requirement.

`Material` is a component with a material asset/reference plus the supported
shading values. Keeping the values in the component makes the simplest agent
edits one-line TSF changes and leaves room for a later material asset format:

```json5
material: {
  model: "unlit",
  base_color: [1.0, 0.0, 0.0, 1.0],
}
```

The `pbr` model adds `metallic` and `roughness`:

```json5
material: {
  model: "pbr",
  base_color: [0.8, 0.2, 0.1, 1.0],
  metallic: 0.0,
  roughness: 0.5,
}
```

`base_color` is linear RGBA in the component model; the renderer performs the
appropriate conversion for its sRGB target. Values are finite and range
checked. PBR uses a single directional light, a fixed ambient term, a
metallic-roughness approximation, and no image-based lighting. The shader
version and material model are included in render stats so golden changes can
be attributed to renderer changes.

### TSF serialization and component registration

The component IDs are the lowercase registered names `camera`,
`directional_light`, `mesh`, and `material`, alongside the existing
`transform`. They are serialized under an entity's `components` object exactly
as specified by design doc 0001. References are objects with a `ref` key,
never bare asset strings. A representative scene fragment is:

```json5
{
  tsf: 1,
  scene: { id: "scene:examples/red_cube" },
  assets: {
    cube_mesh: { path: "../assets/cube.mesh", kind: "geometry" },
  },
  entities: [
    {
      id: "entity:main_camera",
      name: "main",
      components: {
        camera: {
          projection: "perspective",
          vertical_fov_degrees: 60.0,
          near: 0.1,
          far: 100.0,
        },
        transform: {
          translation: [0.0, 0.0, 3.0],
          rotation: [0.0, 0.0, 0.0, 1.0],
        },
      },
    },
    {
      id: "entity:cube",
      components: {
        material: {
          model: "unlit",
          base_color: [1.0, 0.0, 0.0, 1.0],
        },
        mesh: {
          geometry: { ref: "asset:cube_mesh" },
        },
        transform: {
          translation: [0.0, 0.0, 0.0],
        },
      },
    },
  ],
}
```

Canonical component ordering remains the registry-defined built-in order from
0001, extended to `transform`, `velocity`, `camera`, `directional_light`,
`mesh`, `material` (then lexicographic ordering for other components). Each
new component declares canonical payload field order: projection fields and
viewport for `camera`; color, illuminance, ambient for `directional_light`;
geometry, submeshes for `mesh`; and model, base_color, metallic, roughness for
`material`. Omitted fields use schema defaults only when the default is
explicitly documented and deterministic. Asset aliases and relative paths
follow 0001 unchanged.

### CLI commands and structured output

The primary command is:

```text
titan render scene.tsf --camera main --out frame.png
```

It loads and validates TSF, resolves assets, renders one frame without opening
a window, and writes PNG. `--json` writes a structured result to stdout (the
PNG remains at `--out`):

```json
{
  "ok": true,
  "frame": 0,
  "camera": "entity:main_camera",
  "output": "frame.png",
  "width": 640,
  "height": 480,
  "draw_calls": 1,
  "triangles": 12,
  "visible_meshes": 1,
  "active_directional_lights": 0,
  "backend": "wgpu",
  "adapter": "...",
  "shader_version": 1,
  "material_models": { "unlit": 1 }
}
```

The exact adapter name is informational and must not be used as a golden key.
Errors use the repository's structured error shape with TSF paths for invalid
components and asset references. `--stats-json path.json` is an optional
convenience for scripts that want stats in a file while reserving stdout for
normal CLI output; `--json` remains supported on every command.

`run` gains capture options:

```text
titan run scene.tsf --headless --frames 120 --capture-every 10 \
  --capture-dir .titan/cache/captures --camera main --json
```

Captures occur after the fixed-step update for frames `N`, `2N`, and so on;
frame zero is captured only with an explicit `--capture-initial` option. Files
use stable zero-padded names such as `frame-000010.png`. Each capture also
produces a JSONL stats record containing the simulation frame, state seed,
image path, and render statistics. A failed capture fails the run with a
structured error. Capture output belongs in `.titan/cache/` by default, not in
source paths.

`view` is a later thin shell. It creates a window surface, forwards resize and
input events, and invokes the same `RenderService` with a surface target. It
must not duplicate scene loading, camera selection, resource resolution,
shaders, or render-stat computation. Headless rendering is the supported
implementation and test surface.

### Determinism, portability, and CI

Simulation determinism remains the 0002 fixed-timestep contract. Rendering adds
three levels of verification:

- Render stats are exact, structured checks. CI can assert camera resolution,
  visible mesh count, draw calls, triangle count, material model counts, and
  resource errors without comparing pixels.
- PNG goldens are opt-in and compared with a documented per-channel tolerance
  and an allowed differing-pixel percentage. Goldens are pinned to a named
  backend/adapter class, shader version, output size, and color format; they
  are not expected to be byte-identical across GPUs.
- A small set of camera/frustum and asset-resolution tests can run without a
  GPU by testing the CPU-side render extraction and stats input.

The default backend is `wgpu`'s Vulkan/Metal/DX12 selection on native systems.
Headless initialization must not create a window or require a display server.
CI has two supported paths: a GPU-backed runner for pixel goldens and a
software backend such as llvmpipe for portable headless smoke tests and stats.
The first implementation should detect adapter/backend availability and return
an actionable structured `RENDER_NO_ADAPTER` error; it must not silently skip
the render test. If a software adapter is not available on a platform, that
platform runs CPU extraction tests and marks GPU tests as an explicit CI
capability requirement rather than weakening the render contract.

Shader math must avoid unordered iteration and unspecified resource selection.
Entity, light, and material grouping order is stable. Statistics are computed
from the submitted render plan, not driver query timing. Render goldens must
be regenerated intentionally when shader code, wgpu, or the pinned test
adapter changes.

### Phased implementation plan

Keep implementation PRs small and independently reviewable:

1. Add `titan_render` crate boundaries, render request/result types, component
   registry metadata, CPU-side extraction, and stats tests. No GPU image yet.
2. Add `wgpu` device selection and an offscreen color/depth target, including a
   headless adapter smoke test and structured no-adapter errors.
3. Add camera math, transform extraction, fullscreen clear, and a one-triangle
   or fixture-mesh draw. Verify projection/frustum tests and a first PNG.
4. Add geometry asset resolution, mesh GPU buffers, stable draw ordering, and
   exact draw-call/triangle stats.
5. Add unlit and basic PBR shaders, directional-light selection, material
   validation, and the red-cube render fixture.
6. Add `titan render` CLI wiring, PNG encoding, `--json`, camera selection,
   and structured diagnostics.
7. Add `titan run --capture-every`, deterministic frame naming, JSONL capture
   stats, and end-to-end fixed-timestep capture tests.
8. Add CI capability detection, software-backend smoke coverage, and narrowly
   scoped pixel goldens with tolerance. Do not modify workflow files in the
   design-doc PR; workflow changes belong in the implementation PR that adds
   the tests.
9. Later, add `titan view` as a surface-only shell over `RenderService`.

## Alternatives considered

### OpenGL or a window-first renderer

OpenGL has broad availability, but headless context creation and portability
vary by platform, and a window-first abstraction encourages the viewer to become
the real implementation. `wgpu` gives one Rust API over native and software
backends and supports offscreen textures directly. OpenGL is not chosen for the
Phase 2 renderer.

### Bevy renderer or a complete game-engine stack

Bevy could provide a renderer and asset ecosystem quickly, but it would couple
Titan's TSF/ECS contracts to Bevy's world, schedules, asset handles, and window
lifecycle. Titan needs a small renderer service that can be driven by the CLI
and later replaced or optimized behind its own API. Bevy is not chosen as the
rendering boundary.

### Deferred or physically complete PBR rendering

Deferred rendering and full PBR are valuable at scale but add render targets,
lighting passes, and asset requirements before the agent feedback loop exists.
A forward pass with unlit plus basic metallic-roughness PBR is sufficient for
the Phase 2 exit criterion and easier to inspect in stats. More advanced paths
can be added behind the same material interface.

### CPU rasterization as the primary renderer

A CPU rasterizer would make CI simple and pixel output highly reproducible,
but it would create a second shading implementation and would not exercise the
GPU path used by the viewer. It remains useful as a future reference or
fallback, but not as the primary Phase 2 renderer. llvmpipe is the preferred
software fallback because it exercises the same `wgpu` shaders.

### Byte-identical PNG goldens on every machine

This is rejected because floating-point math, shader compilation, color
conversion, and driver behavior can differ across adapters. Exact render stats
and tolerant, adapter-pinned image goldens provide stronger and more useful
verification than pretending cross-GPU byte identity exists.

## Impact on the agent pipeline

Phase 2 adds the first visual observation command to the closed loop:

- An agent edits TSF text, runs `titan scene validate`, renders with
  `titan render`, and receives both a PNG and structured stats.
- An agent can run a deterministic simulation with periodic captures and match
  each image to a frame number, seed, and stats JSONL record.
- Agents can assert visibility-related facts such as one visible mesh, twelve
  cube triangles, one draw call, and an active camera without reading pixels.
- `titan view` remains a human client of the same headless service, so any
  behavior available in the viewer remains available through CLI/API paths.
- TSF changes are ordinary canonical text edits: new components validate
  through the same registry/schema path, references are explicit, and
  `titan scene fmt` preserves stable ordering and formatting.

The Phase 2 exit criterion is a fixture scene with a red cube and `main`
camera that can be rendered in a display-free process. Its end-to-end test
must check successful PNG creation, structured stats, and the expected mesh
and triangle counts; a tolerant pixel check may additionally verify that the
cube occupies the expected image region on the pinned CI adapter.

## Open questions

- Whether the Phase 2 fixture geometry format should remain private test data,
  or become the first public non-procedural mesh asset format before glTF
  import.
- Which CI runner and adapter identity should be pinned for pixel goldens.
- Whether `--camera` should accept only entity names/IDs or also a future
  scene-level camera alias map.
- Whether material values should move to `.tmat` assets once Phase 3 asset
  authoring is designed.
