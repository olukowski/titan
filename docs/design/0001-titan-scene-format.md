# Titan Scene Format

Status: draft

## Context

Phase 1 needs a text scene format that agents can create, inspect, edit, format,
validate, and diff without a GUI. The format stores scenes as entities with typed
components, must validate against a schema, must support references to other
files, and must serialize deterministically so a value change is usually a
one-line diff.

The decision is not only syntax. `titan scene edit` needs stable path addressing,
`titan scene fmt` needs a canonical writer, and error messages need spans that
point back into the source file.

The current Rust ecosystem matters:

- Serde is the common Rust data-model layer for text formats, and describes
  itself as a framework for serializing and deserializing Rust data structures
  across supported formats: <https://serde.rs/>.
- `schemars` can derive JSON Schema from Rust types with `#[derive(JsonSchema)]`
  and `schema_for!`: <https://docs.rs/schemars>.
- `jsonschema` validates JSON instances and schemas, including meta-schema
  validation: <https://docs.rs/jsonschema>.
- JSON5 is a JSON superset that adds ECMAScript-derived conveniences while
  retaining JSON's primitive/object/array data model:
  <https://spec.json5.org/>.

## Goals & non-goals

- Goals: human-readable and pleasant enough to hand-edit; trivially
  machine-editable with path-addressed get/set operations; deterministic
  canonical serialization with stable ordering and formatting; schema validation
  for the file shape and component payloads; comments for intent and temporarily
  disabled data; cross-file references from scenes to assets, scripts, and later
  procedural geometry.
- Non-goals: a scripting or expression language; binary, compressed, or
  streaming scene storage; perfect preservation of arbitrary human formatting
  after `titan scene fmt`; editor-only metadata that cannot be represented
  through the CLI/API.

## Proposed design

TSF v1 is a constrained JSON5 profile with Titan-owned canonical formatting.
Files use the `.tsf` extension. Parsers accept JSON5 syntax, but the semantic
model is JSON: objects, arrays, strings, finite numbers, booleans, and null.
Validation converts the parsed value to JSON and runs Titan's generated JSON
Schema plus semantic validation passes.

This recommendation deliberately chooses JSON5's data model, not arbitrary
JavaScript syntax. TSF v1 forbids values that do not round-trip through JSON:
`Infinity`, `NaN`, hexadecimal numbers, leading-plus numbers, and duplicate
object keys are validation errors. Comments and trailing commas are allowed in
source.

Top-level structure:

```json5
{
  tsf: 1,
  scene: {
    id: "scene:demo/moving_box",
    name: "Moving Box",
  },
  assets: {
    cube_mesh: { path: "../assets/cube.tgeo", kind: "geometry" },
    red_material: { path: "../assets/red.tmat", kind: "material" },
  },
  entities: [
    {
      id: "entity:box",
      name: "Box",
      components: {
        transform: {
          translation: [0.0, 0.0, 0.0],
          rotation: [0.0, 0.0, 0.0, 1.0],
          scale: [1.0, 1.0, 1.0],
        },
        velocity: {
          linear: [1.0, 0.0, 0.0],
          angular: [0.0, 0.0, 0.0],
        },
        mesh: {
          geometry: { ref: "asset:cube_mesh" },
          material: { ref: "asset:red_material" },
        },
      },
    },
  ],
}
```

Required top-level keys, in canonical order: `tsf` integer version, `scene`
object, `assets` object, and `entities` array. `scene` requires `id` and may
include `name` and `metadata`. `assets` is keyed by local alias; each value has
at least `path` and `kind`. Entities require stable `id` strings in
`entity:<slug>` form, may include `name` and `parent`, and store component data
under `components`.

Component mapping to ECS:

- Each key in `components` is a registered component type ID, for example
  `transform`, `velocity`, `camera`, `mesh`.
- Each value is the component's serialized payload.
- Deserialization looks up the component type in the component registry, validates
  the payload against that component's schema, and inserts the typed component
  into the ECS world for that entity.
- Unknown component IDs are errors unless validation is run with an explicit
  compatibility mode for forward migration.
- Component schemas are generated from Rust component types where possible and
  may add semantic validation not expressible in JSON Schema, such as normalized
  quaternions or existing asset references.

IDs are stable across formatting and editing; runtime entity handles are never
serialized. References are objects with one `ref` string, not bare strings, so
they can grow later without changing the data model. Entity references use the
full serialized entity ID, such as `entity:box`, for same-scene entities.
Reference prefixes are `asset:<alias>` for top-level assets,
`scene:<path>#entity:<slug>` for another scene file, and
`file:<relative-path>` for direct file references. Relative paths resolve from
the containing `.tsf` file's directory and are normalized to forward slashes by
`titan scene fmt`.

Canonical ordering and formatting:

- UTF-8, LF line endings, final newline.
- Two-space indentation.
- Objects use one property per line.
- Arrays of scalars with length 4 or less may stay on one line; all other arrays
  use one item per line.
- Top-level object order is `tsf`, `scene`, `assets`, `entities`.
- `scene` order is `id`, `name`, `metadata`.
- Entity order is `id`, `name`, `parent`, `components`.
- Component keys sort lexicographically unless a component registry supplies a
  higher-priority canonical order for built-ins. v1 built-in order:
  `transform`, `velocity`, `mesh`, `camera`, `light`.
- Payload object keys sort lexicographically unless the component schema declares
  a field order. Built-ins should declare field order for readability.
- Asset aliases sort lexicographically.
- Numeric formatting is canonical: finite decimal, no leading plus, no trailing
  decimal point, `-0` normalized to `0`. Component schemas may request fixed
  precision only when needed for deterministic simulation state dumps.
- Strings use double quotes only when required by JSON5 syntax or for all values;
  canonical output may leave simple object keys unquoted.
- Comments are preserved when the parser can attach them unambiguously to a
  following object key, array item, or entity/component block. Floating comments
  may move to the nearest following item. `fmt` may drop whitespace-only layout,
  never semantic data.

Path addressing uses JSON Pointer syntax with Titan extensions for entity IDs:
`/entities/entity:box/components/velocity/linear/0`, `/assets/cube_mesh/path`,
and `/scene/name`. Numeric array segments address array indices, but
`entities/<entity-id>` resolves by entity `id` because entity array order is not
identity. CLI output also reports the resolved canonical JSON Pointer with the
physical array index for precise spans when useful.

Complete Phase 1 example:

```json5
{
  tsf: 1,
  scene: {
    id: "scene:examples/moving_entity",
    name: "Moving Entity",
  },
  assets: {},
  entities: [
    {
      id: "entity:mover",
      name: "Mover",
      components: {
        transform: {
          translation: [0.0, 0.0, 0.0],
          rotation: [0.0, 0.0, 0.0, 1.0],
          scale: [1.0, 1.0, 1.0],
        },
        velocity: {
          linear: [0.1, 0.0, 0.0],
          angular: [0.0, 0.0, 0.0],
        },
      },
    },
  ],
}
```

Sample one-line diff from `titan scene edit
/entities/entity:mover/components/velocity/linear/0 0.2` followed by
`titan scene fmt`:

```diff
-          linear: [0.1, 0.0, 0.0],
+          linear: [0.2, 0.0, 0.0],
```

## Alternatives considered

RON is attractive because it is Rust-shaped, supports Serde's data model, and has
`ron::ser::PrettyConfig` for pretty output
(<https://docs.rs/ron>, <https://docs.rs/ron/latest/ron/ser/struct.PrettyConfig.html>).
It also has comments and trailing commas in normal use. It is not recommended for
TSF v1 because schema tooling is not as standard as JSON Schema, non-Rust
tooling is weaker, and path-addressed editing would need Titan-specific AST and
span machinery anyway. RON is pleasant for Rust developers, less universal for
agents and external tools.

KDL is pleasant for humans and explicitly document-oriented. The official Rust
crate preserves formatting, whitespace, comments, and relative item order, and
describes itself as `toml_edit`-like for KDL (<https://github.com/kdl-org/kdl-rs>).
KDL 2.0 is stable as of 2024-12-21 (<https://kdl.dev/spec/>). It was not chosen
because the Rust ecosystem around Serde-style typed deserialization and schema
validation is less direct than JSON Schema. KDL paths also need a custom query
language because nodes can have names, arguments, properties, and children rather
than a single JSON-like object/array model.

TOML has excellent Rust support. `toml_edit` preserves comments, spaces, and
relative order (<https://docs.rs/toml_edit/>), while `toml` covers the simpler
Serde-oriented use case (<https://epage.github.io/blog/2023/01/toml-vs-toml-edit/>).
Taplo provides TOML parser/analyzer/formatter tooling
(<https://lib.rs/crates/taplo>). TOML was not chosen because arrays of entities
with nested component payloads become noisy and table-heavy. It is ideal for
project config, but scene graphs are more naturally object/array documents.

Strict JSON has the strongest schema and editing ecosystem, especially with
JSON Pointer and JSON Schema, but it lacks comments and trailing commas. That
hurts hand-authored scenes. Strict JSON remains the validation data model and
machine interchange target, but not the source syntax.

JSON5 keeps JSON's data model while adding comments, trailing commas, unquoted
object keys, and single-quoted strings. Rust options include `serde_json5`
(<https://docs.rs/serde_json5>) and `json-five-rs`, which advertises Serde
compatibility, comment/whitespace-preserving round trips, AST edits, token-based
round trips, and formatter support (<https://github.com/spyoungtech/json-five-rs>).
Generic JSON5 alone is too loose for deterministic engine data, so TSF uses a
strict profile and canonical formatter.

A fully custom format could optimize every detail for scenes, but it would force
Titan to own parsing, spans, schema tooling, formatting, editor integrations, and
agent familiarity from day one. That is not a good Phase 1 tradeoff. The useful
custom part is the TSF profile, schema, path semantics, and formatter on top of a
known syntax.

## Impact on the agent pipeline

The CLI surface should be implemented before any GUI editing path:

- `titan scene validate path.tsf --json` parses JSON5, rejects non-TSF profile
  values, validates JSON Schema, validates references, and returns structured
  diagnostics.
- `titan scene query path.tsf /entities/entity:mover/components/velocity --json`
  returns the selected JSON value plus source span and resolved canonical path.
- `titan scene edit path.tsf /entities/entity:mover/components/velocity/linear/0
  0.2 --json` parses the replacement as JSON5 scalar/object/array, validates the
  affected component or file, updates the AST, and writes canonical TSF.
- `titan scene fmt path.tsf --check --json` reports whether the file is already
  canonical; without `--check`, it rewrites the file.

Structured error shape:

```json
{
  "ok": false,
  "errors": [
    {
      "code": "TSF_UNKNOWN_COMPONENT",
      "message": "unknown component 'velocty'",
      "path": "/entities/entity:mover/components/velocty",
      "span": {
        "file": "examples/moving.tsf",
        "start": { "line": 14, "column": 9 },
        "end": { "line": 14, "column": 16 }
      }
    }
  ]
}
```

Deterministic checks:

- `fmt` must be idempotent: formatting an already formatted file produces no
  bytes changed.
- `validate` must fail duplicate keys before schema validation.
- `edit` must not reorder unrelated entities or components beyond canonical
  ordering.
- `query` and `edit` paths must work without knowing entity array indices.
- CI should include golden tests for parse -> fmt -> parse, one-line scalar
  diffs, duplicate-key diagnostics, and broken reference diagnostics.

## Open questions

- Whether v1 should preserve comments through every `edit` or only through
  `fmt` best-effort attachment.
- Whether built-in component field ordering lives in schema metadata or Rust
  component registry metadata.
