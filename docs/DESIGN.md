# Titan — Design & Roadmap

This document is Titan's constitution: the thesis it is built around, the
principles that decide how we build, and the ordered roadmap that follows from
them. It is a living anchor — updated as decisions land, not a one-time
manifesto.

For *what currently exists*, see the README and the code. This document is about
*where we are going and why*.

> **History.** Titan began as a Rust `std` prototype that proved the core thesis
> (deterministic agent loop: input-as-data, structured observation, a CPU
> software renderer, a CSG geometry kernel). That work is archived at
> `titan-engine/titan-prototype` and lives locally at `../titan-prototype`. It is
> a **reference quarry**, not dead weight: we port its ideas — and its working
> code — deliberately, layer by layer, fixing known debt (notably the non-robust
> CSG kernel) on the way in. This repo is a clean restart on a new *foundation*,
> not a new *vision*.

## Thesis

> A Titan game is a **deterministic pure function**:
> `(initial state, seed, action sequence) → (frames, observations)`.

Everything else follows from taking that literally. Titan is an **agent-native**
engine — designed first to be driven by AI agents, with humans and graphical
tools as first-class but optional participants — and determinism is what makes
that practical: reproducible runs, record/replay, exact bug reports, the ability
to replay or revert an action sequence precisely, and tests that assert on
outcomes rather than vibes.

Rendering is not the center of the engine; it is one **observation channel** out
of a simulation whose state is the real product.

## What's new this time: the foundation

The prototype proved the *thesis*. This restart is about the *foundation it runs
on*. Titan is now built as **`no_std`, zero-external-dependency, stable Rust**,
with every interface to the outside world made explicit. This is the long-held
"dream engine" — total control, nothing hidden, understandable top to bottom —
expressed in a language that makes that control *safe* and the explicit
interfaces *nicer* than the C original ever could be (a typed `trait` beats a
hand-rolled struct of function pointers).

The dream's *implementation choice* — write it in C — was nostalgia for control
from before Rust was understood. The dream's *principles* — explicit interfaces,
no hidden state, allocator-passing-for-everything, no dependency you didn't
choose — are exactly what serves the agent thesis, and Rust keeps them while
also giving us memory safety and enforced determinism. We keep the principles
and drop the language.

## Principles

These are the rules we use to resolve design questions. When a choice is
unclear, the option that best honors these wins.

1. **`no_std`, zero external dependencies.** Titan depends on `core` only. No
   `std`, no crates.io, no transitive supply chain. Every layer is ours, which
   means every layer is legible to a human or an agent reading top to bottom.
   Where we need a capability the ecosystem would normally provide (collections,
   allocation, math), we build the minimum we need, when we need it.

2. **Stable Rust only.** No nightly features. This is a deliberate constraint,
   and it is achievable: raw syscalls go through stable `core::arch::asm!`; a
   `no_std` binary builds on stable with `panic = "abort"` and a custom entry
   point; and we define **our own allocator abstraction** rather than waiting on
   the unstable `allocator_api`, since we never interoperate with foreign
   allocator implementations anyway. Tests still use the normal harness via
   `#![cfg_attr(not(test), no_std)]` — the shipped library is `no_std`, the test
   build links `std` and keeps `#[test]`, asserts, and `cargo test`.

3. **No hidden global state. Interfaces are explicit and passed.** There is no
   ambient allocator, no global logger, no implicit clock, no thread-local
   anything that the simulation can reach. Capabilities — allocation, the
   platform, time, randomness — are values passed *into* the code that needs
   them, threaded down the call stack. This is the Zig-allocator discipline
   generalized to every external interface, and it is the same discipline the
   agent thesis demands: a function's behavior is determined by its inputs.

4. **Purity is referential transparency, not immutability.** A function that
   takes `&mut World` and mutates it in place is pure as long as it is a
   deterministic function of its inputs with no reads from hidden global state.
   We mutate freely through exclusive references for performance; we simply never
   reach for a global. Rust's `&mut`-is-exclusive rule makes this *enforceable*
   rather than aspirational — a property C could never give us for free. "Pure
   and fast" is a Rust strength we lean into, not a tension we manage.

5. **Determinism is a tested invariant, not an accident.** The same inputs and
   seed produce byte-identical frames and state across runs and machines. Seeded
   RNG threaded explicitly (no ambient randomness), deterministic iteration
   order, a fixed-timestep core, and replay tests that fail loudly on divergence.

6. **Code is the single source of truth.** Everything needed to build, modify,
   or understand a game lives in readable code. A visual editor may *inspect*
   what the code describes; it is never required to change anything. Assets too:
   geometry is authored in-code via builder + CSG APIs so an agent can construct
   it programmatically.

7. **Everything an agent does is data.** Actions are data fed into the
   simulation; observations are data read out of it. Both are pluggable and
   serializable. This is what turns "drive the game" into "emit an action
   sequence" and "replay" into "store an action log."

8. **Headless-first, library-first.** The primary interface is an in-process
   Rust API: an agent (or test) constructs a world, steps it with actions, and
   reads observations — no window, no real-time pacing required. Windowing and
   wall-clock pacing sit *on top of* this headless core.

## The platform layer

`no_std` + no libc means Titan must reach the operating system itself for the
handful of things it genuinely needs (write to a file/stdout, read input, get a
raw timestamp, map memory). We model the OS as an **explicit, swappable
interface** — a Rust `trait` (the principle-3 discipline applied to the kernel
boundary):

- **Linux:** raw syscalls via stable inline `asm!`. No libc, static, inlinable to
  a few instructions — the original dream, and on Linux the syscall ABI is
  stable and documented, so it's safe to target directly.
- **macOS:** syscalls go through **`libSystem` symbol bindings**, not raw `svc`
  instructions. Apple does *not* provide a stable syscall ABI — direct syscalls
  are forbidden and can break between releases — so on Darwin "no deps" means "no
  deps except the one the platform mandates." We bind the **thinnest symbols that
  are still documented** (the `man 2` set — `write`, `read`, `mmap`, …), since a
  documented symbol is a stable contract. Linkage is just an `extern "C"` block
  under `#[link(name = "System")]`; Rust links `libSystem` itself, no libc crate
  needed. This is the primary dev platform, so it has to be first-class.
- **Unknown / embedded / wasm:** a **user-provided implementation** of the
  platform trait. Anyone can satisfy the interface with their own backend; Titan
  has no opinion about how the bytes leave the process.

Because the platform is a passed-in value, the entire engine above it is
testable against a fake platform and is trivially portable: porting Titan is
implementing one trait.

## Memory & allocation

Allocation is an explicit capability (principle 3), never global. We define our
**own allocator trait** — a small interface (`alloc`/`dealloc` over a `Layout`)
that the platform layer can satisfy from raw `mmap`/`munmap` (Linux) or the
mandated symbols (macOS), and that tests can satisfy with a bump or tracking
allocator. We do **not** use the `alloc` crate at all — not even for early
bring-up. Its `#[global_allocator]` model is exactly the hidden ambient state
we're banning, and a dependency leaned on "just to get started" is one we'd lean
on too hard and resent later. We build our collections properly from day one.

The cost is real and we accept it consciously: it means growing our own
collections (a `Vec`-equivalent, a hash map) that take an allocator parameter, as
we need them — not all at once. This is the same "build the minimum we need, when
we need it" rule as principle 1.

## Two front-ends, one core

Agent-friendly and human-friendly are not in tension — deterministic,
discoverable, structured, replayable, no-hidden-globals is simply good design,
and humans benefit from it too. The only real difference is the *surface*: agents
want structured/programmatic/textual interaction; humans want visual, interactive,
immediate feedback. We serve both without compromise by keeping **one
deterministic core** and putting **two front-ends** on it:

- **Agent front-end:** the programmatic action/observation API —
  `step(action) → observation` plus `render()`, headless, reproducible.
- **Human front-end:** an interactive viewer with hot-reload / immediate feedback,
  built *on* the same core. The CPU software renderer (ported from the prototype)
  is the seed of the "see it instantly" loop, and determinism is what makes the
  human tooling trustworthy too.

Neither front-end owns the core; both are clients of it.

## Rendering as an observation channel

Carried forward from the prototype, unchanged in spirit:

- The **CPU software renderer is the executable specification** — permanent, the
  deterministic GPU-free reference. GPU backends will never match it bit-for-bit
  (GPU float isn't reproducible across hardware), so determinism lives on the CPU
  side; a GPU backend is required only to be **visually faithful within a
  perceptual tolerance**, enforced by differential render tests.
- Shading grows from the prototype's hardcoded Lambert toward a small
  **data-driven material model** (one model, implemented as CPU spec + later
  WGSL, kept honest by differential tests). **No shader IR yet**; a small custom
  IR only later, only if effects demand it. **SPIR-V is rejected** as the
  canonical/authoring form (it's an interchange format; a conformant CPU
  interpreter is the wrong altitude) — relevant only if we ever ingest *external*
  shaders.

## Roadmap

Ordered by leverage and dependency. The early items are about standing up the new
foundation and **de-risking its core premise before building on it**. The middle
and deep tracks are largely *ports* of proven prototype subsystems onto that
foundation.

### Foundation (the new, unproven part — do first)

1. **Prove the `no_std` + platform slice.** A `no_std`, zero-dep, stable-Rust
   binary that performs one syscall — write `"hello\n"` to stdout — through the
   platform `trait`, working on **both Linux (raw syscalls)** and **macOS
   (`libSystem` symbols)**. This single slice de-risks the entire premise:
   freestanding entry point, panic handler, the platform abstraction, and the
   cross-OS story, all at once. Nothing else is built until this is real.

2. **Allocation as a capability.** The allocator trait, backed by the platform
   (raw `mmap` on Linux / mandated symbols on macOS) and by a test allocator.
   Then the first owned collection (a growable array) that takes an allocator
   parameter — the substrate everything above needs.

3. **Math core.** Port the prototype's `titan-math` (vectors, matrices,
   transforms) to `no_std`/`core` — it's the most reusable, lowest-risk port and
   unblocks geometry and rendering.

### Port the thesis (proven in the prototype — re-base onto the foundation)

4. **Deterministic action input** — re-base `titan-input` (action-as-data,
   `InputState`, scripted/recorded sources).
5. **Structured observation** — re-base `titan-observe` (serializable world
   snapshots), and add the explicit `step(action) → observation` interaction
   wrapper with reset/revert the prototype deferred. Serialization is ours,
   `no_std`, dependency-free, and starts **deliberately minimal** — a simple
   string format convention is enough to begin; it evolves as needs grow. We do
   not reach for a format crate or a schema system up front.
6. **Fixed-timestep app loop** — re-base the `Time` / `Game` / `App` driver.
7. **Geometry & CPU renderer** — port `titan-geometry` (mesh, CSG, `Shape`) and
   `titan-render` (software rasterizer). The CSG re-port is the chance to address
   the prototype's non-robust-kernel debt rather than carry it forward.

### Deep tracks (parallel / later)

8. **Determinism hardening** — seeded RNG threaded explicitly; replay/record
   tests asserting byte-identical reproduction.
9. **Human front-end** — interactive viewer + hot-reload on the deterministic
   core.
10. **ECS-backed world** — when entities multiply, shaped by observation needs.
11. **wgpu backend** behind the renderer trait; **physics** (deterministic from
    day one — approach chosen up front); **remote agent-driver protocol** (only
    if the in-process API proves insufficient).

## Non-goals / deferred

- **`std`, libc, or any external crate** in the shipped engine. (Test-only `std`
  is fine; see principle 2.)
- **Nightly Rust.**
- **A global allocator / any ambient global capability.**
- **SPIR-V as the canonical shader form.**
- **A mandatory editor.** Tooling may inspect; it never owns the source of truth.
- **Real-time wall-clock pacing as the core.** The core is headless and
  deterministic; real-time windowing is a layer on top.

Decided since first draft: **no `alloc` crate** (own collections from day one,
see Memory & allocation); **serialization starts as a minimal string format
convention** and evolves (see roadmap 5); **macOS linkage** uses the thinnest
documented (`man 2`) symbols via an `extern "C"` block under
`#[link(name = "System")]` (see platform layer). What remains open:

- **Material model scope** — how far toward full PBR, how soon. *Too early to
  answer; revisit at the rendering port.*
- **Deterministic physics strategy** — fixed-point vs. carefully-managed float.
  *Too early to answer; must be chosen before physics work begins, not after.*

---

*This document supersedes any README stub for forward-looking direction; the
README remains the entry point for what exists today.*
