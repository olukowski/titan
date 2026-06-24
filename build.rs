//! Build script: scope the freestanding link flags to *example* targets only.
//!
//! The `hello` example is a `no_std`, no-libc binary that owns its `_start`
//! entry point, so on Linux it must be linked without the C runtime startup
//! files (`-nostartfiles`) or default libraries (`-nostdlib`), as a static,
//! non-PIE executable (`-static -no-pie`).
//!
//! These flags must NOT reach the library's own test harness, which links `std`
//! (and thus libc) the normal way. `rustc-link-arg-examples` scopes them to
//! example targets, and the `target_os` gate keeps them off macOS, where the
//! example links `libSystem` normally.
//!
//! (A build script is host tooling — it does not make the engine depend on
//! anything at runtime.)
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        for flag in ["-nostartfiles", "-nostdlib", "-static", "-no-pie"] {
            println!("cargo::rustc-link-arg-examples={flag}");
        }
    }
    println!("cargo::rerun-if-changed=build.rs");
}
