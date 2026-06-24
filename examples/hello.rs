//! Foundation slice: a `no_std`, zero-dependency, stable-Rust binary that writes
//! "hello" and exits — entirely through the explicit [`Platform`] boundary.
//!
//! It proves the whole premise at once: a freestanding entry point, a panic
//! handler, and the Linux (raw syscall) / macOS (libSystem) split behind one
//! interface. There is no libc startup, no `std`, and nothing global.
//!
//! Run it with `cargo run --example hello`; it prints `hello` and exits `0`.
#![no_std]
#![no_main]

use titan::platform::{Os, Platform, STDOUT};

/// The actual work, written against the [`Platform`] interface rather than any
/// concrete OS: write the greeting, then exit cleanly through the platform.
fn run(platform: &impl Platform) -> ! {
    platform.write(STDOUT, b"hello\n");
    platform.exit(0)
}

/// Linux: with no libc there is no C runtime, so the kernel jumps straight to
/// `_start` (see `.cargo/config.toml`, which drops the startup files). We own
/// the entry point and never return.
#[cfg(target_os = "linux")]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    run(&Os)
}

/// macOS: libSystem's C runtime provides `start`, which calls `main`. We ignore
/// `argc`/`argv` and exit through the platform instead of returning to the crt.
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn main() -> ! {
    run(&Os)
}

/// Required for a `no_std` binary. Nothing here panics yet; if anything ever
/// does, fail loudly with a distinct status rather than risk undefined behavior.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    Os.exit(101)
}
