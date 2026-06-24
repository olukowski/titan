//! The platform layer: Titan's single, explicit boundary to the operating
//! system.
//!
//! `no_std` and no libc means Titan must reach the OS itself for the handful of
//! things it genuinely needs. We model that as one swappable interface — the
//! [`Platform`] trait — rather than scattering raw syscalls through the engine.
//! Everything above this layer is OS-agnostic and can run against a fake
//! implementation in tests; porting Titan to a new system is implementing this
//! trait.
//!
//! There is no ambient, global platform: an implementation is passed in by
//! reference to the code that needs it (principle 3 in `docs/DESIGN.md`).
//!
//! The concrete backend for the current target is re-exported as [`Os`]:
//! - **Linux** — raw `aarch64` syscalls via inline `asm!`, no libc.
//! - **macOS** — the thinnest documented `libSystem` symbols (`man 2`).

/// Standard output — file descriptor `1`.
pub const STDOUT: i32 = 1;

/// Standard error — file descriptor `2`.
pub const STDERR: i32 = 2;

/// The operating-system capabilities Titan needs, as one explicit interface.
///
/// The engine depends on this trait, never on a concrete OS. Backends are
/// zero-sized and passed in explicitly — there is no global platform.
///
/// As the capability surface grows, this will likely be split into smaller,
/// focused traits (e.g. separate output, process control, memory mapping, clock)
/// so each piece of the engine can depend on only what it actually uses. For the
/// foundation slice, one small trait is enough.
pub trait Platform {
    /// Write `bytes` to file descriptor `fd`.
    ///
    /// Returns the number of bytes written, or a negative error code. As with
    /// the underlying `write(2)`, a *short write* is allowed: the return value
    /// may be less than `bytes.len()`, and callers that need every byte
    /// delivered must loop. We keep that raw contract here; looping convenience
    /// wrappers belong in a layer above, not in the platform boundary.
    fn write(&self, fd: i32, bytes: &[u8]) -> isize;

    /// Terminate the current process with exit status `code`. Never returns.
    fn exit(&self, code: i32) -> !;
}

// The Linux backend is raw `aarch64` syscalls — its syscall numbers and
// register usage are architecture-specific, so it must be gated by arch, not
// just OS. Building for Linux on another architecture is a clear error rather
// than silently wrong syscalls.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod linux;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub use linux::Os;

#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
compile_error!("Titan's Linux platform backend currently supports aarch64 only");

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::Os;
