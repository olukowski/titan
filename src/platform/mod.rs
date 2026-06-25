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

/// Byte output to a file descriptor.
///
/// The engine depends on this trait, never on a concrete OS. Code that only
/// needs to write takes a `&impl Console` and nothing more — least privilege at
/// the type level.
pub trait Console {
    /// Write `bytes` to file descriptor `fd`.
    ///
    /// Returns the number of bytes written, or a negative error code. As with
    /// the underlying `write(2)`, a *short write* is allowed: the return value
    /// may be less than `bytes.len()`, and callers that need every byte
    /// delivered must loop. We keep that raw contract here; looping convenience
    /// wrappers belong in a layer above, not in the platform boundary.
    fn write(&self, fd: i32, bytes: &[u8]) -> isize;
}

/// Process lifetime control.
pub trait Process {
    /// Terminate the current process with exit status `code`. Never returns.
    fn exit(&self, code: i32) -> !;
}

/// Anonymous memory mapping — the raw page supply the engine's allocators are
/// built on (see [`crate::alloc`]).
///
/// This is the OS boundary for memory, not an allocator: it hands back whole
/// page-aligned regions, and there is no notion of layout or reuse here. The
/// allocator layer above turns these regions into typed allocations.
pub trait Memory {
    /// Map `len` bytes of fresh, zero-filled, readable-and-writable private
    /// anonymous memory. Returns a pointer to at least `len` bytes, or null on
    /// failure.
    ///
    /// The mapping is page-granular: the OS rounds `len` up to a page, so the
    /// usable region may be larger than requested. The pointer is suitably
    /// aligned for any type (page alignment ≥ any `Layout` alignment we use).
    fn map(&self, len: usize) -> *mut u8;

    /// Unmap a region previously returned by [`map`](Memory::map).
    ///
    /// # Safety
    /// `ptr` must come from a prior `map` call and `len` must be the same value
    /// passed to that call. The region must not be used afterwards.
    unsafe fn unmap(&self, ptr: *mut u8, len: usize);
}

/// The full operating-system surface: everything a complete Titan backend
/// provides. This is a convenience umbrella over the focused capability traits —
/// most code should depend on just the capabilities it uses ([`Console`],
/// [`Process`], [`Memory`]) rather than on `Platform` as a whole.
///
/// Backends are zero-sized and passed in explicitly — there is no global
/// platform. The blanket impl means any type implementing all three
/// capabilities is automatically a `Platform`.
pub trait Platform: Console + Process + Memory {}

impl<T: Console + Process + Memory> Platform for T {}

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

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the *real* OS backend for the build's target: macOS libSystem on
    // the dev host, Linux raw syscalls in CI. A page is mapped, written across
    // its full length, read back, and unmapped.
    #[test]
    fn map_returns_writable_memory_and_unmap_releases_it() {
        let os = Os;
        let len = 4096;
        let p = os.map(len);
        assert!(!p.is_null(), "mmap failed");
        // SAFETY: `map` returned a region of at least `len` writable bytes; we
        // touch exactly that range, then release it with the same `len`.
        unsafe {
            for i in 0..len {
                p.add(i).write(i as u8);
            }
            assert_eq!(p.read(), 0);
            assert_eq!(p.add(len - 1).read(), (len - 1) as u8);
            os.unmap(p, len);
        }
    }
}
