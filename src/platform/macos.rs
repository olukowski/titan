//! macOS backend: the thinnest documented `libSystem` symbols (`man 2`).
//!
//! Apple does not provide a stable syscall ABI — direct `svc` syscalls are
//! unsupported and may break between releases — so on Darwin "no dependencies"
//! means "no dependency except the one the platform mandates." We bind the
//! documented `write(2)`/`_exit(2)` entry points as bare C symbols. `libSystem`
//! is linked automatically on Apple targets; the `#[link]` attribute states the
//! dependency explicitly. No libc crate is involved.

use super::Platform;

#[link(name = "System")]
extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn _exit(code: i32) -> !;
    // `errno` is a thread-local on Darwin, reached through this accessor.
    fn __error() -> *mut i32;
}

/// The macOS platform: calls libSystem's documented syscall wrappers.
pub struct Os;

impl Platform for Os {
    fn write(&self, fd: i32, bytes: &[u8]) -> isize {
        // SAFETY: this is `write(2)`. `bytes` is valid for `bytes.len()` reads,
        // matching the `buf`/`count` pair the symbol expects.
        let ret = unsafe { write(fd, bytes.as_ptr(), bytes.len()) };
        if ret >= 0 {
            return ret;
        }
        // libSystem returns -1 and sets `errno`; translate that into a negative
        // errno so the result matches `Platform::write`'s contract — the Linux
        // backend returns `-errno` straight from the kernel.
        // SAFETY: `__error()` returns a valid pointer to this thread's errno.
        let errno = unsafe { *__error() };
        -(errno as isize)
    }

    fn exit(&self, code: i32) -> ! {
        // SAFETY: `_exit(2)` terminates the process and never returns.
        unsafe { _exit(code) }
    }
}
