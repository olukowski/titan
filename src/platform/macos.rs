//! macOS backend: the thinnest documented `libSystem` symbols (`man 2`).
//!
//! Apple does not provide a stable syscall ABI — direct `svc` syscalls are
//! unsupported and may break between releases — so on Darwin "no dependencies"
//! means "no dependency except the one the platform mandates." We bind the
//! documented `write(2)`/`_exit(2)` entry points as bare C symbols. `libSystem`
//! is linked automatically on Apple targets; the `#[link]` attribute states the
//! dependency explicitly. No libc crate is involved.

use super::{Console, Memory, Process};

#[link(name = "System")]
unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn _exit(code: i32) -> !;
    // `errno` is a thread-local on Darwin, reached through this accessor.
    fn __error() -> *mut i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

// `mmap` constants. `MAP_ANON` is `0x1000` on Darwin (it differs from Linux's
// `MAP_ANONYMOUS`, which is why these live per-backend).
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const MAP_PRIVATE: i32 = 0x2;
const MAP_ANON: i32 = 0x1000;
// `mmap` returns this sentinel — `(void *) -1` — on failure.
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

/// The macOS platform: calls libSystem's documented syscall wrappers.
pub struct Os;

impl Console for Os {
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
}

impl Process for Os {
    fn exit(&self, code: i32) -> ! {
        // SAFETY: `_exit(2)` terminates the process and never returns.
        unsafe { _exit(code) }
    }
}

impl Memory for Os {
    fn map(&self, len: usize) -> *mut u8 {
        // SAFETY: a private anonymous `mmap`; `addr = null` lets the kernel pick
        // the address, and `MAP_ANON` requires `fd = -1`, `offset = 0`. Only the
        // scalar arguments are read.
        let ret = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        // Normalize the `MAP_FAILED` sentinel to null so the trait's "null on
        // failure" contract holds on both backends.
        if ret == MAP_FAILED {
            return core::ptr::null_mut();
        }
        ret
    }

    unsafe fn unmap(&self, ptr: *mut u8, len: usize) {
        // SAFETY: the trait contract guarantees `ptr`/`len` describe a region from
        // a prior `map`, which is what `munmap` expects. The result is ignored — a
        // correct unmap cannot fail.
        let _ = unsafe { munmap(ptr, len) };
    }
}
