//! Linux backend: raw `aarch64` syscalls via inline `asm!`, no libc.
//!
//! Linux exposes a stable, documented syscall ABI, so we target it directly:
//! arguments in `x0..`, the syscall number in `x8`, then `svc #0`; the result
//! comes back in `x0`. This is the original "inline to a handful of
//! instructions, no dependencies" goal — on stable Rust.

use super::{Console, Memory, Process};

// `aarch64` Linux syscall numbers.
const SYS_MUNMAP: usize = 215;
const SYS_MMAP: usize = 222;
const SYS_WRITE: usize = 64;
const SYS_EXIT_GROUP: usize = 94;

// `mmap` arguments for a fresh private anonymous region.
const PROT_READ: usize = 0x1;
const PROT_WRITE: usize = 0x2;
const MAP_PRIVATE: usize = 0x2;
const MAP_ANONYMOUS: usize = 0x20;

/// The Linux platform: talks to the kernel directly.
pub struct Os;

impl Console for Os {
    fn write(&self, fd: i32, bytes: &[u8]) -> isize {
        let ret: isize;
        // SAFETY: this is a `write(2)` syscall. `bytes` is a valid, readable
        // slice of `bytes.len()` bytes; the kernel only reads from it and
        // writes nothing back through the pointer.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") SYS_WRITE,
                inout("x0") fd as usize => ret,
                in("x1") bytes.as_ptr(),
                in("x2") bytes.len(),
                options(nostack, preserves_flags),
            );
        }
        ret
    }
}

impl Process for Os {
    fn exit(&self, code: i32) -> ! {
        // SAFETY: `exit_group(2)` terminates the process and never returns, so
        // the `noreturn` option (and the unreachable fall-through) are correct.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") SYS_EXIT_GROUP,
                in("x0") code as usize,
                options(noreturn, nostack),
            );
        }
    }
}

impl Memory for Os {
    fn map(&self, len: usize) -> *mut u8 {
        let ret: isize;
        // SAFETY: an anonymous `mmap(2)` with `addr = 0` lets the kernel choose
        // the address; `fd = -1` and `offset = 0` are required for `MAP_ANONYMOUS`.
        // The kernel reads only these scalar arguments and returns either a fresh
        // mapping or a negative errno; nothing is read or written through a
        // caller pointer.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") SYS_MMAP,
                inout("x0") 0usize => ret,           // addr: let the kernel choose
                in("x1") len,                         // length
                in("x2") PROT_READ | PROT_WRITE,      // prot
                in("x3") MAP_PRIVATE | MAP_ANONYMOUS, // flags
                in("x4") -1isize as usize,            // fd
                in("x5") 0usize,                      // offset
                options(nostack, preserves_flags),
            );
        }
        // On error `mmap` returns `-errno` in the range `[-4095, -1]`; any other
        // value is a valid address. (`MAP_FAILED` on Linux is `-1`, covered here.)
        if (-4095..0).contains(&ret) {
            return core::ptr::null_mut();
        }
        ret as *mut u8
    }

    unsafe fn unmap(&self, ptr: *mut u8, len: usize) {
        // SAFETY: per the trait contract `ptr`/`len` name a region from a prior
        // `map`, which is exactly what `munmap(2)` expects. Its result is ignored:
        // a correct unmap cannot fail, and there is no recovery if it somehow did.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") SYS_MUNMAP,
                in("x0") ptr,
                in("x1") len,
                options(nostack, preserves_flags),
            );
        }
    }
}
