//! Linux backend: raw `aarch64` syscalls via inline `asm!`, no libc.
//!
//! Linux exposes a stable, documented syscall ABI, so we target it directly:
//! arguments in `x0..`, the syscall number in `x8`, then `svc #0`; the result
//! comes back in `x0`. This is the original "inline to a handful of
//! instructions, no dependencies" goal — on stable Rust.

use super::Platform;

// `aarch64` Linux syscall numbers.
const SYS_WRITE: usize = 64;
const SYS_EXIT_GROUP: usize = 94;

/// The Linux platform: talks to the kernel directly.
pub struct Os;

impl Platform for Os {
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
