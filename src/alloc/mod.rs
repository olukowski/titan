//! Memory allocation: Titan's own allocator interface and the allocators built
//! on it.
//!
//! Titan never uses the `alloc` crate — a deliberate constitution choice (see
//! `docs/DESIGN.md`): no global allocator, no `Box`/`Vec`/`String` from the
//! standard distribution. Instead an allocator is an explicit value passed to
//! whatever needs to allocate, exactly like the [platform](crate::platform) is.
//! This module defines that interface, [`Allocator`], and the simplest concrete
//! one, [`Bump`]. Owned collections (a growable array, etc.) come next and will
//! be generic over `A: Allocator`.

use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr::NonNull;

/// An explicit source of memory.
///
/// This mirrors the *shape* of the standard library's unstable `Allocator`, but
/// is our own stable-Rust trait: Titan passes allocators by reference rather
/// than relying on a global allocator, so collections store an `&A` and call
/// through this interface.
///
/// # Safety
/// This is an `unsafe trait` because callers rely on its guarantees for memory
/// safety. An implementation must ensure that:
/// - a `Some` return from [`alloc`](Allocator::alloc) points to a block of at
///   least `layout.size()` bytes, aligned to `layout.align()`, that stays valid
///   until it is passed to [`dealloc`](Allocator::dealloc);
/// - distinct live allocations never overlap.
pub unsafe trait Allocator {
    /// Allocate memory fitting `layout`, or return `None` if the request cannot
    /// be satisfied.
    ///
    /// A zero-sized `layout` still returns a non-null, correctly aligned pointer;
    /// it just must not be dereferenced.
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Release a block previously returned by [`alloc`](Allocator::alloc).
    ///
    /// # Safety
    /// `ptr` must have come from `self.alloc(layout)` with the *same* `layout`,
    /// and must not have been deallocated already.
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
}

/// A bump (arena) allocator over a fixed byte region.
///
/// Allocation is a pointer bump: fast, but individual blocks are never
/// reclaimed — [`dealloc`](Allocator::dealloc) is a no-op, and memory is
/// recovered only by [`reset`](Bump::reset)ting or dropping the whole arena.
/// This is both the allocator the collection types are first exercised against
/// and a genuinely useful one for scratch / per-frame data.
///
/// It borrows its backing storage (`&'a mut [u8]`) rather than owning a mapping,
/// so it is OS-agnostic and `no_std`-pure. An allocator backed directly by
/// [`Memory`](crate::platform::Memory) can wrap one of these over a mapped
/// region later.
pub struct Bump<'a> {
    /// Start of the backing region.
    base: NonNull<u8>,
    /// Length of the backing region, in bytes.
    len: usize,
    /// Bytes consumed so far, including alignment padding. `Cell` because
    /// [`alloc`](Allocator::alloc) takes `&self`.
    used: Cell<usize>,
    /// Ties the arena to the exclusive borrow of its backing storage.
    _storage: PhantomData<&'a mut [u8]>,
}

impl<'a> Bump<'a> {
    /// Create an arena that allocates out of `storage`.
    pub fn new(storage: &'a mut [u8]) -> Self {
        Bump {
            // SAFETY: a slice's data pointer is always non-null.
            base: unsafe { NonNull::new_unchecked(storage.as_mut_ptr()) },
            len: storage.len(),
            used: Cell::new(0),
            _storage: PhantomData,
        }
    }

    /// Free every allocation at once by rewinding the arena to empty.
    ///
    /// Takes `&mut self`: this invalidates every pointer previously handed out,
    /// and the exclusive borrow proves none are still held.
    pub fn reset(&mut self) {
        self.used.set(0);
    }

    /// Bytes currently allocated, including alignment padding.
    pub fn used(&self) -> usize {
        self.used.get()
    }
}

// SAFETY: blocks are carved sequentially from a single exclusively-borrowed
// region and so never overlap; a returned pointer stays valid until the arena is
// reset or dropped, both of which require `&mut self` and thus that no block is
// still borrowed.
unsafe impl Allocator for Bump<'_> {
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let used = self.used.get();
        // The cursor as an absolute address — alignment is a property of the
        // address, not of the offset, so we must align this, not `used`.
        // SAFETY: `used <= len`, so this is in-region or one-past-the-end, both
        // valid pointers to form.
        let cursor = unsafe { self.base.as_ptr().add(used) };
        // Padding needed to reach the requested alignment. `align_offset` returns
        // `usize::MAX` only when alignment is impossible, which never happens for
        // the power-of-two alignments `Layout` guarantees — but handle it anyway.
        let offset = cursor.align_offset(layout.align());
        if offset == usize::MAX {
            return None;
        }
        let aligned = used.checked_add(offset)?;
        let end = aligned.checked_add(layout.size())?;
        if end > self.len {
            return None;
        }
        self.used.set(end);
        // SAFETY: `aligned <= end <= len`, so `base + aligned` is in-region (or
        // one-past-the-end for a zero-sized tail allocation); it is aligned by
        // construction and non-null because `base` is.
        Some(unsafe { NonNull::new_unchecked(self.base.as_ptr().add(aligned)) })
    }

    unsafe fn dealloc(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // A bump allocator cannot free individual blocks; reclamation happens in
        // `reset` / drop. Intentionally a no-op.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn aligns_each_block_and_never_overlaps() {
        let mut buf = [0u8; 64];
        let bump = Bump::new(&mut buf);

        let a = bump.alloc(layout(3, 1)).unwrap();
        let b = bump.alloc(layout(8, 8)).unwrap();

        assert_eq!(b.as_ptr() as usize % 8, 0, "second block must be 8-aligned");
        let a_end = a.as_ptr() as usize + 3;
        assert!(b.as_ptr() as usize >= a_end, "blocks must not overlap");
    }

    #[test]
    fn tracks_usage_and_exhausts_without_corrupting_state() {
        let mut buf = [0u8; 16];
        let mut bump = Bump::new(&mut buf);

        assert_eq!(bump.used(), 0);
        bump.alloc(layout(10, 1)).unwrap();
        assert_eq!(bump.used(), 10);

        // Only 6 bytes remain, so a 7-byte request fails and leaves state intact.
        assert!(bump.alloc(layout(7, 1)).is_none());
        assert_eq!(bump.used(), 10);

        bump.reset();
        assert_eq!(bump.used(), 0);
        // The whole region is available again after reset.
        assert!(bump.alloc(layout(16, 1)).is_some());
    }

    #[test]
    fn allocated_memory_is_writable() {
        let mut buf = [0u8; 32];
        let bump = Bump::new(&mut buf);
        let p = bump.alloc(layout(4, 4)).unwrap();
        // SAFETY: `p` points to 4 freshly allocated, writable, 4-aligned bytes.
        unsafe {
            (p.as_ptr() as *mut u32).write(0xDEAD_BEEF);
            assert_eq!((p.as_ptr() as *const u32).read(), 0xDEAD_BEEF);
        }
    }

    #[test]
    fn zero_sized_request_succeeds() {
        let mut buf = [0u8; 8];
        let bump = Bump::new(&mut buf);
        assert!(bump.alloc(layout(0, 1)).is_some());
        assert_eq!(bump.used(), 0, "a zero-sized request consumes nothing");
    }
}
