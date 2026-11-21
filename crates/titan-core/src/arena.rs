//! A fixed-capacity bump allocator for short-lived, frame-scoped data.

use alloc::boxed::Box;
use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::fmt;
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};

/// A bump allocator over one buffer that is allocated up front and never grows.
///
/// Allocating moves an offset forward, so it is a few instructions and never touches the global
/// allocator. Use `&BumpArena` as the allocator of standard collections:
///
/// ```
/// use titan_core::BumpArena;
///
/// let mut arena = BumpArena::new(1024);
/// let mut scratch = Vec::new_in(&arena);
/// scratch.extend([1, 2, 3]);
/// assert_eq!(scratch.iter().sum::<i32>(), 6);
/// drop(scratch);
/// arena.reset(); // Needs `&mut`, so no allocation can outlive it.
/// ```
///
/// Freeing a block reclaims its memory only if it is the most recent one; everything else is
/// reclaimed at once by [`reset`](Self::reset). Growing or shrinking the most recent block
/// happens in place.
pub struct BumpArena {
    /// Start of the buffer, which this arena owns (from `Box::leak`). Every block is derived from
    /// this pointer, so blocks carry provenance for the buffer and never alias a reference.
    base: NonNull<u8>,
    capacity: usize,
    /// Bytes handed out so far: live blocks lie within `base[..offset]`.
    offset: Cell<usize>,
}

impl BumpArena {
    /// Creates an arena that owns a new buffer of `capacity` bytes.
    ///
    /// # Panics
    ///
    /// Aborts or panics like [`Box`] if the global allocator cannot provide the buffer.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let buffer: &mut [MaybeUninit<u8>] = Box::leak(Box::new_uninit_slice(capacity));
        Self {
            base: NonNull::from(buffer).cast(),
            capacity,
            offset: Cell::new(0),
        }
    }

    /// Frees every allocation at once, making the full capacity available again.
    ///
    /// Taking `&mut self` means no collection can still be borrowing the arena.
    pub fn reset(&mut self) {
        *self.offset.get_mut() = 0;
    }

    /// Returns the number of bytes allocated, including alignment padding.
    #[must_use]
    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// Returns the size of the buffer in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of unallocated bytes. Alignment padding may make a request of this
    /// size fail.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity - self.offset.get()
    }

    /// Returns the offset of the first byte at or after `offset` whose address is a multiple of
    /// `align`. The buffer is only byte-aligned, so this aligns the address, not the offset.
    fn align_up(&self, offset: usize, align: usize) -> Option<usize> {
        let base = self.base.addr().get();
        // `base + offset` is at most the end of the buffer, so it cannot overflow.
        Some((base + offset).checked_next_multiple_of(align)? - base)
    }

    /// Returns the offset of the block at `ptr` if it is the most recent one, ending at the
    /// current offset.
    fn tip_start(&self, ptr: NonNull<u8>, size: usize) -> Option<usize> {
        // Wrapping keeps this a pure comparison for dangling zero-size pointers; an address that
        // happens to match yields a zero-size block exactly at the tip, which is harmless.
        let start = ptr.addr().get().wrapping_sub(self.base.addr().get());
        (start.wrapping_add(size) == self.offset.get()).then_some(start)
    }

    /// Returns the block `base[start..start + len]`.
    ///
    /// # Safety
    ///
    /// `start + len` must not exceed `capacity`.
    unsafe fn block(&self, start: usize, len: usize) -> NonNull<[u8]> {
        debug_assert!(
            start
                .checked_add(len)
                .is_some_and(|end| end <= self.capacity)
        );
        // SAFETY: the caller guarantees `start <= capacity`, so the result is within the buffer
        // or one past its end.
        let ptr = unsafe { self.base.add(start) };
        NonNull::slice_from_raw_parts(ptr, len)
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        let buffer = ptr::slice_from_raw_parts_mut(
            self.base.as_ptr().cast::<MaybeUninit<u8>>(),
            self.capacity,
        );
        // SAFETY: `base` and `capacity` describe the slice `new` leaked from a `Box`, and the
        // arena is its only owner. Every block borrowed the arena, so none is still in use.
        drop(unsafe { Box::from_raw(buffer) });
    }
}

impl fmt::Debug for BumpArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BumpArena")
            .field("used", &self.used())
            .field("capacity", &self.capacity)
            .finish()
    }
}

// SAFETY: blocks are disjoint ranges of a buffer that lives as long as the `BumpArena`. Blocks
// stay valid until deallocated: the only operations that recycle memory are `reset` (`&mut self`)
// and drop, and both need every `&BumpArena`, and so every collection using one, to be gone.
// Copies of `&BumpArena` share one arena, so any copy may grow, shrink, or free any block.
unsafe impl Allocator for &BumpArena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }
        let start = self
            .align_up(self.offset.get(), layout.align())
            .ok_or(AllocError)?;
        let end = start.checked_add(layout.size()).ok_or(AllocError)?;
        if end > self.capacity {
            return Err(AllocError);
        }
        self.offset.set(end);
        // SAFETY: `start + size == end <= capacity`.
        Ok(unsafe { self.block(start, layout.size()) })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if let Some(start) = self.tip_start(ptr, layout.size()) {
            self.offset.set(start);
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if let Some(start) = self.tip_start(ptr, old_layout.size())
            && ptr.addr().get().is_multiple_of(new_layout.align())
            && let Some(end) = start.checked_add(new_layout.size())
            && end <= self.capacity
        {
            self.offset.set(end);
            // SAFETY: `start + new size == end <= capacity`.
            return Ok(unsafe { self.block(start, new_layout.size()) });
        }
        let new = self.allocate(new_layout)?;
        // SAFETY: the caller guarantees `ptr` is a live block of `old_layout.size()` bytes, and
        // `new` is at least that large (`grow` requires `new_layout.size() >= old_layout.size()`).
        // `new` starts at or after the old offset, past every live block, so they do not overlap.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new.cast::<u8>().as_ptr(), old_layout.size())
        };
        Ok(new)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !ptr.addr().get().is_multiple_of(new_layout.align()) {
            let new = self.allocate(new_layout)?;
            // SAFETY: as in `grow`, but copying the `new_layout.size() <= old_layout.size()`
            // bytes that fit.
            unsafe {
                ptr::copy_nonoverlapping(ptr.as_ptr(), new.cast::<u8>().as_ptr(), new_layout.size())
            };
            return Ok(new);
        }
        if let Some(start) = self.tip_start(ptr, old_layout.size()) {
            self.offset.set(start + new_layout.size());
        }
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn offset_of(arena: &BumpArena, block: NonNull<[u8]>) -> usize {
        block.cast::<u8>().addr().get() - arena.base.addr().get()
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn vec_grows_in_place_at_the_tip_and_by_copy_elsewhere() {
        let arena = BumpArena::new(4096);
        let mut a = Vec::new_in(&arena);
        a.extend(0..100u32);
        // Every reallocation hit the tip, so the vector never left its first block.
        assert_eq!(arena.used(), a.capacity() * 4);
        // Interleaving two vectors forces each regrowth to allocate and copy.
        let mut b = Vec::new_in(&arena);
        for i in 0..100u32 {
            a.push(1000 + i);
            b.push(i);
        }
        assert!(a.iter().copied().eq((0..100).chain(1000..1100)));
        assert!(b.iter().copied().eq(0..100));
        assert!(arena.used() <= arena.capacity());
    }

    #[test]
    fn box_frees_only_the_most_recent_block() {
        #[repr(align(256))]
        struct Aligned([u8; 3]);

        let arena = BumpArena::new(1024);
        let bytes = Box::new_in([1u8, 2, 3], &arena);
        let byte = Box::new_in(4u8, &arena);
        assert_eq!(arena.used(), 4);
        drop(byte);
        assert_eq!(arena.used(), 3);
        let aligned = Box::new_in(Aligned([5, 6, 7]), &arena);
        assert!((&raw const *aligned).addr().is_multiple_of(256));
        assert_eq!((*bytes, aligned.0), ([1, 2, 3], [5, 6, 7]));
        let used = arena.used();
        // `bytes` is not the most recent block, so freeing it reclaims nothing.
        drop(bytes);
        assert_eq!(arena.used(), used);
        // Freeing `aligned` reclaims its bytes but not the padding in front of it.
        drop(aligned);
        assert!((3..3 + 256).contains(&arena.used()));
    }

    #[test]
    fn zero_sized_allocations_take_no_space() {
        let empty = BumpArena::new(0);
        let block = (&empty).allocate(layout(0, 64)).unwrap();
        assert!(block.cast::<u8>().addr().get().is_multiple_of(64));
        assert_eq!((&empty).allocate(layout(1, 1)), Err(AllocError));
        let arena = BumpArena::new(16);
        let mut units = Vec::new_in(&arena);
        units.extend(core::iter::repeat_n((), 1000));
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn over_aligned_blocks_align_the_address() {
        let arena = BumpArena::new(2048);
        (&arena).allocate(layout(1, 1)).unwrap();
        for align in [64, 256, 1024] {
            let block = (&arena).allocate(layout(3, align)).unwrap();
            assert!(block.cast::<u8>().addr().get().is_multiple_of(align));
            assert!(offset_of(&arena, block) + 3 <= arena.capacity());
        }
    }

    #[test]
    fn out_of_memory_fails_without_consuming_space() {
        let arena = BumpArena::new(64);
        (&arena).allocate(layout(48, 1)).unwrap();
        assert_eq!((&arena).allocate(layout(17, 1)), Err(AllocError));
        assert_eq!(
            (&arena).allocate(layout(1, 1usize << (usize::BITS - 2))),
            Err(AllocError)
        );
        assert_eq!(
            (&arena).allocate(layout(isize::MAX as usize - 8, 1)),
            Err(AllocError)
        );
        assert_eq!(arena.used(), 48);
        let last = (&arena).allocate(layout(16, 1)).unwrap();
        assert_eq!(offset_of(&arena, last), 48);
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn reset_makes_the_full_capacity_available() {
        let mut arena = BumpArena::new(100);
        (&arena).allocate(layout(60, 1)).unwrap();
        assert_eq!((&arena).allocate(layout(60, 1)), Err(AllocError));
        arena.reset();
        assert_eq!(arena.used(), 0);
        let block = (&arena).allocate(layout(100, 1)).unwrap();
        assert_eq!((offset_of(&arena, block), block.len()), (0, 100));
    }

    #[test]
    fn grow_and_shrink_preserve_contents() {
        let arena = BumpArena::new(256);
        let alloc = &arena;
        let old = layout(8, 8);
        let first = alloc.allocate(old).unwrap().cast::<u8>();
        // SAFETY: `first` is a fresh 8-byte block.
        unsafe { first.copy_from_nonoverlapping(NonNull::from(b"abcdefgh").cast(), 8) };

        // At the tip: in place.
        // SAFETY: `first` is live with layout `old`, and 32 >= 8.
        let grown = unsafe { alloc.grow(first, old, layout(32, 8)) }
            .unwrap()
            .cast::<u8>();
        assert_eq!((grown, arena.used()), (first, 32));
        // SAFETY: `grown` is live with size 32, and 4 <= 32.
        let shrunk = unsafe { alloc.shrink(grown, layout(32, 8), layout(4, 8)) }.unwrap();
        assert_eq!((shrunk.cast::<u8>(), arena.used()), (first, 4));

        // Not at the tip: grow copies, shrink stays put.
        let blocker = alloc.allocate(layout(4, 1)).unwrap();
        // SAFETY: `shrunk` is live with layout (4, 8), and 64 >= 4.
        let moved = unsafe { alloc.grow(shrunk.cast(), layout(4, 8), layout(64, 8)) }.unwrap();
        assert_ne!(moved.cast::<u8>(), first);
        // SAFETY: the first 4 bytes of `moved` were copied from `shrunk`.
        assert_eq!(unsafe { moved.cast::<[u8; 4]>().read() }, *b"abcd");
        let used = arena.used();
        // SAFETY: `blocker` is live with layout (4, 1), and 2 <= 4.
        let kept = unsafe { alloc.shrink(blocker.cast(), layout(4, 1), layout(2, 1)) }.unwrap();
        assert_eq!(
            (kept.cast::<u8>(), arena.used()),
            (blocker.cast::<u8>(), used)
        );

        // A stricter alignment that `ptr` does not meet forces a copy, even when shrinking.
        // SAFETY: `moved` is live with layout (64, 8), and 8 <= 64.
        let realigned = unsafe { alloc.shrink(moved.cast(), layout(64, 8), layout(2, 128)) };
        let realigned = realigned.unwrap().cast::<u8>();
        assert!(realigned.addr().get().is_multiple_of(128));
        // SAFETY: the first 2 bytes of `realigned` were copied from `moved`.
        assert_eq!(unsafe { realigned.cast::<[u8; 2]>().read() }, *b"ab");

        // Growth that does not fit fails and leaves the block and the offset alone.
        let used = arena.used();
        // SAFETY: `realigned` is live with layout (2, 128), and 1024 >= 2.
        let too_big = unsafe { alloc.grow(realigned, layout(2, 128), layout(1024, 128)) };
        assert_eq!((too_big, arena.used()), (Err(AllocError), used));
    }
}
