//! Owned collections built on Titan's explicit [`Allocator`](crate::alloc)
//! interface — the standard library's `Vec`/`String` are off the table (no
//! `alloc` crate; see `docs/DESIGN.md`), so we grow our own.
//!
//! Everything here is generic over `A: Allocator` and borrows the allocator
//! (`&'a A`) rather than owning it, matching the rest of the engine: capabilities
//! are passed in, never global. Allocation is fallible — there is no panic-on-OOM
//! path — so the growing operations return a `Result` and hand the value back
//! rather than aborting.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::slice;

use crate::alloc::Allocator;

/// A contiguous, growable array — Titan's `Vec`.
///
/// Stores its elements in a single block from the borrowed allocator `A`,
/// doubling capacity as needed. Because allocation is fallible,
/// [`push`](Array::push) returns the value back on failure instead of panicking.
///
/// Reclamation follows the allocator: with a [`Bump`](crate::alloc::Bump),
/// `dealloc` is a no-op, so each growth leaves the old block stranded until the
/// arena resets — the expected trade-off for arena allocation.
///
/// # Examples
/// ```
/// use titan::alloc::Bump;
/// use titan::collections::Array;
///
/// let mut storage = [0u8; 256];
/// let bump = Bump::new(&mut storage);
///
/// let mut xs = Array::new_in(&bump);
/// xs.push(1).unwrap();
/// xs.push(2).unwrap();
/// xs.push(3).unwrap();
/// assert_eq!(xs.as_slice(), &[1, 2, 3]);
/// assert_eq!(xs.pop(), Some(3));
/// assert_eq!(xs.len(), 2);
/// ```
pub struct Array<'a, T, A: Allocator> {
    /// Pointer to the allocation. Dangling (but aligned) while `cap == 0` and for
    /// zero-sized `T`, never dereferenced in those states.
    ptr: NonNull<T>,
    /// Number of initialized elements.
    len: usize,
    /// Number of elements the allocation can hold. For zero-sized `T` this is
    /// `usize::MAX` and no allocation ever happens.
    cap: usize,
    /// The borrowed allocator every block comes from and returns to.
    alloc: &'a A,
    /// `Array` owns `T` values, so it acts like it contains a `T` for drop-check
    /// and variance.
    _owns: PhantomData<T>,
}

impl<'a, T, A: Allocator> Array<'a, T, A> {
    /// Create an empty array backed by `alloc`. No allocation happens until the
    /// first [`push`](Array::push).
    pub fn new_in(alloc: &'a A) -> Self {
        Array {
            ptr: NonNull::dangling(),
            len: 0,
            // A zero-sized element needs no storage, so the array is "full" of
            // them from the start and the grow path is never taken.
            cap: if mem::size_of::<T>() == 0 {
                usize::MAX
            } else {
                0
            },
            alloc,
            _owns: PhantomData,
        }
    }

    /// Number of elements currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of elements that fit without reallocating.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Append `value`, growing if necessary.
    ///
    /// Returns `Err(value)` — handing the element back, never dropping it — if
    /// the array is full and the allocator cannot satisfy the growth.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && self.grow().is_err() {
            return Err(value);
        }
        // SAFETY: `len < cap` now, so the slot at `len` is owned by this array
        // and currently uninitialized; writing initializes it.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }

    /// Remove and return the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: the element at the new `len` is initialized (it was within
        // `0..old_len`); reading it moves ownership out and we will not read that
        // slot again until it is overwritten.
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    /// View the initialized elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: for a sized `T`, `ptr` is valid for `len` initialized,
        // contiguous elements. For a zero-sized `T`, `len` may be > 0 over a
        // dangling `ptr`; that is sound because a ZST slice needs only a non-null,
        // aligned pointer (which `NonNull::dangling` is) and no backing storage.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// View the initialized elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, with unique access guaranteed by `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Double the capacity (or jump to a small initial size), moving existing
    /// elements into the new block and freeing the old one. Never called for
    /// zero-sized `T`.
    fn grow(&mut self) -> Result<(), ()> {
        debug_assert!(mem::size_of::<T>() != 0, "zero-sized T must never grow");

        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.checked_mul(2).ok_or(())?
        };
        let new_layout = Layout::array::<T>(new_cap).map_err(|_| ())?;
        let new_ptr = self.alloc.alloc(new_layout).ok_or(())?.cast::<T>();

        if self.cap != 0 {
            // SAFETY: the old block holds `len` initialized elements and does not
            // overlap the freshly allocated one; after the move the old block is
            // released with the exact layout it was allocated with.
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
                let old_layout = Layout::array::<T>(self.cap).map_err(|_| ())?;
                self.alloc.dealloc(self.ptr.cast(), old_layout);
            }
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }
}

impl<T, A: Allocator> Deref for Array<'_, T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, A: Allocator> DerefMut for Array<'_, T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, A: Allocator> Drop for Array<'_, T, A> {
    fn drop(&mut self) {
        // Drop the initialized elements in place...
        // SAFETY: exactly `len` elements are initialized and owned by us. Build
        // the slice pointer directly (no intermediate `&mut [T]`) since the
        // elements are about to be dropped.
        unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len)) };
        // ...then release the backing block. Zero-sized `T` and an unallocated
        // array (`cap == 0`) never owned a block.
        if mem::size_of::<T>() != 0
            && self.cap != 0
            && let Ok(layout) = Layout::array::<T>(self.cap)
        {
            // SAFETY: `ptr`/`layout` are exactly what the last `grow` handed to /
            // received from the allocator.
            unsafe { self.alloc.dealloc(self.ptr.cast(), layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::Bump;
    use core::cell::Cell;

    #[test]
    fn push_pop_and_slice() {
        let mut storage = [0u8; 256];
        let bump = Bump::new(&mut storage);
        let mut xs: Array<i32, _> = Array::new_in(&bump);

        assert!(xs.is_empty());
        for v in [10, 20, 30] {
            xs.push(v).unwrap();
        }
        assert_eq!(xs.len(), 3);
        assert_eq!(xs.as_slice(), &[10, 20, 30]);
        assert_eq!(xs[1], 20); // via Deref<[T]>
        assert_eq!(xs.pop(), Some(30));
        assert_eq!(xs.pop(), Some(20));
        assert_eq!(xs.pop(), Some(10));
        assert_eq!(xs.pop(), None);
        assert!(xs.is_empty());
    }

    #[test]
    fn grows_and_preserves_elements() {
        // Enough room for the doubling chain 4+8+16+32+64+128 blocks of i32.
        let mut storage = [0u8; 4096];
        let bump = Bump::new(&mut storage);
        let mut xs: Array<i32, _> = Array::new_in(&bump);

        for i in 0..100 {
            xs.push(i).unwrap();
        }
        assert_eq!(xs.len(), 100);
        assert!(xs.capacity() >= 100);
        assert!(xs.iter().copied().eq(0..100));
    }

    #[test]
    fn push_returns_value_when_allocator_exhausted() {
        // Far too little space for even the initial 4-element block.
        let mut storage = [0u8; 4];
        let bump = Bump::new(&mut storage);
        let mut xs: Array<u64, _> = Array::new_in(&bump);

        match xs.push(7) {
            Err(v) => assert_eq!(v, 7, "the un-pushed value is handed back intact"),
            Ok(()) => panic!("push should have failed on an exhausted allocator"),
        }
        assert!(xs.is_empty());
    }

    #[test]
    fn failed_growth_preserves_existing_elements() {
        // Room for the first 4-element i32 block (16 bytes + a little alignment
        // slack) but not the 32-byte block the next growth would need.
        let mut storage = [0u8; 40];
        let bump = Bump::new(&mut storage);
        let mut xs: Array<i32, _> = Array::new_in(&bump);

        for v in 0..4 {
            assert!(xs.push(v).is_ok());
        }
        // The 5th push must grow 4 -> 8, which the arena cannot satisfy.
        assert_eq!(xs.push(99), Err(99));
        // The failed growth leaves the array exactly as it was.
        assert_eq!(xs.len(), 4);
        assert_eq!(xs.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn drops_each_element_exactly_once() {
        let drops = Cell::new(0);
        struct Bomb<'c>(&'c Cell<u32>);
        impl Drop for Bomb<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let mut storage = [0u8; 1024];
        let bump = Bump::new(&mut storage);
        {
            let mut xs: Array<Bomb, _> = Array::new_in(&bump);
            for _ in 0..5 {
                // `unwrap` would require `Bomb: Debug` for the `Err` arm; assert
                // success directly instead.
                assert!(xs.push(Bomb(&drops)).is_ok());
            }
            // Popping one drops it immediately; the rest drop with the array.
            drop(xs.pop());
            assert_eq!(drops.get(), 1);
        }
        assert_eq!(drops.get(), 5, "every element dropped exactly once");
    }

    #[test]
    fn supports_zero_sized_elements() {
        let mut storage = [0u8; 8];
        let bump = Bump::new(&mut storage);
        let mut xs: Array<(), _> = Array::new_in(&bump);

        for _ in 0..1000 {
            xs.push(()).unwrap(); // never allocates
        }
        assert_eq!(xs.len(), 1000);
        assert_eq!(xs.pop(), Some(()));
        assert_eq!(xs.len(), 999);
    }
}
