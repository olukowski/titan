//! Generational handles and the [`Pool`] that issues them.
//!
//! A [`Handle`] names a value in a [`Pool`] by slot index plus the slot's generation. Removing a
//! value bumps the slot's generation, so old handles to that slot stop resolving instead of
//! silently pointing at whatever is stored there next.

use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem;
use core::num::NonZeroU32;

/// A copyable, typed reference to a value in a [`Pool<T>`].
///
/// A handle is only meaningful to the `Pool` that issued it; another pool may resolve it to an
/// unrelated value. A handle is 8 bytes, and so is `Option<Handle<T>>`. `T` only tags the handle, so `Handle<T>`
/// is `Copy`, `Eq`, `Hash`, and `Debug` for every `T`.
pub struct Handle<T> {
    index: u32,
    generation: NonZeroU32,
    // `fn() -> T` keeps the handle `Send`, `Sync`, and covariant whatever `T` is.
    _marker: PhantomData<fn() -> T>,
}

const _: () = assert!(size_of::<Handle<()>>() == 8);
const _: () = assert!(size_of::<Option<Handle<()>>>() == 8);

impl<T> Handle<T> {
    const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index, self.generation)
    }
}

/// Slot storage that hands out [`Handle`]s and reuses freed slots.
///
/// Insert, lookup, and removal are O(1). Vacant slots form an intrusive free list, so the pool
/// allocates only when it grows past its largest size so far.
#[derive(Debug)]
pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    /// Head of the free list threaded through [`Entry::Vacant`] slots.
    free_head: Option<u32>,
    len: usize,
}

#[derive(Debug)]
struct Slot<T> {
    generation: NonZeroU32,
    entry: Entry<T>,
}

#[derive(Debug)]
enum Entry<T> {
    Occupied(T),
    /// A free slot, or a retired one if its generation is `u32::MAX` and it is off the free list.
    Vacant {
        next_free: Option<u32>,
    },
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    /// Creates an empty pool without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// Creates an empty pool with room for `capacity` values before it reallocates.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
        }
    }

    /// Stores `value` and returns its handle.
    ///
    /// # Panics
    ///
    /// Panics if the pool would need more than `u32::MAX + 1` slots.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        let handle = if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            let Entry::Vacant { next_free } = slot.entry else {
                unreachable!("free list points at an occupied slot");
            };
            self.free_head = next_free;
            slot.entry = Entry::Occupied(value);
            Handle::new(index, slot.generation)
        } else {
            let index =
                u32::try_from(self.slots.len()).expect("Pool is full: slot index exceeds u32");
            self.slots.push(Slot {
                generation: NonZeroU32::MIN,
                entry: Entry::Occupied(value),
            });
            Handle::new(index, NonZeroU32::MIN)
        };
        self.len += 1;
        handle
    }

    /// Returns the value for `handle`, or `None` if it has been removed. `handle` must come from
    /// this pool.
    #[must_use]
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index as usize)? {
            Slot {
                generation,
                entry: Entry::Occupied(value),
            } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Returns the value for `handle` mutably, or `None` if it has been removed.
    #[must_use]
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize)? {
            Slot {
                generation,
                entry: Entry::Occupied(value),
            } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Returns whether `handle` still refers to a value.
    #[must_use]
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Removes and returns the value for `handle`, or `None` if it was already removed.
    /// Every copy of `handle` stops resolving.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        self.get(handle)?;
        Some(self.vacate(handle.index))
    }

    /// Returns the number of stored values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the pool stores no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes every value. Every handle issued so far stops resolving; capacity is kept.
    pub fn clear(&mut self) {
        for index in 0..self.slots.len() {
            if matches!(self.slots[index].entry, Entry::Occupied(_)) {
                // Lossless: `insert` never creates more than `u32::MAX + 1` slots.
                self.vacate(index as u32);
            }
        }
    }

    /// Iterates over every stored value with its handle, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match &slot.entry {
                Entry::Occupied(value) => Some((Handle::new(index as u32, slot.generation), value)),
                Entry::Vacant { .. } => None,
            })
    }

    /// Iterates mutably over every stored value with its handle, in slot order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| match &mut slot.entry {
                Entry::Occupied(value) => Some((Handle::new(index as u32, slot.generation), value)),
                Entry::Vacant { .. } => None,
            })
    }

    /// Takes the value out of the occupied slot `index` and bumps the slot's generation.
    fn vacate(&mut self, index: u32) -> T {
        let slot = &mut self.slots[index as usize];
        // A slot whose generation cannot advance is retired: it stays vacant and off the free
        // list forever, because reusing it would let a stale handle match again (ABA).
        let next_free = match slot.generation.checked_add(1) {
            Some(generation) => {
                slot.generation = generation;
                self.free_head.replace(index)
            }
            None => None,
        };
        let Entry::Occupied(value) = mem::replace(&mut slot.entry, Entry::Vacant { next_free })
        else {
            unreachable!("vacate called on a vacant slot");
        };
        self.len -= 1;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    #[test]
    fn stale_handle_does_not_resolve_after_slot_reuse() {
        let mut pool = Pool::new();
        let old = pool.insert("old");
        assert_eq!(pool.remove(old), Some("old"));
        let new = pool.insert("new");
        // Same slot, new generation.
        assert_eq!(old.index, new.index);
        assert_ne!(old, new);
        assert_eq!(pool.get(old), None);
        assert_eq!(pool.get_mut(old), None);
        assert!(!pool.contains(old));
        assert_eq!(pool.remove(old), None);
        assert_eq!(pool.get(new), Some(&"new"));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn double_remove_returns_none_and_keeps_len() {
        let mut pool = Pool::new();
        let a = pool.insert(1);
        let b = pool.insert(2);
        assert_eq!(pool.remove(a), Some(1));
        assert_eq!(pool.remove(a), None);
        assert_eq!(pool.len(), 1);
        // The free list holds `a`'s slot exactly once: two inserts must not share a slot.
        let c = pool.insert(3);
        let d = pool.insert(4);
        assert_ne!(c.index, d.index);
        assert_eq!(
            [pool.get(b), pool.get(c), pool.get(d)],
            [Some(&2), Some(&3), Some(&4)]
        );
    }

    #[test]
    fn clear_invalidates_every_handle_and_drops_values() {
        let mut pool = Pool::new();
        let handles: Vec<_> = (0..4).map(|i| pool.insert(i.to_string())).collect();
        pool.remove(handles[1]);
        pool.clear();
        assert!(pool.is_empty());
        assert!(handles.iter().all(|&h| pool.get(h).is_none()));
        // All four slots are reused before the pool grows.
        let reused: Vec<_> = (0..4).map(|i| pool.insert(i.to_string()).index).collect();
        assert!(reused.iter().all(|&i| i < 4));
        assert_eq!(pool.slots.len(), 4);
        assert!(handles.iter().all(|&h| pool.get(h).is_none()));
    }

    #[test]
    fn iter_skips_vacant_slots() {
        let mut pool: Pool<String> = Pool::with_capacity(4);
        let handles: Vec<_> = ["a", "b", "c", "d"]
            .map(String::from)
            .map(|s| pool.insert(s))
            .to_vec();
        pool.remove(handles[0]);
        pool.remove(handles[2]);
        for (_, value) in pool.iter_mut() {
            value.push('!');
        }
        let seen: Vec<_> = pool.iter().map(|(h, v)| (h, v.as_str())).collect();
        assert_eq!(seen, [(handles[1], "b!"), (handles[3], "d!")]);
    }

    #[test]
    fn exhausted_generation_retires_the_slot() {
        let mut pool = Pool::new();
        let first = pool.insert('a');
        // Fast-forward the slot to its last generation instead of 4 billion remove/insert cycles.
        pool.slots[0].generation = NonZeroU32::MAX;
        let last = Handle::new(first.index, NonZeroU32::MAX);
        assert_eq!(pool.remove(last), Some('a'));
        assert_eq!(pool.get(last), None);
        assert_eq!(pool.free_head, None);
        // The retired slot is never handed out again, so `last` can never match a new value.
        let next = pool.insert('b');
        assert_ne!(next.index, last.index);
        assert_eq!(pool.get(last), None);
        // `clear` must retire exhausted slots too.
        pool.slots[1].generation = NonZeroU32::MAX;
        pool.clear();
        assert_eq!(pool.free_head, None);
        assert_eq!(pool.insert('c').index, 2);
    }
}
