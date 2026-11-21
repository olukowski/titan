//! GPU resource handles, the pool backends store them in, and the descriptors that create them.

use alloc::vec::Vec;
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use titan_core::{Handle, Pool};

mod private {
    /// Keeps [`Resource`](super::Resource) closed: only this crate's tags can index a
    /// [`ResourcePool`](super::ResourcePool), so a tag always means a slot in `values`.
    pub trait Sealed {
        fn new(slot: u32) -> Self;
        fn slot(&self) -> u32;
    }
}

/// A kind of GPU resource that a [`Handle`] can name: [`Buffer`], [`Texture`], [`Shader`], or
/// [`Pipeline`]. The trait is sealed.
pub trait Resource: private::Sealed {}

macro_rules! resource_tags {
    ($($(#[$meta:meta])* $Tag:ident),+ $(,)?) => {$(
        $(#[$meta])*
        ///
        /// Backends create these inside a [`ResourcePool`]; user code only passes the handle
        /// around.
        #[derive(Debug)]
        pub struct $Tag {
            slot: u32,
        }

        impl private::Sealed for $Tag {
            fn new(slot: u32) -> Self {
                Self { slot }
            }

            fn slot(&self) -> u32 {
                self.slot
            }
        }

        impl Resource for $Tag {}
    )+};
}

resource_tags! {
    /// Tags a [`Handle`] to a GPU buffer.
    Buffer,
    /// Tags a [`Handle`] to a GPU texture and its sampler.
    Texture,
    /// Tags a [`Handle`] to a compiled shader module.
    Shader,
    /// Tags a [`Handle`] to a render pipeline.
    Pipeline,
}

/// Backend storage for one resource kind, addressed by generational [`Handle<R>`]s.
///
/// [`Handle`] cannot be ordered or decomposed outside `titan-core`, so a backend cannot key its
/// own map by it. This pool issues the handles itself and stores the backend's value `V` (a
/// native buffer, a recorded texture, ...) beside them. Lookup is O(1), and a handle stops
/// resolving once its value is removed, even after the slot is reused.
pub struct ResourcePool<R: Resource, V> {
    /// Issues handles; each tag records which `values` slot its handle owns.
    handles: Pool<R>,
    /// Indexed by tag slot. The handle is kept beside the value so iteration needs no lookups.
    values: Vec<Option<(Handle<R>, V)>>,
    free_slots: Vec<u32>,
}

impl<R: Resource, V> ResourcePool<R, V> {
    /// Creates an empty pool without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            handles: Pool::new(),
            values: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    /// Creates an empty pool with room for `capacity` values before it reallocates.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            handles: Pool::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
        }
    }

    /// Stores `value` and returns its handle.
    ///
    /// # Panics
    ///
    /// Panics if the pool would need more than `u32::MAX + 1` slots.
    pub fn insert(&mut self, value: V) -> Handle<R> {
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let slot = u32::try_from(self.values.len())
                .expect("ResourcePool is full: slot index exceeds u32");
            self.values.push(None);
            slot
        });
        let handle = self.handles.insert(R::new(slot));
        self.values[slot as usize] = Some((handle, value));
        handle
    }

    /// Returns the value for `handle`, or `None` if it has been removed.
    #[must_use]
    pub fn get(&self, handle: Handle<R>) -> Option<&V> {
        let slot = self.handles.get(handle)?.slot();
        self.values[slot as usize].as_ref().map(|(_, value)| value)
    }

    /// Returns the value for `handle` mutably, or `None` if it has been removed.
    #[must_use]
    pub fn get_mut(&mut self, handle: Handle<R>) -> Option<&mut V> {
        let slot = self.handles.get(handle)?.slot();
        self.values[slot as usize].as_mut().map(|(_, value)| value)
    }

    /// Returns whether `handle` still refers to a value.
    #[must_use]
    pub fn contains(&self, handle: Handle<R>) -> bool {
        self.handles.contains(handle)
    }

    /// Removes and returns the value for `handle`, or `None` if it was already removed.
    pub fn remove(&mut self, handle: Handle<R>) -> Option<V> {
        let slot = self.handles.remove(handle)?.slot();
        self.free_slots.push(slot);
        self.values[slot as usize].take().map(|(_, value)| value)
    }

    /// Returns the number of stored values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Returns whether the pool stores no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Removes every value. Every handle issued so far stops resolving; capacity is kept.
    pub fn clear(&mut self) {
        self.handles.clear();
        self.values.clear();
        self.free_slots.clear();
    }

    /// Iterates over every stored value with its handle.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<R>, &V)> {
        self.values
            .iter()
            .filter_map(|entry| entry.as_ref().map(|(handle, value)| (*handle, value)))
    }

    /// Iterates mutably over every stored value with its handle.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<R>, &mut V)> {
        self.values
            .iter_mut()
            .filter_map(|entry| entry.as_mut().map(|(handle, value)| (*handle, value)))
    }
}

impl<R: Resource, V> Default for ResourcePool<R, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Resource, V: fmt::Debug> fmt::Debug for ResourcePool<R, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Defines a `u32` bit set with named flags, `const` set operations, the bitwise operators, and
/// a `Debug` that lists flag names.
macro_rules! usage_flags {
    (
        $(#[$meta:meta])*
        $Flags:ident { $($(#[$flag_meta:meta])* $FLAG:ident = $bit:literal;)+ }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $Flags(u32);

        impl $Flags {
            /// No flags.
            pub const EMPTY: Self = Self(0);
            $($(#[$flag_meta])* pub const $FLAG: Self = Self(1 << $bit);)+

            /// Returns the flags set in `self` or `other`.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Returns whether every flag in `other` is also set in `self`.
            #[must_use]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Returns whether no flag is set.
            #[must_use]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Returns the raw bits.
            #[must_use]
            pub const fn bits(self) -> u32 {
                self.0
            }
        }

        impl BitOr for $Flags {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl BitAnd for $Flags {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl BitOrAssign for $Flags {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl BitAndAssign for $Flags {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl fmt::Debug for $Flags {
            /// Prints the set flags, such as `BufferUsage(VERTEX | COPY_DST)`.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(concat!(stringify!($Flags), "("))?;
                let mut names = [$((Self::$FLAG, stringify!($FLAG))),+]
                    .into_iter()
                    .filter(|&(flag, _)| self.contains(flag))
                    .map(|(_, name)| name);
                f.write_str(names.next().unwrap_or("EMPTY"))?;
                for name in names {
                    write!(f, " | {name}")?;
                }
                f.write_str(")")
            }
        }
    };
}

usage_flags! {
    /// How a buffer may be used. Combine flags with `|`.
    BufferUsage {
        /// Bound with `SetVertexBuffer`.
        VERTEX = 0;
        /// Bound with `SetIndexBuffer`.
        INDEX = 1;
        /// Bound with `SetUniformBuffer`.
        UNIFORM = 2;
        /// Bound with `SetStorageBuffer`.
        STORAGE = 3;
        /// The source of a copy.
        COPY_SRC = 4;
        /// The destination of a copy, including CPU writes.
        COPY_DST = 5;
    }
}

usage_flags! {
    /// How a texture may be used. Combine flags with `|`.
    TextureUsage {
        /// Sampled in shaders, bound with `SetTexture`.
        SAMPLED = 0;
        /// Rendered to as a pass's color or depth attachment.
        RENDER_ATTACHMENT = 1;
        /// The source of a copy.
        COPY_SRC = 2;
        /// The destination of a copy, including CPU writes.
        COPY_DST = 3;
        /// Read and written as a storage texture.
        STORAGE = 4;
    }
}

/// Describes a buffer to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDesc<'a> {
    /// A name for debuggers and error messages.
    pub label: Option<&'a str>,
    /// The size in bytes; must be nonzero.
    pub size: u64,
    /// How the buffer may be used; must not be empty.
    pub usage: BufferUsage,
}

/// The pixel format of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// 8-bit RGBA, linear.
    Rgba8Unorm,
    /// 8-bit RGBA, sRGB-encoded color with linear alpha.
    Rgba8UnormSrgb,
    /// 8-bit BGRA, linear.
    Bgra8Unorm,
    /// 8-bit BGRA, sRGB-encoded color with linear alpha. The usual swapchain format.
    Bgra8UnormSrgb,
    /// One 8-bit linear channel.
    R8Unorm,
    /// 32-bit float depth.
    Depth32Float,
    /// At least 24-bit depth plus 8-bit stencil, in a backend-specific layout.
    Depth24PlusStencil8,
}

impl TextureFormat {
    /// Returns whether this is a depth (or depth-stencil) format.
    #[must_use]
    pub const fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float | Self::Depth24PlusStencil8)
    }

    /// Returns the size of one texel in bytes, or `None` if the layout is backend-specific, in
    /// which case the CPU cannot upload to it.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> Option<u32> {
        match self {
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Depth32Float => Some(4),
            Self::R8Unorm => Some(1),
            Self::Depth24PlusStencil8 => None,
        }
    }
}

/// How a sampler picks texels between texel centers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// The nearest texel: crisp pixel art.
    Nearest,
    /// A blend of the nearest texels: smooth scaling.
    #[default]
    Linear,
}

/// How a sampler handles coordinates outside `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Use the edge texel.
    #[default]
    ClampToEdge,
    /// Tile the texture.
    Repeat,
    /// Tile the texture, mirroring every other tile.
    MirrorRepeat,
}

/// How a texture is sampled. Each texture carries its own sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerDesc {
    /// The filter when the texture is drawn smaller than its size.
    pub min_filter: FilterMode,
    /// The filter when the texture is drawn larger than its size.
    pub mag_filter: FilterMode,
    /// Horizontal addressing.
    pub address_u: AddressMode,
    /// Vertical addressing.
    pub address_v: AddressMode,
}

impl SamplerDesc {
    /// Linear filtering, clamped to the edge.
    pub const LINEAR_CLAMP: Self = Self::uniform(FilterMode::Linear, AddressMode::ClampToEdge);
    /// Nearest filtering, clamped to the edge.
    pub const NEAREST_CLAMP: Self = Self::uniform(FilterMode::Nearest, AddressMode::ClampToEdge);
    /// Linear filtering, tiled.
    pub const LINEAR_REPEAT: Self = Self::uniform(FilterMode::Linear, AddressMode::Repeat);
    /// Nearest filtering, tiled.
    pub const NEAREST_REPEAT: Self = Self::uniform(FilterMode::Nearest, AddressMode::Repeat);

    const fn uniform(filter: FilterMode, address: AddressMode) -> Self {
        Self {
            min_filter: filter,
            mag_filter: filter,
            address_u: address,
            address_v: address,
        }
    }
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self::LINEAR_CLAMP
    }
}

/// Describes a 2D texture to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDesc<'a> {
    /// A name for debuggers and error messages.
    pub label: Option<&'a str>,
    /// Width in texels; must be nonzero.
    pub width: u32,
    /// Height in texels; must be nonzero.
    pub height: u32,
    /// The number of mip levels, from 1 to `floor(log2(max(width, height))) + 1`.
    pub mip_level_count: u32,
    /// The texel format.
    pub format: TextureFormat,
    /// How the texture may be used.
    pub usage: TextureUsage,
    /// How the texture is sampled when bound with `SetTexture`.
    pub sampler: SamplerDesc,
}

impl TextureDesc<'_> {
    /// Describes an unlabeled sRGB RGBA texture with one mip level that can be uploaded to and
    /// sampled with linear filtering: the usual sprite or image texture.
    #[must_use]
    pub const fn rgba8_2d(width: u32, height: u32) -> Self {
        Self {
            label: None,
            width,
            height,
            mip_level_count: 1,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
            sampler: SamplerDesc::LINEAR_CLAMP,
        }
    }
}

/// The number of vertex buffer slots. A pipeline has at most this many vertex layouts, and
/// [`RenderCmd::SetVertexBuffer`](crate::RenderCmd::SetVertexBuffer) slots are below it.
pub const MAX_VERTEX_BUFFERS: usize = 8;

/// The MSL `[[buffer(n)]]` index of uniform and storage buffer binding 0. Vertex buffers take
/// the indices below it, so the two never collide.
// Lossless: MAX_VERTEX_BUFFERS is 8.
pub const METAL_BUFFER_BINDING_OFFSET: u32 = MAX_VERTEX_BUFFERS as u32;

/// The source of a shader in every language a backend may need. Each backend compiles the one
/// it understands: Metal reads `msl`, WebGPU reads `wgsl`, and WebGL reads `glsl`.
///
/// # Binding convention
///
/// Every source declares the resources a [`RenderCmd`](crate::RenderCmd) binds at these fixed
/// places, so one command stream drives every backend and no two resources collide:
///
/// | Command | WGSL | MSL |
/// |---|---|---|
/// | `SetVertexBuffer { slot, .. }` | vertex buffer `slot`, feeding its layout's `@location`s | `[[buffer(slot)]]`, feeding its layout's `[[attribute(n)]]`s |
/// | `SetUniformBuffer` / `SetStorageBuffer { binding, .. }` | `@group(0) @binding(binding)`, vertex and fragment | `[[buffer(8 + binding)]]` ([`METAL_BUFFER_BINDING_OFFSET`]), vertex and fragment |
/// | `SetTexture { binding, .. }` | texture at `@group(1) @binding(2 * binding)`, its sampler at `@group(1) @binding(2 * binding + 1)`, fragment | `[[texture(binding)]]` and `[[sampler(binding)]]`, fragment |
///
/// Vertex buffer slots are below [`MAX_VERTEX_BUFFERS`]. Uniform and storage buffers share one
/// binding space, so a binding names at most one buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderSource<'a> {
    /// A name for debuggers and error messages.
    pub label: Option<&'a str>,
    /// Metal Shading Language source.
    pub msl: &'a str,
    /// WGSL source.
    pub wgsl: &'a str,
    /// GLSL ES source, for backends without WGSL.
    pub glsl: Option<&'a str>,
    /// The vertex entry point's name in every source.
    pub vertex_entry: &'a str,
    /// The fragment entry point's name in every source.
    pub fragment_entry: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;

    #[test]
    fn stale_handle_does_not_resolve_after_slot_reuse() {
        let mut pool: ResourcePool<Buffer, &str> = ResourcePool::new();
        let old = pool.insert("old");
        assert_eq!(pool.remove(old), Some("old"));
        let new = pool.insert("new");
        assert_ne!(old, new);
        assert_eq!(pool.get(old), None);
        assert_eq!(pool.get_mut(old), None);
        assert!(!pool.contains(old));
        assert_eq!(pool.remove(old), None);
        assert_eq!(pool.get(new), Some(&"new"));
        // The freed value slot was reused rather than grown.
        assert_eq!(pool.values.len(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn double_remove_frees_the_slot_once() {
        let mut pool: ResourcePool<Texture, i32> = ResourcePool::with_capacity(2);
        let a = pool.insert(1);
        let b = pool.insert(2);
        assert_eq!(pool.remove(a), Some(1));
        assert_eq!(pool.remove(a), None);
        assert_eq!(pool.len(), 1);
        // A slot freed twice would hand the same value slot to both inserts.
        let c = pool.insert(3);
        let d = pool.insert(4);
        assert_eq!(
            [pool.get(b), pool.get(c), pool.get(d)],
            [Some(&2), Some(&3), Some(&4)]
        );
        *pool.get_mut(c).unwrap() = 30;
        assert_eq!(pool.get(c), Some(&30));
        assert_eq!(pool.get(d), Some(&4));
    }

    #[test]
    fn clear_invalidates_every_handle() {
        let mut pool: ResourcePool<Shader, String> = ResourcePool::default();
        let handles: Vec<_> = (0..3).map(|i| pool.insert(format!("{i}"))).collect();
        pool.remove(handles[1]);
        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.iter().count(), 0);
        assert!(handles.iter().all(|&h| !pool.contains(h)));
        let fresh = pool.insert(String::from("fresh"));
        assert!(handles.iter().all(|&h| pool.get(h).is_none()));
        assert_eq!(pool.get(fresh).map(String::as_str), Some("fresh"));
    }

    #[test]
    fn iter_yields_live_values_with_their_handles() {
        let mut pool: ResourcePool<Pipeline, String> = ResourcePool::new();
        let handles: Vec<_> = ["a", "b", "c"]
            .map(|s| pool.insert(String::from(s)))
            .to_vec();
        pool.remove(handles[0]);
        for (_, value) in pool.iter_mut() {
            value.push('!');
        }
        let seen: Vec<_> = pool.iter().map(|(h, v)| (h, v.as_str())).collect();
        assert_eq!(seen, [(handles[1], "b!"), (handles[2], "c!")]);
        // Every yielded handle resolves to the value yielded with it.
        assert!(pool.iter().all(|(h, v)| pool.get(h) == Some(v)));
    }

    #[test]
    fn usage_flags_combine_and_print_their_names() {
        let mut usage = BufferUsage::VERTEX | BufferUsage::COPY_DST;
        assert!(usage.contains(BufferUsage::VERTEX));
        assert!(!usage.contains(BufferUsage::VERTEX | BufferUsage::INDEX));
        assert!(usage.contains(BufferUsage::EMPTY));
        assert_eq!(format!("{usage:?}"), "BufferUsage(VERTEX | COPY_DST)");
        usage &= BufferUsage::INDEX;
        assert!(usage.is_empty());
        assert_eq!(format!("{usage:?}"), "BufferUsage(EMPTY)");
        usage |= BufferUsage::UNIFORM;
        assert_eq!(usage.bits(), 1 << 2);
    }

    #[test]
    fn texture_format_sizes() {
        assert_eq!(TextureFormat::Bgra8UnormSrgb.bytes_per_pixel(), Some(4));
        assert_eq!(TextureFormat::R8Unorm.bytes_per_pixel(), Some(1));
        assert_eq!(TextureFormat::Depth24PlusStencil8.bytes_per_pixel(), None);
        assert!(TextureFormat::Depth32Float.is_depth());
        assert!(!TextureFormat::Rgba8Unorm.is_depth());
    }
}
