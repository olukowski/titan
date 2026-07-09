//! Core runtime types shared by Titan tools and engine crates.

use titan_math::Vec3;

/// A stable identifier for an entity in a Titan world.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EntityId(u64);

impl EntityId {
    /// Creates an entity identifier from a raw numeric value.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric value for serialization and diagnostics.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// A minimal transform component for positioning entities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// World-space translation.
    pub translation: Vec3,
}

impl Transform {
    /// Creates a transform at the provided translation.
    pub const fn from_translation(translation: Vec3) -> Self {
        Self { translation }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EntityId, Transform};
    use titan_math::Vec3;

    #[test]
    fn entity_id_round_trips_raw_value() {
        assert_eq!(EntityId::from_raw(42).raw(), 42);
    }

    #[test]
    fn default_transform_starts_at_origin() {
        assert_eq!(Transform::default().translation, Vec3::ZERO);
    }
}
