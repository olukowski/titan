//! Math primitives for Titan's deterministic core systems.

use std::ops::{Add, Sub};

/// A three-dimensional vector using `f32` components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X axis component.
    pub x: f32,
    /// Y axis component.
    pub y: f32,
    /// Z axis component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Creates a vector from component values.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Computes the dot product of two vectors.
    pub fn dot(self, other: Self) -> f32 {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

#[cfg(test)]
mod tests {
    use super::Vec3;

    #[test]
    fn adds_components() {
        let left = Vec3::new(1.0, 2.0, 3.0);
        let right = Vec3::new(4.0, 5.0, 6.0);

        assert_eq!(left + right, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn computes_dot_product() {
        let left = Vec3::new(1.0, 3.0, -5.0);
        let right = Vec3::new(4.0, -2.0, -1.0);

        assert_eq!(left.dot(right), 3.0);
    }
}
