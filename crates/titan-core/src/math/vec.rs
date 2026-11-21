//! `f32` vectors with `#[repr(C)]` layout, so slices of them can be uploaded to the GPU as packed
//! vertex attributes. WGSL and MSL uniform and storage arrays of 3-component vectors use a 16-byte
//! stride; use [`Vec4`] or explicit padding there.

use super::scalar::sqrt_f64;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
}

/// A 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

/// A 4D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
    /// The w component.
    pub w: f32,
}

/// Implements the API shared by every vector type, component by component.
macro_rules! impl_vec {
    ($Vec:ident { $($f:ident),+ }) => {
        impl $Vec {
            /// All components zero.
            pub const ZERO: Self = Self::splat(0.0);
            /// All components one.
            pub const ONE: Self = Self::splat(1.0);

            /// Creates a vector from its components.
            #[must_use]
            pub const fn new($($f: f32),+) -> Self {
                Self { $($f),+ }
            }

            /// Creates a vector with every component set to `v`.
            #[must_use]
            pub const fn splat(v: f32) -> Self {
                Self { $($f: v),+ }
            }

            /// Returns the dot product of `self` and `rhs`.
            #[must_use]
            pub fn dot(self, rhs: Self) -> f32 {
                0.0 $(+ self.$f * rhs.$f)+
            }

            /// Returns the squared length; cheaper than [`length`](Self::length) for comparisons.
            #[must_use]
            pub fn length_squared(self) -> f32 {
                self.dot(self)
            }

            /// Returns the length.
            #[must_use]
            pub fn length(self) -> f32 {
                self.length_f64() as f32
            }

            /// Squaring in `f64` neither underflows nor overflows for any finite `f32`
            /// components, so extreme but finite vectors keep an accurate length.
            fn length_f64(self) -> f64 {
                sqrt_f64(0.0 $(+ f64::from(self.$f) * f64::from(self.$f))+)
            }

            /// Returns `self` scaled to length one, or `None` if its length is zero or not
            /// finite.
            #[must_use]
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length_f64();
                (len.is_finite() && len > 0.0)
                    .then(|| Self { $($f: (f64::from(self.$f) / len) as f32),+ })
            }

            /// Returns `self` scaled to length one, or zero if its length is zero or not finite.
            #[must_use]
            pub fn normalize_or_zero(self) -> Self {
                self.try_normalize().unwrap_or(Self::ZERO)
            }

            /// Linearly interpolates from `self` (`t = 0`) to `rhs` (`t = 1`), returning each
            /// endpoint exactly.
            #[must_use]
            pub fn lerp(self, rhs: Self, t: f32) -> Self {
                // The mix loses signed zeros and turns an infinite unused endpoint into NaN.
                if t == 0.0 {
                    return self;
                }
                if t == 1.0 {
                    return rhs;
                }
                // In `f64` the products are exact and the sum cannot overflow.
                let t = f64::from(t);
                let mix = |a: f32, b: f32| ((1.0 - t) * f64::from(a) + t * f64::from(b)) as f32;
                Self { $($f: mix(self.$f, rhs.$f)),+ }
            }
        }

        impl Neg for $Vec {
            type Output = Self;
            fn neg(self) -> Self {
                Self { $($f: -self.$f),+ }
            }
        }

        impl Mul<$Vec> for f32 {
            type Output = $Vec;
            fn mul(self, rhs: $Vec) -> $Vec {
                rhs * self
            }
        }

        impl_vec!(@op $Vec { $($f),+ } Add add AddAssign add_assign +);
        impl_vec!(@op $Vec { $($f),+ } Sub sub SubAssign sub_assign -);
        impl_vec!(@op $Vec { $($f),+ } Mul mul MulAssign mul_assign *);
        impl_vec!(@op $Vec { $($f),+ } Div div DivAssign div_assign /);
    };
    // Component-wise `vec op vec` and `vec op f32`, plus the assigning forms of both.
    (@op $Vec:ident { $($f:ident),+ } $Op:ident $op:ident $OpAssign:ident $op_assign:ident $sym:tt) => {
        impl $Op for $Vec {
            type Output = Self;
            fn $op(self, rhs: Self) -> Self {
                Self { $($f: self.$f $sym rhs.$f),+ }
            }
        }

        impl $Op<f32> for $Vec {
            type Output = Self;
            fn $op(self, rhs: f32) -> Self {
                Self { $($f: self.$f $sym rhs),+ }
            }
        }

        impl $OpAssign for $Vec {
            fn $op_assign(&mut self, rhs: Self) {
                *self = *self $sym rhs;
            }
        }

        impl $OpAssign<f32> for $Vec {
            fn $op_assign(&mut self, rhs: f32) {
                *self = *self $sym rhs;
            }
        }
    };
}

impl_vec!(Vec2 { x, y });
impl_vec!(Vec3 { x, y, z });
impl_vec!(Vec4 { x, y, z, w });

impl Vec2 {
    /// The unit x axis.
    pub const X: Self = Self::new(1.0, 0.0);
    /// The unit y axis.
    pub const Y: Self = Self::new(0.0, 1.0);

    /// Appends `z` to make a [`Vec3`].
    #[must_use]
    pub const fn extend(self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }
}

impl Vec3 {
    /// The unit x axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// The unit y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// The unit z axis.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    /// Returns the cross product; `X.cross(Y) == Z` (right-handed).
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Appends `w` to make a [`Vec4`].
    #[must_use]
    pub const fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// Drops `z` to make a [`Vec2`].
    #[must_use]
    pub const fn truncate(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

impl Vec4 {
    /// The unit x axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    /// The unit y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    /// The unit z axis.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    /// The unit w axis.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// Drops `w` to make a [`Vec3`].
    #[must_use]
    pub const fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_rejects_degenerate_lengths() {
        assert_eq!(
            Vec3::new(3.0, 0.0, 4.0).try_normalize(),
            Some(Vec3::new(0.6, 0.0, 0.8))
        );
        assert_eq!(Vec3::ZERO.try_normalize(), None);
        assert_eq!(Vec2::new(f32::INFINITY, 0.0).try_normalize(), None);
        assert_eq!(
            Vec4::new(f32::NAN, 0.0, 0.0, 0.0).normalize_or_zero(),
            Vec4::ZERO
        );
    }

    #[test]
    fn normalize_handles_extreme_finite_lengths() {
        // Squaring these underflows or overflows in f32.
        for x in [1e-22, 1e-30, f32::from_bits(1), 1e30, f32::MAX] {
            assert_eq!(Vec2::new(x, 0.0).try_normalize(), Some(Vec2::X), "{x:e}");
            assert_eq!(Vec3::new(0.0, -x, 0.0).length(), x, "{x:e}");
        }
        let v = Vec4::new(f32::MAX, f32::MAX, -f32::MAX, f32::MAX);
        assert_eq!(v.try_normalize(), Some(Vec4::new(0.5, 0.5, -0.5, 0.5)));
    }

    #[test]
    fn lerp_hits_endpoints_at_extremes() {
        let (a, b) = (Vec2::new(1e8, -f32::MAX), Vec2::new(1.0, f32::MAX));
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        assert_eq!(a.lerp(b, 0.5).y, 0.0);
    }

    #[test]
    fn lerp_returns_endpoints_bitwise() {
        let a = Vec2::new(-0.0, 2.0);
        assert_eq!(a.lerp(Vec2::ONE, 0.0).x.to_bits(), (-0.0f32).to_bits());
        assert_eq!(Vec2::ONE.lerp(a, 1.0).x.to_bits(), (-0.0f32).to_bits());
        // The unused endpoint is infinite, which the weighted mix would turn into NaN.
        let inf = Vec2::splat(f32::INFINITY);
        assert_eq!(a.lerp(inf, 0.0), a);
        assert_eq!(inf.lerp(a, 1.0), a);
    }

    #[test]
    fn cross_is_right_handed() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::Z), Vec3::X);
        assert_eq!(Vec3::Y.cross(Vec3::X), -Vec3::Z);
    }

    #[test]
    fn operators_are_component_wise() {
        let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        v *= Vec4::new(2.0, 2.0, 0.5, 0.0);
        v -= 1.0;
        assert_eq!(v, Vec4::new(1.0, 3.0, 0.5, -1.0));
        assert_eq!(2.0 * Vec2::X + Vec2::Y / 4.0, Vec2::new(2.0, 0.25));
        assert_eq!(
            Vec2::ZERO.lerp(Vec2::new(4.0, -8.0), 0.25),
            Vec2::new(1.0, -2.0)
        );
    }
}
