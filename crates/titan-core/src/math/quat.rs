use super::{
    Vec3, Vec4,
    scalar::{acos, sin_cos},
};
use core::ops::Mul;

/// A rotation quaternion `w + xi + yj + zk`, laid out as `x, y, z, w`.
///
/// Rotations follow the right-hand rule: a positive angle turns counterclockwise when looking
/// down the axis toward the origin. Most methods assume a unit quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Quat {
    /// The `i` component.
    pub x: f32,
    /// The `j` component.
    pub y: f32,
    /// The `k` component.
    pub z: f32,
    /// The real component.
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// No rotation.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Creates a rotation of `angle` radians about the unit vector `axis`.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (sin, cos) = sin_cos(0.5 * angle);
        let v = axis * sin;
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: cos,
        }
    }

    /// Creates a rotation of `angle` radians about the x axis.
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        Self::from_axis_angle(Vec3::X, angle)
    }

    /// Creates a rotation of `angle` radians about the y axis.
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        Self::from_axis_angle(Vec3::Y, angle)
    }

    /// Creates a rotation of `angle` radians about the z axis.
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        Self::from_axis_angle(Vec3::Z, angle)
    }

    /// Returns `self` scaled to unit length. The result is NaN if `self` is zero or not finite.
    #[must_use]
    pub fn normalize(self) -> Self {
        let unit = self.to_vec4().try_normalize();
        Self::from_vec4(unit.unwrap_or(Vec4::splat(f32::NAN)))
    }

    /// Returns the conjugate, which is the inverse rotation for a unit quaternion.
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Spherically interpolates from `self` (`t = 0`) to `end` (`t = 1`) at constant angular
    /// speed, along the shorter of the two arcs.
    #[must_use]
    pub fn slerp(self, end: Self, t: f32) -> Self {
        let (start, mut end) = (self.to_vec4(), end.to_vec4());
        let mut dot = start.dot(end);
        // q and -q are the same rotation; flipping to a non-negative dot takes the short way.
        if dot < 0.0 {
            end = -end;
            dot = -dot;
        }
        // Nearly parallel: sin(θ) → 0 makes the slerp weights unstable, and lerp is exact enough.
        if dot > 0.9995 {
            return Self::from_vec4(start.lerp(end, t)).normalize();
        }
        let theta = acos(dot);
        let (sin_theta, _) = sin_cos(theta);
        let (sin_start, _) = sin_cos((1.0 - t) * theta);
        let (sin_end, _) = sin_cos(t * theta);
        Self::from_vec4((start * sin_start + end * sin_end) / sin_theta)
    }

    fn to_vec4(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.w)
    }

    fn from_vec4(v: Vec4) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }
}

impl Mul for Quat {
    type Output = Self;
    /// Composes rotations: `(a * b) * v` rotates `v` by `b`, then by `a`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self, rhs);
        Self {
            x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        }
    }
}

impl Mul<Vec3> for Quat {
    type Output = Vec3;
    /// Rotates `v` by the unit quaternion `self`.
    fn mul(self, v: Vec3) -> Vec3 {
        let q = Vec3::new(self.x, self.y, self.z);
        let t = 2.0 * q.cross(v);
        v + self.w * t + q.cross(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn assert_vec3_near(got: Vec3, want: Vec3) {
        assert!((got - want).length() < 1e-6, "got {got:?}, want {want:?}");
    }

    fn assert_same_rotation(a: Quat, b: Quat) {
        // q and -q encode the same rotation.
        let dot = a.to_vec4().dot(b.to_vec4());
        assert!((dot.abs() - 1.0).abs() < 1e-6, "{a:?} vs {b:?}");
    }

    #[test]
    fn axis_angle_rotates_by_right_hand_rule() {
        assert_vec3_near(Quat::from_rotation_z(FRAC_PI_2) * Vec3::X, Vec3::Y);
        assert_vec3_near(Quat::from_rotation_x(FRAC_PI_2) * Vec3::Y, Vec3::Z);
        assert_vec3_near(Quat::from_rotation_y(FRAC_PI_2) * Vec3::Z, Vec3::X);
        let diagonal = Vec3::ONE.normalize_or_zero();
        // A third of a turn about (1, 1, 1) cycles the axes.
        assert_vec3_near(
            Quat::from_axis_angle(diagonal, 2.0 * PI / 3.0) * Vec3::X,
            Vec3::Y,
        );
    }

    #[test]
    fn mul_composes_right_to_left_and_conjugate_inverts() {
        let (a, b) = (Quat::from_rotation_x(0.7), Quat::from_rotation_z(-1.3));
        let v = Vec3::new(0.3, -2.0, 5.0);
        assert_vec3_near((a * b) * v, a * (b * v));
        assert_vec3_near(a.conjugate() * (a * v), v);
        assert_same_rotation(a * a.conjugate(), Quat::IDENTITY);
    }

    #[test]
    fn slerp_follows_the_shortest_arc() {
        let start = Quat::IDENTITY;
        let end = Quat::from_rotation_z(FRAC_PI_2);
        assert_same_rotation(start.slerp(end, 0.5), Quat::from_rotation_z(FRAC_PI_4));
        assert_same_rotation(start.slerp(end, 0.0), start);
        assert_same_rotation(start.slerp(end, 1.0), end);
        // -end is the same rotation; without the sign flip this would take the 270° way round.
        let flipped = Quat::from_vec4(-end.to_vec4());
        assert_same_rotation(start.slerp(flipped, 0.5), Quat::from_rotation_z(FRAC_PI_4));
        // Nearly identical inputs take the lerp path and still return a unit quaternion.
        let near = Quat::from_rotation_z(1e-4);
        let mid = start.slerp(near, 0.5);
        assert!((mid.to_vec4().length() - 1.0).abs() < 1e-6);
        assert_same_rotation(mid, Quat::from_rotation_z(5e-5));
    }

    #[test]
    fn from_quat_matches_quat_rotation() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, -2.0, 0.5).normalize_or_zero(), 2.1);
        let v = Vec3::new(-1.0, 4.0, 2.5);
        assert_vec3_near(super::super::Mat4::from_quat(q).transform_vector3(v), q * v);
        let unnormalized = Quat {
            x: 0.0,
            y: 0.0,
            z: 2.0,
            w: 0.0,
        };
        assert_eq!(
            unnormalized.normalize(),
            Quat {
                x: 0.0,
                y: 0.0,
                z: 1.0,
                w: 0.0
            }
        );
    }

    #[test]
    fn normalize_handles_extreme_finite_lengths() {
        for x in [1e-22, 1e-30, 1e30] {
            let q = Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: x,
            };
            assert_eq!(q.normalize(), Quat::IDENTITY, "{x:e}");
        }
    }
}
