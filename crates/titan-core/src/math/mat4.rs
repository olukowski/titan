use super::{
    Quat, Vec3, Vec4,
    scalar::{sin_cos, sqrt_f64},
};
use core::ops::Mul;

/// A 4×4 `f32` matrix stored column-major, laid out as WGSL's and MSL's `float4x4`.
///
/// # Conventions
///
/// - Vectors are columns and transforms compose right to left: `projection * view * model * v`.
/// - World and view space are right-handed with `+Y` up; the camera looks down `-Z`.
/// - Clip-space depth runs from `0.0` at the near plane to `1.0` at the far plane, which is what
///   both Metal and WebGPU expect. (OpenGL's `[-1, 1]` depth is not supported.)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    /// The four columns; `cols[3]` holds the translation of an affine transform.
    pub cols: [Vec4; 4],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    /// The identity transform.
    pub const IDENTITY: Self = Self::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::W);

    /// Creates a matrix from its four columns.
    #[must_use]
    pub const fn from_cols(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Self {
        Self { cols: [x, y, z, w] }
    }

    /// Creates a translation.
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self::from_cols(Vec4::X, Vec4::Y, Vec4::Z, translation.extend(1.0))
    }

    /// Creates a non-uniform scale.
    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self::from_cols(
            Vec4::new(scale.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, scale.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, scale.z, 0.0),
            Vec4::W,
        )
    }

    /// Creates a rotation from a unit quaternion.
    #[must_use]
    pub fn from_quat(rotation: Quat) -> Self {
        let Quat { x, y, z, w } = rotation;
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self::from_cols(
            Vec4::new(1.0 - (yy + zz), xy + wz, xz - wy, 0.0),
            Vec4::new(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0),
            Vec4::new(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0),
            Vec4::W,
        )
    }

    /// Creates the transform that scales, then rotates, then translates.
    #[must_use]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let [x, y, z, _] = Self::from_quat(rotation).cols;
        Self::from_cols(
            x * scale.x,
            y * scale.y,
            z * scale.z,
            translation.extend(1.0),
        )
    }

    /// Creates an orthographic projection of the view-space box `[left, right] × [bottom, top]`
    /// between the planes `z = -near` and `z = -far`, mapping depth `-near → 0` and `-far → 1`.
    #[must_use]
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        // Extents and sums of finite `f32` bounds can overflow `f32`, but not `f64`.
        let [left, right, bottom, top, near, far] =
            [left, right, bottom, top, near, far].map(f64::from);
        let rcp_width = 1.0 / (right - left);
        let rcp_height = 1.0 / (top - bottom);
        let r = 1.0 / (near - far);
        Self::from_cols(
            Vec4::new((2.0 * rcp_width) as f32, 0.0, 0.0, 0.0),
            Vec4::new(0.0, (2.0 * rcp_height) as f32, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r as f32, 0.0),
            Vec4::new(
                (-(left + right) * rcp_width) as f32,
                (-(top + bottom) * rcp_height) as f32,
                (r * near) as f32,
                1.0,
            ),
        )
    }

    /// Creates a perspective projection with vertical field of view `fov_y` (radians) and
    /// `aspect = width / height`, mapping depth `z = -near → 0` and `z = -far → 1` after the
    /// divide by `w`.
    #[must_use]
    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let (sin, cos) = sin_cos(0.5 * fov_y);
        let h = cos / sin;
        let r = far / (near - far);
        Self::from_cols(
            Vec4::new(h / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, -1.0),
            Vec4::new(0.0, 0.0, r * near, 0.0),
        )
    }

    /// Creates a view matrix for a camera at `eye` looking at `target`, with `up` as the
    /// approximate up direction.
    ///
    /// # Panics
    ///
    /// Panics if `eye == target`, if `up` is zero or not finite, or if `up` is parallel to the
    /// view direction, since no camera orientation exists then. Directions within about 1e-6
    /// radians of parallel are numerically indistinguishable from it and panic too.
    #[must_use]
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        // The basis is built in `f64`: in `f32` the rounding residue of a parallel `up` is large
        // enough to pass for a real side axis.
        type V = [f64; 3];
        let wide = |v: Vec3| [v.x, v.y, v.z].map(f64::from);
        let cross = |a: V, b: V| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let unit = |v: V, min_len: f64| {
            let len = sqrt_f64(v.iter().map(|c| c * c).sum());
            (len.is_finite() && len > min_len).then(|| v.map(|c| c / len))
        };
        let narrow = |v: V| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
        let (eye_wide, target, up) = (wide(eye), wide(target), wide(up));
        let forward = [0, 1, 2].map(|i| target[i] - eye_wide[i]);
        let f = unit(forward, 0.0).expect("look_at_rh: eye equals target");
        let up = unit(up, 0.0).expect("look_at_rh: up is zero or not finite");
        // Rounding in `forward` leaves a residue in the cross product of parallel directions.
        // Between unit vectors its length is the sine of their angle, so a scale-free floor far
        // above that residue rejects it instead of normalizing it into a bogus side axis.
        let s = unit(cross(f, up), 1e-6).expect("look_at_rh: up is parallel to the view");
        let (u, f, s) = (narrow(cross(s, f)), narrow(f), narrow(s));
        Self::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        )
    }

    /// Returns the transpose.
    #[must_use]
    pub fn transpose(self) -> Self {
        let [a, b, c, d] = self.cols;
        Self::from_cols(
            Vec4::new(a.x, b.x, c.x, d.x),
            Vec4::new(a.y, b.y, c.y, d.y),
            Vec4::new(a.z, b.z, c.z, d.z),
            Vec4::new(a.w, b.w, c.w, d.w),
        )
    }

    /// Transforms a point (`w = 1`). Assumes `self` is affine, so the result is not divided by
    /// `w`; use `self * point.extend(1.0)` for projections.
    #[must_use]
    pub fn transform_point3(self, point: Vec3) -> Vec3 {
        (self * point.extend(1.0)).truncate()
    }

    /// Transforms a direction (`w = 0`), ignoring translation.
    #[must_use]
    pub fn transform_vector3(self, vector: Vec3) -> Vec3 {
        (self * vector.extend(0.0)).truncate()
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        let [a, b, c, d] = self.cols;
        a * v.x + b * v.y + c * v.z + d * v.w
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            cols: rhs.cols.map(|col| self * col),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;

    fn assert_vec4_near(got: Vec4, want: Vec4) {
        assert!((got - want).length() < 1e-5, "got {got:?}, want {want:?}");
    }

    /// Projects a view-space point and returns normalized device coordinates.
    fn ndc(proj: Mat4, p: Vec3) -> Vec3 {
        let clip = proj * p.extend(1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn orthographic_maps_box_to_zero_one_depth() {
        let proj = Mat4::orthographic_rh(-4.0, 2.0, -1.0, 3.0, 0.5, 10.0);
        assert_eq!(
            ndc(proj, Vec3::new(-4.0, -1.0, -0.5)),
            Vec3::new(-1.0, -1.0, 0.0)
        );
        assert_eq!(
            ndc(proj, Vec3::new(2.0, 3.0, -10.0)),
            Vec3::new(1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn perspective_maps_near_far_to_zero_one_depth() {
        let proj = Mat4::perspective_rh(FRAC_PI_2, 2.0, 0.1, 100.0);
        assert!(ndc(proj, Vec3::new(0.0, 0.0, -0.1)).z.abs() < 1e-6);
        assert!((ndc(proj, Vec3::new(0.0, 0.0, -100.0)).z - 1.0).abs() < 1e-6);
        // A 90° vertical FOV puts y = -z on the top edge; aspect 2 puts x = -2z on the right.
        let corner = ndc(proj, Vec3::new(10.0, 5.0, -5.0));
        assert!((corner.x - 1.0).abs() < 1e-6 && (corner.y - 1.0).abs() < 1e-6);
        // Points behind the camera get negative w and are clipped.
        assert!((proj * Vec4::new(0.0, 0.0, 1.0, 1.0)).w < 0.0);
    }

    #[test]
    fn look_at_puts_target_on_negative_z() {
        let eye = Vec3::new(3.0, 2.0, 5.0);
        let view = Mat4::look_at_rh(eye, Vec3::new(3.0, 2.0, -5.0), Vec3::Y);
        assert_eq!(view.transform_point3(eye), Vec3::ZERO);
        assert_eq!(
            view.transform_point3(Vec3::new(3.0, 2.0, -5.0)),
            Vec3::new(0.0, 0.0, -10.0)
        );
        assert_eq!(view.transform_vector3(Vec3::X), Vec3::X);
    }

    #[test]
    fn orthographic_handles_huge_bounds() {
        let proj = Mat4::orthographic_rh(-2e38, 2e38, -2e38, 2e38, 0.0, 1.0);
        let corner = ndc(proj, Vec3::new(2e38, -2e38, -1.0));
        // The x and y scales are f32 subnormals, so they keep only about 22 bits.
        assert_vec4_near(corner.extend(0.0), Vec4::new(1.0, -1.0, 1.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "up is parallel")]
    fn look_at_rejects_parallel_up() {
        let _ = Mat4::look_at_rh(Vec3::ZERO, Vec3::Y, Vec3::Y);
    }

    #[test]
    #[should_panic(expected = "up is parallel")]
    fn look_at_rejects_parallel_up_off_axis() {
        let dir = Vec3::new(1.0, 5.0, 6.0);
        let _ = Mat4::look_at_rh(Vec3::ZERO, dir, dir);
    }

    #[test]
    #[should_panic(expected = "up is parallel")]
    fn look_at_rejects_parallel_up_with_rounded_forward() {
        // `target - eye` is exactly parallel to `up`, but the f64 cross product keeps a residue.
        let up = Vec3::new(1.0, 5.0, 6.0);
        let _ = Mat4::look_at_rh(up * 2.0f32.powi(-51), up, up);
    }

    #[test]
    fn compose_scale_rotation_translation() {
        let rotation = Quat::from_rotation_z(FRAC_PI_2);
        let (scale, translation) = (Vec3::new(2.0, 3.0, 4.0), Vec3::new(10.0, 20.0, 30.0));
        let srt = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        let chained = Mat4::from_translation(translation)
            * Mat4::from_quat(rotation)
            * Mat4::from_scale(scale);
        for (a, b) in srt.cols.into_iter().zip(chained.cols) {
            assert_vec4_near(a, b);
        }
        // Scale x by 2, rotate +90° about z (x → y), then translate.
        let p = srt.transform_point3(Vec3::X);
        assert!((p - Vec3::new(10.0, 22.0, 30.0)).length() < 1e-5, "{p:?}");
        assert_eq!(srt.transform_vector3(Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.transpose().cols[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(m.transpose().transpose(), m);
        assert_eq!(m * Mat4::IDENTITY, m);
    }
}
