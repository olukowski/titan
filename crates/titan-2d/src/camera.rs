//! The 2D camera: which part of the world the screen shows.

use titan_core::math::sin_cos;
use titan_core::{Mat4, Quat, Vec2, Vec3, Vec4};

/// An orthographic 2D camera.
///
/// # Conventions
///
/// - World space is right-handed with `+X` right and `+Y` up, matching [`Mat4`].
/// - Screen space is in pixels with `(0, 0)` at the top-left corner and `viewport` at the
///   bottom-right, `+X` right and `+Y` down, matching pointer events and
///   [`Viewport`](titan_gpu::Viewport).
///
/// The camera centers the screen on `position`, shows `zoom` pixels per world unit, and turns
/// the view `rotation` radians counter-clockwise, so the world appears to turn clockwise.
///
/// Every method panics unless `viewport` and `zoom` are finite and positive and `position` and
/// `rotation` are finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera2D {
    /// The world point at the center of the screen.
    pub position: Vec2,
    /// The screen size in pixels.
    pub viewport: Vec2,
    /// Pixels per world unit.
    pub zoom: f32,
    /// Counter-clockwise rotation of the view in radians.
    pub rotation: f32,
}

impl Camera2D {
    /// A camera over a screen of `viewport` pixels, one pixel per world unit, centered on the
    /// world origin.
    #[must_use]
    pub fn new(viewport: Vec2) -> Self {
        Self {
            position: Vec2::ZERO,
            viewport,
            zoom: 1.0,
            rotation: 0.0,
        }
    }

    /// A camera over a screen of `viewport` pixels, one pixel per world unit, with the world
    /// origin at the screen's bottom-left corner and world `viewport` at its top-right.
    #[must_use]
    pub fn bottom_left(viewport: Vec2) -> Self {
        Self {
            position: viewport * 0.5,
            ..Self::new(viewport)
        }
    }

    /// Returns the matrix from world space to clip space. World `z = 0` maps to depth `0.5`.
    ///
    /// # Panics
    ///
    /// Panics if the camera is invalid; see [`Camera2D`].
    #[must_use]
    pub fn view_projection(&self) -> Mat4 {
        self.assert_valid();
        // Compute the scale in `f64` so intermediate `2.0 * zoom` or half-extents
        // `viewport / (2.0 * zoom)` do not overflow `f32` when the final scale fits.
        let sx = (2.0 * f64::from(self.zoom) / f64::from(self.viewport.x)) as f32;
        let sy = (2.0 * f64::from(self.zoom) / f64::from(self.viewport.y)) as f32;
        let projection = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            * Mat4::from_scale(Vec3::new(sx, sy, 1.0));
        let view = Mat4::from_quat(Quat::from_rotation_z(-self.rotation))
            * Mat4::from_translation(Vec3::new(-self.position.x, -self.position.y, 0.0));
        projection * view
    }

    /// Returns the screen pixel that shows `world`.
    ///
    /// # Panics
    ///
    /// Panics if the camera is invalid; see [`Camera2D`].
    #[must_use]
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        let ndc = self.view_projection() * Vec4::new(world.x, world.y, 0.0, 1.0);
        Vec2::new(
            (ndc.x + 1.0) * 0.5 * self.viewport.x,
            (1.0 - ndc.y) * 0.5 * self.viewport.y,
        )
    }

    /// Returns the world point that `screen` shows: the inverse of
    /// [`world_to_screen`](Self::world_to_screen).
    ///
    /// # Panics
    ///
    /// Panics if the camera is invalid; see [`Camera2D`].
    #[must_use]
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        self.assert_valid();
        let view_x = (screen.x - self.viewport.x * 0.5) / self.zoom;
        let view_y = (self.viewport.y * 0.5 - screen.y) / self.zoom;
        let (sin, cos) = sin_cos(self.rotation);
        self.position + Vec2::new(view_x * cos - view_y * sin, view_x * sin + view_y * cos)
    }

    fn assert_valid(&self) {
        let positive = |v: f32| v.is_finite() && v > 0.0;
        assert!(
            positive(self.viewport.x) && positive(self.viewport.y),
            "Camera2D viewport must be finite and positive, got {:?}",
            self.viewport
        );
        assert!(
            positive(self.zoom),
            "Camera2D zoom must be finite and positive, got {}",
            self.zoom
        );
        assert!(
            self.position.x.is_finite() && self.position.y.is_finite() && self.rotation.is_finite(),
            "Camera2D position and rotation must be finite, got {:?} and {}",
            self.position,
            self.rotation
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;

    const VIEWPORT: Vec2 = Vec2::new(800.0, 600.0);

    fn clip(camera: &Camera2D, world: Vec2) -> Vec4 {
        camera.view_projection() * Vec4::new(world.x, world.y, 0.0, 1.0)
    }

    fn assert_near(actual: Vec2, expected: Vec2) {
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1.0);
        assert!(
            close(actual.x, expected.x) && close(actual.y, expected.y),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn projection_maps_viewport_corners_to_ndc_corners_at_mid_depth() {
        let camera = Camera2D::new(VIEWPORT);
        for (world, ndc) in [
            (Vec2::new(-400.0, -300.0), Vec4::new(-1.0, -1.0, 0.5, 1.0)),
            (Vec2::new(400.0, 300.0), Vec4::new(1.0, 1.0, 0.5, 1.0)),
            (Vec2::new(-400.0, 300.0), Vec4::new(-1.0, 1.0, 0.5, 1.0)),
            (Vec2::ZERO, Vec4::new(0.0, 0.0, 0.5, 1.0)),
        ] {
            assert_eq!(clip(&camera, world), ndc, "{world:?}");
        }
    }

    #[test]
    fn bottom_left_puts_the_origin_at_the_screen_bottom_left() {
        let camera = Camera2D::bottom_left(VIEWPORT);
        assert_eq!(camera.world_to_screen(Vec2::ZERO), Vec2::new(0.0, 600.0));
        assert_eq!(camera.world_to_screen(VIEWPORT), Vec2::new(800.0, 0.0));
        assert_eq!(
            camera.world_to_screen(Vec2::new(100.0, 50.0)),
            Vec2::new(100.0, 550.0)
        );
        assert_eq!(
            camera.screen_to_world(Vec2::new(100.0, 550.0)),
            Vec2::new(100.0, 50.0)
        );
    }

    #[test]
    fn zoom_and_pan_scale_and_shift_the_view() {
        let zoomed = Camera2D {
            zoom: 2.0,
            ..Camera2D::new(VIEWPORT)
        };
        // 1/150 is inexact, so the zoomed projection rounds.
        assert_near(
            zoomed.world_to_screen(Vec2::new(100.0, 50.0)),
            Vec2::new(600.0, 200.0),
        );
        assert_eq!(
            zoomed.screen_to_world(Vec2::new(600.0, 200.0)),
            Vec2::new(100.0, 50.0)
        );
        let panned = Camera2D {
            position: Vec2::new(50.0, -20.0),
            ..Camera2D::new(VIEWPORT)
        };
        assert_eq!(
            panned.world_to_screen(Vec2::new(50.0, -20.0)),
            Vec2::new(400.0, 300.0)
        );
        assert_eq!(
            panned.world_to_screen(Vec2::new(60.0, -20.0)),
            Vec2::new(410.0, 300.0)
        );
    }

    #[test]
    fn rotating_the_camera_counter_clockwise_turns_the_world_clockwise() {
        let camera = Camera2D {
            rotation: FRAC_PI_2,
            ..Camera2D::new(VIEWPORT)
        };
        // World up now points right on screen, and world right points down.
        assert_near(
            camera.world_to_screen(Vec2::new(0.0, 100.0)),
            Vec2::new(500.0, 300.0),
        );
        assert_near(
            camera.world_to_screen(Vec2::new(100.0, 0.0)),
            Vec2::new(400.0, 400.0),
        );
    }

    #[test]
    fn screen_to_world_inverts_world_to_screen() {
        let camera = Camera2D {
            position: Vec2::new(-37.5, 120.25),
            viewport: Vec2::new(1280.0, 720.0),
            zoom: 3.5,
            rotation: 0.7,
        };
        for world in [
            Vec2::ZERO,
            Vec2::new(-37.5, 120.25),
            Vec2::new(100.0, -250.0),
            Vec2::new(-180.0, 300.0),
        ] {
            assert_near(camera.screen_to_world(camera.world_to_screen(world)), world);
        }
        for screen in [Vec2::ZERO, Vec2::new(1280.0, 720.0), Vec2::new(17.0, 650.0)] {
            assert_near(
                camera.world_to_screen(camera.screen_to_world(screen)),
                screen,
            );
        }
    }

    #[test]
    fn extreme_finite_zooms_still_center_the_origin() {
        for zoom in [f32::MAX, 1e-38] {
            let camera = Camera2D {
                zoom,
                ..Camera2D::new(VIEWPORT)
            };
            assert_eq!(
                camera.world_to_screen(Vec2::ZERO),
                Vec2::new(400.0, 300.0),
                "zoom {zoom:e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "viewport must be finite and positive")]
    fn zero_viewport_panics() {
        let _ = Camera2D::new(Vec2::new(800.0, 0.0)).view_projection();
    }

    #[test]
    #[should_panic(expected = "viewport must be finite and positive")]
    fn infinite_viewport_panics() {
        let _ = Camera2D::new(Vec2::new(f32::INFINITY, 600.0)).world_to_screen(Vec2::ZERO);
    }

    #[test]
    #[should_panic(expected = "zoom must be finite and positive")]
    fn negative_zoom_panics() {
        let camera = Camera2D {
            zoom: -1.0,
            ..Camera2D::new(VIEWPORT)
        };
        let _ = camera.screen_to_world(Vec2::ZERO);
    }

    #[test]
    #[should_panic(expected = "zoom must be finite and positive")]
    fn nan_zoom_panics() {
        let camera = Camera2D {
            zoom: f32::NAN,
            ..Camera2D::new(VIEWPORT)
        };
        let _ = camera.view_projection();
    }

    #[test]
    #[should_panic(expected = "position and rotation must be finite")]
    fn non_finite_rotation_panics() {
        let camera = Camera2D {
            rotation: f32::NAN,
            ..Camera2D::new(VIEWPORT)
        };
        let _ = camera.screen_to_world(Vec2::ZERO);
    }
}
