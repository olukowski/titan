//! Window configuration and surface size.

use alloc::string::String;
use titan_core::Vec2;

/// How a platform backend should open the app's window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowConfig {
    /// The window title.
    pub title: String,
    /// The initial width in logical points, which the platform scales to physical pixels.
    pub width: u32,
    /// The initial height in logical points.
    pub height: u32,
    /// Whether the user can resize the window.
    pub resizable: bool,
}

impl WindowConfig {
    /// Creates a resizable window configuration.
    #[must_use]
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            resizable: true,
        }
    }
}

impl Default for WindowConfig {
    /// A resizable 800 x 600 window titled "Titan".
    fn default() -> Self {
        Self::new("Titan", 800, 600)
    }
}

/// The size of the surface the app renders to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSize {
    /// Width in physical framebuffer pixels.
    pub width: u32,
    /// Height in physical framebuffer pixels.
    pub height: u32,
    /// Physical pixels per logical point, such as `2.0` on a Retina display.
    pub scale_factor: f32,
}

impl SurfaceSize {
    /// Creates a surface size from physical pixels and a scale factor.
    ///
    /// # Panics
    ///
    /// Panics unless `scale_factor` is finite and positive.
    #[must_use]
    pub fn new(width: u32, height: u32, scale_factor: f32) -> Self {
        assert!(
            scale_factor.is_finite() && scale_factor > 0.0,
            "scale_factor must be finite and positive"
        );
        Self {
            width,
            height,
            scale_factor,
        }
    }

    /// Returns the size in logical points.
    #[must_use]
    pub fn logical_size(self) -> Vec2 {
        Vec2::new(self.width as f32, self.height as f32) / self.scale_factor
    }

    /// Returns width / height, or `0.0` for a zero-height (minimized) surface.
    #[must_use]
    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

impl Default for SurfaceSize {
    /// 800 x 600 pixels at a scale factor of 1.
    fn default() -> Self {
        Self::new(800, 600, 1.0)
    }
}

/// What the app wants after a simulation step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AppAction {
    /// Keep running.
    #[default]
    Continue,
    /// Stop: no more steps or frames run.
    Exit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retina_surface_converts_to_logical_points() {
        let size = SurfaceSize::new(2560, 1600, 2.0);
        assert_eq!(size.logical_size(), Vec2::new(1280.0, 800.0));
        assert_eq!(size.aspect_ratio(), 1.6);
        assert_eq!(SurfaceSize::new(100, 0, 1.0).aspect_ratio(), 0.0);
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn zero_scale_factor_panics() {
        let _ = SurfaceSize::new(800, 600, 0.0);
    }
}
