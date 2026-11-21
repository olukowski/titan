use super::Vec2;

/// An axis-aligned rectangle covering the half-open area `[min, max)`.
///
/// Half-open bounds let rectangles tile without overlap: a point on a shared edge belongs to
/// exactly one of them, and rectangles that only touch do not intersect.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Rect {
    /// The corner with the smallest coordinates (inclusive).
    pub min: Vec2,
    /// The corner with the largest coordinates (exclusive).
    pub max: Vec2,
}

impl Rect {
    /// Creates a rectangle from its corners; `min` must not exceed `max` on either axis.
    #[must_use]
    pub const fn from_min_max(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Creates a rectangle from its `min` corner and a non-negative size.
    #[must_use]
    pub fn from_pos_size(pos: Vec2, size: Vec2) -> Self {
        Self {
            min: pos,
            max: pos + size,
        }
    }

    /// Returns `max - min`.
    #[must_use]
    pub fn size(self) -> Vec2 {
        self.max - self.min
    }

    /// Returns the midpoint.
    #[must_use]
    pub fn center(self) -> Vec2 {
        // The sum of two finite `f32`s can overflow; in `f64` it cannot.
        let mid = |a: f32, b: f32| ((f64::from(a) + f64::from(b)) * 0.5) as f32;
        Vec2::new(mid(self.min.x, self.max.x), mid(self.min.y, self.max.y))
    }

    /// Returns whether `point` lies in `[min, max)`.
    #[must_use]
    pub fn contains(self, point: Vec2) -> bool {
        self.min.x <= point.x
            && point.x < self.max.x
            && self.min.y <= point.y
            && point.y < self.max.y
    }

    /// Returns whether the two rectangles share any area; touching edges and empty rectangles do
    /// not count.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        let non_empty = |r: Self| r.min.x < r.max.x && r.min.y < r.max.y;
        non_empty(self)
            && non_empty(other)
            && self.min.x < other.max.x
            && other.min.x < self.max.x
            && self.min.y < other.max.y
            && other.min.y < self.max.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_is_half_open() {
        let r = Rect::from_pos_size(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(
            (r.size(), r.center()),
            (Vec2::new(3.0, 4.0), Vec2::new(2.5, 4.0))
        );
        assert!(r.contains(Vec2::new(1.0, 2.0)));
        assert!(r.contains(Vec2::new(3.999, 5.999)));
        assert!(!r.contains(Vec2::new(4.0, 3.0)));
        assert!(!r.contains(Vec2::new(2.0, 6.0)));
        assert!(!r.contains(Vec2::new(0.999, 3.0)));
        assert!(!r.contains(Vec2::new(f32::NAN, 3.0)));
        // An empty rectangle contains nothing, not even its own corner.
        let empty = Rect::from_min_max(Vec2::ONE, Vec2::ONE);
        assert!(!empty.contains(Vec2::ONE));
    }

    #[test]
    fn intersects_requires_shared_area() {
        let a = Rect::from_min_max(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let overlapping = Rect::from_min_max(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let touching = Rect::from_min_max(Vec2::new(2.0, 0.0), Vec2::new(4.0, 2.0));
        let corner = Rect::from_min_max(Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0));
        let inside = Rect::from_min_max(Vec2::new(0.5, 0.5), Vec2::new(1.0, 1.0));
        let apart = Rect::from_min_max(Vec2::new(0.0, 5.0), Vec2::new(2.0, 6.0));
        assert!(a.intersects(overlapping) && overlapping.intersects(a));
        assert!(a.intersects(inside) && inside.intersects(a));
        assert!(!a.intersects(touching) && !touching.intersects(a));
        assert!(!a.intersects(corner));
        assert!(!a.intersects(apart));
        // Empty rectangles have no area to share, even inside another rectangle.
        let point = Rect::from_min_max(Vec2::ONE, Vec2::ONE);
        let zero_width = Rect::from_min_max(Vec2::new(1.0, 0.5), Vec2::new(1.0, 1.5));
        let zero_height = Rect::from_min_max(Vec2::new(0.5, 1.0), Vec2::new(1.5, 1.0));
        for empty in [point, zero_width, zero_height] {
            assert!(!a.intersects(empty) && !empty.intersects(a), "{empty:?}");
        }
    }

    #[test]
    fn center_of_huge_rect_is_finite() {
        let r = Rect::from_min_max(Vec2::new(2e38, -3e38), Vec2::new(3e38, -2e38));
        assert_eq!(r.center(), Vec2::new(2.5e38, -2.5e38));
    }
}
