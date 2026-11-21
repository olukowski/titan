//! Fixed-timestep simulation clock.

/// Turns variable frame times into a whole number of fixed simulation steps.
///
/// Each frame, call [`advance`](Self::advance) with the elapsed time and run that many fixed
/// updates; then render with [`alpha`](Self::alpha) to interpolate between the last two states.
/// Clamping each frame to `max_frame_dt` bounds the catch-up work after a hitch, so a slow frame
/// cannot snowball into ever more steps (the "spiral of death").
#[derive(Clone, Debug)]
pub struct FixedClock {
    fixed_dt: f32,
    max_frame_dt: f32,
    /// Unsimulated time, always in `[0, fixed_dt)` between calls. `f64` keeps exact multiples of
    /// `fixed_dt` exact and cannot overflow, even with `fixed_dt = f32::MAX`.
    accumulator: f64,
    tick: u64,
}

impl FixedClock {
    /// Creates a clock that steps every `fixed_dt` seconds and counts at most `max_frame_dt`
    /// seconds per frame.
    ///
    /// # Panics
    ///
    /// Panics unless `fixed_dt` is finite and positive and `max_frame_dt` is finite and in
    /// `[fixed_dt, 65536 · fixed_dt]`. The upper bound caps one frame at 65 536 steps.
    #[must_use]
    pub fn new(fixed_dt: f32, max_frame_dt: f32) -> Self {
        assert!(
            fixed_dt.is_finite() && fixed_dt > 0.0,
            "fixed_dt must be finite and positive"
        );
        assert!(
            max_frame_dt.is_finite() && max_frame_dt >= fixed_dt,
            "max_frame_dt must be finite and at least fixed_dt"
        );
        assert!(
            max_frame_dt <= fixed_dt * 65536.0,
            "max_frame_dt allows too many steps per frame"
        );
        Self {
            fixed_dt,
            max_frame_dt,
            accumulator: 0.0,
            tick: 0,
        }
    }

    /// Adds a frame's elapsed time and returns how many fixed steps to run now.
    ///
    /// Frame times above `max_frame_dt` are clamped. Zero, negative, NaN, and infinite frame
    /// times are ignored and return 0, so a bad timer reading cannot corrupt the clock.
    pub fn advance(&mut self, frame_dt: f32) -> u32 {
        if !(frame_dt.is_finite() && frame_dt > 0.0) {
            return 0;
        }
        let dt = f64::from(self.fixed_dt);
        let frame = f64::from(frame_dt.min(self.max_frame_dt));
        // Split the frame into whole steps before adding the carried fraction: adding first can
        // round a sub-step deficit away at large frame times. With `f32` operands the quotient
        // cannot round up to an integer, `as` truncation is floor, and `steps · dt` (17 × 24
        // bits) and the subtraction are exact, so `rem` is in `[0, dt)`.
        let mut steps = (frame / dt) as u32;
        let rem = frame - f64::from(steps) * dt;
        let room = dt - self.accumulator;
        if rem >= room {
            steps += 1;
            self.accumulator = rem - room;
        } else {
            // The true sum is below `dt`, but it can round up to it.
            self.accumulator = (self.accumulator + rem).min(dt.next_down());
        }
        self.tick += u64::from(steps);
        steps
    }

    /// Returns how far the clock is between the last step and the next, in `[0, 1)`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        // `accumulator < fixed_dt`, but narrowing the quotient to `f32` can round it up to 1.
        ((self.accumulator / f64::from(self.fixed_dt)) as f32).min(1.0f32.next_down())
    }

    /// Returns the simulation step length in seconds.
    #[must_use]
    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }

    /// Returns the number of steps taken since creation or the last [`reset`](Self::reset).
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Discards accumulated time and sets the tick count back to zero.
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_step_frames_accumulate() {
        let mut clock = FixedClock::new(0.25, 1.0);
        assert_eq!(clock.advance(0.125), 0);
        assert_eq!(clock.alpha(), 0.5);
        assert_eq!(clock.advance(0.0625), 0);
        assert_eq!(clock.alpha(), 0.75);
        assert_eq!(clock.advance(0.125), 1);
        assert_eq!((clock.tick(), clock.alpha()), (1, 0.25));
    }

    #[test]
    fn long_frames_run_several_steps() {
        let mut clock = FixedClock::new(0.25, 1.0);
        assert_eq!(clock.advance(0.875), 3);
        assert_eq!((clock.tick(), clock.alpha()), (3, 0.5));
        assert_eq!(clock.advance(0.5), 2);
        assert_eq!((clock.tick(), clock.alpha()), (5, 0.5));
    }

    #[test]
    fn spikes_are_clamped_to_max_frame_dt() {
        let mut clock = FixedClock::new(0.25, 1.0);
        assert_eq!(clock.advance(3600.0), 4);
        assert_eq!(clock.alpha(), 0.0);
        // The excess was dropped, not deferred to later frames.
        assert_eq!(clock.advance(0.125), 0);
        assert_eq!(clock.tick(), 4);
    }

    #[test]
    fn invalid_frame_times_are_ignored() {
        let mut clock = FixedClock::new(0.25, 1.0);
        clock.advance(0.125);
        for bad in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(clock.advance(bad), 0);
        }
        assert_eq!((clock.tick(), clock.alpha()), (0, 0.5));
        assert_eq!(clock.advance(0.125), 1);
    }

    #[test]
    fn real_frame_rates_keep_alpha_below_one() {
        let mut clock = FixedClock::new(1.0 / 60.0, 0.25);
        let mut steps = 0;
        for frame in 0..10_000 {
            // Alternate 144 Hz and 30 Hz frames.
            steps += clock.advance(if frame % 2 == 0 {
                1.0 / 144.0
            } else {
                1.0 / 30.0
            });
            assert!(
                (0.0..1.0).contains(&clock.alpha()),
                "alpha = {}",
                clock.alpha()
            );
        }
        assert_eq!(u64::from(steps), clock.tick());
        clock.reset();
        assert_eq!((clock.tick(), clock.alpha()), (0, 0.0));
    }

    #[test]
    fn exact_multiples_step_exactly() {
        let mut clock = FixedClock::new(0.001, 65536.0 * 0.001);
        assert_eq!(clock.advance(4.0 * 0.001), 4);
        assert_eq!(clock.alpha(), 0.0);
        assert_eq!(clock.advance(65536.0 * 0.001), 65536);
        assert_eq!(clock.alpha(), 0.0);
    }

    #[test]
    fn alpha_stays_below_one_just_short_of_a_step() {
        let mut clock = FixedClock::new(1.0, 1.0);
        assert_eq!(clock.advance(0.5), 0);
        // The sum 1 - 2^-25 is short of a step but rounds to 1.0 as an f32.
        assert_eq!(clock.advance(0.5f32.next_down()), 0);
        assert_eq!(clock.alpha(), 1.0f32.next_down());
    }

    #[test]
    fn carried_deficit_survives_a_max_frame() {
        let mut clock = FixedClock::new(1.0, 65536.0);
        // Leaves 1 - 2^-48 s, short of a step by less than an ulp of 65536.
        assert_eq!(clock.advance(1.0f32.next_down()), 0);
        assert_eq!(clock.advance(2.0f32.powi(-24).next_down()), 0);
        assert_eq!(clock.advance(65536.0), 65536);
        assert_eq!((clock.tick(), clock.alpha()), (65536, 1.0f32.next_down()));
    }

    #[test]
    fn huge_steps_do_not_overflow() {
        let mut clock = FixedClock::new(f32::MAX, f32::MAX);
        assert_eq!(clock.advance(f32::MAX / 2.0), 0);
        assert_eq!(clock.advance(f32::MAX), 1);
        assert_eq!((clock.tick(), clock.alpha()), (1, 0.5));
    }

    #[test]
    #[should_panic(expected = "at least fixed_dt")]
    fn max_frame_dt_below_fixed_dt_panics() {
        let _ = FixedClock::new(0.1, 0.05);
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn nan_fixed_dt_panics() {
        let _ = FixedClock::new(f32::NAN, 1.0);
    }
}
