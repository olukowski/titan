//! Platform-independent input events and the per-step [`InputState`] built from them.

use crate::window::SurfaceSize;
use titan_core::Vec2;

/// Defines a fieldless `#[repr(u8)]` enum whose variant names double as its text names, with
/// `ALL`, `name`, and `from_name` generated from the one variant list.
macro_rules! named_enum {
    ($(#[$meta:meta])* $Enum:ident, $what:literal { $($Variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $Enum {
            $(#[doc = concat!("The `", stringify!($Variant), "` ", $what, ".")] $Variant,)+
        }

        impl $Enum {
            /// Every variant, in declaration order.
            pub const ALL: &[Self] = &[$(Self::$Variant),+];

            /// Returns the variant's name, as used in event logs.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$Variant => stringify!($Variant),)+
                }
            }

            /// Returns the variant named `name`, the inverse of [`name`](Self::name).
            #[must_use]
            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $(stringify!($Variant) => Some(Self::$Variant),)+
                    _ => None,
                }
            }
        }
    };
}

named_enum! {
    /// A physical key, named after its position on a US keyboard layout.
    KeyCode, "key" {
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9,
        ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
        Space, Enter, Escape, Tab, Backspace,
        ShiftLeft, ShiftRight, ControlLeft, ControlRight, AltLeft, AltRight, SuperLeft, SuperRight,
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    }
}

named_enum! {
    /// A mouse button.
    MouseButton, "mouse button" { Left, Right, Middle }
}

// Every key and button has a bit in a `u128` set.
const _: () = assert!(KeyCode::ALL.len() <= 128 && MouseButton::ALL.len() <= 128);

/// One input event from the platform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A key was pressed. Platforms may repeat this while the key is held.
    KeyDown(KeyCode),
    /// A key was released.
    KeyUp(KeyCode),
    /// The pointer moved to (`x`, `y`) in logical points from the surface's top-left corner.
    PointerMove {
        /// Horizontal position, increasing rightward.
        x: f32,
        /// Vertical position, increasing downward.
        y: f32,
    },
    /// A mouse button was pressed.
    PointerDown {
        /// The button.
        button: MouseButton,
    },
    /// A mouse button was released.
    PointerUp {
        /// The button.
        button: MouseButton,
    },
    /// The wheel or trackpad scrolled by (`dx`, `dy`) logical points.
    Scroll {
        /// Horizontal scroll.
        dx: f32,
        /// Vertical scroll.
        dy: f32,
    },
    /// The surface changed size or scale factor.
    Resize(SurfaceSize),
    /// The user asked to close the window.
    CloseRequested,
}

/// Held state and per-step edges for a set of buttons, one bit per button.
#[derive(Clone, Copy, Debug, Default)]
struct Buttons {
    down: u128,
    just_pressed: u128,
    just_released: u128,
}

impl Buttons {
    /// Platform key repeat sends presses for held buttons; only the first one is an edge.
    fn press(&mut self, bit: u128) {
        if self.down & bit == 0 {
            self.down |= bit;
            self.just_pressed |= bit;
        }
    }

    fn release(&mut self, bit: u128) {
        if self.down & bit != 0 {
            self.down &= !bit;
            self.just_released |= bit;
        }
    }

    fn end_step(&mut self) {
        self.just_pressed = 0;
        self.just_released = 0;
    }
}

fn key_bit(key: KeyCode) -> u128 {
    1 << key as u8
}

fn button_bit(button: MouseButton) -> u128 {
    1 << button as u8
}

/// The input a simulation step sees: what is held, and what changed since the last step.
///
/// Feed it every platform [`Event`] with [`apply_event`](Self::apply_event), and call
/// [`end_step`](Self::end_step) after each fixed simulation step. Edges ("just pressed", "just
/// released") and pointer and scroll deltas cover everything since the previous step, so a press
/// that arrives during a frame with no step is seen by the next step that runs. It never
/// allocates.
#[derive(Clone, Debug)]
pub struct InputState {
    keys: Buttons,
    mouse: Buttons,
    pointer_position: Vec2,
    /// Whether a pointer position has been seen. The first move only places the pointer, so it
    /// does not produce a jump from the origin.
    has_pointer: bool,
    pointer_delta: Vec2,
    scroll_delta: Vec2,
    surface_size: SurfaceSize,
    close_requested: bool,
}

impl InputState {
    /// Creates a state with nothing held, the pointer unplaced at the origin, and the given
    /// surface size.
    #[must_use]
    pub fn new(surface_size: SurfaceSize) -> Self {
        Self {
            keys: Buttons::default(),
            mouse: Buttons::default(),
            pointer_position: Vec2::ZERO,
            has_pointer: false,
            pointer_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            surface_size,
            close_requested: false,
        }
    }

    /// Updates the state with one platform event.
    pub fn apply_event(&mut self, event: Event) {
        match event {
            Event::KeyDown(key) => self.keys.press(key_bit(key)),
            Event::KeyUp(key) => self.keys.release(key_bit(key)),
            Event::PointerDown { button } => self.mouse.press(button_bit(button)),
            Event::PointerUp { button } => self.mouse.release(button_bit(button)),
            Event::PointerMove { x, y } => {
                let position = Vec2::new(x, y);
                if self.has_pointer {
                    self.pointer_delta += position - self.pointer_position;
                }
                self.pointer_position = position;
                self.has_pointer = true;
            }
            Event::Scroll { dx, dy } => self.scroll_delta += Vec2::new(dx, dy),
            Event::Resize(size) => self.surface_size = size,
            Event::CloseRequested => self.close_requested = true,
        }
    }

    /// Ends a simulation step: clears edges and deltas, keeping held buttons, the pointer
    /// position, the surface size, and a close request.
    pub fn end_step(&mut self) {
        self.keys.end_step();
        self.mouse.end_step();
        self.pointer_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// Returns whether `key` is held.
    #[must_use]
    pub fn is_down(&self, key: KeyCode) -> bool {
        self.keys.down & key_bit(key) != 0
    }

    /// Returns whether `key` was pressed since the last step.
    #[must_use]
    pub fn just_pressed(&self, key: KeyCode) -> bool {
        self.keys.just_pressed & key_bit(key) != 0
    }

    /// Returns whether `key` was released since the last step.
    #[must_use]
    pub fn just_released(&self, key: KeyCode) -> bool {
        self.keys.just_released & key_bit(key) != 0
    }

    /// Returns whether `button` is held.
    #[must_use]
    pub fn is_button_down(&self, button: MouseButton) -> bool {
        self.mouse.down & button_bit(button) != 0
    }

    /// Returns whether `button` was pressed since the last step.
    #[must_use]
    pub fn button_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse.just_pressed & button_bit(button) != 0
    }

    /// Returns whether `button` was released since the last step.
    #[must_use]
    pub fn button_just_released(&self, button: MouseButton) -> bool {
        self.mouse.just_released & button_bit(button) != 0
    }

    /// Returns the pointer position in logical points, or the origin before the first move.
    #[must_use]
    pub fn pointer_position(&self) -> Vec2 {
        self.pointer_position
    }

    /// Returns how far the pointer moved since the last step.
    #[must_use]
    pub fn pointer_delta(&self) -> Vec2 {
        self.pointer_delta
    }

    /// Returns how far the wheel or trackpad scrolled since the last step.
    #[must_use]
    pub fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }

    /// Returns the current surface size.
    #[must_use]
    pub fn surface_size(&self) -> SurfaceSize {
        self.surface_size
    }

    /// Returns whether the user has asked to close the window. It stays set; the app decides
    /// whether to exit.
    #[must_use]
    pub fn close_requested(&self) -> bool {
        self.close_requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> InputState {
        InputState::new(SurfaceSize::default())
    }

    #[test]
    fn key_edges_fire_once_per_press() {
        let mut input = state();
        input.apply_event(Event::KeyDown(KeyCode::Space));
        assert!(input.is_down(KeyCode::Space) && input.just_pressed(KeyCode::Space));
        assert!(!input.is_down(KeyCode::A));
        input.end_step();
        assert!(input.is_down(KeyCode::Space) && !input.just_pressed(KeyCode::Space));

        // OS key repeat while held is not a new press.
        input.apply_event(Event::KeyDown(KeyCode::Space));
        assert!(!input.just_pressed(KeyCode::Space));

        input.apply_event(Event::KeyUp(KeyCode::Space));
        assert!(!input.is_down(KeyCode::Space) && input.just_released(KeyCode::Space));
        input.end_step();
        assert!(!input.just_released(KeyCode::Space));

        // Releasing a key that is not held (pressed before focus, say) is not an edge.
        input.apply_event(Event::KeyUp(KeyCode::Escape));
        assert!(!input.just_released(KeyCode::Escape));
    }

    #[test]
    fn tap_within_one_step_reports_both_edges() {
        let mut input = state();
        input.apply_event(Event::KeyDown(KeyCode::F12));
        input.apply_event(Event::KeyUp(KeyCode::F12));
        assert!(input.just_pressed(KeyCode::F12) && input.just_released(KeyCode::F12));
        assert!(!input.is_down(KeyCode::F12));
    }

    #[test]
    fn every_key_has_its_own_bit() {
        let mut input = state();
        for (i, &key) in KeyCode::ALL.iter().enumerate() {
            input.apply_event(Event::KeyDown(key));
            let held = KeyCode::ALL.iter().filter(|&&k| input.is_down(k)).count();
            assert_eq!(held, i + 1, "{key:?}");
        }
    }

    #[test]
    fn mouse_button_edges_match_keys() {
        let mut input = state();
        let left = MouseButton::Left;
        input.apply_event(Event::PointerDown { button: left });
        input.apply_event(Event::PointerDown { button: left });
        assert!(input.is_button_down(left) && input.button_just_pressed(left));
        assert!(!input.is_button_down(MouseButton::Right));
        input.end_step();
        assert!(input.is_button_down(left) && !input.button_just_pressed(left));
        input.apply_event(Event::PointerUp { button: left });
        assert!(!input.is_button_down(left) && input.button_just_released(left));
        input.end_step();
        input.apply_event(Event::PointerUp {
            button: MouseButton::Middle,
        });
        assert!(!input.button_just_released(MouseButton::Middle));
        assert!(!input.button_just_released(left));
    }

    #[test]
    fn pointer_and_scroll_deltas_accumulate_per_step() {
        let mut input = state();
        input.apply_event(Event::PointerMove { x: 100.0, y: 50.0 });
        // Placing the pointer is not motion.
        assert_eq!(input.pointer_delta(), Vec2::ZERO);
        assert_eq!(input.pointer_position(), Vec2::new(100.0, 50.0));
        input.apply_event(Event::PointerMove { x: 110.0, y: 40.0 });
        input.apply_event(Event::PointerMove { x: 115.0, y: 45.0 });
        assert_eq!(input.pointer_delta(), Vec2::new(15.0, -5.0));
        input.apply_event(Event::Scroll { dx: 0.0, dy: -3.0 });
        input.apply_event(Event::Scroll { dx: 1.5, dy: -1.0 });
        assert_eq!(input.scroll_delta(), Vec2::new(1.5, -4.0));

        input.end_step();
        assert_eq!(input.pointer_delta(), Vec2::ZERO);
        assert_eq!(input.scroll_delta(), Vec2::ZERO);
        assert_eq!(input.pointer_position(), Vec2::new(115.0, 45.0));
        input.apply_event(Event::PointerMove { x: 115.0, y: 50.0 });
        assert_eq!(input.pointer_delta(), Vec2::new(0.0, 5.0));
    }

    #[test]
    fn resize_and_close_persist_across_steps() {
        let mut input = state();
        assert!(!input.close_requested());
        let retina = SurfaceSize::new(2560, 1600, 2.0);
        input.apply_event(Event::Resize(retina));
        input.apply_event(Event::CloseRequested);
        input.end_step();
        assert_eq!(input.surface_size(), retina);
        assert!(input.close_requested());
    }

    #[test]
    fn names_round_trip() {
        for &key in KeyCode::ALL {
            assert_eq!(KeyCode::from_name(key.name()), Some(key));
        }
        for &button in MouseButton::ALL {
            assert_eq!(MouseButton::from_name(button.name()), Some(button));
        }
        assert_eq!(KeyCode::ALL.len(), 65);
        assert_eq!(KeyCode::from_name("a"), None);
        assert_eq!(MouseButton::from_name("Back"), None);
    }
}
