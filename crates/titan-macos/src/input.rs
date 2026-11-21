//! macOS keyboard input as Titan [`Event`]s: virtual key codes and modifier-flag diffs.
//!
//! Pure Rust with no AppKit calls, so its tests run on every platform.

use titan_app::{Event, KeyCode};

// Virtual key codes from `HIToolbox/Events.h` (`kVK_*`).
const KVK_ANSI_Q: u16 = 0x0C;
const KVK_SHIFT: u16 = 0x38;
const KVK_RIGHT_SHIFT: u16 = 0x3C;
const KVK_CONTROL: u16 = 0x3B;
const KVK_RIGHT_CONTROL: u16 = 0x3E;
const KVK_OPTION: u16 = 0x3A;
const KVK_RIGHT_OPTION: u16 = 0x3D;
const KVK_COMMAND: u16 = 0x37;
const KVK_RIGHT_COMMAND: u16 = 0x36;

// Device-independent modifier flags from `AppKit/NSEvent.h` (`NSEventModifierFlag*`).
const FLAG_SHIFT: u64 = 1 << 17;
const FLAG_CONTROL: u64 = 1 << 18;
const FLAG_OPTION: u64 = 1 << 19;
pub(crate) const FLAG_COMMAND: u64 = 1 << 20;

// Device-dependent (left/right) modifier bits from `IOKit/hidsystem/IOLLEvent.h`
// (`NX_DEVICE*KEYMASK`), which AppKit passes through in the low 16 bits of `modifierFlags`.
const DEVICE_LCTL: u64 = 0x0000_0001;
const DEVICE_LSHIFT: u64 = 0x0000_0002;
const DEVICE_RSHIFT: u64 = 0x0000_0004;
const DEVICE_LCMD: u64 = 0x0000_0008;
const DEVICE_RCMD: u64 = 0x0000_0010;
const DEVICE_LALT: u64 = 0x0000_0020;
const DEVICE_RALT: u64 = 0x0000_0040;
const DEVICE_RCTL: u64 = 0x0000_2000;

/// Returns the key at macOS virtual key code `key_code` (a `kVK_*` value from
/// `HIToolbox/Events.h`, as `NSEvent.keyCode` reports it), or `None` for keys Titan does not
/// name.
///
/// Key codes are physical positions, so the mapping ignores the keyboard layout: the key left of
/// `S` is [`KeyCode::A`] on QWERTY and AZERTY keyboards alike.
#[must_use]
pub fn map_macos_keycode(key_code: u16) -> Option<KeyCode> {
    Some(match key_code {
        0x00 => KeyCode::A,
        0x01 => KeyCode::S,
        0x02 => KeyCode::D,
        0x03 => KeyCode::F,
        0x04 => KeyCode::H,
        0x05 => KeyCode::G,
        0x06 => KeyCode::Z,
        0x07 => KeyCode::X,
        0x08 => KeyCode::C,
        0x09 => KeyCode::V,
        0x0B => KeyCode::B,
        KVK_ANSI_Q => KeyCode::Q,
        0x0D => KeyCode::W,
        0x0E => KeyCode::E,
        0x0F => KeyCode::R,
        0x10 => KeyCode::Y,
        0x11 => KeyCode::T,
        0x12 => KeyCode::Digit1,
        0x13 => KeyCode::Digit2,
        0x14 => KeyCode::Digit3,
        0x15 => KeyCode::Digit4,
        0x16 => KeyCode::Digit6,
        0x17 => KeyCode::Digit5,
        0x19 => KeyCode::Digit9,
        0x1A => KeyCode::Digit7,
        0x1C => KeyCode::Digit8,
        0x1D => KeyCode::Digit0,
        0x1F => KeyCode::O,
        0x20 => KeyCode::U,
        0x22 => KeyCode::I,
        0x23 => KeyCode::P,
        0x24 => KeyCode::Enter,
        0x25 => KeyCode::L,
        0x26 => KeyCode::J,
        0x28 => KeyCode::K,
        0x2D => KeyCode::N,
        0x2E => KeyCode::M,
        0x30 => KeyCode::Tab,
        0x31 => KeyCode::Space,
        0x33 => KeyCode::Backspace,
        0x35 => KeyCode::Escape,
        KVK_RIGHT_COMMAND => KeyCode::SuperRight,
        KVK_COMMAND => KeyCode::SuperLeft,
        KVK_SHIFT => KeyCode::ShiftLeft,
        KVK_OPTION => KeyCode::AltLeft,
        KVK_CONTROL => KeyCode::ControlLeft,
        KVK_RIGHT_SHIFT => KeyCode::ShiftRight,
        KVK_RIGHT_OPTION => KeyCode::AltRight,
        KVK_RIGHT_CONTROL => KeyCode::ControlRight,
        0x60 => KeyCode::F5,
        0x61 => KeyCode::F6,
        0x62 => KeyCode::F7,
        0x63 => KeyCode::F3,
        0x64 => KeyCode::F8,
        0x65 => KeyCode::F9,
        0x67 => KeyCode::F11,
        0x6D => KeyCode::F10,
        0x6F => KeyCode::F12,
        0x76 => KeyCode::F4,
        0x78 => KeyCode::F2,
        0x7A => KeyCode::F1,
        0x7B => KeyCode::ArrowLeft,
        0x7C => KeyCode::ArrowRight,
        0x7D => KeyCode::ArrowDown,
        0x7E => KeyCode::ArrowUp,
        _ => return None,
    })
}

/// Returns whether a key-down of `key_code` with `modifier_flags` is Command-Q, which the window
/// turns into [`Event::CloseRequested`] because the app has no menu to handle it.
#[cfg_attr(
    not(target_os = "macos"),
    allow(
        dead_code,
        reason = "only the macOS window calls it; its tests run everywhere"
    )
)]
pub(crate) fn is_command_q(key_code: u16, modifier_flags: u64) -> bool {
    key_code == KVK_ANSI_Q && modifier_flags & FLAG_COMMAND != 0
}

/// One left or right modifier key and the flag bits that report it.
struct Modifier {
    key: KeyCode,
    key_code: u16,
    /// The device-independent flag both sides share, such as `NSEventModifierFlagShift`.
    class: u64,
    /// This side's device-dependent bit.
    device: u64,
    /// The other side's device-dependent bit.
    other_device: u64,
}

/// Every modifier key; a key's index is its bit in [`ModifierTracker`]'s held set.
const MODIFIERS: [Modifier; 8] = [
    modifier(
        KeyCode::ShiftLeft,
        KVK_SHIFT,
        FLAG_SHIFT,
        DEVICE_LSHIFT,
        DEVICE_RSHIFT,
    ),
    modifier(
        KeyCode::ShiftRight,
        KVK_RIGHT_SHIFT,
        FLAG_SHIFT,
        DEVICE_RSHIFT,
        DEVICE_LSHIFT,
    ),
    modifier(
        KeyCode::ControlLeft,
        KVK_CONTROL,
        FLAG_CONTROL,
        DEVICE_LCTL,
        DEVICE_RCTL,
    ),
    modifier(
        KeyCode::ControlRight,
        KVK_RIGHT_CONTROL,
        FLAG_CONTROL,
        DEVICE_RCTL,
        DEVICE_LCTL,
    ),
    modifier(
        KeyCode::AltLeft,
        KVK_OPTION,
        FLAG_OPTION,
        DEVICE_LALT,
        DEVICE_RALT,
    ),
    modifier(
        KeyCode::AltRight,
        KVK_RIGHT_OPTION,
        FLAG_OPTION,
        DEVICE_RALT,
        DEVICE_LALT,
    ),
    modifier(
        KeyCode::SuperLeft,
        KVK_COMMAND,
        FLAG_COMMAND,
        DEVICE_LCMD,
        DEVICE_RCMD,
    ),
    modifier(
        KeyCode::SuperRight,
        KVK_RIGHT_COMMAND,
        FLAG_COMMAND,
        DEVICE_RCMD,
        DEVICE_LCMD,
    ),
];

const fn modifier(
    key: KeyCode,
    key_code: u16,
    class: u64,
    device: u64,
    other_device: u64,
) -> Modifier {
    Modifier {
        key,
        key_code,
        class,
        device,
        other_device,
    }
}

/// Turns AppKit's `flagsChanged:` events into key presses and releases for the eight modifier
/// keys.
///
/// A `flagsChanged:` event carries only the new modifier flags and the key that changed, so the
/// tracker keeps which modifiers are held and reports the difference. The left/right
/// device-dependent bits decide each side; when an event lacks them (some virtual and remote
/// keyboards), the changed key toggles and a cleared shared flag releases both sides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    not(target_os = "macos"),
    allow(
        dead_code,
        reason = "only the macOS window feeds it; its tests run everywhere"
    )
)]
pub(crate) struct ModifierTracker {
    /// Bit `i` is set while `MODIFIERS[i]` is held.
    held: u8,
}

#[cfg_attr(
    not(target_os = "macos"),
    allow(
        dead_code,
        reason = "only the macOS window feeds it; its tests run everywhere"
    )
)]
impl ModifierTracker {
    /// Applies a `flagsChanged:` event for `key_code` with `modifier_flags` and returns the key
    /// events it implies, in a fixed order.
    pub(crate) fn update(
        &mut self,
        key_code: u16,
        modifier_flags: u64,
    ) -> impl Iterator<Item = Event> + use<> {
        let old = self.held;
        let mut new = 0;
        for (i, m) in MODIFIERS.iter().enumerate() {
            let bit = 1 << i;
            let held = if modifier_flags & m.class == 0 {
                false
            } else if modifier_flags & (m.device | m.other_device) != 0 {
                modifier_flags & m.device != 0
            } else if key_code == m.key_code {
                old & bit == 0
            } else {
                old & bit != 0
            };
            if held {
                new |= bit;
            }
        }
        self.held = new;
        MODIFIERS.iter().enumerate().filter_map(move |(i, m)| {
            let bit = 1 << i;
            match (old & bit != 0, new & bit != 0) {
                (false, true) => Some(Event::KeyDown(m.key)),
                (true, false) => Some(Event::KeyUp(m.key)),
                _ => None,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use titan_app::KeyCode::{AltLeft, AltRight, ShiftLeft, ShiftRight, SuperLeft, SuperRight};

    #[test]
    fn every_key_has_exactly_one_key_code() {
        let mut codes: HashMap<KeyCode, Vec<u16>> = HashMap::new();
        for code in 0..=u16::MAX {
            if let Some(key) = map_macos_keycode(code) {
                codes.entry(key).or_default().push(code);
            }
        }
        for &key in KeyCode::ALL {
            assert_eq!(
                codes.get(&key).map(Vec::len),
                Some(1),
                "{key:?} maps from {:?}",
                codes.get(&key)
            );
        }
        assert_eq!(codes.len(), KeyCode::ALL.len());
        assert_eq!(KeyCode::ALL.len(), 65);
    }

    #[test]
    fn spot_check_key_codes_against_events_h() {
        // A few `kVK_*` values whose order is not alphabetical, so a shifted table shows up.
        assert_eq!(map_macos_keycode(0x00), Some(KeyCode::A));
        assert_eq!(map_macos_keycode(0x06), Some(KeyCode::Z));
        assert_eq!(map_macos_keycode(0x16), Some(KeyCode::Digit6));
        assert_eq!(map_macos_keycode(0x17), Some(KeyCode::Digit5));
        assert_eq!(map_macos_keycode(0x24), Some(KeyCode::Enter));
        assert_eq!(map_macos_keycode(0x33), Some(KeyCode::Backspace));
        assert_eq!(map_macos_keycode(0x7A), Some(KeyCode::F1));
        assert_eq!(map_macos_keycode(0x7E), Some(KeyCode::ArrowUp));
        // kVK_ANSI_Equal, kVK_CapsLock, kVK_Function, and kVK_F13 have no Titan key.
        for unnamed in [0x18, 0x39, 0x3F, 0x69] {
            assert_eq!(map_macos_keycode(unnamed), None);
        }
    }

    #[test]
    fn command_q_needs_both_the_key_and_the_flag() {
        assert!(is_command_q(KVK_ANSI_Q, FLAG_COMMAND | DEVICE_LCMD));
        assert!(!is_command_q(KVK_ANSI_Q, FLAG_CONTROL));
        assert!(!is_command_q(0x0D, FLAG_COMMAND));
    }

    fn update(tracker: &mut ModifierTracker, key_code: u16, flags: u64) -> Vec<Event> {
        tracker.update(key_code, flags).collect()
    }

    #[test]
    fn both_sides_of_a_modifier_are_tracked_independently() {
        let mut t = ModifierTracker::default();
        let shift = FLAG_SHIFT;
        assert_eq!(
            update(&mut t, KVK_SHIFT, shift | DEVICE_LSHIFT),
            [Event::KeyDown(ShiftLeft)]
        );
        assert_eq!(
            update(
                &mut t,
                KVK_RIGHT_SHIFT,
                shift | DEVICE_LSHIFT | DEVICE_RSHIFT
            ),
            [Event::KeyDown(ShiftRight)]
        );
        // Releasing the left side keeps the shared flag set; only the device bit says which.
        assert_eq!(
            update(&mut t, KVK_SHIFT, shift | DEVICE_RSHIFT),
            [Event::KeyUp(ShiftLeft)]
        );
        assert_eq!(
            update(&mut t, KVK_RIGHT_SHIFT, 0),
            [Event::KeyUp(ShiftRight)]
        );
        assert_eq!(t, ModifierTracker::default());
    }

    #[test]
    fn every_modifier_presses_and_releases_its_own_key() {
        for m in &MODIFIERS {
            let mut t = ModifierTracker::default();
            assert_eq!(
                update(&mut t, m.key_code, m.class | m.device),
                [Event::KeyDown(m.key)]
            );
            assert_eq!(update(&mut t, m.key_code, 0), [Event::KeyUp(m.key)]);
        }
    }

    #[test]
    fn one_event_can_change_several_modifiers() {
        let mut t = ModifierTracker::default();
        update(&mut t, KVK_COMMAND, FLAG_COMMAND | DEVICE_LCMD);
        // Focus came back with Option held and Command already released.
        assert_eq!(
            update(&mut t, KVK_RIGHT_OPTION, FLAG_OPTION | DEVICE_RALT),
            [Event::KeyDown(AltRight), Event::KeyUp(SuperLeft)]
        );
    }

    #[test]
    fn without_device_bits_the_changed_key_toggles() {
        let mut t = ModifierTracker::default();
        assert_eq!(
            update(&mut t, KVK_OPTION, FLAG_OPTION),
            [Event::KeyDown(AltLeft)]
        );
        assert_eq!(
            update(&mut t, KVK_RIGHT_OPTION, FLAG_OPTION),
            [Event::KeyDown(AltRight)]
        );
        assert_eq!(
            update(&mut t, KVK_OPTION, FLAG_OPTION),
            [Event::KeyUp(AltLeft)]
        );
        // A cleared shared flag releases whatever is still held.
        assert_eq!(update(&mut t, KVK_COMMAND, 0), [Event::KeyUp(AltRight)]);
    }

    #[test]
    fn unrelated_flags_and_keys_change_nothing() {
        let mut t = ModifierTracker::default();
        // Caps Lock and the function key report flagsChanged too.
        assert_eq!(update(&mut t, 0x39, 1 << 16), []);
        assert_eq!(update(&mut t, 0x3F, 1 << 23), []);
        update(&mut t, KVK_RIGHT_COMMAND, FLAG_COMMAND | DEVICE_RCMD);
        // A repeated identical event is not a second press.
        assert_eq!(
            update(&mut t, KVK_RIGHT_COMMAND, FLAG_COMMAND | DEVICE_RCMD),
            []
        );
        assert_eq!(
            update(&mut t, KVK_RIGHT_COMMAND, 0),
            [Event::KeyUp(SuperRight)]
        );
    }
}
