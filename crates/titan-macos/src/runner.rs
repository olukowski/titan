//! The macOS frame loop: [`HeadlessRunner`](titan_app::HeadlessRunner)'s lifecycle, fed by
//! AppKit events and a wall clock, rendering to a [`Window`] through [`MetalGpu`].

use crate::input::FLAG_COMMAND;
use crate::metal::MetalGpu;
use crate::objc::{
    ClassBuilder, Id, NO, NSInteger, NSUInteger, ObjcBool, Sel, YES, autoreleasepool, class,
    msg_send, sel, send_super_void,
};
use crate::window::{Window, WindowEvent};
use std::ffi::c_void;
use std::sync::Once;
use std::time::Instant;
use titan_app::{
    App, AppAction, Event, InputState, KeyCode, MouseButton, SurfaceSize, WindowConfig,
};
use titan_core::FixedClock;
use titan_gpu::{FrameCommands, GpuError};

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    static NSDefaultRunLoopMode: Id;
}

/// The simulation step, as in `HeadlessConfig::default()`.
const FIXED_DT: f32 = 1.0 / 60.0;
/// The longest frame time counted, which bounds the steps per frame.
const MAX_FRAME_DT: f32 = 0.25;
// `NSEventMaskAny` and `NSEventTypeKeyUp` (AppKit/NSEvent.h).
const NS_EVENT_MASK_ANY: NSUInteger = NSUInteger::MAX;
const NS_EVENT_TYPE_KEY_UP: NSUInteger = 11;
// `NSApplicationActivationPolicyRegular` (AppKit/NSRunningApplication.h).
const NS_APPLICATION_ACTIVATION_POLICY_REGULAR: NSInteger = 0;

pub(crate) fn run<A: App>(mut app: A, config: WindowConfig) -> Result<(), GpuError> {
    let WindowConfig {
        title,
        width,
        height,
        resizable,
    } = config;
    autoreleasepool(|| {
        let ns_app = launch()?;
        let window = Window::new(&title, width, height, resizable)?;
        let mut size = window.surface_size();
        let mut gpu = MetalGpu::with_layer(window.layer().clone(), size.width, size.height)?;
        app.init(&mut gpu)?;
        app.resize(&mut gpu, size)?;

        let mut clock = FixedClock::new(FIXED_DT, MAX_FRAME_DT);
        let mut input = InputState::new(size);
        let mut commands = FrameCommands::new();
        let mut events = Vec::new();
        let mut last_frame = Instant::now();
        let mut presented = true;
        loop {
            let visible = window.is_visible();
            // Presenting blocks on the display's refresh, which paces the loop. A frame that
            // presents nothing does not, so then wait for events until the next fixed step.
            let wait_seconds = if visible && size.width > 0 && size.height > 0 && presented {
                0.0
            } else {
                let until_step = f64::from((1.0 - clock.alpha()) * FIXED_DT)
                    - last_frame.elapsed().as_secs_f64();
                until_step.clamp(0.0, f64::from(FIXED_DT))
            };
            autoreleasepool(|| pump_events(ns_app, wait_seconds));
            window.take_events(&mut events);
            for event in events.drain(..) {
                if let Some(new_size) = apply_window_event(&mut input, event) {
                    size = new_size;
                    gpu.resize(size.width, size.height);
                    app.resize(&mut gpu, size)?;
                }
            }
            let now = Instant::now();
            let steps = clock.advance((now - last_frame).as_secs_f32());
            last_frame = now;
            for _ in 0..steps {
                let action = app.step(&input, clock.fixed_dt());
                input.end_step();
                if action == AppAction::Exit {
                    // Report a failure of the last frames, which only `Drop` would see.
                    gpu.wait_idle()?;
                    return Ok(());
                }
            }
            if visible && size.width > 0 && size.height > 0 {
                commands.clear();
                app.render(&mut gpu, &mut commands, clock.alpha())?;
                presented = gpu.submit_frame(commands.view())?;
            }
        }
    })
}

/// Applies one window event to `input` and returns the new surface size if it is a resize.
/// Losing focus releases every held key and button, since their releases go elsewhere.
fn apply_window_event(input: &mut InputState, event: WindowEvent) -> Option<SurfaceSize> {
    match event {
        WindowEvent::Input(event) => {
            input.apply_event(event);
            match event {
                Event::Resize(size) => Some(size),
                _ => None,
            }
        }
        WindowEvent::FocusLost => {
            for &key in KeyCode::ALL {
                if input.is_down(key) {
                    input.apply_event(Event::KeyUp(key));
                }
            }
            for &button in MouseButton::ALL {
                if input.is_button_down(button) {
                    input.apply_event(Event::PointerUp { button });
                }
            }
            None
        }
    }
}

/// Sets up the shared `NSApplication` as a regular, foreground app and returns it.
fn launch() -> Result<Id, GpuError> {
    static LAUNCH: Once = Once::new();
    // SAFETY: `isMainThread` is a `BOOL` class method with no preconditions.
    let main_thread: ObjcBool =
        unsafe { msg_send![ObjcBool, class!(c"NSThread"), c"isMainThread"] };
    if main_thread == NO {
        return Err(GpuError::Backend(String::from(
            "titan_macos::run must be called on the main thread",
        )));
    }
    LAUNCH.call_once(|| {
        let mut builder = ClassBuilder::new(c"TitanApplication", class!(c"NSApplication"));
        let send_event = send_event as extern "C" fn(Id, Sel, Id) as *const c_void;
        // SAFETY: `send_event` has the signature of `- (void)sendEvent:(NSEvent *)event`.
        unsafe { builder.add_method(sel!(c"sendEvent:"), send_event, c"v@:@") };
        builder.register();
    });
    // SAFETY: on the main thread, `sharedApplication` creates `NSApp` (as a TitanApplication
    // unless something created it first) and returns it; the setters take their declared
    // types. `finishLaunching` runs once per process.
    unsafe {
        let ns_app: Id = msg_send![Id, class!(c"TitanApplication"), c"sharedApplication"];
        let _: ObjcBool = msg_send![
            ObjcBool,
            ns_app,
            c"setActivationPolicy:",
            NS_APPLICATION_ACTIVATION_POLICY_REGULAR => NSInteger,
        ];
        static FINISH_LAUNCHING: Once = Once::new();
        FINISH_LAUNCHING.call_once(|| msg_send![(), ns_app, c"finishLaunching"]);
        msg_send![(), ns_app, c"activateIgnoringOtherApps:", YES => ObjcBool];
        Ok(ns_app)
    }
}

/// Dispatches every queued AppKit event, first waiting up to `wait_seconds` for one.
fn pump_events(ns_app: Id, wait_seconds: f64) {
    // SAFETY: on the main thread, `NSDate` factory methods return autoreleased dates (kept by
    // the caller's pool), and `nextEventMatchingMask:untilDate:inMode:dequeue:` returns an
    // autoreleased event or nil; `sendEvent:` dispatches it.
    unsafe {
        let mut until: Id = msg_send![
            Id,
            class!(c"NSDate"),
            c"dateWithTimeIntervalSinceNow:",
            wait_seconds => f64,
        ];
        loop {
            let event: Id = msg_send![
                Id,
                ns_app,
                c"nextEventMatchingMask:untilDate:inMode:dequeue:",
                NS_EVENT_MASK_ANY => NSUInteger,
                until => Id,
                NSDefaultRunLoopMode => Id,
                YES => ObjcBool,
            ];
            if event.is_null() {
                break;
            }
            msg_send![(), ns_app, c"sendEvent:", event => Id];
            until = msg_send![Id, class!(c"NSDate"), c"distantPast"];
        }
    }
}

/// `-[TitanApplication sendEvent:]`: `NSApplication` drops key-up events while Command is held,
/// which would leave keys stuck down, so those go straight to the key window.
extern "C" fn send_event(this: Id, sel: Sel, event: Id) {
    // SAFETY: AppKit calls this on the live shared application with a live `NSEvent`; `type`
    // and `modifierFlags` are `NSUInteger`s, `keyWindow` is nil or a live window, and the
    // superclass `NSApplication` implements `sendEvent:` with this signature.
    unsafe {
        let event_type: NSUInteger = msg_send![NSUInteger, event, c"type"];
        if event_type == NS_EVENT_TYPE_KEY_UP {
            let flags: NSUInteger = msg_send![NSUInteger, event, c"modifierFlags"];
            let key_window: Id = msg_send![Id, this, c"keyWindow"];
            if flags as u64 & FLAG_COMMAND != 0 && !key_window.is_null() {
                msg_send![(), key_window, c"sendEvent:", event => Id];
                return;
            }
        }
        send_super_void(this, class!(c"NSApplication"), sel, event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn losing_focus_releases_held_input_in_event_order() {
        let mut input = InputState::new(SurfaceSize::new(4, 4, 1.0));
        // W was pressed in an earlier step and is still held.
        input.apply_event(Event::KeyDown(KeyCode::W));
        input.end_step();
        let resized = SurfaceSize::new(8, 8, 2.0);
        let resizes: Vec<_> = [
            WindowEvent::Input(Event::KeyDown(KeyCode::ShiftLeft)),
            WindowEvent::Input(Event::PointerDown {
                button: MouseButton::Right,
            }),
            WindowEvent::FocusLost,
            WindowEvent::Input(Event::KeyDown(KeyCode::A)),
            WindowEvent::Input(Event::Resize(resized)),
        ]
        .into_iter()
        .filter_map(|event| apply_window_event(&mut input, event))
        .collect();
        assert_eq!(resizes, [resized]);
        for key in [KeyCode::W, KeyCode::ShiftLeft] {
            assert!(!input.is_down(key) && input.just_released(key), "{key:?}");
        }
        assert!(!input.is_button_down(MouseButton::Right));
        assert!(input.button_just_released(MouseButton::Right));
        assert!(input.is_down(KeyCode::A) && input.just_pressed(KeyCode::A));

        // After focus returns, pressing W again is a fresh press.
        input.end_step();
        apply_window_event(&mut input, WindowEvent::Input(Event::KeyDown(KeyCode::W)));
        assert!(input.just_pressed(KeyCode::W));
    }
}
