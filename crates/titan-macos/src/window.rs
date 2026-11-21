//! The AppKit window: an `NSWindow` whose content view hosts a `CAMetalLayer`, and the two
//! Objective-C classes that turn AppKit callbacks into Titan [`Event`]s.

use crate::input::{ModifierTracker, is_command_q, map_macos_keycode};
use crate::objc::{
    BOOL_METHOD_TYPES, Class, ClassBuilder, Id, NO, NSPoint, NSRect, NSSize, NSUInteger, ObjcBool,
    Retained, Sel, YES, class, ivar_offset, msg_send, ns_string, sel, state_ivar,
};
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, c_void};
use std::ptr::{NonNull, null_mut};
use std::sync::Once;
use std::sync::atomic::{AtomicIsize, Ordering};
use titan_app::{Event, MouseButton, SurfaceSize};
use titan_gpu::GpuError;

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {}

// `CAMetalLayer` lives in QuartzCore.
#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {}

// `NSWindowStyleMask` (AppKit/NSWindow.h).
const NS_WINDOW_STYLE_MASK_TITLED: NSUInteger = 1 << 0;
const NS_WINDOW_STYLE_MASK_CLOSABLE: NSUInteger = 1 << 1;
const NS_WINDOW_STYLE_MASK_MINIATURIZABLE: NSUInteger = 1 << 2;
const NS_WINDOW_STYLE_MASK_RESIZABLE: NSUInteger = 1 << 3;
// `NSBackingStoreBuffered` (AppKit/NSGraphics.h).
const NS_BACKING_STORE_BUFFERED: NSUInteger = 2;
// `NSWindowOcclusionStateVisible` (AppKit/NSWindow.h).
const NS_WINDOW_OCCLUSION_STATE_VISIBLE: NSUInteger = 1 << 1;
/// `-[NSEvent buttonNumber]` of the middle mouse button.
const MIDDLE_BUTTON_NUMBER: isize = 2;
/// Points per line for wheels that scroll by lines (`hasPreciseScrollingDeltas == NO`).
const POINTS_PER_SCROLL_LINE: f64 = 10.0;

/// What the window queues for the runner, in the order AppKit delivered it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum WindowEvent {
    Input(Event),
    /// The window stopped being key: key and button releases now go to another window or app.
    FocusLost,
}

/// What the window's Objective-C objects share with Rust: the events they queue and what they
/// need to produce them. The window owns it; both objects point to it through their
/// `titanState` ivar until the window is dropped.
#[derive(Debug)]
struct WindowState {
    events: RefCell<Vec<WindowEvent>>,
    modifiers: Cell<ModifierTracker>,
    /// The last size reported, so repeated notifications send one resize.
    size: Cell<SurfaceSize>,
    /// Borrowed from [`Window`], which outlives every callback that reads them.
    window: Id,
    view: Id,
    layer: Id,
}

impl WindowState {
    fn push(&self, event: Event) {
        self.events.borrow_mut().push(WindowEvent::Input(event));
    }

    /// Reads the view's size in backing pixels and the scale factor, updates the layer's scale,
    /// and queues an [`Event::Resize`] if either changed.
    fn update_size(&self) {
        // SAFETY: `window`, `view`, and `layer` are live while callbacks run (see the field
        // docs); `contentsScale` is a `CGFloat` property.
        let size = unsafe {
            let size = surface_size(self.window, self.view);
            msg_send![
                (),
                self.layer,
                c"setContentsScale:",
                f64::from(size.scale_factor) => f64,
            ];
            size
        };
        if size != self.size.get() {
            self.size.set(size);
            self.push(Event::Resize(size));
        }
    }
}

/// An open window that renders through a `CAMetalLayer` and queues input as [`Event`]s.
#[derive(Debug)]
pub(crate) struct Window {
    window: Retained,
    view: Retained,
    delegate: Retained,
    layer: Retained,
    /// Owned; freed in `Drop` only after both ivars that point to it are cleared.
    state: NonNull<WindowState>,
}

impl Window {
    /// Opens a centered window `width` x `height` points large, titled `title`, and makes it
    /// key.
    pub(crate) fn new(
        title: &str,
        width: u32,
        height: u32,
        resizable: bool,
    ) -> Result<Self, GpuError> {
        let classes = classes();
        let mut style = NS_WINDOW_STYLE_MASK_TITLED
            | NS_WINDOW_STYLE_MASK_CLOSABLE
            | NS_WINDOW_STYLE_MASK_MINIATURIZABLE;
        if resizable {
            style |= NS_WINDOW_STYLE_MASK_RESIZABLE;
        }
        let rect = NSRect {
            origin: NSPoint::default(),
            size: NSSize {
                width: f64::from(width),
                height: f64::from(height),
            },
        };
        // SAFETY: called on the main thread (checked by the runner). Each `alloc`/`init*` and
        // `new` returns +1, `[CAMetalLayer layer]` is autoreleased and retained here, and every
        // message is sent with its declared C types.
        let (window, view, layer, delegate) = unsafe {
            let window = Retained::from_retained(msg_send![
                Id,
                msg_send![Id, class!(c"NSWindow"), c"alloc"],
                c"initWithContentRect:styleMask:backing:defer:",
                rect => NSRect,
                style => NSUInteger,
                NS_BACKING_STORE_BUFFERED => NSUInteger,
                NO => ObjcBool,
            ])
            .ok_or_else(|| GpuError::Backend(String::from("NSWindow failed to initialize")))?;
            let view = Retained::from_retained(msg_send![
                Id,
                msg_send![Id, classes.view, c"alloc"],
                c"initWithFrame:",
                rect => NSRect,
            ])
            .expect("NSView initWithFrame: returned nil");
            let layer = Retained::retain(msg_send![Id, class!(c"CAMetalLayer"), c"layer"])
                .expect("CAMetalLayer layer returned nil");
            let delegate = Retained::from_retained(msg_send![Id, classes.delegate, c"new"])
                .expect("TitanWindowDelegate new returned nil");

            msg_send![(), &window, c"setReleasedWhenClosed:", NO => ObjcBool];
            msg_send![(), &window, c"setTitle:", ns_string(title).as_ptr() => Id];
            msg_send![(), &window, c"setAcceptsMouseMovedEvents:", YES => ObjcBool];
            // A layer-hosting view: set the layer first, then ask for it.
            msg_send![(), &view, c"setLayer:", layer.as_ptr() => Id];
            msg_send![(), &view, c"setWantsLayer:", YES => ObjcBool];
            msg_send![(), &window, c"setContentView:", view.as_ptr() => Id];
            let _: ObjcBool =
                msg_send![ObjcBool, &window, c"makeFirstResponder:", view.as_ptr() => Id];
            (window, view, layer, delegate)
        };

        // SAFETY: the window and view are live and fully set up.
        let size = unsafe { surface_size(window.as_ptr(), view.as_ptr()) };
        let state = NonNull::from(Box::leak(Box::new(WindowState {
            events: RefCell::new(Vec::new()),
            modifiers: Cell::default(),
            size: Cell::new(size),
            window: window.as_ptr(),
            view: view.as_ptr(),
            layer: layer.as_ptr(),
        })));
        // SAFETY: the view and delegate are live instances of the classes registered by
        // `classes()`, so their state ivars are at the stored offsets; `state` stays valid until
        // `Drop` clears both ivars. Then the window, now able to call back, is shown.
        unsafe {
            *state_ivar(view.as_ptr(), classes.view_offset) = state.as_ptr().cast::<c_void>();
            *state_ivar(delegate.as_ptr(), classes.delegate_offset) =
                state.as_ptr().cast::<c_void>();
            msg_send![
                (),
                layer.as_ptr(),
                c"setContentsScale:",
                f64::from(size.scale_factor) => f64,
            ];
            msg_send![(), &window, c"setDelegate:", delegate.as_ptr() => Id];
            msg_send![(), &window, c"center"];
            msg_send![(), &window, c"makeKeyAndOrderFront:", null_mut() => Id];
        }
        Ok(Self {
            window,
            view,
            delegate,
            layer,
            state,
        })
    }

    fn state(&self) -> &WindowState {
        // SAFETY: `state` is a live allocation owned by `self`; callbacks only ever take shared
        // references to it too.
        unsafe { self.state.as_ref() }
    }

    /// Returns the window's `CAMetalLayer`.
    pub(crate) fn layer(&self) -> &Retained {
        &self.layer
    }

    /// Returns the last surface size reported.
    pub(crate) fn surface_size(&self) -> SurfaceSize {
        self.state().size.get()
    }

    /// Moves every queued event, oldest first, onto the end of `out`.
    pub(crate) fn take_events(&self, out: &mut Vec<WindowEvent>) {
        out.append(&mut self.state().events.borrow_mut());
    }

    /// Returns whether any part of the window is visible: not minimized, hidden, or covered.
    pub(crate) fn is_visible(&self) -> bool {
        // SAFETY: `occlusionState` is an `NSWindowOcclusionState` (`NSUInteger`) property.
        let state: NSUInteger = unsafe { msg_send![NSUInteger, &self.window, c"occlusionState"] };
        state & NS_WINDOW_OCCLUSION_STATE_VISIBLE != 0
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        let classes = classes();
        // SAFETY: clearing both ivars first means no AppKit callback can reach `state` after
        // this; detaching the delegate and closing the window stop further callbacks anyway.
        // Only then is the leaked `Box` reclaimed, exactly once.
        unsafe {
            *state_ivar(self.view.as_ptr(), classes.view_offset) = null_mut();
            *state_ivar(self.delegate.as_ptr(), classes.delegate_offset) = null_mut();
            msg_send![(), &self.window, c"setDelegate:", null_mut() => Id];
            msg_send![(), &self.window, c"close"];
            drop(Box::from_raw(self.state.as_ptr()));
        }
    }
}

/// Reads `view`'s size in backing pixels and `window`'s scale factor.
///
/// # Safety
///
/// `window` and `view` are a live `NSWindow` and `NSView`.
unsafe fn surface_size(window: Id, view: Id) -> SurfaceSize {
    // SAFETY: `bounds` and `convertRectToBacking:` return `NSRect`s (sent with the stret
    // variant), and `backingScaleFactor` a `CGFloat`, on live objects (caller contract).
    let (backing, scale) = unsafe {
        let bounds: NSRect = msg_send![stret NSRect, view, c"bounds"];
        let backing: NSRect =
            msg_send![stret NSRect, view, c"convertRectToBacking:", bounds => NSRect];
        let scale: f64 = msg_send![f64, window, c"backingScaleFactor"];
        (backing, scale)
    };
    // `as` saturates; backing sizes are small, nonnegative, and integral up to rounding.
    SurfaceSize::new(
        backing.size.width.round() as u32,
        backing.size.height.round() as u32,
        scale as f32,
    )
}

/// The registered `TitanView` and `TitanWindowDelegate` classes and their state ivar offsets.
#[derive(Debug)]
struct Classes {
    view: Class,
    delegate: Class,
    view_offset: isize,
    delegate_offset: isize,
}

/// Registers the window classes on first use and returns them.
fn classes() -> Classes {
    static REGISTER: Once = Once::new();
    static VIEW_OFFSET: AtomicIsize = AtomicIsize::new(0);
    static DELEGATE_OFFSET: AtomicIsize = AtomicIsize::new(0);
    REGISTER.call_once(|| {
        let [bool_getter, bool_with_object] = BOOL_METHOD_TYPES;
        let mut view = ClassBuilder::new(c"TitanView", class!(c"NSView"));
        let event: &CStr = c"v@:@";
        let view_methods: [(Sel, *const c_void, &CStr); 16] = [
            (
                sel!(c"acceptsFirstResponder"),
                yes as Getter as _,
                bool_getter,
            ),
            (
                sel!(c"acceptsFirstMouse:"),
                yes_for_event as Query as _,
                bool_with_object,
            ),
            (sel!(c"keyDown:"), key_down as Handler as _, event),
            (sel!(c"keyUp:"), key_up as Handler as _, event),
            (sel!(c"flagsChanged:"), flags_changed as Handler as _, event),
            (sel!(c"mouseDown:"), mouse_down as Handler as _, event),
            (sel!(c"mouseUp:"), mouse_up as Handler as _, event),
            (
                sel!(c"rightMouseDown:"),
                right_mouse_down as Handler as _,
                event,
            ),
            (
                sel!(c"rightMouseUp:"),
                right_mouse_up as Handler as _,
                event,
            ),
            (
                sel!(c"otherMouseDown:"),
                other_mouse_down as Handler as _,
                event,
            ),
            (
                sel!(c"otherMouseUp:"),
                other_mouse_up as Handler as _,
                event,
            ),
            (sel!(c"mouseMoved:"), mouse_moved as Handler as _, event),
            (sel!(c"mouseDragged:"), mouse_moved as Handler as _, event),
            (
                sel!(c"rightMouseDragged:"),
                mouse_moved as Handler as _,
                event,
            ),
            (
                sel!(c"otherMouseDragged:"),
                mouse_moved as Handler as _,
                event,
            ),
            (sel!(c"scrollWheel:"), scroll_wheel as Handler as _, event),
        ];
        for (sel, imp, types) in view_methods {
            // SAFETY: each function's signature matches its type encoding, and each only
            // dereferences its receiver's state ivar, which every TitanView has.
            unsafe { view.add_method(sel, imp, types) };
        }
        let view = view.register();

        let mut delegate = ClassBuilder::new(c"TitanWindowDelegate", class!(c"NSObject"));
        // SAFETY: as for the view's methods, with TitanWindowDelegate receivers.
        unsafe {
            delegate.add_method(
                sel!(c"windowShouldClose:"),
                window_should_close as Query as _,
                bool_with_object,
            );
            delegate.add_method(
                sel!(c"windowDidResize:"),
                window_resized as Handler as _,
                event,
            );
            delegate.add_method(
                sel!(c"windowDidChangeBackingProperties:"),
                window_resized as Handler as _,
                event,
            );
            delegate.add_method(
                sel!(c"windowDidResignKey:"),
                window_resigned_key as Handler as _,
                event,
            );
        }
        let delegate = delegate.register();
        VIEW_OFFSET.store(ivar_offset(view), Ordering::Relaxed);
        DELEGATE_OFFSET.store(ivar_offset(delegate), Ordering::Relaxed);
    });
    Classes {
        view: class!(c"TitanView"),
        delegate: class!(c"TitanWindowDelegate"),
        view_offset: VIEW_OFFSET.load(Ordering::Relaxed),
        delegate_offset: DELEGATE_OFFSET.load(Ordering::Relaxed),
    }
}

/// `- (BOOL)method`.
type Getter = extern "C" fn(Id, Sel) -> ObjcBool;
/// `- (BOOL)method:(id)arg`.
type Query = extern "C" fn(Id, Sel, Id) -> ObjcBool;
/// `- (void)method:(id)arg`: event handlers and notifications.
type Handler = extern "C" fn(Id, Sel, Id);

/// Returns the state `object` points to, or `None` once its window is dropped.
///
/// # Safety
///
/// `object` is a live `TitanView` (when `view`) or `TitanWindowDelegate`. The reference must
/// not outlive the current callback.
unsafe fn state<'a>(object: Id, view: bool) -> Option<&'a WindowState> {
    let classes = classes();
    let offset = if view {
        classes.view_offset
    } else {
        classes.delegate_offset
    };
    // SAFETY: the ivar is at `offset` in `object` (caller contract) and holds null or a
    // `WindowState` that outlives the callback (see `Window::drop`).
    unsafe { (*state_ivar(object, offset)).cast::<WindowState>().as_ref() }
}

extern "C" fn yes(_this: Id, _sel: Sel) -> ObjcBool {
    YES
}

extern "C" fn yes_for_event(_this: Id, _sel: Sel, _event: Id) -> ObjcBool {
    YES
}

extern "C" fn key_down(this: Id, _sel: Sel, event: Id) {
    // SAFETY: AppKit sends `keyDown:` to a live TitanView with a live key `NSEvent`, whose
    // `keyCode` is `unsigned short` and `modifierFlags` an `NSUInteger`.
    unsafe {
        let Some(state) = state(this, true) else {
            return;
        };
        let key_code: u16 = msg_send![u16, event, c"keyCode"];
        let flags: NSUInteger = msg_send![NSUInteger, event, c"modifierFlags"];
        // With no menu bar, nothing else handles Command-Q. Not calling `super` also keeps
        // AppKit from beeping at keys no responder takes.
        if is_command_q(key_code, flags as u64) {
            state.push(Event::CloseRequested);
        } else if let Some(key) = map_macos_keycode(key_code) {
            state.push(Event::KeyDown(key));
        }
    }
}

extern "C" fn key_up(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `key_down`.
    unsafe {
        let Some(state) = state(this, true) else {
            return;
        };
        let key_code: u16 = msg_send![u16, event, c"keyCode"];
        if let Some(key) = map_macos_keycode(key_code) {
            state.push(Event::KeyUp(key));
        }
    }
}

extern "C" fn flags_changed(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `key_down`, for a `flagsChanged:` event.
    unsafe {
        let Some(state) = state(this, true) else {
            return;
        };
        let key_code: u16 = msg_send![u16, event, c"keyCode"];
        let flags: NSUInteger = msg_send![NSUInteger, event, c"modifierFlags"];
        let mut modifiers = state.modifiers.get();
        for key_event in modifiers.update(key_code, flags as u64) {
            state.push(key_event);
        }
        state.modifiers.set(modifiers);
    }
}

/// Converts `event`'s location to logical points from the top-left of the view `this`.
///
/// # Safety
///
/// `this` is a live `NSView` and `event` a live mouse `NSEvent`.
unsafe fn pointer_move(this: Id, event: Id) -> Event {
    // SAFETY: `locationInWindow` and `convertPoint:fromView:` return `NSPoint`s (16 bytes, no
    // stret), and `bounds` an `NSRect` (stret), on live objects (caller contract).
    let (point, bounds) = unsafe {
        let in_window: NSPoint = msg_send![NSPoint, event, c"locationInWindow"];
        let point: NSPoint = msg_send![
            NSPoint,
            this,
            c"convertPoint:fromView:",
            in_window => NSPoint,
            null_mut() => Id,
        ];
        let bounds: NSRect = msg_send![stret NSRect, this, c"bounds"];
        (point, bounds)
    };
    // AppKit's y axis points up from the bottom-left corner; Titan's points down from the top.
    Event::PointerMove {
        x: point.x as f32,
        y: (bounds.size.height - point.y) as f32,
    }
}

/// Queues the pointer position, then `button`'s press or release, for a mouse event.
///
/// # Safety
///
/// `this` is a live `TitanView` and `event` a live mouse `NSEvent`.
unsafe fn mouse_button(this: Id, event: Id, button: Option<MouseButton>, down: bool) {
    // SAFETY: the caller's contract is exactly what `state` and `pointer_move` need.
    unsafe {
        let Some(state) = state(this, true) else {
            return;
        };
        state.push(pointer_move(this, event));
        if let Some(button) = button {
            state.push(if down {
                Event::PointerDown { button }
            } else {
                Event::PointerUp { button }
            });
        }
    }
}

/// Returns the button of an `otherMouse*` event: the middle button, or `None` for the rest.
///
/// # Safety
///
/// `event` is a live mouse `NSEvent`.
unsafe fn other_button(event: Id) -> Option<MouseButton> {
    // SAFETY: `buttonNumber` is an `NSInteger` property of a live mouse event.
    let number: isize = unsafe { msg_send![isize, event, c"buttonNumber"] };
    (number == MIDDLE_BUTTON_NUMBER).then_some(MouseButton::Middle)
}

extern "C" fn mouse_down(this: Id, _sel: Sel, event: Id) {
    // SAFETY: AppKit sends mouse events to a live TitanView with a live mouse `NSEvent`.
    unsafe { mouse_button(this, event, Some(MouseButton::Left), true) };
}

extern "C" fn mouse_up(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`.
    unsafe { mouse_button(this, event, Some(MouseButton::Left), false) };
}

extern "C" fn right_mouse_down(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`.
    unsafe { mouse_button(this, event, Some(MouseButton::Right), true) };
}

extern "C" fn right_mouse_up(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`.
    unsafe { mouse_button(this, event, Some(MouseButton::Right), false) };
}

extern "C" fn other_mouse_down(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`.
    unsafe { mouse_button(this, event, other_button(event), true) };
}

extern "C" fn other_mouse_up(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`.
    unsafe { mouse_button(this, event, other_button(event), false) };
}

extern "C" fn mouse_moved(this: Id, _sel: Sel, event: Id) {
    // SAFETY: as in `mouse_down`; `mouseMoved:` and the `*Dragged:` events carry locations.
    unsafe { mouse_button(this, event, None, false) };
}

extern "C" fn scroll_wheel(this: Id, _sel: Sel, event: Id) {
    // SAFETY: AppKit sends `scrollWheel:` to a live TitanView with a live scroll `NSEvent`,
    // whose deltas are `CGFloat`s and `hasPreciseScrollingDeltas` a `BOOL`.
    unsafe {
        let Some(state) = state(this, true) else {
            return;
        };
        let dx: f64 = msg_send![f64, event, c"scrollingDeltaX"];
        let dy: f64 = msg_send![f64, event, c"scrollingDeltaY"];
        let precise: ObjcBool = msg_send![ObjcBool, event, c"hasPreciseScrollingDeltas"];
        let scale = if precise == NO {
            POINTS_PER_SCROLL_LINE
        } else {
            1.0
        };
        state.push(Event::Scroll {
            dx: (dx * scale) as f32,
            dy: (dy * scale) as f32,
        });
    }
}

extern "C" fn window_should_close(this: Id, _sel: Sel, _window: Id) -> ObjcBool {
    // SAFETY: AppKit sends this to the live TitanWindowDelegate of the window.
    if let Some(state) = unsafe { state(this, false) } {
        state.push(Event::CloseRequested);
    }
    // The app decides: it exits from a step that sees the close request.
    NO
}

extern "C" fn window_resized(this: Id, _sel: Sel, _notification: Id) {
    // SAFETY: AppKit sends window notifications to the live TitanWindowDelegate.
    if let Some(state) = unsafe { state(this, false) } {
        state.update_size();
    }
}

extern "C" fn window_resigned_key(this: Id, _sel: Sel, _notification: Id) {
    // SAFETY: AppKit sends window notifications to the live TitanWindowDelegate.
    if let Some(state) = unsafe { state(this, false) } {
        state.modifiers.set(ModifierTracker::default());
        state.events.borrow_mut().push(WindowEvent::FocusLost);
    }
}
