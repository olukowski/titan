//! The Objective-C runtime: message sends, owned object pointers, autorelease pools, strings,
//! and class registration. Every raw call into `libobjc` lives here.
//!
//! Messages are sent through [`msg_send!`], which casts `objc_msgSend` to the exact C signature
//! of the method; it is never called as a variadic function, whose ABI differs on arm64.

use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, Ordering};

/// An Objective-C object pointer (`id`).
pub(crate) type Id = *mut c_void;
/// A selector (`SEL`).
pub(crate) type Sel = *const c_void;
/// A class (`Class`).
pub(crate) type Class = *const c_void;
/// `BOOL`: `bool` on arm64 and `signed char` on x86-64 (`objc/objc.h`), which differ in how
/// arguments and returns are extended.
#[cfg(target_arch = "aarch64")]
pub(crate) type ObjcBool = bool;
#[cfg(target_arch = "aarch64")]
pub(crate) const NO: ObjcBool = false;
#[cfg(target_arch = "aarch64")]
pub(crate) const YES: ObjcBool = true;
#[cfg(target_arch = "x86_64")]
pub(crate) type ObjcBool = i8;
#[cfg(target_arch = "x86_64")]
pub(crate) const NO: ObjcBool = 0;
#[cfg(target_arch = "x86_64")]
pub(crate) const YES: ObjcBool = 1;
pub(crate) type NSUInteger = usize;
pub(crate) type NSInteger = isize;
/// `CGFloat` is `double` on every 64-bit Apple platform (`CoreGraphics/CGBase.h`).
pub(crate) type CGFloat = f64;

/// `NSPoint` / `CGPoint`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NSPoint {
    pub(crate) x: CGFloat,
    pub(crate) y: CGFloat,
}

/// `NSSize` / `CGSize`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NSSize {
    pub(crate) width: CGFloat,
    pub(crate) height: CGFloat,
}

/// `NSRect` / `CGRect`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NSRect {
    pub(crate) origin: NSPoint,
    pub(crate) size: NSSize,
}

/// `NSUTF8StringEncoding` (`Foundation/NSString.h`).
const NS_UTF8_STRING_ENCODING: NSUInteger = 4;

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    pub(crate) fn objc_msgSend();
    #[cfg(target_arch = "x86_64")]
    fn objc_msgSend_stret();
    fn objc_msgSendSuper();
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_getClass(name: *const c_char) -> Class;
    fn objc_retain(obj: Id) -> Id;
    fn objc_release(obj: Id);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
    fn objc_allocateClassPair(superclass: Class, name: *const c_char, extra_bytes: usize) -> Class;
    fn objc_registerClassPair(class: Class);
    fn class_addIvar(
        class: Class,
        name: *const c_char,
        size: usize,
        log2_alignment: u8,
        types: *const c_char,
    ) -> ObjcBool;
    fn class_addMethod(
        class: Class,
        name: Sel,
        imp: unsafe extern "C" fn(),
        types: *const c_char,
    ) -> ObjcBool;
    fn class_getInstanceVariable(class: Class, name: *const c_char) -> *mut c_void;
    fn ivar_getOffset(ivar: *mut c_void) -> isize;
}

/// The send function for methods that return a struct larger than 16 bytes, such as `NSRect`:
/// x86-64 returns those through a hidden pointer with `objc_msgSend_stret`, while arm64 has no
/// such variant (`objc/message.h`). `double` returns need no variant on either: x86-64's
/// `objc_msgSend_fpret` is only for `long double`.
#[cfg(target_arch = "x86_64")]
pub(crate) const MSG_SEND_STRET: unsafe extern "C" fn() = objc_msgSend_stret;
#[cfg(target_arch = "aarch64")]
pub(crate) const MSG_SEND_STRET: unsafe extern "C" fn() = objc_msgSend;

/// Sends a message: `msg_send![RetType, receiver, c"selector:with:", arg => ArgType, ...]`.
///
/// Prefix the return type with `stret` for methods returning a struct over 16 bytes. The
/// expansion is an unsafe call: the caller guarantees the receiver is a live object (or nil)
/// that responds to the selector, and that the listed types are exactly the method's C types.
macro_rules! msg_send {
    (@send $imp:expr, $ret:ty, $receiver:expr, $sel:literal $(, $arg:expr => $ty:ty)*) => {{
        let imp = ::core::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn($crate::objc::Id, $crate::objc::Sel $(, $ty)*) -> $ret,
        >($imp);
        imp($crate::objc::Receiver::into_id($receiver), $crate::objc::sel!($sel) $(, $arg)*)
    }};
    (stret $ret:ty, $receiver:expr, $sel:literal $(, $arg:expr => $ty:ty)* $(,)?) => {
        $crate::objc::msg_send!(
            @send $crate::objc::MSG_SEND_STRET, $ret, $receiver, $sel $(, $arg => $ty)*
        )
    };
    ($ret:ty, $receiver:expr, $sel:literal $(, $arg:expr => $ty:ty)* $(,)?) => {
        $crate::objc::msg_send!(
            @send $crate::objc::objc_msgSend as unsafe extern "C" fn(),
            $ret, $receiver, $sel $(, $arg => $ty)*
        )
    };
}
pub(crate) use msg_send;

/// The selector named by a C string literal, registered once per call site.
macro_rules! sel {
    ($name:literal) => {{
        static CACHE: ::std::sync::atomic::AtomicPtr<::std::ffi::c_void> =
            ::std::sync::atomic::AtomicPtr::new(::std::ptr::null_mut());
        $crate::objc::cached_sel(&CACHE, $name)
    }};
}
pub(crate) use sel;

/// The class named by a C string literal, looked up once per call site.
///
/// # Panics
///
/// Panics if no class has that name, which means a framework failed to load.
macro_rules! class {
    ($name:literal) => {{
        static CACHE: ::std::sync::atomic::AtomicPtr<::std::ffi::c_void> =
            ::std::sync::atomic::AtomicPtr::new(::std::ptr::null_mut());
        $crate::objc::cached_class(&CACHE, $name)
    }};
}
pub(crate) use class;

pub(crate) fn cached_sel(cache: &AtomicPtr<c_void>, name: &CStr) -> Sel {
    let cached = cache.load(Ordering::Relaxed);
    if !cached.is_null() {
        return cached;
    }
    // SAFETY: `name` is NUL-terminated. Selectors are immortal and unique per name, so a racing
    // thread stores the same value.
    let sel = unsafe { sel_registerName(name.as_ptr()) };
    cache.store(sel.cast_mut(), Ordering::Relaxed);
    sel
}

pub(crate) fn cached_class(cache: &AtomicPtr<c_void>, name: &CStr) -> Class {
    let cached = cache.load(Ordering::Relaxed);
    if !cached.is_null() {
        return cached;
    }
    // SAFETY: `name` is NUL-terminated. Registered classes are never unloaded, so caching the
    // pointer is sound.
    let class = unsafe { objc_getClass(name.as_ptr()) };
    assert!(!class.is_null(), "Objective-C class {name:?} is not loaded");
    cache.store(class.cast_mut(), Ordering::Relaxed);
    class
}

/// Anything [`msg_send!`] can send to: an object, a class, or an owned object.
pub(crate) trait Receiver {
    fn into_id(self) -> Id;
}

impl Receiver for Id {
    fn into_id(self) -> Id {
        self
    }
}

impl Receiver for Class {
    fn into_id(self) -> Id {
        self.cast_mut()
    }
}

impl Receiver for &Retained {
    fn into_id(self) -> Id {
        self.as_ptr()
    }
}

/// An owned reference to an Objective-C object: one retain count, released on drop.
pub(crate) struct Retained(NonNull<c_void>);

impl Retained {
    /// Takes ownership of a +1 reference, as returned by `alloc`/`init*`, `new*`, `copy*`, and
    /// `*Create*` functions. Returns `None` for nil.
    ///
    /// # Safety
    ///
    /// `ptr` is nil or an object whose +1 reference the caller owns and hands over.
    pub(crate) unsafe fn from_retained(ptr: Id) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    /// Retains a borrowed or autoreleased object. Returns `None` for nil.
    ///
    /// # Safety
    ///
    /// `ptr` is nil or a live object.
    pub(crate) unsafe fn retain(ptr: Id) -> Option<Self> {
        // SAFETY: the caller guarantees `ptr` is a live object; `objc_retain` returns it.
        NonNull::new(ptr)
            .map(|ptr| Self(unsafe { NonNull::new_unchecked(objc_retain(ptr.as_ptr())) }))
    }

    pub(crate) fn as_ptr(&self) -> Id {
        self.0.as_ptr()
    }
}

impl Clone for Retained {
    fn clone(&self) -> Self {
        // SAFETY: `self` keeps the object alive, and `objc_retain` returns the same pointer.
        unsafe { objc_retain(self.as_ptr()) };
        Self(self.0)
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        // SAFETY: `self` owns one retain count, given up exactly once here.
        unsafe { objc_release(self.as_ptr()) };
    }
}

impl fmt::Debug for Retained {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Retained({:p})", self.0)
    }
}

/// Runs `f` inside an autorelease pool, releasing what it autoreleased when it returns or
/// unwinds. Taking a closure keeps pools strictly nested, which `objc_autoreleasePoolPop`
/// requires.
pub(crate) fn autoreleasepool<R>(f: impl FnOnce() -> R) -> R {
    /// Pops the pool on drop, so a panic in `f` still pops it.
    struct Pool {
        token: *mut c_void,
        /// Pools belong to the thread that pushed them.
        _not_send: PhantomData<*mut ()>,
    }
    impl Drop for Pool {
        fn drop(&mut self) {
            // SAFETY: `token` came from the matching push on this thread, and pools nested
            // inside `f` were popped before `f` returned.
            unsafe { objc_autoreleasePoolPop(self.token) };
        }
    }
    let _pool = Pool {
        // SAFETY: pushing a pool has no preconditions.
        token: unsafe { objc_autoreleasePoolPush() },
        _not_send: PhantomData,
    };
    f()
}

/// Creates an `NSString` holding `s`. Embedded NULs are kept.
pub(crate) fn ns_string(s: &str) -> Retained {
    // SAFETY: `NSString` responds to `alloc` and `initWithBytes:length:encoding:`, which copies
    // `s.len()` bytes of valid UTF-8 and returns +1 (nil only on invalid input).
    let ptr = unsafe {
        let string: Id = msg_send![Id, class!(c"NSString"), c"alloc"];
        msg_send![
            Id,
            string,
            c"initWithBytes:length:encoding:",
            s.as_ptr().cast::<c_void>() => *const c_void,
            s.len() => NSUInteger,
            NS_UTF8_STRING_ENCODING => NSUInteger,
        ]
    };
    // SAFETY: `init*` returns a +1 reference we now own.
    unsafe { Retained::from_retained(ptr) }.expect("NSString rejected valid UTF-8")
}

/// Copies an `NSString` into a Rust string; nil becomes the empty string.
///
/// # Safety
///
/// `string` is nil or a live `NSString`.
pub(crate) unsafe fn ns_string_to_rust(string: Id) -> String {
    if string.is_null() {
        return String::new();
    }
    // SAFETY: `string` is a live `NSString` (caller contract). `UTF8String` returns a
    // NUL-terminated buffer that lives at least as long as the innermost autorelease pool, and
    // it is copied before returning.
    unsafe {
        let utf8: *const c_char = msg_send![*const c_char, string, c"UTF8String"];
        if utf8.is_null() {
            return String::new();
        }
        CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}

/// Returns an `NSError`'s `localizedDescription`; nil becomes `"unknown error"`.
///
/// # Safety
///
/// `error` is nil or a live `NSError`.
pub(crate) unsafe fn ns_error_description(error: Id) -> String {
    if error.is_null() {
        return String::from("unknown error");
    }
    // SAFETY: `error` is a live `NSError` (caller contract), whose `localizedDescription` is a
    // nonnull `NSString`.
    unsafe {
        let description: Id = msg_send![Id, error, c"localizedDescription"];
        ns_string_to_rust(description)
    }
}

/// The type encoding of `- (BOOL)method` and `- (BOOL)method:(id)arg`: `BOOL` is `signed char`
/// (`c`) on x86-64 and `bool` (`B`) on arm64 (`objc/objc.h`).
#[cfg(target_arch = "x86_64")]
pub(crate) const BOOL_METHOD_TYPES: [&CStr; 2] = [c"c@:", c"c@:@"];
#[cfg(target_arch = "aarch64")]
pub(crate) const BOOL_METHOD_TYPES: [&CStr; 2] = [c"B@:", c"B@:@"];

/// `struct objc_super` (`objc/message.h`): where `objc_msgSendSuper` starts its method lookup.
#[repr(C)]
struct ObjcSuper {
    receiver: Id,
    super_class: Class,
}

/// Sends `- (void)sel:(id)arg` to `receiver`'s superclass implementation in `super_class`.
///
/// # Safety
///
/// `receiver` is a live instance of a subclass of `super_class`, which implements `sel` with
/// that signature, and `arg` is valid for it.
pub(crate) unsafe fn send_super_void(receiver: Id, super_class: Class, sel: Sel, arg: Id) {
    let target = ObjcSuper {
        receiver,
        super_class,
    };
    // SAFETY: `objc_msgSendSuper` takes the `objc_super` pointer in place of the receiver and
    // then the method's own arguments; the caller guarantees the method's signature.
    unsafe {
        let imp = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(*const ObjcSuper, Sel, Id),
        >(objc_msgSendSuper);
        imp(&raw const target, sel, arg);
    }
}

/// Builds and registers a new Objective-C class with one pointer-sized ivar for Rust state.
#[derive(Debug)]
pub(crate) struct ClassBuilder {
    class: Class,
}

impl ClassBuilder {
    /// The name of the ivar [`ivar_offset`] finds.
    pub(crate) const STATE_IVAR: &CStr = c"titanState";

    /// Starts a subclass of `superclass` named `name`, with a `void *` ivar named
    /// [`STATE_IVAR`](Self::STATE_IVAR).
    ///
    /// # Panics
    ///
    /// Panics if a class named `name` already exists. Callers register each class once.
    pub(crate) fn new(name: &CStr, superclass: Class) -> Self {
        // SAFETY: `superclass` is a registered class and `name` is NUL-terminated;
        // `objc_allocateClassPair` returns nil if the name is taken, which is checked.
        let class = unsafe { objc_allocateClassPair(superclass, name.as_ptr(), 0) };
        assert!(
            !class.is_null(),
            "Objective-C class {name:?} already exists"
        );
        let size = size_of::<*mut c_void>();
        // Lossless: log2 of a pointer's alignment is at most 3.
        let log2_alignment = align_of::<*mut c_void>().trailing_zeros() as u8;
        // SAFETY: the class is allocated but not yet registered, the only time ivars can be
        // added; `^v` is the encoding of `void *`, matching `size` and alignment.
        let added = unsafe {
            class_addIvar(
                class,
                Self::STATE_IVAR.as_ptr(),
                size,
                log2_alignment,
                c"^v".as_ptr(),
            )
        };
        assert_eq!(added, YES, "class_addIvar failed for {name:?}");
        Self { class }
    }

    /// Adds or overrides the method `sel` with the C function at `imp`, whose Objective-C type
    /// encoding is `types` (for example `c"v@:@"` for `- (void)method:(id)arg`).
    ///
    /// # Safety
    ///
    /// `imp` points to an `extern "C"` function whose signature is exactly
    /// `(Id, Sel, args...) -> ret` as `types` describes, and it is sound to call with any
    /// receiver of this class.
    pub(crate) unsafe fn add_method(&mut self, sel: Sel, imp: *const c_void, types: &CStr) {
        // SAFETY: the class is allocated and unregistered; the caller vouches for `imp`, and
        // the runtime stores it as an untyped `IMP`, calling it back with its real signature.
        let added = unsafe {
            let imp = std::mem::transmute::<*const c_void, unsafe extern "C" fn()>(imp);
            class_addMethod(self.class, sel, imp, types.as_ptr())
        };
        assert_eq!(added, YES, "class_addMethod failed");
    }

    /// Registers the class and returns it.
    pub(crate) fn register(self) -> Class {
        // SAFETY: the class was allocated by `objc_allocateClassPair` and is registered once.
        unsafe { objc_registerClassPair(self.class) };
        self.class
    }
}

/// Returns the byte offset of [`ClassBuilder::STATE_IVAR`] in instances of `class`.
///
/// # Panics
///
/// Panics if `class` was not built by [`ClassBuilder`].
pub(crate) fn ivar_offset(class: Class) -> isize {
    // SAFETY: `class` is a registered class and the name is NUL-terminated; a missing ivar
    // returns null, which is checked.
    let ivar = unsafe { class_getInstanceVariable(class, ClassBuilder::STATE_IVAR.as_ptr()) };
    assert!(!ivar.is_null(), "class has no titanState ivar");
    // SAFETY: `ivar` is a valid ivar of `class`.
    unsafe { ivar_getOffset(ivar) }
}

/// Returns a pointer to the state ivar of `object`, which lies `offset` bytes into it.
///
/// # Safety
///
/// `object` is a live instance of a class whose state ivar is at `offset` (from
/// [`ivar_offset`]).
pub(crate) unsafe fn state_ivar(object: Id, offset: isize) -> *mut *mut c_void {
    // SAFETY: the ivar lies inside the object's allocation at `offset` (caller contract).
    unsafe { object.cast::<u8>().offset(offset).cast() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;
    use std::slice;

    #[test]
    fn strings_round_trip_through_nsstring_with_nuls_and_unicode() {
        autoreleasepool(|| {
            for s in ["", "Titan", "naïve 🚀", "a\0b"] {
                let ns = ns_string(s);
                // SAFETY: `ns` is a live NSString; `length` returns UTF-16 units.
                let len: NSUInteger = unsafe { msg_send![NSUInteger, &ns, c"length"] };
                assert_eq!(len, s.encode_utf16().count());
                // SAFETY: `dataUsingEncoding:` returns an autoreleased `NSData` (kept alive by
                // the pool) holding the exact UTF-8 bytes, embedded NULs included; `bytes`
                // points to `length` readable bytes, but may be null when `length` is zero.
                let round_trip = unsafe {
                    let data: Id = msg_send![
                        Id,
                        &ns,
                        c"dataUsingEncoding:",
                        NS_UTF8_STRING_ENCODING => NSUInteger,
                    ];
                    let bytes: *const u8 = msg_send![*const u8, data, c"bytes"];
                    let len: NSUInteger = msg_send![NSUInteger, data, c"length"];
                    let utf8 = if len == 0 {
                        &[]
                    } else {
                        slice::from_raw_parts(bytes, len)
                    };
                    String::from_utf8_lossy(utf8).into_owned()
                };
                assert_eq!(round_trip, s);
                if !s.contains('\0') {
                    // SAFETY: `ns` is a live NSString.
                    assert_eq!(unsafe { ns_string_to_rust(ns.as_ptr()) }, s);
                }
            }
            // SAFETY: nil is allowed.
            assert_eq!(unsafe { ns_string_to_rust(null_mut()) }, "");
            // SAFETY: nil is allowed.
            assert_eq!(unsafe { ns_error_description(null_mut()) }, "unknown error");
        });
    }

    #[test]
    fn retained_clone_and_drop_balance_the_retain_count() {
        autoreleasepool(|| {
            // A plain `NSObject`: short strings can be tagged pointers, which ignore retains.
            // SAFETY: `new` returns a +1 `NSObject`.
            let object =
                unsafe { Retained::from_retained(msg_send![Id, class!(c"NSObject"), c"new"]) }
                    .unwrap();
            // SAFETY: `retainCount` is a plain getter on a live object.
            let count = |o: &Retained| unsafe { msg_send![NSUInteger, o, c"retainCount"] };
            let base = count(&object);
            let copy = object.clone();
            assert_eq!(count(&object), base + 1);
            drop(copy);
            assert_eq!(count(&object), base);
            // SAFETY: `object` is live.
            let extra = unsafe { Retained::retain(object.as_ptr()) }.unwrap();
            assert_eq!(count(&object), base + 1);
            drop(extra);
            assert_eq!(count(&object), base);
            // SAFETY: nil is allowed.
            assert!(unsafe { Retained::retain(null_mut()) }.is_none());
        });
    }
}
