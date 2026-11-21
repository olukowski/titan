//! [`export_app!`](crate::export_app): the WASM exports `web/titan.js` calls, and their glue.

/// Exports an [`App`](titan_app::App) to `web/titan.js` from a `cdylib` crate.
///
/// `export_app!(ctor)` runs with [`WebConfig::default`](crate::WebConfig::default);
/// `export_app!(ctor, config)` takes a [`WebConfig`](crate::WebConfig). `ctor` is evaluated once,
/// when the host calls `titan_init`.
///
/// On `wasm32` it defines these exports, which drive one [`WebRunner`](crate::WebRunner):
///
/// | Export | Returns |
/// |---|---|
/// | `titan_init(width: u32, height: u32, scale_factor: f32)` | 0 ok, -1 error |
/// | `titan_frame(dt_secs: f32)` | 0 continue, 1 exit, -1 error |
/// | `titan_resize(width: u32, height: u32, scale_factor: f32)` | 0 ok, -1 error |
/// | `titan_key(code: u32, down: u32)` | |
/// | `titan_pointer_move(x: f32, y: f32)` | |
/// | `titan_pointer_button(button: u32, down: u32)` | |
/// | `titan_scroll(dx: f32, dy: f32)` | |
/// | `titan_audio_mix(frames: u32)` | pointer to `2 * frames` interleaved stereo `f32`s, valid until the next call; null on error |
///
/// An export that returns -1 or null first logs the error through the host's `log` import, so
/// the page can show it. `titan_audio_mix` fails when no `AudioOutput` is playing or `frames`
/// is over 65536. On other targets the macro only type-checks `ctor` and `config`.
///
/// ```ignore
/// titan_wasm::export_app!(MyGame::new());
/// ```
#[macro_export]
macro_rules! export_app {
    ($ctor:expr) => {
        $crate::export_app!($ctor, $crate::WebConfig::default());
    };
    ($ctor:expr, $config:expr) => {
        const _: () = {
            fn check() -> (impl $crate::__App, $crate::WebConfig) {
                ($ctor, $config)
            }
            // Counts as a use on targets without the exports, so `ctor`'s type is not dead code.
            let _ = check;

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_init(width: u32, height: u32, scale_factor: f32) -> i32 {
                let (app, config) = check();
                $crate::__private::init(app, config, width, height, scale_factor)
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_frame(dt_secs: f32) -> i32 {
                $crate::__private::frame(dt_secs)
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_resize(width: u32, height: u32, scale_factor: f32) -> i32 {
                $crate::__private::resize(width, height, scale_factor)
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_key(code: u32, down: u32) {
                $crate::__private::key(code, down);
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_pointer_move(x: f32, y: f32) {
                $crate::__private::pointer_move(x, y);
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_pointer_button(button: u32, down: u32) {
                $crate::__private::pointer_button(button, down);
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_scroll(dx: f32, dy: f32) {
                $crate::__private::scroll(dx, dy);
            }

            #[cfg(target_arch = "wasm32")]
            #[unsafe(no_mangle)]
            extern "C" fn titan_audio_mix(frames: u32) -> *const f32 {
                $crate::__private::audio_mix(frames)
            }
        };
    };
}

/// The runner the exports share, and the functions they call.
#[cfg(target_arch = "wasm32")]
pub mod glue {
    use crate::abi::LOG_ERROR;
    use crate::audio::{MAX_MIX_FRAMES, MIX};
    use crate::bridge::{HostBridge, WasmBridge};
    use crate::runner::{WebConfig, WebRunner, surface_size};
    use alloc::boxed::Box;
    use core::cell::RefCell;
    use core::ptr::null;
    use titan_app::{App, AppAction, InputState, SurfaceSize};
    use titan_gpu::{FrameCommands, GpuDevice, GpuError};

    #[cfg(target_feature = "atomics")]
    compile_error!("titan-wasm's exports assume a single-threaded wasm32 target");

    /// Erases the app's type so the runner fits in a `static`, whose type must be named.
    struct DynApp(Box<dyn App>);

    impl App for DynApp {
        fn init(&mut self, gpu: &mut dyn GpuDevice) -> Result<(), GpuError> {
            self.0.init(gpu)
        }

        fn resize(&mut self, gpu: &mut dyn GpuDevice, size: SurfaceSize) -> Result<(), GpuError> {
            self.0.resize(gpu, size)
        }

        fn step(&mut self, input: &InputState, dt: f32) -> AppAction {
            self.0.step(input, dt)
        }

        fn render(
            &mut self,
            gpu: &mut dyn GpuDevice,
            commands: &mut FrameCommands,
            alpha: f32,
        ) -> Result<(), GpuError> {
            self.0.render(gpu, commands, alpha)
        }
    }

    type Runner = WebRunner<DynApp, WasmBridge>;

    /// The runner, once `titan_init` has succeeded.
    struct Slot(RefCell<Option<Runner>>);

    // SAFETY: without the `atomics` target feature (rejected above) a wasm32 module runs on one
    // thread, so the cell is never accessed concurrently. The host calls one export at a time and
    // no import calls back into an export, so the `RefCell` borrows never overlap either.
    unsafe impl Sync for Slot {}

    static RUNNER: Slot = Slot(RefCell::new(None));

    fn report(context: &str, error: &GpuError) {
        WasmBridge.log(LOG_ERROR, &alloc::format!("{context}: {error}"));
    }

    /// Runs `f` on the runner, or returns `None` before `titan_init` has succeeded.
    fn with_runner<T>(f: impl FnOnce(&mut Runner) -> T) -> Option<T> {
        RUNNER.0.borrow_mut().as_mut().map(f)
    }

    /// `titan_init`: creates the runner, replacing any earlier one.
    pub fn init(
        app: impl App + 'static,
        config: WebConfig,
        width: u32,
        height: u32,
        scale_factor: f32,
    ) -> i32 {
        let runner = surface_size(width, height, scale_factor)
            .and_then(|size| WebRunner::new(DynApp(Box::new(app)), WasmBridge, size, config));
        match runner {
            Ok(runner) => {
                *RUNNER.0.borrow_mut() = Some(runner);
                0
            }
            Err(error) => {
                report("titan_init failed", &error);
                -1
            }
        }
    }

    /// `titan_frame`.
    pub fn frame(dt_secs: f32) -> i32 {
        match with_runner(|r| r.frame(dt_secs)) {
            Some(Ok(AppAction::Continue)) => 0,
            Some(Ok(AppAction::Exit)) => 1,
            Some(Err(error)) => {
                report("titan_frame failed", &error);
                -1
            }
            None => {
                WasmBridge.log(LOG_ERROR, "titan_frame called before titan_init succeeded");
                -1
            }
        }
    }

    /// `titan_resize`.
    pub fn resize(width: u32, height: u32, scale_factor: f32) -> i32 {
        match with_runner(|r| r.on_resize(width, height, scale_factor)) {
            Some(Ok(())) => 0,
            Some(Err(error)) => {
                report("titan_resize failed", &error);
                -1
            }
            None => {
                WasmBridge.log(LOG_ERROR, "titan_resize called before titan_init succeeded");
                -1
            }
        }
    }

    /// `titan_key`: input before `titan_init` succeeds is dropped.
    pub fn key(code: u32, down: u32) {
        with_runner(|r| r.on_key(code, down != 0));
    }

    /// `titan_pointer_move`.
    pub fn pointer_move(x: f32, y: f32) {
        with_runner(|r| r.on_pointer_move(x, y));
    }

    /// `titan_pointer_button`.
    pub fn pointer_button(button: u32, down: u32) {
        with_runner(|r| r.on_pointer_button(button, down != 0));
    }

    /// `titan_scroll`.
    pub fn scroll(dx: f32, dy: f32) {
        with_runner(|r| r.on_scroll(dx, dy));
    }

    /// `titan_audio_mix`.
    pub fn audio_mix(frames: u32) -> *const f32 {
        let mut mix = MIX.0.borrow_mut();
        let Some(mix) = mix.as_mut() else {
            WasmBridge.log(
                LOG_ERROR,
                "titan_audio_mix called with no AudioOutput playing",
            );
            return null();
        };
        match mix(frames) {
            Some(mixed) => mixed,
            None => {
                let message = alloc::format!(
                    "titan_audio_mix: {frames} frames is over the limit of {MAX_MIX_FRAMES}"
                );
                WasmBridge.log(LOG_ERROR, &message);
                null()
            }
        }
    }
}
