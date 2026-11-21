//! [`HostBridge`]: the calls a [`WebGpuDevice`](crate::WebGpuDevice) makes into its host, and
//! `WasmBridge`, which makes them through WASM imports from `web/titan.js`.

use titan_gpu::{GpuError, TextureFormat};

/// The host side of a [`WebGpuDevice`](crate::WebGpuDevice): it owns the WebGPU objects, named by
/// the `u32` ids the device assigns.
///
/// The device validates every call before making it, so the host only fails on what Rust cannot
/// check, such as WGSL compile errors. Descriptor and frame payloads follow the layouts in
/// [`abi`](crate::abi).
pub trait HostBridge {
    /// Returns the canvas surface's format.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] if the host reports a format Titan does not know.
    fn surface_format(&mut self) -> Result<TextureFormat, GpuError>;

    /// Creates buffer `id` of `size` bytes with [`BufferUsage`](titan_gpu::BufferUsage) bits
    /// `usage`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn create_buffer(
        &mut self,
        id: u32,
        size: u32,
        usage: u32,
        label: Option<&str>,
    ) -> Result<(), GpuError>;

    /// Copies `data` into buffer `id` at byte `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn write_buffer(&mut self, id: u32, offset: u32, data: &[u8]) -> Result<(), GpuError>;

    /// Destroys buffer `id`.
    fn destroy_buffer(&mut self, id: u32);

    /// Creates texture `id` from an [`encode_texture_desc`](crate::abi::encode_texture_desc)
    /// payload.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn create_texture(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError>;

    /// Replaces mip level 0 of texture `id` with tightly packed rows of `data`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn write_texture(&mut self, id: u32, data: &[u8]) -> Result<(), GpuError>;

    /// Destroys texture `id`.
    fn destroy_texture(&mut self, id: u32);

    /// Compiles shader `id` from an [`encode_shader_source`](crate::abi::encode_shader_source)
    /// payload.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn create_shader(&mut self, id: u32, encoded_source: &[u8]) -> Result<(), GpuError>;

    /// Destroys shader `id`.
    fn destroy_shader(&mut self, id: u32);

    /// Creates pipeline `id` from an [`encode_pipeline_desc`](crate::abi::encode_pipeline_desc)
    /// payload.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn create_pipeline(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError>;

    /// Destroys pipeline `id`.
    fn destroy_pipeline(&mut self, id: u32);

    /// Encodes and submits a frame payload that starts with
    /// [`begin_frame`](crate::abi::begin_frame).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Backend`] with the host's message.
    fn submit_frame(&mut self, encoded_commands: &[u8]) -> Result<(), GpuError>;

    /// Logs `message` at `level`, one of the [`abi`](crate::abi) `LOG_*` levels. The default
    /// drops it.
    fn log(&mut self, level: u32, message: &str) {
        let _ = (level, message);
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::WasmBridge;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::HostBridge;
    use crate::abi::WireEnum;
    use alloc::string::String;
    use alloc::vec;
    use titan_gpu::{GpuError, TextureFormat};

    /// The functions `web/titan.js` provides as the `titan` import module. Each fallible import
    /// returns 0 on success or the byte length of an error message, which `read_error` then
    /// copies out. Pointers and lengths name ranges of this module's linear memory.
    mod imports {
        #[link(wasm_import_module = "titan")]
        unsafe extern "C" {
            pub(super) safe fn surface_format() -> u32;
            pub(super) unsafe fn create_buffer(
                id: u32,
                size: u32,
                usage: u32,
                label_ptr: *const u8,
                label_len: u32,
            ) -> u32;
            pub(super) unsafe fn write_buffer(
                id: u32,
                offset: u32,
                data_ptr: *const u8,
                data_len: u32,
            ) -> u32;
            pub(super) safe fn destroy_buffer(id: u32);
            pub(super) unsafe fn create_texture(id: u32, desc_ptr: *const u8, desc_len: u32)
            -> u32;
            pub(super) unsafe fn write_texture(id: u32, data_ptr: *const u8, data_len: u32) -> u32;
            pub(super) safe fn destroy_texture(id: u32);
            pub(super) unsafe fn create_shader(id: u32, desc_ptr: *const u8, desc_len: u32) -> u32;
            pub(super) safe fn destroy_shader(id: u32);
            pub(super) unsafe fn create_pipeline(
                id: u32,
                desc_ptr: *const u8,
                desc_len: u32,
            ) -> u32;
            pub(super) safe fn destroy_pipeline(id: u32);
            pub(super) unsafe fn submit_frame(cmd_ptr: *const u8, cmd_len: u32) -> u32;
            pub(super) unsafe fn read_error(dst_ptr: *mut u8, dst_len: u32);
            pub(super) unsafe fn log(level: u32, msg_ptr: *const u8, msg_len: u32);
        }
    }

    /// The [`HostBridge`] of a WASM module running under `web/titan.js`.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct WasmBridge;

    /// Splits a slice into the pointer and length an import takes. Lossless: `usize` is 32 bits
    /// on wasm32.
    fn raw(bytes: &[u8]) -> (*const u8, u32) {
        (bytes.as_ptr(), bytes.len() as u32)
    }

    /// Turns an import's return value into a result, fetching the error message if there is one.
    fn check(error_len: u32) -> Result<(), GpuError> {
        if error_len == 0 {
            return Ok(());
        }
        let mut message = vec![0; error_len as usize];
        // SAFETY: `message` is `error_len` writable bytes, which the import fills.
        unsafe { imports::read_error(message.as_mut_ptr(), error_len) };
        Err(GpuError::Backend(
            String::from_utf8_lossy(&message).into_owned(),
        ))
    }

    impl HostBridge for WasmBridge {
        fn surface_format(&mut self) -> Result<TextureFormat, GpuError> {
            let tag = imports::surface_format();
            u8::try_from(tag)
                .ok()
                .and_then(TextureFormat::from_tag)
                .ok_or_else(|| {
                    GpuError::Backend(alloc::format!("unknown surface format tag {tag}"))
                })
        }

        fn create_buffer(
            &mut self,
            id: u32,
            size: u32,
            usage: u32,
            label: Option<&str>,
        ) -> Result<(), GpuError> {
            let (label_ptr, label_len) = raw(label.unwrap_or("").as_bytes());
            // SAFETY: the label range is live, readable memory for the whole call.
            check(unsafe { imports::create_buffer(id, size, usage, label_ptr, label_len) })
        }

        fn write_buffer(&mut self, id: u32, offset: u32, data: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(data);
            // SAFETY: the data range is live, readable memory for the whole call.
            check(unsafe { imports::write_buffer(id, offset, ptr, len) })
        }

        fn destroy_buffer(&mut self, id: u32) {
            imports::destroy_buffer(id);
        }

        fn create_texture(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(encoded_desc);
            // SAFETY: the payload range is live, readable memory for the whole call.
            check(unsafe { imports::create_texture(id, ptr, len) })
        }

        fn write_texture(&mut self, id: u32, data: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(data);
            // SAFETY: the data range is live, readable memory for the whole call.
            check(unsafe { imports::write_texture(id, ptr, len) })
        }

        fn destroy_texture(&mut self, id: u32) {
            imports::destroy_texture(id);
        }

        fn create_shader(&mut self, id: u32, encoded_source: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(encoded_source);
            // SAFETY: the payload range is live, readable memory for the whole call.
            check(unsafe { imports::create_shader(id, ptr, len) })
        }

        fn destroy_shader(&mut self, id: u32) {
            imports::destroy_shader(id);
        }

        fn create_pipeline(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(encoded_desc);
            // SAFETY: the payload range is live, readable memory for the whole call.
            check(unsafe { imports::create_pipeline(id, ptr, len) })
        }

        fn destroy_pipeline(&mut self, id: u32) {
            imports::destroy_pipeline(id);
        }

        fn submit_frame(&mut self, encoded_commands: &[u8]) -> Result<(), GpuError> {
            let (ptr, len) = raw(encoded_commands);
            // SAFETY: the payload range is live, readable memory for the whole call.
            check(unsafe { imports::submit_frame(ptr, len) })
        }

        fn log(&mut self, level: u32, message: &str) {
            let (ptr, len) = raw(message.as_bytes());
            // SAFETY: the message range is live, readable memory for the whole call.
            unsafe { imports::log(level, ptr, len) };
        }
    }
}

/// A [`HostBridge`] for tests that records every call it accepts.
#[cfg(test)]
pub(crate) mod mock {
    use super::HostBridge;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use titan_gpu::{GpuError, TextureFormat};

    /// One accepted bridge call, with copies of its payloads.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum Call {
        CreateBuffer {
            id: u32,
            size: u32,
            usage: u32,
            label: Option<String>,
        },
        WriteBuffer {
            id: u32,
            offset: u32,
            data: Vec<u8>,
        },
        DestroyBuffer(u32),
        CreateTexture(u32, Vec<u8>),
        WriteTexture(u32, Vec<u8>),
        DestroyTexture(u32),
        CreateShader(u32, Vec<u8>),
        DestroyShader(u32),
        CreatePipeline(u32, Vec<u8>),
        DestroyPipeline(u32),
        SubmitFrame(Vec<u8>),
    }

    #[derive(Debug)]
    pub(crate) struct MockBridge {
        pub(crate) surface_format: TextureFormat,
        pub(crate) calls: Vec<Call>,
        /// Makes the next fallible call fail with this message instead of being recorded.
        pub(crate) fail_next: Option<&'static str>,
    }

    impl MockBridge {
        pub(crate) fn new() -> Self {
            Self {
                surface_format: TextureFormat::Bgra8UnormSrgb,
                calls: Vec::new(),
                fail_next: None,
            }
        }

        pub(crate) fn frames(&self) -> impl Iterator<Item = &[u8]> {
            self.calls.iter().filter_map(|call| match call {
                Call::SubmitFrame(bytes) => Some(bytes.as_slice()),
                _ => None,
            })
        }

        fn accept(&mut self, call: Call) -> Result<(), GpuError> {
            match self.fail_next.take() {
                Some(message) => Err(GpuError::Backend(message.to_string())),
                None => {
                    self.calls.push(call);
                    Ok(())
                }
            }
        }
    }

    impl HostBridge for MockBridge {
        fn surface_format(&mut self) -> Result<TextureFormat, GpuError> {
            Ok(self.surface_format)
        }

        fn create_buffer(
            &mut self,
            id: u32,
            size: u32,
            usage: u32,
            label: Option<&str>,
        ) -> Result<(), GpuError> {
            self.accept(Call::CreateBuffer {
                id,
                size,
                usage,
                label: label.map(String::from),
            })
        }

        fn write_buffer(&mut self, id: u32, offset: u32, data: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::WriteBuffer {
                id,
                offset,
                data: data.to_vec(),
            })
        }

        fn destroy_buffer(&mut self, id: u32) {
            self.calls.push(Call::DestroyBuffer(id));
        }

        fn create_texture(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::CreateTexture(id, encoded_desc.to_vec()))
        }

        fn write_texture(&mut self, id: u32, data: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::WriteTexture(id, data.to_vec()))
        }

        fn destroy_texture(&mut self, id: u32) {
            self.calls.push(Call::DestroyTexture(id));
        }

        fn create_shader(&mut self, id: u32, encoded_source: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::CreateShader(id, encoded_source.to_vec()))
        }

        fn destroy_shader(&mut self, id: u32) {
            self.calls.push(Call::DestroyShader(id));
        }

        fn create_pipeline(&mut self, id: u32, encoded_desc: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::CreatePipeline(id, encoded_desc.to_vec()))
        }

        fn destroy_pipeline(&mut self, id: u32) {
            self.calls.push(Call::DestroyPipeline(id));
        }

        fn submit_frame(&mut self, encoded_commands: &[u8]) -> Result<(), GpuError> {
            self.accept(Call::SubmitFrame(encoded_commands.to_vec()))
        }
    }
}
