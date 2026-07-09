//! Headless GPU device and offscreen-target ownership.

use std::env;

use serde::Serialize;

use crate::{OutputSize, RenderError, ServiceResult, error, validate_output_size};

/// The graphics backend selected for a Titan GPU context.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
    Other,
}

impl AdapterBackend {
    fn from_wgpu(backend: wgpu::Backend) -> Self {
        match backend {
            wgpu::Backend::Vulkan => Self::Vulkan,
            wgpu::Backend::Metal => Self::Metal,
            wgpu::Backend::Dx12 => Self::Dx12,
            wgpu::Backend::Gl => Self::Gl,
            wgpu::Backend::BrowserWebGpu => Self::BrowserWebGpu,
            wgpu::Backend::Noop => Self::Other,
        }
    }
}

/// Titan-owned adapter metadata; no `wgpu` handle crosses the public API.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: AdapterBackend,
}

/// CPU-side interpretation of the two supported adapter-selection variables.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AdapterSelection {
    pub name_substring: Option<String>,
    backends: Option<wgpu::Backends>,
}

impl AdapterSelection {
    /// Reads `WGPU_BACKEND` and `WGPU_ADAPTER_NAME`.
    pub fn from_environment() -> ServiceResult<Self> {
        Self::from_values(
            env::var("WGPU_BACKEND").ok().as_deref(),
            env::var("WGPU_ADAPTER_NAME").ok().as_deref(),
        )
    }

    /// Parses environment values separately so selection can be tested without a GPU.
    pub fn from_values(backend: Option<&str>, name: Option<&str>) -> ServiceResult<Self> {
        let backends = match backend.filter(|value| !value.trim().is_empty()) {
            Some(value) => {
                let backends = wgpu::Backends::from_comma_list(value);
                if backends.is_empty() {
                    return Err(RenderError::new(
                        error::NO_ADAPTER,
                        format!("unsupported WGPU_BACKEND value: {value}"),
                    ));
                }
                Some(backends)
            }
            None => None,
        };
        let name_substring = name
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.to_ascii_lowercase());
        Ok(Self {
            name_substring,
            backends,
        })
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        self.backends
            .is_none_or(|backends| backends.contains(wgpu::Backends::from(info.backend)))
            && self
                .name_substring
                .as_ref()
                .is_none_or(|name| info.name.to_ascii_lowercase().contains(name))
    }
}

/// Owns a headless instance, adapter, device, and queue for `RenderService`.
pub(crate) struct GpuContext {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    _queue: wgpu::Queue,
    adapter_info: AdapterInfo,
}

impl GpuContext {
    /// Creates a context without creating a window, surface, or display connection.
    pub fn new() -> ServiceResult<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> ServiceResult<Self> {
        let selection = AdapterSelection::from_environment()?;
        // The no-env default is the first enumerated matching adapter; deterministic selection
        // is via WGPU_ADAPTER_NAME.
        let backends = selection
            .backends
            .unwrap_or(wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12);
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);

        let adapter = instance
            .enumerate_adapters(backends)
            .await
            .into_iter()
            .find(|adapter| selection.matches(&adapter.get_info()))
            .ok_or_else(|| {
                RenderError::new(
                    error::NO_ADAPTER,
                    "no headless GPU adapter matched the requested selection",
                )
            })?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|_| {
                RenderError::new(
                    error::NO_ADAPTER,
                    "the selected headless adapter could not provide a device",
                )
            })?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            _queue: queue,
            adapter_info: AdapterInfo {
                name: info.name,
                backend: AdapterBackend::from_wgpu(info.backend),
            },
        })
    }

    pub(crate) fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    pub(crate) fn device_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Creates an sRGB color target and a depth target for an offscreen pass.
    #[allow(dead_code, reason = "consumed by the later render-pass implementation")]
    pub(crate) fn create_render_targets(
        &self,
        size: OutputSize,
    ) -> ServiceResult<OffscreenRenderTargets> {
        validate_output_size(size, Some(self.device.limits().max_texture_dimension_2d))?;
        let extent = wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        };
        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("titan headless color target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("titan headless depth target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Ok(OffscreenRenderTargets { size, color, depth })
    }
}

/// GPU-owned color/depth targets with no public `wgpu` types in the contract.
#[expect(
    dead_code,
    reason = "target handles are consumed by later render-pass steps"
)]
pub(crate) struct OffscreenRenderTargets {
    size: OutputSize,
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

impl OffscreenRenderTargets {
    #[allow(dead_code, reason = "consumed by the later render-pass implementation")]
    pub(crate) fn size(&self) -> OutputSize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_case_insensitive_and_uses_name_substrings() {
        let selection = AdapterSelection::from_values(Some("VULKAN"), Some("LlVpMpE")).unwrap();
        assert_eq!(selection.backends, Some(wgpu::Backends::VULKAN));
        assert_eq!(selection.name_substring.as_deref(), Some("llvpmpe"));
    }

    #[test]
    fn selection_accepts_wgpu_aliases_and_lists() {
        let selection = AdapterSelection::from_values(Some("vk, mtl, d3d12, gles"), None).unwrap();
        assert_eq!(
            selection.backends,
            Some(
                wgpu::Backends::VULKAN
                    | wgpu::Backends::METAL
                    | wgpu::Backends::DX12
                    | wgpu::Backends::GL
            )
        );
    }

    #[test]
    fn selection_matches_backend_and_name_substring() {
        let selection = AdapterSelection::from_values(Some("vulkan"), Some("LLVMpipe")).unwrap();
        let matching_info = wgpu::AdapterInfo {
            name: "llvmpipe (LLVM 15.0.7, 256 bits)".to_owned(),
            ..wgpu::AdapterInfo::new(wgpu::DeviceType::Other, wgpu::Backend::Vulkan)
        };
        assert!(selection.matches(&matching_info));

        let wrong_backend = wgpu::AdapterInfo {
            backend: wgpu::Backend::Gl,
            ..matching_info.clone()
        };
        assert!(!selection.matches(&wrong_backend));

        let wrong_name = wgpu::AdapterInfo {
            name: "swiftshader".to_owned(),
            ..matching_info
        };
        assert!(!selection.matches(&wrong_name));
    }

    #[test]
    fn unsupported_backend_is_structured() {
        let error = AdapterSelection::from_values(Some("not-a-backend"), None).unwrap_err();
        assert_eq!(error.code, error::NO_ADAPTER);
    }

    #[test]
    fn output_size_validation_is_cpu_only() {
        assert!(validate_output_size(OutputSize::new(1, 1), Some(4)).is_ok());
        assert_eq!(
            validate_output_size(OutputSize::new(0, 1), Some(4))
                .unwrap_err()
                .code,
            error::INVALID_OUTPUT_SIZE
        );
        assert_eq!(
            validate_output_size(OutputSize::new(5, 1), Some(4))
                .unwrap_err()
                .code,
            error::INVALID_OUTPUT_SIZE
        );
    }

    #[test]
    fn headless_adapter_smoke_reports_no_adapter() {
        match GpuContext::new() {
            Ok(context) => {
                assert!(!context.adapter_info().name.is_empty());
                let targets = context
                    .create_render_targets(OutputSize::new(1, 1))
                    .unwrap();
                assert_eq!(targets.size(), OutputSize::new(1, 1));
            }
            Err(error) => assert_eq!(error.code, error::NO_ADAPTER),
        }
    }
}
