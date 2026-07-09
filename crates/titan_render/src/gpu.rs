//! Headless GPU device and offscreen-target ownership.

use std::env;

use serde::Serialize;

use crate::{DrawItem, Mat4, OutputSize, RenderError, ServiceResult, error, validate_output_size};

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
    queue: wgpu::Queue,
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
            queue,
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

    pub(crate) fn draw_plan(
        &self,
        size: OutputSize,
        clear: [f32; 4],
        view_projection: [[f32; 4]; 4],
        draw_list: &[DrawItem],
    ) -> ServiceResult<Vec<u8>> {
        let targets = self.create_render_targets(size)?;
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("titan first draw shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                "@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;\n@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n  var p = array<vec3<f32>, 3>(vec3<f32>(-0.7, -0.6, 0.0), vec3<f32>(0.0, 0.75, 0.0), vec3<f32>(0.7, -0.6, 0.0));\n  return camera * vec4<f32>(p[i], 1.0);\n}\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.95, 0.2, 0.08, 1.0); }\n",
            )),
        });
        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("titan camera layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_groups = draw_list
            .iter()
            .map(|item| {
                let matrix = (Mat4(view_projection) * item.model).transpose().0;
                let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("titan draw transform uniform"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut bytes = [0u8; 64];
                for (index, value) in matrix.iter().flatten().enumerate() {
                    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
                }
                self.queue.write_buffer(&uniform, 0, &bytes);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("titan draw transform bind group"),
                    layout: &bind_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                (uniform, bind_group)
            })
            .collect::<Vec<_>>();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("titan first draw pipeline"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("titan first draw layout"),
                        bind_group_layouts: &[Some(&bind_layout)],
                        immediate_size: 0,
                    },
                )),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let bytes_per_row = size.width * 4;
        let padded_row = bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("titan pixel readback"),
            size: u64::from(padded_row) * u64::from(size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("titan first draw encoder"),
            });
        let color_view = targets.color.create_view(&Default::default());
        let depth_view = targets.depth.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("titan first draw pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0] as f64,
                            g: clear[1] as f64,
                            b: clear[2] as f64,
                            a: clear[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            for (_, bind_group) in &bind_groups {
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.color,
                mip_level: 0,
                origin: Default::default(),
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::new(error::CAPTURE_UNAVAILABLE, "GPU polling failed"))?;
        receiver
            .recv()
            .map_err(|_| {
                RenderError::new(error::CAPTURE_UNAVAILABLE, "GPU readback callback failed")
            })?
            .map_err(|_| RenderError::new(error::CAPTURE_UNAVAILABLE, "GPU readback failed"))?;
        let mapped = slice.get_mapped_range().map_err(|_| {
            RenderError::new(error::CAPTURE_UNAVAILABLE, "GPU mapped readback failed")
        })?;
        let mut rgba8 = vec![0; (size.width * size.height * 4) as usize];
        for row in 0..size.height as usize {
            let src = row * padded_row as usize;
            let dst = row * bytes_per_row as usize;
            rgba8[dst..dst + bytes_per_row as usize]
                .copy_from_slice(&mapped[src..src + bytes_per_row as usize]);
        }
        drop(mapped);
        readback.unmap();
        Ok(rgba8)
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
        Ok(OffscreenRenderTargets { color, depth })
    }
}

/// GPU-owned color/depth targets with no public `wgpu` types in the contract.
pub(crate) struct OffscreenRenderTargets {
    color: wgpu::Texture,
    depth: wgpu::Texture,
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
                context
                    .create_render_targets(OutputSize::new(1, 1))
                    .unwrap();
            }
            Err(error) => assert_eq!(error.code, error::NO_ADAPTER),
        }
    }
}
