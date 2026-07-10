//! Headless GPU device and offscreen-target ownership.

use std::env;

use serde::Serialize;

use crate::{
    DirectionalLightData, DrawItem, Mat4, MaterialModel, OutputSize, RenderError, ServiceResult,
    error, validate_output_size,
};

const UNIFORM_SIZE: usize = 272;
const MVP_OFFSET: usize = 0;
const MODEL_OFFSET: usize = 64;
const NORMAL_MATRIX_OFFSET: usize = 128;
const CAMERA_POSITION_OFFSET: usize = 192;
const BASE_COLOR_OFFSET: usize = 208;

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
    pub driver: String,
    pub driver_info: String,
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
                driver: info.driver,
                driver_info: info.driver_info,
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
        light: Option<DirectionalLightData>,
        camera_position: [f32; 3],
    ) -> ServiceResult<Vec<u8>> {
        let targets = self.create_render_targets(size)?;
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("titan unlit and basic pbr shaders"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                "struct Uniforms { mvp: mat4x4<f32>, model: mat4x4<f32>, normal_matrix: mat4x4<f32>, camera_position: vec4<f32>, base_color: vec4<f32>, light_color: vec4<f32>, light_direction: vec4<f32>, params: vec4<f32> };\n@group(0) @binding(0) var<uniform> uniforms: Uniforms;\nstruct UnlitIn { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32> };\nstruct PbrIn { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };\nstruct UnlitOut { @builtin(position) position: vec4<f32> };\nstruct PbrOut { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world_position: vec3<f32> };\n@vertex fn vs_unlit(vertex: UnlitIn) -> UnlitOut { var out: UnlitOut; out.position = uniforms.mvp * vec4<f32>(vertex.position, 1.0); return out; }\n@vertex fn vs_pbr(vertex: PbrIn) -> PbrOut { var out: PbrOut; let world = uniforms.model * vec4<f32>(vertex.position, 1.0); out.position = uniforms.mvp * vec4<f32>(vertex.position, 1.0); out.normal = normalize((uniforms.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz); out.world_position = world.xyz; return out; }\n@fragment fn fs_unlit() -> @location(0) vec4<f32> { return uniforms.base_color; }\n@fragment fn fs(in: PbrOut) -> @location(0) vec4<f32> { let base = uniforms.base_color.rgb; let n = normalize(in.normal); let l = normalize(-uniforms.light_direction.xyz); let v = normalize(uniforms.camera_position.xyz - in.world_position); let diffuse = max(dot(n, l), 0.0); let metallic = uniforms.params.y; let roughness = max(uniforms.params.z, 0.04); let ambient = uniforms.params.w; let light = uniforms.light_color.rgb * uniforms.light_color.a * diffuse; let f0 = mix(vec3<f32>(0.04), base, metallic); let specular = f0 * pow(max(dot(reflect(-l, n), v), 0.0), 2.0 / (roughness * roughness)); return vec4<f32>(base * (ambient + light * (1.0 - metallic)) + specular * light, uniforms.base_color.a); }\n",
            )),
        });
        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("titan camera layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(UNIFORM_SIZE as u64),
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
                    size: UNIFORM_SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let material = item.material;
                let model = match material.model {
                    MaterialModel::Unlit => 0.0,
                    MaterialModel::Pbr => 1.0,
                };
                let metallic = material.metallic.unwrap_or(0.0);
                let roughness = material.roughness.unwrap_or(1.0);
                let selected = light.unwrap_or(DirectionalLightData {
                    entity: titan_core::EntityId::from_raw(0),
                    direction: [0.0, 0.0, 1.0],
                    color: [0.0, 0.0, 0.0],
                    illuminance: 0.0,
                    ambient: 0.08,
                });
                let values = [
                    [
                        material.base_color[0],
                        material.base_color[1],
                        material.base_color[2],
                        material.base_color[3],
                    ],
                    [
                        selected.color[0],
                        selected.color[1],
                        selected.color[2],
                        selected.illuminance,
                    ],
                    [
                        selected.direction[0],
                        selected.direction[1],
                        selected.direction[2],
                        0.0,
                    ],
                    [
                        model,
                        metallic,
                        roughness,
                        if light.is_some() {
                            selected.ambient
                        } else {
                            0.08
                        },
                    ],
                ];
                let camera = [
                    camera_position[0],
                    camera_position[1],
                    camera_position[2],
                    0.0,
                ];
                let bytes = pack_uniform_bytes(
                    matrix,
                    item.model,
                    Mat4::normal_from_model(item.model),
                    camera,
                    values,
                );
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
        let mesh_buffers = draw_list
            .iter()
            .map(|item| {
                let vertex_bytes = item
                    .geometry
                    .vertices
                    .iter()
                    .enumerate()
                    .flat_map(|(index, vertex)| {
                        let mut values = vertex
                            .position
                            .into_iter()
                            .chain(vertex.uv)
                            .collect::<Vec<_>>();
                        if matches!(item.material.model, MaterialModel::Pbr) {
                            values = vertex
                                .position
                                .into_iter()
                                .chain(
                                    item.geometry
                                        .normals
                                        .as_ref()
                                        .expect("validated PBR normals")[index],
                                )
                                .chain(vertex.uv)
                                .collect();
                        }
                        values
                    })
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>();
                let index_bytes = item
                    .geometry
                    .indices
                    .iter()
                    .flat_map(|index| index.to_ne_bytes())
                    .collect::<Vec<_>>();
                let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("titan mesh vertex buffer"),
                    size: vertex_bytes.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(&vertex_buffer, 0, &vertex_bytes);
                let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("titan mesh index buffer"),
                    size: index_bytes.len() as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(&index_buffer, 0, &index_bytes);
                (vertex_buffer, index_buffer)
            })
            .collect::<Vec<_>>();
        let pbr_pipeline = self
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
                    entry_point: Some("vs_pbr"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 24,
                                shader_location: 2,
                            },
                        ],
                    })],
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
        let unlit_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("titan unlit draw layout"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });
        let unlit_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("titan unlit pipeline"),
                layout: Some(&unlit_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_unlit"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 20,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_unlit"),
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
            for ((item, (vertex_buffer, index_buffer)), (_, bind_group)) in draw_list
                .iter()
                .zip(mesh_buffers.iter())
                .zip(bind_groups.iter())
            {
                pass.set_pipeline(match item.material.model {
                    MaterialModel::Unlit => &unlit_pipeline,
                    MaterialModel::Pbr => &pbr_pipeline,
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                for submesh in &item.geometry.submeshes {
                    pass.draw_indexed(
                        submesh.index_start..submesh.index_start + submesh.index_count,
                        0,
                        0..1,
                    );
                }
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

fn pack_uniform_bytes(
    mvp: [[f32; 4]; 4],
    model: Mat4,
    normal_matrix: Mat4,
    camera_position: [f32; 4],
    values: [[f32; 4]; 4],
) -> [u8; UNIFORM_SIZE] {
    let mut bytes = [0u8; UNIFORM_SIZE];
    for (offset, matrix) in [
        (MVP_OFFSET, mvp),
        (MODEL_OFFSET, model.transpose().0),
        (NORMAL_MATRIX_OFFSET, normal_matrix.transpose().0),
    ] {
        for (index, value) in matrix.iter().flatten().enumerate() {
            let byte_index = offset + index * 4;
            bytes[byte_index..byte_index + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
    for (offset, vector) in [(CAMERA_POSITION_OFFSET, camera_position)]
        .into_iter()
        .chain(
            values
                .into_iter()
                .enumerate()
                .map(|(index, vector)| (BASE_COLOR_OFFSET + index * 16, vector)),
        )
    {
        for (component, value) in vector.into_iter().enumerate() {
            let byte_index = offset + component * 4;
            bytes[byte_index..byte_index + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
    bytes
}

/// GPU-owned color/depth targets with no public `wgpu` types in the contract.
pub(crate) struct OffscreenRenderTargets {
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn uniform_buffer_layout_has_expected_field_offsets() {
        let mvp = [[1.0; 4]; 4];
        let model = Mat4([[2.0; 4]; 4]);
        let normal_matrix = Mat4([[3.0; 4]; 4]);
        let camera = [4.0, 5.0, 6.0, 7.0];
        let values = [
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
        ];
        let bytes = pack_uniform_bytes(mvp, model, normal_matrix, camera, values);

        assert_eq!(bytes.len(), 272);
        assert_eq!(f32_at(&bytes, 0), 1.0);
        assert_eq!(f32_at(&bytes, 64), 2.0);
        assert_eq!(f32_at(&bytes, 128), 3.0);
        assert_eq!(
            &bytes[192..208],
            &camera
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(f32_at(&bytes, 208), 8.0);
        assert_eq!(f32_at(&bytes, 224), 12.0);
        assert_eq!(f32_at(&bytes, 240), 16.0);
        assert_eq!(f32_at(&bytes, 256), 20.0);
    }

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
