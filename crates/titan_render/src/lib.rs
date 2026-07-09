//! Renderer-owned, headless rendering contracts.
//!
//! This crate owns the renderer API, the deterministic CPU render plan, and
//! the headless GPU context used by GPU-backed services.

use std::{collections::BTreeMap, fmt};

use serde::Serialize;
use titan_core::{Camera, CameraProjection, Component, EntityId, Mesh, Transform, Velocity, World};
use titan_math::Vec3;

mod gpu;

pub use gpu::{AdapterBackend, AdapterInfo, AdapterSelection};

/// Stable codes used by structured renderer diagnostics.
pub mod error {
    // These phase-1 codes are transitional; reconcile them with the design
    // document's final error taxonomy when the backend lands in step 2.
    pub const CAMERA_UNAVAILABLE: &str = "RENDER_CAMERA_UNAVAILABLE";
    pub const CAPTURE_UNAVAILABLE: &str = "RENDER_CAPTURE_UNAVAILABLE";
    pub const INVALID_OUTPUT_SIZE: &str = "RENDER_INVALID_OUTPUT_SIZE";
    pub const NO_ADAPTER: &str = "RENDER_NO_ADAPTER";
    pub const UNKNOWN_BUILTIN: &str = "RENDER_UNKNOWN_BUILTIN";
    pub const MISSING_NORMALS: &str = "RENDER_MISSING_NORMALS";
    pub const INVALID_NORMALS: &str = "RENDER_INVALID_NORMALS";
}

/// A structured renderer failure, suitable for a CLI error envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RenderError {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl RenderError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
        }
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RenderError {}

/// Result type for renderer operations.
pub type ServiceResult<T> = Result<T, RenderError>;

/// Camera selection reserved for the camera component implementation.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CameraSelection {
    Entity(EntityId),
    Name(String),
    /// Let the renderer choose its default camera once cameras exist.
    #[default]
    Default,
}

/// Requested output dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OutputSize {
    pub width: u32,
    pub height: u32,
}

impl OutputSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

pub(crate) fn validate_output_size(
    output_size: OutputSize,
    max_texture_dimension_2d: Option<u32>,
) -> ServiceResult<()> {
    if output_size.width == 0 || output_size.height == 0 {
        return Err(RenderError::new(
            error::INVALID_OUTPUT_SIZE,
            "render output dimensions must be greater than zero",
        ));
    }
    if let Some(max_dimension) = max_texture_dimension_2d
        && (output_size.width > max_dimension || output_size.height > max_dimension)
    {
        return Err(RenderError::new(
            error::INVALID_OUTPUT_SIZE,
            format!("render output dimensions must not exceed the device limit of {max_dimension}"),
        ));
    }
    Ok(())
}

impl Default for OutputSize {
    fn default() -> Self {
        Self::new(640, 480)
    }
}

/// RGBA clear color in the renderer's linear component model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ClearColor(pub [f32; 4]);

impl Default for ClearColor {
    fn default() -> Self {
        Self([0.0, 0.0, 0.0, 1.0])
    }
}

/// Whether a future backend should produce pixels in addition to stats.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    #[default]
    StatsOnly,
    Image,
}

/// Backend-independent render request.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RenderRequest {
    pub camera: CameraSelection,
    pub output_size: OutputSize,
    pub clear_color: ClearColor,
    pub capture: CaptureMode,
}

/// A renderer-neutral entity extracted from the ECS.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ExtractedEntity {
    pub entity: EntityId,
    pub transform: Transform,
    pub model: Mat4,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<Vec3>,
}

/// Future GPU draw input. Mesh/material fields will be added with their TSF
/// components; keeping this type separate prevents renderer state leaking into
/// `titan_core`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DrawItem {
    pub entity: EntityId,
}

/// Deterministic CPU render plan.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RenderScene {
    pub entities: Vec<ExtractedEntity>,
    pub draw_list: Vec<DrawItem>,
}

/// Exact stats computed from a render plan, not GPU timing.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub triangles: u64,
    pub visible_meshes: u32,
    pub active_directional_lights: u32,
    pub ignored_directional_lights: u32,
    pub material_models: BTreeMap<String, u32>,
    pub active_camera: Option<EntityId>,
}

/// Result returned by the service. Pixels are deliberately unavailable until
/// the backend phase; stats and the CPU plan are already usable in CI.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderResult {
    pub camera: Option<EntityId>,
    pub output_size: OutputSize,
    pub stats: RenderStats,
    pub scene: RenderScene,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgba8: Option<Vec<u8>>,
}

/// Metadata for components understood by extraction. TSF aliases remain owned
/// exclusively by the alias registry in `titan_scene`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderComponentMetadata {
    pub registered_name: &'static str,
    pub schema_version: u32,
}

/// Renderer-side metadata, kept separate from the ECS and TSF registries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderComponentRegistry {
    components: Vec<RenderComponentMetadata>,
}

/// A deterministic row-major 4x4 matrix, with column vectors on the right.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    pub const IDENTITY: Self = Self([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    pub fn from_transform(transform: &Transform) -> Self {
        let [x, y, z, w] = transform.rotation;
        Self([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                transform.translation.x,
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                transform.translation.y,
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
                transform.translation.z,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn view_from_transform(transform: &Transform) -> Self {
        let model = Self::from_transform(transform).0;
        let mut view = Self::IDENTITY.0;
        for row in 0..3 {
            for column in 0..3 {
                view[row][column] = model[column][row];
            }
        }
        for values in view.iter_mut().take(3) {
            values[3] =
                -(values[0] * model[0][3] + values[1] * model[1][3] + values[2] * model[2][3]);
        }
        Self(view)
    }

    pub fn projection(projection: &CameraProjection, aspect: f32) -> ServiceResult<Self> {
        if !aspect.is_finite() || aspect <= 0.0 {
            return Err(RenderError::new(
                error::INVALID_OUTPUT_SIZE,
                "camera aspect ratio must be positive",
            ));
        }
        match *projection {
            CameraProjection::Perspective {
                vertical_fov_degrees,
                near,
                far,
                ..
            } => {
                let f = 1.0 / (vertical_fov_degrees.to_radians() * 0.5).tan();
                if !f.is_finite() || near <= 0.0 || far <= near {
                    return Err(RenderError::new(
                        error::CAMERA_UNAVAILABLE,
                        "invalid perspective camera parameters",
                    ));
                }
                Ok(Self([
                    [f / aspect, 0.0, 0.0, 0.0],
                    [0.0, f, 0.0, 0.0],
                    [0.0, 0.0, far / (near - far), (near * far) / (near - far)],
                    [0.0, 0.0, -1.0, 0.0],
                ]))
            }
            CameraProjection::Orthographic {
                height, near, far, ..
            } => {
                if !height.is_finite() || height <= 0.0 || near <= 0.0 || far <= near {
                    return Err(RenderError::new(
                        error::CAMERA_UNAVAILABLE,
                        "invalid orthographic camera parameters",
                    ));
                }
                let half = height * 0.5;
                Ok(Self([
                    [1.0 / (half * aspect), 0.0, 0.0, 0.0],
                    [0.0, 1.0 / half, 0.0, 0.0],
                    [0.0, 0.0, 1.0 / (near - far), near / (near - far)],
                    [0.0, 0.0, 0.0, 1.0],
                ]))
            }
        }
    }

    fn transpose(self) -> Self {
        let mut result = [[0.0; 4]; 4];
        for (row, values) in result.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = self.0[column][row];
            }
        }
        Self(result)
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0; 4]; 4];
        for (row, values) in out.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = (0..4).map(|i| self.0[row][i] * rhs.0[i][column]).sum();
            }
        }
        Self(out)
    }
}

fn resolve_camera(
    world: &World,
    selection: &CameraSelection,
) -> ServiceResult<Option<(EntityId, Camera, Transform)>> {
    let candidates: Vec<EntityId> = match selection {
        CameraSelection::Entity(entity) => vec![*entity],
        CameraSelection::Name(name) => world.scene_entities_named(name),
        CameraSelection::Default => world
            .query::<&'static Camera>()
            .map(|(entity, _)| entity)
            .collect(),
    };
    if candidates.len() > 1 && matches!(selection, CameraSelection::Name(_)) {
        return Err(RenderError::new(
            error::CAMERA_UNAVAILABLE,
            "camera name is ambiguous",
        ));
    }
    if candidates.is_empty() {
        if matches!(selection, CameraSelection::Default) {
            return Ok(None);
        }
        return Err(RenderError::new(
            error::CAMERA_UNAVAILABLE,
            "selected camera was not found",
        ));
    }
    let entity = candidates[0];
    let camera = world
        .get::<Camera>(entity)
        .map_err(|error| RenderError::new(error::CAMERA_UNAVAILABLE, error.to_string()))?
        .ok_or_else(|| {
            RenderError::new(
                error::CAMERA_UNAVAILABLE,
                "selected entity has no camera component",
            )
        })?;
    let transform = world
        .get::<Transform>(entity)
        .map_err(|error| RenderError::new(error::CAMERA_UNAVAILABLE, error.to_string()))?
        .ok_or_else(|| {
            RenderError::new(
                error::CAMERA_UNAVAILABLE,
                "selected camera has no transform",
            )
        })?;
    Ok(Some((entity, *camera, *transform)))
}

fn projection_viewport(projection: &CameraProjection) -> Option<OutputSize> {
    match *projection {
        CameraProjection::Perspective { viewport, .. }
        | CameraProjection::Orthographic { viewport, .. } => {
            viewport.map(|v| OutputSize::new(v.width, v.height))
        }
    }
}

impl RenderComponentRegistry {
    pub fn phase1() -> Self {
        Self {
            components: vec![
                RenderComponentMetadata {
                    registered_name: Transform::NAME,
                    schema_version: Transform::SCHEMA_VERSION,
                },
                RenderComponentMetadata {
                    registered_name: Velocity::NAME,
                    schema_version: Velocity::SCHEMA_VERSION,
                },
            ],
        }
    }

    pub fn phase2() -> Self {
        let mut registry = Self::phase1();
        registry.components.extend([
            RenderComponentMetadata {
                registered_name: Camera::NAME,
                schema_version: Camera::SCHEMA_VERSION,
            },
            RenderComponentMetadata {
                registered_name: titan_core::DirectionalLight::NAME,
                schema_version: titan_core::DirectionalLight::SCHEMA_VERSION,
            },
            RenderComponentMetadata {
                registered_name: Mesh::NAME,
                schema_version: Mesh::SCHEMA_VERSION,
            },
            RenderComponentMetadata {
                registered_name: titan_core::Material::NAME,
                schema_version: titan_core::Material::SCHEMA_VERSION,
            },
        ]);
        registry
    }

    pub fn components(&self) -> &[RenderComponentMetadata] {
        &self.components
    }
}

/// CPU-side scene extraction, stable by `EntityId` through the core query API.
pub fn extract_scene(world: &World) -> RenderScene {
    let entities = world
        .query::<&'static Transform>()
        .map(|(entity, transform)| ExtractedEntity {
            entity,
            transform: *transform,
            model: Mat4::from_transform(transform),
            velocity: match world.get::<Velocity>(entity) {
                Ok(Some(velocity)) => Some(velocity.linear),
                Ok(None) => None,
                Err(error) => {
                    debug_assert!(false, "velocity lookup failed: {error}");
                    None
                }
            },
        })
        .collect();

    let draw_list = world
        .query::<&'static Mesh>()
        .map(|(entity, _)| DrawItem { entity })
        .collect();
    RenderScene {
        entities,
        draw_list,
    }
}

/// Headless render service boundary. CPU-only services are available for
/// extraction and stats tests; GPU-backed services own their device context.
pub struct RenderService {
    pub components: RenderComponentRegistry,
    gpu: Option<gpu::GpuContext>,
}

impl RenderService {
    /// Creates a GPU-backed service, reporting `RENDER_NO_ADAPTER` when no
    /// suitable headless device is available.
    pub fn new() -> ServiceResult<Self> {
        Ok(Self {
            components: RenderComponentRegistry::phase1(),
            gpu: Some(gpu::GpuContext::new()?),
        })
    }

    /// Creates the CPU-only service used by stats and extraction tests.
    pub fn cpu_only() -> Self {
        Self {
            components: RenderComponentRegistry::phase2(),
            gpu: None,
        }
    }

    /// Returns adapter metadata for a GPU-backed service.
    pub fn adapter_info(&self) -> Option<&AdapterInfo> {
        self.gpu.as_ref().map(gpu::GpuContext::adapter_info)
    }

    /// Renders the CPU-side phase-1 plan. GPU-backed services share this plan
    /// until the draw pipeline is added.
    pub fn render(&self, world: &World, request: RenderRequest) -> ServiceResult<RenderResult> {
        let max_texture_dimension = self
            .gpu
            .as_ref()
            .map(|gpu| gpu.device_limits().max_texture_dimension_2d);
        if request.capture == CaptureMode::Image && self.gpu.is_none() {
            return Err(RenderError::new(
                error::CAPTURE_UNAVAILABLE,
                "image capture requires a GPU-backed render service",
            ));
        }
        let camera = resolve_camera(world, &request.camera)?;
        let output_size = if request.output_size == OutputSize::default() {
            camera
                .as_ref()
                .and_then(|(_, camera, _)| projection_viewport(&camera.projection))
                .unwrap_or(request.output_size)
        } else {
            request.output_size
        };
        validate_output_size(output_size, max_texture_dimension)?;
        let scene = extract_scene(world);
        let view_projection = camera
            .as_ref()
            .map(|(_, camera, transform)| {
                Mat4::projection(
                    &camera.projection,
                    output_size.width as f32 / output_size.height as f32,
                )
                .map(|projection| projection * Mat4::view_from_transform(transform))
            })
            .transpose()?
            .unwrap_or(Mat4::IDENTITY);
        let stats = RenderStats {
            visible_meshes: scene.draw_list.len() as u32,
            draw_calls: scene.draw_list.len() as u32,
            triangles: scene.draw_list.len() as u64,
            active_camera: camera.as_ref().map(|(id, _, _)| *id),
            ..RenderStats::default()
        };
        let rgba8 = match (&self.gpu, request.capture) {
            (Some(gpu), CaptureMode::Image) if camera.is_some() => Some(gpu.draw_triangle(
                output_size,
                request.clear_color.0,
                view_projection.transpose().0,
            )?),
            (_, CaptureMode::Image) if camera.is_none() => {
                return Err(RenderError::new(
                    error::CAMERA_UNAVAILABLE,
                    "a camera is required for image rendering",
                ));
            }
            (_, CaptureMode::Image) => unreachable!("CPU image capture was rejected above"),
            _ => None,
        };
        Ok(RenderResult {
            camera: camera.as_ref().map(|(id, _, _)| *id),
            output_size,
            stats,
            scene,
            rgba8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with_components() -> World {
        let mut world = World::new(titan_core::phase1_component_registry().unwrap());
        let first = world.spawn_with_id(EntityId::from_raw(7)).unwrap();
        world
            .insert(first, Transform::from_translation(Vec3::new(1.0, 2.0, 3.0)))
            .unwrap();
        world
            .insert(first, Velocity::new(Vec3::new(4.0, 5.0, 6.0)))
            .unwrap();
        let second = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        world.insert(second, Transform::default()).unwrap();
        world
    }

    #[test]
    fn extraction_is_stable_and_keeps_cpu_components() {
        let scene = extract_scene(&world_with_components());
        assert_eq!(scene.entities[0].entity, EntityId::from_raw(2));
        assert_eq!(scene.entities[1].entity, EntityId::from_raw(7));
        assert_eq!(scene.entities[1].velocity, Some(Vec3::new(4.0, 5.0, 6.0)));
        assert!(scene.draw_list.is_empty());
    }

    #[test]
    fn stats_reflect_the_current_empty_draw_plan() {
        let output = RenderService::cpu_only()
            .render(&world_with_components(), RenderRequest::default())
            .unwrap();
        assert_eq!(output.stats.draw_calls, 0);
        assert_eq!(output.stats.triangles, 0);
        assert_eq!(output.stats.visible_meshes, 0);
        assert!(output.stats.material_models.is_empty());
    }

    #[test]
    fn invalid_requests_use_stable_codes() {
        let world = world_with_components();
        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    output_size: OutputSize::new(0, 10),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::INVALID_OUTPUT_SIZE);

        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    output_size: OutputSize::new(10, 0),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::INVALID_OUTPUT_SIZE);

        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    camera: CameraSelection::Entity(EntityId::from_raw(7)),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::CAMERA_UNAVAILABLE);

        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    capture: CaptureMode::Image,
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::CAPTURE_UNAVAILABLE);
    }

    #[test]
    fn default_service_registers_phase2_components() {
        assert_eq!(
            RenderService::cpu_only().components,
            RenderService::cpu_only().components
        );
        assert_eq!(RenderService::cpu_only().components.components().len(), 6);
        assert_eq!(
            RenderService::cpu_only().components.components()[0].registered_name,
            Transform::NAME
        );
        assert_eq!(
            RenderService::cpu_only().components.components()[1].registered_name,
            Velocity::NAME
        );
    }

    #[test]
    fn perspective_view_projection_places_origin_at_center() {
        let camera = CameraProjection::Perspective {
            vertical_fov_degrees: 60.0,
            near: 0.1,
            far: 100.0,
            viewport: None,
        };
        let projection = Mat4::projection(&camera, 16.0 / 9.0).unwrap();
        let view =
            Mat4::view_from_transform(&Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)));
        let clip = (projection * view).0;
        assert!((clip[0][3]).abs() < 1e-6);
        assert!((clip[1][3]).abs() < 1e-6);
        assert!((clip[3][3] - 3.0).abs() < 1e-6);
        assert!((projection.0[0][0] - projection.0[1][1] / (16.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn orthographic_projection_uses_requested_height() {
        let projection = Mat4::projection(
            &CameraProjection::Orthographic {
                height: 10.0,
                near: 1.0,
                far: 11.0,
                viewport: None,
            },
            2.0,
        )
        .unwrap();
        assert!((projection.0[0][0] - 0.1).abs() < 1e-6);
        assert!((projection.0[1][1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn named_camera_resolves_and_ambiguous_names_fail() {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        let first = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
        let second = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        let camera = Camera {
            projection: CameraProjection::Perspective {
                vertical_fov_degrees: 60.0,
                near: 0.1,
                far: 10.0,
                viewport: None,
            },
        };
        for entity in [first, second] {
            world.insert(entity, camera).unwrap();
            world
                .insert(
                    entity,
                    Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)),
                )
                .unwrap();
        }
        world.bind_scene_entity_name("main", first).unwrap();
        let result = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    camera: CameraSelection::Name("main".into()),
                    ..RenderRequest::default()
                },
            )
            .unwrap();
        assert_eq!(result.camera, Some(first));
        world.bind_scene_entity_name("main", second).unwrap();
        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    camera: CameraSelection::Name("main".into()),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::CAMERA_UNAVAILABLE);
    }

    #[test]
    fn extraction_contains_models_and_meshes_in_entity_order() {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        for id in [7, 2] {
            let entity = world.spawn_with_id(EntityId::from_raw(id)).unwrap();
            world
                .insert(
                    entity,
                    Transform::from_translation(Vec3::new(id as f32, 0.0, 0.0)),
                )
                .unwrap();
            world
                .insert(
                    entity,
                    Mesh {
                        geometry: titan_core::AssetReference::new("asset:fixture"),
                        submeshes: None,
                    },
                )
                .unwrap();
        }
        let scene = extract_scene(&world);
        assert_eq!(
            scene
                .draw_list
                .iter()
                .map(|item| item.entity.raw())
                .collect::<Vec<_>>(),
            vec![2, 7]
        );
        assert_eq!(
            scene
                .entities
                .iter()
                .map(|item| item.entity.raw())
                .collect::<Vec<_>>(),
            vec![2, 7]
        );
        assert!((scene.entities[0].model.0[0][3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn gpu_smoke_returns_pixels_or_no_adapter() {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        let camera = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
        world
            .insert(
                camera,
                Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)),
            )
            .unwrap();
        world
            .insert(
                camera,
                Camera {
                    projection: CameraProjection::Perspective {
                        vertical_fov_degrees: 60.0,
                        near: 0.1,
                        far: 10.0,
                        viewport: Some(titan_core::Viewport {
                            width: 32,
                            height: 32,
                        }),
                    },
                },
            )
            .unwrap();
        let mesh = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        world.insert(mesh, Transform::default()).unwrap();
        world
            .insert(
                mesh,
                Mesh {
                    geometry: titan_core::AssetReference::new("asset:fixture"),
                    submeshes: None,
                },
            )
            .unwrap();
        match RenderService::new() {
            Err(error) => assert_eq!(error.code, error::NO_ADAPTER),
            Ok(service) => {
                let result = service
                    .render(
                        &world,
                        RenderRequest {
                            camera: CameraSelection::Entity(camera),
                            capture: CaptureMode::Image,
                            clear_color: ClearColor([0.1, 0.2, 0.3, 1.0]),
                            ..RenderRequest::default()
                        },
                    )
                    .unwrap();
                let pixels = result.rgba8.unwrap();
                assert_eq!(pixels.len(), 32 * 32 * 4);
                assert_ne!(&pixels[0..4], &[26, 51, 76, 255]);
                assert!(
                    pixels
                        .chunks_exact(4)
                        .any(|pixel| pixel[0] > 200 && pixel[2] < 120)
                );
            }
        }
    }
}
