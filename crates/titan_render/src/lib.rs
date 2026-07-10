//! Renderer-owned, headless rendering contracts.
//!
//! This crate owns the renderer API, the deterministic CPU render plan, and
//! the headless GPU context used by GPU-backed services.

use std::{collections::BTreeMap, fmt};

use serde::Serialize;
use titan_core::{
    Camera, CameraProjection, Component, DirectionalLight, EntityId, Material, MaterialModel, Mesh,
    Transform, Velocity, World,
};
use titan_math::Vec3;

mod geometry;
mod gpu;

pub use geometry::{
    AssetResolver, CUBE_V1_PATH, GeometryResolver, MeshAsset, MeshVertex, Submesh, cube_v1,
};
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
    pub const ASSET_UNAVAILABLE: &str = "RENDER_ASSET_UNAVAILABLE";
    pub const INVALID_GEOMETRY: &str = "RENDER_INVALID_GEOMETRY";
    pub const INVALID_MATERIAL: &str = "RENDER_INVALID_MATERIAL";
    pub const MISSING_MATERIAL: &str = "RENDER_MISSING_MATERIAL";
    pub const MISSING_LIGHT_TRANSFORM: &str = "RENDER_MISSING_LIGHT_TRANSFORM";
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

    fn with_path(code: &'static str, message: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: Some(path.into()),
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

/// Camera selection for a render request.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CameraSelection {
    Entity(EntityId),
    Name(String),
    /// Resolve the only camera in the world.
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
    pub output_size: Option<OutputSize>,
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

/// One planned fixture draw, including the entity's model transform.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DrawItem {
    pub entity: EntityId,
    pub model: Mat4,
    pub geometry: MeshAsset,
    pub material: Material,
}

/// Deterministic CPU render plan.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RenderScene {
    pub entities: Vec<ExtractedEntity>,
    pub draw_list: Vec<DrawItem>,
    pub directional_light: Option<DirectionalLightData>,
}

/// The selected light, including its entity transform-derived direction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DirectionalLightData {
    pub entity: EntityId,
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub illuminance: f32,
    pub ambient: f32,
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
    pub shader_version: u32,
}

pub const SHADER_VERSION: u32 = 1;

pub fn validate_material(material: &Material) -> ServiceResult<()> {
    if material
        .base_color
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(RenderError::new(
            error::INVALID_MATERIAL,
            "material.base_color must contain finite values in the range 0..=1",
        ));
    }
    match material.model {
        MaterialModel::Unlit => {
            if material.metallic.is_some() || material.roughness.is_some() {
                return Err(RenderError::new(
                    error::INVALID_MATERIAL,
                    "unlit materials must not specify metallic or roughness",
                ));
            }
        }
        MaterialModel::Pbr => {
            for (name, value) in [
                ("metallic", material.metallic),
                ("roughness", material.roughness),
            ] {
                let Some(value) = value else {
                    return Err(RenderError::new(
                        error::INVALID_MATERIAL,
                        format!("pbr materials require {name}"),
                    ));
                };
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    return Err(RenderError::new(
                        error::INVALID_MATERIAL,
                        format!("material.{name} must be finite and in the range 0..=1"),
                    ));
                }
            }
        }
    }
    Ok(())
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

    /// The inverse-transpose of a rigid transform's upper-left 3x3 matrix.
    /// Transforms currently contain rotation and translation only, so the
    /// inverse-transpose is the rotation itself.
    pub fn normal_from_transform(transform: &Transform) -> Self {
        Self::normal_from_model(Self::from_transform(transform))
    }

    pub fn normal_from_model(model: Self) -> Self {
        let model = model.0;
        let mut normal = Self::IDENTITY.0;
        for (row, values) in normal.iter_mut().take(3).enumerate() {
            for (column, value) in values.iter_mut().take(3).enumerate() {
                *value = model[row][column];
            }
        }
        Self(normal)
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
                if !(vertical_fov_degrees.is_finite()
                    && near.is_finite()
                    && far.is_finite()
                    && 0.0 < vertical_fov_degrees
                    && vertical_fov_degrees < 180.0
                    && 0.0 < near
                    && near < far)
                {
                    return Err(RenderError::new(
                        error::CAMERA_UNAVAILABLE,
                        "invalid perspective camera parameters",
                    ));
                }
                let f = 1.0 / (vertical_fov_degrees.to_radians() * 0.5).tan();
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
                if !(height.is_finite()
                    && near.is_finite()
                    && far.is_finite()
                    && height > 0.0
                    && 0.0 < near
                    && near < far)
                {
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
) -> ServiceResult<(EntityId, Camera, Transform)> {
    let candidates: Vec<EntityId> = match selection {
        CameraSelection::Entity(entity) => vec![*entity],
        CameraSelection::Name(name) => world.scene_entities_named(name),
        CameraSelection::Default => world
            .query::<&'static Camera>()
            .map(|(entity, _)| entity)
            .collect(),
    };
    if candidates.len() > 1 {
        return Err(RenderError::new(
            error::CAMERA_UNAVAILABLE,
            "camera selection is ambiguous",
        ));
    }
    if candidates.is_empty() {
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
    Ok((entity, *camera, *transform))
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
pub fn extract_scene(world: &World) -> ServiceResult<RenderScene> {
    let resolver = GeometryResolver::new(
        world
            .scene_base_dir()
            .unwrap_or_else(|| std::path::Path::new(".")),
    );
    extract_scene_with_resolver(world, |path| resolver.resolve(path))
}

fn extract_scene_with_resolver(
    world: &World,
    resolve: impl Fn(&str) -> ServiceResult<MeshAsset>,
) -> ServiceResult<RenderScene> {
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

    let directional_light = world
        .query::<&'static DirectionalLight>()
        .next()
        .map(|(entity, light)| {
            let transform = world
                .get::<Transform>(entity)
                .map_err(|error| {
                    RenderError::new(error::MISSING_LIGHT_TRANSFORM, error.to_string())
                })?
                .ok_or_else(|| {
                    RenderError::with_path(
                        error::MISSING_LIGHT_TRANSFORM,
                        "directional light requires a transform",
                        format!("entity:{}/transform", entity.raw()),
                    )
                })?;
            Ok(DirectionalLightData {
                entity,
                direction: forward_direction(transform),
                color: light.color,
                illuminance: light.illuminance,
                ambient: light.ambient,
            })
        })
        .transpose()?;
    let draw_list = world
        .query::<&'static Mesh>()
        .map(|(entity, mesh)| {
            let alias = mesh.geometry.ref_.strip_prefix("asset:").ok_or_else(|| {
                RenderError::with_path(
                    error::ASSET_UNAVAILABLE,
                    format!(
                        "mesh geometry reference is not an asset reference: '{}'",
                        mesh.geometry.ref_
                    ),
                    &mesh.geometry.ref_,
                )
            })?;
            let asset = world.scene_asset(alias).ok_or_else(|| {
                RenderError::with_path(
                    error::ASSET_UNAVAILABLE,
                    format!("unknown geometry asset reference '{alias}'"),
                    format!("asset:{alias}"),
                )
            })?;
            if asset.kind != "geometry" {
                return Err(RenderError::with_path(
                    error::ASSET_UNAVAILABLE,
                    format!("asset '{alias}' is not a geometry asset"),
                    format!("asset:{alias}"),
                ));
            }
            let geometry =
                select_submeshes(resolve(&asset.path)?, mesh.submeshes.as_deref(), alias)?;
            Ok(DrawItem {
                entity,
                model: world
                    .get::<Transform>(entity)
                    .ok()
                    .flatten()
                    .map_or(Mat4::IDENTITY, Mat4::from_transform),
                geometry,
                material: world
                    .get::<Material>(entity)
                    .map_err(|error| RenderError::new(error::MISSING_MATERIAL, error.to_string()))?
                    .ok_or_else(|| {
                        RenderError::with_path(
                            error::MISSING_MATERIAL,
                            "mesh requires a material",
                            format!("entity:{}/material", entity.raw()),
                        )
                    })
                    .copied()?,
            })
        })
        .collect::<ServiceResult<Vec<_>>>()?;
    Ok(RenderScene {
        entities,
        draw_list,
        directional_light,
    })
}

fn select_submeshes(
    mut geometry: MeshAsset,
    selection: Option<&[u32]>,
    alias: &str,
) -> ServiceResult<MeshAsset> {
    geometry.validate(alias)?;
    let Some(selection) = selection else {
        return Ok(geometry);
    };
    if selection.is_empty() {
        return Err(RenderError::with_path(
            error::INVALID_GEOMETRY,
            "submesh selection must not be empty",
            format!("asset:{alias}/submeshes"),
        ));
    }
    let mut selected = Vec::with_capacity(selection.len());
    for &index in selection {
        let submesh = geometry.submeshes.get(index as usize).ok_or_else(|| {
            RenderError::with_path(
                error::ASSET_UNAVAILABLE,
                format!("submesh index {index} is out of range for asset '{alias}'"),
                format!("asset:{alias}/submeshes"),
            )
        })?;
        selected.push(*submesh);
    }
    geometry.submeshes = selected;
    geometry.validate(alias)?;
    Ok(geometry)
}

fn forward_direction(transform: &Transform) -> [f32; 3] {
    let [x, y, z, w] = transform.rotation;
    [
        -2.0 * (x * z + y * w),
        -2.0 * (y * z - x * w),
        -1.0 + 2.0 * (x * x + y * y),
    ]
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
            components: RenderComponentRegistry::phase2(),
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

    /// Renders the deterministic CPU draw plan and, when requested, submits it
    /// to the GPU backend.
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
        let output_size = if let Some(output_size) = request.output_size {
            output_size
        } else {
            projection_viewport(&camera.1.projection).unwrap_or_default()
        };
        validate_output_size(output_size, max_texture_dimension)?;
        let scene = extract_scene(world)?;
        for item in &scene.draw_list {
            validate_material(&item.material)?;
            if matches!(item.material.model, MaterialModel::Pbr) {
                validate_normals(&item.geometry)?;
            }
        }
        let view_projection = Mat4::projection(
            &camera.1.projection,
            output_size.width as f32 / output_size.height as f32,
        )
        .map(|projection| projection * Mat4::view_from_transform(&camera.2))?;
        let stats = stats_for_scene(
            &scene,
            camera.0,
            world
                .query::<&'static DirectionalLight>()
                .count()
                .saturating_sub(usize::from(scene.directional_light.is_some())) as u32,
        );
        let rgba8 = match (&self.gpu, request.capture) {
            (Some(gpu), CaptureMode::Image) => Some(gpu.draw_plan(
                output_size,
                request.clear_color.0,
                view_projection.0,
                &scene.draw_list,
                scene.directional_light,
                [
                    camera.2.translation.x,
                    camera.2.translation.y,
                    camera.2.translation.z,
                ],
            )?),
            _ => None,
        };
        Ok(RenderResult {
            camera: Some(camera.0),
            output_size,
            stats,
            scene,
            rgba8,
        })
    }
}

fn stats_for_scene(
    scene: &RenderScene,
    camera: EntityId,
    ignored_directional_lights: u32,
) -> RenderStats {
    RenderStats {
        visible_meshes: scene.draw_list.len() as u32,
        draw_calls: scene
            .draw_list
            .iter()
            .map(|item| item.geometry.submeshes.len() as u32)
            .sum(),
        triangles: scene
            .draw_list
            .iter()
            .map(|item| item.geometry.triangle_count())
            .sum(),
        active_directional_lights: u32::from(scene.directional_light.is_some()),
        ignored_directional_lights,
        shader_version: SHADER_VERSION,
        material_models: scene
            .draw_list
            .iter()
            .fold(BTreeMap::new(), |mut counts, item| {
                let name = match item.material.model {
                    MaterialModel::Unlit => "unlit",
                    MaterialModel::Pbr => "pbr",
                };
                *counts.entry(name.to_owned()).or_insert(0) += 1;
                counts
            }),
        active_camera: Some(camera),
    }
}

fn validate_normals(mesh: &MeshAsset) -> ServiceResult<()> {
    if mesh.vertices.is_empty() {
        return Err(RenderError::new(
            error::MISSING_NORMALS,
            "pbr mesh has no normals",
        ));
    }
    if mesh.vertices.iter().any(|vertex| {
        vertex.normal.iter().any(|value| !value.is_finite())
            || (vertex.normal.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() > 1e-4
    }) {
        return Err(RenderError::new(
            error::INVALID_NORMALS,
            "pbr mesh normals must be finite unit vectors",
        ));
    }
    Ok(())
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

    fn world_with_camera() -> World {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        let entity = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
        world
            .insert(
                entity,
                Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)),
            )
            .unwrap();
        world
            .insert(
                entity,
                Camera {
                    projection: CameraProjection::Perspective {
                        vertical_fov_degrees: 60.0,
                        near: 0.1,
                        far: 100.0,
                        viewport: Some(titan_core::Viewport {
                            width: 320,
                            height: 200,
                        }),
                    },
                },
            )
            .unwrap();
        world
    }

    fn default_test_material() -> Material {
        Material {
            model: MaterialModel::Unlit,
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: None,
            roughness: None,
        }
    }

    #[test]
    fn extraction_is_stable_and_keeps_cpu_components() {
        let scene = extract_scene(&world_with_components()).unwrap();
        assert_eq!(scene.entities[0].entity, EntityId::from_raw(2));
        assert_eq!(scene.entities[1].entity, EntityId::from_raw(7));
        assert_eq!(scene.entities[1].velocity, Some(Vec3::new(4.0, 5.0, 6.0)));
        assert!(scene.draw_list.is_empty());
    }

    #[test]
    fn stats_reflect_the_current_empty_draw_plan() {
        let output = RenderService::cpu_only()
            .render(&world_with_camera(), RenderRequest::default())
            .unwrap();
        assert_eq!(output.stats.draw_calls, 0);
        assert_eq!(output.stats.triangles, 0);
        assert_eq!(output.stats.visible_meshes, 0);
        assert!(output.stats.material_models.is_empty());
    }

    #[test]
    fn default_camera_requires_exactly_one_camera() {
        let world = world_with_components();
        let error = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap_err();
        assert_eq!(error.code, error::CAMERA_UNAVAILABLE);

        let mut world = world_with_camera();
        let second = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        world.insert(second, Transform::default()).unwrap();
        world
            .insert(
                second,
                Camera {
                    projection: CameraProjection::Perspective {
                        vertical_fov_degrees: 60.0,
                        near: 0.1,
                        far: 100.0,
                        viewport: None,
                    },
                },
            )
            .unwrap();
        let error = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap_err();
        assert_eq!(error.code, error::CAMERA_UNAVAILABLE);
    }

    #[test]
    fn invalid_requests_use_stable_codes() {
        let world = world_with_camera();
        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    output_size: Some(OutputSize::new(0, 10)),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::INVALID_OUTPUT_SIZE);

        let error = RenderService::cpu_only()
            .render(
                &world,
                RenderRequest {
                    output_size: Some(OutputSize::new(10, 0)),
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
    fn invalid_projection_parameters_return_structured_errors() {
        for projection in [
            CameraProjection::Perspective {
                vertical_fov_degrees: 0.0,
                near: 0.1,
                far: 10.0,
                viewport: None,
            },
            CameraProjection::Perspective {
                vertical_fov_degrees: 180.0,
                near: 0.1,
                far: 10.0,
                viewport: None,
            },
            CameraProjection::Perspective {
                vertical_fov_degrees: f32::NAN,
                near: 0.1,
                far: 10.0,
                viewport: None,
            },
            CameraProjection::Perspective {
                vertical_fov_degrees: 60.0,
                near: f32::INFINITY,
                far: 10.0,
                viewport: None,
            },
            CameraProjection::Orthographic {
                height: 0.0,
                near: 0.1,
                far: 10.0,
                viewport: None,
            },
            CameraProjection::Orthographic {
                height: 10.0,
                near: 10.0,
                far: 10.0,
                viewport: None,
            },
        ] {
            let error = Mat4::projection(&projection, 1.0).unwrap_err();
            assert_eq!(error.code, error::CAMERA_UNAVAILABLE);
        }
    }

    #[test]
    fn explicit_output_size_overrides_camera_viewport() {
        let result = RenderService::cpu_only()
            .render(
                &world_with_camera(),
                RenderRequest {
                    output_size: Some(OutputSize::new(640, 480)),
                    ..RenderRequest::default()
                },
            )
            .unwrap();
        assert_eq!(result.output_size, OutputSize::new(640, 480));
    }

    #[test]
    fn named_camera_resolves_and_ambiguous_names_fail() {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        world.set_scene_asset(
            "fixture",
            titan_core::AssetEntry {
                path: CUBE_V1_PATH.into(),
                kind: "geometry".into(),
            },
        );
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
        world.set_scene_asset(
            "fixture",
            titan_core::AssetEntry {
                path: CUBE_V1_PATH.into(),
                kind: "geometry".into(),
            },
        );
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
            world.insert(entity, default_test_material()).unwrap();
        }
        let scene = extract_scene(&world).unwrap();
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
        assert_eq!(scene.draw_list[0].geometry, cube_v1());
    }

    #[test]
    fn tsf_load_bridge_extracts_builtin_cube_and_rejects_unknown_builtin() {
        let source = |path: &str| {
            format!(
                r#"{{
  tsf: 1,
  scene: {{ id: "scene:test" }},
  assets: {{ mesh: {{ path: "{path}", kind: "geometry" }} }},
  entities: [{{
    id: "entity:cube",
    components: {{
      transform: {{ translation: [0.0, 0.0, 0.0] }},
      mesh: {{ geometry: {{ ref: "asset:mesh" }} }},
      material: {{ model: "unlit", base_color: [1.0, 0.0, 0.0, 1.0] }},
    }},
  }}],
}}"#
            )
        };
        let registry = titan_scene::phase2_component_registry().unwrap();

        let document = titan_scene::parse(Some("/tmp/scene.tsf"), &source(CUBE_V1_PATH)).unwrap();
        let world = titan_scene::load_world(&document, registry.clone()).unwrap();
        assert_eq!(
            extract_scene(&world).unwrap().draw_list[0].geometry,
            cube_v1()
        );

        let document = titan_scene::parse(
            Some("/tmp/scene.tsf"),
            &source("__builtin__/geometry/not-a-real-version"),
        )
        .unwrap();
        let world = titan_scene::load_world(&document, registry).unwrap();
        let error = extract_scene(&world).unwrap_err();
        assert_eq!(error.code, error::UNKNOWN_BUILTIN);
    }

    #[test]
    fn extraction_resolves_relative_assets_from_the_tsf_directory() {
        let base = std::env::temp_dir().join(format!("titan-render-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let asset = base.join("mesh.mesh");
        std::fs::write(&asset, b"placeholder").unwrap();
        let file = base.join("scene.tsf");
        let source = r#"{
  tsf: 1,
  scene: { id: "scene:test" },
  assets: { mesh: { path: "mesh.mesh", kind: "geometry" } },
  entities: [{ id: "entity:cube", components: { mesh: { geometry: { ref: "asset:mesh" } } } }],
}"#;
        let document = titan_scene::parse(Some(file.to_str().unwrap()), source).unwrap();
        let world =
            titan_scene::load_world(&document, titan_scene::phase2_component_registry().unwrap())
                .unwrap();
        let error = extract_scene(&world).unwrap_err();
        assert!(error.message.contains(&asset.display().to_string()));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn extraction_reports_unknown_builtin_and_dangling_asset_references() {
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        let entity = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
        world
            .insert(
                entity,
                Mesh {
                    geometry: titan_core::AssetReference::new("asset:missing"),
                    submeshes: None,
                },
            )
            .unwrap();
        let error = extract_scene(&world).unwrap_err();
        assert_eq!(error.code, error::ASSET_UNAVAILABLE);
        assert_eq!(error.path.as_deref(), Some("asset:missing"));

        world.set_scene_asset(
            "missing",
            titan_core::AssetEntry {
                path: "__builtin__/geometry/not-a-real-version".into(),
                kind: "geometry".into(),
            },
        );
        let error = extract_scene(&world).unwrap_err();
        assert_eq!(error.code, error::UNKNOWN_BUILTIN);
        assert_eq!(
            error.path.as_deref(),
            Some("__builtin__/geometry/not-a-real-version")
        );
    }

    #[test]
    fn selected_submeshes_drive_exact_draw_and_triangle_stats() {
        let geometry = MeshAsset {
            vertices: vec![
                MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                };
                6
            ],
            indices: vec![0; 9],
            submeshes: vec![
                Submesh {
                    index_start: 0,
                    index_count: 3,
                },
                Submesh {
                    index_start: 3,
                    index_count: 6,
                },
            ],
        };
        let mut world = World::new(titan_core::phase2_component_registry().unwrap());
        world.set_scene_asset(
            "multi",
            titan_core::AssetEntry {
                path: "multi.mesh".into(),
                kind: "geometry".into(),
            },
        );
        let entity = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
        world.insert(entity, Transform::default()).unwrap();
        world
            .insert(
                entity,
                Mesh {
                    geometry: titan_core::AssetReference::new("asset:multi"),
                    submeshes: Some(vec![1]),
                },
            )
            .unwrap();
        world.insert(entity, default_test_material()).unwrap();
        let scene = extract_scene_with_resolver(&world, |path| {
            assert_eq!(path, "multi.mesh");
            Ok(geometry.clone())
        })
        .unwrap();
        assert_eq!(scene.draw_list[0].geometry.submeshes.len(), 1);
        let stats = stats_for_scene(&scene, EntityId::from_raw(9), 0);
        assert_eq!(stats.draw_calls, 1);
        assert_eq!(stats.triangles, 2);

        let error = select_submeshes(cube_v1(), Some(&[1]), "cube").unwrap_err();
        assert_eq!(error.code, error::ASSET_UNAVAILABLE);

        let error = select_submeshes(cube_v1(), Some(&[]), "cube").unwrap_err();
        assert_eq!(error.code, error::INVALID_GEOMETRY);
        assert_eq!(error.path.as_deref(), Some("asset:cube/submeshes"));
    }

    #[test]
    fn extraction_rejects_malformed_resolved_geometry() {
        let cases = [
            MeshAsset {
                vertices: vec![MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                }],
                indices: vec![0, 0, 0],
                submeshes: vec![Submesh {
                    index_start: u32::MAX,
                    index_count: 1,
                }],
            },
            MeshAsset {
                vertices: vec![MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                }],
                indices: vec![0, 0, 0],
                submeshes: vec![Submesh {
                    index_start: 1,
                    index_count: 3,
                }],
            },
            MeshAsset {
                vertices: vec![MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                }],
                indices: vec![0, 0, 0],
                submeshes: vec![Submesh {
                    index_start: 0,
                    index_count: 2,
                }],
            },
            MeshAsset {
                vertices: vec![MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                }],
                indices: vec![1, 0, 0],
                submeshes: vec![Submesh {
                    index_start: 0,
                    index_count: 3,
                }],
            },
            MeshAsset {
                vertices: Vec::new(),
                indices: Vec::new(),
                submeshes: Vec::new(),
            },
            MeshAsset {
                vertices: vec![MeshVertex {
                    position: [0.0; 3],
                    normal: [0.0; 3],
                    uv: [0.0; 2],
                }],
                indices: vec![0, 0, 0],
                submeshes: Vec::new(),
            },
        ];
        for geometry in cases {
            let mut world = World::new(titan_core::phase2_component_registry().unwrap());
            world.set_scene_asset(
                "bad",
                titan_core::AssetEntry {
                    path: "bad.mesh".into(),
                    kind: "geometry".into(),
                },
            );
            let entity = world.spawn_with_id(EntityId::from_raw(1)).unwrap();
            world
                .insert(
                    entity,
                    Mesh {
                        geometry: titan_core::AssetReference::new("asset:bad"),
                        submeshes: None,
                    },
                )
                .unwrap();
            let error = extract_scene_with_resolver(&world, |_| Ok(geometry.clone())).unwrap_err();
            assert_eq!(error.code, error::INVALID_GEOMETRY);
        }
    }

    #[test]
    fn stats_are_derived_from_the_stable_mesh_draw_plan() {
        let mut world = world_with_camera();
        world.set_scene_asset(
            "fixture",
            titan_core::AssetEntry {
                path: CUBE_V1_PATH.into(),
                kind: "geometry".into(),
            },
        );
        for id in [7, 3] {
            let entity = world.spawn_with_id(EntityId::from_raw(id)).unwrap();
            world.insert(entity, Transform::default()).unwrap();
            world
                .insert(
                    entity,
                    Mesh {
                        geometry: titan_core::AssetReference::new("asset:fixture"),
                        submeshes: None,
                    },
                )
                .unwrap();
            world.insert(entity, default_test_material()).unwrap();
        }
        let output = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap();
        assert_eq!(
            output
                .scene
                .draw_list
                .iter()
                .map(|item| item.entity.raw())
                .collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert_eq!(output.stats.visible_meshes, 2);
        assert_eq!(output.stats.draw_calls, 2);
        assert_eq!(output.stats.triangles, 24);
    }

    #[test]
    fn light_selection_and_material_validation_are_structured_and_stable() {
        let mut world = world_with_camera();
        for (id, ambient) in [(2, 0.1), (7, 0.2)] {
            let entity = world.spawn_with_id(EntityId::from_raw(id)).unwrap();
            world.insert(entity, Transform::default()).unwrap();
            world
                .insert(
                    entity,
                    DirectionalLight {
                        color: [1.0, 1.0, 1.0],
                        illuminance: 1.0,
                        ambient,
                    },
                )
                .unwrap();
        }
        let output = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap();
        assert_eq!(output.stats.active_directional_lights, 1);
        assert_eq!(output.stats.ignored_directional_lights, 1);
        assert_eq!(output.scene.directional_light.unwrap().entity.raw(), 2);
        assert_eq!(
            output.scene.directional_light.unwrap().direction,
            [0.0, 0.0, -1.0]
        );

        let invalid = Material {
            model: MaterialModel::Pbr,
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: Some(0.5),
            roughness: None,
        };
        assert_eq!(
            validate_material(&invalid).unwrap_err().code,
            error::INVALID_MATERIAL
        );
    }

    #[test]
    fn first_directional_light_without_transform_is_not_skipped() {
        let mut world = world_with_camera();
        let missing_transform = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        world
            .insert(
                missing_transform,
                DirectionalLight {
                    color: [1.0, 1.0, 1.0],
                    illuminance: 1.0,
                    ambient: 0.1,
                },
            )
            .unwrap();
        let higher = world.spawn_with_id(EntityId::from_raw(7)).unwrap();
        world.insert(higher, Transform::default()).unwrap();
        world
            .insert(
                higher,
                DirectionalLight {
                    color: [1.0, 1.0, 1.0],
                    illuminance: 1.0,
                    ambient: 0.2,
                },
            )
            .unwrap();

        let error = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap_err();
        assert_eq!(error.code, error::MISSING_LIGHT_TRANSFORM);
        assert_eq!(error.path.as_deref(), Some("entity:2/transform"));
    }

    #[test]
    fn mesh_without_material_is_a_structured_error() {
        let mut world = world_with_camera();
        world.set_scene_asset(
            "fixture",
            titan_core::AssetEntry {
                path: CUBE_V1_PATH.into(),
                kind: "geometry".into(),
            },
        );
        let mesh_entity = world.spawn_with_id(EntityId::from_raw(2)).unwrap();
        world.insert(mesh_entity, Transform::default()).unwrap();
        world
            .insert(
                mesh_entity,
                Mesh {
                    geometry: titan_core::AssetReference::new("asset:fixture"),
                    submeshes: None,
                },
            )
            .unwrap();

        let error = RenderService::cpu_only()
            .render(&world, RenderRequest::default())
            .unwrap_err();
        assert_eq!(error.code, error::MISSING_MATERIAL);
        assert_eq!(error.path.as_deref(), Some("entity:2/material"));
    }

    #[test]
    fn normal_matrix_rotates_normals_with_the_mesh() {
        let transform = Transform {
            translation: Vec3::new(4.0, 5.0, 6.0),
            rotation: [0.0, 0.70710677, 0.0, 0.70710677],
        };
        let model = Mat4::from_transform(&transform).0;
        let normal = Mat4::normal_from_transform(&transform).0;
        for row in 0..3 {
            for column in 0..3 {
                assert!((normal[row][column] - model[row][column]).abs() < 1e-5);
            }
        }
        let local_normal = [1.0, 0.0, 0.0];
        let world_normal = [
            model[0][0] * local_normal[0]
                + model[0][1] * local_normal[1]
                + model[0][2] * local_normal[2],
            model[1][0] * local_normal[0]
                + model[1][1] * local_normal[1]
                + model[1][2] * local_normal[2],
            model[2][0] * local_normal[0]
                + model[2][1] * local_normal[1]
                + model[2][2] * local_normal[2],
        ];
        let transformed_normal = [
            normal[0][0] * local_normal[0]
                + normal[0][1] * local_normal[1]
                + normal[0][2] * local_normal[2],
            normal[1][0] * local_normal[0]
                + normal[1][1] * local_normal[1]
                + normal[1][2] * local_normal[2],
            normal[2][0] * local_normal[0]
                + normal[2][1] * local_normal[1]
                + normal[2][2] * local_normal[2],
        ];
        for (actual, expected) in transformed_normal.into_iter().zip(world_normal) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(normal[0][3], 0.0);
        assert_eq!(normal[3][0], 0.0);
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
        world.set_scene_asset(
            "fixture",
            titan_core::AssetEntry {
                path: CUBE_V1_PATH.into(),
                kind: "geometry".into(),
            },
        );
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
        world.insert(mesh, default_test_material()).unwrap();
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
