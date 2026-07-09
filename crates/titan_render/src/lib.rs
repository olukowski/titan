//! Renderer-owned, headless rendering contracts.
//!
//! This crate intentionally has no graphics backend yet. It owns the API and
//! the deterministic CPU render plan that later GPU implementations consume.

use std::{collections::BTreeMap, fmt};

use serde::Serialize;
use titan_core::{Component, EntityId, Transform, Velocity, World};
use titan_math::Vec3;

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

    RenderScene {
        entities,
        draw_list: Vec::new(),
    }
}

/// Headless render service boundary. Backend ownership arrives in a later PR.
#[derive(Clone, Debug)]
pub struct RenderService {
    pub components: RenderComponentRegistry,
}

impl RenderService {
    /// Creates the stateless phase-1 CPU renderer.
    ///
    /// Step 2 will add a fallible, device-backed constructor (for adapter and
    /// device acquisition) and interior-mutable resource caches. Keeping the
    /// render operation shared-reference based now leaves that cache boundary
    /// explicit without making phase-1 construction pretend to acquire a GPU.
    pub fn new() -> Self {
        Self {
            components: RenderComponentRegistry::phase1(),
        }
    }

    /// Renders from the phase-1 stateless CPU plan. The `&self` boundary is
    /// intentional: step 2's caches will use interior mutability while its
    /// fallible device-backed constructor reports `RENDER_NO_ADAPTER`.
    pub fn render(&self, world: &World, request: RenderRequest) -> ServiceResult<RenderResult> {
        if request.output_size.width == 0 || request.output_size.height == 0 {
            return Err(RenderError::new(
                error::INVALID_OUTPUT_SIZE,
                "render output dimensions must be greater than zero",
            ));
        }
        if !matches!(request.camera, CameraSelection::Default) {
            return Err(RenderError::new(
                error::CAMERA_UNAVAILABLE,
                "camera components are not available in the phase 1 render extraction",
            ));
        }
        if request.capture == CaptureMode::Image {
            return Err(RenderError::new(
                error::CAPTURE_UNAVAILABLE,
                "image capture is not available before GPU backend initialization",
            ));
        }

        let scene = extract_scene(world);
        let stats = RenderStats {
            visible_meshes: scene.draw_list.len() as u32,
            draw_calls: scene.draw_list.len() as u32,
            ..RenderStats::default()
        };
        Ok(RenderResult {
            camera: None,
            output_size: request.output_size,
            stats,
            scene,
            rgba8: None,
        })
    }
}

impl Default for RenderService {
    fn default() -> Self {
        Self::new()
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
        let output = RenderService::new()
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
        let error = RenderService::new()
            .render(
                &world,
                RenderRequest {
                    output_size: OutputSize::new(0, 10),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::INVALID_OUTPUT_SIZE);

        let error = RenderService::new()
            .render(
                &world,
                RenderRequest {
                    output_size: OutputSize::new(10, 0),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::INVALID_OUTPUT_SIZE);

        let error = RenderService::new()
            .render(
                &world,
                RenderRequest {
                    camera: CameraSelection::Entity(EntityId::from_raw(7)),
                    ..RenderRequest::default()
                },
            )
            .unwrap_err();
        assert_eq!(error.code, error::CAMERA_UNAVAILABLE);

        let error = RenderService::new()
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
    fn default_service_registers_phase1_components() {
        assert_eq!(
            RenderService::default().components,
            RenderService::new().components
        );
        assert_eq!(RenderService::default().components.components().len(), 2);
        assert_eq!(
            RenderService::default().components.components()[0].registered_name,
            Transform::NAME
        );
        assert_eq!(
            RenderService::default().components.components()[1].registered_name,
            Velocity::NAME
        );
    }
}
