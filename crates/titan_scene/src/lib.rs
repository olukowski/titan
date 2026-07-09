//! Scene graph types for Titan's text-first authoring pipeline.

pub mod tsf;

pub use tsf::{
    Diagnostic, DiagnosticSpan, Document, Position, QueryResult, Span, TsfError, TsfResult, Value,
    edit, fmt, parse, query, validate,
};

use titan_core::{EntityId, Transform};

/// A named scene containing entities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    name: String,
    entities: Vec<SceneEntity>,
}

impl Scene {
    /// Creates an empty scene with a stable human-readable name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entities: Vec::new(),
        }
    }

    /// Returns the scene name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds an entity to the scene.
    pub fn add_entity(&mut self, entity: SceneEntity) {
        self.entities.push(entity);
    }

    /// Returns all entities in insertion order.
    pub fn entities(&self) -> &[SceneEntity] {
        &self.entities
    }
}

/// An entity entry in a scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneEntity {
    /// Stable entity identifier.
    pub id: EntityId,
    /// Human-readable label for diagnostics and editor views.
    pub label: String,
    /// Initial transform for the entity.
    pub transform: Transform,
}

impl SceneEntity {
    /// Creates a scene entity with a default transform.
    pub fn new(id: EntityId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            transform: Transform::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Scene, SceneEntity};
    use titan_core::EntityId;

    #[test]
    fn scene_tracks_entities_in_insertion_order() {
        let mut scene = Scene::new("demo");

        scene.add_entity(SceneEntity::new(EntityId::from_raw(1), "camera"));
        scene.add_entity(SceneEntity::new(EntityId::from_raw(2), "player"));

        assert_eq!(scene.name(), "demo");
        assert_eq!(scene.entities()[0].label, "camera");
        assert_eq!(scene.entities()[1].label, "player");
    }
}
