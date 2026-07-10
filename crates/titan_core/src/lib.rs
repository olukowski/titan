//! Deterministic core runtime types shared by Titan tools and engine crates.

use std::{
    any::{Any, TypeId, type_name},
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use titan_math::Vec3;

/// Result type used by Titan core APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by Titan core APIs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Error {
    EntityAlreadyExists(EntityId),
    EntityNotFound(EntityId),
    ComponentAlreadyRegistered(&'static str),
    ComponentNameConflict(&'static str),
    ComponentNotRegistered(String),
    ComponentTypeConflict(&'static str),
    ComponentStoreMissing(&'static str),
    ComponentSerialize { name: &'static str, message: String },
    ComponentDeserialize { name: &'static str, message: String },
    System { name: &'static str, message: String },
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntityAlreadyExists(id) => write!(f, "entity {} already exists", id.raw()),
            Self::EntityNotFound(id) => write!(f, "entity {} does not exist", id.raw()),
            Self::ComponentAlreadyRegistered(name) => {
                write!(f, "component {name} is already registered")
            }
            Self::ComponentNameConflict(name) => {
                write!(f, "component name {name} is registered for another type")
            }
            Self::ComponentNotRegistered(name) => write!(f, "component {name} is not registered"),
            Self::ComponentTypeConflict(name) => {
                write!(f, "component store {name} has an unexpected type")
            }
            Self::ComponentStoreMissing(name) => write!(f, "component store {name} is missing"),
            Self::ComponentSerialize { name, message } => {
                write!(f, "failed to serialize component {name}: {message}")
            }
            Self::ComponentDeserialize { name, message } => {
                write!(f, "failed to deserialize component {name}: {message}")
            }
            Self::System { name, message } => write!(f, "system {name} failed: {message}"),
        }
    }
}

/// A stable serialized identifier for an entity in a Titan world.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct EntityId(u64);

impl EntityId {
    /// Creates an entity identifier from a raw numeric value.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric value for serialization and diagnostics.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// A serializable ECS component.
pub trait Component: 'static + Send + Sync + Serialize + DeserializeOwned {
    /// Stable component name used in state dumps, scenes, and command logs.
    const NAME: &'static str;
    /// Component schema version for serialized payloads.
    const SCHEMA_VERSION: u32;
}

/// Registered metadata and type-erased helpers for a component.
#[derive(Clone)]
pub struct ComponentMeta {
    name: &'static str,
    schema_version: u32,
    type_id: TypeId,
    type_name: &'static str,
    serialize: fn(&dyn Any) -> Result<Value>,
    deserialize: fn(Value) -> Result<Box<dyn Any + Send + Sync>>,
    clone_value: fn(&dyn Any) -> Result<Box<dyn Any + Send + Sync>>,
    equals: fn(&dyn Any, &dyn Any) -> Result<bool>,
    debug: fn(&dyn Any) -> Result<String>,
}

impl ComponentMeta {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    pub fn serialize(&self, value: &dyn Any) -> Result<Value> {
        (self.serialize)(value)
    }

    pub fn deserialize(&self, value: Value) -> Result<Box<dyn Any + Send + Sync>> {
        (self.deserialize)(value)
    }

    pub fn clone_value(&self, value: &dyn Any) -> Result<Box<dyn Any + Send + Sync>> {
        (self.clone_value)(value)
    }

    pub fn equals(&self, left: &dyn Any, right: &dyn Any) -> Result<bool> {
        (self.equals)(left, right)
    }

    pub fn debug_value(&self, value: &dyn Any) -> Result<String> {
        (self.debug)(value)
    }
}

/// Registry of serializable/introspectable components.
#[derive(Clone, Default)]
pub struct ComponentRegistry {
    by_name: BTreeMap<&'static str, ComponentMeta>,
    by_type: BTreeMap<TypeId, &'static str>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self) -> Result<()>
    where
        T: Component + Clone + fmt::Debug + PartialEq,
    {
        let type_id = TypeId::of::<T>();
        if self.by_type.contains_key(&type_id) {
            return Err(Error::ComponentAlreadyRegistered(T::NAME));
        }
        if self.by_name.contains_key(T::NAME) {
            return Err(Error::ComponentNameConflict(T::NAME));
        }

        let meta = ComponentMeta {
            name: T::NAME,
            schema_version: T::SCHEMA_VERSION,
            type_id,
            type_name: type_name::<T>(),
            serialize: serialize_component::<T>,
            deserialize: deserialize_component::<T>,
            clone_value: clone_component::<T>,
            equals: equal_component::<T>,
            debug: debug_component::<T>,
        };
        self.by_name.insert(T::NAME, meta);
        self.by_type.insert(type_id, T::NAME);
        Ok(())
    }

    pub fn meta<T: Component>(&self) -> Result<&ComponentMeta> {
        self.meta_by_type(TypeId::of::<T>(), T::NAME)
    }

    pub fn meta_by_name(&self, name: &str) -> Result<&ComponentMeta> {
        self.by_name
            .get(name)
            .ok_or_else(|| Error::ComponentNotRegistered(name.to_string()))
    }

    fn meta_by_type(&self, type_id: TypeId, name: &'static str) -> Result<&ComponentMeta> {
        let registered_name = self
            .by_type
            .get(&type_id)
            .ok_or_else(|| Error::ComponentNotRegistered(name.to_string()))?;
        self.by_name
            .get(registered_name)
            .ok_or_else(|| Error::ComponentNotRegistered(name.to_string()))
    }
}

fn downcast_component<'a, T: Component>(value: &'a dyn Any, name: &'static str) -> Result<&'a T> {
    value
        .downcast_ref::<T>()
        .ok_or(Error::ComponentTypeConflict(name))
}

fn serialize_component<T: Component>(value: &dyn Any) -> Result<Value> {
    let component = downcast_component::<T>(value, T::NAME)?;
    serde_json::to_value(component).map_err(|source| Error::ComponentSerialize {
        name: T::NAME,
        message: source.to_string(),
    })
}

fn deserialize_component<T: Component>(value: Value) -> Result<Box<dyn Any + Send + Sync>> {
    serde_json::from_value::<T>(value)
        .map(|component| Box::new(component) as Box<dyn Any + Send + Sync>)
        .map_err(|source| Error::ComponentDeserialize {
            name: T::NAME,
            message: source.to_string(),
        })
}

fn clone_component<T>(value: &dyn Any) -> Result<Box<dyn Any + Send + Sync>>
where
    T: Component + Clone,
{
    Ok(Box::new(downcast_component::<T>(value, T::NAME)?.clone()))
}

fn equal_component<T>(left: &dyn Any, right: &dyn Any) -> Result<bool>
where
    T: Component + PartialEq,
{
    Ok(downcast_component::<T>(left, T::NAME)? == downcast_component::<T>(right, T::NAME)?)
}

fn debug_component<T>(value: &dyn Any) -> Result<String>
where
    T: Component + fmt::Debug,
{
    Ok(format!("{:?}", downcast_component::<T>(value, T::NAME)?))
}

struct ComponentStore {
    components: BTreeMap<EntityId, Box<dyn Any + Send + Sync>>,
}

impl ComponentStore {
    fn new() -> Self {
        Self {
            components: BTreeMap::new(),
        }
    }
}

/// Deterministic ECS world.
pub struct World {
    registry: ComponentRegistry,
    entities: BTreeSet<EntityId>,
    allocated_entities: BTreeSet<EntityId>,
    next_id: u64,
    stores_by_type: BTreeMap<TypeId, ComponentStore>,
    store_names: BTreeMap<&'static str, TypeId>,
    event_log: EventLog,
    frame: u64,
    seed: u64,
    scene_entity_ids: BTreeMap<String, u64>,
    scene_entity_names: BTreeMap<String, Vec<EntityId>>,
}

impl World {
    pub fn new(registry: ComponentRegistry) -> Self {
        Self {
            registry,
            entities: BTreeSet::new(),
            allocated_entities: BTreeSet::new(),
            next_id: 1,
            stores_by_type: BTreeMap::new(),
            store_names: BTreeMap::new(),
            event_log: EventLog::default(),
            frame: 0,
            seed: 0,
            scene_entity_ids: BTreeMap::new(),
            scene_entity_names: BTreeMap::new(),
        }
    }

    pub fn spawn(&mut self) -> EntityId {
        let id = self.allocate_entity_id();
        self.allocated_entities.insert(id);
        self.entities.insert(id);
        self.event_log.push(EventKind::EntitySpawned { entity: id });
        id
    }

    pub fn spawn_with_id(&mut self, id: EntityId) -> Result<EntityId> {
        if self.allocated_entities.contains(&id) {
            return Err(Error::EntityAlreadyExists(id));
        }
        self.next_id = self.next_id.max(id.raw().saturating_add(1));
        self.allocated_entities.insert(id);
        self.entities.insert(id);
        self.event_log.push(EventKind::EntitySpawned { entity: id });
        Ok(id)
    }

    fn allocate_entity_id(&mut self) -> EntityId {
        loop {
            let id = EntityId(self.next_id);
            if !self.allocated_entities.contains(&id) {
                self.next_id = self.next_id.saturating_add(1);
                return id;
            }
            if self.next_id == u64::MAX {
                panic!("entity id space exhausted");
            }
            self.next_id += 1;
        }
    }

    pub fn despawn(&mut self, id: EntityId) -> Result<()> {
        if !self.entities.remove(&id) {
            return Err(Error::EntityNotFound(id));
        }
        for store in self.stores_by_type.values_mut() {
            store.components.remove(&id);
        }
        self.event_log
            .push(EventKind::EntityDespawned { entity: id });
        Ok(())
    }

    pub fn insert<T: Component>(&mut self, id: EntityId, component: T) -> Result<()> {
        self.require_entity(id)?;
        self.require_store_mut::<T>()?
            .components
            .insert(id, Box::new(component));
        self.event_log.push(EventKind::ComponentInserted {
            entity: id,
            component: T::NAME.to_string(),
        });
        Ok(())
    }

    pub fn insert_serialized(
        &mut self,
        id: EntityId,
        component_name: &str,
        value: Value,
    ) -> Result<()> {
        self.require_entity(id)?;
        let meta = self.registry.meta_by_name(component_name)?.clone();
        let component = meta.deserialize(value)?;
        self.store_names.insert(meta.name, meta.type_id);
        self.stores_by_type
            .entry(meta.type_id)
            .or_insert_with(ComponentStore::new)
            .components
            .insert(id, component);
        self.event_log.push(EventKind::ComponentInserted {
            entity: id,
            component: meta.name.to_string(),
        });
        Ok(())
    }

    pub fn remove<T: Component>(&mut self, id: EntityId) -> Result<Option<T>> {
        self.require_entity(id)?;
        let removed = match self.require_store_mut::<T>()?.components.remove(&id) {
            Some(component) => Some(
                *component
                    .downcast::<T>()
                    .map_err(|_| Error::ComponentTypeConflict(T::NAME))?,
            ),
            None => None,
        };
        if removed.is_some() {
            self.event_log.push(EventKind::ComponentRemoved {
                entity: id,
                component: T::NAME.to_string(),
            });
        }
        Ok(removed)
    }

    pub fn get<T: Component>(&self, id: EntityId) -> Result<Option<&T>> {
        self.require_entity(id)?;
        self.store::<T>().map(|store| {
            store
                .and_then(|store| store.components.get(&id))
                .map(|component| {
                    component
                        .downcast_ref::<T>()
                        .ok_or(Error::ComponentTypeConflict(T::NAME))
                })
                .transpose()
        })?
    }

    pub fn get_mut<T: Component>(&mut self, id: EntityId) -> Result<Option<&mut T>> {
        self.require_entity(id)?;
        self.store_mut::<T>().map(|store| {
            store
                .and_then(|store| store.components.get_mut(&id))
                .map(|component| {
                    component
                        .downcast_mut::<T>()
                        .ok_or(Error::ComponentTypeConflict(T::NAME))
                })
                .transpose()
        })?
    }

    pub fn set<T: Component>(&mut self, id: EntityId, component: T) -> Result<()> {
        self.require_entity(id)?;
        self.require_store_mut::<T>()?
            .components
            .insert(id, Box::new(component));
        self.event_log.push(EventKind::ComponentSet {
            entity: id,
            component: T::NAME.to_string(),
        });
        Ok(())
    }

    pub fn query<'a, Q: Query + 'a>(&'a self) -> QueryIter<'a, Q> {
        QueryIter::new(Q::collect(self))
    }

    pub fn query_mut<'a, Q: QueryMut + 'a>(&'a mut self) -> QueryMutIter<'a, Q> {
        QueryMutIter::new(Q::collect(self))
    }

    pub fn dump_state(&self) -> Result<StateDump> {
        let mut entities = Vec::new();
        for entity in &self.entities {
            let mut components = BTreeMap::new();
            for (name, type_id) in &self.store_names {
                let store = self
                    .stores_by_type
                    .get(type_id)
                    .ok_or(Error::ComponentStoreMissing(name))?;
                if let Some(component) = store.components.get(entity) {
                    let meta = self.registry.meta_by_name(name)?;
                    components.insert(
                        name.to_string(),
                        ComponentDump {
                            schema_version: meta.schema_version,
                            value: meta.serialize(component.as_ref())?,
                        },
                    );
                }
            }
            entities.push(EntityDump {
                id: entity.raw(),
                components,
            });
        }
        let entity_ids = self
            .scene_entity_ids
            .iter()
            .filter(|(_, entity)| self.entities.contains(&EntityId::from_raw(**entity)))
            .map(|(scene_id, entity)| (scene_id.clone(), *entity))
            .collect();
        Ok(StateDump {
            frame: self.frame,
            seed: self.seed,
            entity_ids,
            entities,
        })
    }

    pub fn event_log(&self) -> &EventLog {
        &self.event_log
    }

    pub fn set_runtime_metadata(&mut self, frame: u64, seed: u64) {
        self.frame = frame;
        self.seed = seed;
    }

    pub fn bind_scene_entity_id(
        &mut self,
        scene_id: impl Into<String>,
        entity: EntityId,
    ) -> Result<()> {
        self.require_entity(entity)?;
        self.scene_entity_ids.insert(scene_id.into(), entity.raw());
        Ok(())
    }

    /// Associates a serialized entity name with an entity for name-based APIs.
    pub fn bind_scene_entity_name(
        &mut self,
        name: impl Into<String>,
        entity: EntityId,
    ) -> Result<()> {
        self.require_entity(entity)?;
        let entities = self.scene_entity_names.entry(name.into()).or_default();
        if !entities.contains(&entity) {
            entities.push(entity);
        }
        Ok(())
    }

    /// Returns all entities carrying a serialized name, in stable ID order.
    pub fn scene_entities_named(&self, name: &str) -> Vec<EntityId> {
        let mut entities = self
            .scene_entity_names
            .get(name)
            .cloned()
            .unwrap_or_default();
        entities.sort_unstable();
        entities.retain(|entity| self.entities.contains(entity));
        entities
    }

    fn require_entity(&self, id: EntityId) -> Result<()> {
        if self.entities.contains(&id) {
            Ok(())
        } else {
            Err(Error::EntityNotFound(id))
        }
    }

    fn ensure_store<T: Component>(&mut self) -> Result<()> {
        let meta = self.registry.meta::<T>()?;
        if !self.stores_by_type.contains_key(&meta.type_id) {
            self.store_names.insert(meta.name, meta.type_id);
            self.stores_by_type
                .insert(meta.type_id, ComponentStore::new());
        }
        Ok(())
    }

    fn require_store_mut<T: Component>(&mut self) -> Result<&mut ComponentStore> {
        self.ensure_store::<T>()?;
        self.stores_by_type
            .get_mut(&TypeId::of::<T>())
            .ok_or(Error::ComponentStoreMissing(T::NAME))
    }

    fn store<T: Component>(&self) -> Result<Option<&ComponentStore>> {
        self.registry.meta::<T>()?;
        Ok(self.stores_by_type.get(&TypeId::of::<T>()))
    }

    fn store_mut<T: Component>(&mut self) -> Result<Option<&mut ComponentStore>> {
        self.registry.meta::<T>()?;
        Ok(self.stores_by_type.get_mut(&TypeId::of::<T>()))
    }
}

/// Immutable deterministic query.
pub trait Query {
    type Item<'a>
    where
        Self: 'a;

    fn collect<'w>(world: &'w World) -> Vec<(EntityId, Self::Item<'w>)>;
}

/// Mutable deterministic query.
pub trait QueryMut {
    type Item<'a>
    where
        Self: 'a;

    fn collect<'w>(world: &'w mut World) -> Vec<(EntityId, Self::Item<'w>)>;
}

/// Query iterator that yields matching entities in ascending `EntityId` order.
pub struct QueryIter<'a, Q: Query + 'a> {
    inner: std::vec::IntoIter<(EntityId, Q::Item<'a>)>,
}

impl<'a, Q: Query + 'a> QueryIter<'a, Q> {
    fn new(items: Vec<(EntityId, Q::Item<'a>)>) -> Self {
        Self {
            inner: items.into_iter(),
        }
    }
}

impl<'a, Q: Query + 'a> Iterator for QueryIter<'a, Q> {
    type Item = (EntityId, Q::Item<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Mutable query iterator that yields matching entities in ascending `EntityId` order.
pub struct QueryMutIter<'a, Q: QueryMut + 'a> {
    inner: std::vec::IntoIter<(EntityId, Q::Item<'a>)>,
}

impl<'a, Q: QueryMut + 'a> QueryMutIter<'a, Q> {
    fn new(items: Vec<(EntityId, Q::Item<'a>)>) -> Self {
        Self {
            inner: items.into_iter(),
        }
    }
}

impl<'a, Q: QueryMut + 'a> Iterator for QueryMutIter<'a, Q> {
    type Item = (EntityId, Q::Item<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T: Component> Query for &'static T {
    type Item<'a>
        = &'a T
    where
        Self: 'a;

    fn collect<'w>(world: &'w World) -> Vec<(EntityId, Self::Item<'w>)> {
        match world.store::<T>() {
            Ok(Some(store)) => store
                .components
                .iter()
                .filter(|(id, _)| world.entities.contains(id))
                .filter_map(|(id, component)| {
                    component.downcast_ref::<T>().map(|value| (*id, value))
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl<A: Component, B: Component> Query for (&'static A, &'static B) {
    type Item<'a>
        = (&'a A, &'a B)
    where
        Self: 'a;

    fn collect<'w>(world: &'w World) -> Vec<(EntityId, Self::Item<'w>)> {
        let Ok(Some(left)) = world.store::<A>() else {
            return Vec::new();
        };
        let Ok(Some(right)) = world.store::<B>() else {
            return Vec::new();
        };
        left.components
            .iter()
            .filter(|(id, _)| world.entities.contains(id))
            .filter_map(|(id, left_component)| {
                let left_value = left_component.downcast_ref::<A>()?;
                let right_value = right.components.get(id)?.downcast_ref::<B>()?;
                Some((*id, (left_value, right_value)))
            })
            .collect()
    }
}

impl<T: Component> QueryMut for &'static mut T {
    type Item<'a>
        = &'a mut T
    where
        Self: 'a;

    fn collect<'w>(world: &'w mut World) -> Vec<(EntityId, Self::Item<'w>)> {
        if world.ensure_store::<T>().is_err() {
            return Vec::new();
        }
        let entities = &world.entities;
        match world.stores_by_type.get_mut(&TypeId::of::<T>()) {
            Some(store) => store
                .components
                .iter_mut()
                .filter(|(id, _)| entities.contains(id))
                .filter_map(|(id, component)| {
                    component.downcast_mut::<T>().map(|value| (*id, value))
                })
                .collect(),
            None => Vec::new(),
        }
    }
}

impl<A: Component, B: Component> QueryMut for (&'static mut A, &'static B) {
    type Item<'a>
        = (&'a mut A, &'a B)
    where
        Self: 'a;

    fn collect<'w>(world: &'w mut World) -> Vec<(EntityId, Self::Item<'w>)> {
        if TypeId::of::<A>() == TypeId::of::<B>() {
            return Vec::new();
        }
        if world.ensure_store::<A>().is_err() || world.ensure_store::<B>().is_err() {
            return Vec::new();
        }

        let left_ptr = match world.stores_by_type.get_mut(&TypeId::of::<A>()) {
            Some(store) => store as *mut ComponentStore,
            None => return Vec::new(),
        };
        let right_ptr = match world.stores_by_type.get(&TypeId::of::<B>()) {
            Some(store) => store as *const ComponentStore,
            None => return Vec::new(),
        };
        let entities = &world.entities as *const BTreeSet<EntityId>;

        // The stores are disjoint because equal TypeIds returned above. Iteration is still
        // ordered by the mutable store's BTreeMap and filtered against the immutable store.
        unsafe {
            let entities = &*entities;
            let left = &mut *left_ptr;
            let right = &*right_ptr;
            left.components
                .iter_mut()
                .filter(|(id, _)| entities.contains(id))
                .filter_map(|(id, left_component)| {
                    let left_value = left_component.downcast_mut::<A>()?;
                    let right_value = right.components.get(id)?.downcast_ref::<B>()?;
                    Some((*id, (left_value, right_value)))
                })
                .collect()
        }
    }
}

/// Structured deterministic world dump.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StateDump {
    pub frame: u64,
    pub seed: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub entity_ids: BTreeMap<String, u64>,
    pub entities: Vec<EntityDump>,
}

/// Entity entry in a state dump.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EntityDump {
    pub id: u64,
    pub components: BTreeMap<String, ComponentDump>,
}

/// Component entry in a state dump.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComponentDump {
    pub schema_version: u32,
    pub value: Value,
}

/// Ordered event log.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventLog {
    records: Vec<EventRecord>,
    next_sequence: u64,
}

impl EventLog {
    pub fn records(&self) -> &[EventRecord] {
        &self.records
    }

    pub fn to_jsonl(&self) -> Result<String> {
        let mut output = String::new();
        for record in &self.records {
            let line =
                serde_json::to_string(record).map_err(|source| Error::ComponentSerialize {
                    name: "titan.core.EventLog",
                    message: source.to_string(),
                })?;
            output.push_str(&line);
            output.push('\n');
        }
        Ok(output)
    }

    fn push(&mut self, kind: EventKind) {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.records.push(EventRecord { sequence, kind });
    }
}

/// A single ordered event log record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventRecord {
    pub sequence: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

/// Serializable ECS event kinds.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    EntitySpawned { entity: EntityId },
    EntityDespawned { entity: EntityId },
    ComponentInserted { entity: EntityId, component: String },
    ComponentSet { entity: EntityId, component: String },
    ComponentRemoved { entity: EntityId, component: String },
    SystemError { system: String, message: String },
}

/// Deterministic fixed-step context passed to systems.
#[derive(Clone, Copy, Debug)]
pub struct FixedStepContext {
    pub fixed_dt: f32,
    pub frame: u64,
    pub seed: u64,
    pub rng: DeterministicRng,
}

impl FixedStepContext {
    pub fn new(fixed_dt: f32, frame: u64, seed: u64) -> Self {
        Self {
            fixed_dt,
            frame,
            seed,
            rng: DeterministicRng::for_frame(seed, frame),
        }
    }
}

/// Small deterministic PRNG with a fixed xorshift64* algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn for_frame(seed: u64, frame: u64) -> Self {
        Self::new(splitmix64(seed ^ frame.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
    }

    fn for_system(seed: u64, frame: u64, system_index: u64) -> Self {
        Self::new(splitmix64(
            seed ^ frame.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ system_index.wrapping_mul(0xbf58_476d_1ce4_e5b9),
        ))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / 16_777_216.0
    }

    pub fn seed(&self) -> u64 {
        self.state
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub type SystemFn = fn(&mut SystemWorld<'_>, &mut CommandBuffer, FixedStepContext) -> Result<()>;

/// Default seed used by deterministic Phase 1 headless runs.
pub const DEFAULT_RUN_SEED: u64 = 0x5449_5441_4e50_4831;

/// Default fixed timestep used by deterministic Phase 1 headless runs.
pub const DEFAULT_FIXED_DT: f32 = 1.0 / 60.0;

/// Deterministic insertion-ordered system schedule.
#[derive(Default)]
pub struct Schedule {
    systems: Vec<(&'static str, SystemFn)>,
}

impl Schedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_system(&mut self, name: &'static str, system: SystemFn) {
        self.systems.push((name, system));
    }

    pub fn run_fixed_step(&mut self, world: &mut World, ctx: FixedStepContext) -> Result<()> {
        world.frame = ctx.frame;
        world.seed = ctx.seed;
        for (system_index, (name, system)) in self.systems.iter().enumerate() {
            let mut commands = CommandBuffer::for_world(world);
            let mut system_ctx = ctx;
            system_ctx.rng = DeterministicRng::for_system(ctx.seed, ctx.frame, system_index as u64);
            let result = {
                let mut system_world = SystemWorld { world };
                system(&mut system_world, &mut commands, system_ctx)
            };
            if let Err(error) = result {
                let message = error.to_string();
                return Err(log_system_error(world, name, message));
            }
            if let Err(error) = commands.apply(world) {
                return Err(log_system_error(world, name, error.to_string()));
            }
        }
        Ok(())
    }
}

fn log_system_error(world: &mut World, name: &'static str, message: String) -> Error {
    world.event_log.push(EventKind::SystemError {
        system: name.to_string(),
        message: message.clone(),
    });
    Error::System { name, message }
}

/// Restricted system view of the world.
pub struct SystemWorld<'a> {
    world: &'a mut World,
}

impl SystemWorld<'_> {
    pub fn get<T: Component>(&self, id: EntityId) -> Result<Option<&T>> {
        self.world.get::<T>(id)
    }

    pub fn get_mut<T: Component>(&mut self, id: EntityId) -> Result<Option<&mut T>> {
        self.world.get_mut::<T>(id)
    }

    pub fn query<'a, Q: Query + 'a>(&'a self) -> QueryIter<'a, Q> {
        self.world.query::<Q>()
    }

    pub fn query_mut<'a, Q: QueryMut + 'a>(&'a mut self) -> QueryMutIter<'a, Q> {
        self.world.query_mut::<Q>()
    }
}

type Command = Box<dyn FnOnce(&mut World) -> Result<()> + Send>;

/// Structural changes recorded by systems and applied after each system.
pub struct CommandBuffer {
    commands: Vec<Command>,
    allocated_entities: BTreeSet<EntityId>,
    next_id: u64,
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self {
            commands: Vec::new(),
            allocated_entities: BTreeSet::new(),
            next_id: 1,
        }
    }
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    fn for_world(world: &World) -> Self {
        Self {
            commands: Vec::new(),
            allocated_entities: world.allocated_entities.clone(),
            next_id: world.next_id,
        }
    }

    pub fn spawn(&mut self) -> EntityId {
        let id = self.allocate_entity_id();
        self.commands
            .push(Box::new(move |world| world.spawn_with_id(id).map(|_| ())));
        id
    }

    pub fn spawn_with_id(&mut self, id: EntityId) {
        self.next_id = self.next_id.max(id.raw().saturating_add(1));
        self.allocated_entities.insert(id);
        self.commands
            .push(Box::new(move |world| world.spawn_with_id(id).map(|_| ())));
    }

    pub fn despawn(&mut self, id: EntityId) {
        self.commands.push(Box::new(move |world| world.despawn(id)));
    }

    pub fn insert<T: Component>(&mut self, id: EntityId, component: T) {
        self.commands
            .push(Box::new(move |world| world.insert(id, component)));
    }

    pub fn remove<T: Component>(&mut self, id: EntityId) {
        self.commands
            .push(Box::new(move |world| world.remove::<T>(id).map(|_| ())));
    }

    pub fn set<T: Component>(&mut self, id: EntityId, component: T) {
        self.commands
            .push(Box::new(move |world| world.set(id, component)));
    }

    fn allocate_entity_id(&mut self) -> EntityId {
        loop {
            let id = EntityId(self.next_id);
            if !self.allocated_entities.contains(&id) {
                self.next_id = self.next_id.saturating_add(1);
                self.allocated_entities.insert(id);
                return id;
            }
            if self.next_id == u64::MAX {
                panic!("entity id space exhausted");
            }
            self.next_id += 1;
        }
    }

    fn apply(self, world: &mut World) -> Result<()> {
        for command in self.commands {
            command(world)?;
        }
        Ok(())
    }
}

/// A minimal transform component for positioning entities.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Transform {
    /// World-space translation.
    pub translation: Vec3,
    /// World-space orientation as a normalized `[x, y, z, w]` quaternion.
    #[serde(default = "identity_rotation")]
    pub rotation: [f32; 4],
}

const fn identity_rotation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

impl Transform {
    /// Creates a transform at the provided translation.
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: identity_rotation(),
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: identity_rotation(),
        }
    }
}

impl Component for Transform {
    const NAME: &'static str = "titan.core.Transform";
    const SCHEMA_VERSION: u32 = 2;
}

/// A minimal velocity component used by deterministic simulation tests.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Velocity {
    pub linear: Vec3,
}

/// Projection parameters for an entity camera.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "projection", rename_all = "lowercase")]
pub enum CameraProjection {
    Perspective {
        vertical_fov_degrees: f32,
        near: f32,
        far: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        viewport: Option<Viewport>,
    },
    Orthographic {
        height: f32,
        near: f32,
        far: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        viewport: Option<Viewport>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Camera {
    #[serde(flatten)]
    pub projection: CameraProjection,
}

impl Component for Camera {
    const NAME: &'static str = "titan.core.Camera";
    const SCHEMA_VERSION: u32 = 1;
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionalLight {
    pub color: [f32; 3],
    pub illuminance: f32,
    pub ambient: f32,
}

impl Component for DirectionalLight {
    const NAME: &'static str = "titan.core.DirectionalLight";
    const SCHEMA_VERSION: u32 = 1;
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssetReference {
    #[serde(rename = "ref")]
    pub ref_: String,
}

impl AssetReference {
    pub fn new(reference: impl Into<String>) -> Self {
        Self {
            ref_: reference.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    pub geometry: AssetReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submeshes: Option<Vec<u32>>,
}

impl Component for Mesh {
    const NAME: &'static str = "titan.core.Mesh";
    const SCHEMA_VERSION: u32 = 1;
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MaterialModel {
    Unlit,
    Pbr,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    pub model: MaterialModel,
    pub base_color: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metallic: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roughness: Option<f32>,
}

impl Component for Material {
    const NAME: &'static str = "titan.core.Material";
    const SCHEMA_VERSION: u32 = 1;
}

impl Velocity {
    pub const fn new(linear: Vec3) -> Self {
        Self { linear }
    }
}

impl Component for Velocity {
    const NAME: &'static str = "titan.core.Velocity";
    const SCHEMA_VERSION: u32 = 1;
}

/// Creates the built-in Phase 1 component registry.
pub fn phase1_component_registry() -> Result<ComponentRegistry> {
    let mut registry = ComponentRegistry::new();
    registry.register::<Transform>()?;
    registry.register::<Velocity>()?;
    Ok(registry)
}

/// Creates the built-in Phase 2 component registry.
pub fn phase2_component_registry() -> Result<ComponentRegistry> {
    let mut registry = phase1_component_registry()?;
    registry.register::<Camera>()?;
    registry.register::<DirectionalLight>()?;
    registry.register::<Mesh>()?;
    registry.register::<Material>()?;
    Ok(registry)
}

/// Phase 1 deterministic velocity integration.
pub fn velocity_integration_system(
    world: &mut SystemWorld<'_>,
    _commands: &mut CommandBuffer,
    ctx: FixedStepContext,
) -> Result<()> {
    for (_, (transform, velocity)) in world.query_mut::<(&mut Transform, &Velocity)>() {
        transform.translation.x += velocity.linear.x * ctx.fixed_dt;
        transform.translation.y += velocity.linear.y * ctx.fixed_dt;
        transform.translation.z += velocity.linear.z * ctx.fixed_dt;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register::<Transform>().unwrap();
        registry.register::<Velocity>().unwrap();
        registry
    }

    fn integrate(
        world: &mut SystemWorld<'_>,
        commands: &mut CommandBuffer,
        ctx: FixedStepContext,
    ) -> Result<()> {
        for (_, (transform, velocity)) in world.query_mut::<(&mut Transform, &Velocity)>() {
            transform.translation.x += velocity.linear.x * ctx.fixed_dt;
            transform.translation.y += velocity.linear.y * ctx.fixed_dt;
            transform.translation.z += velocity.linear.z * ctx.fixed_dt;
        }
        if ctx.frame == 2 {
            commands.spawn_with_id(EntityId::from_raw(100));
        }
        Ok(())
    }

    fn invalid_command(
        _world: &mut SystemWorld<'_>,
        commands: &mut CommandBuffer,
        _ctx: FixedStepContext,
    ) -> Result<()> {
        commands.despawn(EntityId::from_raw(999));
        Ok(())
    }

    fn spawn_and_initialize(
        _world: &mut SystemWorld<'_>,
        commands: &mut CommandBuffer,
        _ctx: FixedStepContext,
    ) -> Result<()> {
        let entity = commands.spawn();
        commands.insert(
            entity,
            Transform::from_translation(Vec3::new(4.0, 5.0, 6.0)),
        );
        Ok(())
    }

    fn write_random_x(
        world: &mut SystemWorld<'_>,
        _commands: &mut CommandBuffer,
        ctx: FixedStepContext,
    ) -> Result<()> {
        let mut rng = ctx.rng;
        for (_, transform) in world.query_mut::<&mut Transform>() {
            transform.translation.x = rng.next_f32();
        }
        Ok(())
    }

    fn write_random_y(
        world: &mut SystemWorld<'_>,
        _commands: &mut CommandBuffer,
        ctx: FixedStepContext,
    ) -> Result<()> {
        let mut rng = ctx.rng;
        for (_, transform) in world.query_mut::<&mut Transform>() {
            transform.translation.y = rng.next_f32();
        }
        Ok(())
    }

    #[test]
    fn entity_id_round_trips_raw_value() {
        assert_eq!(EntityId::from_raw(42).raw(), 42);
    }

    #[test]
    fn default_transform_starts_at_origin() {
        assert_eq!(Transform::default().translation, Vec3::ZERO);
    }

    #[test]
    fn fixed_step_rng_is_repeatable_but_frame_distinct() {
        let mut first_frame = FixedStepContext::new(0.5, 1, 1234).rng;
        let mut same_frame = FixedStepContext::new(0.5, 1, 1234).rng;
        let mut next_frame = FixedStepContext::new(0.5, 2, 1234).rng;

        assert_eq!(first_frame.next_u64(), same_frame.next_u64());
        assert_ne!(first_frame.next_u64(), next_frame.next_u64());
    }

    #[test]
    fn registry_lookup_accepts_runtime_component_names() {
        let registry = registry();
        let name = String::from(Transform::NAME);

        let meta = registry.meta_by_name(&name).unwrap();

        assert_eq!(meta.name(), Transform::NAME);
    }

    #[test]
    fn same_setup_seed_and_frames_produce_byte_identical_dumps() {
        fn run() -> String {
            let mut world = World::new(registry());
            let entity = world.spawn_with_id(EntityId::from_raw(10)).unwrap();
            world.insert(entity, Transform::default()).unwrap();
            world
                .insert(entity, Velocity::new(Vec3::new(1.0, 2.0, 3.0)))
                .unwrap();

            let mut schedule = Schedule::new();
            schedule.add_system("integrate", integrate);
            for frame in 1..=5 {
                schedule
                    .run_fixed_step(&mut world, FixedStepContext::new(0.5, frame, 1234))
                    .unwrap();
            }
            serde_json::to_string(&world.dump_state().unwrap()).unwrap()
        }

        assert_eq!(run(), run());
    }

    #[test]
    fn query_iteration_is_ascending_after_mixed_operations() {
        let mut world = World::new(registry());
        for id in [5, 2, 8, 1] {
            world.spawn_with_id(EntityId::from_raw(id)).unwrap();
        }
        world
            .insert(
                EntityId::from_raw(5),
                Transform::from_translation(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        world
            .insert(
                EntityId::from_raw(1),
                Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        world
            .insert(
                EntityId::from_raw(8),
                Transform::from_translation(Vec3::new(8.0, 0.0, 0.0)),
            )
            .unwrap();
        world.remove::<Transform>(EntityId::from_raw(5)).unwrap();
        world.despawn(EntityId::from_raw(2)).unwrap();
        world
            .insert(
                EntityId::from_raw(5),
                Transform::from_translation(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();

        let ids: Vec<u64> = world
            .query::<&Transform>()
            .map(|(id, _)| id.raw())
            .collect();
        assert_eq!(ids, vec![1, 5, 8]);
    }

    #[test]
    fn state_dump_entity_and_component_order_is_stable() {
        let mut world = World::new(registry());
        let high = world.spawn_with_id(EntityId::from_raw(9)).unwrap();
        let low = world.spawn_with_id(EntityId::from_raw(3)).unwrap();
        world
            .insert(high, Velocity::new(Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        world.insert(high, Transform::default()).unwrap();
        world.insert(low, Transform::default()).unwrap();

        let dump = world.dump_state().unwrap();
        assert_eq!(
            dump.entities
                .iter()
                .map(|entity| entity.id)
                .collect::<Vec<_>>(),
            vec![3, 9]
        );
        assert_eq!(
            dump.entities[1]
                .components
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                "titan.core.Transform".to_string(),
                "titan.core.Velocity".to_string()
            ]
        );
    }

    #[test]
    fn state_dump_omits_scene_ids_for_despawned_entities() {
        let mut world = World::new(registry());
        let live = world.spawn_with_id(EntityId::from_raw(3)).unwrap();
        let despawned = world.spawn_with_id(EntityId::from_raw(9)).unwrap();
        world.bind_scene_entity_id("entity:live", live).unwrap();
        world
            .bind_scene_entity_id("entity:despawned", despawned)
            .unwrap();
        world.despawn(despawned).unwrap();

        let dump = world.dump_state().unwrap();

        assert_eq!(dump.entity_ids.get("entity:live"), Some(&3));
        assert!(!dump.entity_ids.contains_key("entity:despawned"));
        assert_eq!(
            dump.entities
                .iter()
                .map(|entity| entity.id)
                .collect::<Vec<_>>(),
            vec![3]
        );
    }

    #[test]
    fn duplicate_scene_entity_name_binding_returns_one_entity() {
        let mut world = World::new(registry());
        let entity = world.spawn_with_id(EntityId::from_raw(3)).unwrap();

        world.bind_scene_entity_name("player", entity).unwrap();
        world.bind_scene_entity_name("player", entity).unwrap();

        assert_eq!(world.scene_entities_named("player"), vec![entity]);
    }

    #[test]
    fn event_log_order_is_stable_for_core_operations() {
        let mut world = World::new(registry());
        let entity = world.spawn_with_id(EntityId::from_raw(7)).unwrap();
        world.insert(entity, Transform::default()).unwrap();
        world
            .set(
                entity,
                Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        world.remove::<Transform>(entity).unwrap();
        world.despawn(entity).unwrap();

        let events: Vec<&'static str> = world
            .event_log()
            .records()
            .iter()
            .map(|record| match record.kind {
                EventKind::EntitySpawned { .. } => "entity_spawned",
                EventKind::ComponentInserted { .. } => "component_inserted",
                EventKind::ComponentSet { .. } => "component_set",
                EventKind::ComponentRemoved { .. } => "component_removed",
                EventKind::EntityDespawned { .. } => "entity_despawned",
                EventKind::SystemError { .. } => "system_error",
            })
            .collect();
        assert_eq!(
            events,
            vec![
                "entity_spawned",
                "component_inserted",
                "component_set",
                "component_removed",
                "entity_despawned"
            ]
        );
    }

    #[test]
    fn runtime_spawns_skip_reserved_max_id_and_exhaust_cleanly() {
        let mut world = World::new(registry());
        world
            .spawn_with_id(EntityId::from_raw(u64::MAX - 1))
            .unwrap();

        let max = world.spawn();
        assert_eq!(max, EntityId::from_raw(u64::MAX));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.spawn();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn runtime_spawns_do_not_reuse_reserved_max_id() {
        let mut world = World::new(registry());
        world.spawn_with_id(EntityId::from_raw(u64::MAX)).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.spawn();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn schedule_logs_command_buffer_failures_as_system_errors() {
        let mut world = World::new(registry());
        let mut schedule = Schedule::new();
        schedule.add_system("invalid_command", invalid_command);

        let error = schedule
            .run_fixed_step(&mut world, FixedStepContext::new(0.25, 1, 99))
            .unwrap_err();

        assert_eq!(
            error,
            Error::System {
                name: "invalid_command",
                message: "entity 999 does not exist".to_string()
            }
        );
        assert_eq!(world.event_log().records().len(), 1);
        assert_eq!(
            world.event_log().records()[0].kind,
            EventKind::SystemError {
                system: "invalid_command".to_string(),
                message: "entity 999 does not exist".to_string()
            }
        );
    }

    #[test]
    fn command_buffer_spawn_returns_reserved_id_for_initialization() {
        let mut world = World::new(registry());
        let mut schedule = Schedule::new();
        schedule.add_system("spawn_and_initialize", spawn_and_initialize);

        schedule
            .run_fixed_step(&mut world, FixedStepContext::new(0.25, 1, 99))
            .unwrap();

        let transform = world
            .get::<Transform>(EntityId::from_raw(1))
            .unwrap()
            .unwrap();
        assert_eq!(transform.translation, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn schedule_gives_systems_distinct_rng_streams_per_frame() {
        let mut world = World::new(registry());
        let entity = world.spawn();
        world.insert(entity, Transform::default()).unwrap();

        let mut schedule = Schedule::new();
        schedule.add_system("write_random_x", write_random_x);
        schedule.add_system("write_random_y", write_random_y);
        schedule
            .run_fixed_step(&mut world, FixedStepContext::new(0.25, 1, 99))
            .unwrap();

        let translation = world.get::<Transform>(entity).unwrap().unwrap().translation;
        assert_ne!(translation.x, translation.y);
    }

    #[test]
    fn schedule_integrates_velocity_and_applies_commands_after_system() {
        let mut world = World::new(registry());
        let entity = world.spawn();
        world.insert(entity, Transform::default()).unwrap();
        world
            .insert(entity, Velocity::new(Vec3::new(2.0, 0.0, 0.0)))
            .unwrap();

        let mut schedule = Schedule::new();
        schedule.add_system("integrate", integrate);
        for frame in 1..=3 {
            schedule
                .run_fixed_step(&mut world, FixedStepContext::new(0.25, frame, 99))
                .unwrap();
        }

        assert_eq!(
            world.get::<Transform>(entity).unwrap().unwrap().translation,
            Vec3::new(1.5, 0.0, 0.0)
        );
        assert!(
            world
                .get::<Transform>(EntityId::from_raw(100))
                .unwrap()
                .is_none()
        );
    }
}
