use std::{collections::BTreeMap, fmt};

use titan_core::ComponentRegistry;

use crate::tsf::{Diagnostic, Value};

pub type Diagnostics = Vec<Diagnostic>;
pub type TsfComponentValidator = fn(&Value, &str, &mut Diagnostics);

/// A stable lowercase TSF alias for a registered component.
#[derive(Clone, Copy, Debug)]
pub struct TsfComponentBinding {
    pub alias: &'static str,
    pub registered_name: &'static str,
    pub schema_version: u32,
    pub validate: TsfComponentValidator,
}

/// Errors found while constructing the immutable TSF alias map.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TsfComponentRegistryError {
    DuplicateAlias(&'static str),
    DuplicateRegisteredName(&'static str),
    UnsupportedCoreRegistry,
    ComponentNotRegistered(String),
    SchemaVersionMismatch {
        name: &'static str,
        expected: u32,
        actual: u32,
    },
}

impl fmt::Display for TsfComponentRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAlias(alias) => write!(f, "TSF component alias '{alias}' is duplicated"),
            Self::DuplicateRegisteredName(name) => {
                write!(f, "registered component '{name}' is duplicated")
            }
            Self::UnsupportedCoreRegistry => {
                write!(
                    f,
                    "core component registry is not a supported Titan scene registry"
                )
            }
            Self::ComponentNotRegistered(name) => write!(f, "component {name} is not registered"),
            Self::SchemaVersionMismatch {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "component {name} has schema {actual}, expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for TsfComponentRegistryError {}

/// Immutable alias registry used by TSF loading, validation, and formatting.
#[derive(Clone)]
pub struct TsfComponentRegistry {
    component_registry: ComponentRegistry,
    by_alias: BTreeMap<&'static str, TsfComponentBinding>,
    component_order: Vec<&'static str>,
}

pub trait IntoTsfComponentRegistry {
    fn into_tsf_component_registry(self)
    -> Result<TsfComponentRegistry, TsfComponentRegistryError>;
}

impl IntoTsfComponentRegistry for TsfComponentRegistry {
    fn into_tsf_component_registry(
        self,
    ) -> Result<TsfComponentRegistry, TsfComponentRegistryError> {
        Ok(self)
    }
}

impl IntoTsfComponentRegistry for ComponentRegistry {
    fn into_tsf_component_registry(
        self,
    ) -> Result<TsfComponentRegistry, TsfComponentRegistryError> {
        registry_for_core(self)
    }
}

impl TsfComponentRegistry {
    pub fn new(
        component_registry: ComponentRegistry,
        bindings: impl IntoIterator<Item = TsfComponentBinding>,
    ) -> Result<Self, TsfComponentRegistryError> {
        let mut by_alias = BTreeMap::new();
        let mut component_order = Vec::new();
        let mut names = BTreeMap::new();
        for binding in bindings {
            if by_alias.insert(binding.alias, binding).is_some() {
                return Err(TsfComponentRegistryError::DuplicateAlias(binding.alias));
            }
            if names.insert(binding.registered_name, ()).is_some() {
                return Err(TsfComponentRegistryError::DuplicateRegisteredName(
                    binding.registered_name,
                ));
            }
            component_order.push(binding.alias);
            let meta = component_registry
                .meta_by_name(binding.registered_name)
                .map_err(|_| {
                    TsfComponentRegistryError::ComponentNotRegistered(
                        binding.registered_name.to_owned(),
                    )
                })?;
            if meta.schema_version() != binding.schema_version {
                return Err(TsfComponentRegistryError::SchemaVersionMismatch {
                    name: binding.registered_name,
                    expected: binding.schema_version,
                    actual: meta.schema_version(),
                });
            }
        }
        Ok(Self {
            component_registry,
            by_alias,
            component_order,
        })
    }

    pub fn component_registry(&self) -> &ComponentRegistry {
        &self.component_registry
    }
    pub fn binding(&self, alias: &str) -> Option<&TsfComponentBinding> {
        self.by_alias.get(alias)
    }
    pub fn registered_name(&self, alias: &str) -> Option<&'static str> {
        self.binding(alias).map(|binding| binding.registered_name)
    }
    pub fn bindings(&self) -> impl Iterator<Item = &TsfComponentBinding> {
        self.by_alias.values()
    }
    pub(crate) fn component_order(&self) -> &[&'static str] {
        &self.component_order
    }
    pub fn into_component_registry(self) -> ComponentRegistry {
        self.component_registry
    }
}

pub const PHASE1_BINDINGS: [TsfComponentBinding; 2] = [
    TsfComponentBinding {
        alias: "transform",
        registered_name: "titan.core.Transform",
        schema_version: 2,
        validate: crate::tsf::validate_transform_binding,
    },
    TsfComponentBinding {
        alias: "velocity",
        registered_name: "titan.core.Velocity",
        schema_version: 1,
        validate: crate::tsf::validate_velocity_binding,
    },
];

pub const PHASE2_BINDINGS: [TsfComponentBinding; 6] = [
    PHASE1_BINDINGS[0],
    PHASE1_BINDINGS[1],
    TsfComponentBinding {
        alias: "camera",
        registered_name: "titan.core.Camera",
        schema_version: 1,
        validate: crate::tsf::validate_camera_binding,
    },
    TsfComponentBinding {
        alias: "directional_light",
        registered_name: "titan.core.DirectionalLight",
        schema_version: 1,
        validate: crate::tsf::validate_directional_light_binding,
    },
    TsfComponentBinding {
        alias: "mesh",
        registered_name: "titan.core.Mesh",
        schema_version: 1,
        validate: crate::tsf::validate_mesh_binding,
    },
    TsfComponentBinding {
        alias: "material",
        registered_name: "titan.core.Material",
        schema_version: 1,
        validate: crate::tsf::validate_material_binding,
    },
];

pub fn phase1_component_registry() -> Result<TsfComponentRegistry, TsfComponentRegistryError> {
    TsfComponentRegistry::new(
        titan_core::phase1_component_registry()
            .map_err(|e| TsfComponentRegistryError::ComponentNotRegistered(e.to_string()))?,
        PHASE1_BINDINGS,
    )
}

pub fn phase2_component_registry() -> Result<TsfComponentRegistry, TsfComponentRegistryError> {
    TsfComponentRegistry::new(
        titan_core::phase2_component_registry()
            .map_err(|e| TsfComponentRegistryError::ComponentNotRegistered(e.to_string()))?,
        PHASE2_BINDINGS,
    )
}

pub(crate) fn registry_for_core(
    registry: ComponentRegistry,
) -> Result<TsfComponentRegistry, TsfComponentRegistryError> {
    let bindings = if PHASE2_BINDINGS
        .iter()
        .all(|binding| registry.meta_by_name(binding.registered_name).is_ok())
    {
        PHASE2_BINDINGS.to_vec()
    } else if PHASE1_BINDINGS
        .iter()
        .all(|binding| registry.meta_by_name(binding.registered_name).is_ok())
    {
        PHASE1_BINDINGS.to_vec()
    } else {
        return Err(TsfComponentRegistryError::UnsupportedCoreRegistry);
    };
    TsfComponentRegistry::new(registry, bindings)
}
