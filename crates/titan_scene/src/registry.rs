use std::{collections::BTreeMap, fmt};

use titan_core::ComponentRegistry;

use crate::tsf::{Diagnostic, Value};

pub type Diagnostics = Vec<Diagnostic>;
pub type TsfComponentValidator = fn(&Value, &str, &mut Diagnostics);

fn no_op_validator(_: &Value, _: &str, _: &mut Diagnostics) {}

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
    pub by_alias: BTreeMap<&'static str, TsfComponentBinding>,
}

impl TsfComponentRegistry {
    pub fn new(
        component_registry: ComponentRegistry,
        bindings: impl IntoIterator<Item = TsfComponentBinding>,
    ) -> Result<Self, TsfComponentRegistryError> {
        let mut by_alias = BTreeMap::new();
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
            if let Ok(meta) = component_registry.meta_by_name(binding.registered_name)
                && meta.schema_version() != binding.schema_version
            {
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
        })
    }

    pub fn component_registry(&self) -> &ComponentRegistry {
        &self.component_registry
    }
    pub fn binding(&self, alias: &str) -> Option<&TsfComponentBinding> {
        self.by_alias.get(alias)
    }
    pub fn bindings(&self) -> impl Iterator<Item = &TsfComponentBinding> {
        self.by_alias.values()
    }
    pub fn into_component_registry(self) -> ComponentRegistry {
        self.component_registry
    }
}

pub const BUILTIN_COMPONENT_ORDER: [&str; 6] = [
    "transform",
    "velocity",
    "camera",
    "directional_light",
    "mesh",
    "material",
];

pub const PHASE1_BINDINGS: [TsfComponentBinding; 2] = [
    TsfComponentBinding {
        alias: "transform",
        registered_name: "titan.core.Transform",
        schema_version: 2,
        validate: no_op_validator,
    },
    TsfComponentBinding {
        alias: "velocity",
        registered_name: "titan.core.Velocity",
        schema_version: 1,
        validate: no_op_validator,
    },
];

pub const PHASE2_BINDINGS: [TsfComponentBinding; 6] = [
    PHASE1_BINDINGS[0],
    PHASE1_BINDINGS[1],
    TsfComponentBinding {
        alias: "camera",
        registered_name: "titan.core.Camera",
        schema_version: 1,
        validate: no_op_validator,
    },
    TsfComponentBinding {
        alias: "directional_light",
        registered_name: "titan.core.DirectionalLight",
        schema_version: 1,
        validate: no_op_validator,
    },
    TsfComponentBinding {
        alias: "mesh",
        registered_name: "titan.core.Mesh",
        schema_version: 1,
        validate: no_op_validator,
    },
    TsfComponentBinding {
        alias: "material",
        registered_name: "titan.core.Material",
        schema_version: 1,
        validate: no_op_validator,
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

pub(crate) fn registry_for_core(registry: ComponentRegistry) -> TsfComponentRegistry {
    TsfComponentRegistry::new(registry, PHASE2_BINDINGS).unwrap_or_else(|error| {
        panic!("built-in component registry must match TSF bindings: {error}")
    })
}
