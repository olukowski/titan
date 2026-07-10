use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Serialize;

use crate::{RenderError, ServiceResult, error};

/// A vertex in the renderer's resolved mesh interface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// One indexed range in a resolved mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Submesh {
    pub index_start: u32,
    pub index_count: u32,
}

/// Renderer-neutral output shared by builtins and future TitanGeo/glTF loaders.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeshAsset {
    pub vertices: Vec<MeshVertex>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<Submesh>,
}

impl MeshAsset {
    pub fn validate(&self, path: &str) -> ServiceResult<()> {
        if self.vertices.is_empty() || self.indices.is_empty() || self.submeshes.is_empty() {
            return Err(RenderError::with_path(
                error::INVALID_GEOMETRY,
                "geometry must contain vertices, indices, and submeshes",
                path,
            ));
        }
        for submesh in &self.submeshes {
            let end = submesh
                .index_start
                .checked_add(submesh.index_count)
                .ok_or_else(|| {
                    RenderError::with_path(
                        error::INVALID_GEOMETRY,
                        "submesh index range overflows",
                        path,
                    )
                })?;
            if submesh.index_count == 0
                || submesh.index_count % 3 != 0
                || end as usize > self.indices.len()
            {
                return Err(RenderError::with_path(
                    error::INVALID_GEOMETRY,
                    format!(
                        "invalid submesh index range {}..{}",
                        submesh.index_start, end
                    ),
                    path,
                ));
            }
            if self.indices[submesh.index_start as usize..end as usize]
                .iter()
                .any(|&index| index as usize >= self.vertices.len())
            {
                return Err(RenderError::with_path(
                    error::INVALID_GEOMETRY,
                    "submesh index references a vertex outside the vertex buffer",
                    path,
                ));
            }
        }
        Ok(())
    }

    pub fn triangle_count(&self) -> u64 {
        self.submeshes
            .iter()
            .map(|submesh| u64::from(submesh.index_count / 3))
            .sum()
    }
}

/// The only geometry builtin reserved by Phase 2.
pub const CUBE_V1_PATH: &str = "__builtin__/geometry/cube-v1";

/// Returns the deterministic unit cube fixture.
pub fn cube_v1() -> MeshAsset {
    let faces = [
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut vertices = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_index, (normal, positions)) in faces.into_iter().enumerate() {
        let start = (face_index * 4) as u32;
        vertices.extend(
            positions
                .into_iter()
                .zip(uvs)
                .map(|(position, uv)| MeshVertex { position, uv }),
        );
        normals.extend([normal; 4]);
        indices.extend([start, start + 2, start + 1, start, start + 3, start + 2]);
    }
    MeshAsset {
        vertices,
        normals: Some(normals),
        indices,
        submeshes: vec![Submesh {
            index_start: 0,
            index_count: 36,
        }],
    }
}

/// Resolves renderer geometry paths. Builtin paths are handled before any filesystem access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryResolver {
    base_dir: PathBuf,
}

impl GeometryResolver {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    pub fn resolve(&self, virtual_relative_path: &str) -> ServiceResult<MeshAsset> {
        if !is_valid_virtual_relative_path(virtual_relative_path) {
            return Err(RenderError::with_path(
                error::ASSET_UNAVAILABLE,
                format!("geometry asset path is not virtual-relative: '{virtual_relative_path}'"),
                virtual_relative_path,
            ));
        }
        if virtual_relative_path == CUBE_V1_PATH {
            let asset = cube_v1();
            asset.validate(virtual_relative_path)?;
            return Ok(asset);
        }
        if virtual_relative_path.split('/').next() == Some("__builtin__") {
            return Err(RenderError::with_path(
                error::UNKNOWN_BUILTIN,
                format!("unknown builtin geometry path '{virtual_relative_path}'"),
                virtual_relative_path,
            ));
        }
        let path = self.base_dir.join(virtual_relative_path);
        let _ = fs::metadata(&path).map_err(|_| {
            RenderError::with_path(
                error::ASSET_UNAVAILABLE,
                format!("geometry asset is unavailable: {}", path.display()),
                virtual_relative_path,
            )
        })?;
        Err(RenderError::with_path(
            error::ASSET_UNAVAILABLE,
            format!(
                "filesystem geometry loading is not implemented: {}",
                path.display()
            ),
            virtual_relative_path,
        ))
    }
}

pub type AssetResolver = GeometryResolver;

pub(crate) fn is_valid_virtual_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.contains(':')
        && Path::new(path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_v1_has_the_complete_deterministic_mesh() {
        let first = cube_v1();
        let second = cube_v1();
        assert_eq!(first, second);
        assert_eq!(first.vertices.len(), 24);
        assert_eq!(first.indices.len(), 36);
        assert_eq!(
            first.submeshes,
            vec![Submesh {
                index_start: 0,
                index_count: 36
            }]
        );
        assert_eq!(first.triangle_count(), 12);
        assert!(
            first
                .vertices
                .iter()
                .zip(first.normals.as_ref().unwrap())
                .all(|(vertex, normal)| {
                    vertex
                        .position
                        .iter()
                        .all(|value| (-0.5..=0.5).contains(value))
                        && (normal.iter().map(|value| value * value).sum::<f32>() - 1.0).abs()
                            < 1e-6
                })
        );
        for face in first.vertices.chunks_exact(4) {
            let start = first
                .vertices
                .iter()
                .position(|vertex| vertex == &face[0])
                .unwrap();
            assert!(
                first.normals.as_ref().unwrap()[start..start + 4]
                    .iter()
                    .all(|normal| *normal == first.normals.as_ref().unwrap()[start])
            );
            assert_eq!(
                face.iter().map(|vertex| vertex.uv).collect::<Vec<_>>(),
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            );
        }
        for triangle in first.indices.chunks_exact(3) {
            assert!(
                triangle
                    .iter()
                    .all(|&index| (index as usize) < first.vertices.len())
            );
            let a = first.vertices[triangle[0] as usize].position;
            let b = first.vertices[triangle[1] as usize].position;
            let c = first.vertices[triangle[2] as usize].position;
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let geometric_normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let stored_normal = first.normals.as_ref().unwrap()[triangle[0] as usize];
            let agreement = geometric_normal[0] * stored_normal[0]
                + geometric_normal[1] * stored_normal[1]
                + geometric_normal[2] * stored_normal[2];
            let outward = geometric_normal[0] * a[0]
                + geometric_normal[1] * a[1]
                + geometric_normal[2] * a[2];
            assert!(agreement > 0.0);
            assert!(outward > 0.0);
        }
        assert!(
            first
                .vertices
                .iter()
                .any(|vertex| vertex.position == [-0.5, -0.5, -0.5])
        );
        assert!(
            first
                .vertices
                .iter()
                .any(|vertex| vertex.position == [0.5, 0.5, 0.5])
        );
    }

    #[test]
    fn builtin_resolution_precedes_filesystem_access() {
        let resolver = GeometryResolver::new("/definitely/nonexistent/titan-base");
        let asset = resolver.resolve(CUBE_V1_PATH).unwrap();
        assert_eq!(asset.indices.len(), 36);
    }

    #[test]
    fn unknown_builtin_is_structured_and_never_falls_through_to_disk() {
        let resolver = GeometryResolver::new("/definitely/nonexistent/titan-base");
        let error = resolver
            .resolve("__builtin__/geometry/not-a-real-version")
            .unwrap_err();
        assert_eq!(error.code, error::UNKNOWN_BUILTIN);
        assert_eq!(
            error.path.as_deref(),
            Some("__builtin__/geometry/not-a-real-version")
        );
    }

    #[test]
    fn parent_dir_geometry_path_is_rejected_before_filesystem_access() {
        let resolver = GeometryResolver::new("/definitely/nonexistent/titan-base");
        let error = resolver.resolve("../../etc/passwd").unwrap_err();

        assert_eq!(error.code, error::ASSET_UNAVAILABLE);
        assert_eq!(error.path.as_deref(), Some("../../etc/passwd"));
        assert!(error.message.contains("not virtual-relative"));
    }
}
