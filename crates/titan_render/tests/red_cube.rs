use std::path::Path;

use titan_render::{CameraSelection, CaptureMode, RenderRequest, RenderService, error};
use titan_scene::{load_world, parse, phase2_component_registry, validate};

const RED_CUBE: &str = include_str!("fixtures/red_cube.tsf");

#[test]
fn red_cube_fixture_renders_with_deterministic_stats() {
    let document = parse(Some("red_cube.tsf"), RED_CUBE).expect("fixture parses");
    validate(&document).expect("fixture validates");
    let world = load_world(&document, phase2_component_registry().expect("registry"))
        .expect("fixture loads");
    let service = RenderService::cpu_only();
    let result = service
        .render(
            &world,
            RenderRequest {
                camera: CameraSelection::Name("main".to_owned()),
                ..RenderRequest::default()
            },
        )
        .expect("fixture renders");
    assert_eq!(result.output_size.width, 64);
    assert_eq!(result.stats.draw_calls, 1);
    assert_eq!(result.stats.triangles, 12);
    assert_eq!(result.stats.visible_meshes, 1);
    assert_eq!(result.stats.active_directional_lights, 0);
    assert_eq!(result.stats.shader_version, 1);
    assert_eq!(result.stats.material_models["unlit"], 1);

    match RenderService::new() {
        Err(error) => assert_eq!(error.code, error::NO_ADAPTER),
        Ok(service) => {
            let result = service
                .render(
                    &world,
                    RenderRequest {
                        camera: CameraSelection::Name("main".to_owned()),
                        capture: CaptureMode::Image,
                        ..RenderRequest::default()
                    },
                )
                .expect("GPU fixture render");
            let pixels = result.rgba8.expect("captured pixels");
            let center = (64 / 2 * 64 + 64 / 2) * 4;
            assert!(pixels[center] > 180, "center pixel should be red");
            assert!(pixels[center + 1] < 80, "center pixel should not be green");
            assert!(pixels[center + 2] < 80, "center pixel should not be blue");
        }
    }
}

#[test]
fn red_cube_fixture_has_no_filesystem_geometry_dependency() {
    assert!(!Path::new("__builtin__/geometry/cube-v1").exists());
}
