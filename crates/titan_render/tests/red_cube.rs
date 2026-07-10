use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use png::{BitDepth, ColorType, Decoder, Encoder};
use serde::Deserialize;
use titan_render::{CameraSelection, CaptureMode, RenderRequest, RenderService, error};
use titan_scene::{load_world, parse, phase2_component_registry, validate};

const RED_CUBE: &str = include_str!("fixtures/red_cube.tsf");
const GOLDEN_VERSION: u32 = 1;
const GOLDEN_FORMAT: &str = "Rgba8UnormSrgb sRGB 8-bit";
const MAX_CHANNEL_DIFF: u8 = 2;
const MAX_DIFFERING_PIXEL_RATIO: f64 = 0.001;

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
}

#[test]
fn red_cube_fixture_has_no_filesystem_geometry_dependency() {
    assert!(!Path::new("__builtin__/geometry/cube-v1").exists());
}

/// The only pixel test in this crate. It is ignored by default so macOS and
/// ordinary CI jobs remain stats-only; the required llvmpipe job invokes it
/// with `--ignored` after setting the adapter-selection environment variables.
///
/// To intentionally create or refresh the fixture in that pinned environment:
/// `TITAN_REGEN_GOLDENS=1 cargo test -p titan_render --test red_cube -- --ignored`
/// Review the resulting PNG and run the same command without the flag before
/// committing it. A missing golden without the flag is an error.
#[test]
#[ignore = "requires the pinned Vulkan llvmpipe adapter"]
fn red_cube_fixture_matches_llvmpipe_golden() {
    let document = parse(Some("red_cube.tsf"), RED_CUBE).expect("fixture parses");
    validate(&document).expect("fixture validates");
    let world = load_world(&document, phase2_component_registry().expect("registry"))
        .expect("fixture loads");

    let service = RenderService::new()
        .unwrap_or_else(|render_error| panic!("{}: {}", render_error.code, render_error.message));
    let adapter = service
        .adapter_info()
        .expect("GPU service has adapter metadata");
    if !adapter.name.to_ascii_lowercase().contains("llvmpipe")
        || adapter.backend != titan_render::AdapterBackend::Vulkan
    {
        panic!(
            "{}: expected Vulkan llvmpipe, selected {} ({:?})",
            error::NO_ADAPTER,
            adapter.name,
            adapter.backend
        );
    }

    let result = service
        .render(
            &world,
            RenderRequest {
                camera: CameraSelection::Name("main".to_owned()),
                capture: CaptureMode::Image,
                ..RenderRequest::default()
            },
        )
        .expect("llvmpipe fixture render");
    let pixels = result.rgba8.as_ref().expect("captured pixels");
    let golden_path = golden_path();
    let metadata_path = metadata_path();

    if std::env::var_os("TITAN_REGEN_GOLDENS").is_some() {
        write_png(
            &golden_path,
            result.output_size.width,
            result.output_size.height,
            pixels,
        );
        write_metadata(&metadata_path, &result, adapter);
        return;
    }

    assert!(
        golden_path.is_file(),
        "missing golden at {}; run with TITAN_REGEN_GOLDENS=1 in the pinned llvmpipe CI environment",
        golden_path.display()
    );
    assert!(
        metadata_path.is_file(),
        "missing golden metadata at {}; run with TITAN_REGEN_GOLDENS=1 in the pinned llvmpipe CI environment",
        metadata_path.display()
    );
    let metadata = read_metadata(&metadata_path);
    assert_eq!(metadata.golden_version, GOLDEN_VERSION);
    assert_eq!(
        metadata.comparison_policy.max_channel_diff,
        MAX_CHANNEL_DIFF
    );
    assert_eq!(
        metadata.comparison_policy.max_differing_pixel_ratio,
        MAX_DIFFERING_PIXEL_RATIO
    );
    assert_eq!(metadata.width, result.output_size.width);
    assert_eq!(metadata.height, result.output_size.height);
    assert_eq!(metadata.format, GOLDEN_FORMAT);
    assert_eq!(metadata.shader_version, result.stats.shader_version);
    let expected_backend = format!("{:?}", adapter.backend).to_ascii_lowercase();
    let environment_changed = "environment changed; intentionally regenerate the golden with TITAN_REGEN_GOLDENS=1 and bump the golden version";
    assert_eq!(
        metadata.wgpu_version,
        wgpu_version(),
        "{environment_changed}: wgpu version differs"
    );
    assert_eq!(
        metadata.adapter.name, adapter.name,
        "{environment_changed}: adapter name differs"
    );
    assert_eq!(
        metadata.adapter.backend, expected_backend,
        "{environment_changed}: adapter backend differs"
    );
    assert_eq!(
        metadata.adapter.driver, adapter.driver,
        "{environment_changed}: adapter driver differs"
    );
    assert_eq!(
        metadata.adapter.driver_info, adapter.driver_info,
        "{environment_changed}: adapter driver info differs"
    );

    let (width, height, expected) = read_png(&golden_path);
    assert_eq!(
        (width, height),
        (result.output_size.width, result.output_size.height)
    );
    assert_eq!(expected.len(), pixels.len());

    let mut max_difference = 0u8;
    let mut different_pixels = 0usize;
    for (actual, expected) in pixels.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let pixel_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .expect("RGBA pixel");
        max_difference = max_difference.max(pixel_difference);
        if pixel_difference > 0 {
            different_pixels += 1;
        }
    }
    let pixel_count = (width as usize) * (height as usize);
    assert!(
        max_difference <= MAX_CHANNEL_DIFF,
        "max channel difference was {max_difference}"
    );
    assert!(
        (different_pixels as f64) / (pixel_count as f64) <= MAX_DIFFERING_PIXEL_RATIO,
        "{different_pixels} of {pixel_count} pixels differ; limit is {MAX_DIFFERING_PIXEL_RATIO:.1}%"
    );
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/red_cube.png")
}

fn metadata_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/red_cube.golden.json")
}

#[derive(Debug, Deserialize)]
struct GoldenMetadata {
    golden_version: u32,
    wgpu_version: String,
    adapter: AdapterMetadata,
    shader_version: u32,
    width: u32,
    height: u32,
    format: String,
    comparison_policy: ComparisonPolicy,
}

#[derive(Debug, Deserialize)]
struct AdapterMetadata {
    name: String,
    backend: String,
    driver: String,
    driver_info: String,
}

#[derive(Debug, Deserialize)]
struct ComparisonPolicy {
    max_channel_diff: u8,
    max_differing_pixel_ratio: f64,
}

fn read_metadata(path: &Path) -> GoldenMetadata {
    let contents = fs::read_to_string(path).expect("read golden metadata");
    serde_json::from_str(&contents).expect("parse golden metadata")
}

fn write_metadata(
    path: &Path,
    result: &titan_render::RenderResult,
    adapter: &titan_render::AdapterInfo,
) {
    let adapter = BTreeMap::from([
        (
            "backend",
            serde_json::json!(format!("{:?}", adapter.backend).to_ascii_lowercase()),
        ),
        ("driver", serde_json::json!(adapter.driver)),
        ("driver_info", serde_json::json!(adapter.driver_info)),
        ("name", serde_json::json!(adapter.name)),
    ]);
    let comparison_policy = BTreeMap::from([
        ("max_channel_diff", serde_json::json!(MAX_CHANNEL_DIFF)),
        (
            "max_differing_pixel_ratio",
            serde_json::json!(MAX_DIFFERING_PIXEL_RATIO),
        ),
    ]);
    let metadata = BTreeMap::from([
        ("adapter", serde_json::json!(adapter)),
        ("comparison_policy", serde_json::json!(comparison_policy)),
        ("format", serde_json::json!(GOLDEN_FORMAT)),
        ("golden_version", serde_json::json!(GOLDEN_VERSION)),
        ("height", serde_json::json!(result.output_size.height)),
        (
            "shader_version",
            serde_json::json!(result.stats.shader_version),
        ),
        ("wgpu_version", serde_json::json!(wgpu_version())),
        ("width", serde_json::json!(result.output_size.width)),
    ]);
    let contents = serde_json::to_string_pretty(&metadata).expect("serialize golden metadata");
    fs::write(path, format!("{contents}\n")).expect("write golden metadata");
}

fn wgpu_version() -> String {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    let contents = fs::read_to_string(&lock_path).unwrap_or_else(|error| {
        panic!(
            "read workspace Cargo.lock at {} while resolving wgpu version: {error}",
            lock_path.display()
        )
    });
    let package = contents
        .split("[[package]]")
        .find(|package| package.lines().any(|line| line.trim() == "name = \"wgpu\""))
        .unwrap_or_else(|| panic!("find the wgpu package in {}", lock_path.display()));
    package
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("version = \"")
                .and_then(|version| version.strip_suffix('"'))
        })
        .map(str::to_owned)
        .unwrap_or_else(|| {
            panic!(
                "read the version from the wgpu package in {}",
                lock_path.display()
            )
        })
}

fn read_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new(fs::File::open(path).expect("open golden PNG"));
    let mut reader = decoder.read_info().expect("read golden PNG header");
    let info = reader.info().clone();
    assert_eq!(info.color_type, ColorType::Rgba, "golden must be RGBA");
    assert_eq!(info.bit_depth, BitDepth::Eight, "golden must be 8-bit");
    let mut pixels = vec![0; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut pixels).expect("read golden PNG");
    pixels.truncate(frame.buffer_size());
    (frame.width, frame.height, pixels)
}

fn write_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
    let file = fs::File::create(path).expect("create golden PNG");
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().expect("write golden PNG header");
    writer
        .write_image_data(pixels)
        .expect("write golden PNG pixels");
}
