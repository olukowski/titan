// Checks titan.js's decoders against the wire layouts and tags in crates/titan-wasm/src/abi.rs.
// Run with `node --test web/titan.test.mjs`.
import test from "node:test";
import assert from "node:assert/strict";
import { KEY_MAP, mapMouseButton, decodeTextureDesc, decodeShaderSource, decodePipelineDesc,
  decodeFrame, shaderBindings } from "./titan.js";

// The WebGPU name of each Rust wire enum variant, indexed by its tag in abi.rs.
const TEXTURE_FORMATS = ["rgba8unorm", "rgba8unorm-srgb", "bgra8unorm", "bgra8unorm-srgb",
  "r8unorm", "depth32float", "depth24plus-stencil8"];
const FILTER_MODES = ["nearest", "linear"];
const ADDRESS_MODES = ["clamp-to-edge", "repeat", "mirror-repeat"];
const VERTEX_FORMATS = ["float32", "float32x2", "float32x3", "float32x4", "uint32", "sint32",
  "unorm8x4"];
const STEP_MODES = ["vertex", "instance"];
const TOPOLOGIES = ["triangle-list", "triangle-strip", "line-list", "line-strip", "point-list"];
const FRONT_FACES = ["ccw", "cw"];
const CULL_MODES = ["none", "front", "back"];
const COMPARE_FUNCTIONS = ["never", "less", "less-equal", "equal", "greater-equal", "greater",
  "not-equal", "always"];
const BLEND_FACTORS = ["zero", "one", "src-alpha", "one-minus-src-alpha", "dst-alpha",
  "one-minus-dst-alpha", "src", "one-minus-src", "dst", "one-minus-dst"];
const BLEND_OPERATIONS = ["add", "subtract", "reverse-subtract", "min", "max"];
const INDEX_FORMATS = ["uint16", "uint32"];

// Little-endian encoders mirroring abi.rs's `put_*`. Payload parts nest freely; bare numbers are u8s.
const le = (setter, size) => (...values) => values.flatMap((value) => {
  const view = new DataView(new ArrayBuffer(size));
  view[setter](0, value, true);
  return [...new Uint8Array(view.buffer)];
});
const [u32, i32, f32, f64] = [le("setUint32", 4), le("setInt32", 4), le("setFloat32", 4),
  le("setFloat64", 8)];
const str = (s) => { const utf8 = new TextEncoder().encode(s); return [u32(utf8.length), ...utf8]; };
const bytes = (...parts) => Uint8Array.from(parts.flat(Infinity));
// The same payload behind a non-zero, unaligned byteOffset, like a view into WASM memory.
const atOffset = (payload) => new Uint8Array([0xee, 0xee, 0xee, ...payload]).subarray(3);
// Asserts that `decodeTag(tag)` yields `names[tag]` for every tag.
const sweep = (names, decodeTag) =>
  names.forEach((name, tag) => assert.equal(decodeTag(tag), name, `tag ${tag}`));

test("KEY_MAP gives each DOM code its index in KeyCode::ALL", () => {
  const rustKeys = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "Digit0", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6", "Digit7", "Digit8", "Digit9",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "Enter", "Escape", "Tab", "Backspace",
    "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight",
    "SuperLeft", "SuperRight", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"];
  const domCodes = rustKeys.map((key) => (key.length === 1 ? `Key${key}` : key.replace("Super", "Meta")));
  assert.equal(domCodes.length, 65);
  assert.deepEqual([...KEY_MAP], domCodes.map((code, index) => [code, index]));
  assert.equal(KEY_MAP.get("MetaLeft"), 51);
  assert.equal(KEY_MAP.get("MetaRight"), 52);
  for (const code of ["Numpad0", "IntlBackslash", "keya", "A", ""]) assert.equal(KEY_MAP.get(code), undefined);
});

test("mapMouseButton swaps DOM middle and right to Titan's order", () => {
  assert.deepEqual([0, 1, 2].map(mapMouseButton), [0, 2, 1]);
  for (const button of [3, 4, -1]) assert.equal(mapMouseButton(button), undefined);
});

const texture = (format, usage, sampler, label = "") =>
  bytes(u32(640, 480, 14), format, u32(usage), sampler, str(label));

test("decodeTextureDesc reads every field, from any byteOffset", () => {
  const payload = texture(3, 0b10011, [1, 0, 2, 1], "albédo");
  const expected = { width: 640, height: 480, mipLevelCount: 14, format: "bgra8unorm-srgb",
    usage: 0b10011, sampler: { minFilter: "linear", magFilter: "nearest",
      addressModeU: "mirror-repeat", addressModeV: "repeat" }, label: "albédo" };
  assert.deepEqual(decodeTextureDesc(payload), expected);
  assert.deepEqual(decodeTextureDesc(atOffset(payload)), expected);
  assert.equal(decodeTextureDesc(texture(0, 0, [0, 0, 0, 0])).label, undefined);
});

test("decodeTextureDesc maps every format, filter, and address tag, and rejects unknown ones", () => {
  const sampler = (tags) => decodeTextureDesc(texture(0, 0, tags)).sampler;
  sweep(TEXTURE_FORMATS, (tag) => decodeTextureDesc(texture(tag, 0, [0, 0, 0, 0])).format);
  sweep(FILTER_MODES, (tag) => sampler([tag, 0, 0, 0]).minFilter);
  sweep(FILTER_MODES, (tag) => sampler([0, tag, 0, 0]).magFilter);
  sweep(ADDRESS_MODES, (tag) => sampler([0, 0, tag, 0]).addressModeU);
  sweep(ADDRESS_MODES, (tag) => sampler([0, 0, 0, tag]).addressModeV);
  for (const payload of [texture(7, 0, [0, 0, 0, 0]), texture(0, 0, [2, 0, 0, 0]),
    texture(0, 0, [0, 2, 0, 0]), texture(0, 0, [0, 0, 3, 0]), texture(0, 0, [0, 0, 0, 3])]) {
    assert.throws(() => decodeTextureDesc(payload), /unknown wire tag/);
  }
});

test("decodeShaderSource reads its strings, from any byteOffset", () => {
  const wgsl = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }";
  const glsl = "#ifdef TITAN_VERTEX\nvoid main() {}\n#endif";
  const payload = bytes(str(wgsl), str(glsl), str("vs"), str("fs_main"), str("sprite ✨"));
  const expected = { wgsl, glsl, vertexEntry: "vs", fragmentEntry: "fs_main", label: "sprite ✨" };
  assert.deepEqual(decodeShaderSource(payload), expected);
  assert.deepEqual(decodeShaderSource(atOffset(payload)), expected);
  assert.deepEqual(decodeShaderSource(bytes(str(wgsl), str(""), str("vs"), str("fs"), str(""))),
    { wgsl, glsl: undefined, vertexEntry: "vs", fragmentEntry: "fs", label: undefined });
});

test("decodeShaderSource reads the exact bytes abi.rs encodes, with and without GLSL", () => {
  // The byte strings abi.rs `shader_source_matches_the_bytes_titan_js_decodes` asserts, verbatim.
  const rust = (latin1) => Uint8Array.from(latin1, (c) => c.charCodeAt(0));
  const common = { wgsl: "w", vertexEntry: "vs", fragmentEntry: "fs", label: "l" };
  assert.deepEqual(decodeShaderSource(rust("\x01\0\0\0w\x01\0\0\0g\x02\0\0\0vs\x02\0\0\0fs\x01\0\0\0l")),
    { ...common, glsl: "g" });
  assert.deepEqual(decodeShaderSource(rust("\x01\0\0\0w\0\0\0\0\x02\0\0\0vs\x02\0\0\0fs\x01\0\0\0l")),
    { ...common, glsl: undefined });
});

// `blend` is six tags (color then alpha src, dst, op), `depth` is [format, write, compare], and
// each layout is [stride, stepMode, [[format, offset, location], ...]].
const pipeline = ({ color = 0, blend, depth, prim = [0, 0, 0], layouts = [], label = "" }) => bytes(
  u32(77), color, blend ? [1, blend] : 0, depth ? [1, depth] : 0, prim, layouts.length,
  layouts.map(([stride, step, attrs]) => [u32(stride), step, attrs.length,
    attrs.map(([format, offset, location]) => [format, u32(offset, location)])]),
  str(label));
const decodePipeline = (opts) => decodePipelineDesc(pipeline(opts));

test("decodePipelineDesc reads a full descriptor, from any byteOffset", () => {
  const payload = pipeline({ color: 3, blend: [8, 7, 2, 1, 1, 4], depth: [6, 0, 4], prim: [3, 1, 2],
    layouts: [[16, 0, [[2, 4, 0]]], [20, 1, [[3, 0, 2], [6, 16, 3]]]], label: "lit" });
  const expected = {
    shaderId: 77, colorFormat: "bgra8unorm-srgb",
    blend: { color: { srcFactor: "dst", dstFactor: "one-minus-src", operation: "reverse-subtract" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "max" } },
    depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: false, depthCompare: "greater-equal" },
    primitive: { topology: "line-strip", frontFace: "cw", cullMode: "back" },
    buffers: [
      { arrayStride: 16, stepMode: "vertex", attributes: [{ format: "float32x3", offset: 4, shaderLocation: 0 }] },
      { arrayStride: 20, stepMode: "instance", attributes: [
        { format: "float32x4", offset: 0, shaderLocation: 2 },
        { format: "unorm8x4", offset: 16, shaderLocation: 3 }] },
    ],
    label: "lit",
  };
  assert.deepEqual(decodePipelineDesc(payload), expected);
  assert.deepEqual(decodePipelineDesc(atOffset(payload)), expected);
});

test("decodePipelineDesc reads absent optional sections", () => {
  assert.deepEqual(decodePipeline({ color: 0xff }), { shaderId: 77, colorFormat: null, blend: null,
    depthStencil: null, primitive: { topology: "triangle-list", frontFace: "ccw", cullMode: "none" },
    buffers: [], label: undefined });
  const depthOnly = decodePipeline({ color: 0xff, depth: [5, 1, 1] });
  assert.equal(depthOnly.colorFormat, null);
  assert.deepEqual(depthOnly.depthStencil, { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" });
});

test("decodePipelineDesc maps every tag and rejects unknown ones", () => {
  sweep(TEXTURE_FORMATS, (tag) => decodePipeline({ color: tag }).colorFormat);
  sweep(BLEND_FACTORS, (tag) => decodePipeline({ blend: [tag, 0, 0, 0, 0, 0] }).blend.color.srcFactor);
  sweep(BLEND_FACTORS, (tag) => decodePipeline({ blend: [0, 0, 0, 0, tag, 0] }).blend.alpha.dstFactor);
  sweep(BLEND_OPERATIONS, (tag) => decodePipeline({ blend: [0, 0, tag, 0, 0, 0] }).blend.color.operation);
  sweep(BLEND_OPERATIONS, (tag) => decodePipeline({ blend: [0, 0, 0, 0, 0, tag] }).blend.alpha.operation);
  sweep(TEXTURE_FORMATS, (tag) => decodePipeline({ depth: [tag, 0, 0] }).depthStencil.format);
  sweep(COMPARE_FUNCTIONS, (tag) => decodePipeline({ depth: [5, 1, tag] }).depthStencil.depthCompare);
  sweep(TOPOLOGIES, (tag) => decodePipeline({ prim: [tag, 0, 0] }).primitive.topology);
  sweep(FRONT_FACES, (tag) => decodePipeline({ prim: [0, tag, 0] }).primitive.frontFace);
  sweep(CULL_MODES, (tag) => decodePipeline({ prim: [0, 0, tag] }).primitive.cullMode);
  sweep(STEP_MODES, (tag) => decodePipeline({ layouts: [[4, tag, []]] }).buffers[0].stepMode);
  sweep(VERTEX_FORMATS, (tag) => decodePipeline({ layouts: [[4, 0, [[tag, 0, 0]]]] }).buffers[0].attributes[0].format);
  for (const color of [7, 0xfe]) {
    assert.throws(() => decodePipeline({ color }), new RegExp(`unknown color format tag ${color}`));
  }
  for (const opts of [{ blend: [10, 0, 0, 0, 0, 0] }, { blend: [0, 0, 0, 0, 0, 5] }, { depth: [7, 0, 0] },
    { depth: [5, 0, 8] }, { prim: [5, 0, 0] }, { prim: [0, 2, 0] }, { prim: [0, 0, 3] },
    { layouts: [[4, 2, []]] }, { layouts: [[4, 0, [[7, 0, 0]]]] }]) {
    assert.throws(() => decodePipeline(opts), /unknown wire tag/, JSON.stringify(opts));
  }
});

// Every opcode with its encoded operands, as abi.rs `encode_cmd` writes them.
const COMMANDS = [
  [[0, u32(9)], { op: "setPipeline", pipeline: 9 }],
  [[1, u32(7, 2, 64)], { op: "setVertexBuffer", slot: 7, buffer: 2, offset: 64 }],
  [[2, u32(3), 0, u32(2)], { op: "setIndexBuffer", buffer: 3, format: "uint16", offset: 2 }],
  [[2, u32(3), 1, u32(4)], { op: "setIndexBuffer", buffer: 3, format: "uint32", offset: 4 }],
  [[3, u32(0, 4, 256)], { op: "setUniformBuffer", binding: 0, buffer: 4, offset: 256 }],
  [[4, u32(1, 5, 0xffff_ffff)], { op: "setStorageBuffer", binding: 1, buffer: 5, offset: 0xffff_ffff }],
  [[5, u32(2, 6)], { op: "setTexture", binding: 2, texture: 6 }],
  [[6, f32(-1.5, 2.25, 640, 480, 0.25, 0.75)],
    { op: "setViewport", x: -1.5, y: 2.25, width: 640, height: 480, minDepth: 0.25, maxDepth: 0.75 }],
  [[7, u32(1, 2, 3, 4)], { op: "setScissor", x: 1, y: 2, width: 3, height: 4 }],
  [[8, u32(3, 2, 1, 5)], { op: "draw", vertexCount: 3, instanceCount: 2, firstVertex: 1, firstInstance: 5 }],
  [[9, u32(6, 1, 12), i32(-4), u32(0)],
    { op: "drawIndexed", indexCount: 6, instanceCount: 1, firstIndex: 12, baseVertex: -4, firstInstance: 0 }],
];
const DRAW = COMMANDS.find(([, cmd]) => cmd.op === "draw");

test("decodeFrame reads every attachment kind and opcode across passes, from any byteOffset", () => {
  const payload = bytes(u32(3),
    // Color clear (f64 RGBA) and depth clear (f32), then every command.
    1, u32(0), 1, f64(0.1, 0.2, 0.3, 1), 0, 1, u32(11), 1, f32(0.1), 1,
    u32(COMMANDS.length), COMMANDS.map(([encoded]) => encoded),
    // Color load, no depth, no commands.
    1, u32(12), 0, 1, 0, u32(0),
    // No color, depth load.
    0, 1, u32(11), 0, 0, u32(1), DRAW[0]);
  assert.deepEqual(decodeFrame(atOffset(payload)), [
    { color: { target: 0, loadOp: "clear", clearValue: { r: 0.1, g: 0.2, b: 0.3, a: 1 }, storeOp: "store" },
      depth: { target: 11, loadOp: "clear", clearValue: Math.fround(0.1), storeOp: "discard" },
      commands: COMMANDS.map(([, cmd]) => cmd) },
    { color: { target: 12, loadOp: "load", clearValue: undefined, storeOp: "discard" }, depth: null,
      commands: [] },
    { color: null, depth: { target: 11, loadOp: "load", clearValue: undefined, storeOp: "store" },
      commands: [DRAW[1]] },
  ]);
  assert.deepEqual(decodeFrame(bytes(u32(0))), []);
  sweep(INDEX_FORMATS, (tag) => decodeFrame(bytes(u32(1), 0, 0, u32(1), 2, u32(3), tag, u32(0)))[0]
    .commands[0].format);
});

test("decodeFrame rejects unknown opcodes and tags", () => {
  assert.throws(() => decodeFrame(bytes(u32(1), 0, 0, u32(1), 10)), /unknown opcode 10/);
  for (const payload of [
    bytes(u32(1), 1, u32(0), 2), // color loadOp
    bytes(u32(1), 1, u32(0), 0, 2), // color storeOp
    bytes(u32(1), 0, 1, u32(0), 2), // depth loadOp
    bytes(u32(1), 0, 1, u32(0), 0, 2), // depth storeOp
    bytes(u32(1), 0, 0, u32(1), 2, u32(3), 2, u32(0)), // index format
  ]) assert.throws(() => decodeFrame(payload), /unknown wire tag/);
});

const VERTEX = 1;
const FRAGMENT = 2;
const keys = (map) => [...map.keys()].sort((a, b) => a - b);

test("shaderBindings derives group 0 buffer and group 1 texture/sampler layout entries", () => {
  const [group0, group1] = shaderBindings(`
    @group(0) @binding(0) var<uniform> u: vec4f;
    @group(0) @binding(1) var<storage, read> ro: array<f32>;
    @group(0) @binding(2) var<storage, read_write> rw: array<f32>;
    @group(0) @binding(3) var<storage> implicitRead: array<f32>;
    @group(1) @binding(0) var tf: texture_2d<f32>;
    @group(1) @binding(1) var s: sampler;
    @group(1) @binding(2) var tu: texture_2d< u32 >;
    @group(1) @binding(4) var ti: texture_2d<i32>;
    @group(1) @binding(6) var td: texture_depth_2d;`);
  const both = VERTEX | FRAGMENT;
  assert.deepEqual([...group0.values()], [
    { binding: 0, visibility: both, buffer: { type: "uniform" } },
    { binding: 1, visibility: both, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: FRAGMENT, buffer: { type: "storage" } }, // Vertex stages cannot write.
    { binding: 3, visibility: both, buffer: { type: "read-only-storage" } },
  ]);
  assert.deepEqual([...group1.values()], [
    { binding: 0, visibility: FRAGMENT, texture: { sampleType: "float" } },
    { binding: 1, visibility: FRAGMENT, sampler: { type: "filtering" } },
    { binding: 2, visibility: FRAGMENT, texture: { sampleType: "uint" } },
    { binding: 4, visibility: FRAGMENT, texture: { sampleType: "sint" } },
    { binding: 6, visibility: FRAGMENT, texture: { sampleType: "depth" } },
  ]);
});

test("shaderBindings keeps only resources the entry points reach", () => {
  const wgsl = `
    @group(0) @binding(0) var<uniform> a: f32;
    @group(0) @binding(1) var<uniform> b: f32;
    @group(0) @binding(2) var<uniform> unused: f32;
    @group(0) @binding(3) var<uniform> c: f32;
    fn leaf() -> f32 { return c; }
    fn helper() -> f32 { return b + leaf(); }
    fn other() -> f32 { return unused; }
    @vertex fn vs() -> @builtin(position) vec4f { return vec4f(a); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(helper()); }`;
  assert.deepEqual(keys(shaderBindings(wgsl, "vs", "fs")[0]), [0, 1, 3]);
  assert.deepEqual(keys(shaderBindings(wgsl, "vs", undefined)[0]), [0]);
  assert.deepEqual(keys(shaderBindings(wgsl)[0]), [0, 1, 2, 3]);
});

test("shaderBindings ignores names that locals, parameters, or members shadow", () => {
  const [group0] = shaderBindings(`
    @group(0) @binding(0) var<uniform> p: f32;
    @group(0) @binding(1) var<uniform> l: f32;
    @group(0) @binding(2) var<uniform> v: f32;
    @group(0) @binding(3) var<uniform> f: f32;
    @group(0) @binding(4) var<uniform> k: f32;
    @group(0) @binding(5) var<uniform> i: f32;
    @group(0) @binding(6) var<uniform> n: f32;
    @group(0) @binding(7) var<uniform> init: f32;
    @group(0) @binding(8) var<uniform> m: f32;
    struct S { m: f32 }
    @fragment fn fs(@location(0) p: f32) -> @location(0) vec4f {
      let l = p; var v = l; var<function> f = v; const k = 1.0;
      for (var i = 0; i < 2; i++) { _ = i; }
      for (var n = 0; n < 2; n++) { }
      _ = n; // The loop's \`n\` is out of scope here.
      let init = init; // An initializer sees the outer \`init\`.
      let s = S(); _ = s.m;
      return vec4f(f + k);
    }`, undefined, "fs");
  assert.deepEqual(keys(group0), [6, 7]);
});

test("shaderBindings ignores line and nested block comments", () => {
  const [group0] = shaderBindings(`
    // @group(0) @binding(0) var<uniform> line: f32;
    /* outer /* inner */ @group(0) @binding(1) var<uniform> nested: f32; */
    @group(0) @binding(2) var<uniform> real: f32;`);
  assert.deepEqual(keys(group0), [2]);
});

const PRELUDE = "const BASE = 2u; const SIX: u32 = BASE * 3; const ZERO = BASE - 2;";
const bindingOf = (expr, prelude = PRELUDE) =>
  keys(shaderBindings(`${prelude} @group(ZERO) @binding(${expr}) var<uniform> x: f32;`)[0]);

test("shaderBindings evaluates integer constant expressions in @group and @binding", () => {
  for (const [expr, value] of [["0x1F", 31], ["7u", 7], ["7i", 7], ["0x10u", 16], ["BASE + 1", 3],
    ["SIX", 6], ["1 + 2 * 3", 7], ["(1 + 2) * 3", 9], ["1 << (2 + 1)", 8], ["256 >> 4", 16],
    ["-2 + 5", 3], ["17 % 5", 2], ["17 / 5", 3], ["4, ", 4], ["0xFFFFFFFF", 0xffff_ffff]]) {
    assert.deepEqual(bindingOf(expr), [value], expr);
  }
});

test("shaderBindings resolves chained type aliases", () => {
  const [, group1] = shaderBindings(`
    alias Tex = texture_2d<u32>; alias MyTex = Tex; alias Samp = sampler;
    @group(1) @binding(0) var t: MyTex;
    @group(1) @binding(1) var s: Samp;`);
  assert.deepEqual([...group1.values()], [
    { binding: 0, visibility: FRAGMENT, texture: { sampleType: "uint" } },
    { binding: 1, visibility: FRAGMENT, sampler: { type: "filtering" } },
  ]);
});

test("shaderBindings rejects bindings outside Titan's convention and unsupported expressions", () => {
  assert.throws(() => shaderBindings("@group(2) @binding(0) var<uniform> x: f32;"), /@group\(2\)/);
  assert.throws(() => shaderBindings("@group(1) @binding(1) var s: sampler_comparison;"),
    /sampler_comparison/);
  for (const expr of ["1.5", "f(1)", "MISSING", "1 +", "()", "1 2", "-1", "0x100000000",
    "0xFFFFFFFF + 1", "1 << 64", "1 / 0", "1 % 0", "A"]) {
    assert.throws(() => bindingOf(expr, `${PRELUDE} const A = B; const B = A;`),
      /unsupported WGSL integer expression/, expr);
  }
});
