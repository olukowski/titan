// Titan's browser host: runs a Titan WASM app on WebGPU with no dependencies.
//
// Rust validates every call and sends resource ids and byte payloads, laid out as documented in
// crates/titan-wasm/src/abi.rs. This module owns the WebGPU objects, the canvas, and DOM input.
// The decoders are exported so they can be checked against real Rust-encoded bytes in Node.

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
const STORE_OPS = ["store", "discard"];
const LOAD_OPS = ["load", "clear"];
const NO_COLOR_FORMAT = 0xff;
const SURFACE_ID = 0;
const LOG_ERROR = 0;
const LOG_WARN = 1;

// Titan usage bit n maps to the WebGPU flag at index n. Numeric so Node can load this module.
const BUFFER_USAGE = [0x20 /* VERTEX */, 0x10 /* INDEX */, 0x40 /* UNIFORM */,
  0x80 /* STORAGE */, 0x04 /* COPY_SRC */, 0x08 /* COPY_DST */];
const TEXTURE_USAGE = [0x04 /* TEXTURE_BINDING */, 0x10 /* RENDER_ATTACHMENT */,
  0x01 /* COPY_SRC */, 0x02 /* COPY_DST */, 0x08 /* STORAGE_BINDING */];
const VERTEX_STAGE = 1;
const FRAGMENT_STAGE = 2;

/** DOM `KeyboardEvent.code` to Titan `KeyCode` wire code: its index in `KeyCode::ALL`. */
export const KEY_MAP = new Map([
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map((c) => `Key${c}`),
  ..."0123456789".split("").map((d) => `Digit${d}`),
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Space", "Enter", "Escape", "Tab", "Backspace",
  "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight",
  "MetaLeft", "MetaRight",
  ...Array.from({ length: 12 }, (_, i) => `F${i + 1}`),
].map((code, index) => [code, index]));

/** DOM `PointerEvent.button` (0 left, 1 middle, 2 right) to Titan's (0 left, 1 right, 2 middle). */
export function mapMouseButton(domButton) {
  return [0, 2, 1][domButton];
}

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

class Reader {
  constructor(bytes) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.pos = 0;
  }
  u8() { return this.view.getUint8(this.pos++); }
  u32() { const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  i32() { const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  f32() { const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  f64() { const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }
  bool() { return this.u8() !== 0; }
  tag(names) {
    const tag = this.u8();
    if (tag >= names.length) throw new Error(`unknown wire tag ${tag} for ${names.join("|")}`);
    return names[tag];
  }
  str() {
    const len = this.u32();
    const s = textDecoder.decode(this.bytes.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }
  optionalStr() { return this.str() || undefined; }
  list(count, read) { return Array.from({ length: count }, () => read(this)); }
}

/** Decodes an `encode_texture_desc` payload. */
export function decodeTextureDesc(bytes) {
  const r = new Reader(bytes);
  return {
    width: r.u32(), height: r.u32(), mipLevelCount: r.u32(), format: r.tag(TEXTURE_FORMATS),
    usage: r.u32(),
    sampler: { minFilter: r.tag(FILTER_MODES), magFilter: r.tag(FILTER_MODES),
      addressModeU: r.tag(ADDRESS_MODES), addressModeV: r.tag(ADDRESS_MODES) },
    label: r.optionalStr(),
  };
}

/** Decodes an `encode_shader_source` payload. */
export function decodeShaderSource(bytes) {
  const r = new Reader(bytes);
  return { wgsl: r.str(), glsl: r.optionalStr(), vertexEntry: r.str(), fragmentEntry: r.str(), label: r.optionalStr() };
}

/** Decodes an `encode_pipeline_desc` payload into WebGPU descriptor pieces. */
export function decodePipelineDesc(bytes) {
  const r = new Reader(bytes);
  const component = () => ({ srcFactor: r.tag(BLEND_FACTORS), dstFactor: r.tag(BLEND_FACTORS),
    operation: r.tag(BLEND_OPERATIONS) });
  const shaderId = r.u32();
  const colorTag = r.u8();
  const colorFormat = colorTag === NO_COLOR_FORMAT ? null : TEXTURE_FORMATS[colorTag];
  if (colorFormat === undefined) throw new Error(`unknown color format tag ${colorTag}`);
  const blend = r.bool() ? { color: component(), alpha: component() } : null;
  const depthStencil = r.bool() ? { format: r.tag(TEXTURE_FORMATS), depthWriteEnabled: r.bool(),
    depthCompare: r.tag(COMPARE_FUNCTIONS) } : null;
  const primitive = { topology: r.tag(TOPOLOGIES), frontFace: r.tag(FRONT_FACES),
    cullMode: r.tag(CULL_MODES) };
  const buffers = r.list(r.u8(), () => ({
    arrayStride: r.u32(), stepMode: r.tag(STEP_MODES),
    attributes: r.list(r.u8(), () => ({ format: r.tag(VERTEX_FORMATS), offset: r.u32(),
      shaderLocation: r.u32() })),
  }));
  return { shaderId, colorFormat, blend, depthStencil, primitive, buffers, label: r.optionalStr() };
}

const COMMANDS = [
  (r) => ({ op: "setPipeline", pipeline: r.u32() }),
  (r) => ({ op: "setVertexBuffer", slot: r.u32(), buffer: r.u32(), offset: r.u32() }),
  (r) => ({ op: "setIndexBuffer", buffer: r.u32(), format: r.tag(INDEX_FORMATS), offset: r.u32() }),
  (r) => ({ op: "setUniformBuffer", binding: r.u32(), buffer: r.u32(), offset: r.u32() }),
  (r) => ({ op: "setStorageBuffer", binding: r.u32(), buffer: r.u32(), offset: r.u32() }),
  (r) => ({ op: "setTexture", binding: r.u32(), texture: r.u32() }),
  (r) => ({ op: "setViewport", x: r.f32(), y: r.f32(), width: r.f32(), height: r.f32(),
    minDepth: r.f32(), maxDepth: r.f32() }),
  (r) => ({ op: "setScissor", x: r.u32(), y: r.u32(), width: r.u32(), height: r.u32() }),
  (r) => ({ op: "draw", vertexCount: r.u32(), instanceCount: r.u32(), firstVertex: r.u32(),
    firstInstance: r.u32() }),
  (r) => ({ op: "drawIndexed", indexCount: r.u32(), instanceCount: r.u32(), firstIndex: r.u32(),
    baseVertex: r.i32(), firstInstance: r.u32() }),
];

/** Decodes a frame payload into its passes and their commands. */
export function decodeFrame(bytes) {
  const r = new Reader(bytes);
  return r.list(r.u32(), () => {
    const color = r.bool() ? { target: r.u32(), loadOp: r.tag(LOAD_OPS) } : null;
    if (color) {
      color.clearValue = color.loadOp === "clear" ? { r: r.f64(), g: r.f64(), b: r.f64(), a: r.f64() } : undefined;
      color.storeOp = r.tag(STORE_OPS);
    }
    const depth = r.bool() ? { target: r.u32(), loadOp: r.tag(LOAD_OPS) } : null;
    if (depth) {
      depth.clearValue = depth.loadOp === "clear" ? r.f32() : undefined;
      depth.storeOp = r.tag(STORE_OPS);
    }
    const commands = r.list(r.u32(), () => {
      const opcode = r.u8();
      if (opcode >= COMMANDS.length) throw new Error(`unknown opcode ${opcode}`);
      return COMMANDS[opcode](r);
    });
    return { color, depth, commands };
  });
}

// WGSL line breaks, any of which ends a `//` comment.
const LINE_BREAK = /[\n\v\f\r\u0085\u2028\u2029]/g;

/** Removes WGSL comments. Block comments nest. */
function stripComments(wgsl) {
  let code = "";
  let depth = 0;
  for (let i = 0; i < wgsl.length; i++) {
    if (wgsl.startsWith("/*", i)) {
      depth++;
      i++;
    } else if (depth && wgsl.startsWith("*/", i)) {
      depth--;
      i++;
      if (!depth) code += " ";
    } else if (!depth && wgsl.startsWith("//", i)) {
      LINE_BREAK.lastIndex = i;
      const end = LINE_BREAK.exec(wgsl)?.index;
      if (end === undefined) break;
      code += "\n"; // Rather than the break itself, as JS `\s` does not match U+0085.
      i = end;
    } else if (!depth) {
      code += wgsl[i];
    }
  }
  return code;
}

// A WGSL identifier. It also matches a lone `_`, which never names a declaration.
const IDENT = String.raw`[_\p{XID_Start}]\p{XID_Continue}*`;
const IS_IDENT = new RegExp(`^${IDENT}$`, "u");
// WGSL tokens, coarsely: suffixed numbers, identifiers, shifts, and other single characters.
const TOKEN = new RegExp(String.raw`\d\p{XID_Continue}*|${IDENT}|<<|>>|\S`, "gu");

/** The index of the `)` closing the `(` at `tokens[open]`, or -1 if it is never closed. */
function closingParen(tokens, open) {
  let depth = 0;
  for (let k = open; k < tokens.length; k++) {
    if (tokens[k] === "(") depth++;
    else if (tokens[k] === ")" && --depth === 0) return k;
  }
  return -1;
}

/** The module-scope declaration that `scope` ends with: the text after its last `;` or `}`. */
const lastDeclaration = (scope, end = scope.length) =>
  scope.slice(Math.max(scope.lastIndexOf(";", end - 1), scope.lastIndexOf("}", end - 1)) + 1, end);

/** The name and parameter tokens of the `fn name(params) -> type` header that `declaration` is. */
function fnHeader(declaration) {
  const tokens = declaration.match(TOKEN) ?? [];
  const fn = tokens.indexOf("fn");
  if (fn < 0 || !IS_IDENT.test(tokens[fn + 1] ?? "") || tokens[fn + 2] !== "(") return undefined;
  const close = closingParen(tokens, fn + 2);
  return close < 0 ? undefined : { name: tokens[fn + 1], params: tokens.slice(fn + 3, close) };
}

/**
 * Splits comment-free WGSL into its module scope, with every `{...}` block emptied, and its
 * functions by name, each as `{ params, body }`: parameter tokens and body source text.
 */
function splitModule(code) {
  const fns = new Map();
  let scope = "";
  let depth = 0;
  let open = 0;
  let header;
  for (let i = 0; i < code.length; i++) {
    if (code[i] === "{") {
      if (depth++ === 0) [open, header] = [i, fnHeader(lastDeclaration(scope))];
    } else if (code[i] === "}") {
      if (--depth > 0) continue;
      if (header) fns.set(header.name, { params: header.params, body: code.slice(open, i + 1) });
      scope += "{}";
    } else if (depth === 0) {
      scope += code[i];
    }
  }
  return { scope, fns };
}

/**
 * The functions and module-scope identifiers that `entries` reference, directly or through calls.
 * Parameters and local `let`, `var`, and `const` declarations shadow module-scope names.
 */
function reachable(fns, entries) {
  const seen = new Set(entries);
  const queue = [...entries];
  while (queue.length) {
    const fn = fns.get(queue.pop());
    if (!fn) continue;
    const { params } = fn;
    // Parameter names are the identifiers before a `:`; attribute arguments never are.
    const scopes = [new Set(params.filter((token, k) => params[k + 1] === ":"))];
    let declaring; // A local enters scope at the `;` ending its declaration, after its initializer.
    let forLoop = false; // A `for` header's declaration is scoped to the loop, so its `{` opens none.
    const tokens = fn.body.match(TOKEN);
    for (let k = 0; k < tokens.length; k++) {
      const token = tokens[k];
      if (token === "{") {
        if (forLoop) forLoop = false;
        else scopes.push(new Set());
      } else if (token === "}") {
        if (scopes.length > 1) scopes.pop(); // Malformed WGSL may close more than it opened.
      } else if (token === "for") {
        scopes.push(new Set());
        forLoop = true;
      } else if (token === ";") {
        if (declaring) scopes.at(-1).add(declaring);
        declaring = undefined;
      } else if (token === "let" || token === "var" || token === "const") {
        if (tokens[k + 1] === "<") { // `var<function>`
          const close = tokens.indexOf(">", k + 2);
          if (close < 0) break; // Malformed; WebGPU reports the error at shader creation.
          k = close;
        }
        declaring = tokens[++k];
      } else if (IS_IDENT.test(token) && tokens[k - 1] !== "." && !seen.has(token)
        && !scopes.some((names) => names.has(token))) {
        // Skipped `.member` accesses never name a module-scope variable.
        seen.add(token);
        if (fns.has(token)) queue.push(token);
      }
    }
  }
  return seen;
}

const shiftable = (b) => b >= 0n && b < 64n; // Larger shifts overflow any WGSL integer.
// Binary operators by increasing precedence. Each returns undefined where WGSL fails.
const INT_OPERATORS = [
  { "<<": (a, b) => (shiftable(b) ? a << b : undefined), ">>": (a, b) => (shiftable(b) ? a >> b : undefined) },
  { "+": (a, b) => a + b, "-": (a, b) => a - b },
  { "*": (a, b) => a * b, "/": (a, b) => (b ? a / b : undefined), "%": (a, b) => (b ? a % b : undefined) },
];

/** Evaluates a WGSL integer constant expression, such as `0u`, `0x1`, or `BASE + 1`, to a u32. */
function evalInt(expr, consts) {
  const unsupported = () => {
    throw new Error(`unsupported WGSL integer expression "${expr.trim()}" in @group/@binding`);
  };
  // Exact BigInts never wrap: in WGSL, overflow in a constant expression fails shader creation,
  // so every suffix (`u`, `i`, or none) gives the same value for shaders WebGPU accepts.
  const visiting = new Set(); // Constants being evaluated, as WGSL rejects cyclic ones.
  const evaluate = (source) => {
    const tokens = source.match(TOKEN) ?? [];
    let pos = 0;
    const binary = (level) => {
      if (level === INT_OPERATORS.length) return unary();
      let value = binary(level + 1);
      while (Object.hasOwn(INT_OPERATORS[level], tokens[pos] ?? "")) {
        value = INT_OPERATORS[level][tokens[pos++]](value, binary(level + 1)) ?? unsupported();
      }
      return value;
    };
    const unary = () => {
      const token = tokens[pos++];
      if (token === "-") return -unary();
      if (token === "(") {
        const value = binary(0);
        if (tokens[pos++] !== ")") unsupported();
        return value;
      }
      const literal = /^(0[xX][\da-fA-F]+|\d+)[iu]?$/.exec(token);
      if (literal) return BigInt(literal[1]);
      if (consts.has(token) && !visiting.has(token)) {
        visiting.add(token);
        const value = evaluate(consts.get(token));
        visiting.delete(token);
        return value;
      }
      return unsupported();
    };
    const value = binary(0);
    if (pos !== tokens.length) unsupported();
    return value;
  };
  const value = evaluate(expr);
  if (value < 0n || value > 0xFFFF_FFFFn) unsupported();
  return Number(value);
}

// `var<space> name: type`, after its attributes. Exact WGSL keyword boundaries handle Unicode too.
const RESOURCE = new RegExp(String.raw`(?<![\p{XID_Continue}_])var(?![\p{XID_Continue}_])\s*(<[^>]*>)?\s*(${IDENT})\s*:\s*([^;={}]+)`, "gu");
const CONST = new RegExp(String.raw`(?<![\p{XID_Continue}_])const(?![\p{XID_Continue}_])\s+(${IDENT})\s*(?::[^=;]*)?=([^;]*);`, "gu");
const ALIAS = new RegExp(String.raw`(?<![\p{XID_Continue}_])alias(?![\p{XID_Continue}_])\s+(${IDENT})\s*=([^;]*);`, "gu");

/** Maps each `@name(args)` attribute in `source` to its argument source, without a trailing comma. */
function attributes(source) {
  const matches = Array.from(source.matchAll(TOKEN));
  const tokens = matches.map(([token]) => token);
  const args = new Map();
  for (let k = 0; k < tokens.length; k++) {
    if (tokens[k] !== "@" || tokens[k + 2] !== "(") continue;
    const close = closingParen(tokens, k + 2);
    if (close < 0) break;
    const arg = source.slice(matches[k + 2].index + 1, matches[close].index);
    args.set(tokens[k + 1], arg.replace(/,\s*$/, ""));
    k = close;
  }
  return args;
}

/**
 * Finds the `@group(g) @binding(b)` resources a WGSL shader declares and returns, per group 0
 * and 1, a map from binding to its bind group layout entry. Given entry points, keeps only the
 * resources they (or functions they call) reference, as WebGPU validates against those alone.
 * Titan binds uniform and storage buffers in group 0, and texture `n` and its sampler at group 1
 * bindings `2n` and `2n + 1`.
 */
export function shaderBindings(wgsl, vertexEntry, fragmentEntry) {
  const { scope, fns } = splitModule(stripComments(wgsl));
  const consts = new Map(Array.from(scope.matchAll(CONST), ([, name, expr]) => [name, expr]));
  const aliases = new Map(Array.from(scope.matchAll(ALIAS), ([, name, type]) => [name, type.trim()]));
  const entries = [vertexEntry, fragmentEntry].filter((entry) => entry !== undefined);
  const used = entries.length ? reachable(fns, entries) : null;
  const groups = [new Map(), new Map()];
  for (const resource of scope.matchAll(RESOURCE)) {
    const [, space = "", name, declaredType] = resource;
    if (used && !used.has(name)) continue;
    const args = attributes(lastDeclaration(scope, resource.index));
    if (!args.has("group") || !args.has("binding")) continue;
    const [group, binding] = [evalInt(args.get("group"), consts), evalInt(args.get("binding"), consts)];
    let type = declaredType.trim();
    // WGSL rejects cyclic aliases; the bound only keeps one from hanging the page.
    for (let n = 0; n < aliases.size && aliases.has(type); n++) type = aliases.get(type);
    if (group > 1) throw new Error(`@group(${group}) is outside Titan's binding convention`);
    const entry = { binding, visibility: FRAGMENT_STAGE };
    if (group === 0) {
      const readWrite = space.includes("read_write");
      const kind = space.includes("uniform") ? "uniform" : readWrite ? "storage" : "read-only-storage";
      // WebGPU forbids writable storage buffers in vertex shaders.
      if (!readWrite) entry.visibility |= VERTEX_STAGE;
      entry.buffer = { type: kind };
    } else if (type === "sampler_comparison") {
      throw new Error("sampler_comparison is not supported because SamplerDesc has no compare function");
    } else if (type.startsWith("sampler")) {
      entry.sampler = { type: "filtering" };
    } else {
      const sampleType = type.startsWith("texture_depth") ? "depth"
        : /<\s*u32/.test(type) ? "uint" : /<\s*i32/.test(type) ? "sint" : "float";
      entry.texture = { sampleType };
    }
    groups[group].set(binding, entry);
  }
  return groups;
}

function gpuUsage(bits, flags) {
  return flags.reduce((usage, flag, bit) => (bits & (1 << bit) ? usage | flag : usage), 0);
}

function lookup(map, id, kind) {
  const value = map.get(id);
  if (value === undefined) throw new Error(`unknown ${kind} id ${id}`);
  return value;
}

/**
 * Loads the Titan app at `wasmUrl` and runs it on `canvas` until it exits or fails.
 * Resolves to `{ dispose }` once running; rejects if WebGPU or the module cannot start.
 * Later failures, including WebGPU validation errors, stop the app and call `onError`.
 */
export async function runTitan({ canvas, wasmUrl, onError = console.error }) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser. Try a recent Chrome, Edge, or " +
      "Safari, or enable WebGPU in the browser's settings.");
  }
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Could not acquire a WebGPU canvas context.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU is available, but no GPU adapter was found.");
  const device = await adapter.requestDevice();
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  // Render through an sRGB view so shader output is gamma-encoded like on native surfaces.
  const surfaceFormat = canvasFormat.endsWith("-srgb") ? canvasFormat : `${canvasFormat}-srgb`;

  // "starting" until runTitan resolves, then "running" until exit, failure, or dispose.
  let state = "starting";
  let startupError = null;
  let frameHandle = 0;
  let observer = null;
  let lastError = null;
  const listeners = [];
  const shutdown = () => {
    if (state === "stopped") return;
    // Stopped first, so the device loss from destroy() below is not reported as a failure.
    state = "stopped";
    cancelAnimationFrame(frameHandle);
    observer?.disconnect();
    for (const [target, type, fn] of listeners) target.removeEventListener(type, fn);
    device.destroy();
  };
  const fail = (error) => {
    if (state === "starting") {
      startupError ??= error;
    } else if (state === "running") {
      shutdown();
      onError(error);
    }
  };
  device.lost.then((info) => fail(new Error(`The GPU device was lost (${info.reason}): ${info.message}`)));
  device.addEventListener("uncapturederror", (e) => fail(new Error(`WebGPU error: ${e.error.message}`)));

  const buffers = new Map();
  const textures = new Map();
  const shaders = new Map();
  const pipelines = new Map();
  let memory = null;
  let pendingError = new Uint8Array(0);
  const bytes = (ptr, len) => new Uint8Array(memory.buffer, ptr, len);
  // Wraps an import so a thrown error becomes Titan's error convention: the message's length.
  const fallible = (fn) => (...args) => {
    try {
      fn(...args);
      return 0;
    } catch (error) {
      pendingError = textEncoder.encode(String(error?.message ?? error) || "unknown host error");
      return pendingError.length;
    }
  };

  // Rust checks that an offset lies inside its buffer; these are the device's WebGPU limits.
  function bufferBinding(entry, { buffer: id, offset }) {
    const { buffer, size } = lookup(buffers, id, "buffer");
    const uniform = entry.buffer.type === "uniform";
    const { limits } = device;
    const alignment = uniform ? limits.minUniformBufferOffsetAlignment : limits.minStorageBufferOffsetAlignment;
    const maxSize = uniform ? limits.maxUniformBufferBindingSize : limits.maxStorageBufferBindingSize;
    const bound = size - offset;
    const where = `${entry.buffer.type} buffer at @group(0) @binding(${entry.binding})`;
    if (offset % alignment) throw new Error(`${where}: offset ${offset} must be a multiple of ${alignment}`);
    if (bound <= 0 || bound > maxSize) {
      throw new Error(`${where}: the ${bound} bytes after offset ${offset} must be 1 to ${maxSize}`);
    }
    if (!uniform && bound % 4) throw new Error(`${where}: the ${bound} bytes after offset ${offset} must be a multiple of 4`);
    return { buffer, offset };
  }

  // WebGPU needs an indexed strip draw's index format in the pipeline, so strip pipelines get a
  // variant per format, made on first use.
  function stripVariant(entry, format) {
    let variant = entry.strips.get(format);
    if (!variant) {
      variant = device.createRenderPipeline({ ...entry.desc,
        primitive: { ...entry.desc.primitive, stripIndexFormat: format } });
      entry.strips.set(format, variant);
    }
    return variant;
  }

  function runPass(pass, commands) {
    let pipeline = null;
    let active = null; // The GPURenderPipeline set on the pass: `pipeline`'s or a strip variant.
    let indexFormat;
    let dirty = false;
    const bound = [new Map(), new Map()];
    const resource = (group, entry) => {
      const value = bound[group].get(group === 0 ? entry.binding : entry.binding >> 1);
      if (!value) throw new Error(`a draw needs a resource for @group(${group}) @binding(${entry.binding})`);
      if (group === 0) return bufferBinding(entry, value);
      return entry.binding % 2 ? value.sampler : value.view;
    };
    const flushBindGroups = () => {
      if (!pipeline) throw new Error("draw without a pipeline");
      if (!dirty) return;
      dirty = false;
      pipeline.layouts.forEach((layout, group) => {
        const entries = [...pipeline.bindings[group].values()].map((entry) => ({ binding: entry.binding, resource: resource(group, entry) }));
        pass.setBindGroup(group, device.createBindGroup({ layout, entries }));
      });
    };
    for (const c of commands) {
      switch (c.op) {
        case "setPipeline":
          pipeline = lookup(pipelines, c.pipeline, "pipeline");
          pass.setPipeline(active = pipeline.pipeline);
          dirty = true;
          break;
        case "setVertexBuffer": pass.setVertexBuffer(c.slot, lookup(buffers, c.buffer, "buffer").buffer, c.offset); break;
        case "setIndexBuffer":
          indexFormat = c.format;
          pass.setIndexBuffer(lookup(buffers, c.buffer, "buffer").buffer, c.format, c.offset);
          break;
        case "setUniformBuffer":
        case "setStorageBuffer": bound[0].set(c.binding, c); dirty = true; break;
        case "setTexture": bound[1].set(c.binding, lookup(textures, c.texture, "texture")); dirty = true; break;
        case "setViewport": pass.setViewport(c.x, c.y, c.width, c.height, c.minDepth, c.maxDepth); break;
        case "setScissor": pass.setScissorRect(c.x, c.y, c.width, c.height); break;
        case "draw":
          flushBindGroups();
          pass.draw(c.vertexCount, c.instanceCount, c.firstVertex, c.firstInstance);
          break;
        case "drawIndexed": {
          flushBindGroups();
          const wanted = pipeline.strips ? stripVariant(pipeline, indexFormat) : pipeline.pipeline;
          if (wanted !== active) pass.setPipeline(active = wanted);
          pass.drawIndexed(c.indexCount, c.instanceCount, c.firstIndex, c.baseVertex, c.firstInstance);
          break;
        }
      }
    }
  }

  function submitFrame(passes) {
    const encoder = device.createCommandEncoder();
    let surfaceView = null;
    for (const { color, depth, commands } of passes) {
      const desc = { colorAttachments: [] };
      if (color) {
        const view = color.target === SURFACE_ID
          ? (surfaceView ??= context.getCurrentTexture().createView({ format: surfaceFormat }))
          : lookup(textures, color.target, "texture").attachmentView;
        desc.colorAttachments.push({ view, loadOp: color.loadOp, clearValue: color.clearValue, storeOp: color.storeOp });
      }
      if (depth) {
        const target = lookup(textures, depth.target, "texture");
        desc.depthStencilAttachment = { view: target.attachmentView, depthLoadOp: depth.loadOp,
          depthClearValue: depth.clearValue, depthStoreOp: depth.storeOp };
        if (target.format === "depth24plus-stencil8") {
          Object.assign(desc.depthStencilAttachment, { stencilLoadOp: "clear", stencilStoreOp: "discard" });
        }
      }
      const pass = encoder.beginRenderPass(desc);
      runPass(pass, commands);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
  }

  const imports = {
    surface_format: () => TEXTURE_FORMATS.indexOf(surfaceFormat),
    create_buffer: fallible((id, size, usage, labelPtr, labelLen) => {
      const label = textDecoder.decode(bytes(labelPtr, labelLen)) || undefined;
      buffers.set(id, { buffer: device.createBuffer({ label, size, usage: gpuUsage(usage, BUFFER_USAGE) }), size });
    }),
    write_buffer: fallible((id, offset, ptr, len) => {
      device.queue.writeBuffer(lookup(buffers, id, "buffer").buffer, offset, memory.buffer, ptr, len);
    }),
    destroy_buffer: (id) => { buffers.get(id)?.buffer.destroy(); buffers.delete(id); },
    create_texture: fallible((id, ptr, len) => {
      const d = decodeTextureDesc(bytes(ptr, len));
      const texture = device.createTexture({ label: d.label, size: [d.width, d.height],
        mipLevelCount: d.mipLevelCount, format: d.format, usage: gpuUsage(d.usage, TEXTURE_USAGE) });
      const full = texture.createView();
      textures.set(id, {
        texture, format: d.format, width: d.width, height: d.height,
        // Shaders sample a combined depth-stencil format through its depth aspect only.
        view: d.format === "depth24plus-stencil8" ? texture.createView({ aspect: "depth-only" }) : full,
        attachmentView: d.mipLevelCount > 1 ? texture.createView({ mipLevelCount: 1 }) : full,
        sampler: device.createSampler({ ...d.sampler, mipmapFilter: d.sampler.minFilter }),
        bytesPerPixel: d.format === "r8unorm" ? 1 : 4,
      });
    }),
    write_texture: fallible((id, ptr, len) => {
      const t = lookup(textures, id, "texture");
      device.queue.writeTexture({ texture: t.texture }, memory.buffer,
        { offset: ptr, bytesPerRow: t.width * t.bytesPerPixel, rowsPerImage: t.height }, [t.width, t.height]);
    }),
    destroy_texture: (id) => { textures.get(id)?.texture.destroy(); textures.delete(id); },
    create_shader: fallible((id, ptr, len) => {
      const s = decodeShaderSource(bytes(ptr, len));
      shaders.set(id, { module: device.createShaderModule({ label: s.label, code: s.wgsl }),
        vertexEntry: s.vertexEntry, fragmentEntry: s.fragmentEntry,
        bindings: shaderBindings(s.wgsl, s.vertexEntry, s.fragmentEntry) });
    }),
    destroy_shader: (id) => { shaders.delete(id); },
    create_pipeline: fallible((id, ptr, len) => {
      const d = decodePipelineDesc(bytes(ptr, len));
      const shader = lookup(shaders, d.shaderId, "shader");
      for (const { srcFactor, dstFactor, operation } of d.blend ? [d.blend.color, d.blend.alpha] : []) {
        if ((operation === "min" || operation === "max") && (srcFactor !== "one" || dstFactor !== "one")) {
          throw new Error(`${operation} blending needs src and dst factors of one`);
        }
      }
      // Explicit layouts, so declared-but-unused bindings stay in them and every bind group fits.
      const groupCount = shader.bindings[1].size ? 2 : shader.bindings[0].size ? 1 : 0;
      const layouts = shader.bindings.slice(0, groupCount)
        .map((group) => device.createBindGroupLayout({ entries: [...group.values()] }));
      const desc = {
        label: d.label,
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module: shader.module, entryPoint: shader.vertexEntry, buffers: d.buffers },
        fragment: { module: shader.module, entryPoint: shader.fragmentEntry,
          targets: d.colorFormat ? [{ format: d.colorFormat, blend: d.blend ?? undefined }] : [] },
        primitive: d.primitive,
        depthStencil: d.depthStencil ?? undefined,
      };
      const strip = d.primitive.topology.endsWith("-strip");
      pipelines.set(id, { pipeline: device.createRenderPipeline(desc), layouts, bindings: shader.bindings,
        desc, strips: strip ? new Map() : null });
    }),
    destroy_pipeline: (id) => { pipelines.delete(id); },
    submit_frame: fallible((ptr, len) => submitFrame(decodeFrame(bytes(ptr, len)))),
    read_error: (ptr, len) => bytes(ptr, len).set(pendingError.subarray(0, len)),
    log: (level, ptr, len) => {
      const message = textDecoder.decode(bytes(ptr, len));
      if (level === LOG_ERROR) {
        lastError = message;
        console.error(message);
      } else if (level === LOG_WARN) {
        console.warn(message);
      } else {
        console.info(message);
      }
    },
  };
  // Every import argument is a Rust `u32`, which JS receives as a signed i32: addresses past
  // 2 GiB would otherwise turn negative.
  const unsignedImports = Object.fromEntries(Object.entries(imports)
    .map(([name, fn]) => [name, (...args) => fn(...args.map((arg) => arg >>> 0))]));

  const failed = (name) => new Error(lastError ?? `${name} failed`);
  const maxSize = device.limits.maxTextureDimension2D;
  let lastScale = 0;
  const resizeCanvas = (width, height, scale) => {
    [width, height] = [Math.min(width, maxSize), Math.min(height, maxSize)];
    if (width === canvas.width && height === canvas.height && scale === lastScale) return false;
    [canvas.width, canvas.height, lastScale] = [width, height, scale];
    return true;
  };

  let app;
  try {
    context.configure({ device, format: canvasFormat, viewFormats: [surfaceFormat], alphaMode: "opaque" });
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`Could not load ${wasmUrl}: HTTP ${response.status} ${response.statusText}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { titan: unsignedImports });
    // Device loss or a WebGPU error while awaiting above aborts startup; nothing awaits below.
    if (startupError) throw startupError;
    app = instance.exports;
    memory = app.memory;
    const rect = canvas.getBoundingClientRect();
    resizeCanvas(Math.round(rect.width * devicePixelRatio), Math.round(rect.height * devicePixelRatio), devicePixelRatio);
    if (app.titan_init(canvas.width, canvas.height, lastScale) !== 0) throw failed("titan_init");
  } catch (error) {
    shutdown();
    throw error;
  }
  state = "running";

  // Calls an app export. A trap, such as a Rust panic, stops the app and returns undefined.
  const callExport = (name, ...args) => {
    if (state !== "running") return undefined;
    try {
      return app[name](...args);
    } catch (error) {
      fail(error);
      return undefined;
    }
  };
  const resize = (width, height) => {
    if (resizeCanvas(width, height, devicePixelRatio)
      && callExport("titan_resize", canvas.width, canvas.height, lastScale) < 0) {
      fail(failed("titan_resize"));
    }
  };

  // The CSS size and whether the observer measures device pixels, for pixel-ratio changes in tick.
  let { width: cssWidth, height: cssHeight } = canvas.getBoundingClientRect();
  let hasDevicePixelBox = false;
  observer = new ResizeObserver(([entry]) => {
    const box = entry.devicePixelContentBoxSize?.[0];
    hasDevicePixelBox = Boolean(box);
    if (box) resize(box.inlineSize, box.blockSize);
    else {
      [cssWidth, cssHeight] = [entry.contentRect.width, entry.contentRect.height];
      resize(Math.round(cssWidth * devicePixelRatio), Math.round(cssHeight * devicePixelRatio));
    }
  });
  try {
    observer.observe(canvas, { box: "device-pixel-content-box" });
  } catch {
    observer.observe(canvas); // Safari lacks device-pixel-content-box.
  }

  const listen = (target, type, fn, options) => {
    target.addEventListener(type, fn, options);
    listeners.push([target, type, fn]);
  };
  const heldKeys = new Set();
  const onKey = (down) => (e) => {
    const code = KEY_MAP.get(e.code);
    if (code === undefined) return;
    if (e.target === canvas) e.preventDefault(); // Keep Space and arrows from scrolling the page.
    if (down) heldKeys.add(code); else heldKeys.delete(code);
    callExport("titan_key", code, down ? 1 : 0);
  };
  // DOM `buttons` bits 1, 2, and 4 (left, right, middle) are Titan's buttons 0, 1, and 2.
  let heldButtons = 0;
  const syncButtons = (buttons) => {
    for (let button = 0; button < 3; button++) {
      const bit = 1 << button;
      if ((buttons ^ heldButtons) & bit) callExport("titan_pointer_button", button, buttons & bit ? 1 : 0);
    }
    heldButtons = buttons;
  };
  // Chorded presses and releases arrive as pointermove, so every event reports the whole mask.
  const pointerButtons = (e, down) => {
    if (typeof e.buttons === "number") return e.buttons & 7;
    const button = mapMouseButton(e.button);
    if (button === undefined) return heldButtons;
    return down ? heldButtons | (1 << button) : heldButtons & ~(1 << button);
  };
  listen(window, "keydown", onKey(true));
  listen(window, "keyup", onKey(false));
  // Keys and buttons released while the page is unfocused send no event; release them so none stick.
  // The pointer whose press the canvas accepted. Buttons of any other pointer, such as a drag
  // that began outside the canvas, belong to someone else.
  let activePointerId = null;
  const releaseButtons = () => {
    syncButtons(0);
    activePointerId = null;
  };
  const releasePointer = (e) => {
    if (activePointerId === null || e.pointerId === activePointerId) releaseButtons();
  };
  listen(window, "blur", () => {
    for (const code of heldKeys) callExport("titan_key", code, 0);
    heldKeys.clear();
    releaseButtons();
  });
  listen(canvas, "pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    callExport("titan_pointer_move", e.clientX - r.left, e.clientY - r.top);
    if (e.pointerId === activePointerId) syncButtons(pointerButtons(e, false));
  });
  listen(canvas, "pointerdown", (e) => {
    activePointerId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    canvas.focus();
    syncButtons(pointerButtons(e, true));
  });
  listen(canvas, "pointerup", (e) => {
    if (e.pointerId !== activePointerId) return;
    syncButtons(pointerButtons(e, false));
    // The pointer stays ours while any button is down, including side buttons Titan ignores.
    if ((e.buttons ?? 0) === 0) activePointerId = null;
  });
  listen(canvas, "pointercancel", releasePointer);
  listen(canvas, "lostpointercapture", releasePointer);
  listen(canvas, "wheel", (e) => {
    e.preventDefault();
    const unit = [1, 16, canvas.clientHeight][e.deltaMode] ?? 1; // Pixels, lines, or pages.
    callExport("titan_scroll", e.deltaX * unit, e.deltaY * unit);
  }, { passive: false });
  listen(canvas, "contextmenu", (e) => e.preventDefault());

  let previous = performance.now();
  const tick = (now) => {
    if (state !== "running") return;
    const dtSecs = (now - previous) / 1000;
    previous = now;
    // Zoom can change the pixel ratio without resizing the device-pixel box the observer watches;
    // a CSS content box keeps its size, but now spans a different number of pixels.
    if (devicePixelRatio !== lastScale) {
      if (hasDevicePixelBox) resize(canvas.width, canvas.height);
      else resize(Math.round(cssWidth * devicePixelRatio), Math.round(cssHeight * devicePixelRatio));
    }
    if (canvas.width > 0 && canvas.height > 0) { // Otherwise there is nothing to draw into.
      const result = callExport("titan_frame", dtSecs);
      if (result === 1) shutdown();
      else if (result < 0) fail(failed("titan_frame"));
    }
    if (state === "running") frameHandle = requestAnimationFrame(tick);
  };
  frameHandle = requestAnimationFrame(tick);

  return { dispose: shutdown };
}
