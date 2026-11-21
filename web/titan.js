// Titan's browser host: runs a Titan WASM app on WebGPU, or on WebGL2 where WebGPU is
// unavailable, with no dependencies.
//
// Rust validates every call and sends resource ids and byte payloads, laid out as documented in
// crates/titan-wasm/src/abi.rs. This module decodes them once and hands them to a backend that
// owns the GPU objects; it also owns the canvas and DOM input. The decoders and the WebGL2
// translation are exported so they can be checked in Node.

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

/** Creates the WebGPU backend: the device, the configured canvas context, and every GPU object. */
async function createWebGPUBackend(canvas, adapter, fail) {
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Could not acquire a WebGPU canvas context.");
  const device = await adapter.requestDevice();
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  // Render through an sRGB view so shader output is gamma-encoded like on native surfaces.
  const surfaceFormat = canvasFormat.endsWith("-srgb") ? canvasFormat : `${canvasFormat}-srgb`;
  device.lost.then((info) => fail(new Error(`The GPU device was lost (${info.reason}): ${info.message}`)));
  device.addEventListener("uncapturederror", (e) => fail(new Error(`WebGPU error: ${e.error.message}`)));
  try {
    context.configure({ device, format: canvasFormat, viewFormats: [surfaceFormat], alphaMode: "opaque" });
  } catch (error) {
    device.destroy(); // runTitan never receives this backend, so its shutdown cannot.
    throw error;
  }

  const buffers = new Map();
  const textures = new Map();
  const shaders = new Map();
  const pipelines = new Map();

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

  return {
    name: "webgpu",
    surfaceFormat,
    maxTextureSize: device.limits.maxTextureDimension2D,
    createBuffer(id, size, usage, label) {
      buffers.set(id, { buffer: device.createBuffer({ label, size, usage: gpuUsage(usage, BUFFER_USAGE) }), size });
    },
    writeBuffer(id, offset, data) {
      device.queue.writeBuffer(lookup(buffers, id, "buffer").buffer, offset, data);
    },
    destroyBuffer(id) { buffers.get(id)?.buffer.destroy(); buffers.delete(id); },
    createTexture(id, d) {
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
    },
    writeTexture(id, data) {
      const t = lookup(textures, id, "texture");
      device.queue.writeTexture({ texture: t.texture }, data,
        { bytesPerRow: t.width * t.bytesPerPixel, rowsPerImage: t.height }, [t.width, t.height]);
    },
    destroyTexture(id) { textures.get(id)?.texture.destroy(); textures.delete(id); },
    createShader(id, s) {
      shaders.set(id, { module: device.createShaderModule({ label: s.label, code: s.wgsl }),
        vertexEntry: s.vertexEntry, fragmentEntry: s.fragmentEntry,
        bindings: shaderBindings(s.wgsl, s.vertexEntry, s.fragmentEntry) });
    },
    destroyShader(id) { shaders.delete(id); },
    createPipeline(id, d) {
      const shader = lookup(shaders, d.shaderId, "shader");
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
    },
    destroyPipeline(id) { pipelines.delete(id); },
    submitFrame(passes) {
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
    },
    destroy() { device.destroy(); },
  };
}

// WebGL2 enum names per WebGPU value. The backend resolves each as `gl[name]`, so this module
// needs no numeric GL constants.
export const GL_NAMES = {
  blendFactor: { zero: "ZERO", one: "ONE", "src-alpha": "SRC_ALPHA", "one-minus-src-alpha": "ONE_MINUS_SRC_ALPHA",
    "dst-alpha": "DST_ALPHA", "one-minus-dst-alpha": "ONE_MINUS_DST_ALPHA", src: "SRC_COLOR",
    "one-minus-src": "ONE_MINUS_SRC_COLOR", dst: "DST_COLOR", "one-minus-dst": "ONE_MINUS_DST_COLOR" },
  blendOperation: { add: "FUNC_ADD", subtract: "FUNC_SUBTRACT", "reverse-subtract": "FUNC_REVERSE_SUBTRACT",
    min: "MIN", max: "MAX" },
  compare: { never: "NEVER", less: "LESS", "less-equal": "LEQUAL", equal: "EQUAL", "greater-equal": "GEQUAL",
    greater: "GREATER", "not-equal": "NOTEQUAL", always: "ALWAYS" },
  // WebGL2 restarts every draw at the maximum index value, lists included: a uint16 list cannot
  // use index 65535, so meshes with 65536+ vertices need uint32 indices.
  topology: { "triangle-list": "TRIANGLES", "triangle-strip": "TRIANGLE_STRIP", "line-list": "LINES",
    "line-strip": "LINE_STRIP", "point-list": "POINTS" },
  cullFace: { none: null, front: "FRONT", back: "BACK" },
  frontFace: { ccw: "CCW", cw: "CW" },
  indexFormat: { uint16: "UNSIGNED_SHORT", uint32: "UNSIGNED_INT" },
  addressMode: { "clamp-to-edge": "CLAMP_TO_EDGE", repeat: "REPEAT", "mirror-repeat": "MIRRORED_REPEAT" },
  // `integer` attributes go through vertexAttribIPointer, so shaders read them as uint or int.
  vertexFormat: {
    float32: { size: 1, type: "FLOAT", normalized: false, integer: false },
    float32x2: { size: 2, type: "FLOAT", normalized: false, integer: false },
    float32x3: { size: 3, type: "FLOAT", normalized: false, integer: false },
    float32x4: { size: 4, type: "FLOAT", normalized: false, integer: false },
    uint32: { size: 1, type: "UNSIGNED_INT", normalized: false, integer: true },
    sint32: { size: 1, type: "INT", normalized: false, integer: true },
    unorm8x4: { size: 4, type: "UNSIGNED_BYTE", normalized: true, integer: false },
  },
  textureFormat: {
    rgba8unorm: { internalFormat: "RGBA8", format: "RGBA", type: "UNSIGNED_BYTE" },
    "rgba8unorm-srgb": { internalFormat: "SRGB8_ALPHA8", format: "RGBA", type: "UNSIGNED_BYTE" },
    r8unorm: { internalFormat: "R8", format: "RED", type: "UNSIGNED_BYTE" },
    depth32float: { internalFormat: "DEPTH_COMPONENT32F", format: "DEPTH_COMPONENT", type: "FLOAT" },
    "depth24plus-stencil8": { internalFormat: "DEPTH24_STENCIL8", format: "DEPTH_STENCIL", type: "UNSIGNED_INT_24_8" },
  },
};

/** The `{ internalFormat, format, type }` GL names for a WebGPU texture format. */
export function glTextureFormat(format) {
  const names = GL_NAMES.textureFormat[format];
  if (!names) throw new Error(`texture format ${format} is not supported on WebGL2; use rgba8unorm instead`);
  return names;
}

/** The samplerParameteri name/value pairs for a decoded sampler, as GL names. */
export function glSamplerParams(sampler, mipLevelCount) {
  const filter = (mode) => (mode === "linear" ? "LINEAR" : "NEAREST");
  const min = filter(sampler.minFilter);
  return {
    // WebGPU's sampler filters between mip levels with the min filter too.
    TEXTURE_MIN_FILTER: mipLevelCount > 1 ? `${min}_MIPMAP_${min}` : min,
    TEXTURE_MAG_FILTER: filter(sampler.magFilter),
    TEXTURE_WRAP_S: GL_NAMES.addressMode[sampler.addressModeU],
    TEXTURE_WRAP_T: GL_NAMES.addressMode[sampler.addressModeV],
  };
}

/** One stage of a Titan GLSL source, which wraps each stage in `#ifdef TITAN_<STAGE>`. */
export const glslStage = (stage, source) => `#version 300 es\n#define TITAN_${stage}\n${source}`;

/** The Titan binding a GLSL uniform block or sampler names with its `_binding<N>` suffix. */
export function glslBinding(name) {
  const match = /_binding(0|[1-9]\d*)$/.exec(name);
  if (!match) throw new Error(`GLSL uniform "${name}" must end in _binding<N> to name its Titan binding`);
  return Number(match[1]);
}

/**
 * The `[x, y, width, height]` GL rect of a top-left-origin WebGPU rect, with GL's bottom-left
 * origin. GL truncates fractional rects, so each edge is rounded to the nearest pixel instead.
 */
export function glRect(x, y, width, height, targetHeight) {
  const [x0, x1, y0, y1] = [x, x + width, y, y + height].map(Math.round);
  return [x0, targetHeight - y1, x1 - x0, y1 - y0];
}

const [VERTEX_USAGE, INDEX_USAGE, UNIFORM_USAGE, STORAGE_USAGE] = [1, 2, 4, 8];
const NO_STORAGE = "storage buffers are not supported on WebGL2";
const SAMPLER_TYPES = ["SAMPLER_2D", "SAMPLER_2D_SHADOW", "INT_SAMPLER_2D", "UNSIGNED_INT_SAMPLER_2D"];

// Copies the sRGB surface texture to the canvas. Fetching decodes it to linear, so this encodes
// it again: the default framebuffer stores what it is given.
const PRESENT_VERTEX = `#version 300 es
void main() {
  gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0);
}`;
const PRESENT_FRAGMENT = `#version 300 es
precision highp float;
uniform highp sampler2D surface;
out vec4 color;
void main() {
  vec3 c = texelFetch(surface, ivec2(gl_FragCoord.xy), 0).rgb;
  color = vec4(mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)), 1.0);
}`;

/**
 * Creates the WebGL2 backend on `gl`. Surface passes draw into an sRGB texture, so blending
 * happens in linear space like WebGPU's -srgb view, and each frame that drew it ends by copying
 * it to the canvas.
 */
export function createWebGL2Backend(gl, fail) {
  const onContextLost = () => fail(new Error("The WebGL2 context was lost."));
  gl.canvas.addEventListener("webglcontextlost", onContextLost);
  const uniformAlignment = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT);
  const maxUniformSize = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
  const buffers = new Map();
  const textures = new Map();
  const shaders = new Map();
  const pipelines = new Map();
  const framebuffers = new Map(); // "<color id>/<depth id>" to { framebuffer, ids }.
  let surface = null; // { texture, width, height }, sized to the drawing buffer.
  let enabledAttributes = new Set();

  const compile = (type, stage, source, label) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`${label}: the GLSL ${stage} stage failed to compile:\n${log}`);
    }
    return shader;
  };
  const link = ([vertex, fragment], label) => {
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`${label}: the GLSL program failed to link:\n${log}`);
    }
    return program;
  };
  const presentStages = [compile(gl.VERTEX_SHADER, "vertex", PRESENT_VERTEX, "present"),
    compile(gl.FRAGMENT_SHADER, "fragment", PRESENT_FRAGMENT, "present")];
  const presentProgram = link(presentStages, "present");
  const vao = gl.createVertexArray();
  const presentVao = gl.createVertexArray();
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

  const dropFramebuffers = (id) => {
    for (const [key, { framebuffer, ids }] of framebuffers) {
      if (!ids.includes(id)) continue;
      gl.deleteFramebuffer(framebuffer);
      framebuffers.delete(key);
    }
  };
  const target = (id) => {
    if (id !== SURFACE_ID) return lookup(textures, id, "texture");
    const { drawingBufferWidth: width, drawingBufferHeight: height } = gl;
    if (surface?.width !== width || surface.height !== height) {
      if (surface) {
        gl.deleteTexture(surface.texture);
        dropFramebuffers(SURFACE_ID);
      }
      surface = { texture: gl.createTexture(), width, height };
      gl.bindTexture(gl.TEXTURE_2D, surface.texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.SRGB8_ALPHA8, width, height);
    }
    return surface;
  };
  const bindFramebuffer = (colorId, depthId) => {
    const key = `${colorId}/${depthId}`;
    let entry = framebuffers.get(key);
    if (!entry) {
      entry = { framebuffer: gl.createFramebuffer(), ids: [colorId, depthId] };
      gl.bindFramebuffer(gl.FRAMEBUFFER, entry.framebuffer);
      if (colorId !== undefined) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target(colorId).texture, 0);
      }
      if (depthId !== undefined) {
        const depth = target(depthId);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, depth.stencil ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT,
          gl.TEXTURE_2D, depth.texture, 0);
      }
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`WebGL2 cannot render to this pass's attachments (status ${status})`);
      }
      framebuffers.set(key, entry);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, entry.framebuffer);
  };

  // GL stores rendered rows bottom-up, so a texture target sampled later is upside down relative
  // to WebGPU. Only the surface, which the present pass copies row for row, is unaffected.
  function runPass({ color, depth, commands }) {
    const colorTarget = color && target(color.target);
    const depthTarget = depth && target(depth.target);
    const { width, height } = colorTarget ?? depthTarget;
    bindFramebuffer(color?.target, depth?.target);
    // WebGPU starts every pass with a full viewport and no scissor.
    gl.viewport(0, 0, width, height);
    gl.depthRange(0, 1);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    let clear = 0;
    if (color?.loadOp === "clear") {
      const { r, g, b, a } = color.clearValue;
      gl.clearColor(r, g, b, a);
      clear |= gl.COLOR_BUFFER_BIT;
    }
    if (depth?.loadOp === "clear") {
      gl.depthMask(true);
      gl.clearDepth(depth.clearValue);
      clear |= gl.DEPTH_BUFFER_BIT;
    }
    if (depthTarget?.stencil) clear |= gl.STENCIL_BUFFER_BIT; // As the WebGPU backend's stencilLoadOp.
    if (clear) gl.clear(clear);

    let pipeline = null;
    let index = null;
    const vertexBuffers = [];
    const uniforms = new Map();
    const boundTextures = new Map();
    // WebGL2 has no base vertex or first instance, so they move the attribute offsets instead.
    // gl_VertexID and gl_InstanceID cannot see that move.
    const bindResources = (baseVertex, firstInstance) => {
      if (!pipeline) throw new Error("draw without a pipeline");
      if (baseVertex && pipeline.usesVertexId) {
        throw new Error(`${pipeline.label}: WebGL2 cannot apply baseVertex ${baseVertex} to gl_VertexID`);
      }
      if (firstInstance && pipeline.usesInstanceId) {
        throw new Error(`${pipeline.label}: WebGL2 cannot apply firstInstance ${firstInstance} to gl_InstanceID`);
      }
      for (const binding of pipeline.uniformBindings) {
        const c = uniforms.get(binding);
        if (!c) throw new Error(`a draw needs a uniform buffer at binding ${binding}`);
        const { buffer, size } = lookup(buffers, c.buffer, "buffer");
        const bound = size - c.offset;
        const where = `uniform buffer at binding ${binding}`;
        if (c.offset % uniformAlignment) {
          throw new Error(`${where}: offset ${c.offset} must be a multiple of ${uniformAlignment}`);
        }
        if (bound <= 0) throw new Error(`${where}: offset ${c.offset} leaves no bytes of its ${size}-byte buffer`);
        // No block can exceed MAX_UNIFORM_BLOCK_SIZE (16 KiB at minimum), so bind at most that much.
        gl.bindBufferRange(gl.UNIFORM_BUFFER, binding, buffer, c.offset, Math.min(bound, maxUniformSize));
      }
      for (const binding of pipeline.textureBindings) {
        const t = boundTextures.get(binding);
        if (!t) throw new Error(`a draw needs a texture at binding ${binding}`);
        gl.activeTexture(gl.TEXTURE0 + binding);
        gl.bindTexture(gl.TEXTURE_2D, t.texture);
        gl.bindSampler(binding, t.sampler);
      }
      const used = new Set();
      pipeline.buffers.forEach(({ arrayStride, stepMode, attributes }, slot) => {
        const instance = stepMode === "instance";
        const start = vertexBuffers[slot].offset + (instance ? firstInstance : baseVertex) * arrayStride;
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffers[slot].buffer);
        for (const { location, offset, size, type, normalized, integer } of attributes) {
          if (start + offset < 0) {
            throw new Error(`baseVertex ${baseVertex} moves attribute ${location} before its buffer's start`);
          }
          gl.enableVertexAttribArray(location);
          used.add(location);
          if (integer) gl.vertexAttribIPointer(location, size, type, arrayStride, start + offset);
          else gl.vertexAttribPointer(location, size, type, normalized, arrayStride, start + offset);
          gl.vertexAttribDivisor(location, instance ? 1 : 0);
        }
      });
      for (const location of enabledAttributes) if (!used.has(location)) gl.disableVertexAttribArray(location);
      enabledAttributes = used;
    };
    for (const c of commands) {
      switch (c.op) {
        case "setPipeline": {
          pipeline = lookup(pipelines, c.pipeline, "pipeline");
          const { program, blend, depthTest, cullFace, frontFace } = pipeline;
          gl.useProgram(program);
          if (blend) {
            gl.enable(gl.BLEND);
            gl.blendFuncSeparate(...blend.factors);
            gl.blendEquationSeparate(...blend.operations);
          } else gl.disable(gl.BLEND);
          if (depthTest) {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(depthTest.func);
            gl.depthMask(depthTest.write);
          } else gl.disable(gl.DEPTH_TEST);
          if (cullFace) {
            gl.enable(gl.CULL_FACE);
            gl.cullFace(cullFace);
          } else gl.disable(gl.CULL_FACE);
          gl.frontFace(frontFace);
          break;
        }
        case "setVertexBuffer":
          vertexBuffers[c.slot] = { buffer: lookup(buffers, c.buffer, "buffer").buffer, offset: c.offset };
          break;
        case "setIndexBuffer": index = { ...c, buffer: lookup(buffers, c.buffer, "buffer").buffer }; break;
        case "setUniformBuffer": uniforms.set(c.binding, c); break;
        case "setStorageBuffer": throw new Error(NO_STORAGE);
        case "setTexture": boundTextures.set(c.binding, lookup(textures, c.texture, "texture")); break;
        case "setViewport":
          gl.viewport(...glRect(c.x, c.y, c.width, c.height, height));
          gl.depthRange(c.minDepth, c.maxDepth);
          break;
        case "setScissor":
          gl.enable(gl.SCISSOR_TEST);
          gl.scissor(...glRect(c.x, c.y, c.width, c.height, height));
          break;
        case "draw":
          bindResources(0, c.firstInstance);
          gl.drawArraysInstanced(pipeline.mode, c.firstVertex, c.vertexCount, c.instanceCount);
          break;
        case "drawIndexed": {
          bindResources(c.baseVertex, c.firstInstance);
          // Bound per draw: buffer writes rebind ELEMENT_ARRAY_BUFFER, which is VAO state.
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index.buffer);
          const indexSize = index.format === "uint16" ? 2 : 4;
          gl.drawElementsInstanced(pipeline.mode, c.indexCount, gl[GL_NAMES.indexFormat[index.format]],
            index.offset + c.firstIndex * indexSize, c.instanceCount);
          break;
        }
      }
    }
  }

  return {
    name: "webgl2",
    surfaceFormat: "rgba8unorm-srgb",
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    createBuffer(id, size, usage) {
      if (usage & STORAGE_USAGE) throw new Error(NO_STORAGE);
      const index = usage & INDEX_USAGE;
      if (index && usage & (VERTEX_USAGE | UNIFORM_USAGE)) {
        throw new Error("an index buffer cannot also be a vertex or uniform buffer on WebGL2");
      }
      // A WebGL buffer first bound to ELEMENT_ARRAY_BUFFER can never be bound to another target.
      const bufferTarget = index ? gl.ELEMENT_ARRAY_BUFFER : gl.COPY_WRITE_BUFFER;
      const buffer = gl.createBuffer();
      gl.bindBuffer(bufferTarget, buffer);
      gl.bufferData(bufferTarget, size, gl.DYNAMIC_DRAW);
      buffers.set(id, { buffer, size, target: bufferTarget });
    },
    writeBuffer(id, offset, data) {
      const b = lookup(buffers, id, "buffer");
      gl.bindBuffer(b.target, b.buffer);
      gl.bufferSubData(b.target, offset, data);
    },
    destroyBuffer(id) { gl.deleteBuffer(buffers.get(id)?.buffer ?? null); buffers.delete(id); },
    createTexture(id, d) {
      const { internalFormat, format, type } = glTextureFormat(d.format);
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, d.mipLevelCount, gl[internalFormat], d.width, d.height);
      const sampler = gl.createSampler();
      for (const [param, value] of Object.entries(glSamplerParams(d.sampler, d.mipLevelCount))) {
        gl.samplerParameteri(sampler, gl[param], gl[value]);
      }
      textures.set(id, { texture, sampler, width: d.width, height: d.height, format: gl[format], type: gl[type],
        stencil: d.format === "depth24plus-stencil8" });
    },
    writeTexture(id, data) {
      const t = lookup(textures, id, "texture");
      gl.bindTexture(gl.TEXTURE_2D, t.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, t.width, t.height, t.format, t.type, data);
    },
    destroyTexture(id) {
      const t = textures.get(id);
      if (!t) return;
      gl.deleteTexture(t.texture);
      gl.deleteSampler(t.sampler);
      dropFramebuffers(id);
      textures.delete(id);
    },
    createShader(id, s) {
      const label = s.label ? `shader "${s.label}"` : `shader ${id}`;
      const stages = s.glsl && [compile(gl.VERTEX_SHADER, "vertex", glslStage("VERTEX", s.glsl), label),
        compile(gl.FRAGMENT_SHADER, "fragment", glslStage("FRAGMENT", s.glsl), label)];
      shaders.set(id, { label, stages });
    },
    destroyShader(id) {
      for (const shader of shaders.get(id)?.stages ?? []) gl.deleteShader(shader);
      shaders.delete(id);
    },
    createPipeline(id, d) {
      const shader = lookup(shaders, d.shaderId, "shader");
      if (!shader.stages) throw new Error(`${shader.label} has no GLSL source, which WebGL2 needs`);
      const program = link(shader.stages, shader.label);
      // The linked program lists the built-ins it reads as active attributes, so comments and
      // dead code that mention gl_VertexID or gl_InstanceID do not count.
      const attributes = Array.from({ length: gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) },
        (_, i) => gl.getActiveAttrib(program, i).name);
      const uniformBindings = [];
      for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORM_BLOCKS); i++) {
        const binding = glslBinding(gl.getActiveUniformBlockName(program, i));
        gl.uniformBlockBinding(program, i, binding);
        uniformBindings.push(binding);
      }
      const textureBindings = [];
      gl.useProgram(program);
      for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
        const { name, type } = gl.getActiveUniform(program, i);
        if (!SAMPLER_TYPES.some((samplerType) => gl[samplerType] === type)) continue;
        const binding = glslBinding(name);
        gl.uniform1i(gl.getUniformLocation(program, name), binding); // Texture unit `binding`.
        textureBindings.push(binding);
      }
      const factor = (name) => gl[GL_NAMES.blendFactor[name]];
      const operation = (name) => gl[GL_NAMES.blendOperation[name]];
      const { blend, depthStencil, primitive } = d;
      pipelines.set(id, {
        program, uniformBindings, textureBindings, label: shader.label,
        usesVertexId: attributes.includes("gl_VertexID"),
        usesInstanceId: attributes.includes("gl_InstanceID"),
        blend: blend && {
          factors: [blend.color.srcFactor, blend.color.dstFactor, blend.alpha.srcFactor, blend.alpha.dstFactor].map(factor),
          operations: [blend.color.operation, blend.alpha.operation].map(operation),
        },
        depthTest: depthStencil && { func: gl[GL_NAMES.compare[depthStencil.depthCompare]],
          write: depthStencil.depthWriteEnabled },
        cullFace: primitive.cullMode === "none" ? null : gl[GL_NAMES.cullFace[primitive.cullMode]],
        frontFace: gl[GL_NAMES.frontFace[primitive.frontFace]],
        mode: gl[GL_NAMES.topology[primitive.topology]],
        buffers: d.buffers.map(({ arrayStride, stepMode, attributes }) => ({
          arrayStride, stepMode,
          attributes: attributes.map(({ format, offset, shaderLocation }) => {
            const { size, type, normalized, integer } = GL_NAMES.vertexFormat[format];
            return { location: shaderLocation, offset, size, type: gl[type], normalized, integer };
          }),
        })),
      });
    },
    destroyPipeline(id) { gl.deleteProgram(pipelines.get(id)?.program ?? null); pipelines.delete(id); },
    submitFrame(passes) {
      gl.bindVertexArray(vao);
      for (const pass of passes) runPass(pass);
      if (passes.some(({ color }) => color?.target === SURFACE_ID)) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, surface.width, surface.height);
        for (const cap of [gl.SCISSOR_TEST, gl.BLEND, gl.DEPTH_TEST, gl.CULL_FACE]) gl.disable(cap);
        gl.useProgram(presentProgram);
        gl.bindVertexArray(presentVao);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, surface.texture);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      const error = gl.getError();
      if (error !== gl.NO_ERROR) throw new Error(`WebGL2 error 0x${error.toString(16)} while drawing a frame`);
    },
    // Deletes every GL object instead of losing the context, so a later run can reuse the canvas.
    destroy() {
      gl.useProgram(null); // A bound program outlives deleteProgram.
      gl.canvas.removeEventListener("webglcontextlost", onContextLost);
      for (const { buffer } of buffers.values()) gl.deleteBuffer(buffer);
      for (const { texture, sampler } of textures.values()) {
        gl.deleteTexture(texture);
        gl.deleteSampler(sampler);
      }
      for (const { stages } of shaders.values()) for (const shader of stages ?? []) gl.deleteShader(shader);
      for (const { program } of pipelines.values()) gl.deleteProgram(program);
      for (const { framebuffer } of framebuffers.values()) gl.deleteFramebuffer(framebuffer);
      if (surface) gl.deleteTexture(surface.texture);
      for (const shader of presentStages) gl.deleteShader(shader);
      gl.deleteProgram(presentProgram);
      gl.deleteVertexArray(vao);
      gl.deleteVertexArray(presentVao);
    },
  };
}

/**
 * Loads the Titan app at `wasmUrl` and runs it on `canvas` until it exits or fails.
 * Renders with WebGPU, or with WebGL2 where WebGPU has no adapter (WebGL2 needs GLSL shaders).
 * Resolves to `{ dispose, backend }` once running, where `backend` is "webgpu" or "webgl2";
 * rejects if neither API or the module can start. Later failures, including WebGPU validation
 * errors and WebGL2 context loss, stop the app and call `onError`.
 */
export async function runTitan({ canvas, wasmUrl, onError = console.error }) {
  // "starting" until runTitan resolves, then "running" until exit, failure, or dispose.
  let state = "starting";
  let startupError = null;
  let frameHandle = 0;
  let observer = null;
  let lastError = null;
  let backend = null;
  const listeners = [];
  const shutdown = () => {
    if (state === "stopped") return;
    // Stopped first, so the device or context loss from destroy() is not reported as a failure.
    state = "stopped";
    cancelAnimationFrame(frameHandle);
    observer?.disconnect();
    for (const [target, type, fn] of listeners) target.removeEventListener(type, fn);
    backend?.destroy();
  };
  const fail = (error) => {
    if (state === "starting") {
      startupError ??= error;
    } else if (state === "running") {
      shutdown();
      onError(error);
    }
  };

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

  const imports = {
    surface_format: () => TEXTURE_FORMATS.indexOf(backend.surfaceFormat),
    create_buffer: fallible((id, size, usage, labelPtr, labelLen) => {
      backend.createBuffer(id, size, usage, textDecoder.decode(bytes(labelPtr, labelLen)) || undefined);
    }),
    write_buffer: fallible((id, offset, ptr, len) => backend.writeBuffer(id, offset, bytes(ptr, len))),
    destroy_buffer: (id) => backend.destroyBuffer(id),
    create_texture: fallible((id, ptr, len) => backend.createTexture(id, decodeTextureDesc(bytes(ptr, len)))),
    write_texture: fallible((id, ptr, len) => backend.writeTexture(id, bytes(ptr, len))),
    destroy_texture: (id) => backend.destroyTexture(id),
    create_shader: fallible((id, ptr, len) => backend.createShader(id, decodeShaderSource(bytes(ptr, len)))),
    destroy_shader: (id) => backend.destroyShader(id),
    create_pipeline: fallible((id, ptr, len) => {
      const d = decodePipelineDesc(bytes(ptr, len));
      for (const { srcFactor, dstFactor, operation } of d.blend ? [d.blend.color, d.blend.alpha] : []) {
        if ((operation === "min" || operation === "max") && (srcFactor !== "one" || dstFactor !== "one")) {
          throw new Error(`${operation} blending needs src and dst factors of one`);
        }
      }
      backend.createPipeline(id, d);
    }),
    destroy_pipeline: (id) => backend.destroyPipeline(id),
    submit_frame: fallible((ptr, len) => backend.submitFrame(decodeFrame(bytes(ptr, len)))),
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
  let lastScale = 0;
  const resizeCanvas = (width, height, scale) => {
    const maxSize = backend.maxTextureSize;
    [width, height] = [Math.min(width, maxSize), Math.min(height, maxSize)];
    if (width === canvas.width && height === canvas.height && scale === lastScale) return false;
    [canvas.width, canvas.height, lastScale] = [width, height, scale];
    return true;
  };

  let app;
  try {
    // Before any getContext call: a canvas with a "webgpu" context can no longer give a WebGL2 one.
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      backend = await createWebGPUBackend(canvas, adapter, fail);
    } else {
      const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false });
      if (!gl) {
        throw new Error(`This browser supports neither WebGPU${navigator.gpu ? " (no GPU adapter found)" : ""} ` +
          "nor WebGL2. Try a recent Chrome, Edge, Firefox, or Safari, with hardware acceleration enabled.");
      }
      backend = createWebGL2Backend(gl, fail);
    }
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`Could not load ${wasmUrl}: HTTP ${response.status} ${response.statusText}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { titan: unsignedImports });
    // Device or context loss while awaiting above aborts startup; nothing awaits below.
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

  return { dispose: shutdown, backend: backend.name };
}
