// snaidhm WebGPU runtime — WASM import implementations for gpu.almd
//
// Almide Int = i64 = BigInt in JS. B() wraps returns, N() unwraps args.

// Fonts are read by Almide (src/sfnt.almd) from a file this page hands over
// through WASI (wasi.js, snaidhm's host/wasi.js).

import { createWasi } from "./wasi.js";

const handles = [null];
function h(obj) { handles.push(obj); return handles.length - 1; }
function g(id) { return handles[Number(id)]; }
const B = (n) => BigInt(n);
const N = (b) => Number(b);

let _device, _context, _format, _wasmMemory;

// Streaming data builder
let _dataChunks = [];
let _dataIsF32 = [];

// Binding entries: { kind: 'buffer'|'texture'|'sampler', obj }
let _bindingEntries = [];

// Shaders loaded from files at init
let SHADERS = [];

// Text resources (populated during init)
let _textResources = null;

export function createImports(canvas) {
  return { gpu: {
    get_preferred_format: () => B(h(_format)),

    configure_canvas(deviceId, _fmtId) {
      _context = canvas.getContext("webgpu");
      _context.configure({ device: g(deviceId), format: _format, alphaMode: "premultiplied" });
      return B(h(_context));
    },

    create_shader(deviceId, shaderId, _) {
      const code = SHADERS[N(shaderId)] || SHADERS[0];
      return B(h(g(deviceId).createShaderModule({ code })));
    },

    create_buffer(deviceId, size, usage) {
      const sz = N(size), us = N(usage);
      const buf = g(deviceId).createBuffer({ size: sz, usage: us });
      return B(h(buf));
    },

    write_buffer(deviceId, bufferId, dataPtr, dataLen) {
      g(deviceId).queue.writeBuffer(g(bufferId), 0,
        new Uint8Array(_wasmMemory.buffer, N(dataPtr), N(dataLen)));
    },

    create_compute_pipeline(deviceId, shaderId, entryId) {
      return B(h(g(deviceId).createComputePipeline({
        layout: "auto",
        compute: { module: g(shaderId), entryPoint: "fine" },
      })));
    },

    create_render_pipeline(deviceId, shaderId, _vp, _vl, _fp, _fl, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_fullscreen" },
        fragment: { module: g(shaderId), entryPoint: "fs_fullscreen", targets: [{ format: _format }] },
        primitive: { topology: "triangle-list" },
      })));
    },

    // Text render pipeline: vertex buffers + alpha blending
    create_text_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: {
          module: g(shaderId), entryPoint: "vs_main",
          buffers: [{
            arrayStride: 32,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" },   // pos
              { shaderLocation: 1, offset: 8, format: "float32x2" },   // uv
              { shaderLocation: 2, offset: 16, format: "float32x4" },  // color
            ],
          }],
        },
        fragment: {
          module: g(shaderId), entryPoint: "fs_main",
          targets: [{ format: _format, blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          }}],
        },
        primitive: { topology: "triangle-list" },
      })));
    },

    // Image render pipeline: pos(2) + uv(2) = 16 bytes, alpha blend, texture-only
    create_image_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: {
          module: g(shaderId), entryPoint: "vs_main",
          buffers: [{
            arrayStride: 16,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" },
              { shaderLocation: 1, offset: 8, format: "float32x2" },
            ],
          }],
        },
        fragment: {
          module: g(shaderId), entryPoint: "fs_main",
          targets: [{ format: _format, blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          }}],
        },
        primitive: { topology: "triangle-list" },
      })));
    },

    // Explicit bind group API (supports buffer + texture + sampler)
    begin_bindings() { _bindingEntries = []; },

    add_buffer_binding(bufferId) {
      _bindingEntries.push({ kind: "buffer", obj: g(bufferId) });
    },

    add_texture_binding(textureId) {
      _bindingEntries.push({ kind: "texture", obj: g(textureId) });
    },

    add_sampler_binding(samplerId) {
      _bindingEntries.push({ kind: "sampler", obj: g(samplerId) });
    },

    create_bound_group(deviceId, pipelineId, groupIdx) {
      const layout = g(pipelineId).getBindGroupLayout(N(groupIdx));
      const entries = _bindingEntries.map((e, i) => {
        if (e.kind === "buffer") return { binding: i, resource: { buffer: e.obj } };
        if (e.kind === "texture") return { binding: i, resource: e.obj.createView() };
        if (e.kind === "sampler") return { binding: i, resource: e.obj };
      });
      _bindingEntries = [];
      return B(h(g(deviceId).createBindGroup({ layout, entries })));
    },

    set_bind_group(passId, index, bgId) { g(passId).setBindGroup(N(index), g(bgId)); },

    begin_encoder: (deviceId) => B(h(g(deviceId).createCommandEncoder())),

    begin_compute_pass: (encoderId) => B(h(g(encoderId).beginComputePass())),

    dispatch_workgroups(passId, x, y, z) { g(passId).dispatchWorkgroups(N(x), N(y), N(z)); },

    begin_render_pass(encoderId, r, g_, b, a) {
      return B(h(g(encoderId).beginRenderPass({
        colorAttachments: [{
          view: _context.getCurrentTexture().createView(),
          clearValue: { r, g: g_, b, a },
          loadOp: "clear", storeOp: "store",
        }],
      })));
    },

    set_pipeline(passId, pipelineId) { g(passId).setPipeline(g(pipelineId)); },
    draw(passId, n) { g(passId).draw(N(n)); },

    set_vertex_buffer(passId, slot, bufferId) { g(passId).setVertexBuffer(N(slot), g(bufferId)); },
    set_index_buffer(passId, bufferId) { g(passId).setIndexBuffer(g(bufferId), "uint32"); },
    draw_indexed(passId, count) { g(passId).drawIndexed(N(count)); },

    end_pass(passId) { g(passId).end(); },
    finish_and_submit(deviceId, encoderId) {
      g(deviceId).queue.submit([g(encoderId).finish()]);
    },

    // Streaming data builder
    begin_data() { _dataChunks = []; _dataIsF32 = []; },
    push_f32(v) { _dataChunks.push(v); _dataIsF32.push(true); },
    push_u32(v) { _dataChunks.push(N(v)); _dataIsF32.push(false); },
    flush_to_buffer(deviceId, bufferId) {
      const buf = new ArrayBuffer(_dataChunks.length * 4);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      for (let i = 0; i < _dataChunks.length; i++) {
        if (_dataIsF32[i]) f32[i] = _dataChunks[i];
        else u32[i] = _dataChunks[i];
      }
      g(deviceId).queue.writeBuffer(g(bufferId), 0, new Uint8Array(buf));
      _dataChunks = []; _dataIsF32 = [];
    },

    log_int(v) { console.log("[gpu]", N(v)); },
    log_str(ptr, len) {
      console.log("[gpu]", new TextDecoder().decode(
        new Uint8Array(_wasmMemory.buffer, N(ptr), N(len))));
    },
  }};
}


// ── Image textures (procedural card thumbnails) ──

function generateCardImages() {
  const S = 64;
  const atlas = new Uint8Array(S * 3 * S * 4);
  for (let img = 0; img < 3; img++) {
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = (y * S * 3 + img * S + x) * 4;
        const u = x / S, v = y / S;
        let r, g, b;
        if (img === 0) {
          const dx = u - 0.5, dy = v - 0.35;
          const sun = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) * 4.5);
          r = 0.92 - v * 0.35 + sun * 0.4;
          g = 0.35 + sun * 0.55 - v * 0.15;
          b = 0.25 + v * 0.45;
        } else if (img === 1) {
          const wave = Math.sin(u * 12 + v * 4) * 0.04;
          r = 0.1 + v * 0.15;
          g = 0.35 + u * 0.2 + wave;
          b = 0.55 + v * 0.25 + wave;
        } else {
          const band = Math.sin(v * 8 + u * 3) * 0.1;
          r = 0.25 + v * 0.35 + band;
          g = 0.15 + (1 - v) * 0.4 + band * 0.5;
          b = 0.4 + v * 0.25;
        }
        atlas[i]     = Math.min(255, Math.max(0, r * 255));
        atlas[i + 1] = Math.min(255, Math.max(0, g * 255));
        atlas[i + 2] = Math.min(255, Math.max(0, b * 255));
        atlas[i + 3] = 255;
      }
    }
  }
  return { data: atlas, width: S * 3, height: S };
}

function buildImageQuads() {
  const cards = [
    { x: -0.81, y: -0.81, w: 0.40, h: 0.30, uOff: 0 },
    { x: -0.26, y: -0.81, w: 0.40, h: 0.30, uOff: 1 / 3 },
    { x:  0.29, y: -0.81, w: 0.48, h: 0.30, uOff: 2 / 3 },
  ];
  const verts = [], idxs = [];
  for (const c of cards) {
    const vi = verts.length / 4;
    const u0 = c.uOff, u1 = c.uOff + 1 / 3;
    verts.push(c.x, c.y, u0, 1,           c.x + c.w, c.y, u1, 1,
               c.x + c.w, c.y + c.h, u1, 0, c.x, c.y + c.h, u0, 0);
    idxs.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
  }
  return { vertices: new Float32Array(verts), indices: new Uint32Array(idxs) };
}

function initImageResources(device) {
  const cardImages = generateCardImages();
  const imageQuads = buildImageQuads();

  const imgTexture = device.createTexture({
    size: [cardImages.width, cardImages.height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const imgBytesPerRow = Math.ceil(cardImages.width * 4 / 256) * 256;
  const imgAligned = new Uint8Array(imgBytesPerRow * cardImages.height);
  for (let row = 0; row < cardImages.height; row++) {
    imgAligned.set(
      cardImages.data.subarray(row * cardImages.width * 4, (row + 1) * cardImages.width * 4),
      row * imgBytesPerRow,
    );
  }
  device.queue.writeTexture({ texture: imgTexture }, imgAligned, { bytesPerRow: imgBytesPerRow }, [cardImages.width, cardImages.height]);

  const imgSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const imgVertBuf = device.createBuffer({
    size: imageQuads.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(imgVertBuf, 0, imageQuads.vertices);

  const imgIdxBuf = device.createBuffer({
    size: imageQuads.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(imgIdxBuf, 0, imageQuads.indices);

  console.log(`Image atlas: ${cardImages.width}x${cardImages.height} | Image quads: ${imageQuads.indices.length / 6}`);

  return {
    texture: h(imgTexture),
    sampler: h(imgSampler),
    vertexBuffer: h(imgVertBuf),
    indexBuffer: h(imgIdxBuf),
    indexCount: imageQuads.indices.length,
  };
}


export async function init(wasmUrl, canvas) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice();
  _format = navigator.gpu.getPreferredCanvasFormat();

  // Load shaders (no more text.wgsl needed for SDF)
  const [rasterCode, imageCode] = await Promise.all([
    fetch("./raster.wgsl").then(r => r.text()),
    fetch("./image.wgsl").then(r => r.text()),
  ]);
  SHADERS = [rasterCode, rasterCode, rasterCode, imageCode];

  // The font `render` reads as /font.ttf.
  const fontBytes = await fetch("font.ttf").then(r => r.arrayBuffer());

  const _imageResources = initImageResources(_device);

  const wasi = createWasi({ files: { "/font.ttf": fontBytes } });
  const imports = { wasi_snapshot_preview1: wasi.imports, ...createImports(canvas) };
  const { instance } = await WebAssembly.instantiate(
    await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;
  wasi.setMemory(_wasmMemory);

  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}
  if (instance.exports.render) {
    try {
      const im = _imageResources;
      instance.exports.render(
        B(h(_device)),
        B(im.vertexBuffer),
        B(im.indexBuffer),
        B(im.indexCount),
        B(im.texture),
        B(im.sampler),
      );
      console.log("snaidhm: paths-are-paths text rendering complete");
    } catch (e) {
      console.error("render error:", e);
    }
  }
}

// Text lines — must match main.almd text_lines (same positions, sizes, alignment)
// Semantic roles for accessibility: heading, label, text (default)
const TEXT_LINES = [
  { text: "snaidhm",                size: 40, x: 0.0,   y: 0.64,  align: "center", color: [1,1,1,1], role: "heading", level: 1 },
  { text: "Almide > WASM > WebGPU", size: 13, x: 0.0,   y: 0.57,  align: "center", color: [0.8,0.85,1.0,0.9], role: "heading", level: 2 },
  { text: "Red",                    size: 13, x: -0.5,  y: -0.02, align: "center", color: [1,1,1,1], role: "label", labelFor: "red-circle" },
  { text: "Green",                  size: 13, x: 0.0,   y: -0.02, align: "center", color: [1,1,1,1], role: "label", labelFor: "green-circle" },
  { text: "Blue",                   size: 13, x: 0.5,   y: -0.02, align: "center", color: [1,1,1,1], role: "label", labelFor: "blue-circle" },
  { text: "Card A",                 size: 11, x: -0.8,  y: -0.4,  align: "left",   color: [0.3,0.3,0.4,1], role: "heading", level: 3 },
  { text: "Card B",                 size: 11, x: -0.25, y: -0.4,  align: "left",   color: [0.3,0.3,0.4,1], role: "heading", level: 3 },
  { text: "Card C",                 size: 11, x: 0.3,   y: -0.4,  align: "left",   color: [0.3,0.3,0.4,1], role: "heading", level: 3 },
  { text: "Shapes + Text",          size: 16, x: 0.0,   y: -0.68, align: "center", color: [0.5,0.5,0.6,1] },
  { text: "all from Almide",        size: 16, x: 0.0,   y: -0.78, align: "center", color: [0.5,0.5,0.6,1] },
];

export function createTextOverlay(overlay, canvas) {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;

  const ndcToX = (nx) => (nx + 1) * 0.5 * W;
  const ndcToY = (ny) => (1 - ny) * 0.5 * H;

  // Canvas itself is a figure with an accessible label
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "snaidhm GPU renderer demo: shapes, gradients, and text rendered via Almide to WASM to WebGPU");

  // Overlay acts as a document structure for screen readers
  overlay.setAttribute("role", "document");
  overlay.setAttribute("aria-label", "snaidhm UI text content");

  for (const line of TEXT_LINES) {
    // Choose semantic element based on role
    let el;
    if (line.role === "heading" && line.level) {
      const tag = `h${Math.min(line.level, 6)}`;
      el = document.createElement(tag);
      // Reset heading styles — visual rendering is on canvas
      el.style.margin = "0";
      el.style.fontWeight = "normal";
    } else {
      el = document.createElement("span");
    }

    el.textContent = line.text;
    el.style.fontSize = line.size + "px";
    el.style.fontFamily = "sans-serif";
    el.style.color = "transparent";

    if (line.role === "label") {
      el.setAttribute("role", "note");
      el.setAttribute("aria-label", `${line.text} — ${line.labelFor} label`);
    }

    overlay.appendChild(el);

    const elW = el.offsetWidth;
    const baselinePx = ndcToY(line.y);
    const topPx = baselinePx - line.size * 0.75;

    let leftPx;
    if (line.align === "center") {
      leftPx = ndcToX(line.x) - elW / 2;
    } else {
      leftPx = ndcToX(line.x);
    }

    el.style.left = leftPx + "px";
    el.style.top = topPx + "px";
  }
}

// ── Text input (hidden textarea + IME) ──
// Card C area in NDC: (0.25, -0.85, 0.6, 0.55) with label at y=-0.4
// Input field below the Card C label

export function setupTextInput(overlay, textarea, canvas) {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const ndcToX = (nx) => (nx + 1) * 0.5 * W;
  const ndcToY = (ny) => (1 - ny) * 0.5 * H;

  // Position input field inside Card C, below the "Card C" label
  const inputArea = document.createElement("div");
  inputArea.className = "input-area";
  inputArea.setAttribute("role", "textbox");
  inputArea.setAttribute("aria-label", "Text input field");
  inputArea.style.left = ndcToX(0.29) + "px";
  inputArea.style.top = ndcToY(-0.48) + "px";
  inputArea.style.width = (ndcToX(0.81) - ndcToX(0.29)) + "px";
  inputArea.style.fontSize = "11px";
  inputArea.style.fontFamily = "sans-serif";

  const display = document.createElement("div");
  display.className = "input-display";
  inputArea.appendChild(display);

  overlay.appendChild(inputArea);

  let focused = false;

  function renderPlaceholder() {
    display.innerHTML = "";
    if (!focused && textarea.value === "") {
      const ph = document.createElement("span");
      ph.className = "placeholder";
      ph.textContent = "Type here...";
      display.appendChild(ph);
    } else {
      display.innerHTML = "";
    }
  }

  // Position textarea directly over the input area
  textarea.style.left = inputArea.style.left;
  textarea.style.top = inputArea.style.top;
  textarea.style.width = inputArea.style.width;
  textarea.style.height = "1.4em";

  inputArea.addEventListener("click", () => {
    textarea.style.pointerEvents = "auto";
    textarea.focus();
  });

  textarea.addEventListener("focus", () => {
    focused = true;
    textarea.style.color = "#333";
    textarea.style.caretColor = "#333";
    inputArea.style.outline = "1.5px solid rgba(66, 133, 244, 0.6)";
    inputArea.style.outlineOffset = "2px";
    inputArea.style.borderRadius = "2px";
    renderPlaceholder();
  });

  textarea.addEventListener("blur", () => {
    focused = false;
    textarea.style.color = "transparent";
    textarea.style.caretColor = "transparent";
    textarea.style.pointerEvents = "none";
    inputArea.style.outline = "none";
    // Show committed text in display when not focused
    display.innerHTML = "";
    if (textarea.value) {
      display.appendChild(document.createTextNode(textarea.value));
    } else {
      renderPlaceholder();
    }
  });

  textarea.addEventListener("input", () => {
    renderPlaceholder();
  });

  renderPlaceholder();
}
