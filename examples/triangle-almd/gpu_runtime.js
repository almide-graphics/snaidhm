// snaidhm WebGPU runtime — WASM import implementations for gpu.almd
//
// Almide Int = i64 = BigInt in JS. B() wraps returns, N() unwraps args.

const handles = [null];
function h(obj) { handles.push(obj); return handles.length - 1; }
function g(id) { return handles[Number(id)]; }
const B = (n) => BigInt(n);
const N = (b) => Number(b);

let _device, _context, _format, _wasmMemory;

// Buffers tracked for auto-binding
let _lastBuffers = [];

// Streaming data builder
let _dataChunks = [];
let _dataIsF32 = [];
let _bindingEntries = [];

// Shaders loaded from files at init
let SHADERS = [];

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
      console.log(`create_buffer: size=${sz} usage=0x${us.toString(16)}`);
      const buf = g(deviceId).createBuffer({ size: sz, usage: us });
      const id = h(buf);
      _lastBuffers.push(id);
      return B(id);
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

    // Explicit bind group API
    begin_bindings() { _bindingEntries = []; },
    add_buffer_binding(bufferId) { _bindingEntries.push(g(bufferId)); },
    create_bound_group(deviceId, pipelineId, groupIdx) {
      const layout = g(pipelineId).getBindGroupLayout(N(groupIdx));
      const entries = _bindingEntries.map((buf, i) => ({ binding: i, resource: { buffer: buf } }));
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

export async function init(wasmUrl, canvas) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice();
  _format = navigator.gpu.getPreferredCanvasFormat();

  // Load shader (single file has both compute + render entry points)
  const shaderCode = await fetch("./raster.wgsl").then(r => r.text());
  SHADERS = [shaderCode, shaderCode];

  const wasi = new Proxy({}, { get(_, n) {
    if (n === "proc_exit") return () => {};
    if (n === "fd_prestat_get") return () => 8;
    return () => 0;
  }});

  const imports = { wasi_snapshot_preview1: wasi, ...createImports(canvas) };
  const { instance } = await WebAssembly.instantiate(
    await fetch(wasmUrl).then(r => r.arrayBuffer()), imports);
  _wasmMemory = instance.exports.memory;

  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}
  if (instance.exports.render) {
    try {
      instance.exports.render(B(h(_device)));
      console.log("snaidhm: Almide path rasterizer complete");
    } catch (e) {
      console.error("render error:", e);
    }
  }
}
