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

const COMPUTE_SHADER = `
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<uniform> params: vec2<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params.x;
  let h = params.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let u = f32(gid.x) / f32(w);
  let v = f32(gid.y) / f32(h);

  // Gradient with circle
  let dx = u - 0.5;
  let dy = v - 0.5;
  let d = sqrt(dx*dx + dy*dy);
  let ring = smoothstep(0.28, 0.3, d) - smoothstep(0.3, 0.32, d);

  let r = u32(clamp((0.15 + u * 0.4 + ring * 0.6) * 255.0, 0.0, 255.0));
  let g = u32(clamp((0.1 + v * 0.3 + ring * 0.8) * 255.0, 0.0, 255.0));
  let b = u32(clamp((0.3 + (1.0-v) * 0.5 + ring * 0.5) * 255.0, 0.0, 255.0));
  pixels[gid.y * w + gid.x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
`;

const QUAD_SHADER = `
struct Params { width: u32, height: u32 }

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>, 6>(
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0),
  );
  var uv = array<vec2<f32>, 6>(
    vec2(0.0,1.0), vec2(1.0,1.0), vec2(0.0,0.0),
    vec2(0.0,0.0), vec2(1.0,1.0), vec2(1.0,0.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(p[i], 0.0, 1.0);
  out.uv = uv[i];
  return out;
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let px = u32(uv.x * f32(params.width));
  let py = u32(uv.y * f32(params.height));
  let c = pixels[py * params.width + px];
  return vec4<f32>(
    f32(c & 0xFFu) / 255.0,
    f32((c >> 8u) & 0xFFu) / 255.0,
    f32((c >> 16u) & 0xFFu) / 255.0,
    1.0
  );
}
`;

const SHADERS = [COMPUTE_SHADER, QUAD_SHADER];

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
      const buf = g(deviceId).createBuffer({ size: N(size), usage: N(usage) });
      const id = h(buf);
      _lastBuffers.push(id);
      // Auto-upload params for uniform buffers (width, height)
      if (N(usage) & 0x0040) { // UNIFORM
        g(deviceId).queue.writeBuffer(buf, 0, new Uint32Array([512, 512]));
      }
      return B(id);
    },

    write_buffer(deviceId, bufferId, dataPtr, dataLen) {
      g(deviceId).queue.writeBuffer(g(bufferId), 0,
        new Uint8Array(_wasmMemory.buffer, N(dataPtr), N(dataLen)));
    },

    create_compute_pipeline(deviceId, shaderId, entryId) {
      return B(h(g(deviceId).createComputePipeline({
        layout: "auto",
        compute: { module: g(shaderId), entryPoint: "main" },
      })));
    },

    create_render_pipeline(deviceId, shaderId, _vp, _vl, _fp, _fl, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs" },
        fragment: { module: g(shaderId), entryPoint: "fs", targets: [{ format: _format }] },
        primitive: { topology: "triangle-list" },
      })));
    },

    create_bind_group(deviceId, pipelineId, groupIdx, _ptr, count) {
      const pipeline = g(pipelineId);
      const layout = pipeline.getBindGroupLayout(N(groupIdx));
      const entries = [];
      // Auto-bind: use the last N created buffers
      const bufs = _lastBuffers.slice(-N(count));
      for (let i = 0; i < bufs.length; i++) {
        entries.push({ binding: i, resource: { buffer: g(bufs[i]) } });
      }
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
    instance.exports.render(B(h(_device)));
    console.log("snaidhm: Almide compute + render complete");
  }
}
