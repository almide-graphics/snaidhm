// snaidhm WebGPU runtime — provides WASM imports for gpu.almd @extern declarations
//
// Handle table: GPU objects are stored here, referenced by integer handles from WASM.
// Almide Int = i64 = BigInt in JS. All imports/exports need BigInt<->Number conversion.

const handles = [null]; // index 0 = invalid
function h(obj) { handles.push(obj); return handles.length - 1; }
function g(id) { return handles[Number(id)]; }
const B = (n) => BigInt(n);  // Number → BigInt (return to WASM)
const N = (b) => Number(b);  // BigInt → Number (from WASM)

let _device = null;
let _context = null;
let _format = null;

const TRIANGLE_SHADER = `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>( 0.0,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
  );
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.3, 0.7, 1.0, 1.0);
}
`;

export function createImports(canvas) {
  return {
    gpu: {
      get_preferred_format: () => B(h(_format)),

      configure_canvas(deviceId, _formatId) {
        _context = canvas.getContext("webgpu");
        _context.configure({ device: g(deviceId), format: _format, alphaMode: "premultiplied" });
        return B(h(_context));
      },

      create_shader(deviceId, _codePtr, _codeLen) {
        return B(h(g(deviceId).createShaderModule({ code: TRIANGLE_SHADER })));
      },

      create_render_pipeline(deviceId, shaderId, _vsPtr, _vsLen, _fsPtr, _fsLen, _formatId) {
        return B(h(g(deviceId).createRenderPipeline({
          layout: "auto",
          vertex: { module: g(shaderId), entryPoint: "vs" },
          fragment: { module: g(shaderId), entryPoint: "fs", targets: [{ format: _format }] },
          primitive: { topology: "triangle-list" },
        })));
      },

      create_buffer(deviceId, size, usage) {
        return B(h(g(deviceId).createBuffer({ size: N(size), usage: N(usage) })));
      },

      write_buffer(deviceId, bufferId, dataPtr, dataLen) {
        const mem = new Uint8Array(_wasmMemory.buffer, N(dataPtr), N(dataLen));
        g(deviceId).queue.writeBuffer(g(bufferId), 0, mem);
      },

      begin_encoder: (deviceId) => B(h(g(deviceId).createCommandEncoder())),

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
      draw(passId, vertexCount) { g(passId).draw(N(vertexCount)); },
      end_pass(passId) { g(passId).end(); },

      finish_and_submit(deviceId, encoderId) {
        g(deviceId).queue.submit([g(encoderId).finish()]);
      },

      log_int(v) { console.log("[gpu]", N(v)); },
      log_str(ptr, len) {
        console.log("[gpu]", new TextDecoder().decode(new Uint8Array(_wasmMemory.buffer, N(ptr), N(len))));
      },
    },
  };
}

let _wasmMemory = null;

export async function init(wasmUrl, canvas) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  _device = await adapter.requestDevice();
  _format = navigator.gpu.getPreferredCanvasFormat();
  const deviceHandle = h(_device);

  const wasi = new Proxy({}, {
    get(_, name) {
      if (name === "proc_exit") return () => {};
      if (name === "fd_prestat_get") return () => 8;
      return () => 0;
    },
  });

  const imports = { wasi_snapshot_preview1: wasi, ...createImports(canvas) };
  const wasmBytes = await fetch(wasmUrl).then(r => r.arrayBuffer());
  const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
  _wasmMemory = instance.exports.memory;

  if (instance.exports._start) try { instance.exports._start(); } catch (_) {}
  if (instance.exports.render) {
    instance.exports.render(B(deviceHandle));
    console.log("snaidhm: Almide -> WASM -> WebGPU render complete");
  }
}
