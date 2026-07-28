// Every host entry point must RUN, not merely exist.
//
// `host-contract.mjs` checks that each declared extern has an implementation.
// It cannot see an implementation that throws the moment it is called — and one
// did: `create_texture(deviceId, w, h)` named a parameter `h`, shadowing the
// handle-registering helper of the same name, so `h(...)` called a number.
// "h is not a function", at runtime, from inside a wasm call, where nothing on
// the Almide side can catch or report it.
//
// This drives every entry point against a permissive fake device. It does not
// check that the WebGPU calls are correct — only that the host code itself is
// well-formed enough to reach them.
//
// Usage: node test/host-runs.mjs

import { readFile } from "node:fs/promises";

// A stand-in for anything the WebGPU API would return: every property is a
// method, every method returns another one of these.
const anything = () => new Proxy(function () {}, {
  get: (_t, k) => (k === "then" ? undefined : anything()),
  apply: () => anything(),
  has: () => true,
});

globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_DST: 4, COPY_SRC: 8 };
globalThis.GPUBufferUsage = { VERTEX: 32, INDEX: 16, UNIFORM: 64, COPY_DST: 8, STORAGE: 128 };
globalThis.createImageBitmap = async () => ({ width: 4, height: 4, close() {} });
globalThis.Blob = globalThis.Blob ?? class { constructor() {} };

const src = await readFile("host/gpu.js", "utf8");
const mod = await import(`data:text/javascript,${encodeURIComponent(src)}`);

const host = mod.createGpuHost(anything());
host.setFormat("bgra8unorm");
host.setMemory({ buffer: new ArrayBuffer(1024) });
host.registerShader("// wgsl");
const dev = host.register(anything());

// Real handles, threaded from the calls that make them — passing 0 would hand
// every entry point the null slot and test nothing but the null path.
const I = host.imports;
const shader = I.create_shader(dev, 0n, 0n);
const buffer = I.create_buffer(dev, 256n, 64n);
const texture = I.create_texture(dev, 8n, 8n);
const sampler = I.create_sampler(dev, 1n, 1n);
const pipeline = I.create_mesh_pipeline(dev, shader, 0n, 1n);
I.begin_bindings();
I.add_buffer_binding(buffer);
const bindGroup = I.create_bound_group(dev, pipeline, 0n);
I.set_depth_size(dev, 8n, 8n);
const encoder = I.begin_encoder(dev);
I.configure_canvas(dev, 0n);
const pass = I.begin_render_pass_3d(encoder, 0, 0, 0, 1, 0n);

const ARGS = {
  configure_canvas: [dev, 0n],
  create_shader: [dev, 0n, 0n],
  create_buffer: [dev, 256n, 64n],
  write_buffer: [dev, buffer, 0n, 16n],
  write_buffer_at: [dev, buffer, 0n, 0n, 16n],
  write_f32_at: [dev, buffer, 0n, 1.0],
  write_u32_at: [dev, buffer, 0n, 1n],
  create_render_pipeline: [dev, shader, 0n, 0n, 0n, 0n, 0n],
  create_compute_pipeline: [dev, shader, 0n],
  create_text_pipeline: [dev, shader, 0n],
  create_image_pipeline: [dev, shader, 0n],
  create_mesh_pipeline: [dev, shader, 0n, 1n],
  create_bound_group: [dev, pipeline, 0n],
  set_bind_group: [pass, 0n, bindGroup],
  begin_encoder: [dev],
  begin_compute_pass: [encoder],
  dispatch_workgroups: [pass, 1n, 1n, 1n],
  begin_render_pass: [encoder, 0, 0, 0, 1],
  begin_render_pass_3d: [encoder, 0, 0, 0, 1, 0n],
  set_depth_size: [dev, 8n, 8n],
  create_texture: [dev, 8n, 8n],
  upload_encoded_image: [dev, texture, 0n, 16n],
  create_sampler: [dev, 1n, 1n],
  set_pipeline: [pass, pipeline],
  set_vertex_buffer: [pass, 0n, buffer],
  set_index_buffer: [pass, buffer],
  set_index_buffer_u16: [pass, buffer],
  draw: [pass, 3n],
  draw_indexed: [pass, 3n],
  draw_indexed_from: [pass, 0n, 3n],
  end_pass: [pass],
  finish_and_submit: [dev, encoder],
  push_f32: [1.0],
  push_u32: [1n],
  flush_to_buffer: [dev, buffer],
  add_buffer_binding: [buffer],
  add_texture_binding: [texture],
  add_sampler_binding: [sampler],
  log_int: [0n],
  log_str: [0n, 0n],
  get_preferred_format: [],
  get_state: [0n],
  set_state: [0n, 0n],
  begin_bindings: [],
  begin_data: [],
};

// `end_pass` and `finish_and_submit` consume their handles, so run them last.
const CONSUMING = new Set(["end_pass", "finish_and_submit"]);

const failures = [];
const names = Object.keys(host.imports);
const ordered = [...names.filter((n) => !CONSUMING.has(n)), ...names.filter((n) => CONSUMING.has(n))];
for (const name of ordered) {
  const args = ARGS[name] ?? [];
  try {
    host.imports[name](...args);
  } catch (e) {
    failures.push(`${name}(${args.length} args) threw: ${e.message}`);
  }
}

console.log(`snaidhm host — drove ${names.length} entry points\n`);
for (const f of failures) console.log(`  FAIL ${f}`);
if (failures.length) {
  console.error(`\nFAILED — ${failures.length} entry point(s) throw when called.`);
  console.error("A host function that throws is invisible to the wasm side: the call");
  console.error("unwinds through it with no way to report or recover.");
  process.exit(1);
}
console.log("passed — every entry point runs");
