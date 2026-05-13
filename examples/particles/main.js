// snaidhm Phase 0.5 — 10K Particles (compute shader + double buffer)
//
// Demonstrates:
// - Compute shader dispatch (@compute @workgroup_size)
// - Storage buffers (read + read_write)
// - Double buffering (ping-pong between two buffers)
// - Uniform buffer (per-frame dt)
// - Point rendering from storage buffer

const PARTICLE_COUNT = 10000;
const PARTICLE_SIZE = 6 * 4; // pos(2) + vel(2) + life(1) + pad(1) = 6 floats = 24 bytes
const WORKGROUP_SIZE = 256;

async function main() {
  if (!navigator.gpu) {
    document.body.textContent = "WebGPU not supported";
    return;
  }

  const canvas = document.getElementById("canvas");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  // Load shaders
  const shaderSource = await fetch("particles.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Initialize particle data
  const initialData = new Float32Array(PARTICLE_COUNT * 6);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const o = i * 6;
    initialData[o + 0] = Math.random() * 2 - 1;  // pos.x
    initialData[o + 1] = Math.random() * 2 - 1;  // pos.y
    initialData[o + 2] = (Math.random() - 0.5) * 0.3;  // vel.x
    initialData[o + 3] = Math.random() * 0.5;           // vel.y
    initialData[o + 4] = Math.random() * 5;             // life
    initialData[o + 5] = 0;                             // pad
  }

  // Double-buffered storage: A and B
  const bufferSize = PARTICLE_COUNT * PARTICLE_SIZE;
  const bufferA = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
  });
  const bufferB = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
  });
  device.queue.writeBuffer(bufferA, 0, initialData);
  device.queue.writeBuffer(bufferB, 0, initialData);

  // Uniform buffer: SimParams { dt, count, seed, _pad }
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ── Compute pipeline ──

  const computeBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
    compute: { module: shaderModule, entryPoint: "simulate" },
  });

  // Two bind groups for ping-pong: A→B and B→A
  const computeBG_AtoB = device.createBindGroup({
    layout: computeBGL,
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });
  const computeBG_BtoA = device.createBindGroup({
    layout: computeBGL,
    entries: [
      { binding: 0, resource: { buffer: bufferB } },
      { binding: 1, resource: { buffer: bufferA } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  // ── Render pipeline ──

  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: shaderModule, entryPoint: "vs_render" },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_render",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "point-list" },
  });

  const renderBG_A = device.createBindGroup({
    layout: renderBGL,
    entries: [{ binding: 0, resource: { buffer: bufferA } }],
  });
  const renderBG_B = device.createBindGroup({
    layout: renderBGL,
    entries: [{ binding: 0, resource: { buffer: bufferB } }],
  });

  // ── Frame loop ──

  let frame_idx = 0;
  let lastTime = performance.now();

  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05); // cap at 50ms
    lastTime = now;

    // Upload sim params
    const params = new Float32Array([dt, 0, now * 0.001, 0]);
    new Uint32Array(params.buffer)[1] = PARTICLE_COUNT;
    device.queue.writeBuffer(uniformBuffer, 0, params);

    // Ping-pong: even frames A→B, odd frames B→A
    const isEven = (frame_idx & 1) === 0;
    const computeBG = isEven ? computeBG_AtoB : computeBG_BtoA;
    const renderBG = isEven ? renderBG_B : renderBG_A; // render from destination
    frame_idx++;

    const encoder = device.createCommandEncoder();

    // Compute pass
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBG);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
    computePass.end();

    // Render pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBG);
    renderPass.draw(PARTICLE_COUNT);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
