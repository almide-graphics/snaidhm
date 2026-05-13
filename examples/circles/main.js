// snaidhm Phase 0.7 — 1000 circles via compute SDF rasterizer
//
// Demonstrates:
// - Tiled compute rasterization (workgroup = 16×16 tile)
// - SDF evaluation with smooth antialiasing
// - Storage buffer as framebuffer (compute → render)
// - Large-scale parallel rendering (512×512 = 262K threads)

const WIDTH = 512;
const HEIGHT = 512;
const CIRCLE_COUNT = 1000;
const CIRCLE_STRIDE = 8 * 4; // 8 floats × 4 bytes = 32 bytes per circle

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
  context.configure({ device, format });

  const shaderSource = await fetch("circles.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Generate circle data
  const circleData = new Float32Array(CIRCLE_COUNT * 8);
  for (let i = 0; i < CIRCLE_COUNT; i++) {
    const o = i * 8;
    circleData[o + 0] = Math.random() * 2 - 1;     // pos.x
    circleData[o + 1] = Math.random() * 2 - 1;     // pos.y
    circleData[o + 2] = 0.01 + Math.random() * 0.06; // radius
    circleData[o + 3] = 0.3 + Math.random() * 0.7;  // color_r
    circleData[o + 4] = 0.3 + Math.random() * 0.7;  // color_g
    circleData[o + 5] = 0.3 + Math.random() * 0.7;  // color_b
    circleData[o + 6] = 0;                           // _pad0
    circleData[o + 7] = 0;                           // _pad1
  }

  // Buffers
  const circleBuffer = device.createBuffer({
    size: circleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(circleBuffer, 0, circleData);

  const pixelBuffer = device.createBuffer({
    size: WIDTH * HEIGHT * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const paramsData = new Uint32Array([WIDTH, HEIGHT, CIRCLE_COUNT, 0]);
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

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
    compute: { module: shaderModule, entryPoint: "rasterize" },
  });

  const computeBG = device.createBindGroup({
    layout: computeBGL,
    entries: [
      { binding: 0, resource: { buffer: circleBuffer } },
      { binding: 1, resource: { buffer: pixelBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  // ── Render pipeline (fullscreen quad) ──

  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: shaderModule, entryPoint: "vs_fullscreen" },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_fullscreen",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const renderBG = device.createBindGroup({
    layout: renderBGL,
    entries: [
      { binding: 0, resource: { buffer: pixelBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  // ── Animate: move circles ──

  let t = 0;
  function frame() {
    t += 0.005;

    // Animate circles
    for (let i = 0; i < CIRCLE_COUNT; i++) {
      const o = i * 8;
      const speed = 0.2 + (i % 7) * 0.05;
      const phase = i * 0.1;
      circleData[o + 0] += Math.sin(t * speed + phase) * 0.002;
      circleData[o + 1] += Math.cos(t * speed * 0.7 + phase) * 0.002;
      // Wrap around
      if (circleData[o + 0] > 1.2) circleData[o + 0] = -1.2;
      if (circleData[o + 0] < -1.2) circleData[o + 0] = 1.2;
      if (circleData[o + 1] > 1.2) circleData[o + 1] = -1.2;
      if (circleData[o + 1] < -1.2) circleData[o + 1] = 1.2;
    }
    device.queue.writeBuffer(circleBuffer, 0, circleData);

    const encoder = device.createCommandEncoder();

    // Compute pass: rasterize circles to pixel buffer
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBG);
    computePass.dispatchWorkgroups(
      Math.ceil(WIDTH / 16),
      Math.ceil(HEIGHT / 16),
    );
    computePass.end();

    // Render pass: display pixel buffer
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
    renderPass.draw(6); // fullscreen quad
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
