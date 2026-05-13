// snaidhm Phase 1 — Path renderer
//
// CPU: cubic bezier → line segments (Wang's formula, sorted by path_id)
// GPU: per-pixel winding number → fill color
//
// Test scene: circles, rounded rects, heart — all cubic bezier paths

const WIDTH = 512;
const HEIGHT = 512;

// ── Bezier math (CPU) ──

function cubicEval(p0, p1, p2, p3, t) {
  const u = 1 - t, uu = u * u, tt = t * t;
  return [
    uu * u * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + tt * t * p3[0],
    uu * u * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + tt * t * p3[1],
  ];
}

function wangSegments(p0, p1, p2, p3, tol) {
  const d1 = [p2[0] - 2 * p1[0] + p0[0], p2[1] - 2 * p1[1] + p0[1]];
  const d2 = [p3[0] - 2 * p2[0] + p1[0], p3[1] - 2 * p2[1] + p1[1]];
  const dd = Math.max(Math.hypot(d1[0], d1[1]), Math.hypot(d2[0], d2[1]));
  return Math.max(1, Math.min(256, Math.ceil(Math.sqrt(3 * dd / (4 * tol)))));
}

// ── Path builder ──

function buildScene() {
  const beziers = [];
  let pathId = 0;

  function addCubic(p0, p1, p2, p3, color) {
    beziers.push({ p0, p1, p2, p3, color, pathId });
  }
  function closePath() { pathId++; }

  function circle(cx, cy, r, color) {
    const k = 0.5522847498 * r;
    addCubic([cx, cy + r], [cx + k, cy + r], [cx + r, cy + k], [cx + r, cy], color);
    addCubic([cx + r, cy], [cx + r, cy - k], [cx + k, cy - r], [cx, cy - r], color);
    addCubic([cx, cy - r], [cx - k, cy - r], [cx - r, cy - k], [cx - r, cy], color);
    addCubic([cx - r, cy], [cx - r, cy + k], [cx - k, cy + r], [cx, cy + r], color);
    closePath();
  }

  function roundRect(x, y, w, h, r, color) {
    const k = 0.5522847498 * r;
    addCubic([x + r, y + h], [x + r, y + h], [x + w - r, y + h], [x + w - r, y + h], color);
    addCubic([x + w - r, y + h], [x + w - r + k, y + h], [x + w, y + h - r + k], [x + w, y + h - r], color);
    addCubic([x + w, y + h - r], [x + w, y + h - r], [x + w, y + r], [x + w, y + r], color);
    addCubic([x + w, y + r], [x + w, y + r - k], [x + w - r + k, y], [x + w - r, y], color);
    addCubic([x + w - r, y], [x + w - r, y], [x + r, y], [x + r, y], color);
    addCubic([x + r, y], [x + r - k, y], [x, y + r - k], [x, y + r], color);
    addCubic([x, y + r], [x, y + r], [x, y + h - r], [x, y + h - r], color);
    addCubic([x, y + h - r], [x, y + h - r + k], [x + r - k, y + h], [x + r, y + h], color);
    closePath();
  }

  function heart(cx, cy, s, color) {
    addCubic([cx, cy - s * 0.4], [cx, cy + s * 0.4], [cx - s, cy + s * 0.8], [cx - s, cy], color);
    addCubic([cx - s, cy], [cx - s, cy - s * 0.8], [cx, cy - s * 0.6], [cx, cy - s * 1.2], color);
    addCubic([cx, cy - s * 1.2], [cx, cy - s * 0.6], [cx + s, cy - s * 0.8], [cx + s, cy], color);
    addCubic([cx + s, cy], [cx + s, cy + s * 0.8], [cx, cy + s * 0.4], [cx, cy - s * 0.4], color);
    closePath();
  }

  // Scene
  circle(0, 0, 0.6, [0.25, 0.35, 0.78, 1.0]);
  circle(-0.3, 0.3, 0.15, [1.0, 0.3, 0.3, 1.0]);
  circle(0.3, 0.3, 0.15, [0.3, 1.0, 0.3, 1.0]);
  circle(0.0, -0.2, 0.15, [1.0, 1.0, 0.3, 1.0]);
  roundRect(-0.8, -0.85, 0.5, 0.3, 0.06, [0.85, 0.45, 0.2, 1.0]);
  roundRect(0.3, -0.85, 0.5, 0.3, 0.08, [0.35, 0.8, 0.55, 1.0]);
  heart(0.0, 0.68, 0.12, [1.0, 0.2, 0.35, 1.0]);

  for (let i = 0; i < 50; i++) {
    const angle = (i / 50) * Math.PI * 2;
    const cx = Math.cos(angle) * 0.85;
    const cy = Math.sin(angle) * 0.85;
    const hue = i / 50;
    circle(cx, cy, 0.03, [
      0.5 + 0.5 * Math.sin(hue * 6.28),
      0.5 + 0.5 * Math.sin(hue * 6.28 + 2.09),
      0.5 + 0.5 * Math.sin(hue * 6.28 + 4.18),
      1.0,
    ]);
  }

  return beziers;
}

// ── Flatten on CPU (sorted by path_id) ──

function flattenBeziers(beziers) {
  const tol = 0.5 / WIDTH; // half-pixel tolerance in NDC
  const segments = [];

  for (const bez of beziers) {
    const n = wangSegments(bez.p0, bez.p1, bez.p2, bez.p3, tol);
    for (let i = 0; i < n; i++) {
      const t0 = i / n;
      const t1 = (i + 1) / n;
      const a = cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, t0);
      const b = cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, t1);
      segments.push({ p0: a, p1: b, color: bez.color, pathId: bez.pathId });
    }
  }

  return segments;
}

// ── WebGPU ──

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

  const shaderSource = await fetch("paths.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Build and flatten paths
  const beziers = buildScene();
  const segments = flattenBeziers(beziers);
  const segCount = segments.length;

  // Pack segments: p0(2) p1(2) color(4) path_id(1) pad(3) = 12 floats per segment
  const segData = new Float32Array(segCount * 12);
  for (let i = 0; i < segCount; i++) {
    const s = segments[i];
    const o = i * 12;
    segData[o + 0] = s.p0[0]; segData[o + 1] = s.p0[1];
    segData[o + 2] = s.p1[0]; segData[o + 3] = s.p1[1];
    segData[o + 4] = s.color[0]; segData[o + 5] = s.color[1];
    segData[o + 6] = s.color[2]; segData[o + 7] = s.color[3];
    new Uint32Array(segData.buffer)[i * 12 + 8] = s.pathId;
  }

  // Buffers
  const segBuffer = device.createBuffer({
    size: segData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(segBuffer, 0, segData);

  const pixelBuffer = device.createBuffer({
    size: WIDTH * HEIGHT * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const paramsData = new Uint32Array([WIDTH, HEIGHT, segCount, 0]);
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  // Rasterize pipeline
  const rasterBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const rasterPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [rasterBGL] }),
    compute: { module: shaderModule, entryPoint: "rasterize" },
  });

  const rasterBG = device.createBindGroup({
    layout: rasterBGL,
    entries: [
      { binding: 0, resource: { buffer: segBuffer } },
      { binding: 1, resource: { buffer: pixelBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  // Render pipeline (fullscreen quad)
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

  // Execute
  const encoder = device.createCommandEncoder();

  const rasterPass = encoder.beginComputePass();
  rasterPass.setPipeline(rasterPipeline);
  rasterPass.setBindGroup(0, rasterBG);
  rasterPass.dispatchWorkgroups(Math.ceil(WIDTH / 16), Math.ceil(HEIGHT / 16));
  rasterPass.end();

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
  renderPass.draw(6);
  renderPass.end();

  device.queue.submit([encoder.finish()]);

  console.log(`Rendered ${beziers.length} beziers → ${segCount} segments → ${WIDTH}×${HEIGHT} pixels`);
}

main();
