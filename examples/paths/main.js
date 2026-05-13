// snaidhm Phase 1 — Tiled path renderer
//
// CPU: flatten (Wang's formula) + coarse tile assignment (AABB, sorted)
// GPU: fine rasterize (per-tile winding number → fill)
// GPU: fullscreen quad display

const WIDTH = 512;
const HEIGHT = 512;
const TILE_SIZE = 16;
const TILES_X = WIDTH / TILE_SIZE;
const TILES_Y = HEIGHT / TILE_SIZE;
const NUM_TILES = TILES_X * TILES_Y;
const MAX_SEGS_PER_TILE = 512;

// ── Bezier math ──

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

// ── Scene ──

function buildScene() {
  const beziers = [];
  let pathId = 0;

  function addCubic(p0, p1, p2, p3, color) {
    beziers.push({ p0, p1, p2, p3, color, pathId });
  }
  function closePath() { pathId++; }

  function circle(cx, cy, r, color) {
    const k = 0.5522847498 * r;
    addCubic([cx, cy+r], [cx+k, cy+r], [cx+r, cy+k], [cx+r, cy], color);
    addCubic([cx+r, cy], [cx+r, cy-k], [cx+k, cy-r], [cx, cy-r], color);
    addCubic([cx, cy-r], [cx-k, cy-r], [cx-r, cy-k], [cx-r, cy], color);
    addCubic([cx-r, cy], [cx-r, cy+k], [cx-k, cy+r], [cx, cy+r], color);
    closePath();
  }

  function roundRect(x, y, w, h, r, color) {
    const k = 0.5522847498 * r;
    addCubic([x+r, y+h], [x+r, y+h], [x+w-r, y+h], [x+w-r, y+h], color);
    addCubic([x+w-r, y+h], [x+w-r+k, y+h], [x+w, y+h-r+k], [x+w, y+h-r], color);
    addCubic([x+w, y+h-r], [x+w, y+h-r], [x+w, y+r], [x+w, y+r], color);
    addCubic([x+w, y+r], [x+w, y+r-k], [x+w-r+k, y], [x+w-r, y], color);
    addCubic([x+w-r, y], [x+w-r, y], [x+r, y], [x+r, y], color);
    addCubic([x+r, y], [x+r-k, y], [x, y+r-k], [x, y+r], color);
    addCubic([x, y+r], [x, y+r], [x, y+h-r], [x, y+h-r], color);
    addCubic([x, y+h-r], [x, y+h-r+k], [x+r-k, y+h], [x+r, y+h], color);
    closePath();
  }

  function heart(cx, cy, s, color) {
    addCubic([cx, cy-s*0.4], [cx, cy+s*0.4], [cx-s, cy+s*0.8], [cx-s, cy], color);
    addCubic([cx-s, cy], [cx-s, cy-s*0.8], [cx, cy-s*0.6], [cx, cy-s*1.2], color);
    addCubic([cx, cy-s*1.2], [cx, cy-s*0.6], [cx+s, cy-s*0.8], [cx+s, cy], color);
    addCubic([cx+s, cy], [cx+s, cy+s*0.8], [cx, cy+s*0.4], [cx, cy-s*0.4], color);
    closePath();
  }

  circle(0, 0, 0.6, [0.25, 0.35, 0.78, 1.0]);
  circle(-0.3, 0.3, 0.15, [1.0, 0.3, 0.3, 1.0]);
  circle(0.3, 0.3, 0.15, [0.3, 1.0, 0.3, 1.0]);
  circle(0.0, -0.2, 0.15, [1.0, 1.0, 0.3, 1.0]);
  roundRect(-0.8, -0.85, 0.5, 0.3, 0.06, [0.85, 0.45, 0.2, 1.0]);
  roundRect(0.3, -0.85, 0.5, 0.3, 0.08, [0.35, 0.8, 0.55, 1.0]);
  heart(0.0, 0.68, 0.12, [1.0, 0.2, 0.35, 1.0]);

  for (let i = 0; i < 50; i++) {
    const a = (i / 50) * Math.PI * 2, h = i / 50;
    circle(Math.cos(a) * 0.85, Math.sin(a) * 0.85, 0.03, [
      0.5 + 0.5 * Math.sin(h * 6.28),
      0.5 + 0.5 * Math.sin(h * 6.28 + 2.09),
      0.5 + 0.5 * Math.sin(h * 6.28 + 4.18), 1.0,
    ]);
  }

  for (let i = 0; i < 100; i++) {
    circle(
      (Math.random() * 2 - 1) * 0.9,
      (Math.random() * 2 - 1) * 0.9,
      0.01 + Math.random() * 0.04,
      [Math.random(), Math.random(), Math.random(), 0.7],
    );
  }

  return beziers;
}

// ── Flatten (CPU) ──

function flattenBeziers(beziers) {
  const tol = 0.5 / WIDTH;
  const segments = [];
  for (const bez of beziers) {
    const n = wangSegments(bez.p0, bez.p1, bez.p2, bez.p3, tol);
    for (let i = 0; i < n; i++) {
      segments.push({
        p0: cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, i / n),
        p1: cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, (i + 1) / n),
        color: bez.color,
        pathId: bez.pathId,
      });
    }
  }
  return segments;
}

// ── Coarse tile assignment (CPU, preserves path_id order) ──

function assignTiles(segments) {
  const tileCounts = new Uint32Array(NUM_TILES);
  const tileSegIds = new Uint32Array(NUM_TILES * MAX_SEGS_PER_TILE);

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    // AABB in NDC → pixel → tile
    const minX = Math.min(seg.p0[0], seg.p1[0]);
    const maxX = Math.max(seg.p0[0], seg.p1[0]);
    const minY = Math.min(seg.p0[1], seg.p1[1]);
    const maxY = Math.max(seg.p0[1], seg.p1[1]);

    // NDC to pixel
    const pxMinX = (minX + 1) * 0.5 * WIDTH;
    const pxMaxX = (maxX + 1) * 0.5 * WIDTH;
    const pxMinY = (1 - maxY) * 0.5 * HEIGHT;  // flip Y
    const pxMaxY = (1 - minY) * 0.5 * HEIGHT;

    const tMaxX = Math.min(TILES_X - 1, Math.floor(pxMaxX / TILE_SIZE));
    const tMinY = Math.max(0, Math.floor(pxMinY / TILE_SIZE));
    const tMaxY = Math.min(TILES_Y - 1, Math.floor(pxMaxY / TILE_SIZE));

    // Winding uses rightward ray: segment can affect any tile to its LEFT.
    // Assign to all columns 0..maxX for each overlapping row.
    for (let ty = tMinY; ty <= tMaxY; ty++) {
      for (let tx = 0; tx <= tMaxX; tx++) {
        const tileId = ty * TILES_X + tx;
        const slot = tileCounts[tileId];
        if (slot < MAX_SEGS_PER_TILE) {
          tileSegIds[tileId * MAX_SEGS_PER_TILE + slot] = si;
          tileCounts[tileId]++;
        }
      }
    }
  }

  return { tileCounts, tileSegIds };
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

  // Build scene
  const beziers = buildScene();
  const segments = flattenBeziers(beziers);
  const { tileCounts, tileSegIds } = assignTiles(segments);
  const segCount = segments.length;

  console.log(`Scene: ${beziers.length} beziers → ${segCount} segments, ${NUM_TILES} tiles`);
  const maxPerTile = Math.max(...tileCounts);
  console.log(`Max segments per tile: ${maxPerTile}`);

  // Pack segments
  const segData = new Float32Array(segCount * 12);
  for (let i = 0; i < segCount; i++) {
    const s = segments[i], o = i * 12;
    segData[o] = s.p0[0]; segData[o+1] = s.p0[1];
    segData[o+2] = s.p1[0]; segData[o+3] = s.p1[1];
    segData[o+4] = s.color[0]; segData[o+5] = s.color[1];
    segData[o+6] = s.color[2]; segData[o+7] = s.color[3];
    new Uint32Array(segData.buffer)[i * 12 + 8] = s.pathId;
  }

  // Buffers
  const segBuffer = device.createBuffer({ size: segData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(segBuffer, 0, segData);

  const tileCountBuffer = device.createBuffer({ size: tileCounts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(tileCountBuffer, 0, tileCounts);

  const tileSegBuffer = device.createBuffer({ size: tileSegIds.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(tileSegBuffer, 0, tileSegIds);

  const pixelBuffer = device.createBuffer({ size: WIDTH * HEIGHT * 4, usage: GPUBufferUsage.STORAGE });

  const paramsData = new Uint32Array([WIDTH, HEIGHT, segCount, TILES_X, TILES_Y, 0, 0, 0]);
  const paramsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  // Fine pipeline
  const fineBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const finePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [fineBGL] }),
    compute: { module: shaderModule, entryPoint: "fine" },
  });

  const fineBG = device.createBindGroup({
    layout: fineBGL,
    entries: [
      { binding: 0, resource: { buffer: segBuffer } },
      { binding: 1, resource: { buffer: tileCountBuffer } },
      { binding: 2, resource: { buffer: tileSegBuffer } },
      { binding: 3, resource: { buffer: pixelBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ],
  });

  // Render pipeline
  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: { module: shaderModule, entryPoint: "vs_fullscreen" },
    fragment: { module: shaderModule, entryPoint: "fs_fullscreen", targets: [{ format }] },
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

  const finePass = encoder.beginComputePass();
  finePass.setPipeline(finePipeline);
  finePass.setBindGroup(0, fineBG);
  finePass.dispatchWorkgroups(TILES_X, TILES_Y);
  finePass.end();

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
  });
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBG);
  renderPass.draw(6);
  renderPass.end();

  device.queue.submit([encoder.finish()]);
}

main();
