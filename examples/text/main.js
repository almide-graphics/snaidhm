// snaidhm — Text rendering via path pipeline
//
// TTF → glyph outlines → cubic beziers → tiled winding fill
// Proves: "paths are paths" — text is just shapes

import { TTFFont, contoursToCubicBeziers } from "./ttf.js";

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

// ── Text → bezier paths ──

function textToBeziers(font, text, fontSize, startX, startY, color, startPathId) {
  const beziers = [];
  const scale = fontSize / font.unitsPerEm;
  let x = startX;
  let pathId = startPathId;

  for (const ch of text) {
    const charCode = ch.codePointAt(0);
    const glyphId = font.charToGlyphId(charCode);
    const advance = font.getAdvanceWidth(glyphId);

    const contours = font.getGlyphOutline(glyphId);
    if (contours) {
      // TTF Y is up, NDC Y is up — but our renderer flips Y in the shader
      // So we negate Y here to get correct orientation
      const glyphBeziers = contoursToCubicBeziers(contours, scale, x, startY);

      for (const bez of glyphBeziers) {
        beziers.push({ ...bez, color, pathId });
      }
      pathId++;
    }

    x += advance * scale;
  }

  return { beziers, nextPathId: pathId, endX: x };
}

// ── Flatten + tile assignment (reused from paths example) ──

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

function assignTiles(segments) {
  const tileCounts = new Uint32Array(NUM_TILES);
  const tileSegIds = new Uint32Array(NUM_TILES * MAX_SEGS_PER_TILE);

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const minX = Math.min(seg.p0[0], seg.p1[0]);
    const maxX = Math.max(seg.p0[0], seg.p1[0]);
    const minY = Math.min(seg.p0[1], seg.p1[1]);
    const maxY = Math.max(seg.p0[1], seg.p1[1]);

    const pxMinX = (minX + 1) * 0.5 * WIDTH;
    const pxMaxX = (maxX + 1) * 0.5 * WIDTH;
    const pxMinY = (1 - maxY) * 0.5 * HEIGHT;
    const pxMaxY = (1 - minY) * 0.5 * HEIGHT;

    const tMaxX = Math.min(TILES_X - 1, Math.floor(pxMaxX / TILE_SIZE));
    const tMinY = Math.max(0, Math.floor(pxMinY / TILE_SIZE));
    const tMaxY = Math.min(TILES_Y - 1, Math.floor(pxMaxY / TILE_SIZE));

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

  // Load font — use a system font or bundled TTF
  let fontBuffer;
  try {
    // Try loading a local font file
    const resp = await fetch("font.ttf");
    if (!resp.ok) throw new Error("font.ttf not found");
    fontBuffer = await resp.arrayBuffer();
  } catch {
    // Fallback: fetch a web font (Noto Sans)
    const resp = await fetch("https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf");
    fontBuffer = await resp.arrayBuffer();
  }

  const font = new TTFFont(fontBuffer);
  console.log(`Font loaded: ${font.numGlyphs} glyphs, ${font.unitsPerEm} units/em`);

  // Build scene: text + decorative shapes
  const allBeziers = [];
  let pathId = 0;

  // Title text
  const title = textToBeziers(font, "snaidhm", 0.25, -0.75, 0.3, [0.15, 0.25, 0.7, 1.0], pathId);
  allBeziers.push(...title.beziers);
  pathId = title.nextPathId;

  // Subtitle
  const subtitle = textToBeziers(font, "GPU path renderer", 0.12, -0.55, 0.0, [0.5, 0.5, 0.5, 1.0], pathId);
  allBeziers.push(...subtitle.beziers);
  pathId = subtitle.nextPathId;

  // More text
  const line3 = textToBeziers(font, "Almide + WebGPU + Winding Fill", 0.08, -0.8, -0.25, [0.3, 0.6, 0.3, 1.0], pathId);
  allBeziers.push(...line3.beziers);
  pathId = line3.nextPathId;

  // Numbers
  const line4 = textToBeziers(font, "0123456789", 0.15, -0.6, -0.5, [0.8, 0.3, 0.2, 1.0], pathId);
  allBeziers.push(...line4.beziers);
  pathId = line4.nextPathId;

  // Japanese (if font supports it)
  const line5 = textToBeziers(font, "paths are paths", 0.1, -0.55, -0.75, [0.6, 0.4, 0.8, 1.0], pathId);
  allBeziers.push(...line5.beziers);
  pathId = line5.nextPathId;

  console.log(`Text: ${allBeziers.length} beziers from ${pathId} glyphs`);

  // Flatten and tile
  const segments = flattenBeziers(allBeziers);
  const { tileCounts, tileSegIds } = assignTiles(segments);
  console.log(`Flattened: ${segments.length} segments, max/tile: ${Math.max(...tileCounts)}`);

  // ── WebGPU setup (same as paths example) ──

  const canvas = document.getElementById("canvas");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  const shaderSource = await fetch("paths.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  const segCount = segments.length;
  const segData = new Float32Array(segCount * 12);
  for (let i = 0; i < segCount; i++) {
    const s = segments[i], o = i * 12;
    segData[o] = s.p0[0]; segData[o+1] = s.p0[1];
    segData[o+2] = s.p1[0]; segData[o+3] = s.p1[1];
    segData[o+4] = s.color[0]; segData[o+5] = s.color[1];
    segData[o+6] = s.color[2]; segData[o+7] = s.color[3];
    new Uint32Array(segData.buffer)[i * 12 + 8] = s.pathId;
  }

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
