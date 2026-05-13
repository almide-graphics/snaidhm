// snaidhm — Combined renderer: path fill/stroke + SDF text
//
// Pass 1: Compute — tiled winding fill for shapes
// Pass 2: Render — fullscreen quad (path output) + SDF text quads on top
//
// This is the target architecture: shapes via compute path renderer,
// text via SDF atlas, both in the same frame.

import { TTFFont } from "./ttf.js";
import { generateSDFAtlas } from "./sdf.js";

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

// ── Scene: shapes ──

function buildShapes() {
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

  // Background card
  roundRect(-0.9, -0.9, 1.8, 1.8, 0.08, [1.0, 1.0, 1.0, 1.0]);

  // Header bar
  roundRect(-0.85, 0.55, 1.7, 0.3, 0.04, [0.2, 0.35, 0.75, 1.0]);

  // Decorative circles
  circle(-0.5, 0.0, 0.18, [1.0, 0.35, 0.3, 0.9]);
  circle(0.0, 0.0, 0.18, [0.3, 0.8, 0.4, 0.9]);
  circle(0.5, 0.0, 0.18, [0.3, 0.5, 1.0, 0.9]);

  // Bottom cards
  roundRect(-0.85, -0.85, 0.5, 0.55, 0.04, [0.95, 0.95, 0.98, 1.0]);
  roundRect(-0.3, -0.85, 0.5, 0.55, 0.04, [0.95, 0.95, 0.98, 1.0]);
  roundRect(0.25, -0.85, 0.6, 0.55, 0.04, [0.95, 0.95, 0.98, 1.0]);

  // Accent dots
  for (let i = 0; i < 5; i++) {
    const x = -0.6 + i * 0.3;
    circle(x, 0.4, 0.025, [1.0, 1.0, 1.0, 0.8]);
  }

  return { beziers, pathCount: pathId };
}

// ── Scene: shadows (SDF-based analytical soft shadows) ──

function buildShadows() {
  const shadows = [];

  function addRectShadow(x, y, w, h, r, ox, oy, blur, color) {
    shadows.push({
      centerX: x + w / 2, centerY: y + h / 2,
      halfW: w / 2, halfH: h / 2,
      cornerRadius: r, offsetX: ox, offsetY: oy, blur, color,
    });
  }

  function addCircleShadow(cx, cy, radius, ox, oy, blur, color) {
    shadows.push({
      centerX: cx, centerY: cy,
      halfW: radius, halfH: radius,
      cornerRadius: radius, offsetX: ox, offsetY: oy, blur, color,
    });
  }

  // Main card — large soft shadow
  addRectShadow(-0.9, -0.9, 1.8, 1.8, 0.08,  0.01, -0.02, 0.08, [0, 0, 0, 0.2]);

  // Circles — medium shadow
  addCircleShadow(-0.5, 0.0, 0.18,  0.008, -0.015, 0.04, [0, 0, 0, 0.25]);
  addCircleShadow( 0.0, 0.0, 0.18,  0.008, -0.015, 0.04, [0, 0, 0, 0.25]);
  addCircleShadow( 0.5, 0.0, 0.18,  0.008, -0.015, 0.04, [0, 0, 0, 0.25]);

  // Bottom cards — subtle shadow
  addRectShadow(-0.85, -0.85, 0.5, 0.55, 0.04,  0.005, -0.01, 0.03, [0, 0, 0, 0.15]);
  addRectShadow(-0.3,  -0.85, 0.5, 0.55, 0.04,  0.005, -0.01, 0.03, [0, 0, 0, 0.15]);
  addRectShadow( 0.25, -0.85, 0.6, 0.55, 0.04,  0.005, -0.01, 0.03, [0, 0, 0, 0.15]);

  // Pack into Float32Array: 12 floats per shadow (48 bytes, vec4-aligned)
  const data = new Float32Array(shadows.length * 12);
  for (let i = 0; i < shadows.length; i++) {
    const s = shadows[i], o = i * 12;
    data[o + 0] = s.centerX;
    data[o + 1] = s.centerY;
    data[o + 2] = s.halfW;
    data[o + 3] = s.halfH;
    data[o + 4] = s.cornerRadius;
    data[o + 5] = s.offsetX;
    data[o + 6] = s.offsetY;
    data[o + 7] = s.blur;
    data[o + 8] = s.color[0];
    data[o + 9] = s.color[1];
    data[o + 10] = s.color[2];
    data[o + 11] = s.color[3];
  }
  return { data, count: shadows.length };
}

// ── Paint descriptors (per-path) ──

function buildPaints(pathCount) {
  // 16 floats per paint (64 bytes, vec4-aligned)
  // Layout: [type,pad,pad,pad, color0.rgba, color1.rgba, params.xyzw]
  const data = new Float32Array(pathCount * 16);
  const u32 = new Uint32Array(data.buffer);

  function solid(pid, r, g, b, a) {
    const o = pid * 16;
    u32[o] = 0;
    data[o+4] = r; data[o+5] = g; data[o+6] = b; data[o+7] = a;
  }

  function linearGrad(pid, c0, c1, sx, sy, ex, ey) {
    const o = pid * 16;
    u32[o] = 1;
    data[o+4] = c0[0]; data[o+5] = c0[1]; data[o+6] = c0[2]; data[o+7] = c0[3];
    data[o+8] = c1[0]; data[o+9] = c1[1]; data[o+10] = c1[2]; data[o+11] = c1[3];
    data[o+12] = sx; data[o+13] = sy; data[o+14] = ex; data[o+15] = ey;
  }

  function radialGrad(pid, c0, c1, cx, cy, radius) {
    const o = pid * 16;
    u32[o] = 2;
    data[o+4] = c0[0]; data[o+5] = c0[1]; data[o+6] = c0[2]; data[o+7] = c0[3];
    data[o+8] = c1[0]; data[o+9] = c1[1]; data[o+10] = c1[2]; data[o+11] = c1[3];
    data[o+12] = cx; data[o+13] = cy; data[o+14] = radius;
  }

  // 0: Background card — solid white
  solid(0, 1.0, 1.0, 1.0, 1.0);

  // 1: Header bar — linear gradient (deep blue top → lighter blue bottom)
  linearGrad(1,
    [0.12, 0.22, 0.58, 1.0],
    [0.28, 0.48, 0.88, 1.0],
    -0.85, 0.85, -0.85, 0.55);

  // 2: Red circle — radial gradient (bright center → deeper edge)
  radialGrad(2, [1.0, 0.55, 0.5, 0.9], [0.82, 0.18, 0.12, 0.9], -0.5, 0.0, 0.18);

  // 3: Green circle
  radialGrad(3, [0.45, 0.92, 0.55, 0.9], [0.12, 0.62, 0.22, 0.9], 0.0, 0.0, 0.18);

  // 4: Blue circle
  radialGrad(4, [0.5, 0.7, 1.0, 0.9], [0.15, 0.35, 0.82, 0.9], 0.5, 0.0, 0.18);

  // 5-7: Bottom cards — solid
  solid(5, 0.95, 0.95, 0.98, 1.0);
  solid(6, 0.95, 0.95, 0.98, 1.0);
  solid(7, 0.95, 0.95, 0.98, 1.0);

  // 8-12: Accent dots
  for (let i = 8; i < pathCount; i++) solid(i, 1.0, 1.0, 1.0, 0.8);

  return data;
}

// ── Image textures (procedural thumbnails) ──

function generateCardImages() {
  const S = 64;
  const atlas = new Uint8Array(S * 3 * S * 4); // 3 images side by side: 192×64

  for (let img = 0; img < 3; img++) {
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = (y * S * 3 + img * S + x) * 4;
        const u = x / S, v = y / S;
        let r, g, b;

        if (img === 0) {
          // Sunset: warm gradient with sun glow
          const dx = u - 0.5, dy = v - 0.35;
          const sun = Math.max(0, 1 - Math.sqrt(dx*dx + dy*dy) * 4.5);
          r = 0.92 - v * 0.35 + sun * 0.4;
          g = 0.35 + sun * 0.55 - v * 0.15;
          b = 0.25 + v * 0.45;
        } else if (img === 1) {
          // Ocean: cool blues and teals
          const wave = Math.sin(u * 12 + v * 4) * 0.04;
          r = 0.1 + v * 0.15;
          g = 0.35 + u * 0.2 + wave;
          b = 0.55 + v * 0.25 + wave;
        } else {
          // Aurora: purple-green bands
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
  // 3 image quads, one per card. Vertex: pos(2) + uv(2) = 4 floats, 16 bytes
  const cards = [
    { x: -0.82, y: -0.82, w: 0.42, h: 0.32, uOff: 0 },
    { x: -0.27, y: -0.82, w: 0.42, h: 0.32, uOff: 1/3 },
    { x:  0.28, y: -0.82, w: 0.50, h: 0.32, uOff: 2/3 },
  ];

  const verts = [], idxs = [];
  for (const c of cards) {
    const vi = verts.length / 4;
    const u0 = c.uOff, u1 = c.uOff + 1/3;
    verts.push(c.x, c.y, u0, 1,           c.x+c.w, c.y, u1, 1,
               c.x+c.w, c.y+c.h, u1, 0,   c.x, c.y+c.h, u0, 0);
    idxs.push(vi, vi+1, vi+2, vi, vi+2, vi+3);
  }

  return { vertices: new Float32Array(verts), indices: new Uint32Array(idxs) };
}

// ── Flatten + tile ──

function flattenBeziers(beziers) {
  const tol = 0.5 / WIDTH;
  const segments = [];
  for (const bez of beziers) {
    const n = wangSegments(bez.p0, bez.p1, bez.p2, bez.p3, tol);
    for (let i = 0; i < n; i++) {
      segments.push({
        p0: cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, i / n),
        p1: cubicEval(bez.p0, bez.p1, bez.p2, bez.p3, (i + 1) / n),
        color: bez.color, pathId: bez.pathId,
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
    const maxX = Math.max(seg.p0[0], seg.p1[0]);
    const minY = Math.min(seg.p0[1], seg.p1[1]);
    const maxY = Math.max(seg.p0[1], seg.p1[1]);
    const tMaxX = Math.min(TILES_X - 1, Math.floor((maxX + 1) * 0.5 * WIDTH / TILE_SIZE));
    const tMinY = Math.max(0, Math.floor((1 - maxY) * 0.5 * HEIGHT / TILE_SIZE));
    const tMaxY = Math.min(TILES_Y - 1, Math.floor((1 - minY) * 0.5 * HEIGHT / TILE_SIZE));
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

// ── SDF text quads ──

function measureText(text, size, font, atlas) {
  const scale = size * 2 / HEIGHT / font.unitsPerEm;
  let width = 0;
  for (const ch of text) {
    const glyph = atlas.glyphs.get(ch);
    width += (glyph?.advance || font.unitsPerEm * 0.3) * scale;
  }
  return width;
}

function buildTextQuads(font, atlas) {
  // x: anchor position. align: "left" (default), "center", "right"
  // y: baseline position in NDC
  const lines = [
    { text: "snaidhm", size: 40, x: 0.0, y: 0.64, align: "center", color: [1, 1, 1, 1] },
    { text: "GPU path renderer", size: 14, x: 0.0, y: 0.57, align: "center", color: [0.8, 0.85, 1, 0.9] },
    { text: "Fill", size: 13, x: -0.5, y: -0.02, align: "center", color: [1, 1, 1, 1] },
    { text: "Stroke", size: 13, x: 0.0, y: -0.02, align: "center", color: [1, 1, 1, 1] },
    { text: "SDF", size: 13, x: 0.5, y: -0.02, align: "center", color: [1, 1, 1, 1] },
    { text: "Card A", size: 11, x: -0.8, y: -0.4, color: [0.3, 0.3, 0.4, 1] },
    { text: "Card B", size: 11, x: -0.25, y: -0.4, color: [0.3, 0.3, 0.4, 1] },
    { text: "Card C", size: 11, x: 0.3, y: -0.4, color: [0.3, 0.3, 0.4, 1] },
    { text: "Shapes + Text", size: 16, x: 0.0, y: -0.68, align: "center", color: [0.5, 0.5, 0.6, 1] },
    { text: "in one frame", size: 16, x: 0.0, y: -0.78, align: "center", color: [0.5, 0.5, 0.6, 1] },
  ];

  const vertices = [], indices = [];
  for (const line of lines) {
    const scale = line.size * 2 / HEIGHT / font.unitsPerEm;
    const textWidth = measureText(line.text, line.size, font, atlas);
    let cursorX = line.x;
    if (line.align === "center") cursorX -= textWidth / 2;
    else if (line.align === "right") cursorX -= textWidth;
    for (const ch of line.text) {
      const glyph = atlas.glyphs.get(ch);
      if (!glyph || glyph.atlasW === 0) {
        cursorX += (glyph?.advance || font.unitsPerEm * 0.3) * scale;
        continue;
      }
      const pad = glyph.padding / glyph.sdfScale;
      const x0 = cursorX + (glyph.bounds.xMin - pad) * scale;
      const y0 = line.y + (glyph.bounds.yMin - pad) * scale;
      const x1 = cursorX + (glyph.bounds.xMax + pad) * scale;
      const y1 = line.y + (glyph.bounds.yMax + pad) * scale;
      const u0 = glyph.atlasX / atlas.atlasWidth;
      const v0 = glyph.atlasY / atlas.atlasHeight;
      const u1 = (glyph.atlasX + glyph.atlasW) / atlas.atlasWidth;
      const v1 = (glyph.atlasY + glyph.atlasH) / atlas.atlasHeight;
      const c = line.color, vi = vertices.length / 8;
      vertices.push(x0, y0, u0, v1, c[0], c[1], c[2], c[3]);
      vertices.push(x1, y0, u1, v1, c[0], c[1], c[2], c[3]);
      vertices.push(x1, y1, u1, v0, c[0], c[1], c[2], c[3]);
      vertices.push(x0, y1, u0, v0, c[0], c[1], c[2], c[3]);
      indices.push(vi, vi+1, vi+2, vi, vi+2, vi+3);
      cursorX += glyph.advance * scale;
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

// ── WebGPU ──

async function main() {
  if (!navigator.gpu) { document.body.textContent = "WebGPU not supported"; return; }

  // Load font
  const fontBuffer = await fetch("font.ttf").then(r => r.arrayBuffer());
  const font = new TTFFont(fontBuffer);

  // SDF atlas
  const chars = [];
  for (let i = 32; i < 127; i++) chars.push(String.fromCharCode(i));
  const atlas = generateSDFAtlas(font, chars, 48, 6);

  // Build scene
  const { beziers, pathCount } = buildShapes();
  const segments = flattenBeziers(beziers);
  const { tileCounts, tileSegIds } = assignTiles(segments);
  const textQuads = buildTextQuads(font, atlas);
  const shadowScene = buildShadows();
  const paintData = buildPaints(pathCount);
  const cardImages = generateCardImages();
  const imageQuads = buildImageQuads();

  console.log(`Shapes: ${segments.length} segments | Text: ${textQuads.indices.length / 6} glyphs`);

  // WebGPU setup
  const canvas = document.getElementById("canvas");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  const pathShader = device.createShaderModule({ code: await fetch("paths.wgsl").then(r => r.text()) });
  const textShader = device.createShaderModule({ code: await fetch("text.wgsl").then(r => r.text()) });
  const imageShader = device.createShaderModule({ code: await fetch("image.wgsl").then(r => r.text()) });

  // ── Path renderer buffers ──

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
  const tileCountBuf = device.createBuffer({ size: tileCounts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(tileCountBuf, 0, tileCounts);
  const tileSegBuf = device.createBuffer({ size: tileSegIds.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(tileSegBuf, 0, tileSegIds);
  const pixelBuf = device.createBuffer({ size: WIDTH * HEIGHT * 4, usage: GPUBufferUsage.STORAGE });
  const paramsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const shadowBuf = device.createBuffer({ size: shadowScene.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shadowBuf, 0, shadowScene.data);
  const paintBuf = device.createBuffer({ size: paintData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paintBuf, 0, paintData);
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([WIDTH, HEIGHT, segCount, TILES_X, TILES_Y, shadowScene.count, 0, 0]));

  // Path compute pipeline
  const pathBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ]});
  const pathPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [pathBGL] }),
    compute: { module: pathShader, entryPoint: "fine" },
  });
  const pathBG = device.createBindGroup({ layout: pathBGL, entries: [
    { binding: 0, resource: { buffer: segBuffer } },
    { binding: 1, resource: { buffer: tileCountBuf } },
    { binding: 2, resource: { buffer: tileSegBuf } },
    { binding: 3, resource: { buffer: pixelBuf } },
    { binding: 4, resource: { buffer: paramsBuf } },
    { binding: 5, resource: { buffer: shadowBuf } },
    { binding: 6, resource: { buffer: paintBuf } },
  ]});

  // Fullscreen quad pipeline (path output)
  const quadBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
  ]});
  const quadPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [quadBGL] }),
    vertex: { module: pathShader, entryPoint: "vs_fullscreen" },
    fragment: { module: pathShader, entryPoint: "fs_fullscreen", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const quadBG = device.createBindGroup({ layout: quadBGL, entries: [
    { binding: 0, resource: { buffer: pixelBuf } },
    { binding: 1, resource: { buffer: paramsBuf } },
  ]});

  // ── SDF text pipeline ──

  const sdfTexture = device.createTexture({
    size: [atlas.atlasWidth, atlas.atlasHeight],
    format: "r8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const bytesPerRow = Math.ceil(atlas.atlasWidth / 256) * 256;
  const alignedData = new Uint8Array(bytesPerRow * atlas.atlasHeight);
  for (let row = 0; row < atlas.atlasHeight; row++) {
    alignedData.set(atlas.atlasData.subarray(row * atlas.atlasWidth, row * atlas.atlasWidth + atlas.atlasWidth), row * bytesPerRow);
  }
  device.queue.writeTexture({ texture: sdfTexture }, alignedData, { bytesPerRow }, [atlas.atlasWidth, atlas.atlasHeight]);

  const sdfSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const textVertBuf = device.createBuffer({ size: textQuads.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(textVertBuf, 0, textQuads.vertices);
  const textIdxBuf = device.createBuffer({ size: textQuads.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(textIdxBuf, 0, textQuads.indices);
  const textParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(textParamsBuf, 0, new Float32Array([atlas.atlasWidth, atlas.atlasHeight, WIDTH, HEIGHT]));

  const textBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
  ]});
  const textPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [textBGL] }),
    vertex: {
      module: textShader, entryPoint: "vs_main",
      buffers: [{ arrayStride: 32, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
        { shaderLocation: 2, offset: 16, format: "float32x4" },
      ]}],
    },
    fragment: {
      module: textShader, entryPoint: "fs_main",
      targets: [{ format, blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      }}],
    },
    primitive: { topology: "triangle-list" },
  });
  const textBG = device.createBindGroup({ layout: textBGL, entries: [
    { binding: 0, resource: { buffer: textParamsBuf } },
    { binding: 1, resource: sdfTexture.createView() },
    { binding: 2, resource: sdfSampler },
  ]});

  // ── Image pipeline ──

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
  const imgVertBuf = device.createBuffer({ size: imageQuads.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(imgVertBuf, 0, imageQuads.vertices);
  const imgIdxBuf = device.createBuffer({ size: imageQuads.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(imgIdxBuf, 0, imageQuads.indices);

  const imgBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
  ]});
  const imgPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [imgBGL] }),
    vertex: {
      module: imageShader, entryPoint: "vs_main",
      buffers: [{ arrayStride: 16, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x2" },
        { shaderLocation: 1, offset: 8, format: "float32x2" },
      ]}],
    },
    fragment: {
      module: imageShader, entryPoint: "fs_main",
      targets: [{ format, blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      }}],
    },
    primitive: { topology: "triangle-list" },
  });
  const imgBG = device.createBindGroup({ layout: imgBGL, entries: [
    { binding: 0, resource: imgTexture.createView() },
    { binding: 1, resource: imgSampler },
  ]});

  // ── Render frame ──

  const encoder = device.createCommandEncoder();

  // Pass 1: Compute path rasterization
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(pathPipeline);
  computePass.setBindGroup(0, pathBG);
  computePass.dispatchWorkgroups(TILES_X, TILES_Y);
  computePass.end();

  // Pass 2: Render — fullscreen quad (paths) then SDF text on top
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
  });

  // Draw path output as fullscreen quad
  renderPass.setPipeline(quadPipeline);
  renderPass.setBindGroup(0, quadBG);
  renderPass.draw(6);

  // Draw image textures on cards
  renderPass.setPipeline(imgPipeline);
  renderPass.setBindGroup(0, imgBG);
  renderPass.setVertexBuffer(0, imgVertBuf);
  renderPass.setIndexBuffer(imgIdxBuf, "uint32");
  renderPass.drawIndexed(imageQuads.indices.length);

  // Draw SDF text on top
  renderPass.setPipeline(textPipeline);
  renderPass.setBindGroup(0, textBG);
  renderPass.setVertexBuffer(0, textVertBuf);
  renderPass.setIndexBuffer(textIdxBuf, "uint32");
  renderPass.drawIndexed(textQuads.indices.length);

  renderPass.end();
  device.queue.submit([encoder.finish()]);
}

main();
