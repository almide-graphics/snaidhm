// snaidhm — SDF text rendering
//
// TTF → glyph outlines → SDF atlas → GPU textured quads with smoothstep AA
// Produces crisp text at any size without path rendering artifacts.

import { TTFFont } from "./ttf.js";
import { generateSDFAtlas } from "./sdf.js";

const WIDTH = 512;
const HEIGHT = 512;

async function main() {
  if (!navigator.gpu) {
    document.body.textContent = "WebGPU not supported";
    return;
  }

  // Load font
  let fontBuffer;
  try {
    const resp = await fetch("font.ttf");
    if (!resp.ok) throw new Error();
    fontBuffer = await resp.arrayBuffer();
  } catch {
    document.body.textContent = "font.ttf not found — copy Arial.ttf to examples/text/font.ttf";
    return;
  }
  const font = new TTFFont(fontBuffer);
  console.log(`Font: ${font.numGlyphs} glyphs, ${font.unitsPerEm} UPM`);

  // Generate SDF atlas for ASCII printable chars
  const chars = [];
  for (let i = 32; i < 127; i++) chars.push(String.fromCharCode(i));
  console.time("SDF atlas");
  const atlas = generateSDFAtlas(font, chars, 48, 6);
  console.timeEnd("SDF atlas");
  console.log(`Atlas: ${atlas.atlasWidth}×${atlas.atlasHeight}, ${atlas.glyphs.size} glyphs`);



  // Build text quads
  const lines = [
    { text: "snaidhm", size: 72, x: -0.72, y: 0.35, color: [0.15, 0.25, 0.7, 1] },
    { text: "GPU path renderer", size: 32, x: -0.52, y: 0.05, color: [0.4, 0.4, 0.4, 1] },
    { text: "SDF text — crisp at any size", size: 22, x: -0.65, y: -0.15, color: [0.3, 0.6, 0.3, 1] },
    { text: "0123456789 ABCDEF", size: 36, x: -0.62, y: -0.38, color: [0.8, 0.3, 0.2, 1] },
    { text: "paths are paths", size: 28, x: -0.45, y: -0.58, color: [0.6, 0.4, 0.8, 1] },
    { text: "Figma? Hold my beer.", size: 18, x: -0.48, y: -0.78, color: [0.5, 0.5, 0.5, 1] },
  ];

  const vertices = []; // 8 floats per vertex: pos(2) + uv(2) + color(4)
  const indices = [];

  for (const line of lines) {
    let cursorX = line.x;
    // Convert pixel font size to NDC scale:
    // unitsPerEm → size pixels → size * 2/HEIGHT NDC units
    const scale = line.size * 2 / HEIGHT / font.unitsPerEm;

    for (const ch of line.text) {
      const glyph = atlas.glyphs.get(ch);
      if (!glyph || glyph.atlasW === 0) {
        cursorX += (glyph?.advance || font.unitsPerEm * 0.3) * scale;
        continue;
      }

      // Glyph quad in NDC
      const pad = glyph.padding / glyph.sdfScale;
      const x0 = cursorX + (glyph.bounds.xMin - pad) * scale;
      const y0 = line.y + (glyph.bounds.yMin - pad) * scale;
      const x1 = cursorX + (glyph.bounds.xMax + pad) * scale;
      const y1 = line.y + (glyph.bounds.yMax + pad) * scale;

      // Atlas UVs (WebGPU: UV origin = top-left, V down)
      const u0 = glyph.atlasX / atlas.atlasWidth;
      const v0 = glyph.atlasY / atlas.atlasHeight;
      const u1 = (glyph.atlasX + glyph.atlasW) / atlas.atlasWidth;
      const v1 = (glyph.atlasY + glyph.atlasH) / atlas.atlasHeight;

      const c = line.color;
      const vi = vertices.length / 8;

      // 4 vertices per quad
      // NDC: y0=bottom, y1=top. UV: v0=top of SDF, v1=bottom of SDF
      vertices.push(x0, y0, u0, v1, c[0], c[1], c[2], c[3]); // bottom-left
      vertices.push(x1, y0, u1, v1, c[0], c[1], c[2], c[3]); // bottom-right
      vertices.push(x1, y1, u1, v0, c[0], c[1], c[2], c[3]); // top-right
      vertices.push(x0, y1, u0, v0, c[0], c[1], c[2], c[3]); // top-left

      // 2 triangles
      indices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);

      cursorX += glyph.advance * scale;
    }
  }

  console.log(`Quads: ${indices.length / 6} glyphs, ${vertices.length / 8} vertices`);

  // ── WebGPU ──

  const canvas = document.getElementById("canvas");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  const shaderSource = await fetch("text.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Atlas texture (R8 → sampled as float)
  const texture = device.createTexture({
    size: [atlas.atlasWidth, atlas.atlasHeight],
    format: "r8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // WebGPU requires bytesPerRow aligned to 256
  const bytesPerRow = Math.ceil(atlas.atlasWidth / 256) * 256;
  // Repack data with proper row alignment
  const alignedData = new Uint8Array(bytesPerRow * atlas.atlasHeight);
  for (let row = 0; row < atlas.atlasHeight; row++) {
    alignedData.set(
      atlas.atlasData.subarray(row * atlas.atlasWidth, row * atlas.atlasWidth + atlas.atlasWidth),
      row * bytesPerRow,
    );
  }
  device.queue.writeTexture(
    { texture },
    alignedData,
    { bytesPerRow },
    [atlas.atlasWidth, atlas.atlasHeight],
  );

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // Vertex + index buffers
  const vertData = new Float32Array(vertices);
  const idxData = new Uint32Array(indices);

  const vertBuffer = device.createBuffer({
    size: vertData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertBuffer, 0, vertData);

  const idxBuffer = device.createBuffer({
    size: idxData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(idxBuffer, 0, idxData);

  // Uniform
  const paramsData = new Float32Array([atlas.atlasWidth, atlas.atlasHeight, WIDTH, HEIGHT]);
  const paramsBuffer = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  // Pipeline
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },     // pos
          { shaderLocation: 1, offset: 2 * 4, format: "float32x2" }, // uv
          { shaderLocation: 2, offset: 4 * 4, format: "float32x4" }, // color
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: sampler },
    ],
  });

  // Render
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.95, g: 0.95, b: 0.97, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertBuffer);
  pass.setIndexBuffer(idxBuffer, "uint32");
  pass.drawIndexed(indices.length);
  pass.end();

  device.queue.submit([encoder.finish()]);
}

main();
