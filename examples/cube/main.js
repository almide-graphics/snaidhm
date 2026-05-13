// snaidhm Phase 0.3 — Rotating cube with uniform buffer (hand-written host)

// ── Matrix math (what lumen.mat4 will provide) ──

function mat4_identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function mat4_multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[j * 4 + i] =
        a[0 * 4 + i] * b[j * 4 + 0] +
        a[1 * 4 + i] * b[j * 4 + 1] +
        a[2 * 4 + i] * b[j * 4 + 2] +
        a[3 * 4 + i] * b[j * 4 + 3];
    }
  }
  return out;
}

function mat4_rotation_y(angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0,
     0, 0, 0, 1,
  ]);
}

function mat4_rotation_x(angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([
    1, 0,  0, 0,
    0, c, -s, 0,
    0, s,  c, 0,
    0, 0,  0, 1,
  ]);
}

function mat4_perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4_translate(x, y, z) {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

// ── Cube geometry ──

// 8 vertices: position (vec3) + color (vec3) = 6 floats per vertex
// prettier-ignore
const VERTICES = new Float32Array([
  // pos              color
  -1, -1, -1,    1, 0, 0,  // 0 red
   1, -1, -1,    0, 1, 0,  // 1 green
   1,  1, -1,    0, 0, 1,  // 2 blue
  -1,  1, -1,    1, 1, 0,  // 3 yellow
  -1, -1,  1,    1, 0, 1,  // 4 magenta
   1, -1,  1,    0, 1, 1,  // 5 cyan
   1,  1,  1,    1, 1, 1,  // 6 white
  -1,  1,  1,    0.5, 0.5, 0.5,  // 7 gray
]);

// 12 triangles (36 indices)
// prettier-ignore
const INDICES = new Uint16Array([
  0,1,2, 0,2,3,  // front
  4,6,5, 4,7,6,  // back
  0,4,5, 0,5,1,  // bottom
  2,6,7, 2,7,3,  // top
  0,3,7, 0,7,4,  // left
  1,5,6, 1,6,2,  // right
]);

// ── WebGPU setup ──

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

  // Shader
  const shaderSource = await fetch("cube.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Vertex buffer
  const vertexBuffer = device.createBuffer({
    size: VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, VERTICES);

  // Index buffer
  const indexBuffer = device.createBuffer({
    size: INDICES.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, INDICES);

  // Uniform buffer (mat4x4 = 64 bytes)
  const uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Bind group layout + bind group
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform" },
    }],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // Depth texture
  const depthTexture = device.createTexture({
    size: [512, 512],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Pipeline
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4, // 6 floats * 4 bytes
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },     // position
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" }, // color
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  // Projection matrix (static)
  const proj = mat4_perspective(Math.PI / 4, 1, 0.1, 100);
  const view = mat4_translate(0, 0, -5);

  // Render loop
  let t = 0;
  function frame() {
    t += 0.01;

    // MVP = proj * view * rotY * rotX
    const rotY = mat4_rotation_y(t);
    const rotX = mat4_rotation_x(t * 0.7);
    const model = mat4_multiply(rotY, rotX);
    const mv = mat4_multiply(view, model);
    const mvp = mat4_multiply(proj, mv);
    device.queue.writeBuffer(uniformBuffer, 0, mvp);

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint16");
    renderPass.drawIndexed(36);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
