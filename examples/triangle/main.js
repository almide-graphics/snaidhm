// snaidhm Phase 0 — WebGPU triangle (hand-written host)
// This is what the Almide compiler should eventually generate.

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

  // Load WGSL shader
  const shaderSource = await fetch("triangle.wgsl").then(r => r.text());
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // Render
  function frame() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    renderPass.setPipeline(pipeline);
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
