// The JS side of snaidhm's `gpu` extern namespace.
//
// `src/web/gpu.almd` declares the contract; this implements it. They belong in
// one place: the implementation used to live inside ceangal's demo, so a repo
// that merely CONSUMES snaidhm owned the only copy of snaidhm's own host, and
// every fix to it had to be discovered rather than inherited.
//
// GPU objects are opaque i64 handles on the wasm side. The table here is
// private to this namespace — `dom` handles are a separate space and the two
// never cross, so they do not need to share one.
//
// This module knows nothing about UI, fonts or the DOM. It is the browser
// boundary for the GPU and nothing else.

/// Build the `gpu` import object for a canvas.
///
/// The caller wires the returned pieces up:
///   host.setMemory(instance.exports.memory)   after instantiation
///   host.setFormat(navigator.gpu.getPreferredCanvasFormat())
///   host.registerShader(wgslSource)           -> the index create_shader resolves
///   host.register(gpuObject)                  -> a handle for something JS made
///   host.beginFrame()                         at the top of each frame
export function createGpuHost(canvas) {
  const handles = [null];
  const h = (obj) => { handles.push(obj); return handles.length - 1; };
  const g = (id) => handles[Number(id)];
  const B = (n) => BigInt(n);
  const N = (b) => Number(b);

  let _context = null, _format = null, _wasmMemory = null;
  // Depth target for the 3D pass, rebuilt when the drawing buffer resizes.
  let _depth = null;
  // When a 3D pass has already cleared the colour target this frame, the 2D
  // pass must LOAD instead of clearing again or it erases that output.
  let _clearedThisFrame = false;
  let _dataChunks = [], _dataIsF32 = [], _bindingEntries = [];
  let SHADERS = [];

  const imports = {
    get_preferred_format: () => B(h(_format)),
    configure_canvas(deviceId, _fmtId) {
      _context = canvas.getContext("webgpu");
      _context.configure({ device: g(deviceId), format: _format, alphaMode: "premultiplied" });
      return B(h(_context));
    },
    create_shader(deviceId, shaderId, _) {
      return B(h(g(deviceId).createShaderModule({ code: SHADERS[N(shaderId)] || SHADERS[0] })));
    },
    create_buffer(deviceId, size, usage) {
      return B(h(g(deviceId).createBuffer({ size: N(size), usage: N(usage) })));
    },
    create_stream_buffer(deviceId, size, usage) {
      return B(h(g(deviceId).createBuffer({ size: N(size), usage: N(usage) })));
    },
    destroy_buffer(_deviceId, bufferId) {
      const b = g(bufferId);
      if (b && typeof b.destroy === "function" && b instanceof GPUBuffer) {
        b.destroy();
        handles[Number(bufferId)] = null;
      }
    },
    write_buffer(deviceId, bufferId, dataPtr, dataLen) {
      g(deviceId).queue.writeBuffer(g(bufferId), 0, new Uint8Array(_wasmMemory.buffer, N(dataPtr), N(dataLen)));
    },
    write_f32_at(deviceId, bufferId, byteOffset, value) {
      g(deviceId).queue.writeBuffer(g(bufferId), N(byteOffset), new Float32Array([value]));
    },
    write_u32_at(deviceId, bufferId, byteOffset, value) {
      g(deviceId).queue.writeBuffer(g(bufferId), N(byteOffset), new Uint32Array([N(value)]));
    },
    create_compute_pipeline(deviceId, shaderId, _) {
      const p = g(deviceId).createComputePipeline({ layout: "auto", compute: { module: g(shaderId), entryPoint: "fine" } });
      return B(h(p));
    },
    create_render_pipeline(deviceId, shaderId, _vp, _vl, _fp, _fl, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_fullscreen" },
        // Blend, so the pass composites over whatever is already in the target
        // instead of overwriting it. Without this the coverage the fragment
        // shader reports is discarded and the layer is always opaque.
        fragment: { module: g(shaderId), entryPoint: "fs_fullscreen", targets: [{ format: _format, blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }}] },
        primitive: { topology: "triangle-list" },
      })));
    },
    create_text_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_main", buffers: [{ arrayStride: 32, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
          { shaderLocation: 2, offset: 16, format: "float32x4" },
        ]}] },
        fragment: { module: g(shaderId), entryPoint: "fs_main", targets: [{ format: _format, blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }}] },
        primitive: { topology: "triangle-list" },
      })));
    },
    create_image_pipeline(deviceId, shaderId, _fmt) {
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: { module: g(shaderId), entryPoint: "vs_main", buffers: [{ arrayStride: 16, attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32x2" },
        ]}] },
        fragment: { module: g(shaderId), entryPoint: "fs_main", targets: [{ format: _format, blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        }}] },
        primitive: { topology: "triangle-list" },
      })));
    },
    begin_bindings() { _bindingEntries = []; },
    add_buffer_binding(bufferId) { _bindingEntries.push({ kind: "buffer", obj: g(bufferId) }); },
    add_texture_binding(textureId) { _bindingEntries.push({ kind: "texture", obj: g(textureId) }); },
    add_sampler_binding(samplerId) { _bindingEntries.push({ kind: "sampler", obj: g(samplerId) }); },
    create_bound_group(deviceId, pipelineId, groupIdx) {
      const layout = g(pipelineId).getBindGroupLayout(N(groupIdx));
      const entries = _bindingEntries.map((e, i) => {
        if (e.kind === "buffer") return { binding: i, resource: { buffer: e.obj } };
        if (e.kind === "texture") return { binding: i, resource: e.obj.createView() };
        if (e.kind === "sampler") return { binding: i, resource: e.obj };
      });
      _bindingEntries = [];
      return B(h(g(deviceId).createBindGroup({ layout, entries })));
    },
    set_bind_group(passId, index, bgId) { g(passId).setBindGroup(N(index), g(bgId)); },
    begin_encoder: (deviceId) => B(h(g(deviceId).createCommandEncoder())),
    begin_compute_pass: (encoderId) => B(h(g(encoderId).beginComputePass())),
    dispatch_workgroups(passId, x, y, z) {
      g(passId).dispatchWorkgroups(N(x), N(y), N(z));
    },
    begin_render_pass(encoderId, r, g_, b, a) {
      return B(h(g(encoderId).beginRenderPass({
        colorAttachments: [{ view: _context.getCurrentTexture().createView(),
          clearValue: { r, g: g_, b, a },
          loadOp: _clearedThisFrame ? "load" : "clear", storeOp: "store" }],
      })));
    },
    set_pipeline(passId, pipelineId) { g(passId).setPipeline(g(pipelineId)); },
    draw(passId, n) { g(passId).draw(N(n)); },
    set_vertex_buffer(passId, slot, bufferId) { g(passId).setVertexBuffer(N(slot), g(bufferId)); },
    set_index_buffer(passId, bufferId) { g(passId).setIndexBuffer(g(bufferId), "uint32"); },
    draw_indexed(passId, count) { g(passId).drawIndexed(N(count)); },
    end_pass(passId) { g(passId).end(); },
    finish_and_submit(deviceId, encoderId) { g(deviceId).queue.submit([g(encoderId).finish()]); },
    begin_data() { _dataChunks = []; _dataIsF32 = []; },
    push_f32(v) { _dataChunks.push(v); _dataIsF32.push(true); },
    push_u32(v) { _dataChunks.push(N(v)); _dataIsF32.push(false); },
    flush_to_buffer(deviceId, bufferId) {
      const buf = new ArrayBuffer(_dataChunks.length * 4);
      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      for (let i = 0; i < _dataChunks.length; i++) {
        if (_dataIsF32[i]) f32[i] = _dataChunks[i]; else u32[i] = _dataChunks[i];
      }
      g(deviceId).queue.writeBuffer(g(bufferId), 0, new Uint8Array(buf));
      _dataChunks = []; _dataIsF32 = [];
    },
    flush_to_texture(deviceId, texId, x, y, w, h) {
      const width = N(w), height = N(h);
      const px = new Uint32Array(_dataChunks.length);
      for (let i = 0; i < _dataChunks.length; i++) px[i] = _dataChunks[i];
      if (width > 0 && height > 0 && px.length >= width * height) {
        g(deviceId).queue.writeTexture(
          { texture: g(texId), origin: [N(x), N(y)] },
          new Uint8Array(px.buffer, 0, width * height * 4),
          { bytesPerRow: width * 4 }, [width, height]);
      }
      _dataChunks = []; _dataIsF32 = [];
    },
    // ── Optional 3D pass ──
    //
    // ceangal's own pipeline is a fullscreen quad with no vertex buffers and no
    // depth attachment, so a mesh cannot go through it. These are the entry
    // points an app needs to render geometry UNDER the 2D layer, declared
    // against the same `gpu` namespace snaidhm owns. Pure boundary translation:
    // no application logic lives here.

    // NOTE: parameters must not be named `h`, `g`, `B` or `N` — those are the
    // handle-table helpers this module closes over, and a parameter of the same
    // name shadows them. `create_texture(deviceId, w, h)` did exactly that and
    // failed at runtime with "h is not a function", which the wasm side cannot
    // see and no signature check would catch.
    set_depth_size(deviceId, width_, height_) {
      const width = Math.max(1, N(width_)), height = Math.max(1, N(height_));
      if (_depth && _depth.width === width && _depth.height === height) return;
      if (_depth) _depth.tex.destroy();
      const tex = g(deviceId).createTexture({
        size: [width, height], format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      _depth = { tex, view: tex.createView(), width, height };
    },

    // Fixed mesh vertex layout: pos(3) + normal(3) + uv(2), 32-byte stride,
    // depth-tested with `less`, back faces culled, glTF's CCW front.
    create_mesh_pipeline(deviceId, shaderId, _fmt, cull, blend, depthWrite) {
      const cullMode = N(cull) === 0 ? "none" : N(cull) === 2 ? "front" : "back";
      const target = N(blend) === 1
        ? { format: _format, blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          } }
        : { format: _format };
      return B(h(g(deviceId).createRenderPipeline({
        layout: "auto",
        vertex: {
          module: g(shaderId), entryPoint: "vs_main",
          buffers: [{ arrayStride: 32, attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ]}],
        },
        fragment: { module: g(shaderId), entryPoint: "fs_main", targets: [target] },
        primitive: { topology: "triangle-list", cullMode, frontFace: "ccw" },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: N(depthWrite) !== 0,
          depthCompare: "less",
        },
      })));
    },

    // begin_render_pass with the depth attachment bound. `load` chooses whether
    // the colour target is cleared (0) or preserved (1). Depth always clears.
    begin_render_pass_3d(encoderId, r, g_, b, a, load) {
      if (!_depth) throw new Error("begin_render_pass_3d before set_depth_size");
      if (N(load) !== 1) _clearedThisFrame = true;
      return B(h(g(encoderId).beginRenderPass({
        colorAttachments: [{
          view: _context.getCurrentTexture().createView(),
          clearValue: { r, g: g_, b, a },
          loadOp: N(load) === 1 ? "load" : "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: _depth.view, depthClearValue: 1.0,
          depthLoadOp: "clear", depthStoreOp: "store",
        },
      })));
    },

    // ── Textures ──

    create_texture(deviceId, width_, height_) {
      return B(h(g(deviceId).createTexture({
        size: [Math.max(1, N(width_)), Math.max(1, N(height_))],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT,
      })));
    },

    // Async by nature: `createImageBitmap` is a promise and a wasm call is not.
    // The texture already exists at its final size, so every bind group built
    // from it stays valid — only its contents arrive late.
    upload_encoded_image(deviceId, textureId, ptr, len) {
      const device = g(deviceId), texture = g(textureId);
      if (!device || !texture) return;
      const encoded = new Uint8Array(_wasmMemory.buffer, N(ptr), N(len)).slice();
      createImageBitmap(new Blob([encoded]), { premultiplyAlpha: "none", colorSpaceConversion: "none" })
        .then((bmp) => {
          device.queue.copyExternalImageToTexture(
            { source: bmp },
            { texture },
            [Math.min(bmp.width, texture.width), Math.min(bmp.height, texture.height)],
          );
          bmp.close?.();
        })
        .catch((e) => console.warn("[gpu] image decode failed:", e.message));
    },

    create_sampler(deviceId, filter, wrap) {
      const f = N(filter) === 0 ? "nearest" : "linear";
      const w = N(wrap) === 1 ? "repeat" : "clamp-to-edge";
      return B(h(g(deviceId).createSampler({
        magFilter: f, minFilter: f, addressModeU: w, addressModeV: w,
      })));
    },

    draw_indexed_from(passId, first, count) {
      g(passId).drawIndexed(N(count), 1, N(first), 0, 0);
    },

    // u16 indices — half the bandwidth of the u32 path, and glTF's common case.
    set_index_buffer_u16(passId, bufferId) { g(passId).setIndexBuffer(g(bufferId), "uint16"); },

    // Upload from linear memory at an offset, so geometry built in a Bytes
    // arena reaches the GPU without an intermediate copy.
    write_buffer_at(deviceId, bufferId, dstOffset, srcPtr, len) {
      g(deviceId).queue.writeBuffer(g(bufferId), N(dstOffset),
        new Uint8Array(_wasmMemory.buffer, N(srcPtr), N(len)));
    },

    log_int(v) { console.log("[gpu]", N(v)); },
    log_str(ptr, len) { console.log("[gpu]", new TextDecoder().decode(new Uint8Array(_wasmMemory.buffer, N(ptr), N(len)))); },
  };

  return {
    imports,
    /// Hand something JS created (a texture, a sampler, a buffer) to the wasm
    /// side as a handle.
    register: h,
    resolve: g,
    setMemory(memory) { _wasmMemory = memory; },
    setFormat(format) { _format = format; },
    /// Append a WGSL module and return the index `create_shader` resolves it
    /// by. An app renders its own pass by adding its module here rather than
    /// forking this file.
    registerShader(code) { SHADERS.push(code); return SHADERS.length - 1; },
    /// Reset per-frame clear ownership. Call at the top of every frame.
    beginFrame() { _clearedThisFrame = false; },
    get context() { return _context; },
    get depthSize() { return _depth ? [_depth.width, _depth.height] : null; },
  };
}
