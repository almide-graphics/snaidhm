# Phase 0: Triangle — Design Notes

## Goal

Prove end-to-end: `@gpu fn` in Almide → WGSL output → WebGPU triangle on screen.

## Compiler Requirements

### 1. `@gpu(vertex)` / `@gpu(fragment)` extraction

The compiler must detect `@gpu` annotated functions and:
- Emit them as WGSL source (string embedded in host output, or separate `.wgsl` file)
- NOT include them in the host WASM output as callable functions

### 2. Type mapping: Almide → WGSL

| Almide | WGSL |
|--------|------|
| `Vec2` | `vec2<f32>` |
| `Vec4` | `vec4<f32>` |
| `UInt32` | `u32` |
| `Float` | `f32` |
| `[T; N]` (fixed array) | `array<T, N>` |
| struct with `@builtin` / `@location` | WGSL struct with same annotations |

### 3. Built-in passthrough

`@builtin(position)`, `@builtin(vertex_index)`, `@location(N)` pass through directly to WGSL output. The compiler validates they appear only in `@gpu` functions.

### 4. Pipeline reference resolution

When host code writes `vertex: vs_main`, the compiler resolves this to:
- The WGSL source string for that function
- The entry point name (`"vs_main"`)

### 5. No bind groups in Phase 0

This triangle has no uniforms or storage buffers — just vertex_index.
Bind groups come in Phase 0.3 (uniform for MVP matrix).

## Expected WGSL Output

```wgsl
struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>(0.5, -0.5),
  );
  var colors = array<vec4<f32>, 3>(
    vec4<f32>(1.0, 0.0, 0.0, 1.0),
    vec4<f32>(0.0, 1.0, 0.0, 1.0),
    vec4<f32>(0.0, 0.0, 1.0, 1.0),
  );
  var output: VertexOutput;
  output.pos = vec4<f32>(positions[idx].x, positions[idx].y, 0.0, 1.0);
  output.color = colors[idx];
  return output;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
  return color;
}
```

## Expected Host Behavior (WASM)

1. Get canvas element
2. Request WebGPU adapter + device
3. Create shader module from embedded WGSL string
4. Create render pipeline (vertex + fragment entry points)
5. Each frame: begin render pass → draw(3) → end → submit

## Implementation Strategy

Since the Almide compiler's WGSL codegen doesn't exist yet, Phase 0 is split:

### Step 1: Hand-write the target outputs
- Write the expected `.wgsl` file manually
- Write the host WebGPU code manually (in Almide → WASM, or Rust/TS as temp scaffold)
- Prove the triangle renders

### Step 2: Implement `@gpu` → WGSL in the compiler
- Add WGSL as a codegen target in `almide/almide`
- Parse `@gpu` annotations
- Emit WGSL from the AST of annotated functions
- Embed WGSL string in host output

### Step 3: Compile `main.almd` end-to-end
- `almide build --target wasm examples/triangle/main.almd`
- Output: `.wasm` (host) + `.wgsl` (shaders) or single `.wasm` with embedded WGSL
- Serve in browser, triangle appears
