# Phase 0.3: Rotating Cube — Design Notes

## Goal

Express uniform buffers, vertex input, and matrix types in Almide so that
`almide main.almd --target wgsl` produces the hand-written `cube.wgsl`.

## The Hard Question

Phase 0 was pure codegen — Almide already had the syntax, we just emitted WGSL.
Phase 0.3 requires **new language constructs** because GPU resources
(uniform buffers, storage buffers, textures) don't exist in Almide today.

The design here directly constrains:
- Phase 0.5: Frame Arena + double-buffered storage buffers
- Phase 1+: Full path renderer with compute shaders
- The "Third Way" memory model (@read/@write, @persistent)

Getting this wrong means rework. Getting this right means the rest falls into place.

## What cube.wgsl Needs That Almide Doesn't Have

### 1. Module-level GPU variable declarations

```wgsl
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
```

Almide has `let` (immutable) and `var` (mutable) at module level, but no
concept of a GPU-resident variable with an address space qualifier
(`uniform`, `storage`, `private`, `workgroup`).

### 2. Vertex input with @location

```wgsl
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
) -> VertexOutput { ... }
```

This is solved — @location on params already works.

### 3. Vec/Mat types

```wgsl
vec3<f32>, vec4<f32>, mat4x4<f32>
```

Almide doesn't have these as built-in types. They come from lumen.

## Design Options

### Option A: `@gpu` module-level let with address space annotation

```almide
@group(0) @binding(0) @uniform
let uniforms: Uniforms
```

- `@uniform` / `@storage` / `@workgroup` as annotations on let/var
- The WGSL emitter sees the annotation and emits `var<uniform>`
- No new syntax needed — reuses existing let + annotation system
- **Concern**: `let` implies immutable value binding, not GPU resource declaration

### Option B: Dedicated `gpu` block

```almide
gpu {
  @group(0) @binding(0)
  uniform uniforms: Uniforms

  @group(0) @binding(1)
  storage(read) particles: [Particle]
}
```

- Explicit block separates GPU declarations from CPU code
- Keywords `uniform`, `storage`, `workgroup` instead of annotations
- **Concern**: new syntax, parser work, different from rest of Almide

### Option C: Annotation-only, inferred from @gpu context

```almide
@gpu(vertex)
fn vs_main(
  @group(0) @binding(0) @uniform uniforms: Uniforms,
  @location(0) position: Vec3,
  @location(1) color: Vec3,
) -> VertexOutput = { ... }
```

- Uniform buffer appears as a parameter with @uniform annotation
- Compiler separates it into a module-level `var<uniform>` in WGSL
- From Almide's perspective, the shader function just takes all its inputs
  as parameters — whether they come from vertex buffers or uniform buffers
  is determined by the annotation
- **Advantage**: no new syntax at all, LLM sees "function with annotated params"
- **Concern**: mixing buffer bindings with vertex inputs in the same param list

### Option D: Top-level `@gpu` declarations (recommended direction)

```almide
@gpu @group(0) @binding(0)
let uniforms: Uniforms = uniform()

@gpu @group(0) @binding(1)
var particles: [Particle] = storage(read_write)
```

- Uses `let`/`var` + `@gpu` annotation to mark GPU-resident
- Address space expressed as a "constructor": `uniform()`, `storage(read)`, `storage(read_write)`
- `let` = read-only (uniform), `var` = read-write (storage)
- Aligns with Frame Arena design: `var` without `@gpu` = frame arena, `@persistent` = dedicated
- **Advantage**: minimal syntax, leverages existing let/var semantics

## Recommended: Option C (params) + Option A (module-level)

Combine two approaches for different use cases:

### For uniforms/storage that the shader reads:
```almide
@group(0) @binding(0) @uniform
let uniforms: Uniforms
```

### For vertex inputs:
```almide
@gpu(vertex)
fn vs_main(
  @location(0) position: Vec3,
  @location(1) color: Vec3,
) -> VertexOutput = { ... }
```

### Rationale:
- Uniforms are **shared state** (multiple functions read them) → module-level
- Vertex inputs are **per-invocation** → function parameters
- This matches WGSL's own model exactly
- `@uniform` on `let` is natural: uniforms are read-only from the shader's perspective
- Future: `@storage(read_write)` on `var` for compute shader buffers

## Vec/Mat Types

### Option: Named types resolved by WGSL emitter

```almide
type Vec3 = { x: Float, y: Float, z: Float }
type Vec4 = { x: Float, y: Float, z: Float, w: Float }
type Mat4 = { /* 16 floats */ }
```

This is wrong — WGSL vec/mat are primitive types with hardware support,
not structs.

### Better: Built-in type names that the WGSL emitter maps

The type checker accepts `Vec3`, `Vec4`, `Mat4` as `Named` types.
The WGSL emitter maps them:
- `Vec2` → `vec2<f32>`
- `Vec3` → `vec3<f32>`
- `Vec4` → `vec4<f32>`
- `Mat4` → `mat4x4<f32>`

This is what emit_wgsl already does. The missing piece is that the
type checker rejects unknown `Named` types. Solutions:
1. Register Vec2/Vec3/Vec4/Mat4 as known types in the checker (when target=wgsl)
2. Use lumen as a dependency (lumen defines these types)
3. Allow `@gpu` functions to use unresolved Named types (lenient checking)

**For Phase 0.3**: option 3 (lenient) is fastest, option 2 (lumen) is cleanest long-term.

## Frame Arena Implications

The uniform/storage declaration model directly feeds into Frame Arena:

```almide
// Frame-scoped (default): buffer recycled each frame
@group(0) @binding(0) @uniform
let mvp: Mat4

// Persistent: buffer survives across frames
@persistent @group(0) @binding(1) @storage(read)
let glyph_cache: [PathSegment]
```

The `@persistent` annotation from the earlier design session maps naturally
to this model. Without `@persistent`, GPU variables follow the frame arena.

## Compute Shader Implications (Phase 0.5+)

```almide
@group(0) @binding(0) @storage(read)
let prev: [Particle]

@group(0) @binding(1) @storage(read_write)
var next: [Particle]

@gpu(compute, workgroup = [256, 1, 1])
fn simulate(@uniform dt: Float) = {
  let id = global_invocation_id().x
  let p = prev[id]
  next[id] = Particle {
    pos: p.pos + p.vel * dt,
    vel: p.vel + Vec2 { x: 0.0, y: -9.8 } * dt,
    life: p.life - dt,
  }
}
```

The @read/@write/@uniform param annotations from the earlier design
are actually **syntactic sugar** for the module-level declarations.
The compiler can:
1. See `@uniform dt: Float` in a compute function
2. Auto-generate the `@group/@binding` module-level declaration
3. Or require the user to declare them explicitly (more control)

## Decision Needed

Before implementing, decide:

1. **Module-level vs param-only for uniforms**: explicit `let uniforms` at module
   level, or implicit via `@uniform` param annotation on the function?
2. **Vec/Mat types**: lumen dependency, built-in registration, or lenient checking?
3. **@group/@binding numbering**: manual (user assigns), or auto-assigned by compiler?

## Next Steps

1. Pick a design direction
2. Implement the minimal parser/checker changes
3. Write cube's main.almd
4. Compile and verify against cube.wgsl
