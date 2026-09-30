// Glyph-cache text with the atlas in a buffer (snaidhm text.almd,
// `new_buffer_layer`)
//
// text.wgsl, reading coverage from a storage buffer instead of a texture: the
// 1024 x 1024 atlas one byte per pixel, four to a word, rows 256 words long.
// Quads sit on whole pixels, so a fragment's uv falls inside exactly one atlas
// pixel — what nearest sampling of the texture gave.
//
// Why a buffer: on Metal, writing a texture from the CPU is a blit, and the
// first blit makes the driver set up ~50 MB of GPU memory. A buffer is written
// in place.

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> atlas: array<u32>;

const ATLAS: f32 = 1024.0;

// Vertex data: pos (normal space) 2 + atlas uv 2 + colour 4 = 8 floats.
@vertex
fn vs_main(
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = uv;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let p = vec2<u32>(clamp(floor(in.uv * ATLAS), vec2<f32>(0.0), vec2<f32>(ATLAS - 1.0)));
  let i = p.y * u32(ATLAS) + p.x;
  let coverage = f32((atlas[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu) / 255.0;
  return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
