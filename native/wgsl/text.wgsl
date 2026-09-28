// Glyph-cache text — blit coverage from the glyph atlas (snaidhm text.almd)
//
// Each glyph was rasterized once at the size it is shown at, into the atlas's
// alpha channel as exact area coverage (glyph.almd). Quads are placed on
// whole pixels and sampled nearest, so a glyph reaches the screen as it was
// rasterized: the coverage is the alpha, the run's colour the colour.

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

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
  let coverage = textureSample(atlas, atlas_sampler, in.uv).a;
  return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
