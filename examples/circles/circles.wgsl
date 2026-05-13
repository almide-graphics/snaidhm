// Phase 0.7: 1000 circles — tiled SDF compute rasterizer
//
// Each workgroup = one 16×16 tile.
// Each thread = one pixel within the tile.
// For each pixel, evaluate SDF against all circles, accumulate color.

struct Circle {
  pos: vec2<f32>,     // center in NDC [-1, 1]
  radius: f32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  _pad0: f32,
  _pad1: f32,
}

struct RasterParams {
  width: u32,
  height: u32,
  circle_count: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read>       circles: array<Circle>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(2) var<uniform>             params: RasterParams;

// Pack RGBA into u32 (ABGR layout for little-endian readback)
fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  let ri = u32(clamp(r * 255.0, 0.0, 255.0));
  let gi = u32(clamp(g * 255.0, 0.0, 255.0));
  let bi = u32(clamp(b * 255.0, 0.0, 255.0));
  let ai = u32(clamp(a * 255.0, 0.0, 255.0));
  return ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

@compute @workgroup_size(16, 16)
fn rasterize(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }

  // Pixel center in NDC [-1, 1]
  let uv = vec2<f32>(
    f32(px) / f32(params.width) * 2.0 - 1.0,
    1.0 - f32(py) / f32(params.height) * 2.0,  // flip Y
  );

  // Accumulate color from all circles (back to front)
  var color = vec3<f32>(0.02, 0.02, 0.04);  // background
  var alpha_acc = 0.0;

  for (var i = 0u; i < params.circle_count; i++) {
    let c = circles[i];
    let dist = length(uv - c.pos) - c.radius;

    // Smooth edge: 1px antialiasing
    let pixel_size = 2.0 / f32(params.width);
    let aa = smoothstep(pixel_size, -pixel_size, dist);

    if (aa > 0.001) {
      let circle_color = vec3<f32>(c.color_r, c.color_g, c.color_b);
      color = mix(color, circle_color, aa * (1.0 - alpha_acc) * 0.8);
      alpha_acc = min(alpha_acc + aa * 0.8, 1.0);
    }
  }

  let idx = py * params.width + px;
  pixels[idx] = pack_color(color.x, color.y, color.z, 1.0);
}

// Fullscreen quad: render the pixel buffer to screen
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VSOut {
  // Two-triangle fullscreen quad from vertex index
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),  vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uvs[idx];
  return out;
}

@group(0) @binding(0) var<storage, read> render_pixels: array<u32>;
@group(0) @binding(1) var<uniform>       render_params: RasterParams;

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let px = u32(uv.x * f32(render_params.width));
  let py = u32(uv.y * f32(render_params.height));
  let idx = py * render_params.width + px;
  let packed = render_pixels[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;
  return vec4<f32>(r, g, b, a);
}
