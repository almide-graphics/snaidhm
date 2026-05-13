// snaidhm Phase 1 — Tiled path renderer
//
// CPU: flatten + coarse tile assignment (segments sorted by path_id within each tile)
// GPU: fine rasterize (per-tile winding number fill)

const TILE_SIZE: u32 = 16u;
const MAX_SEGS_PER_TILE: u32 = 512u;

struct LineSeg {
  p0: vec2<f32>,
  p1: vec2<f32>,
  color: vec4<f32>,
  path_id: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

struct Params {
  width: u32,
  height: u32,
  seg_count: u32,
  tiles_x: u32,
  tiles_y: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

// Fine rasterize: per-tile winding fill
@group(0) @binding(0) var<storage, read>       segments: array<LineSeg>;
@group(0) @binding(1) var<storage, read>       tile_counts: array<u32>;
@group(0) @binding(2) var<storage, read>       tile_seg_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;

fn wind_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let e = b - a;
  let w = p - a;
  if (a.y <= p.y) {
    if (b.y > p.y) {
      if (e.x * w.y - e.y * w.x > 0.0) { return 1.0; }
    }
  } else {
    if (b.y <= p.y) {
      if (e.x * w.y - e.y * w.x < 0.0) { return -1.0; }
    }
  }
  return 0.0;
}

fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  let ri = u32(clamp(r * 255.0, 0.0, 255.0));
  let gi = u32(clamp(g * 255.0, 0.0, 255.0));
  let bi = u32(clamp(b * 255.0, 0.0, 255.0));
  let ai = u32(clamp(a * 255.0, 0.0, 255.0));
  return ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

@compute @workgroup_size(16, 16)
fn fine(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }

  let tile_id = wg.y * params.tiles_x + wg.x;
  let count = min(tile_counts[tile_id], MAX_SEGS_PER_TILE);
  let tile_base = tile_id * MAX_SEGS_PER_TILE;

  let p = vec2<f32>(
    f32(px) / f32(params.width) * 2.0 - 1.0,
    1.0 - f32(py) / f32(params.height) * 2.0,
  );

  var color = vec3<f32>(0.95, 0.95, 0.97);
  var winding = 0.0;
  var current_path = 0xFFFFFFFFu;
  var current_color = vec4<f32>(0.0);

  for (var i = 0u; i < count; i++) {
    let seg_idx = tile_seg_ids[tile_base + i];
    let seg = segments[seg_idx];

    if (seg.path_id != current_path) {
      if (abs(winding) > 0.5) {
        color = mix(color, current_color.rgb, current_color.a);
      }
      winding = 0.0;
      current_path = seg.path_id;
      current_color = seg.color;
    }

    winding += wind_line(p, seg.p0, seg.p1);
  }

  if (abs(winding) > 0.5) {
    color = mix(color, current_color.rgb, current_color.a);
  }

  pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, 1.0);
}

// ── Fullscreen quad ──

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VSOut {
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
@group(0) @binding(1) var<uniform>       render_params: Params;

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let px = u32(uv.x * f32(render_params.width));
  let py = u32(uv.y * f32(render_params.height));
  let idx = py * render_params.width + px;
  let packed = render_pixels[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec4<f32>(r, g, b, 1.0);
}
