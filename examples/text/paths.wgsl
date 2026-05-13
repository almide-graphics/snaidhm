// snaidhm — Tiled path renderer with backdrop propagation
//
// Inspired by Vello (linebender/vello):
// - Backdrop prefix sum propagates winding across tile columns
// - Fine pass uses backdrop as initial winding per path

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
  path_count: u32,
  _pad1: u32,
  _pad2: u32,
}

// ── Backdrop prefix sum (Vello-style) ──
// One workgroup per tile row. Each thread handles one tile column.
// Computes inclusive prefix sum of backdrop deltas across the row,
// separately per path.

@group(0) @binding(0) var<storage, read_write> backdrops: array<i32>;
@group(0) @binding(1) var<uniform>             backdrop_params: Params;

const BACKDROP_WG: u32 = 64u;
var<workgroup> sh_backdrop: array<i32, 64u>;

@compute @workgroup_size(64)
fn backdrop(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
) {
  // wg_id.x = tile_row * path_count + path_index
  let path_idx = wg_id.x % backdrop_params.path_count;
  let tile_row = wg_id.x / backdrop_params.path_count;
  let tiles_x = backdrop_params.tiles_x;

  let tile_col = local_id.x;
  let tile_id = tile_row * tiles_x + tile_col;
  let flat_idx = tile_id * backdrop_params.path_count + path_idx;

  var val: i32 = 0;
  if (tile_col < tiles_x) {
    val = backdrops[flat_idx];
  }
  sh_backdrop[local_id.x] = val;

  // Inclusive prefix sum (Hillis-Steele)
  for (var stride = 1u; stride < BACKDROP_WG; stride <<= 1u) {
    workgroupBarrier();
    if (local_id.x >= stride) {
      val += sh_backdrop[local_id.x - stride];
    }
    workgroupBarrier();
    sh_backdrop[local_id.x] = val;
  }

  if (tile_col < tiles_x) {
    backdrops[flat_idx] = val;
  }
}

// ── Fine rasterize ──

@group(0) @binding(0) var<storage, read>       segments: array<LineSeg>;
@group(0) @binding(1) var<storage, read>       tile_counts: array<u32>;
@group(0) @binding(2) var<storage, read>       tile_seg_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;
@group(0) @binding(5) var<storage, read>       fine_backdrops: array<i32>;

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

fn eval_winding_at(p: vec2<f32>, tile_base: u32, count: u32, target_path: u32) -> f32 {
  var winding = 0.0;
  for (var i = 0u; i < count; i++) {
    let seg = segments[tile_seg_ids[tile_base + i]];
    if (seg.path_id == target_path) {
      winding += wind_line(p, seg.p0, seg.p1);
    }
  }
  return winding;
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
  let path_count = params.path_count;

  let px_ndc = f32(px) / f32(params.width) * 2.0 - 1.0;
  let py_ndc = 1.0 - f32(py) / f32(params.height) * 2.0;
  let half_px = 1.0 / f32(params.width);

  // 4x supersampling offsets
  let offsets = array<vec2<f32>, 4>(
    vec2<f32>(-0.375 * half_px, -0.125 * half_px),
    vec2<f32>( 0.125 * half_px, -0.375 * half_px),
    vec2<f32>( 0.375 * half_px,  0.125 * half_px),
    vec2<f32>(-0.125 * half_px,  0.375 * half_px),
  );

  var color = vec3<f32>(0.95, 0.95, 0.97);

  // Process each unique path in this tile
  var prev_path = 0xFFFFFFFFu;
  for (var i = 0u; i < count; i++) {
    let seg = segments[tile_seg_ids[tile_base + i]];
    if (seg.path_id == prev_path) { continue; }
    prev_path = seg.path_id;

    // Get backdrop (accumulated winding from tiles to the left)
    let backdrop_idx = tile_id * path_count + seg.path_id;
    var base_winding = 0;
    if (seg.path_id < path_count) {
      base_winding = fine_backdrops[backdrop_idx];
    }

    // 4x supersampling with backdrop
    var coverage = 0.0;
    for (var s = 0u; s < 4u; s++) {
      let sp = vec2<f32>(px_ndc + offsets[s].x, py_ndc + offsets[s].y);
      let local_w = eval_winding_at(sp, tile_base, count, seg.path_id);
      let total_w = f32(base_winding) + local_w;
      if (abs(total_w) >= 0.5) { coverage += 0.25; }
    }

    if (coverage > 0.0) {
      color = mix(color, seg.color.rgb, coverage * seg.color.a);
    }
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
