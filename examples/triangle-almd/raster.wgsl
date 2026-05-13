// snaidhm — Minimal path rasterizer (no shadows/paints, for Almide demo)

const TILE_SIZE: u32 = 16u;
const MAX_SEGS_PER_TILE: u32 = 64u;

struct LineSeg {
  p0: vec2<f32>, p1: vec2<f32>,
  color: vec4<f32>,
  path_id: u32, _p0: u32, _p1: u32, _p2: u32,
}

struct Params {
  width: u32, height: u32, seg_count: u32,
  tiles_x: u32, tiles_y: u32,
  _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read>       segments: array<LineSeg>;
@group(0) @binding(1) var<storage, read>       tile_counts: array<u32>;
@group(0) @binding(2) var<storage, read>       tile_seg_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;

fn seg_area(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
  let y = p0.y;
  let delta = p1 - p0;
  let y0 = clamp(y, 0.0, 1.0);
  let y1 = clamp(y + delta.y, 0.0, 1.0);
  let dy = y0 - y1;
  if abs(dy) < 1e-9 { return 0.0; }
  let inv_dy = 1.0 / delta.y;
  let t0 = (y0 - y) * inv_dy;
  let t1 = (y1 - y) * inv_dy;
  let x0 = p0.x + t0 * delta.x;
  let x1 = p0.x + t1 * delta.x;
  let xmin = min(min(x0, x1), 1.0) - 1e-6;
  let xmax = max(x0, x1);
  let b = min(xmax, 1.0);
  let c = max(b, 0.0);
  let d = max(xmin, 0.0);
  let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
  return (1.0 - a) * dy;
}

fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  return u32(clamp(r*255.0,0.0,255.0)) | (u32(clamp(g*255.0,0.0,255.0))<<8u) |
         (u32(clamp(b*255.0,0.0,255.0))<<16u) | (u32(clamp(a*255.0,0.0,255.0))<<24u);
}

@compute @workgroup_size(16, 16)
fn fine(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
  let px = gid.x; let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }

  let tile_id = wg.y * params.tiles_x + wg.x;
  let count = min(tile_counts[tile_id], MAX_SEGS_PER_TILE);
  let tile_base = tile_id * MAX_SEGS_PER_TILE;

  let ndc_to_px = 0.5 * f32(params.width);
  let ndc_to_py = 0.5 * f32(params.height);
  let px_f = f32(px); let py_f = f32(py);

  var color = vec3<f32>(0.12, 0.12, 0.18);
  var area = 0.0;
  var current_path = 0xFFFFFFFFu;
  var current_color = vec4<f32>(0.0);

  for (var i = 0u; i < count; i++) {
    let seg = segments[tile_seg_ids[tile_base + i]];
    if (seg.path_id != current_path) {
      let cov = min(abs(area), 1.0);
      if cov > 1e-4 {
        color = mix(color, current_color.rgb, cov * current_color.a);
      }
      area = 0.0;
      current_path = seg.path_id;
      current_color = seg.color;
    }
    let sp0 = vec2<f32>((seg.p0.x+1.0)*ndc_to_px - px_f, (1.0-seg.p0.y)*ndc_to_py - py_f);
    let sp1 = vec2<f32>((seg.p1.x+1.0)*ndc_to_px - px_f, (1.0-seg.p1.y)*ndc_to_py - py_f);
    area += seg_area(sp0, sp1);
  }
  // Last path
  let last_cov = min(abs(area), 1.0);
  if last_cov > 1e-4 {
    color = mix(color, current_color.rgb, last_cov * current_color.a);
  }

  pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, 1.0);
}

// ── Fullscreen quad ──
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>,6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(-1.0,1.0),vec2(-1.0,1.0),vec2(1.0,-1.0),vec2(1.0,1.0));
  var uv = array<vec2<f32>,6>(vec2(0.0,1.0),vec2(1.0,1.0),vec2(0.0,0.0),vec2(0.0,0.0),vec2(1.0,1.0),vec2(1.0,0.0));
  var out: VSOut; out.pos = vec4<f32>(p[i],0.0,1.0); out.uv = uv[i]; return out;
}

@group(0) @binding(0) var<storage, read> render_pixels: array<u32>;
@group(0) @binding(1) var<uniform> render_params: Params;

@fragment fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let px = u32(uv.x * f32(render_params.width));
  let py = u32(uv.y * f32(render_params.height));
  let c = render_pixels[py * render_params.width + px];
  return vec4<f32>(f32(c&0xFFu)/255.0, f32((c>>8u)&0xFFu)/255.0, f32((c>>16u)&0xFFu)/255.0, 1.0);
}
