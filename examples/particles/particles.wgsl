// Phase 0.5: 10K Particles — compute shader simulation + vertex rendering

struct Particle {
  pos: vec2<f32>,
  vel: vec2<f32>,
  life: f32,
  _pad: f32,  // align to 16 bytes
}

struct SimParams {
  dt: f32,
  count: u32,
  seed: f32,
  _pad: f32,
}

// Compute: simulate particles
@group(0) @binding(0) var<storage, read>       prev: array<Particle>;
@group(0) @binding(1) var<storage, read_write> next: array<Particle>;
@group(0) @binding(2) var<uniform>             params: SimParams;

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= params.count) { return; }

  var p = prev[id];
  let gravity = vec2<f32>(0.0, -2.0);

  p.vel = p.vel + gravity * params.dt;
  p.pos = p.pos + p.vel * params.dt;
  p.life = p.life - params.dt;

  // Respawn dead particles at top with random horizontal position
  if (p.life <= 0.0) {
    let hash = fract(sin(f32(id) * 43758.5453 + params.seed) * 43758.5453);
    p.pos = vec2<f32>(hash * 2.0 - 1.0, 1.0);
    p.vel = vec2<f32>((hash - 0.5) * 0.5, -0.5 - hash * 0.5);
    p.life = 2.0 + hash * 3.0;
  }

  next[id] = p;
}

// Render: draw particles as points
struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) alpha: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;

@vertex
fn vs_render(@builtin(vertex_index) idx: u32) -> VertexOutput {
  let p = particles[idx];
  var out: VertexOutput;
  out.pos = vec4<f32>(p.pos, 0.0, 1.0);
  out.alpha = clamp(p.life / 3.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs_render(@location(0) alpha: f32) -> @location(0) vec4<f32> {
  return vec4<f32>(0.3, 0.7, 1.0, alpha);
}
