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
