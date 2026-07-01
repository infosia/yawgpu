// tiled_msaa — scene pass (subpass 0).
//
// A centred triangle drawn into a 4x MSAA color attachment. Its diagonal edges
// alias badly without MSAA. The attachment is a pure subpass intermediate: it is
// written here and consumed as a *multisampled* input attachment by the
// per-sample subpass — on a TBDR GPU it never leaves tile memory.

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(0.0, 0.75), vec2(-0.75, -0.6), vec2(0.75, -0.6));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.55, 0.15, 1.0); // warm orange on the cleared background
}
