// Tiled deferred — G-buffer pass (subpass 0).
//
// A full-screen triangle writes a procedural G-buffer: an "albedo" colour and a
// "normal" (here a simple radial normal), to two color attachments that the
// lighting subpass then reads back as input attachments (tile memory — never
// written to main memory between subpasses on a TBDR GPU).

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  // Full-screen triangle in clip space; uv spans [0,1] over the framebuffer.
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  var out: VsOut;
  out.pos = vec4<f32>(p[i], 0.0, 1.0);
  out.uv = (p[i] * 0.5) + vec2<f32>(0.5, 0.5);
  return out;
}

struct GBuffer {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
}

@fragment
fn fs(in: VsOut) -> GBuffer {
  // Albedo: a smooth colour gradient across the frame.
  let albedo = vec3<f32>(in.uv.x, in.uv.y, 1.0 - in.uv.x);
  // Normal: a hemisphere pointing out of a centred bump (encoded to [0,1]).
  let d = in.uv - vec2<f32>(0.5, 0.5);
  let z = sqrt(max(0.0, 1.0 - dot(d, d) * 4.0));
  let n = normalize(vec3<f32>(d.x, d.y, z + 0.25));
  var out: GBuffer;
  out.albedo = vec4<f32>(albedo, 1.0);
  out.normal = vec4<f32>(n * 0.5 + vec3<f32>(0.5, 0.5, 0.5), 1.0);
  return out;
}
