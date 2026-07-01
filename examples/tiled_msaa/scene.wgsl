// tiled_msaa — scene pass (subpass 0).
//
// Draws thin colored lines radiating from the centre into a 4x MSAA color
// attachment. Thin near-diagonal lines are where MSAA is most visible: without
// it their edges stair-step badly; with it (resolved by the later subpasses) they
// are smooth. The line endpoints/colours are generated procedurally from the
// vertex index — no vertex buffer — so subpass 0 is drawn as a LINE LIST of
// `LINE_COUNT * 2` vertices (see main.c, which must match `LINE_COUNT`).
//
// The attachment is a pure subpass intermediate: written here and consumed as a
// *multisampled* input attachment by the per-sample subpass — on a TBDR GPU it
// never leaves tile memory.

const LINE_COUNT: u32 = 60u;
const PI: f32 = 3.14159265;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let line = vi / 2u;
  let is_outer = (vi % 2u) == 1u;
  let t = f32(line) / f32(LINE_COUNT);
  let angle = t * 2.0 * PI;
  let dir = vec2<f32>(cos(angle), sin(angle));

  var out: VsOut;
  if (is_outer) {
    out.pos = vec4<f32>(dir * 0.9, 0.0, 1.0);
    out.color = vec3<f32>(cos(angle) * 0.5 + 0.5, sin(angle) * 0.5 + 0.5, 1.0);
  } else {
    out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.color = vec3<f32>(1.0, sin(angle) * 0.5 + 0.5, cos(angle) * 0.5 + 0.5);
  }
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
