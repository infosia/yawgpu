// tiled_msaa — resolve pass (subpass 2).
//
// Reads the 4x MSAA intermediate as a multisampled input attachment and averages
// its four samples in-shader (a custom resolve), writing the single-sample
// output. This produces anti-aliased edges without a hardware resolve attachment
// — the whole pass stays on-tile until this final write.
//
// This subpass rasterizes single-sampled (multisample.count = 1); it reads all
// four samples explicitly with the 2-arg `inputAttachmentLoad(hdr, i)`. Its
// pipeline still declares the input attachment `multisampled` in its explicit
// bind-group layout.

enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var hdr: input_attachment<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(2) vec4<f32> {
  var sum = vec4<f32>(0.0);
  for (var i = 0u; i < 4u; i = i + 1u) {
    sum = sum + inputAttachmentLoad(hdr, i);
  }
  return sum * 0.25;
}
