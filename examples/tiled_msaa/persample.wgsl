// tiled_msaa — per-sample pass (subpass 1).
//
// Reads the 4x MSAA scene attachment PER SAMPLE via the 2-arg
// `inputAttachmentLoad(scene, sample_index)`. `@builtin(sample_index)` lowers to
// the SPIR-V `SampleId` built-in, which promotes this fragment to per-sample
// invocation (Vulkan `sampleRateShading`) — so it runs once per MSAA sample.
//
// A mild per-sample tint stands in for a real per-sample post-process (e.g. an
// HDR tonemap applied before the resolve, which reduces edge artefacts vs
// tonemapping after a hardware resolve). The output is itself a 4x MSAA
// attachment, consumed by the resolve subpass.
//
// This is a yawgpu vendor extension surface (Vulkan-only): declare the input
// attachment as `multisampled` in the pipeline's explicit bind-group layout —
// WGSL cannot express an input attachment's multisampled-ness in the type.

enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var scene: input_attachment<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(sample_index) s: u32) -> @location(1) vec4<f32> {
  let c = inputAttachmentLoad(scene, s);
  let tint = 0.90 + 0.10 * f32(s) / 3.0;
  return vec4<f32>(c.rgb * tint, c.a);
}
