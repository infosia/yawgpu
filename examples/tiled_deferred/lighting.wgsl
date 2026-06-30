// Tiled deferred — lighting pass (subpass 1).
//
// Reads the G-buffer (albedo + normal) written by subpass 0 as INPUT ATTACHMENTS:
// on Metal these lower to `[[color(N)]]` programmable-blend tile reads, on Vulkan
// to `SubpassData` INPUT_ATTACHMENT descriptors — both reading tile memory, never
// a round-trip through main memory. Writes the shaded result to the final target.
//
// Portable contract: the fragment writes its GLOBAL color slot via `@location(2)`;
// `fragment.targets` carries only that written slot (the input attachments are
// supplied by the core per backend), and no bind group is set for the
// input-attachment-only group 0.

enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var g_albedo: input_attachment<f32>;
@group(0) @binding(1) @input_attachment_index(1) var g_normal: input_attachment<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(2) vec4<f32> {
  let albedo = inputAttachmentLoad(g_albedo).rgb;
  let normal = inputAttachmentLoad(g_normal).rgb * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
  // Simple directional light + ambient.
  let light_dir = normalize(vec3<f32>(0.4, 0.6, 1.0));
  let diffuse = max(0.0, dot(normalize(normal), light_dir));
  let shade = albedo * (0.25 + 0.75 * diffuse);
  return vec4<f32>(shade, 1.0);
}
