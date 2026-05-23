// Composite pass: reads HDR lit result from tile memory via input attachment.
// Applies Reinhard tonemapping, outputs to sRGB swapchain (hardware gamma).
//
// Block 55's "dual convention accepted" rule allows two fragment output
// conventions for subpass pipelines. Vulkan uses the subpass-local
// @location(0), while Metal needs the flat MTL color slot directly because
// naga's MSL backend does not subpass-remap fragment outputs.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle from vertex_index
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, -y, 0.0, 1.0);
    return out;
}

// Input attachment: HDR lit color from subpass 1 (render-pass color slot 2)
@group(0) @binding(0) var t_lit_color: subpass_input<f32>;

fn composite_color(in: VertexOutput) -> vec4<f32> {
    let hdr = subpassLoad(t_lit_color).rgb;

    // Reinhard tonemapping (linear output — sRGB swapchain handles gamma)
    let mapped = hdr / (hdr + vec3<f32>(1.0));

    return vec4<f32>(mapped, 1.0);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return composite_color(in);
}

@fragment
fn fs_metal(in: VertexOutput) -> @location(3) vec4<f32> {
    return composite_color(in);
}
