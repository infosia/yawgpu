// Deferred lighting pass: reads G-Buffer via input attachments.
// Reconstructs world position from depth + inverse view-projection.
// Blinn-Phong shading with 4 orbiting point lights.
// Outputs HDR color to an intermediate Rgba16Float target (tonemapping in next subpass).
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

// Input attachments: read from tile memory (G-Buffer output from subpass 0)
@group(0) @binding(0) var t_albedo: subpass_input<f32>;
@group(0) @binding(1) var t_normal: subpass_input<f32>;

struct LightParams {
    lights: array<vec4<f32>, 4>,   // xyz = position, w = intensity
    camera_pos: vec3<f32>,
    time: f32,
    inv_view_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
}

@group(1) @binding(0) var<uniform> light: LightParams;

fn shade_lighting(in: VertexOutput) -> vec4<f32> {
    let albedo_sample = subpassLoad(t_albedo);

    // Use albedo alpha to detect background (0.0 = background, 1.0 = geometry)
    if albedo_sample.a < 0.5 {
        return vec4<f32>(0.02, 0.02, 0.04, 1.0);
    }

    let normal_sample = subpassLoad(t_normal);
    let albedo = albedo_sample.rgb;
    let raw_normal = normal_sample.xyz;
    // Guard against zero-length normal (e.g. cleared but not written).
    let normal = select(normalize(raw_normal), vec3<f32>(0.0, 1.0, 0.0), dot(raw_normal, raw_normal) < 0.0001);
    let depth = normal_sample.a;

    // Reconstruct world position from screen coords + stored depth
    let uv = in.position.xy / light.screen_size;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = light.inv_view_proj * ndc;
    // Guard against w ≈ 0 (point at infinity / degenerate inverse).
    let safe_w = select(world_h.w, 1.0, abs(world_h.w) < 0.0001);
    let world_pos = world_h.xyz / safe_w;

    // Hemisphere ambient
    let sky_color = vec3<f32>(0.12, 0.15, 0.25);
    let ground_color = vec3<f32>(0.04, 0.03, 0.02);
    let hemisphere = normal.y * 0.5 + 0.5;
    var result = mix(ground_color, sky_color, hemisphere) * albedo;

    let to_camera = light.camera_pos - world_pos;
    let cam_dist = max(length(to_camera), 0.0001);
    let view_dir = to_camera / cam_dist;

    // 4 point lights with Blinn-Phong and distance attenuation
    for (var i = 0u; i < 4u; i = i + 1u) {
        let light_pos = light.lights[i].xyz;
        let intensity = light.lights[i].w;

        let to_light = light_pos - world_pos;
        let dist = max(length(to_light), 0.0001);
        let light_dir = to_light / dist;
        let attenuation = intensity / (1.0 + dist * dist);

        // Diffuse (Lambertian)
        let diff = max(dot(normal, light_dir), 0.0);

        // Specular (Blinn-Phong)
        let half_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(normal, half_dir), 0.0), 64.0);

        let light_color = vec3<f32>(1.0, 0.95, 0.85);

        result = result + albedo * diff * light_color * attenuation;
        result = result + spec * light_color * attenuation * 0.3;
    }

    // Output HDR — tonemapping happens in the composite subpass
    return vec4<f32>(result, 1.0);
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade_lighting(in);
}

@fragment
fn fs_metal(in: VertexOutput) -> @location(2) vec4<f32> {
    return shade_lighting(in);
}
