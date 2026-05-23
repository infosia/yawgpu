// G-Buffer pass: outputs albedo color and world-space normal to two render targets.
// Supports instanced rendering of a cube grid via instance_index.
//
// Albedo (Rgba8Unorm):   RGB = color, A = 1.0 (geometry flag)
// Normal (Rgba16Float):  RGB = world normal [-1,1], A = clip-space depth for reconstruction

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
    // 5x5 grid layout: compute row/col from instance_index
    let grid_size = 5u;
    let spacing = 3.0;
    let half_grid = f32(grid_size - 1u) * spacing * 0.5;

    let col = instance % grid_size;
    let row = instance / grid_size;
    let offset = vec3<f32>(
        f32(col) * spacing - half_grid,
        0.0,
        f32(row) * spacing - half_grid,
    );

    let world_pos = in.position + offset;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    return out;
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;
    out.albedo = vec4<f32>(in.color, 1.0);
    // Store normal directly in RGB (Rgba16Float supports [-1, 1] natively).
    // Store window-space depth in alpha for world-position reconstruction.
    out.normal = vec4<f32>(in.world_normal, in.clip_position.z);
    return out;
}
