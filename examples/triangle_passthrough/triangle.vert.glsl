// SPIR-V passthrough demo — vertex stage.
//
// Hand-written GLSL that compiles via `glslangValidator -V` to
// `triangle.vert.spv`. yawgpu's SPIR-V passthrough does NOT reflect the
// SPIR-V (unlike the WGSL-to-SPIR-V path), so the entry-point name
// recorded in the SPIR-V module ("main" by default for glslang) is what
// the pipeline descriptor must reference at `entryPoint`.
//
// To regenerate the .spv:
//
//   glslangValidator -V -o triangle.vert.spv triangle.vert.glsl
//
// (The pre-compiled .spv is checked in alongside this source so the
// example builds without a Vulkan SDK install.)

#version 450

void main() {
    // Three positions derived from gl_VertexIndex, matching the WGSL
    // reference (`@vertex fn vs_main(@builtin(vertex_index) i): @builtin(position)`):
    //   vertex 0 -> (-1, -1), vertex 1 -> (0, +1), vertex 2 -> (+1, -1).
    // The Y component is negated below to compensate for Vulkan's
    // clip-space convention (+Y points down in clip space, vs WebGPU /
    // Metal where +Y points up). With the negation, all three backends
    // display the same triangle pointing up.
    float x = float(int(gl_VertexIndex) - 1);
    float y = float(int(gl_VertexIndex & 1) * 2 - 1);
    gl_Position = vec4(x, -y, 0.0, 1.0);
}
