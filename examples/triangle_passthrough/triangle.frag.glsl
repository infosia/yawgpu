// SPIR-V passthrough demo — fragment stage.
//
// Hand-written GLSL that compiles via `glslangValidator -V` to
// `triangle.frag.spv`. The entry-point name baked into the SPIR-V is
// "main" (glslang's default); the pipeline descriptor's fragment
// `entryPoint` must therefore be "main".
//
// To regenerate the .spv:
//
//   glslangValidator -V -o triangle.frag.spv triangle.frag.glsl

#version 450

layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(v_color, 1.0);
}
