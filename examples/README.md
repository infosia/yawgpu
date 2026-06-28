# yawgpu C examples

These examples are a separate CMake tree and are not Cargo workspace members.
Select a backend at runtime with `YAWGPU_BACKEND=noop|metal|vulkan`, for
example `YAWGPU_BACKEND=metal ./examples/build/compute/compute`.

## macOS

Install prerequisites with `brew install cmake glfw`, then configure and build:

```sh
cmake -S examples -B examples/build
cmake --build examples/build
```

Windowed examples use GLFW plus `CAMetalLayer` on macOS.

## Windows

Install the MSVC toolchain, CMake, LLVM with `LIBCLANG_PATH` pointing at the
LLVM `bin` directory, and the Vulkan SDK plus a Vulkan-capable GPU driver/ICD.
Configure the examples with the Vulkan feature:

```powershell
cmake -S examples -B examples/build -DYAWGPU_FEATURE=vulkan
cmake --build examples/build
```

Run Vulkan examples with:

```powershell
$env:YAWGPU_BACKEND = "vulkan"
examples\build\triangle\Debug\triangle.exe
```

The CMake build copies `yawgpu.dll` next to each example executable on Windows.
If running binaries from another directory, put the corresponding
`target-vulkan\debug` or `target-vulkan\release` directory on `PATH`, or copy
`yawgpu.dll` beside the executable. Windows windowed examples use native Win32
windowing and do not require GLFW; macOS/Linux windowed examples still require
GLFW, and Linux windowed examples are not enabled in this phase.

`capture` renders a solid color to an offscreen RGBA8 texture, copies it via T2B to a readback buffer, and writes `red.png` in the binary's current working directory.

`tiled_deferred` demonstrates yawgpu's tiled extension from C: it records a two-subpass offscreen pass, reads a G-buffer through a subpass input, copies the persistent output texture to a readback buffer, validates the center pixel, and writes `tiled_deferred.png`. Build it with `-DYAWGPU_EXTENSIONS=tiled` on a backend with native subpass-input reads such as Metal or Vulkan; MoltenVK does not support this path. Without that extension the example builds as a stub that prints `tiled extension not enabled`.

`surface_smoke` opens a window, clears the swapchain to a slate color for about 60 frames or until the window is closed, and then exits.

`triangle` opens a window, draws an RGB-corner gradient triangle (red / green / blue at the three vertices, smoothly interpolated across the surface) on a black background for about 60 frames or until the window is closed, and then exits with status 0.

`hello_triangle` is the Dawn HelloTriangle port and has the same prerequisites as `triangle`. It draws the same RGB-corner gradient triangle for about 60 frames, but feeds positions **and** per-vertex colors from a real (interleaved) vertex buffer instead of deriving them in the shader from P9.3's `@builtin(vertex_index)`.

`triangle_passthrough` draws the same RGB-corner gradient triangle, but feeds the GPU **native shader bytecode** instead of WGSL, through yawgpu's opt-in `shader-passthrough` vendor feature: hand-written **SPIR-V** (`triangle.{vert,frag}.spv`, compiled from the bundled GLSL with `glslangValidator -V`) on Vulkan, and hand-written **MSL** (`triangle.msl`) on Metal. It is **opt-in** — configure with `-DYAWGPU_SHADER_PASSTHROUGH=ON`, which both adds the `shader-passthrough` cargo feature to `libyawgpu` and enables this example:

```sh
cmake -S examples -B examples/build -DYAWGPU_FEATURE=metal -DYAWGPU_SHADER_PASSTHROUGH=ON
cmake --build examples/build
YAWGPU_BACKEND=metal  ./examples/build/triangle_passthrough/triangle_passthrough
YAWGPU_BACKEND=vulkan ./examples/build/triangle_passthrough/triangle_passthrough   # with -DYAWGPU_FEATURE=vulkan
```

It self-skips (exit 0) on backends other than Metal / Vulkan, since passthrough has no Noop shader compiler to feed.
