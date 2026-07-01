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

The CMake build copies `yawgpu.dll` (and, for the default Tint-linked build,
its `tint_shim.dll` dependency) next to each example executable on Windows.
If running binaries from another directory, put the corresponding
`target-vulkan\debug` or `target-vulkan\release` directory on `PATH`, or copy
both `yawgpu.dll` and `tint_shim.dll` beside the executable. Windows windowed examples use native Win32
windowing and do not require GLFW; macOS/Linux windowed examples still require
GLFW, and Linux windowed examples are not enabled in this phase.

`capture` renders a solid color to an offscreen RGBA8 texture, copies it via T2B to a readback buffer, and writes `red.png` in the binary's current working directory.

`tiled_deferred` demonstrates yawgpu's `tiled` (TBDR / multi-subpass) vendor extension from C. It records a **two-subpass** offscreen pass: subpass 0 writes a G-buffer (albedo + a packed normal) to two color attachments, then subpass 1 reads them back as **input attachments** (Metal `[[color(N)]]` programmable-blend tile reads; Vulkan `SubpassData` `INPUT_ATTACHMENT` descriptors — tile memory, never a round-trip through main memory) and writes the shaded result. By default it **opens a window** and presents the shaded result every frame; with `--verify` it instead renders one frame offscreen, copies the final target to a readback buffer, prints the center pixel, and writes `tiled_deferred.png`. The same C code + shaders run on **both Metal and Vulkan** (including MoltenVK, which executes genuine multi-subpass input attachments) and produce identical output — the portable contract: the fragment writes its global `@location`, `fragment.targets` lists only the written slots, and no bind group is set for the input-attachment-only group. Opt-in (off by default), and — like the other windowed examples — needs GLFW:

```sh
cmake -S examples -B examples/build -DYAWGPU_FEATURE=metal  -DYAWGPU_TILED=ON   # or -DYAWGPU_FEATURE=vulkan
cmake --build examples/build --target tiled_deferred
YAWGPU_BACKEND=metal  ./examples/build/tiled_deferred/tiled_deferred            # windowed
YAWGPU_BACKEND=metal  ./examples/build/tiled_deferred/tiled_deferred --verify   # offscreen + tiled_deferred.png
```

`-DYAWGPU_TILED=ON` adds the `tiled` cargo feature to `libyawgpu` and enables this example.

`tiled_msaa` demonstrates **per-sample MSAA subpass input** (a `tiled` vendor surface, **Vulkan-only**). It records a **three-subpass** pass: subpass 0 draws a centred triangle into a 4× MSAA color attachment (its diagonal edges alias without MSAA); subpass 1 reads that attachment **per sample** via the 2-arg `inputAttachmentLoad(scene, @builtin(sample_index))` — `sample_index` lowers to SPIR-V `SampleId`, promoting the fragment to per-sample invocation (Vulkan `sampleRateShading`) — applies a per-sample tint, and writes a 4× MSAA intermediate; subpass 2 reads that intermediate as a multisampled input, averages its four samples in-shader (a **custom resolve**), and writes the single-sample, anti-aliased output. The multisampled input attachments are declared via an **explicit pipeline layout** whose input-attachment binding is `multisampled` (WGSL cannot express an input attachment's multisampled-ness in the type). By default it opens a window; with `--verify` it renders one frame offscreen, prints the center pixel, and writes `tiled_msaa.png`. It errors on non-Vulkan backends. Opt-in via `-DYAWGPU_TILED=ON`; needs GLFW like the other windowed examples:

```sh
cmake -S examples -B examples/build -DYAWGPU_FEATURE=vulkan -DYAWGPU_TILED=ON
cmake --build examples/build --target tiled_msaa
YAWGPU_BACKEND=vulkan ./examples/build/tiled_msaa/tiled_msaa            # windowed
YAWGPU_BACKEND=vulkan ./examples/build/tiled_msaa/tiled_msaa --verify   # offscreen + tiled_msaa.png
```

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
