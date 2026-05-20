# yawgpu C examples

These examples are a separate CMake tree and are not Cargo workspace members. Install prerequisites with `brew install cmake glfw`, then configure and build with `cmake -S examples -B examples/build && cmake --build examples/build`. Select a backend at runtime with `YAWGPU_BACKEND=noop|metal|vulkan`, for example `YAWGPU_BACKEND=metal ./examples/build/compute/compute`; P9.0 examples are headless only, while GLFW/windowed examples come in later slices.

`capture` renders a solid color to an offscreen RGBA8 texture, copies it via T2B to a readback buffer, and writes `red.png` in the binary's current working directory.

`surface_smoke` is built when GLFW is available (`brew install glfw`); it opens a window, clears the swapchain to a slate color for about 60 frames or until the window is closed, and then exits.

`triangle` is built when GLFW is available (`brew install glfw`). Select `YAWGPU_BACKEND=metal` or `YAWGPU_BACKEND=vulkan` at runtime; for Vulkan, source `$VULKAN_SDK/setup-env.sh` first. It opens a window, draws a red triangle on a black background for about 60 frames or until the window is closed, and then exits with status 0.
