# yawgpu C examples

These examples are a separate CMake tree and are not Cargo workspace members. Install prerequisites with `brew install cmake glfw`, then configure and build with `cmake -S examples -B examples/build && cmake --build examples/build`. Select a backend at runtime with `YAWGPU_BACKEND=noop|metal|vulkan`, for example `YAWGPU_BACKEND=metal ./examples/build/compute/compute`; P9.0 examples are headless only, while GLFW/windowed examples come in later slices.
