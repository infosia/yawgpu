add_library(framework framework.c)

target_include_directories(framework PUBLIC
    "${CMAKE_SOURCE_DIR}/framework"
    "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
)

if(MSVC)
    target_compile_options(framework PRIVATE /W4)
else()
    target_compile_options(framework PRIVATE -Wall -Wextra -Wpedantic)
endif()
