add_library(framework framework.c)

if(YAWGPU_GLFW_FOUND AND APPLE)
    target_sources(framework PRIVATE framework_macos.m)
endif()

if(WIN32)
    target_sources(framework PRIVATE framework_windows.c)
endif()

target_include_directories(framework PUBLIC
    "${CMAKE_SOURCE_DIR}/framework"
    "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
)
if(YAWGPU_EXTENSION_COMPILE_DEFS)
    target_compile_definitions(framework PUBLIC ${YAWGPU_EXTENSION_COMPILE_DEFS})
endif()

if(MSVC)
    target_compile_options(framework PRIVATE /W4)
else()
    target_compile_options(framework PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(YAWGPU_GLFW_FOUND)
    target_link_libraries(framework PUBLIC glfw)
endif()

if(WIN32)
    target_link_libraries(framework PUBLIC user32 gdi32)
endif()
