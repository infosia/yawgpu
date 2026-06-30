add_executable(tiled_deferred main.c)
# YAWGPU_HAS_TILED exposes the tiled / multi-subpass vendor C ABI in yawgpu.h.
target_compile_definitions(tiled_deferred PRIVATE STB_IMAGE_WRITE_IMPLEMENTATION YAWGPU_HAS_TILED)
target_include_directories(tiled_deferred PRIVATE "${CMAKE_SOURCE_DIR}/capture")
yawgpu_add_example(tiled_deferred)

# Copy the WGSL shaders next to the built binary so it runs from any cwd.
foreach(shader gbuffer.wgsl lighting.wgsl)
    add_custom_command(TARGET tiled_deferred POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            "$<TARGET_FILE_DIR:tiled_deferred>/${shader}"
    )
endforeach()
