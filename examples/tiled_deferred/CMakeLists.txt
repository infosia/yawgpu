add_executable(tiled_deferred main.c)
yawgpu_add_example(tiled_deferred)

target_include_directories(tiled_deferred PRIVATE "${CMAKE_SOURCE_DIR}/capture")
target_compile_definitions(tiled_deferred PRIVATE STB_IMAGE_WRITE_IMPLEMENTATION)

foreach(shader gbuffer.wgsl lighting.wgsl composite.wgsl)
    add_custom_command(TARGET tiled_deferred POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            "$<TARGET_FILE_DIR:tiled_deferred>/${shader}"
    )
endforeach()
