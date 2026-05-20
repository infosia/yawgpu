add_executable(triangle main.c)
yawgpu_add_example(triangle)

add_custom_command(TARGET triangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/shader.wgsl"
        "$<TARGET_FILE_DIR:triangle>/shader.wgsl"
)
