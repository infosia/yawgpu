add_executable(compute main.c)
yawgpu_add_example(compute)

add_custom_command(TARGET compute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/shader.wgsl"
        "$<TARGET_FILE_DIR:compute>/shader.wgsl"
)
