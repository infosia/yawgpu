add_executable(hello_triangle main.c)
yawgpu_add_example(hello_triangle)

add_custom_command(TARGET hello_triangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/shader.wgsl"
        "$<TARGET_FILE_DIR:hello_triangle>/shader.wgsl"
)
