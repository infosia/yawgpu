add_executable(triangle_passthrough main.c)
yawgpu_add_example(triangle_passthrough)

# Copy the prebuilt SPIR-V binaries + MSL source next to the binary so
# the example can fopen them with the same relative paths from any cwd.
foreach(shader triangle.vert.spv triangle.frag.spv triangle.msl)
    add_custom_command(TARGET triangle_passthrough POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            "$<TARGET_FILE_DIR:triangle_passthrough>/${shader}"
    )
endforeach()
