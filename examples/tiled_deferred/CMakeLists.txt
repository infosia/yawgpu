add_executable(tiled_deferred main.c)
target_include_directories(tiled_deferred PRIVATE "${CMAKE_SOURCE_DIR}/capture")
target_compile_definitions(tiled_deferred PRIVATE STB_IMAGE_WRITE_IMPLEMENTATION)
yawgpu_add_example(tiled_deferred)
