cmake_minimum_required(VERSION 3.20)
project(yawgpu_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(BENCH_BACKEND "yawgpu" CACHE STRING "WebGPU implementation to link (yawgpu|dawn)")
set_property(CACHE BENCH_BACKEND PROPERTY STRINGS "yawgpu" "dawn")

# yawgpu: defaults to this checkout; the release dylib is the shipped artifact.
set(BENCH_YAWGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Path to a yawgpu checkout")
set(BENCH_YAWGPU_LIB "" CACHE FILEPATH "Explicit path to libyawgpu")

# Dawn: no default — a Dawn checkout is a local prerequisite, never a repo path.
set(BENCH_DAWN_DIR "" CACHE PATH "Path to a Dawn checkout")
set(BENCH_DAWN_BUILD_DIR "" CACHE PATH "Path to a Dawn build directory")
set(BENCH_DAWN_LIB "" CACHE FILEPATH "Explicit path to libwebgpu_dawn")

if(BENCH_BACKEND STREQUAL "yawgpu")
    set(BENCH_HEADER_DIR "${BENCH_YAWGPU_DIR}/yawgpu/ffi")
    if(NOT EXISTS "${BENCH_HEADER_DIR}/webgpu-headers/webgpu.h")
        message(FATAL_ERROR "Could not find webgpu-headers/webgpu.h under ${BENCH_YAWGPU_DIR}")
    endif()

    if(BENCH_YAWGPU_LIB)
        set(BENCH_LIB "${BENCH_YAWGPU_LIB}")
    else()
        find_library(YAWGPU_LIB
            NAMES yawgpu
            PATHS
                "${BENCH_YAWGPU_DIR}/target/release"
                "${BENCH_YAWGPU_DIR}/target-metal/release"
                "${BENCH_YAWGPU_DIR}/target-vulkan/release"
            NO_DEFAULT_PATH
        )
        set(BENCH_LIB "${YAWGPU_LIB}")
    endif()
    if(NOT BENCH_LIB)
        message(FATAL_ERROR "Could not find libyawgpu. Run `cargo build -p yawgpu --release` or set BENCH_YAWGPU_LIB.")
    endif()

    set(BENCH_BACKEND_SOURCE src/backend_yawgpu.cpp)
    set(BENCH_BACKEND_DEFINE YAWGPU_BENCH_BACKEND_YAWGPU=1)
elseif(BENCH_BACKEND STREQUAL "dawn")
    if(NOT BENCH_DAWN_DIR OR NOT BENCH_DAWN_BUILD_DIR)
        message(FATAL_ERROR "BENCH_DAWN_DIR and BENCH_DAWN_BUILD_DIR are required for BENCH_BACKEND=dawn")
    endif()
    set(BENCH_HEADER_DIR "${BENCH_DAWN_DIR}/include" "${BENCH_DAWN_BUILD_DIR}/gen/include")

    if(BENCH_DAWN_LIB)
        set(BENCH_LIB "${BENCH_DAWN_LIB}")
    else()
        find_library(DAWN_WEBGPU_LIB
            NAMES webgpu_dawn
            PATHS
                "${BENCH_DAWN_BUILD_DIR}/src/dawn/native"
                "${BENCH_DAWN_BUILD_DIR}/src/dawn"
                "${BENCH_DAWN_BUILD_DIR}"
            NO_DEFAULT_PATH
        )
        set(BENCH_LIB "${DAWN_WEBGPU_LIB}")
    endif()
    if(NOT BENCH_LIB)
        message(FATAL_ERROR "Could not find libwebgpu_dawn. Build Dawn or set BENCH_DAWN_LIB.")
    endif()

    set(BENCH_BACKEND_SOURCE src/backend_dawn.cpp)
    set(BENCH_BACKEND_DEFINE YAWGPU_BENCH_BACKEND_DAWN=1)
else()
    message(FATAL_ERROR "Unknown BENCH_BACKEND: ${BENCH_BACKEND}")
endif()

add_executable(bench src/main.cpp ${BENCH_BACKEND_SOURCE})
target_include_directories(bench PRIVATE ${BENCH_HEADER_DIR} src)
target_compile_definitions(bench PRIVATE ${BENCH_BACKEND_DEFINE})
target_link_libraries(bench PRIVATE "${BENCH_LIB}")

if(APPLE)
    # libyawgpu.dylib carries an @rpath install_name and loads the Tint shim
    # relative to itself; teach the executable where the linked library lives.
    get_filename_component(_bench_lib_dir "${BENCH_LIB}" DIRECTORY)
    set_target_properties(bench PROPERTIES
        BUILD_RPATH "${_bench_lib_dir}"
        INSTALL_RPATH "${_bench_lib_dir}")
    target_link_libraries(bench PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework IOSurface")
endif()

message(STATUS "bench backend: ${BENCH_BACKEND}")
message(STATUS "bench library: ${BENCH_LIB}")
