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
        set(BENCH_DAWN_LIB_DIRS
            "${BENCH_DAWN_BUILD_DIR}/src/dawn/native"
            "${BENCH_DAWN_BUILD_DIR}/src/dawn"
            "${BENCH_DAWN_BUILD_DIR}"
        )
        # A multi-config generator (Visual Studio, Xcode) writes the library into
        # a per-configuration subdirectory of the directory a single-config
        # generator writes it into directly. Search both layouts, preferring a
        # Release Dawn since the benchmark itself defaults to Release.
        foreach(config Release RelWithDebInfo MinSizeRel Debug)
            list(APPEND BENCH_DAWN_LIB_DIRS
                "${BENCH_DAWN_BUILD_DIR}/src/dawn/native/${config}"
                "${BENCH_DAWN_BUILD_DIR}/src/dawn/${config}"
                "${BENCH_DAWN_BUILD_DIR}/${config}"
            )
        endforeach()

        find_library(DAWN_WEBGPU_LIB
            NAMES webgpu_dawn
            PATHS ${BENCH_DAWN_LIB_DIRS}
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

if(WIN32)
    # bench.exe links an import library and Windows has no rpath, so every
    # runtime DLL must sit next to the executable. Without this the loader
    # aborts the process before main with 0xC0000135 and no diagnostic, and the
    # only workaround is putting the cargo target dir (or Dawn's build dir) on
    # PATH by hand. Mirrors examples/CMakeLists.txt's POST_BUILD copy.
    get_filename_component(BENCH_LIB_DIR "${BENCH_LIB}" DIRECTORY)
    set(BENCH_RUNTIME_DIRS "${BENCH_LIB_DIR}")

    if(BENCH_BACKEND STREQUAL "dawn")
        # webgpu_dawn.dll lands at the build root (under the per-configuration
        # subdirectory for a multi-config generator), not beside its import
        # library in src/dawn/native.
        list(APPEND BENCH_RUNTIME_DIRS "${BENCH_DAWN_BUILD_DIR}")
        foreach(config Release RelWithDebInfo MinSizeRel Debug)
            list(APPEND BENCH_RUNTIME_DIRS "${BENCH_DAWN_BUILD_DIR}/${config}")
        endforeach()
        set(BENCH_RUNTIME_DLLS webgpu_dawn.dll)
        set(BENCH_OPTIONAL_DLLS)
    else()
        # The default Tint-linked yawgpu.dll loads tint_shim.dll at startup;
        # yawgpu-tint's build.rs places it beside yawgpu.dll. A non-Tint stub
        # build produces no shim, so that one is optional.
        set(BENCH_RUNTIME_DLLS yawgpu.dll)
        set(BENCH_OPTIONAL_DLLS tint_shim.dll)
    endif()

    foreach(dll ${BENCH_RUNTIME_DLLS} ${BENCH_OPTIONAL_DLLS})
        set(BENCH_FOUND_DLL "")
        foreach(dir ${BENCH_RUNTIME_DIRS})
            if(NOT BENCH_FOUND_DLL AND EXISTS "${dir}/${dll}")
                set(BENCH_FOUND_DLL "${dir}/${dll}")
            endif()
        endforeach()

        if(BENCH_FOUND_DLL)
            add_custom_command(TARGET bench POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${BENCH_FOUND_DLL}" "$<TARGET_FILE_DIR:bench>"
                VERBATIM
            )
            message(STATUS "bench runtime: ${BENCH_FOUND_DLL}")
        elseif(dll IN_LIST BENCH_RUNTIME_DLLS)
            message(WARNING "Could not find ${dll} near ${BENCH_LIB}; bench.exe "
                            "will not start unless its directory is on PATH")
        endif()
    endforeach()
endif()

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
