cmake_minimum_required(VERSION 3.20)
project(yawgpu_examples LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    set(YAWGPU_GLFW_FOUND OFF)
    set(YAWGPU_WINDOWED_EXAMPLES ON)
elseif(APPLE)
    find_package(glfw3 QUIET)
    if(glfw3_FOUND)
        set(YAWGPU_GLFW_FOUND ON)
        set(YAWGPU_WINDOWED_EXAMPLES ON)
        enable_language(OBJC)
    else()
        set(YAWGPU_GLFW_FOUND OFF)
        set(YAWGPU_WINDOWED_EXAMPLES OFF)
        message(STATUS "glfw3 not found; skipping windowed examples")
    endif()
else()
    find_package(glfw3 QUIET)
    set(YAWGPU_GLFW_FOUND OFF)
    set(YAWGPU_WINDOWED_EXAMPLES OFF)
    message(STATUS "windowed examples are enabled on macOS with GLFW or on Windows")
endif()

cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH YAWGPU_ROOT)

set(YAWGPU_PROFILE "debug" CACHE STRING "Cargo profile used for libyawgpu")
set_property(CACHE YAWGPU_PROFILE PROPERTY STRINGS debug release)
set(YAWGPU_FEATURE "" CACHE STRING "Optional yawgpu cargo feature: metal or vulkan")
# Opt-in: also build libyawgpu with the `shader-passthrough` vendor feature and
# enable the `triangle_passthrough` example (SPIR-V/MSL native shader passthrough).
option(YAWGPU_SHADER_PASSTHROUGH "Build the shader-passthrough example (adds the shader-passthrough cargo feature)" OFF)
# Opt-in: also build libyawgpu with the `tiled` vendor feature and enable the
# `tiled_deferred` example (multi-subpass deferred rendering via input attachments).
option(YAWGPU_TILED "Build the tiled-deferred example (adds the tiled cargo feature)" OFF)

if(YAWGPU_PROFILE STREQUAL "release")
    set(YAWGPU_CARGO_PROFILE_ARG --release)
    set(YAWGPU_TARGET_PROFILE release)
else()
    set(YAWGPU_CARGO_PROFILE_ARG)
    set(YAWGPU_TARGET_PROFILE debug)
endif()

set(YAWGPU_CARGO_FEATURES)
if(NOT YAWGPU_FEATURE STREQUAL "")
    list(APPEND YAWGPU_CARGO_FEATURES ${YAWGPU_FEATURE})
endif()
if(YAWGPU_SHADER_PASSTHROUGH)
    list(APPEND YAWGPU_CARGO_FEATURES shader-passthrough)
endif()
if(YAWGPU_TILED)
    list(APPEND YAWGPU_CARGO_FEATURES tiled)
endif()
list(REMOVE_DUPLICATES YAWGPU_CARGO_FEATURES)

if(YAWGPU_CARGO_FEATURES)
    list(JOIN YAWGPU_CARGO_FEATURES "," YAWGPU_CARGO_FEATURES_ARG)
    set(YAWGPU_CARGO_FEATURE_ARGS --features "${YAWGPU_CARGO_FEATURES_ARG}")
else()
    set(YAWGPU_CARGO_FEATURE_ARGS)
endif()

# Use a feature-specific cargo target directory so dylibs from different backend
# configurations do not overwrite each other in target/${profile}.
if(YAWGPU_FEATURE STREQUAL "")
    set(YAWGPU_TARGET_ROOT "${YAWGPU_ROOT}/target")
else()
    set(YAWGPU_TARGET_ROOT "${YAWGPU_ROOT}/target-${YAWGPU_FEATURE}")
endif()

if(DEFINED ENV{CARGO_BUILD_TARGET})
    set(YAWGPU_TARGET_DIR "${YAWGPU_TARGET_ROOT}/$ENV{CARGO_BUILD_TARGET}/${YAWGPU_TARGET_PROFILE}")
else()
    set(YAWGPU_TARGET_DIR "${YAWGPU_TARGET_ROOT}/${YAWGPU_TARGET_PROFILE}")
endif()

if(WIN32)
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/yawgpu.dll")
    set(YAWGPU_IMPORT_LIB "${YAWGPU_TARGET_DIR}/yawgpu.dll.lib")
    # The default Tint-linked yawgpu.dll loads tint_shim.dll at runtime; build.rs
    # places it next to yawgpu.dll in the target dir. Copied per-example below.
    set(YAWGPU_TINT_SHIM "${YAWGPU_TARGET_DIR}/tint_shim.dll")
elseif(APPLE)
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/libyawgpu.dylib")
else()
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/libyawgpu.so")
endif()

add_custom_target(yawgpu_cargo_build
    COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${YAWGPU_TARGET_ROOT}
            cargo build -p yawgpu ${YAWGPU_CARGO_PROFILE_ARG} ${YAWGPU_CARGO_FEATURE_ARGS}
    WORKING_DIRECTORY "${YAWGPU_ROOT}"
    BYPRODUCTS "${YAWGPU_SHARED_LIB}"
    COMMENT "Building libyawgpu with Cargo (target dir: ${YAWGPU_TARGET_ROOT})"
    VERBATIM
)

# libyawgpu references the Tint shim as @rpath/libtint_shim.<ext>, but cargo
# emits it into a hashed build-script output dir rather than next to libyawgpu.
# Copy it alongside libyawgpu (already on the examples' rpath) so every example
# loads it without a manual DYLD_LIBRARY_PATH. Windows is handled by build.rs.
if(APPLE OR (UNIX AND NOT WIN32))
    if(APPLE)
        set(YAWGPU_TINT_SHIM_NAME "libtint_shim.dylib")
    else()
        set(YAWGPU_TINT_SHIM_NAME "libtint_shim.so")
    endif()
    add_custom_command(TARGET yawgpu_cargo_build POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DTARGET_DIR=${YAWGPU_TARGET_DIR}
            -DSHIM_NAME=${YAWGPU_TINT_SHIM_NAME}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_tint_shim.cmake"
        COMMENT "Copying Tint shim next to libyawgpu"
        VERBATIM
    )
endif()

add_library(yawgpu_c_api SHARED IMPORTED GLOBAL)
set_target_properties(yawgpu_c_api PROPERTIES
    IMPORTED_LOCATION "${YAWGPU_SHARED_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
)
if(WIN32)
    set_target_properties(yawgpu_c_api PROPERTIES
        IMPORTED_IMPLIB "${YAWGPU_IMPORT_LIB}"
    )
endif()
add_dependencies(yawgpu_c_api yawgpu_cargo_build)

if(APPLE)
    set(YAWGPU_PLATFORM_LIBS "-framework Cocoa" "-framework CoreFoundation" "-framework QuartzCore" "-framework Metal")
elseif(UNIX)
    set(YAWGPU_PLATFORM_LIBS m dl)
else()
    set(YAWGPU_PLATFORM_LIBS)
endif()

function(yawgpu_add_example TARGET_NAME)
    target_link_libraries(${TARGET_NAME} PRIVATE framework yawgpu_c_api ${YAWGPU_PLATFORM_LIBS})
    target_include_directories(${TARGET_NAME} PRIVATE
        "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
        "${CMAKE_SOURCE_DIR}/framework"
    )
    if(APPLE)
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${YAWGPU_TARGET_DIR}"
            INSTALL_RPATH "${YAWGPU_TARGET_DIR}"
        )
    elseif(UNIX)
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${YAWGPU_TARGET_DIR}"
        )
    endif()
    if(WIN32)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${YAWGPU_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/yawgpu.dll"
            COMMAND ${CMAKE_COMMAND}
                -DSRC=${YAWGPU_TINT_SHIM}
                -DDEST_DIR=$<TARGET_FILE_DIR:${TARGET_NAME}>
                -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_dll.cmake"
        )
    endif()
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4)
        # Delay-load yawgpu.dll so the framework's failure hook can report a
        # missing yawgpu.dll / tint_shim.dll instead of the loader killing the
        # process pre-main with status 0xC0000135 and no output.
        target_link_options(${TARGET_NAME} PRIVATE /DELAYLOAD:yawgpu.dll)
        target_link_libraries(${TARGET_NAME} PRIVATE delayimp)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_subdirectory(framework)
add_subdirectory(enumerate_adapters)
add_subdirectory(compute)
add_subdirectory(device_info)
add_subdirectory(capture)
if(YAWGPU_WINDOWED_EXAMPLES)
    add_subdirectory(surface_smoke)
    add_subdirectory(triangle)
    add_subdirectory(hello_triangle)
    if(YAWGPU_SHADER_PASSTHROUGH)
        add_subdirectory(triangle_passthrough)
    endif()
    # tiled_deferred / tiled_msaa present to a window by default (and offscreen
    # with --verify), so they need the windowing framework (GLFW).
    if(YAWGPU_TILED)
        add_subdirectory(tiled_deferred)
        # tiled_msaa demonstrates per-sample MSAA subpass input (Vulkan-only).
        add_subdirectory(tiled_msaa)
    endif()
elseif(YAWGPU_TILED)
    message(STATUS "YAWGPU_TILED set but windowed examples are off; tiled_deferred "
                   "/ tiled_msaa need GLFW for their window — install GLFW or run with "
                   "--verify after enabling windowed examples")
endif()
