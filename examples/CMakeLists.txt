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
set(YAWGPU_EXTENSIONS "" CACHE STRING "Optional yawgpu extension cargo features: shader-passthrough;tiled;mobile")

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

string(REPLACE " " ";" YAWGPU_EXTENSIONS_LIST "${YAWGPU_EXTENSIONS}")
set(YAWGPU_EXTENSION_COMPILE_DEFS)
foreach(YAWGPU_EXTENSION IN LISTS YAWGPU_EXTENSIONS_LIST)
    if(YAWGPU_EXTENSION STREQUAL "")
        continue()
    endif()
    list(APPEND YAWGPU_CARGO_FEATURES ${YAWGPU_EXTENSION})
    if(YAWGPU_EXTENSION STREQUAL "shader-passthrough")
        list(APPEND YAWGPU_EXTENSION_COMPILE_DEFS YAWGPU_HAS_SHADER_PASSTHROUGH)
    elseif(YAWGPU_EXTENSION STREQUAL "tiled")
        list(APPEND YAWGPU_EXTENSION_COMPILE_DEFS YAWGPU_HAS_TILED)
    elseif(YAWGPU_EXTENSION STREQUAL "mobile")
        list(APPEND YAWGPU_EXTENSION_COMPILE_DEFS
            YAWGPU_HAS_SHADER_PASSTHROUGH
            YAWGPU_HAS_TILED
        )
    else()
        message(FATAL_ERROR "Unknown YAWGPU_EXTENSIONS entry: ${YAWGPU_EXTENSION}")
    endif()
endforeach()
list(REMOVE_DUPLICATES YAWGPU_CARGO_FEATURES)
list(REMOVE_DUPLICATES YAWGPU_EXTENSION_COMPILE_DEFS)

if(YAWGPU_CARGO_FEATURES)
    list(JOIN YAWGPU_CARGO_FEATURES "," YAWGPU_CARGO_FEATURES_ARG)
    set(YAWGPU_CARGO_FEATURE_ARGS --features "${YAWGPU_CARGO_FEATURES_ARG}")
else()
    set(YAWGPU_CARGO_FEATURE_ARGS)
endif()

# Use a feature-specific cargo target directory so that --features metal and
# --features vulkan dylibs do not overwrite each other in target/${profile}.
# Without this, the surface_smoke binary in build-metal would dlopen whichever
# dylib was built last and silently fall back to the Noop arm in
# `#[cfg(not(feature = "metal"))]`.
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

add_library(yawgpu_c_api SHARED IMPORTED GLOBAL)
set_target_properties(yawgpu_c_api PROPERTIES
    IMPORTED_LOCATION "${YAWGPU_SHARED_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
)
if(YAWGPU_EXTENSION_COMPILE_DEFS)
    target_compile_definitions(yawgpu_c_api INTERFACE ${YAWGPU_EXTENSION_COMPILE_DEFS})
endif()
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
        )
    endif()
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4)
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
endif()
