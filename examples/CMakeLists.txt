cmake_minimum_required(VERSION 3.20)
project(yawgpu_examples LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH YAWGPU_ROOT)

set(YAWGPU_PROFILE "debug" CACHE STRING "Cargo profile used for libyawgpu")
set_property(CACHE YAWGPU_PROFILE PROPERTY STRINGS debug release)
set(YAWGPU_FEATURE "" CACHE STRING "Optional yawgpu cargo feature: metal or vulkan")

if(YAWGPU_PROFILE STREQUAL "release")
    set(YAWGPU_CARGO_PROFILE_ARG --release)
    set(YAWGPU_TARGET_PROFILE release)
else()
    set(YAWGPU_CARGO_PROFILE_ARG)
    set(YAWGPU_TARGET_PROFILE debug)
endif()

if(YAWGPU_FEATURE STREQUAL "")
    set(YAWGPU_CARGO_FEATURE_ARGS)
else()
    set(YAWGPU_CARGO_FEATURE_ARGS --features ${YAWGPU_FEATURE})
endif()

if(DEFINED ENV{CARGO_BUILD_TARGET})
    set(YAWGPU_TARGET_DIR "${YAWGPU_ROOT}/target/$ENV{CARGO_BUILD_TARGET}/${YAWGPU_TARGET_PROFILE}")
else()
    set(YAWGPU_TARGET_DIR "${YAWGPU_ROOT}/target/${YAWGPU_TARGET_PROFILE}")
endif()

if(WIN32)
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/yawgpu.dll")
elseif(APPLE)
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/libyawgpu.dylib")
else()
    set(YAWGPU_SHARED_LIB "${YAWGPU_TARGET_DIR}/libyawgpu.so")
endif()

add_custom_target(yawgpu_cargo_build
    COMMAND cargo build -p yawgpu ${YAWGPU_CARGO_PROFILE_ARG} ${YAWGPU_CARGO_FEATURE_ARGS}
    WORKING_DIRECTORY "${YAWGPU_ROOT}"
    BYPRODUCTS "${YAWGPU_SHARED_LIB}"
    COMMENT "Building libyawgpu with Cargo"
    VERBATIM
)

add_library(yawgpu_c_api SHARED IMPORTED GLOBAL)
set_target_properties(yawgpu_c_api PROPERTIES
    IMPORTED_LOCATION "${YAWGPU_SHARED_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${YAWGPU_ROOT}/yawgpu/ffi/webgpu-headers"
)
add_dependencies(yawgpu_c_api yawgpu_cargo_build)

if(APPLE)
    set(YAWGPU_PLATFORM_LIBS "-framework CoreFoundation" "-framework QuartzCore" "-framework Metal")
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
