cmake_minimum_required(VERSION 3.20)
project(yawgpu_tint_shim CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED YAWGPU_DAWN_DIR)
    message(FATAL_ERROR "YAWGPU_DAWN_DIR must be set to a Dawn checkout")
endif()

# --- Minimal Tint configuration -------------------------------------------
# We want ONLY Tint's WGSL reader + MSL/SPIR-V/GLSL writers + inspector. Dawn's
# native GPU backends, HLSL, validators, command tools, tests, samples, fuzzers
# and protobuf are all off. The writer flags default to the corresponding
# DAWN_ENABLE_* backend, so we set them explicitly to ON regardless of backends.
set(DAWN_ENABLE_METAL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_NODE_BINDINGS OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TINTD OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_IR_BINARY OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_READER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_GLSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_NULL_WRITER OFF CACHE BOOL "" FORCE)

# EXCLUDE_FROM_ALL: nothing in Dawn builds unless `tint_shim` depends on it, so
# building target `tint_shim` pulls in only tint_api + tint_api_helpers and their
# transitive deps (abseil + the linked SPIRV-Tools subset).
add_subdirectory(${YAWGPU_DAWN_DIR} ${CMAKE_BINARY_DIR}/dawn EXCLUDE_FROM_ALL)

# Build the shim as a self-contained shared library linking tint_api PRIVATE,
# which bundles the needed Tint code behind one dylib and avoids the
# 56-static-library link-order problem.
add_library(tint_shim SHARED tint_shim.cpp)
target_compile_features(tint_shim PRIVATE cxx_std_20)

# On Apple targets, make the dylib's own LC_ID_DYLIB loader-relative. The linker
# copies this id string into whatever links against tint_shim, so
# `libyawgpu.dylib` records `@loader_path/libtint_shim.dylib` at link time — the
# shim is then resolved next to whatever image loaded it (test binary, example,
# or the cdylib), with no rpath/DYLD_FALLBACK involvement. ELF (SONAME) and
# Windows have no install names, so guard to Apple.
if(APPLE)
    # BUILD_WITH_INSTALL_NAME_DIR is required in addition to INSTALL_NAME_DIR:
    # CMake's default MACOSX_RPATH gives a *build-tree* dylib the install name
    # `@rpath/libtint_shim.dylib`, and INSTALL_NAME_DIR otherwise only applies at
    # `install` time. We consume the build-tree dylib directly (never install it),
    # so this flag makes the build-tree LC_ID_DYLIB use INSTALL_NAME_DIR now.
    set_target_properties(tint_shim PROPERTIES
        INSTALL_NAME_DIR "@loader_path"
        BUILD_WITH_INSTALL_NAME_DIR ON)
endif()
target_include_directories(tint_shim PRIVATE
    ${YAWGPU_DAWN_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tint_shim PRIVATE
    tint_api
    tint_api_helpers
    tint_lang_glsl_writer_helpers
)
