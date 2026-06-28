# copy_tint_shim.cmake — run at build time (cmake -P) after the cargo build.
#
# libyawgpu links the Tint shim as `@rpath/libtint_shim.<ext>`, but cargo emits
# the shim into a hashed build-script output directory
# (`<target>/<profile>/build/yawgpu-tint-*/out/build/`), not next to
# libyawgpu itself. The examples' rpath already includes the cargo target dir
# (where libyawgpu lives), so copying the shim there makes every example load it
# without a manual DYLD_LIBRARY_PATH / LD_LIBRARY_PATH. (Windows is handled by
# yawgpu-tint's build.rs, which copies tint_shim.dll next to the artifacts.)
#
# Inputs (passed via -D): TARGET_DIR (cargo target/profile dir, also the dest),
# SHIM_NAME (libtint_shim.dylib | libtint_shim.so).

file(GLOB _shims
    "${TARGET_DIR}/build/yawgpu-tint-*/out/build/${SHIM_NAME}"
    "${TARGET_DIR}/build/yawgpu-tint-*/out/${SHIM_NAME}")

if(_shims)
    list(GET _shims 0 _shim)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_shim}" "${TARGET_DIR}/${SHIM_NAME}")
    message(STATUS "copied Tint shim: ${_shim} -> ${TARGET_DIR}/${SHIM_NAME}")
else()
    message(STATUS "Tint shim ${SHIM_NAME} not found under ${TARGET_DIR}/build/yawgpu-tint-*/ (skipping copy)")
endif()
