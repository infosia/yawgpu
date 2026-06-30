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

# When the feature set changes, cargo creates a second build-script output dir
# (a new `yawgpu-tint-<hash>`), so the glob can return several shims. Pick the
# NEWEST by timestamp — taking the first glob result risks copying a stale shim
# whose ABI no longer matches the freshly-built libyawgpu (an arg-shift crash in
# yawgpu_tint_program_create).
if(_shims)
    set(_shim "")
    set(_newest "")
    foreach(_candidate IN LISTS _shims)
        file(TIMESTAMP "${_candidate}" _ts "%Y%m%d%H%M%S")
        if(_ts STRGREATER _newest)
            set(_newest "${_ts}")
            set(_shim "${_candidate}")
        endif()
    endforeach()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_shim}" "${TARGET_DIR}/${SHIM_NAME}")
    message(STATUS "copied Tint shim: ${_shim} -> ${TARGET_DIR}/${SHIM_NAME}")
else()
    message(STATUS "Tint shim ${SHIM_NAME} not found under ${TARGET_DIR}/build/yawgpu-tint-*/ (skipping copy)")
endif()
