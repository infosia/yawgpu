# copy_runtime_dll.cmake — run at build time (cmake -P) to place an optional
# runtime DLL next to an example executable, skipping silently when it is
# absent.
#
# On Windows, libyawgpu (the default, Tint-linked build) depends on
# tint_shim.dll at load time. yawgpu-tint's build.rs copies tint_shim.dll next
# to yawgpu.dll in the cargo target dir, but the example's own POST_BUILD must
# carry it next to the executable so the app launches without putting the cargo
# target dir on PATH. A non-Tint stub build produces no shim, so the copy is
# guarded by EXISTS rather than failing the build.
#
# Inputs (passed via -D): SRC (full path to the DLL), DEST_DIR (executable dir).
if(EXISTS "${SRC}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${SRC}" "${DEST_DIR}/")
endif()
