/* C ABI for driving Dawn's Tint from Rust.
 *
 * Phase 1 smoke surface: just enough to prove the build/link/FFI path
 * (WGSL -> MSL). The full reflection + multi-target codegen ABI lands in
 * Phase 1b. All returned `char*` strings are heap-allocated and must be freed
 * with `yawgpu_tint_string_free`. No function aborts across the boundary;
 * failures are reported via the `err` out-parameter. */
#ifndef YAWGPU_TINT_SHIM_H
#define YAWGPU_TINT_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initializes the Tint runtime. Idempotent; safe to call more than once. */
void yawgpu_tint_initialize(void);

/* Compiles a WGSL module to MSL for `entry_point`. Returns a heap-allocated MSL
 * string on success, or NULL on failure with `*err` set to a heap-allocated
 * diagnostic message (NULL if none). Free both with yawgpu_tint_string_free. */
char* yawgpu_tint_wgsl_to_msl(const char* wgsl, const char* entry_point, char** err);

/* Frees a string previously returned by this library. NULL is ignored. */
void yawgpu_tint_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif /* YAWGPU_TINT_SHIM_H */
