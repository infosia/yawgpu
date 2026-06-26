/* C ABI for driving Dawn's Tint from Rust.
 *
 * All returned `char*` strings are heap-allocated and must be freed with
 * `yawgpu_tint_string_free`. Returned `uint32_t*` arrays must be freed with
 * `yawgpu_tint_u32_free`. Failures are reported as false/NULL with `*err` set
 * to a heap-allocated diagnostic string when an error out-parameter is present.
 */
#ifndef YAWGPU_TINT_SHIM_H
#define YAWGPU_TINT_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YawgpuTintProgram YawgpuTintProgram;

/* Initializes the Tint runtime. Idempotent; safe to call more than once.
 * Tint ICEs terminate the process by design; this Tint revision exposes only
 * per-ICE callbacks, not a global callback setter that the shim can install. */
void yawgpu_tint_initialize(void);

/* Parse + validate WGSL. Returns NULL on failure with *err set.
 * If shader_f16 is true, Tint parses with wgsl::AllowedFeatures::Everything(). */
YawgpuTintProgram* yawgpu_tint_program_create(const char* wgsl,
                                              size_t wgsl_len,
                                              bool shader_f16,
                                              char** err);

/* Destroys a program returned by yawgpu_tint_program_create. NULL is ignored. */
void yawgpu_tint_program_destroy(YawgpuTintProgram*);

/* Entry point stage. Mirrors tint/lang/wgsl/inspector/entry_point.h:
 * 0=kVertex, 1=kFragment, 2=kCompute. */
typedef struct {
    const char* name;          /* borrowed; valid until program destroyed */
    uint8_t stage;             /* 0=vertex 1=fragment 2=compute */
    bool has_workgroup_size;
    uint32_t wg_x, wg_y, wg_z;
    bool frag_depth_used, sample_mask_used;
} YawgpuTintEntryPoint;

size_t yawgpu_tint_entry_point_count(const YawgpuTintProgram*);
bool yawgpu_tint_entry_point_get(const YawgpuTintProgram*, size_t i, YawgpuTintEntryPoint* out);

/* Resource enums mirror tint/lang/wgsl/inspector/resource_binding.h:
 * resource_type:
 *   0=kUniformBuffer, 1=kStorageBuffer, 2=kReadOnlyStorageBuffer, 3=kSampler,
 *   4=kSampledTexture, 5=kMultisampledTexture, 6=kWriteOnlyStorageTexture,
 *   7=kReadOnlyStorageTexture, 8=kReadWriteStorageTexture, 9=kDepthTexture,
 *   10=kDepthMultisampledTexture, 11=kExternalTexture, 12=kReadOnlyTexelBuffer,
 *   13=kReadWriteTexelBuffer, 14=kInputAttachment.
 * dim:
 *   0=k1d, 1=k2d, 2=k2dArray, 3=k3d, 4=kCube, 5=kCubeArray, 6=kNone.
 * sampled_kind:
 *   0=kFloat, 1=kUInt, 2=kSInt, 3=kFilterable, 4=kUnfilterable,
 *   5=kUnknownFilterable.
 * sampler_type:
 *   0=kComparison, 1=kFiltering, 2=kNonFiltering, 3=kUnknownFiltering.
 * texel_format:
 *   0=kR8Snorm, 1=kR8Uint, 2=kR8Sint, 3=kRg8Unorm, 4=kRg8Snorm,
 *   5=kRg8Uint, 6=kRg8Sint, 7=kR16Unorm, 8=kR16Snorm, 9=kR16Uint,
 *   10=kR16Sint, 11=kR16Float, 12=kRg16Unorm, 13=kRg16Snorm,
 *   14=kRg16Uint, 15=kRg16Sint, 16=kRg16Float, 17=kBgra8Unorm,
 *   18=kRgba8Unorm, 19=kRgba8Snorm, 20=kRgba8Uint, 21=kRgba8Sint,
 *   22=kRgba16Unorm, 23=kRgba16Snorm, 24=kRgba16Uint, 25=kRgba16Sint,
 *   26=kRgba16Float, 27=kR32Uint, 28=kR32Sint, 29=kR32Float,
 *   30=kRg32Uint, 31=kRg32Sint, 32=kRg32Float, 33=kRgba32Uint,
 *   34=kRgba32Sint, 35=kRgba32Float, 36=kR8Unorm, 37=kRgb10A2Uint,
 *   38=kRgb10A2Unorm, 39=kRg11B10Ufloat, 40=kNone. */
typedef struct {
    uint32_t group, binding;
    uint8_t resource_type;
    uint8_t dim, sampled_kind, sampler_type, texel_format;
    uint64_t size;
    bool has_array_size;
    uint32_t array_size;
} YawgpuTintResourceBinding;

size_t yawgpu_tint_resource_binding_count(const YawgpuTintProgram*, const char* ep);
bool yawgpu_tint_resource_binding_get(const YawgpuTintProgram*,
                                      const char* ep,
                                      size_t i,
                                      YawgpuTintResourceBinding* out);

/* Override type_class mirrors tint::inspector::Override::Type:
 * 0=kBool, 1=kFloat32, 2=kUint32, 3=kInt32, 4=kFloat16.
 * default_value is populated from sem::GlobalVariable::ConstantValue() when
 * has_default is true; otherwise it is 0.0. Boolean defaults are encoded as
 * 0.0 or 1.0. */
typedef struct {
    const char* name;          /* borrowed; valid until program destroyed */
    uint16_t id;
    uint8_t type_class;
    bool has_default;
    double default_value;
} YawgpuTintOverride;

size_t yawgpu_tint_override_count(const YawgpuTintProgram*);
bool yawgpu_tint_override_get(const YawgpuTintProgram*, size_t i, YawgpuTintOverride* out);

typedef struct {
    uint32_t group, binding, dst_group, dst_binding;
} YawgpuTintBindingRemap;

typedef struct {
    const YawgpuTintBindingRemap* uniform;
    size_t n_uniform;
    const YawgpuTintBindingRemap* storage;
    size_t n_storage;
    const YawgpuTintBindingRemap* texture;
    size_t n_texture;
    const YawgpuTintBindingRemap* storage_texture;
    size_t n_storage_texture;
    const YawgpuTintBindingRemap* sampler;
    size_t n_sampler;
} YawgpuTintBindings;

typedef struct {
    const char* name;
    double value;
} YawgpuTintOverrideValue;

typedef struct {
    char* msl;
    bool needs_storage_buffer_sizes;
} YawgpuTintMslOutput;

bool yawgpu_tint_generate_msl(const YawgpuTintProgram*,
                              const char* ep,
                              const YawgpuTintBindings*,
                              const YawgpuTintOverrideValue* ov,
                              size_t n_ov,
                              bool disable_robustness,
                              YawgpuTintMslOutput* out,
                              char** err);

bool yawgpu_tint_generate_spirv(const YawgpuTintProgram*,
                                const char* ep,
                                const YawgpuTintBindings*,
                                const YawgpuTintOverrideValue* ov,
                                size_t n_ov,
                                bool disable_robustness,
                                uint32_t** words_out,
                                size_t* n_words_out,
                                char** err);

bool yawgpu_tint_generate_glsl(const YawgpuTintProgram*,
                               const char* ep,
                               const YawgpuTintBindings*,
                               const YawgpuTintOverrideValue* ov,
                               size_t n_ov,
                               char** glsl_out,
                               char** err);

void yawgpu_tint_string_free(char*);
void yawgpu_tint_u32_free(uint32_t*);

#ifdef __cplusplus
}
#endif

#endif /* YAWGPU_TINT_SHIM_H */
