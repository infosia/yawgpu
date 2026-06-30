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

/* Export marker for the shim's public C ABI. This header is only ever compiled
 * into the shim itself (the Rust side declares the same functions by hand), so
 * the marker is always the producer/export side. On MSVC a shared library
 * exports nothing by default, so without `__declspec(dllexport)` no import
 * library is generated and dependents fail to link (LNK1181). On other
 * toolchains symbols default to public; the visibility attribute keeps the
 * intent explicit and is a no-op there. */
#if defined(_WIN32)
#  define YAWGPU_TINT_API __declspec(dllexport)
#else
#  define YAWGPU_TINT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YawgpuTintProgram YawgpuTintProgram;

/* Initializes the Tint runtime. Idempotent; safe to call more than once.
 * Tint ICEs terminate the process by design; this Tint revision exposes only
 * per-ICE callbacks, not a global callback setter that the shim can install. */
YAWGPU_TINT_API void yawgpu_tint_initialize(void);

/* Parse + validate WGSL. Returns NULL on failure with *err set.
 * If shader_f16 is true, Tint allows the f16 WGSL extension.
 * If allow_framebuffer_fetch is true, Tint allows the framebuffer-fetch WGSL extension.
 * lang_features contains WGPUWGSLLanguageFeatureName numeric values. */
YAWGPU_TINT_API YawgpuTintProgram* yawgpu_tint_program_create(const char* wgsl,
                                              size_t wgsl_len,
                                              bool shader_f16,
                                              bool allow_framebuffer_fetch,
                                              const uint32_t* lang_features,
                                              size_t n_lang_features,
                                              char** err);

/* Destroys a program returned by yawgpu_tint_program_create. NULL is ignored. */
YAWGPU_TINT_API void yawgpu_tint_program_destroy(YawgpuTintProgram*);

/* Entry point stage. Mirrors tint/lang/wgsl/inspector/entry_point.h:
 * 0=kVertex, 1=kFragment, 2=kCompute. */
typedef struct {
    const char* name;          /* borrowed; valid until program destroyed */
    uint8_t stage;             /* 0=vertex 1=fragment 2=compute */
    bool has_workgroup_size;
    uint32_t wg_x, wg_y, wg_z;
    bool frag_depth_used, sample_mask_used;
    bool input_sample_mask_used, front_facing_used, sample_index_used;
    bool primitive_index_used, subgroup_invocation_id_used, subgroup_size_used;
} YawgpuTintEntryPoint;

YAWGPU_TINT_API size_t yawgpu_tint_entry_point_count(const YawgpuTintProgram*);
YAWGPU_TINT_API bool yawgpu_tint_entry_point_get(const YawgpuTintProgram*, size_t i, YawgpuTintEntryPoint* out);

/* Stage-variable enums mirror tint/lang/wgsl/inspector/entry_point.h:
 * component_type:
 *   0=kF32, 1=kU32, 2=kI32, 3=kF16, 4=kUnknown.
 * composition_type:
 *   0=kScalar, 1=kVec2, 2=kVec3, 3=kVec4, 4=kUnknown.
 * interpolation_type:
 *   0=kPerspective, 1=kLinear, 2=kFlat, 3=kUnknown.
 * interpolation_sampling:
 *   0=kNone, 1=kCenter, 2=kCentroid, 3=kSample, 4=kFirst, 5=kEither,
 *   6=kUnknown. */
typedef struct {
    bool has_location;
    uint32_t location;
    bool has_color;
    uint32_t color;
    uint8_t component_type;
    uint8_t composition_type;
    uint8_t interpolation_type;
    uint8_t interpolation_sampling;
} YawgpuTintStageVariable;

YAWGPU_TINT_API size_t yawgpu_tint_entry_point_input_count(const YawgpuTintProgram*, const char* ep);
YAWGPU_TINT_API bool yawgpu_tint_entry_point_input_get(const YawgpuTintProgram*,
                                       const char* ep,
                                       size_t i,
                                       YawgpuTintStageVariable* out);
YAWGPU_TINT_API size_t yawgpu_tint_entry_point_output_count(const YawgpuTintProgram*, const char* ep);
YAWGPU_TINT_API bool yawgpu_tint_entry_point_output_get(const YawgpuTintProgram*,
                                        const char* ep,
                                        size_t i,
                                        YawgpuTintStageVariable* out);

/* Program diagnostics collected during successful parsing.
 * severity: 0=note/info, 1=warning. Error diagnostics fail program creation. */
typedef struct {
    const char* message;       /* borrowed; valid until program destroyed */
    uint8_t severity;
} YawgpuTintDiagnostic;

YAWGPU_TINT_API size_t yawgpu_tint_diagnostic_count(const YawgpuTintProgram*);
YAWGPU_TINT_API bool yawgpu_tint_diagnostic_get(const YawgpuTintProgram*, size_t i, YawgpuTintDiagnostic* out);

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
 *   38=kRgb10A2Unorm, 39=kRg11B10Ufloat, 40=kNone.
 * sample_usage:
 *   0=load, 1=sample, 2=gather. Strongest usage per texture binding for the
 *   requested entry point, computed from Tint sem call graph and AST calls.
 * input_attachment_index:
 *   Meaningful only when resource_type == 14 kInputAttachment; otherwise 0. */
typedef struct {
    uint32_t group, binding;
    uint8_t resource_type;
    uint8_t dim, sampled_kind, sampler_type, texel_format;
    uint8_t sample_usage;
    uint64_t size;
    bool has_array_size;
    uint32_t array_size;
    uint32_t input_attachment_index;
} YawgpuTintResourceBinding;

YAWGPU_TINT_API size_t yawgpu_tint_resource_binding_count(const YawgpuTintProgram*, const char* ep);
YAWGPU_TINT_API bool yawgpu_tint_resource_binding_get(const YawgpuTintProgram*,
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
    /* True only when the override has an explicit `@id(N)` attribute. Tint
     * always assigns a numeric id (sequential when `@id` is absent), so callers
     * that need WebGPU's "key by numeric id only for @id overrides" rule must use
     * this flag rather than treating every id as explicit. */
    bool has_explicit_id;
    uint8_t type_class;
    bool has_default;
    double default_value;
} YawgpuTintOverride;

YAWGPU_TINT_API size_t yawgpu_tint_override_count(const YawgpuTintProgram*);
YAWGPU_TINT_API bool yawgpu_tint_override_get(const YawgpuTintProgram*, size_t i, YawgpuTintOverride* out);

typedef struct {
    uint32_t group, binding, dst_group, dst_binding;
} YawgpuTintBindingRemap;

typedef struct {
    uint32_t src_group;     /* WGSL group of the texture_external */
    uint32_t src_binding;   /* WGSL binding of the texture_external */
    uint32_t plane0_slot;   /* MSL texture slot for plane 0 */
    uint32_t plane1_slot;   /* MSL texture slot for plane 1 */
    uint32_t params_slot;   /* MSL buffer slot for the params (metadata) UBO */
} YawgpuTintExternalTextureRemap;

typedef struct {
    uint32_t group;        /* WGSL @group of the input_attachment var */
    uint32_t binding;      /* WGSL @binding of the input_attachment var */
    uint32_t color_slot;   /* Metal [[color(N)]] slot to lower it to */
} YawgpuTintInputAttachmentColorIndex;

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
    const YawgpuTintExternalTextureRemap* external_texture;
    size_t n_external_texture;
    const YawgpuTintInputAttachmentColorIndex* input_attachment_color_index;
    size_t n_input_attachment_color_index;
} YawgpuTintBindings;

typedef struct {
    const char* name;
    double value;
} YawgpuTintOverrideValue;

typedef struct {
    uint8_t format;
    uint32_t offset;
    uint32_t shader_location;
} YawgpuTintVertexAttribute;

typedef struct {
    uint32_t slot;
    uint32_t metal_index;
    uint32_t array_stride;
    uint8_t step_mode;
    const YawgpuTintVertexAttribute* attributes;
    size_t n_attributes;
} YawgpuTintVertexBuffer;

typedef struct {
    char* msl;
    char* entry_point;
    bool needs_storage_buffer_sizes;
    uint32_t* buffer_size_bindings;
    size_t n_buffer_size_bindings;
    uint32_t* workgroup_allocations;    /* per-index threadgroup allocation sizes (compute) */
    size_t n_workgroup_allocations;
    bool has_frag_depth_clamp;          /* true if this fragment EP writes frag_depth and clamp was enabled */
    uint32_t frag_depth_clamp_slot;     /* MSL buffer index of the depth-range immediate block (valid iff has_frag_depth_clamp) */
} YawgpuTintMslOutput;

YAWGPU_TINT_API bool yawgpu_tint_generate_msl(const YawgpuTintProgram*,
                              const char* ep,
                              const YawgpuTintBindings*,
                              const YawgpuTintOverrideValue* ov,
                              size_t n_ov,
                              uint32_t buffer_sizes_slot,
                              bool disable_robustness,
                              bool emit_vertex_point_size,
                              const YawgpuTintVertexBuffer* vertex_buffers,
                              size_t n_vertex_buffers,
                              uint32_t fixed_sample_mask,
                              YawgpuTintMslOutput* out,
                              char** err);

YAWGPU_TINT_API bool yawgpu_tint_generate_spirv(const YawgpuTintProgram*,
                                const char* ep,
                                const YawgpuTintBindings*,
                                const YawgpuTintOverrideValue* ov,
                                size_t n_ov,
                                bool disable_robustness,
                                bool use_vulkan_memory_model,
                                uint32_t framebuffer_fetch_descriptor_set,
                                bool multisampled_input_attachment,
                                uint32_t** words_out,
                                size_t* n_words_out,
                                char** err);

/* Returns the module's total var<workgroup> storage size in bytes (0 if none).
   Returns false + sets *err on hard failure. */
YAWGPU_TINT_API bool yawgpu_tint_workgroup_storage_size(const YawgpuTintProgram*,
                                        const YawgpuTintOverrideValue* ov,
                                        size_t n_ov,
                                        uint64_t* out,
                                        char** err);

YAWGPU_TINT_API bool yawgpu_tint_generate_glsl(const YawgpuTintProgram*,
                               const char* ep,
                               const YawgpuTintBindings*,
                               const YawgpuTintOverrideValue* ov,
                               size_t n_ov,
                               char** glsl_out,
                               char** err);

YAWGPU_TINT_API void yawgpu_tint_string_free(char*);
YAWGPU_TINT_API void yawgpu_tint_u32_free(uint32_t*);

#ifdef __cplusplus
}
#endif

#endif /* YAWGPU_TINT_SHIM_H */
