// Tint C shim implementation. Mirrors dawn/src/tint/cmd/tint/main.cc for
// Parse, ProgramToLoweredIR, GenerateBindings, SubstituteOverrides, and writer
// option setup.

#include <cstdlib>
#include <cstring>
#include <exception>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "src/tint/api/common/substitute_overrides_config.h"
#include "src/tint/api/common/vertex_pulling_config.h"
#include "src/tint/api/helpers/generate_bindings.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/constant/value.h"
#include "src/tint/lang/core/ir/reflection.h"
#include "src/tint/lang/core/ir/referenced_module_vars.h"
#include "src/tint/lang/core/ir/transform/substitute_overrides.h"
#include "src/tint/lang/core/type/pointer.h"
#include "src/tint/lang/glsl/writer/helpers/generate_bindings.h"
#include "src/tint/lang/glsl/writer/writer.h"
#include "src/tint/lang/msl/writer/common/options.h"
#include "src/tint/lang/msl/writer/writer.h"
#include "src/tint/lang/spirv/writer/writer.h"
#include "src/tint/lang/wgsl/ast/id_attribute.h"
#include "src/tint/lang/wgsl/ast/identifier.h"
#include "src/tint/lang/wgsl/ast/module.h"
#include "src/tint/lang/wgsl/ast/override.h"
#include "src/tint/lang/wgsl/enums.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/wgsl/sem/builtin_fn.h"
#include "src/tint/lang/wgsl/sem/call.h"
#include "src/tint/lang/wgsl/sem/function.h"
#include "src/tint/lang/wgsl/sem/module.h"
#include "src/tint/lang/wgsl/sem/variable.h"
#include "src/tint/utils/containers/hashmap.h"
#include "src/tint/utils/containers/unique_vector.h"
#include "src/tint/utils/rtti/switch.h"

#include "tint_shim.h"

// ---------------------------------------------------------------------------
// ABI drift guards -- enum ordinals.
//
// The reflection structs below export several Tint enums as raw `uint8_t`
// ordinals via `static_cast` (see fill_entry_point, fill_stage_variable,
// fill_resource_binding, fill_override further down). `yawgpu-tint`'s Rust
// side (`yawgpu-tint/src/lib.rs`, `raw_enum!` blocks) re-hardcodes the same
// numeric values by hand -- there is no bindgen or shared enum definition.
// Tint gives no ABI stability guarantee for these enums across Dawn
// revisions: a reorder, insertion, or removal of a variant would silently
// corrupt reflection (wrong texture dimension, wrong resource type, ...)
// with no compile error anywhere in the pipeline.
//
// The static_asserts below pin every ordinal this shim casts to `uint8_t` to
// the exact value the Rust `raw_enum!` mappings expect. If Dawn reorders a
// Tint enum, one of these fires and the C++ build breaks instead of shipping
// silently-wrong reflection. Each block is labeled with the header-comment
// table it protects (`tint_shim.h:83-156`) and the fill_* function that
// performs the cast.
//
// See "Dawn rev bump" at the top of tint_shim.h for the fix procedure when
// one of these fires.
// ---------------------------------------------------------------------------

// Message shared by every ordinal guard below; undef'd at the end of this
// section so it doesn't leak into the rest of the translation unit.
#define YAWGPU_TINT_ORDINAL_MSG \
    "tint enum ordinal changed; update tint_shim.h + yawgpu-tint raw_enum! mappings"

// tint::inspector::EntryPoint::stage -- tint_shim.h:62-63 ("0=kVertex,
// 1=kFragment, 2=kCompute"); cast in fill_entry_point (~L388).
// Source: third_party/dawn/src/tint/lang/wgsl/inspector/entry_point.h,
// `enum class PipelineStage`.
static_assert(static_cast<uint8_t>(tint::inspector::PipelineStage::kVertex) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::PipelineStage::kFragment) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::PipelineStage::kCompute) == 2,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::StageVariable::component_type -- tint_shim.h:84-85
// ("0=kF32, 1=kU32, 2=kI32, 3=kF16, 4=kUnknown"); cast in fill_stage_variable
// (~L460). Source: entry_point.h, `enum class ComponentType`.
static_assert(static_cast<uint8_t>(tint::inspector::ComponentType::kF32) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ComponentType::kU32) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ComponentType::kI32) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ComponentType::kF16) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ComponentType::kUnknown) == 4,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::StageVariable::composition_type -- tint_shim.h:86-87
// ("0=kScalar, 1=kVec2, 2=kVec3, 3=kVec4, 4=kUnknown"); cast in
// fill_stage_variable (~L461). Source: entry_point.h, `enum class
// CompositionType`.
static_assert(static_cast<uint8_t>(tint::inspector::CompositionType::kScalar) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::CompositionType::kVec2) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::CompositionType::kVec3) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::CompositionType::kVec4) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::CompositionType::kUnknown) == 4,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::StageVariable::interpolation_type -- tint_shim.h:88-89
// ("0=kPerspective, 1=kLinear, 2=kFlat, 3=kUnknown"); cast in
// fill_stage_variable (~L462). Source: entry_point.h, `enum class
// InterpolationType`.
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationType::kPerspective) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationType::kLinear) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationType::kFlat) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationType::kUnknown) == 3,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::StageVariable::interpolation_sampling -- tint_shim.h:90-92
// ("0=kNone, 1=kCenter, 2=kCentroid, 3=kSample, 4=kFirst, 5=kEither,
// 6=kUnknown"); cast in fill_stage_variable (~L463). Source: entry_point.h,
// `enum class InterpolationSampling`.
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kNone) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kCenter) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kCentroid) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kSample) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kFirst) == 4,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kEither) == 5,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::InterpolationSampling::kUnknown) == 6,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::ResourceBinding::resource_type -- tint_shim.h:128-133
// ("0=kUniformBuffer, ... 14=kInputAttachment"); cast in
// fill_resource_binding (~L599). Source: resource_binding.h, `enum class
// ResourceType` (nested in `struct ResourceBinding`).
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kUniformBuffer) == 0,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kStorageBuffer) == 1,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kReadOnlyStorageBuffer) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kSampler) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kSampledTexture) == 4,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kMultisampledTexture) == 5,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kWriteOnlyStorageTexture) == 6,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kReadOnlyStorageTexture) == 7,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kReadWriteStorageTexture) == 8,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kDepthTexture) == 9,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kDepthMultisampledTexture) == 10,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kExternalTexture) == 11,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kReadOnlyTexelBuffer) == 12,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(
                  tint::inspector::ResourceBinding::ResourceType::kReadWriteTexelBuffer) == 13,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::ResourceType::kInputAttachment) == 14,
    YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::ResourceBinding::dim -- tint_shim.h:134-135 ("0=k1d,
// 1=k2d, 2=k2dArray, 3=k3d, 4=kCube, 5=kCubeArray, 6=kNone"); cast in
// fill_resource_binding (~L600). Source: resource_binding.h, `enum class
// TextureDimension` (nested in `struct ResourceBinding`).
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::k1d) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::k2d) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::k2dArray) == 2,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::k3d) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::kCube) == 4,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::kCubeArray) == 5,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TextureDimension::kNone) == 6,
    YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::ResourceBinding::sampled_kind -- tint_shim.h:136-138
// ("0=kFloat, 1=kUInt, 2=kSInt, 3=kFilterable, 4=kUnfilterable,
// 5=kUnknownFilterable"); cast in fill_resource_binding (~L601). Source:
// resource_binding.h, `enum class SampledKind` (nested in `struct
// ResourceBinding`).
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kFloat) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kUInt) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kSInt) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kFilterable) == 3,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kUnfilterable) == 4,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SampledKind::kUnknownFilterable) == 5,
    YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::ResourceBinding::sampler_type -- tint_shim.h:139-140
// ("0=kComparison, 1=kFiltering, 2=kNonFiltering, 3=kUnknownFiltering"); cast
// in fill_resource_binding (~L602). Source: resource_binding.h, `enum class
// SamplerType` (nested in `struct ResourceBinding`).
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SamplerType::kComparison) == 0,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SamplerType::kFiltering) == 1,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SamplerType::kNonFiltering) == 2,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::SamplerType::kUnknownFiltering) == 3,
    YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::ResourceBinding::image_format (texel_format) --
// tint_shim.h:141-151 (0=kR8Snorm .. 40=kNone, ~40 variants); cast in
// fill_resource_binding (~L603). Source: resource_binding.h, `enum class
// TexelFormat` (nested in `struct ResourceBinding`).
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR8Snorm) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR8Uint) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR8Sint) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg8Unorm) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg8Snorm) == 4,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg8Uint) == 5,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg8Sint) == 6,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR16Unorm) == 7,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR16Snorm) == 8,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR16Uint) == 9,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR16Sint) == 10,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR16Float) == 11,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg16Unorm) == 12,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg16Snorm) == 13,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg16Uint) == 14,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg16Sint) == 15,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg16Float) == 16,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kBgra8Unorm) == 17,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba8Unorm) == 18,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba8Snorm) == 19,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba8Uint) == 20,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba8Sint) == 21,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba16Unorm) == 22,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba16Snorm) == 23,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba16Uint) == 24,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba16Sint) == 25,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba16Float) == 26,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR32Uint) == 27,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR32Sint) == 28,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR32Float) == 29,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg32Uint) == 30,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg32Sint) == 31,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg32Float) == 32,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba32Uint) == 33,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba32Sint) == 34,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgba32Float) == 35,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kR8Unorm) == 36,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgb10A2Uint) == 37,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRgb10A2Unorm) == 38,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(
    static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kRg11B10Ufloat) == 39,
    YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::ResourceBinding::TexelFormat::kNone) == 40,
              YAWGPU_TINT_ORDINAL_MSG);

// tint::inspector::Override::type_class -- tint_shim.h:174-175 ("0=kBool,
// 1=kFloat32, 2=kUint32, 3=kInt32, 4=kFloat16"); cast in fill_override
// (~L652). Source: entry_point.h, `enum class Type` (nested in `struct
// Override`).
static_assert(static_cast<uint8_t>(tint::inspector::Override::Type::kBool) == 0,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::Override::Type::kFloat32) == 1,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::Override::Type::kUint32) == 2,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::Override::Type::kInt32) == 3,
              YAWGPU_TINT_ORDINAL_MSG);
static_assert(static_cast<uint8_t>(tint::inspector::Override::Type::kFloat16) == 4,
              YAWGPU_TINT_ORDINAL_MSG);

#undef YAWGPU_TINT_ORDINAL_MSG

// ---------------------------------------------------------------------------
// ABI drift guards -- FFI struct layout.
//
// The `YawgpuTintXxx` structs declared in tint_shim.h are hand-mirrored on
// the Rust side as `#[repr(C)]` `RawXxx` structs (yawgpu-tint/src/lib.rs).
// There is no bindgen keeping the two definitions in sync, so a field
// reorder or insertion on either side would silently desynchronize the
// layout. These asserts pin `sizeof`/`offsetof` for every FFI struct on this
// (64-bit LP64/LLP64) target; the matching Rust-side asserts live next to
// each `RawXxx` definition in lib.rs. All FFI structs here use only 8-byte
// pointers/`size_t`, 1-byte `bool`, and standard scalar alignment, which is
// identical between the Itanium (macOS/Linux) and Microsoft (Windows, LLP64)
// 64-bit ABIs, so these constants hold on every yawgpu target platform.
// ---------------------------------------------------------------------------

static_assert(sizeof(YawgpuTintEntryPoint) == 40, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, name) == 0, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, stage) == 8, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, has_workgroup_size) == 9,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, wg_x) == 12, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, wg_y) == 16, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, wg_z) == 20, "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, frag_depth_used) == 24,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, sample_mask_used) == 25,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, input_sample_mask_used) == 26,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, front_facing_used) == 27,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, sample_index_used) == 28,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, primitive_index_used) == 29,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, subgroup_invocation_id_used) == 30,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, subgroup_size_used) == 31,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, frag_position_used) == 32,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, has_clip_distances) == 33,
              "YawgpuTintEntryPoint layout changed");
static_assert(offsetof(YawgpuTintEntryPoint, clip_distances_size) == 36,
              "YawgpuTintEntryPoint layout changed");

static_assert(sizeof(YawgpuTintStageVariable) == 28, "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, has_location) == 0,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, location) == 4,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, has_color) == 8,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, color) == 12,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, has_blend_src) == 16,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, blend_src) == 20,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, component_type) == 24,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, composition_type) == 25,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, interpolation_type) == 26,
              "YawgpuTintStageVariable layout changed");
static_assert(offsetof(YawgpuTintStageVariable, interpolation_sampling) == 27,
              "YawgpuTintStageVariable layout changed");

static_assert(sizeof(YawgpuTintDiagnostic) == 16, "YawgpuTintDiagnostic layout changed");
static_assert(offsetof(YawgpuTintDiagnostic, message) == 0,
              "YawgpuTintDiagnostic layout changed");
static_assert(offsetof(YawgpuTintDiagnostic, severity) == 8,
              "YawgpuTintDiagnostic layout changed");

static_assert(sizeof(YawgpuTintResourceBinding) == 40,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, group) == 0,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, binding) == 4,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, resource_type) == 8,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, dim) == 9,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, sampled_kind) == 10,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, sampler_type) == 11,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, texel_format) == 12,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, sample_usage) == 13,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, size) == 16,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, has_array_size) == 24,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, array_size) == 28,
              "YawgpuTintResourceBinding layout changed");
static_assert(offsetof(YawgpuTintResourceBinding, input_attachment_index) == 32,
              "YawgpuTintResourceBinding layout changed");

static_assert(sizeof(YawgpuTintOverride) == 24, "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, name) == 0, "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, id) == 8, "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, has_explicit_id) == 10,
              "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, type_class) == 11,
              "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, has_default) == 12,
              "YawgpuTintOverride layout changed");
static_assert(offsetof(YawgpuTintOverride, default_value) == 16,
              "YawgpuTintOverride layout changed");

static_assert(sizeof(YawgpuTintBindingRemap) == 16, "YawgpuTintBindingRemap layout changed");
static_assert(offsetof(YawgpuTintBindingRemap, group) == 0,
              "YawgpuTintBindingRemap layout changed");
static_assert(offsetof(YawgpuTintBindingRemap, binding) == 4,
              "YawgpuTintBindingRemap layout changed");
static_assert(offsetof(YawgpuTintBindingRemap, dst_group) == 8,
              "YawgpuTintBindingRemap layout changed");
static_assert(offsetof(YawgpuTintBindingRemap, dst_binding) == 12,
              "YawgpuTintBindingRemap layout changed");

static_assert(sizeof(YawgpuTintExternalTextureRemap) == 20,
              "YawgpuTintExternalTextureRemap layout changed");
static_assert(offsetof(YawgpuTintExternalTextureRemap, src_group) == 0,
              "YawgpuTintExternalTextureRemap layout changed");
static_assert(offsetof(YawgpuTintExternalTextureRemap, src_binding) == 4,
              "YawgpuTintExternalTextureRemap layout changed");
static_assert(offsetof(YawgpuTintExternalTextureRemap, plane0_slot) == 8,
              "YawgpuTintExternalTextureRemap layout changed");
static_assert(offsetof(YawgpuTintExternalTextureRemap, plane1_slot) == 12,
              "YawgpuTintExternalTextureRemap layout changed");
static_assert(offsetof(YawgpuTintExternalTextureRemap, params_slot) == 16,
              "YawgpuTintExternalTextureRemap layout changed");

static_assert(sizeof(YawgpuTintInputAttachmentColorIndex) == 12,
              "YawgpuTintInputAttachmentColorIndex layout changed");
static_assert(offsetof(YawgpuTintInputAttachmentColorIndex, group) == 0,
              "YawgpuTintInputAttachmentColorIndex layout changed");
static_assert(offsetof(YawgpuTintInputAttachmentColorIndex, binding) == 4,
              "YawgpuTintInputAttachmentColorIndex layout changed");
static_assert(offsetof(YawgpuTintInputAttachmentColorIndex, color_slot) == 8,
              "YawgpuTintInputAttachmentColorIndex layout changed");

static_assert(sizeof(YawgpuTintBindings) == 112, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, uniform) == 0, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_uniform) == 8, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, storage) == 16, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_storage) == 24, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, texture) == 32, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_texture) == 40, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, storage_texture) == 48,
              "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_storage_texture) == 56,
              "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, sampler) == 64, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_sampler) == 72, "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, external_texture) == 80,
              "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_external_texture) == 88,
              "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, input_attachment_color_index) == 96,
              "YawgpuTintBindings layout changed");
static_assert(offsetof(YawgpuTintBindings, n_input_attachment_color_index) == 104,
              "YawgpuTintBindings layout changed");

static_assert(sizeof(YawgpuTintOverrideValue) == 16, "YawgpuTintOverrideValue layout changed");
static_assert(offsetof(YawgpuTintOverrideValue, name) == 0,
              "YawgpuTintOverrideValue layout changed");
static_assert(offsetof(YawgpuTintOverrideValue, value) == 8,
              "YawgpuTintOverrideValue layout changed");

static_assert(sizeof(YawgpuTintVertexAttribute) == 12, "YawgpuTintVertexAttribute layout changed");
static_assert(offsetof(YawgpuTintVertexAttribute, format) == 0,
              "YawgpuTintVertexAttribute layout changed");
static_assert(offsetof(YawgpuTintVertexAttribute, offset) == 4,
              "YawgpuTintVertexAttribute layout changed");
static_assert(offsetof(YawgpuTintVertexAttribute, shader_location) == 8,
              "YawgpuTintVertexAttribute layout changed");

static_assert(sizeof(YawgpuTintVertexBuffer) == 32, "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, slot) == 0,
              "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, metal_index) == 4,
              "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, array_stride) == 8,
              "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, step_mode) == 12,
              "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, attributes) == 16,
              "YawgpuTintVertexBuffer layout changed");
static_assert(offsetof(YawgpuTintVertexBuffer, n_attributes) == 24,
              "YawgpuTintVertexBuffer layout changed");

static_assert(sizeof(YawgpuTintMslOutput) == 64, "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, msl) == 0, "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, entry_point) == 8,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, needs_storage_buffer_sizes) == 16,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, buffer_size_bindings) == 24,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, n_buffer_size_bindings) == 32,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, workgroup_allocations) == 40,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, n_workgroup_allocations) == 48,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, has_frag_depth_clamp) == 56,
              "YawgpuTintMslOutput layout changed");
static_assert(offsetof(YawgpuTintMslOutput, frag_depth_clamp_slot) == 60,
              "YawgpuTintMslOutput layout changed");

struct YawgpuTintProgram {
    // Must outlive `program`: Tint Source objects keep pointers into this file.
    std::unique_ptr<tint::Source::File> file;
    tint::Program program;
    std::vector<tint::inspector::EntryPoint> entry_points;
    std::vector<tint::inspector::Override> overrides;
    std::vector<std::string> diagnostic_messages;
    std::vector<uint8_t> diagnostic_severities;
};

namespace {

char* dup_string(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

template <typename Failure>
void set_error(char** err, const Failure& failure) {
    if (err != nullptr) {
        std::stringstream ss;
        ss << failure;
        *err = dup_string(ss.str());
    }
}

void set_error_string(char** err, const std::string& message) {
    if (err != nullptr) {
        *err = dup_string(message);
    }
}

std::string cstr_or_empty(const char* s) {
    return s != nullptr ? std::string(s) : std::string();
}

bool all_remaps_empty(const YawgpuTintBindings* bindings) {
    return bindings == nullptr ||
           (bindings->n_uniform == 0 && bindings->n_storage == 0 && bindings->n_texture == 0 &&
            bindings->n_storage_texture == 0 && bindings->n_sampler == 0 &&
            bindings->n_external_texture == 0 && bindings->n_input_attachment_color_index == 0);
}

void fill_binding_map(tint::BindingMap& map,
                      const YawgpuTintBindingRemap* remaps,
                      size_t count) {
    if (remaps == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& r = remaps[i];
        map.emplace(tint::BindingPoint{.group = r.group, .binding = r.binding},
                    tint::BindingPoint{.group = r.dst_group, .binding = r.dst_binding});
    }
}

tint::Bindings make_bindings(const YawgpuTintBindings* bindings) {
    tint::Bindings out;
    if (bindings == nullptr) {
        return out;
    }
    fill_binding_map(out.uniform, bindings->uniform, bindings->n_uniform);
    fill_binding_map(out.storage, bindings->storage, bindings->n_storage);
    fill_binding_map(out.texture, bindings->texture, bindings->n_texture);
    fill_binding_map(out.storage_texture, bindings->storage_texture, bindings->n_storage_texture);
    fill_binding_map(out.sampler, bindings->sampler, bindings->n_sampler);
    if (bindings->external_texture != nullptr) {
        for (size_t i = 0; i < bindings->n_external_texture; ++i) {
            const auto& e = bindings->external_texture[i];
            out.external_texture[tint::BindingPoint{e.src_group, e.src_binding}] =
                tint::ExternalMultiplanarTexture{
                    /*metadata=*/tint::BindingPoint{0u, e.params_slot},
                    /*plane0=*/tint::BindingPoint{0u, e.plane0_slot},
                    /*plane1=*/tint::BindingPoint{0u, e.plane1_slot},
                };
        }
    }
    return out;
}

tint::VertexFormat to_tint_vertex_format(uint8_t format) {
    switch (format) {
        case 0:
            return tint::VertexFormat::kUint8;
        case 1:
            return tint::VertexFormat::kUint8x2;
        case 2:
            return tint::VertexFormat::kUint8x4;
        case 3:
            return tint::VertexFormat::kSint8;
        case 4:
            return tint::VertexFormat::kSint8x2;
        case 5:
            return tint::VertexFormat::kSint8x4;
        case 6:
            return tint::VertexFormat::kUnorm8;
        case 7:
            return tint::VertexFormat::kUnorm8x2;
        case 8:
            return tint::VertexFormat::kUnorm8x4;
        case 9:
            return tint::VertexFormat::kSnorm8;
        case 10:
            return tint::VertexFormat::kSnorm8x2;
        case 11:
            return tint::VertexFormat::kSnorm8x4;
        case 12:
            return tint::VertexFormat::kUint16;
        case 13:
            return tint::VertexFormat::kUint16x2;
        case 14:
            return tint::VertexFormat::kUint16x4;
        case 15:
            return tint::VertexFormat::kSint16;
        case 16:
            return tint::VertexFormat::kSint16x2;
        case 17:
            return tint::VertexFormat::kSint16x4;
        case 18:
            return tint::VertexFormat::kUnorm16;
        case 19:
            return tint::VertexFormat::kUnorm16x2;
        case 20:
            return tint::VertexFormat::kUnorm16x4;
        case 21:
            return tint::VertexFormat::kSnorm16;
        case 22:
            return tint::VertexFormat::kSnorm16x2;
        case 23:
            return tint::VertexFormat::kSnorm16x4;
        case 24:
            return tint::VertexFormat::kFloat16;
        case 25:
            return tint::VertexFormat::kFloat16x2;
        case 26:
            return tint::VertexFormat::kFloat16x4;
        case 27:
            return tint::VertexFormat::kFloat32;
        case 28:
            return tint::VertexFormat::kFloat32x2;
        case 29:
            return tint::VertexFormat::kFloat32x3;
        case 30:
            return tint::VertexFormat::kFloat32x4;
        case 31:
            return tint::VertexFormat::kUint32;
        case 32:
            return tint::VertexFormat::kUint32x2;
        case 33:
            return tint::VertexFormat::kUint32x3;
        case 34:
            return tint::VertexFormat::kUint32x4;
        case 35:
            return tint::VertexFormat::kSint32;
        case 36:
            return tint::VertexFormat::kSint32x2;
        case 37:
            return tint::VertexFormat::kSint32x3;
        case 38:
            return tint::VertexFormat::kSint32x4;
        case 39:
            return tint::VertexFormat::kUnorm10_10_10_2;
        case 40:
            return tint::VertexFormat::kUnorm8x4BGRA;
        default:
            return tint::VertexFormat::kUint8;
    }
}

tint::VertexStepMode to_tint_step_mode(uint8_t step_mode) {
    switch (step_mode) {
        case 1:
            return tint::VertexStepMode::kInstance;
        case 0:
        default:
            return tint::VertexStepMode::kVertex;
    }
}

tint::diag::Result<tint::SubstituteOverridesConfig> make_override_config(
    const YawgpuTintProgram* program,
    const YawgpuTintOverrideValue* values,
    size_t count) {
    tint::SubstituteOverridesConfig cfg;
    if (count == 0) {
        return cfg;
    }

    tint::inspector::Inspector inspector(program->program);
    auto override_names = inspector.GetNamedOverrideIds();
    cfg.map.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (values == nullptr || values[i].name == nullptr || values[i].name[0] == '\0') {
            return tint::diag::Failure("empty override name");
        }
        std::string name(values[i].name);
        char* end = nullptr;
        unsigned long parsed = std::strtoul(name.c_str(), &end, 10);
        if (end != nullptr && *end == '\0' && parsed <= UINT16_MAX) {
            cfg.map.emplace(tint::OverrideId{static_cast<uint16_t>(parsed)}, values[i].value);
            continue;
        }
        auto found = override_names.find(name);
        if (found == override_names.end()) {
            return tint::diag::Failure("unknown override '" + name + "'");
        }
        cfg.map.emplace(found->second, values[i].value);
    }
    return cfg;
}

std::optional<tint::wgsl::LanguageFeature> to_tint_language_feature(uint32_t feature) {
    switch (feature) {
        case 1:
            return tint::wgsl::LanguageFeature::kReadonlyAndReadwriteStorageTextures;
        case 2:
            return tint::wgsl::LanguageFeature::kPacked4X8IntegerDotProduct;
        case 3:
            return tint::wgsl::LanguageFeature::kUnrestrictedPointerParameters;
        case 4:
            return tint::wgsl::LanguageFeature::kPointerCompositeAccess;
        case 5:
            return tint::wgsl::LanguageFeature::kUniformBufferStandardLayout;
        case 6:
            return tint::wgsl::LanguageFeature::kSubgroupId;
        case 7:
            return tint::wgsl::LanguageFeature::kTextureAndSamplerLet;
        case 8:
            return tint::wgsl::LanguageFeature::kSubgroupUniformity;
        case 9:
            return tint::wgsl::LanguageFeature::kTextureFormatsTier1;
        case 10:
            return tint::wgsl::LanguageFeature::kLinearIndexing;
        default:
            return std::nullopt;
    }
}

tint::Result<tint::core::ir::Module> lower_ir(const YawgpuTintProgram* program) {
    tint::wgsl::reader::IROptions options{
        .dump_ir_when_validating = false,
        .enable_validation_asserts = false,
    };
    return tint::wgsl::reader::ProgramToLoweredIR(program->program, options);
}

const tint::core::ir::Function* find_ir_entry_point(const tint::core::ir::Module& ir,
                                                    const std::string& ep_name) {
    for (auto* f : ir.functions) {
        if (f != nullptr && f->IsEntryPoint() && ir.NameOf(f).NameView() == ep_name) {
            return f;
        }
    }
    return nullptr;
}

tint::msl::writer::ArrayLengthOptions generate_array_length_from_constants(
    tint::core::ir::Module& ir,
    const std::string& ep_name,
    uint32_t buffer_sizes_slot,
    std::vector<tint::BindingPoint>& ordered_bindings) {
    tint::msl::writer::ArrayLengthOptions options{
        .ubo_binding = buffer_sizes_slot,
    };

    const tint::core::ir::Function* ep_func = find_ir_entry_point(ir, ep_name);
    if (ep_func == nullptr) {
        return options;
    }

    tint::core::ir::ReferencedModuleVars<const tint::core::ir::Module> referenced_module_vars{ir};
    auto& refs = referenced_module_vars.TransitiveReferences(ep_func);

    std::unordered_set<tint::BindingPoint> storage_bindings;
    for (auto* var : refs) {
        auto bp = var->BindingPoint();
        if (!bp.has_value()) {
            continue;
        }

        auto* ty = var->Result()->Type()->As<tint::core::type::Pointer>();
        if (ty != nullptr && ty->AddressSpace() == tint::core::AddressSpace::kStorage &&
            !ty->HasFixedFootprint()) {
            if (storage_bindings.insert(*bp).second) {
                auto size_index = static_cast<uint32_t>(ordered_bindings.size());
                options.bindpoint_to_size_index.emplace(*bp, size_index);
                ordered_bindings.push_back(*bp);
            }
        }
    }

    return options;
}

void mark_used_buffer_bindings(const tint::BindingMap& map, std::vector<bool>& used) {
    for (const auto& entry : map) {
        if (entry.second.group == 0 && entry.second.binding < used.size()) {
            used[entry.second.binding] = true;
        }
    }
}

std::optional<tint::BindingPoint> choose_immediate_binding_point(const tint::Bindings& bindings,
                                                                 uint32_t buffer_sizes_slot) {
    std::vector<bool> used(256, false);
    if (buffer_sizes_slot < used.size()) {
        used[buffer_sizes_slot] = true;
    }
    mark_used_buffer_bindings(bindings.uniform, used);
    mark_used_buffer_bindings(bindings.storage, used);

    for (uint32_t binding = 0; binding < used.size(); ++binding) {
        if (!used[binding]) {
            return tint::BindingPoint{.group = 0u, .binding = binding};
        }
    }
    return std::nullopt;
}

uint32_t* dup_binding_pairs(const std::vector<tint::BindingPoint>& bindings) {
    if (bindings.empty()) {
        return nullptr;
    }
    uint32_t* out = static_cast<uint32_t*>(std::malloc(bindings.size() * 2 * sizeof(uint32_t)));
    if (out == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
        out[i * 2] = bindings[i].group;
        out[i * 2 + 1] = bindings[i].binding;
    }
    return out;
}

void fill_entry_point(const tint::inspector::EntryPoint& ep, YawgpuTintEntryPoint* out) {
    out->name = ep.name.c_str();
    out->stage = static_cast<uint8_t>(ep.stage);
    out->has_workgroup_size = ep.workgroup_size.has_value();
    out->wg_x = ep.workgroup_size ? ep.workgroup_size->x : 0;
    out->wg_y = ep.workgroup_size ? ep.workgroup_size->y : 0;
    out->wg_z = ep.workgroup_size ? ep.workgroup_size->z : 0;
    out->frag_depth_used = ep.frag_depth_used;
    out->sample_mask_used = ep.input_sample_mask_used || ep.output_sample_mask_used;
    out->input_sample_mask_used = ep.input_sample_mask_used;
    out->front_facing_used = ep.front_facing_used;
    out->sample_index_used = ep.sample_index_used;
    out->primitive_index_used = ep.primitive_index_used;
    out->subgroup_invocation_id_used = ep.subgroup_invocation_id_used;
    out->subgroup_size_used = ep.subgroup_size_used;
    out->frag_position_used = ep.frag_position_used;
    out->has_clip_distances = ep.clip_distances_size.has_value();
    out->clip_distances_size = ep.clip_distances_size.value_or(0);
}

uint8_t diagnostic_severity(tint::diag::Severity severity) {
    switch (severity) {
        case tint::diag::Severity::Warning:
            return 1;
        case tint::diag::Severity::Note:
        case tint::diag::Severity::Error:
            return 0;
    }
    return 0;
}

const tint::inspector::EntryPoint* find_entry_point(const YawgpuTintProgram* program,
                                                    const char* ep) {
    if (program == nullptr || ep == nullptr) {
        return nullptr;
    }
    std::string name(ep);
    for (const auto& entry : program->entry_points) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

std::unordered_map<uint32_t, tint::BindingPoint> color_bindings_for_entry_point(
    const YawgpuTintProgram* program,
    const char* ep,
    uint32_t framebuffer_fetch_descriptor_set) {
    std::unordered_map<uint32_t, tint::BindingPoint> bindings;
    const auto* entry = find_entry_point(program, ep);
    if (entry == nullptr || entry->stage != tint::inspector::PipelineStage::kFragment) {
        return bindings;
    }

    for (const auto& input : entry->input_variables) {
        if (input.attributes.color.has_value()) {
            uint32_t slot = input.attributes.color.value();
            bindings.emplace(
                slot,
                tint::BindingPoint{.group = framebuffer_fetch_descriptor_set, .binding = slot});
        }
    }
    return bindings;
}

void fill_stage_variable(const tint::inspector::StageVariable& variable,
                         YawgpuTintStageVariable* out) {
    out->has_location = variable.attributes.location.has_value();
    out->location = variable.attributes.location.value_or(0);
    out->has_color = variable.attributes.color.has_value();
    out->color = variable.attributes.color.value_or(0);
    out->has_blend_src = variable.attributes.blend_src.has_value();
    out->blend_src = variable.attributes.blend_src.value_or(0);
    out->component_type = static_cast<uint8_t>(variable.component_type);
    out->composition_type = static_cast<uint8_t>(variable.composition_type);
    out->interpolation_type = static_cast<uint8_t>(variable.interpolation_type);
    out->interpolation_sampling = static_cast<uint8_t>(variable.interpolation_sampling);
}

uint8_t max_sample_usage(uint8_t lhs, uint8_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

uint8_t builtin_sample_usage(tint::wgsl::BuiltinFn builtin) {
    switch (builtin) {
        case tint::wgsl::BuiltinFn::kTextureGather:
        case tint::wgsl::BuiltinFn::kTextureGatherCompare:
            return 2;
        case tint::wgsl::BuiltinFn::kTextureSample:
        case tint::wgsl::BuiltinFn::kTextureSampleBias:
        case tint::wgsl::BuiltinFn::kTextureSampleCompare:
        case tint::wgsl::BuiltinFn::kTextureSampleCompareLevel:
        case tint::wgsl::BuiltinFn::kTextureSampleGrad:
        case tint::wgsl::BuiltinFn::kTextureSampleLevel:
        case tint::wgsl::BuiltinFn::kTextureSampleBaseClampToEdge:
            return 1;
        case tint::wgsl::BuiltinFn::kTextureLoad:
            return 0;
        default:
            return 0;
    }
}

std::map<std::pair<uint32_t, uint32_t>, uint8_t> texture_sample_usages(
    const tint::Program& program,
    const std::string& entry_point) {
    std::map<std::pair<uint32_t, uint32_t>, uint8_t> usages;
    const auto& sem = program.Sem();
    auto entry_point_symbol = program.Symbols().Get(entry_point);
    if (!entry_point_symbol.IsValid()) {
        return usages;
    }

    using GlobalSet = tint::UniqueVector<const tint::sem::GlobalVariable*, 4>;
    tint::Hashmap<const tint::sem::Function*,
                  tint::Hashmap<const tint::sem::Parameter*, GlobalSet, 2>,
                  8>
        globals_for_handle_parameters;

    auto add_globals_as_parameter = [&](const tint::sem::Function* fn,
                                        const tint::sem::Parameter* param,
                                        const GlobalSet* vars) {
        auto& globals = globals_for_handle_parameters.GetOrAddZero(fn).GetOrAddZero(param);
        for (const auto* var : *vars) {
            globals.Add(var);
        }
    };

    auto get_globals_for_argument = [&](const tint::sem::Function* fn,
                                        const tint::sem::ValueExpression* argument,
                                        GlobalSet* scratch_global) -> const GlobalSet* {
        auto* identifier = argument->RootIdentifier();
        auto* local = identifier != nullptr ? identifier->As<tint::sem::LocalVariable>() : nullptr;
        while (local != nullptr) {
            identifier = local->Initializer()->RootIdentifier();
            if (identifier == nullptr) {
                return scratch_global;
            }
            local = identifier->As<tint::sem::LocalVariable>();
        }

        if (auto* global =
                identifier != nullptr ? identifier->As<tint::sem::GlobalVariable>() : nullptr) {
            scratch_global->Add(global);
            return scratch_global;
        }
        if (auto* parameter =
                identifier != nullptr ? identifier->As<tint::sem::Parameter>() : nullptr) {
            if (auto by_fn = globals_for_handle_parameters.Get(fn)) {
                if (auto globals = by_fn.value->Get(parameter)) {
                    return globals.value;
                }
            }
        }
        return scratch_global;
    };

    auto declarations = sem.Module()->DependencyOrderedDeclarations();
    for (auto rit = declarations.rbegin(); rit != declarations.rend(); rit++) {
        auto* fn = sem.Get<tint::sem::Function>(*rit);
        if ((fn == nullptr) || !fn->HasCallGraphEntryPoint(entry_point_symbol)) {
            continue;
        }

        for (auto* call : fn->DirectCalls()) {
            tint::Switch(
                call->Target(),
                [&](const tint::sem::Function* callee) {
                    for (size_t i = 0; i < call->Arguments().Length(); i++) {
                        auto* parameter = sem.Get(callee->Declaration()->params[i]);
                        if (parameter == nullptr || !parameter->Type()->IsHandle()) {
                            continue;
                        }
                        GlobalSet scratch_global;
                        const auto* globals =
                            get_globals_for_argument(fn, call->Arguments()[i], &scratch_global);
                        add_globals_as_parameter(callee, parameter, globals);
                    }
                },
                [&](const tint::sem::BuiltinFn* builtin) {
                    const auto& signature = builtin->Signature();
                    int texture_index = signature.IndexOf(tint::core::ParameterUsage::kTexture);
                    if (texture_index == -1 ||
                        call->Arguments()[static_cast<size_t>(texture_index)]
                            ->Is<tint::sem::Call>()) {
                        return;
                    }
                    uint8_t usage = builtin_sample_usage(builtin->Fn());
                    GlobalSet scratch_global;
                    const auto* texture_globals = get_globals_for_argument(
                        fn, call->Arguments()[static_cast<size_t>(texture_index)],
                        &scratch_global);
                    for (const auto* texture : *texture_globals) {
                        auto binding_point = texture->Attributes().binding_point;
                        if (!binding_point.has_value()) {
                            continue;
                        }
                        auto key =
                            std::make_pair(binding_point->group, binding_point->binding);
                        usages[key] = max_sample_usage(usages[key], usage);
                    }
                });
        }
    }
    return usages;
}

void fill_resource_binding(const tint::inspector::ResourceBinding& binding,
                           uint8_t sample_usage,
                           YawgpuTintResourceBinding* out) {
    out->group = binding.bind_group;
    out->binding = binding.binding;
    out->resource_type = static_cast<uint8_t>(binding.resource_type);
    out->dim = static_cast<uint8_t>(binding.dim);
    out->sampled_kind = static_cast<uint8_t>(binding.sampled_kind);
    out->sampler_type = static_cast<uint8_t>(binding.sampler_type);
    out->texel_format = static_cast<uint8_t>(binding.image_format);
    out->sample_usage = sample_usage;
    out->size = binding.size;
    out->has_array_size = binding.array_size.has_value();
    out->array_size = binding.array_size.value_or(0);
    out->input_attachment_index = binding.input_attachment_index;
}

double override_default_value(const tint::sem::GlobalVariable* global,
                              tint::inspector::Override::Type type) {
    if (global == nullptr || global->Initializer() == nullptr ||
        global->Initializer()->ConstantValue() == nullptr) {
        return 0.0;
    }
    const auto* value = global->Initializer()->ConstantValue();
    switch (type) {
        case tint::inspector::Override::Type::kBool:
            return value->ValueAs<bool>() ? 1.0 : 0.0;
        case tint::inspector::Override::Type::kFloat32:
            return static_cast<double>(value->ValueAs<tint::core::f32>());
        case tint::inspector::Override::Type::kUint32:
            return static_cast<double>(value->ValueAs<tint::core::u32>());
        case tint::inspector::Override::Type::kInt32:
            return static_cast<double>(value->ValueAs<tint::core::i32>());
        case tint::inspector::Override::Type::kFloat16:
            return static_cast<double>(static_cast<float>(value->ValueAs<tint::core::f16>()));
    }
    return 0.0;
}

const tint::sem::GlobalVariable* find_override_global(const tint::Program& program,
                                                      const tint::inspector::Override& ov) {
    for (auto* decl : program.AST().Globals<tint::ast::Override>()) {
        auto* global = program.Sem().Get(decl);
        if (global == nullptr) {
            continue;
        }
        if (decl->name->symbol.Name() == ov.name) {
            return global;
        }
    }
    return nullptr;
}

void fill_override(const tint::Program& program,
                   const tint::inspector::Override& ov,
                   YawgpuTintOverride* out) {
    out->name = ov.name.c_str();
    out->id = ov.id.value;
    out->type_class = static_cast<uint8_t>(ov.type);
    out->has_default = ov.is_initialized;
    const tint::sem::GlobalVariable* global = find_override_global(program, ov);
    // Tint assigns an id to every override (sequential without `@id`); only an
    // explicit `@id(N)` attribute counts as an explicit id for WebGPU's
    // constant-key rules.
    out->has_explicit_id =
        global != nullptr &&
        tint::ast::HasAttribute<tint::ast::IdAttribute>(global->Declaration()->attributes);
    out->default_value = ov.is_initialized ? override_default_value(global, ov.type) : 0.0;
}

}  // namespace

extern "C" {

void yawgpu_tint_initialize(void) {
    // Tint's InternalCompilerError destructor is [[noreturn]]. This Dawn/Tint
    // revision exposes optional per-ICE callbacks, but no global setter, so the
    // shim cannot make ICEs catchable or install a process-wide capture hook.
    tint::Initialize();
}

YawgpuTintProgram* yawgpu_tint_program_create(const char* wgsl,
                                              size_t wgsl_len,
                                              bool shader_f16,
                                              bool subgroups,
                                              bool dual_source_blending,
                                              bool clip_distances,
                                              bool primitive_index,
                                              bool allow_framebuffer_fetch,
                                              const uint32_t* lang_features,
                                              size_t n_lang_features,
                                              char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    try {
        if (wgsl == nullptr) {
            set_error_string(err, "WGSL source pointer is NULL");
            return nullptr;
        }

        auto out = std::make_unique<YawgpuTintProgram>();
        out->file =
            std::make_unique<tint::Source::File>("shader.wgsl", std::string(wgsl, wgsl_len));

        tint::wgsl::reader::Options options;
        if (n_lang_features > 0 && lang_features == nullptr) {
            set_error_string(err, "WGSL language feature pointer is NULL");
            return nullptr;
        }
        for (size_t i = 0; i < n_lang_features; ++i) {
            if (auto feature = to_tint_language_feature(lang_features[i])) {
                options.allowed_features.features.insert(*feature);
            }
        }
        if (shader_f16) {
            options.allowed_features.extensions.insert(tint::wgsl::Extension::kF16);
        }
        if (subgroups) {
            options.allowed_features.extensions.insert(tint::wgsl::Extension::kSubgroups);
        }
        if (dual_source_blending) {
            options.allowed_features.extensions.insert(
                tint::wgsl::Extension::kDualSourceBlending);
        }
        if (clip_distances) {
            options.allowed_features.extensions.insert(tint::wgsl::Extension::kClipDistances);
        }
        if (primitive_index) {
            options.allowed_features.extensions.insert(tint::wgsl::Extension::kPrimitiveIndex);
        }
        if (allow_framebuffer_fetch) {
            options.allowed_features.extensions.insert(
                tint::wgsl::Extension::kChromiumExperimentalFramebufferFetch);
            options.allowed_features.extensions.insert(
                tint::wgsl::Extension::kChromiumInternalInputAttachments);
        }
        tint::Program parsed = tint::wgsl::reader::Parse(out->file.get(), options);
        if (!parsed.IsValid()) {
            set_error_string(err, parsed.Diagnostics().Str());
            return nullptr;
        }

        out->program = std::move(parsed);
        for (const auto& diagnostic : out->program.Diagnostics()) {
            if (diagnostic.severity == tint::diag::Severity::Error) {
                continue;
            }
            out->diagnostic_messages.push_back(diagnostic.message.Plain());
            out->diagnostic_severities.push_back(diagnostic_severity(diagnostic.severity));
        }
        tint::inspector::Inspector inspector(out->program);
        out->entry_points = inspector.GetEntryPoints();
        out->overrides = inspector.Overrides();
        if (inspector.has_error()) {
            set_error_string(err, inspector.error());
            return nullptr;
        }
        return out.release();
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return nullptr;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return nullptr;
    }
}

void yawgpu_tint_program_destroy(YawgpuTintProgram* program) {
    delete program;
}

size_t yawgpu_tint_entry_point_count(const YawgpuTintProgram* program) {
    return program != nullptr ? program->entry_points.size() : 0;
}

bool yawgpu_tint_entry_point_get(const YawgpuTintProgram* program,
                                 size_t i,
                                 YawgpuTintEntryPoint* out) {
    if (program == nullptr || out == nullptr || i >= program->entry_points.size()) {
        return false;
    }
    fill_entry_point(program->entry_points[i], out);
    return true;
}

size_t yawgpu_tint_entry_point_input_count(const YawgpuTintProgram* program, const char* ep) {
    const auto* entry = find_entry_point(program, ep);
    return entry != nullptr ? entry->input_variables.size() : 0;
}

bool yawgpu_tint_entry_point_input_get(const YawgpuTintProgram* program,
                                       const char* ep,
                                       size_t i,
                                       YawgpuTintStageVariable* out) {
    const auto* entry = find_entry_point(program, ep);
    if (entry == nullptr || out == nullptr || i >= entry->input_variables.size()) {
        return false;
    }
    fill_stage_variable(entry->input_variables[i], out);
    return true;
}

size_t yawgpu_tint_entry_point_output_count(const YawgpuTintProgram* program, const char* ep) {
    const auto* entry = find_entry_point(program, ep);
    return entry != nullptr ? entry->output_variables.size() : 0;
}

bool yawgpu_tint_entry_point_output_get(const YawgpuTintProgram* program,
                                        const char* ep,
                                        size_t i,
                                        YawgpuTintStageVariable* out) {
    const auto* entry = find_entry_point(program, ep);
    if (entry == nullptr || out == nullptr || i >= entry->output_variables.size()) {
        return false;
    }
    fill_stage_variable(entry->output_variables[i], out);
    return true;
}

size_t yawgpu_tint_diagnostic_count(const YawgpuTintProgram* program) {
    return program != nullptr ? program->diagnostic_messages.size() : 0;
}

bool yawgpu_tint_diagnostic_get(const YawgpuTintProgram* program,
                                size_t i,
                                YawgpuTintDiagnostic* out) {
    if (program == nullptr || out == nullptr || i >= program->diagnostic_messages.size() ||
        i >= program->diagnostic_severities.size()) {
        return false;
    }
    out->message = program->diagnostic_messages[i].c_str();
    out->severity = program->diagnostic_severities[i];
    return true;
}

size_t yawgpu_tint_resource_binding_count(const YawgpuTintProgram* program, const char* ep) {
    if (program == nullptr) {
        return 0;
    }
    tint::inspector::Inspector inspector(program->program);
    return inspector.GetResourceBindings(cstr_or_empty(ep)).size();
}

bool yawgpu_tint_resource_binding_get(const YawgpuTintProgram* program,
                                      const char* ep,
                                      size_t i,
                                      YawgpuTintResourceBinding* out) {
    if (program == nullptr || out == nullptr) {
        return false;
    }
    tint::inspector::Inspector inspector(program->program);
    auto bindings = inspector.GetResourceBindings(cstr_or_empty(ep));
    if (i >= bindings.size()) {
        return false;
    }
    auto usages = texture_sample_usages(program->program, cstr_or_empty(ep));
    auto key = std::make_pair(bindings[i].bind_group, bindings[i].binding);
    auto found = usages.find(key);
    fill_resource_binding(bindings[i], found != usages.end() ? found->second : 0, out);
    return true;
}

size_t yawgpu_tint_override_count(const YawgpuTintProgram* program) {
    return program != nullptr ? program->overrides.size() : 0;
}

bool yawgpu_tint_override_get(const YawgpuTintProgram* program, size_t i, YawgpuTintOverride* out) {
    if (program == nullptr || out == nullptr || i >= program->overrides.size()) {
        return false;
    }
    fill_override(program->program, program->overrides[i], out);
    return true;
}

bool yawgpu_tint_generate_msl(const YawgpuTintProgram* program,
                              const char* ep,
                              const YawgpuTintBindings* bindings,
                              const YawgpuTintOverrideValue* ov,
                              size_t n_ov,
                              uint32_t buffer_sizes_slot,
                              bool disable_robustness,
                              bool emit_vertex_point_size,
                              const YawgpuTintVertexBuffer* vertex_buffers,
                              size_t n_vertex_buffers,
                              uint32_t fixed_sample_mask,
                              YawgpuTintMslOutput* out,
                              char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (out != nullptr) {
        out->msl = nullptr;
        out->entry_point = nullptr;
        out->needs_storage_buffer_sizes = false;
        out->buffer_size_bindings = nullptr;
        out->n_buffer_size_bindings = 0;
        out->workgroup_allocations = nullptr;
        out->n_workgroup_allocations = 0;
        out->has_frag_depth_clamp = false;
        out->frag_depth_clamp_slot = 0;
    }
    try {
        if (program == nullptr || out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        std::string remapped_entry_point = "tint_" + entry_point;
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::msl::writer::Options options;
        options.entry_point_name = entry_point;
        options.remapped_entry_point_name = remapped_entry_point;
        options.disable_robustness = disable_robustness;
        // Point-list topology on Metal requires every vertex to set [[point_size]];
        // Tint emits it (= 1.0) when asked. yawgpu threads this from the render
        // pipeline's force_point_size.
        options.emit_vertex_point_size = emit_vertex_point_size;
        options.fixed_sample_mask = fixed_sample_mask;
        options.bindings = all_remaps_empty(bindings)
                               ? tint::GenerateBindings(ir.Get(), entry_point, true, true)
                               : make_bindings(bindings);
        if (bindings != nullptr && bindings->input_attachment_color_index != nullptr) {
            for (size_t i = 0; i < bindings->n_input_attachment_color_index; ++i) {
                const auto& e = bindings->input_attachment_color_index[i];
                options.input_attachment_to_color_index[tint::BindingPoint{e.group, e.binding}] =
                    e.color_slot;
            }
        }
        options.immediate_binding_point =
            choose_immediate_binding_point(options.bindings, buffer_sizes_slot);
        const tint::inspector::EntryPoint* ep_info = find_entry_point(program, entry_point.c_str());
        const bool has_frag_depth_clamp =
            ep_info != nullptr && ep_info->frag_depth_used &&
            options.immediate_binding_point.has_value();
        uint32_t frag_depth_clamp_slot = 0;
        if (has_frag_depth_clamp) {
            options.depth_range_offsets =
                tint::msl::writer::Options::RangeOffsets{/*min=*/0u, /*max=*/4u};
            frag_depth_clamp_slot = options.immediate_binding_point->binding;
        }
        std::vector<tint::BindingPoint> ordered_size_bindings;
        options.array_length_from_constants = generate_array_length_from_constants(
            ir.Get(), entry_point, buffer_sizes_slot, ordered_size_bindings);
        const auto num_storage = static_cast<uint32_t>(ordered_size_bindings.size());
        if (n_vertex_buffers > 0) {
            if (vertex_buffers == nullptr) {
                set_error_string(err, "vertex buffer pointer is NULL");
                return false;
            }
            tint::VertexPullingConfig config;
            config.pulling_group = 4u;
            uint32_t max_slot = 0;
            for (size_t i = 0; i < n_vertex_buffers; ++i) {
                max_slot = std::max(max_slot, vertex_buffers[i].slot);
            }
            config.vertex_state.resize(max_slot + 1);
            for (size_t i = 0; i < n_vertex_buffers; ++i) {
                const auto& buffer = vertex_buffers[i];
                std::vector<tint::VertexAttributeDescriptor> attributes;
                attributes.reserve(buffer.n_attributes);
                if (buffer.n_attributes > 0 && buffer.attributes == nullptr) {
                    set_error_string(err, "vertex attribute pointer is NULL");
                    return false;
                }
                for (size_t attr_idx = 0; attr_idx < buffer.n_attributes; ++attr_idx) {
                    const auto& attribute = buffer.attributes[attr_idx];
                    attributes.push_back(tint::VertexAttributeDescriptor{
                        .format = to_tint_vertex_format(attribute.format),
                        .offset = attribute.offset,
                        .shader_location = attribute.shader_location,
                    });
                }
                config.vertex_state[buffer.slot] = tint::VertexBufferLayoutDescriptor{
                    buffer.array_stride,
                    to_tint_step_mode(buffer.step_mode),
                    std::move(attributes),
                };
                tint::BindingPoint src{.group = 4u, .binding = buffer.slot};
                options.bindings.storage[src] =
                    tint::BindingPoint{.group = 0u, .binding = buffer.metal_index};
                options.array_length_from_constants.bindpoint_to_size_index[src] =
                    num_storage + static_cast<uint32_t>(i);
            }
            options.vertex_pulling_config = std::move(config);
        }
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::msl::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        out->msl = dup_string(result->msl);
        if (out->msl == nullptr) {
            set_error_string(err, "failed to allocate MSL output");
            return false;
        }
        out->entry_point = dup_string(remapped_entry_point);
        if (out->entry_point == nullptr) {
            std::free(out->msl);
            out->msl = nullptr;
            set_error_string(err, "failed to allocate MSL entry point output");
            return false;
        }
        out->needs_storage_buffer_sizes = result->needs_storage_buffer_sizes;
        out->has_frag_depth_clamp = has_frag_depth_clamp;
        out->frag_depth_clamp_slot = frag_depth_clamp_slot;
        out->buffer_size_bindings = dup_binding_pairs(ordered_size_bindings);
        out->n_buffer_size_bindings = ordered_size_bindings.size();
        if (!ordered_size_bindings.empty() && out->buffer_size_bindings == nullptr) {
            std::free(out->msl);
            std::free(out->entry_point);
            out->msl = nullptr;
            out->entry_point = nullptr;
            out->n_buffer_size_bindings = 0;
            set_error_string(err, "failed to allocate MSL buffer size bindings");
            return false;
        }
        out->n_workgroup_allocations = result->workgroup_allocations.size();
        if (result->workgroup_allocations.empty()) {
            out->workgroup_allocations = nullptr;
            out->n_workgroup_allocations = 0;
        } else {
            out->workgroup_allocations = static_cast<uint32_t*>(
                std::malloc(result->workgroup_allocations.size() * sizeof(uint32_t)));
            if (out->workgroup_allocations == nullptr) {
                std::free(out->msl);
                std::free(out->entry_point);
                std::free(out->buffer_size_bindings);
                out->msl = nullptr;
                out->entry_point = nullptr;
                out->buffer_size_bindings = nullptr;
                out->n_buffer_size_bindings = 0;
                out->n_workgroup_allocations = 0;
                set_error_string(err, "failed to allocate MSL workgroup allocations");
                return false;
            }
            std::memcpy(out->workgroup_allocations,
                        result->workgroup_allocations.data(),
                        result->workgroup_allocations.size() * sizeof(uint32_t));
        }
        return true;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

bool yawgpu_tint_generate_spirv(const YawgpuTintProgram* program,
                                const char* ep,
                                const YawgpuTintBindings* bindings,
                                const YawgpuTintOverrideValue* ov,
                                size_t n_ov,
                                bool disable_robustness,
                                bool use_vulkan_memory_model,
                                uint32_t framebuffer_fetch_descriptor_set,
                                bool multisampled_input_attachment,
                                bool has_polyfill_pixel_center,
                                uint32_t polyfill_pixel_center,
                                uint32_t** words_out,
                                size_t* n_words_out,
                                char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (words_out != nullptr) {
        *words_out = nullptr;
    }
    if (n_words_out != nullptr) {
        *n_words_out = 0;
    }
    try {
        if (program == nullptr || words_out == nullptr || n_words_out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::spirv::writer::Options options;
        options.entry_point_name = entry_point;
        options.disable_robustness = disable_robustness;
        options.extensions.use_vulkan_memory_model = use_vulkan_memory_model;
        options.multisampled_input_attachment = multisampled_input_attachment;
        // Pixel-center polyfill for @builtin(position): under Vulkan sample-rate
        // shading FragCoord.xy/z/w reflect the sample location, but WebGPU
        // requires the pixel-center (fragment) position. When set, Tint's SPIR-V
        // shader_io raise reconstructs the pixel center from a center-sampled
        // interpolant at this free inter-stage location (matches Dawn's Vulkan
        // backend, which sets the same option).
        if (has_polyfill_pixel_center) {
            options.polyfill_pixel_center = polyfill_pixel_center;
            // The pixel-center reconstruction maps the recovered NDC depth into
            // the viewport depth range (min/max depth), which are supplied at
            // draw time as push constants. Two f32 immediates at byte offsets 0
            // (min) and 4 (max); the Vulkan HAL declares a matching fragment
            // push-constant range and writes the viewport min/max depth. Matches
            // Dawn's ClampFragDepthArgs layout.
            options.depth_range_offsets =
                tint::spirv::writer::Options::RangeOffsets{/*min=*/0u, /*max=*/4u};
        }
        options.bindings = all_remaps_empty(bindings)
                               ? tint::GenerateBindings(ir.Get(), entry_point, false, false)
                               : make_bindings(bindings);
        options.colour_index_to_binding_point =
            color_bindings_for_entry_point(
                program, entry_point.c_str(), framebuffer_fetch_descriptor_set);
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::spirv::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        size_t byte_len = result->spirv.size() * sizeof(uint32_t);
        auto* words = static_cast<uint32_t*>(std::malloc(byte_len));
        if (words == nullptr && byte_len != 0) {
            set_error_string(err, "failed to allocate SPIR-V output");
            return false;
        }
        if (byte_len != 0) {
            std::memcpy(words, result->spirv.data(), byte_len);
        }
        *words_out = words;
        *n_words_out = result->spirv.size();
        return true;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

bool yawgpu_tint_workgroup_storage_size(const YawgpuTintProgram* program,
                                        const YawgpuTintOverrideValue* ov,
                                        size_t n_ov,
                                        uint64_t* out,
                                        char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (out != nullptr) {
        *out = 0;
    }
    try {
        if (program == nullptr || out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        auto cfg = make_override_config(program, ov, n_ov);
        if (cfg != tint::Success) {
            *out = 0;
            return true;
        }
        auto sub = tint::core::ir::transform::SubstituteOverrides(ir.Get(), cfg.Get());
        if (sub != tint::Success) {
            *out = 0;
            return true;
        }
        auto wi = tint::core::ir::GetWorkgroupInfo(ir.Get());
        *out = (wi == tint::Success) ? wi->storage_size : 0;
        return true;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

bool yawgpu_tint_generate_glsl(const YawgpuTintProgram* program,
                               const char* ep,
                               const YawgpuTintBindings* bindings,
                               const YawgpuTintOverrideValue* ov,
                               size_t n_ov,
                               char** glsl_out,
                               char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (glsl_out != nullptr) {
        *glsl_out = nullptr;
    }
    try {
        if (program == nullptr || glsl_out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::glsl::writer::Options options;
        options.entry_point_name = entry_point;
        options.version = tint::glsl::writer::Version();
        if (all_remaps_empty(bindings)) {
            auto generated = tint::glsl::writer::GenerateBindings(ir.Get(), entry_point);
            options.bindings = std::move(generated.bindings);
            options.texture_builtins_from_uniform =
                std::move(generated.texture_builtins_from_uniform);
        } else {
            options.bindings = make_bindings(bindings);
        }
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::glsl::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        *glsl_out = dup_string(result->glsl);
        return *glsl_out != nullptr;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

void yawgpu_tint_string_free(char* s) {
    std::free(s);
}

void yawgpu_tint_u32_free(uint32_t* words) {
    std::free(words);
}

}  // extern "C"
