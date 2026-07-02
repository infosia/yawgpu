# Block 93 — WGSL language features

Owns the instance-level WGSL language-feature surface:
`wgpuInstanceGetWGSLLanguageFeatures` / `wgpuInstanceHasWGSLLanguageFeature`
and the invariant that ties it to the Tint frontend.

## Contract

1. **Single canonical set.** `yawgpu-core::SUPPORTED_WGSL_LANGUAGE_FEATURES`
   (`yawgpu-core/src/wgsl_language_features.rs`) is the one source of truth.
   The FFI queries read it, and the Tint WGSL reader's
   `allowed_features.features` is built from the same set
   (`shader_tint.rs` → `yawgpu_tint_program_create`). By construction the
   API answer and compiler acceptance can never diverge (the failure mode
   CTS `shader,validation,parse,requires:wgsl_matches_api` checks).
2. **Advertising bar.** A feature enters the set only when yawgpu both
   *compiles* it and either *executes* it correctly or *deterministically
   validates* it at pipeline creation (established when the subgroup
   language features were withheld until the subgroup runtime existed).
3. Enum values are the canonical `webgpu.h` values (Dawn `dawn.json`
   `"wgsl language feature name"`); the core unit test
   `supported_wgsl_language_features_match_canonical_api_values` pins them.

## Current set

`readonly_and_readwrite_storage_textures` (1),
`packed_4x8_integer_dot_product` (2), `unrestricted_pointer_parameters` (3),
`pointer_composite_access` (4), `uniform_buffer_standard_layout` (5),
`subgroup_id` (6), `texture_and_sampler_let` (7), `subgroup_uniformity` (8),
`texture_formats_tier1` (9), `linear_indexing` (10),
`immediate_address_space` (11 — this block).

## immediate_address_space (0x0B)

Dawn advertises `immediate_address_space` (Tint status:
shipped-with-killswitch) while reporting `maxImmediateSize = 0` on the same
builds yawgpu is compared against: the *parser* accepts
`requires immediate_address_space;` and `var<immediate>` declarations, and
*pipeline creation* rejects any actual use because the shader's immediate
size exceeds the layout's `immediateSize` budget. yawgpu adopts the same
posture (yawgpu also reports `maxImmediateSize = 0` on every backend —
`HalLimits::DEFAULT`, no backend overrides it).

### Behaviour contract

- `wgpuInstanceHasWGSLLanguageFeature(instance, ImmediateAddressSpace)`
  returns true; the feature is present in
  `wgpuInstanceGetWGSLLanguageFeatures`.
- `createShaderModule` accepts `requires immediate_address_space;` and
  modules declaring `var<immediate>` globals (no validation error).
- **New pipeline-creation rule (both compute and render, every stage):**
  the entry point's *immediate data size* — the total byte size of all
  immediate-address-space variables it statically accesses, as reported by
  Tint reflection (`inspector::EntryPoint::immediate_data_size`) — must be
  ≤ the pipeline layout's `immediate_size`. Violation → captured
  **validation error** at `create*Pipeline` (never a HAL error, never a
  panic). This rule is backend-independent (fires identically on Noop).
  With `maxImmediateSize = 0` today, every layout has `immediate_size = 0`,
  so any entry point that actually touches a `var<immediate>` is rejected
  deterministically; a module that merely *declares* one (unused by the
  entry point) has size 0 and passes.
- Default (auto) pipeline layouts budget the **device's
  `maxImmediateSize`** (not 0) — CTS
  `pipeline,immediates:pipeline_creation_immediate_size_mismatch` requires
  a shader using up to `maxImmediateSize` to succeed under an auto layout.
  (Historical note: this section originally said auto layouts reserve 0
  bytes; that was unobservable — and wrong — while every backend reported
  `maxImmediateSize = 0`, and was corrected during Block 94 S2 when the
  Metal limit became 64 and the CTS cases started running.)

### Implementation notes

- `yawgpu/ffi/webgpu-headers/webgpu.h`: add
  `WGPUWGSLLanguageFeatureName_ImmediateAddressSpace = 0x0000000B` (bindgen
  regenerates the Rust binding).
- `yawgpu-core/src/wgsl_language_features.rs`: constant + list entry + test
  update.
- `yawgpu-tint/shim`: `to_tint_language_feature` case 11 →
  `tint::wgsl::LanguageFeature::kImmediateAddressSpace`; expose the entry
  point's `immediate_data_size` through the entry-point reflection FFI
  (extends the cached per-entry reflection, Block 90 unit-test rules apply).
- `yawgpu-core`: surface `immediate_data_size` on the reflected entry point
  and enforce the pipeline-creation rule in compute/render pipeline
  validation.
- GLES (Tier 2) note: Tint's GLSL backend uses an *internal* immediate slot
  (`tint_immediates[0]`, Block 67) for `first_instance`. Because user
  immediates are rejected at core validation while `maxImmediateSize = 0`,
  no user immediate ever reaches the GLES HAL; revisit the slot layout only
  if `maxImmediateSize` is ever raised.

### Out of scope (future work if `maxImmediateSize` is raised)

`wgpu*SetImmediateData` command surface, HAL immediate binding
(Metal `setBytes` / Vulkan push constants), and non-zero
`maxImmediateSize` reporting. The vendored `webgpu.h` does not declare the
`SetImmediateData` entry points yet.

## Unit tests (Block 90 discipline)

- Core: list contains 11 / canonical-value test updated; pipeline-creation
  rule — `var<immediate>` used by entry point → validation error (explicit
  layout and auto layout); declared-but-unused → pipeline creates fine;
  `requires immediate_address_space;` module compiles.
- FFI: `wgpuInstanceHasWGSLLanguageFeature(ImmediateAddressSpace)` true +
  present in `GetWGSLLanguageFeatures`.
- Shim/`yawgpu-tint`: `immediate_data_size` reflection (0 without use,
  correct byte size with use).
