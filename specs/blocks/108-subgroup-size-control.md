# Block 108 — `subgroup-size-control` on Vulkan (`@subgroup_size`)

Status: **S1 DONE (2026-09-26)** — core + FFI + Tint shim + Noop (commit
"Block 108 S1"); coding agent: Claude subagent (codex usage limit). S1
findings: **R4 rule 4 is core-enforced** — Tint rejects a zero /
non-power-of-two *const* `@subgroup_size` at parse time, but
`SubstituteOverrides` does not check an override-driven value (`sg=6` →
`Some(6)`, `sg=0` → `Some(0)`), so `validate_explicit_subgroup_size`
rejects it before rule 1; the Inspector does not reflect the attribute, so
the shim reads its presence from the AST and the override-free fast path
defers such entry points to the IR resolve. Gates: `cargo test --workspace`
green, fmt + clippy default / `vulkan` / `tiled` / `gles` clean; `metal`
not built (Windows host — Mac compile check owed).
**S2 + S3 DONE (2026-09-26)** — Vulkan HAL (advertisement predicate incl.
`requiredSubgroupSizeStages ∋ COMPUTE`, device enable, `RequiredSubgroupSize`
+ `REQUIRE_FULL_SUBGROUPS` / `ALLOW_VARYING_SUBGROUP_SIZE`) and
`e2e_vulkan_subgroup_size_control.rs`. Native NVIDIA caps
`{min 32, max 32, maxComputeWorkgroupSubgroups 32}`. Under
`VK_LAYER_KHRONOS_validation` with 0 VUID lines: e2e 3/3, the
`e2e_vulkan_subgroups` 4/4 + `e2e_vulkan_compute` 3/3 `ALLOW_VARYING`
regression guard, and all 55 ignored Vulkan HAL tests. A layer negative check
(a `(1,1,1)` module with required size 32 → VUID-02757) proves the chain
reaches the driver. S4 (CTS) next.
Raised by the 2026-09-26 native-Vulkan CTS skip audit (Windows 11, NVIDIA
RTX 5060 Ti, yawgpu `2c6ea6f`): after the ~381k ASTC / ETC2 / EAC hardware
skips and the structural / C-API-N/A skips are removed,
`subgroup-size-control` is the only skipped WebGPU feature yawgpu can
implement on this host.
Ledger: `specs/tracking/subgroups.md` ("Block 108" section).

## Problem

`WGPUFeatureName_SubgroupSizeControl = 0x17` is in the pinned
`webgpu.h`, but yawgpu has no `Feature::SubgroupSizeControl`, never
advertises it, and the Tint shim never allows the
`subgroup_size_control` WGSL extension. A shader containing
`enable subgroup_size_control;` or `@subgroup_size(N)` is therefore
rejected on every device, and the CTS cases that need the feature skip:

- `webgpu:shader,execution,shader_io,compute_builtins:subgroup_size_attribute:*`
  (6 cases; the port also compiles the body out for non-Dawn backends — see S4)
- `webgpu:shader,validation,extension,subgroup_size_control:*` (skips via a
  behavioural probe: `enable subgroup_size_control;` does not compile)
- `webgpu:api,validation,capability_checks,features,subgroup_size_control:*`
  (`.unimplemented(...)` in the port — see S4)

The extension itself (Tint doc
`third_party/dawn/docs/tint/extensions/subgroup_size_control.md`): a
compute entry point may carry `@subgroup_size(S)`; the pipeline then runs
with exactly subgroup size `S`, and `@builtin(subgroup_size)` reads `S`.

## Oracle (Dawn)

| Aspect | Dawn | Source |
|---|---|---|
| Backends | D3D12 (SM 6.6 `[WaveSize]`) and **Vulkan**; **not Metal** | `docs/tint/extensions/subgroup_size_control.md` "Availability" |
| Vulkan advertisement | `Subgroups` supported **and** `VK_EXT_subgroup_size_control` with `subgroupSizeControl` **and** `computeFullSubgroups` | `native/vulkan/PhysicalDeviceVk.cpp` (`hasComputeFullSubgroups`, `EnableFeature(Feature::SubgroupSizeControl)`) |
| Implied feature | enabling `SubgroupSizeControl` enables `Subgroups` | `native/Device.cpp` (feature implications) |
| WGSL gate | `Extension::kSubgroupSizeControl` allowed iff the device has the feature | `native/Device.cpp` (`mWGSLAllowedFeatures`) |
| Validation 1 | `workgroup_size.x % S == 0` | `native/ShaderModule.cpp` `ValidateComputeStageWorkgroupSize` |
| Validation 2 | `S` in `[minSubgroupSize, maxSubgroupSize]` of `VkPhysicalDeviceSubgroupSizeControlProperties` | `native/vulkan/ShaderModuleVk.cpp` (after `Generate`, i.e. after override substitution) |
| Validation 3 | `(x·y·z) / S <= maxComputeWorkgroupSubgroups` | same |
| Pipeline (explicit `S`) | chain `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{S}` + `REQUIRE_FULL_SUBGROUPS_BIT` | `native/vulkan/ComputePipelineVk.cpp` |
| Pipeline (no `S`, ext present) | `ALLOW_VARYING_SUBGROUP_SIZE_BIT` (so `subgroup_size` reports the real width) | same |
| Exposure | **Experimental** — gated behind the `allow_unsafe_apis` toggle | `native/Features.cpp`, `native/PhysicalDevice.cpp` |

**Oracle caveat.** The webgpu-native-cts Dawn build does not set
`allow_unsafe_apis`, so the Dawn oracle **skips** these cases too. yawgpu
results are checked by the CTS cases' own assertions (the execution case
checks itself: it compares `subgroup_size` with the attribute) and by
source parity with the rows above, not by a Dawn run. (Optional: a Dawn
oracle run with the toggle forced on; see Open questions.)

**Exposure decision.** yawgpu has no unsafe-API toggle. The enum is in
the canonical `webgpu.h`, so yawgpu advertises the feature whenever the
hardware condition holds (R2). This matches how `texture-component-swizzle`
and `primitive-index` are advertised.

## Behaviour contract

### R1 — Feature plumbing (core + FFI)

- `yawgpu-core`: new `Feature::SubgroupSizeControl`, advertised by
  `Adapter` iff `HalAdapter::subgroup_size_control_caps()` (R2) is `Some`.
- `apply_feature_implications`: `SubgroupSizeControl` ⇒ `Subgroups`
  (Dawn parity). Advertisement guarantees the adapter also supports
  `Subgroups`.
- `yawgpu/src/conv/feature.rs`: `WGPUFeatureName_SubgroupSizeControl` ↔
  `Feature::SubgroupSizeControl`, in both directions.
- Requesting the feature on an adapter that lacks it fails `requestDevice`,
  like every other unsupported feature. No new code path is needed for this.
- **No new `WGPULimits` / `WGPUAdapterInfo` fields.** `webgpu.h` has no
  `maxComputeWorkgroupSubgroups` or explicit-size range. The explicit
  range **is** `WGPUAdapterInfo.subgroupMinSize` / `subgroupMaxSize`,
  which on Vulkan already come from
  `VkPhysicalDeviceSubgroupSizeControlProperties` (Block 62). The CTS
  enumerates candidate sizes from exactly those two fields.
  `maxComputeWorkgroupSubgroups` stays internal to core (R4).

### R2 — HAL advertisement

New `HalAdapter::subgroup_size_control_caps() -> Option<HalSubgroupSizeControlCaps>`
(static enum dispatch), where
`HalSubgroupSizeControlCaps { min_size, max_size, max_compute_workgroup_subgroups }`.

- **Vulkan** — `Some` iff **all** of the following hold:
  - `supports_subgroups()`
  - `VK_EXT_subgroup_size_control` is in the device-extension list. The
    extension is **required**: the instance requests
    `YAWGPU_VULKAN_API_VERSION = 1.1`, so the Vulkan 1.3 core promotion
    cannot be enabled at device creation.
  - `VkPhysicalDeviceSubgroupSizeControlFeatures.subgroupSizeControl == TRUE`
  - `...computeFullSubgroups == TRUE`
  - `VkPhysicalDeviceSubgroupSizeControlProperties.requiredSubgroupSizeStages`
    contains `COMPUTE`. This is **stricter than Dawn**, which does not check
    it. It is a hard Vulkan requirement for chaining a required size on a
    compute stage (VUID-VkPipelineShaderStageCreateInfo-pNext-02755), so a
    driver without it must not advertise the feature. Added in the S2 review.
  - Values come from `VkPhysicalDeviceSubgroupSizeControlProperties`
    `{minSubgroupSize, maxSubgroupSize, maxComputeWorkgroupSubgroups}`.
    `min_size` / `max_size` must equal `subgroup_size_range()` (same
    source, same `[4, 128]` validation).
  - Implement the decision as a pure predicate with a unit test, in the
    style of `subgroup_size_control_available`.
- **Metal** — `None` (Dawn: not supported; Apple GPUs have a fixed width of 32).
- **GLES** — `None` (Tier 2; add a catalogue row to Block 67's mapping
  matrix: "`subgroup-size-control` — not advertised").
- **Noop** — `Some { min_size: 4, max_size: 4, max_compute_workgroup_subgroups: 64 }`.
  These are nominal values chosen so the core rules in R4 can be tested
  without a GPU (mirrors `Subgroups` Noop = `4..=4`).

### R3 — Shader-module gate (Tint shim)

- `yawgpu_tint_program_create` gains a `subgroup_size_control` flag,
  threaded exactly like `subgroups` (`Device::create_shader_module` →
  `ShaderModule::from_wgsl` → `Program::parse` → the shim). When the flag is
  set, the shim inserts `tint::wgsl::Extension::kSubgroupSizeControl`.
- A device **without** the feature rejects `enable subgroup_size_control;`
  at `createShaderModule`, identically on every backend, Noop and GLES
  included.

### R4 — Compute pipeline validation (core, tier-independent)

This applies when the compute entry point has `@subgroup_size(S)`.

- `S` is read **after override substitution**, from Tint's
  `WorkgroupInfo.subgroup_size`, on the same path as the workgroup size.
  Extend `yawgpu_tint_resolved_workgroup_size` (or add a sibling query) to
  also return the resolved `subgroup_size`, and expose it on
  `ReflectedWorkgroupSize`.
- The literal-size fast path in `resolved_compute_workgroup_size` must not
  drop the attribute. Either reflect it there, or take the IR path whenever
  the entry point declares `@subgroup_size`.

Rules, checked in Dawn's order. All errors are **validation errors** from
`createComputePipeline` (sync and async), captured by error scopes:

1. `workgroup_size.x % S != 0` → error:
   "x-dimension of workgroup invocations (X) is not a multiple of the
   subgroup_size attribute (S)".
2. `S < caps.min_size || S > caps.max_size` → error:
   "subgroup_size attribute (S) is not in the allowed range ([min, max])".
3. `(x·y·z) / S > caps.max_compute_workgroup_subgroups` → error:
   "number of subgroups per workgroup (N) exceeds the maximum (M)".
4. **Power of two / non-zero.** Tint enforces this for const expressions.
   For override expressions, first establish which layer rejects it (Tint
   during `SubstituteOverrides`, or not at all). If Tint does not reject,
   core rejects `S == 0` or a non-power-of-two `S` **before** rule 1. The
   CTS `subgroup_size_override_*` cases pin the expected outcome. Record
   the finding in the S1 REPORT.

The caps come from the device's adapter. The feature is only enabled where
`caps` is `Some` (R1/R2), so the rules cannot run without caps.

`HalComputePipelineDescriptor` / `HalDevice::create_compute_pipeline`
gain `required_subgroup_size: Option<u32>`, which is `Some(S)` exactly
when the entry point declares the attribute.

### R5 — Vulkan HAL lowering

- **Device creation:** when R2's predicate holds, push
  `VK_EXT_subgroup_size_control` and chain
  `VkPhysicalDeviceSubgroupSizeControlFeatures{subgroupSizeControl, computeFullSubgroups}`
  into `VkDeviceCreateInfo`. This happens independently of whether the
  WebGPU feature was requested, matching Dawn's device-level extension use.
- **`create_compute_pipeline`, `Some(S)`:** chain
  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{requiredSubgroupSize: S}`
  into the stage and set `REQUIRE_FULL_SUBGROUPS_BIT`.
- **`create_compute_pipeline`, `None`, extension enabled on the device:**
  set `ALLOW_VARYING_SUBGROUP_SIZE_BIT` (Dawn parity; without it
  `@builtin(subgroup_size)` may report a width that differs from the
  dispatched one on variable-width GPUs).
  - This changes behaviour for every existing subgroup compute pipeline on
    such hosts. On NVIDIA and Apple (min == max) it is a no-op. It closes
    the deviation noted in Block 62 ("does not yet create
    varying-subgroup-size pipelines"); annotate Block 62, do not rewrite it.
  - Never set both flags (VUID-VkPipelineShaderStageCreateInfo-pNext-02754).
- **Metal / GLES:** `Some(_)` is unreachable by validation (feature not
  advertised) → return `HalError`, never ignore silently. **Noop** ignores it.
- **Out of scope:** Dawn's `FindDefaultComputeSubgroupSize` heuristic, which
  forces `2·min` on variable-width GPUs (Intel only) when no attribute is
  given. yawgpu leaves the driver's choice.

## Slices

| Slice | Owner | Content |
|---|---|---|
| S1 | coding agent | R1 + R3 + R4 + the HAL descriptor field + R2's Noop / Metal / GLES arms; shim changes; inline unit tests |
| S2 | coding agent | R2 Vulkan arm + R5; pure-predicate unit tests; HAL `#[ignore]` real-device smoke (pipeline with `Some(min_size)` creates) |
| S3 | Claude | `yawgpu/tests/e2e_vulkan_subgroup_size_control.rs` (see Tests); native NVIDIA run under `VK_LAYER_KHRONOS_validation` |
| S4 | Claude | webgpu-native-cts port update (separate repo, separate commit) + CTS runs |
| S5 | Claude | Phase Review (fresh-context reviewer over the block's range) + fixes via the coding agent |

S1 and S2 may be one handoff if the diff stays reviewable; S1 alone must
be green on Noop.

## Tests

### Unit (`#[cfg(test)]`, Noop / GPU-free)

- `conv/feature.rs`: round-trip `0x17` ↔ `Feature::SubgroupSizeControl`.
- `adapter.rs`:
  - The Noop adapter advertises `SubgroupSizeControl`.
  - `apply_feature_implications` adds `Subgroups`.
  - A base feature set without HAL support does not contain it.
- Shader module:
  - Without the feature, `enable subgroup_size_control;` → error-sink validation error.
  - With the feature, the same module compiles.
  - `@subgroup_size` without `enable` → error.
- Compute pipeline (Noop caps `4..4`, 64 subgroups max):
  - `@workgroup_size(8) @subgroup_size(4)` succeeds.
  - `@workgroup_size(6) @subgroup_size(4)` → rule 1 error.
  - `@subgroup_size(8)` → rule 2 error.
  - `@workgroup_size(256) @subgroup_size(4)` succeeds (64 subgroups);
    `@workgroup_size(260)` or a 3D shape with 65 subgroups → rule 3 error.
  - Override-driven `S`: valid succeeds, an out-of-range value fails, a
    non-power-of-two value fails.
  - `required_subgroup_size` reaches the HAL descriptor as `Some(S)` /
    `None`. Use a Noop recorder or a pure helper.
  - The async variant reports the same errors.
- Vulkan (GPU-free pure fns):
  - The advertisement predicate: every missing condition → `None`.
  - Stage-flag selection: `Some` → `REQUIRE_FULL` only; `None` + extension →
    `ALLOW_VARYING` only; `None` without extension → empty.

### e2e (`yawgpu/tests/e2e_vulkan_subgroup_size_control.rs`, `#[ignore]`, run under the Khronos layer)

1. Feature advertised on this host, and `subgroupMinSize` / `subgroupMaxSize`
   match the driver (NVIDIA: 32 / 32).
2. For every power of two `S` in `[min, max]`, with `wgx = S·k` for
   `k ∈ {1, 2, 4}`: every invocation writes
   `subgroup_size == S && subgroupAdd(1u) == S` (all invocations active).
   All pass.
3. Validation errors for rules 1, 2 and 4 (override-driven) surface on the
   device error sink (rule 3 is unreachable behind the workgroup-size limit on
   the verified hosts, so it is Noop-tested only).
4. A plain `subgroups` compute pipeline (no attribute) still passes the
   Block 62 e2e (`e2e_vulkan_subgroups`) under the layer — the
   `ALLOW_VARYING` regression guard.
5. 0 VUID lines.

MoltenVK re-check owed on the Mac (MoltenVK exposes the extension; record
whether `computeFullSubgroups` is reported).

### CTS (release `--features vulkan`, native NVIDIA)

- Targets:
  - `webgpu:shader,execution,shader_io,compute_builtins:subgroup_size_attribute:*`
  - `webgpu:shader,validation,extension,subgroup_size_control:*`
  - `webgpu:api,validation,capability_checks,features,subgroup_size_control:*`
- Regression, fail 0 expected:
  - `webgpu:shader,execution,expression,call,builtin,subgroup*:*`
  - `webgpu:shader,execution,expression,call,builtin,quad*:*`
  - `webgpu:shader,validation,extension,*`
  - `webgpu:api,validation,capability_checks,features,*`
  - `webgpu:api,validation,compute_pipeline:*`
- Metal: the targets must still skip (feature not advertised), and the
  full `shader,validation` count must stay subcase-identical to Dawn.

## S4 — webgpu-native-cts port changes (not yawgpu code)

- `compute_builtins.spec.cpp` `subgroup_size_attribute`: drop the
  `#if !defined(CTS_BACKEND_DAWN)` unconditional skip. The enum is now in
  the pinned header; gate on
  `wgpuDeviceHasFeature(WGPUFeatureName_SubgroupSizeControl)` as upstream does.
- `capability_checks/features/subgroup_size_control.spec.cpp`
  `enables_subgroups`: implement it from upstream `b507bd1` (a device
  created with only `SubgroupSizeControl` reports `Subgroups`).
- `shader/validation/extension/subgroup_size_control.spec.cpp`: optionally
  replace the behavioural probe with the feature query. Keep the probe if
  wgpu-native builds lack the enum.
- Update `docs/FINDINGS.md` and the README tables after the sweep.

## Out of scope

- Metal (Dawn does not support it; fixed width 32 on Apple GPUs).
- GLES (Tier 2, not advertised).
- Exposing `maxComputeWorkgroupSubgroups` or the explicit range through a
  vendor `yawgpu.h` chained struct, as Dawn's
  `AdapterPropertiesExplicitComputeSubgroupSizeConfigs` does. Revisit if a
  consumer needs `maxComputeWorkgroupSubgroups`.
- Dawn's default-compute-subgroup-size heuristic (R5).
- Clamping `maxComputeInvocationsPerWorkgroup` by
  `maxComputeWorkgroupSubgroups · defaultComputeSubgroupSize`. Dawn does
  this only when that heuristic is active.

## Open questions

1. Should a Dawn oracle run with `allow_unsafe_apis` forced on (a local
   harness patch) be part of S4, to get a real oracle diff for the three
   target files? Default: no; the execution case checks itself.
2. R4 rule 4: which layer rejects a non-power-of-two override value?
   Resolve in S1 (see R4).
