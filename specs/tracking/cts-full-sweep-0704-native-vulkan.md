# CTS full-sweep 2026-07-04 (first Tint-enabled native run) — three yawgpu findings

Status: **lavapipe-VERIFIED + native-ANV re-measured — four fix rounds landed
2026-07-04; all repro queries emit zero VUIDs under lavapipe + validation layer**
(see "Re-verification" below). Round 1: sampled-read barriers,
`sampleRateShading`/`fragmentStoresAndAtomics` enables, zero-dim dispatch skip.
Round 2: render-pass storage-texture GENERAL transition + format-aware
`map_texture_usage` (depth-stencil attachment usage). Round 3: combined
depth-stencil aspect handling (image-own `aspect_flags` on `VulkanTextureInner`
for the default view and whole-image barriers). Round 4: `VERTEX_SHADER` in the
GENERAL-layout stage mask. Finding 3a verified fixed harness-side. The native-ANV
cluster re-runs (see below) confirm the finding-2 target subtree fixed and
`readonly_depth_stencil` down to 1; the residual ANV failures are driver-suspect
signatures pending Dawn-oracle comparison, plus finding 4. The two zero-dim
quarantined files and the finding-4 diagnosis remain. Surfaced by the first full CTS sweep with a working Tint frontend on
native Linux/Vulkan (Intel Iris 5100, Haswell, Mesa ANV), 2026-07-04. Earlier sweeps ran
with the stub compiler, so every shader/pipeline path silently skipped — these are
pre-existing bugs newly exposed, not regressions.

Sweep result: pass=867,249 fail=3,318 (~0.38% of executed), crash=0 real (2 timeouts, see
finding 3). Raw JSONL + per-file logs are kept locally on the sweep host (git-ignored, not
part of this repo).

Attribution method: each failing cluster was re-run on lavapipe (software Vulkan) and
under `VK_LAYER_KHRONOS_validation`. Findings 1–2 are yawgpu-side,
validation-layer-proven spec violations. Finding 3a was initially attributed to yawgpu
(driver-independent, reproduces on lavapipe) but the 2026-07-04 code investigation
re-attributed it to the CTS harness — see 3a below.

Repro environment for all commands below:

```sh
cd webgpu-native-cts
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json   # lavapipe
Y=/path/to/yawgpu/target-vulkan/release
export LD_LIBRARY_PATH="$(ls -d $Y/build/yawgpu-tint-*/out/build | head -1):$Y"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
./build-yawgpu-release/cts --workers 1 '<query>'
```

---

## Finding 1 — missing image-layout transition to SHADER_READ_ONLY before sampled reads

**Impact: largest failure cluster.** 1,335 fails in
`api,operation,texture_view,texture_component_swizzle` (int formats × textureGather/
textureLoad: gather=993, load=342), plus a likely share of the texture-builtin fails
(`textureLoad` 286, `textureGather` 126, `textureNumLayers` 78, `textureDimensions` 45,
`readonly_depth_stencil` 4 — re-measure these after the fix).

The validation layer reports, per draw that samples a freshly-uploaded texture:

```
UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout:
  command buffer expects VkImage (COLOR aspect, layer 0, mip 0) to be in layout
  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL -- instead, current layout is
  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
VUID-VkDescriptorImageInfo-imageLayout-00344   (descriptor layout mismatch at access time)
VUID-vkCmdDraw-None-08114                      (descriptor not valid when statically used)
```

i.e. the upload path leaves the image in `TRANSFER_DST_OPTIMAL` and the sampling path
never inserts a barrier transitioning it to `SHADER_READ_ONLY_OPTIMAL`.

**Key evidence — PASSING cases violate too.** A float-format control case
(`format="r8unorm";func="textureSample"`, passes everywhere) emits the *same* 3 VUIDs.
lavapipe and most ANV float paths tolerate the wrong layout by luck; ANV int-format
gather does not: uint texels come back as the bit pattern of f32 1.0 (`expected 1, got
1065353216 = 0x3F800000`). So the pass/fail split by format is driver mercy, not
correctness — the barrier is missing on every sampled read after upload.

Also emitted on the same runs (secondary, likely same code area):

```
VUID-RuntimeSpirv-NonWritable-06340
  fragmentStoresAndAtomics not enabled => fragment-stage storage buffers/images must be
  decorated NonWritable (SPIR-V decoration missing, or enable the feature when present).
```

Repro (fails on native ANV, passes-with-VUIDs on lavapipe; VUID counts per single case ~912/684/684/684):

```
webgpu:api,operation,texture_view,texture_component_swizzle:read_swizzle:format="r8uint";func="textureGather"
```

Control (passes, still emits the layout VUIDs — use to verify the barrier fix removes them):

```
webgpu:api,operation,texture_view,texture_component_swizzle:read_swizzle:format="r8unorm";func="textureSample"
```

Note: the 2026-06-20 F-126 review found the *copy* encode path validation-clean — that
review covered copies only, not the upload→sample transition, so it does not contradict
this finding.

## Finding 2 — MSAA / sample_mask path: multiple object-creation spec violations

**Impact: 1,398 fails** in `api,operation,render_pipeline,sample_mask` — the whole
`fragment_output_mask` subtree (both `interpolated=` variants) renders all-zero
(`GPU buffer mismatch at byte N: expected 255, got 0`; also
`alpha <= 0 result did not match zero coverage` ×90). The sibling `final_output` tests
pass (1,614), so plain MSAA rasterization works; the shader-output-mask path does not.

VUID histogram for ONE case (`fragment_output_mask:interpolated=false;sampleCount=1;rasterizationMask=1`):

```
729  VUID-VkImageViewCreateInfo-usage-08931          (view usage not a subset of image usage)
729  VUID-vkCmdBeginRenderPass-initialLayout-01758   (attachment initialLayout mismatch)
648  UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout   (finding 1 again)
486  VUID-VkShaderModuleCreateInfo-pCode-08740       (SPIR-V uses capabilities for features not enabled)
243  VUID-VkImageViewCreateInfo-subresourceRange-09594
243  VUID-VkImageViewCreateInfo-pNext-02662
243  VUID-VkImageMemoryBarrier-oldLayout-01209       (barrier oldLayout != actual layout)
243  VUID-VkImageCreateInfo-imageCreateMaxMipLevels-02251  (mipLevels > max for extent/samples)
243  VUID-VkFramebufferCreateInfo-pAttachments-02633
```

02251 is notable: creating an image with more mip levels than legal for its
extent/sampleCount (multisampled images must have exactly 1 mip). 08740 suggests the
sample-mask/per-sample SPIR-V needs a capability/feature (e.g. `sampleRateShading`)
that is not being enabled on the device. Expect one root object-creation bug in the
MSAA texture/view setup plus a feature-enable gap, not nine independent bugs.

Repro:

```
webgpu:api,operation,render_pipeline,sample_mask:fragment_output_mask:interpolated=false;sampleCount=1;rasterizationMask=1
```

## Finding 3 — popErrorScope never resolves for `current_scope` (hang), + zero-dim dispatch workaround

### 3a. popErrorScope hang — RE-ATTRIBUTED to the CTS harness (not a yawgpu bug)

`webgpu:api,validation,error_scope:current_scope:errorFilter="validation"` and
`errorFilter="out-of-memory"` hang until the harness 30s case timeout (the sweep's only
2 "crashes"). Initial hypothesis (popErrorScope future never completed) was **disproven**
by the 2026-07-04 code investigation:

- yawgpu's pop path is fully synchronous and O(1): `wgpuDevicePopErrorScope`
  (`yawgpu/src/ffi/device.rs`) pops the scope, reads the already-captured error, and
  registers the callback via `register_callback` (`yawgpu/src/ffi/mod.rs`), which marks
  the future Complete **in the same call**. The callback fires on the first
  `wgpuInstanceProcessEvents`. There is no deferral, queue dependency, or later event.
- The real cause: `current_scope` is the only error_scope test using
  `stackDepth=100000` (all others cap at 1000 — exactly the 2-hang / 15-pass split). Its
  body performs ~100,001 sequential synchronous pops, and the harness's `pumpUntil`
  (`webgpu-native-cts/src/common/webgpu/sync.cpp`) executes one unconditional 1 ms sleep
  per pop even when the callback fired on the first `processEvents` iteration.
  100,001 × ~1 ms ≈ 100 s ≫ the 20–30 s case timeout → reported as a hang.
- Corroboration: if the callback truly never fired, the first `popErrorScopeSync` would
  hit its own 5 s internal timeout and fail fast at ~5 s, not time out silently.

**Fix belongs in webgpu-native-cts**, not yawgpu: re-check `done()` before sleeping in
`pumpUntil` (or skip the sleep for already-signalled futures). No yawgpu change needed.

Repro (completes in <60 s via timeout; safe everywhere):

```
./build-yawgpu-release/cts --isolate --workers 1 --case-timeout-ms 20000 \
  'webgpu:api,validation,error_scope:current_scope:*'
```

### 3b. zero-dim dispatch early-out (Haswell freeze workaround)

Mesa ANV on Haswell hard-wedges the GPU (whole-machine freeze, no reset) on any
`vkCmdDispatch` with 0 in a dimension. Driver bug (proven with a hand-written pure-Vulkan
repro), but yawgpu can dodge the entire class with a spec-legal optimization: a dispatch
with any groupCount of 0 does nothing, so **skip emitting `vkCmdDispatch` when
x==0 || y==0 || z==0** in the HAL encoder. Two CTS files currently must be quarantined on
the Haswell host because of this (`api,validation,encoding,cmds,compute_pass`,
`api,validation,encoding,programmable,pipeline_bind_group_compat` — the latter dispatches
`(0,1,1)` from its `doCall()` helper and froze the machine twice on 2026-07-04); the
early-out would let both run. Same applies to `vkCmdDispatchIndirect`-free paths only —
indirect dispatches can't be pre-checked CPU-side, leave those alone.

## Finding 4 (minor) — multisampled sint textureLoad: command encoding errors on ANV only

18 cases `shader,execution,...,textureLoad:multisampled` fail with
`uncaptured error: queue submit cannot use an error command buffer` — exclusively **sint**
formats (r8/rg8/rgba8/r16/rg16/rgba16 sint) × all 3 stages, native ANV only (the same cases
pass on lavapipe). Something in the multisampled-sint path errors during
creation/encoding on ANV (Haswell has limited MSAA integer support — suspect a missing
`vkGetPhysicalDeviceFormatProperties` capability check, or an unconditional usage flag the
format doesn't support there), and the error surfaces as a poisoned command buffer rather
than a clean skip/error. Same 6-format × 3-stage pattern also contributes 6 fails to
`textureDimensions`. Low priority; diagnose with validation layers on the Intel host after
finding 1 lands.

Repro (native ANV):

```
webgpu:shader,execution,expression,call,builtin,textureLoad:multisampled:stage="c";texture_type="texture_multisampled_2d";format="r8sint"
```

---

## Re-verification (2026-07-04 evening, lavapipe + VK_LAYER_KHRONOS_validation)

The landed fixes were re-measured with the repro queries against the rebuilt
`target-vulkan/release` library:

- **Finding 3a — verified fixed.** With the harness-side `pumpUntil` eager
  `done()` re-check built into webgpu-native-cts, both
  `error_scope:current_scope` variants pass in seconds
  (`--isolate --workers 1 --case-timeout-ms 20000`: pass=2).
- **Finding 1 — sampled-read barrier verified effective.**
  `read_swizzle:format="r8uint";func="textureGather"` no longer emits
  `UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout` (912 → 0) or
  `VUID-RuntimeSpirv-NonWritable-06340` (the `fragmentStoresAndAtomics`
  enable). **Residual:** 684× `VUID-VkDescriptorImageInfo-imageLayout-00344` +
  684× `VUID-vkCmdDraw-None-08114` against a *different* image — the
  fragment-stage storage texture, whose descriptor declares GENERAL while the
  image stays in TRANSFER_DST. Root cause: `transition_storage_textures` is
  called only by the compute-pass encoder (`encode.rs`); `encode_render_pass`
  transitions sampled textures only. This is known-gap (a) below.
- **Finding 2 — `sampleRateShading` enable verified effective.**
  `VUID-VkShaderModuleCreateInfo-pCode-08740` is gone and the repro case now
  passes on lavapipe (pass=81). **Residual:** the object-creation VUID cluster
  (08931/01758/01209/09594/02662/03320/02251/02633 + 162 InvalidImageLayout)
  was NOT a cascade of 08740. Full message texts pin it to one root bug:
  `map_texture_usage` (`yawgpu-hal/src/vulkan/format.rs`) maps
  `render_attachment` to `COLOR_ATTACHMENT | INPUT_ATTACHMENT` regardless of
  format, so depth/stencil textures (`D32_SFLOAT_S8_UINT` in the repro) are
  created with color-attachment usage; every downstream view/barrier/
  framebuffer check then fails. `02251` (mip levels) to be re-measured after
  the usage fix — likely, but not confirmed, part of the same cascade.

### Rounds 2–3 outcome (same day)

Round 2 (render-pass storage-texture GENERAL transition + format-aware
`map_texture_usage`) removed the finding-1 residual entirely and collapsed the
finding-2 cluster from 8 VUID types to 3 — `02251` (mip levels), `08931`,
`01758`, `02662`, `01209`, `02633` were all cascades of the usage bug, as
suspected. The 3 residuals (`09594`, `03320`, `InvalidImageLayout` on the
stencil aspect) were all combined depth+stencil aspect bugs: the default image
view was created with a hardcoded COLOR aspect, and whole-image layout barriers
derived their aspect from the aspect-specific *bound* format (missing the other
aspect of a combined image, VUID-03320). Round 3 fixed both by storing the
image's own full aspect mask on `VulkanTextureInner` and using it for the
default view and all whole-image transitions.

**Final re-measure after round 3: all three verification queries — the
finding-1 repro (`r8uint`/`textureGather`), the float control
(`r8unorm`/`textureSample`), and the finding-2 repro (`sample_mask`) — emit
zero VUID lines and pass on lavapipe + validation layer.** This meets the bar
set below; what remains is the native-ANV cluster re-runs.

## Native-ANV cluster re-runs (2026-07-04 late evening, after all fix rounds)

Conditions: Intel Iris 5100 (Haswell, hasvk ICD), no validation layers, `--workers 4`
(`--workers 1` for the small clusters). Raw fail counts with no expectations file —
method differs from the sweep aggregation, so compare signatures, not absolute counts.

| Cluster | Sweep | Re-run | Verdict |
|---|---|---|---|
| `render_pipeline,sample_mask` | 1,398 | **90** | `fragment_output_mask` subtree (the finding-2 target) fully fixed. All 90 residuals are `alpha_to_coverage_mask` "alpha <= 0 result did not match zero coverage" — the separate signature already noted in the original finding. Driver-suspect; diagnose separately. |
| `texture_view,texture_component_swizzle` | 1,335 | 1,335 | Same count, different problem: every fail involves an alpha component in the swizzle on an integer format, and the missing-alpha default comes back as the f32 1.0 bit pattern `0x3F800000` instead of integer 1. Barriers are now validation-clean, so this is a Haswell integer-sampling default-alpha quirk. Candidate yawgpu-side workaround: map swizzle components absent from the format to explicit `VK_COMPONENT_SWIZZLE_ZERO`/`ONE` in `VkComponentMapping` instead of relying on the hardware default. |
| `builtin,textureGather` | 126 | 621 | All 621 are `format="rg32float"`: near-miss numeric mismatches (e.g. diff 0.05 vs tolerance 0.035), not garbage — looks like Haswell texel-selection/precision behaviour on 64-bit texel gathers. Needs a Dawn-oracle comparison on this host before treating as a yawgpu bug. |
| `builtin,textureLoad` | 286 | 2,001 | 264 `multisampled` (finding-4 territory, now visibly including float formats whose MSAA alpha default reads 0) + 1,737 `storage_textures_*` where **every single fail is `stage="v"`** (vertex-stage read-only storage textures read 0). **Pre-existing, not a regression:** the pre-round-2 build (dfcf93a HAL) fails the identical vertex-stage set, and lavapipe passes all of it. Driver-suspect (Haswell vertex-stage storage-image reads). Investigating it did expose a real spec gap — `stage_mask_for_layout(GENERAL)` lacked `VERTEX_SHADER` — fixed in round 4, which does not change the ANV outcome. |
| `memory_sync,texture,readonly_depth_stencil` | 4 | **1** | Remaining fail is `depthReadOnly=false;stencilReadOnly=true` (write depth while sampling read-only stencil) — consistent with known-gap 3 (whole-image layout tracking cannot express split per-aspect layouts; would need `separateDepthStencilLayouts`). |

Still to do on the Intel host: the two zero-dim-dispatch quarantined files
(`api,validation,encoding,cmds,compute_pass`,
`api,validation,encoding,programmable,pipeline_bind_group_compat`) — the round-1
early-out should make them safe, but the failure mode if not is a whole-machine
freeze, so run them deliberately; finding-4 diagnosis (validation layers on the
multisampled-sint cases); and Dawn-oracle comparisons for the remaining
driver-suspect signatures (rg32* gather, alpha-to-coverage — the default-alpha
swizzle signature is resolved, see below).

## Swizzle cluster re-run (2026-07-05, native ANV, after ad231fb)

`texture_view,texture_component_swizzle` re-run against the rebuilt library
(HEAD f672322, `--workers 4`, no expectations file; artifacts in
`webgpu-native-cts/build-yawgpu-release/run-linux-vulkan/rerun-0705-swizzle/`):
**1,335 → 543 fails.** The explicit ZERO/ONE component substitution (ad231fb)
eliminates the default-alpha class entirely — no residual has the
`0x3F800000` missing-alpha signature, and integer-format `textureGather`
swizzle cases (e.g. `r8sint`/`textureGather`) now pass. No Dawn-oracle
comparison needed for this signature anymore.

The 543 residuals decompose into two already-known signatures, now with wider
scope than previously recorded:

1. **342 = the six sint formats (r8/rg8/rgba8/r16/rg16/rgba16 sint) ×
   `func="textureLoad"`, all `queue submit cannot use an error command
   buffer`** — this is finding 4 exactly, not a widening of it: the
   `read_swizzle` test expands the `input` subcase parameter to both
   `texture_2d<i32>` and `texture_multisampled_2d<i32>` for
   possibly-multisampled formats (texture_component_swizzle.spec.cpp,
   `expand("input", ...)`), and the failing half of the subcases (57 of 114
   per format) is precisely the `texture_multisampled_2d` input; the
   single-sampled half passes, as do the same formats' `textureGather` cases
   (gather never takes a multisampled input). Attribute these 342 to the
   finding-4 diagnosis — sint + multisampled on ANV/hasvk.
2. **201 = rg32uint/rg32sint/rg32float (67 each) × `func="textureGather"`,
   wrong-texel value mismatches** (e.g. expected 551.814, got 431.099 — a
   different texel, not garbage) — extends the "rg32float gather" driver-suspect
   signature from the `builtin,textureGather` cluster to **all three rg32
   formats** (8-byte texels), reinforcing the Haswell
   64-bit-texel-gather-selection theory. Still needs the Dawn-oracle
   comparison on this host before treating as a yawgpu bug.

## Finding-4 root cause + driver-suspect triage (2026-07-05, native ANV, validation layer)

- **Finding 4 root cause confirmed: missing sample-count capability check.**
  hasvk reports `sampledImageIntegerSampleCounts = SAMPLE_COUNT_1_BIT` only
  (Haswell hardware cannot sample multisampled integer images; lavapipe
  supports 1+4, explaining the ANV-only failure). Running the finding-4 repro
  under `VK_LAYER_KHRONOS_validation` on native ANV emits **24×
  `VUID-VkImageCreateInfo-samples-02258`** — yawgpu calls `vkCreateImage`
  with `samples=4` on a format whose
  `vkGetPhysicalDeviceImageFormatProperties` does not include that sample
  count. That is invalid API usage (UB); the downstream failure surfaces as
  the "error command buffer" message. yawgpu-side fix: validate the requested
  sample count against image format properties at Vulkan texture creation and
  raise a clean `HalError`/device error instead (the cases still cannot pass
  on this hardware — WebGPU mandates integer MSAA — so the CTS marks them as
  host-specific expected failures; see webgpu-native-cts F-147).
- **The other three signatures are validation-clean on native ANV** (zero
  VUID lines, zero Validation Error lines, failures unchanged; logs in
  `rerun-0705-swizzle/diag-*.log`): rg32* textureGather wrong-texel
  (67 fails), `alpha_to_coverage_mask` (90 fails), vertex-stage read-only
  storage-texture loads returning 0 (NonWritable decoration correctly
  present — no 06341; `vertexPipelineStoresAndAtomics=false` is irrelevant
  to reads). All three are therefore driver/hardware-suspect
  (webgpu-native-cts F-148/F-149/F-150); no yawgpu action.

## Task: vulkan — sample-count capability check at texture creation (finding 4 / CTS F-147)

Goal: stop creating Vulkan images with unsupported sample counts (UB); surface
a clean device error instead.

Inputs to read:
- this file: "Finding 4" + "Finding-4 root cause" sections
- yawgpu-hal/src/vulkan/texture.rs (creation path), yawgpu-hal/src/vulkan/mod.rs
  (adapter / physical-device handle plumbing)

Produce:
- At Vulkan texture creation, when `sample_count > 1`, query
  `vkGetPhysicalDeviceImageFormatProperties` for the exact (format, type,
  tiling, usage, flags) about to be passed to `vkCreateImage`; if the
  requested sample count is not in the returned `sampleCounts` (or the query
  returns FORMAT_NOT_SUPPORTED), return a `HalError` (message naming the
  format + sample count) instead of calling `vkCreateImage`. No panics.
- Inline unit test(s) for any new/changed public fn per CLAUDE.md principle 1
  (the caps comparison itself should be a testable helper).

Out of scope: emulating integer MSAA, other backends, core validation changes
(WebGPU-level sampleCount rules stay as they are), spec edits, commits.

Acceptance criteria:
- [ ] `cargo test` green on Noop (no GPU), `-j 1`, output redirected to a file
- [ ] `cargo build --release -p yawgpu --features vulkan -j 1` clean
- [ ] no new clippy warnings (`-D warnings` gate unchanged)
- [ ] on native ANV the F-147 repro emits **zero** `VUID-VkImageCreateInfo-samples-02258`
      under the validation layer and fails with a clean device error naming the
      format/sample count (verified by Claude post-handoff)

Report back: files changed, how verified.

**Outcome (2026-07-05): implemented and verified.** `check_sample_count_support`
helper + capability query in `yawgpu-hal/src/vulkan/texture.rs`, new
`HalError::TextureCreationFailed` variant; 5 new inline unit tests, hal vulkan
test suite green, clippy clean, workspace test failure set byte-identical to
baseline (all pre-existing Tint-stub environment failures). Native-ANV repro
under the validation layer: **`VUID-VkImageCreateInfo-samples-02258` 24 → 0,
zero VUID lines total**. The cases still fail (the hardware cannot do integer
MSAA — expected, host-limitation xfail on the CTS side, F-147). Note: the
CTS-visible failure message is still the downstream `queue submit cannot use
an error command buffer`; the creation-time Internal device error (which
carries the format/sample-count message) is dispatched correctly per
`dispatch_hal_allocation_error` but is not the error the CTS case reports —
harness-side reporting nuance, no further yawgpu action.

## Zero-dim dispatch supervised re-try (2026-07-05): direct fixed, INDIRECT still wedges

The user re-tried the quarantined `compute_pass` file on the Haswell host with the
round-1 early-out in place — **immediate whole-machine freeze again** (rebooted).
Analysis: `dispatch_sizes` combines `dispatchType ∈ {direct, indirect}`; the round-1
early-out covers only direct dispatches (by design — indirect dims live in a GPU
buffer), so the first `vkCmdDispatchIndirect` whose args contain a zero dimension is
the trigger. **The ANV-Haswell zero-dim wedge covers indirect dispatches too.** Both
CTS files remain quarantined on that host.

Possible durable fix (NOT yet a task — needs a scope decision): hasvk exposes
`VK_EXT_conditional_rendering` (rev 2), which predicates `vkCmdDispatchIndirect`.
Sketch: on affected hardware (quirk-gated), before an indirect dispatch, run a
one-workgroup predicate shader reading the 3 indirect u32s and writing
`pred = (x && y && z)` to a scratch buffer, barrier, then wrap the real dispatch in
`vkCmdBeginConditionalRenderingEXT`/`End`. Zero-dim dispatches get culled before the
broken hardware path. Costs: predicate pipeline + scratch buffer management +
extension enable + quirk detection, and verifying it actually dodges the wedge is
itself a freeze-risk supervised run on the only affected hardware we have. The
alternative is accepting permanent quarantine of indirect-zero-dim CTS files on
Haswell (2 files today).

## Known related gaps (noted during the 2026-07-04 implementation review)

Observations from the fix-round implementation review, recorded so later CTS
failures can be matched against them:

1. **Render-pass storage textures are never transitioned to GENERAL** —
   `transition_storage_textures` is compute-only while render-pass storage
   descriptors declare GENERAL. *Fixed in round 2 (see above).*
2. **Tiled subpass path** (`tiled` feature) lacks the same sampled-texture
   transition that `encode_render_pass` now performs; sampled reads inside
   multi-subpass passes can still hit stale layouts.
3. **Layout tracking is per-texture, not per-subresource** — a single
   `AtomicU8` per `VulkanTextureInner` means per-mip/per-layer layout
   divergence (e.g. copy to mip 0 while sampling mip 1) is not representable;
   transitions barrier the whole image. (Round 3 made whole-image barriers
   aspect-complete, which keeps the single tracked state self-consistent, but
   the granularity limit itself remains.)
4. **External-texture planes get no layout transition** (Vulkan external
   textures are currently rejected before reaching the driver, so this is
   latent until a real Vulkan re-home; see the block-36 posture).
5. **Copy/clear transition sites still derive barrier aspect from the copy's
   (possibly aspect-narrowed) format** — `encode_texture_clear` and the
   buffer↔texture / texture↔texture copy paths use
   `buffer_texture_copy_aspect_flags(copy.format, copy.aspect)` for their
   whole-image transitions. A `DepthOnly`/`StencilOnly` copy on a *combined*
   depth-stencil image emits a single-aspect whole-image barrier — the same
   VUID-03320 class round 3 fixed for the bind-group paths. The
   `VkBufferImageCopy` subresource itself is correct; only the transition
   aspect would need the image-own `aspect_flags`. Next candidate if CTS
   depth/stencil copy tests show validation errors.

## Suggested order

1. Finding 1 barrier fix (upload→sample transition) — biggest win, then re-run the
   texture-builtin + readonly_depth_stencil clusters to see what remains.
2. Finding 2 MSAA object creation + feature enables. Root cause identified 2026-07-04:
   `sampleRateShading` device-feature enable is gated behind `#[cfg(feature = "tiled")]`
   in `yawgpu-hal/src/vulkan/mod.rs`, so the default Vulkan build never enables it while
   Tint emits `OpCapability SampleRateShading` for `@builtin(sample_mask)` output
   (→ VUID-08740). The other VUIDs in the histogram are expected to be cascade errors
   from the invalid pipeline — re-measure after the feature fix.
3. Finding 3b zero-dim dispatch early-out (one-liner, unblocks 2 quarantined files).
   (3a is harness-side — fix in webgpu-native-cts, see above.)

Verification for 1–2: the repro queries above under lavapipe +
`VK_LAYER_KHRONOS_validation` must emit **zero** VUID lines (matches the bar set by
[[vulkan-buffer-texture-usage-vuids]]), then re-run the failing clusters on native ANV.

## Addendum (2026-07-04, after the CTS-side tolerance fix)

The CTS harness's missing unorm-decode tolerance was fixed CTS-side (upstream 3-encoded-ULP
rule). After it, `textureLoad:*` on lavapipe still shows two yawgpu-suspect residuals:

- **13 deterministic fails, all `texture_depth_multisampled_2d` loads** — every sample reads
  ~0.5 (looks like a clear/default value) where the test wrote e.g. 0.125. Likely the MSAA
  depth write path (render-pass depth output → sampled read), adjacent to finding 2.
- **~30-40 additional depth-format fails appear only at `--workers 4` (in-process), flaky
  counts, single cases pass 5/5 in isolation** — a concurrency-dependent race on depth
  write→sample visibility, consistent with finding 1's missing-barrier theme (on lavapipe a
  missing barrier shows as a data race rather than a layout fault).
