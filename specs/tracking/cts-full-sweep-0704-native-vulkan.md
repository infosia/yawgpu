# CTS full-sweep 2026-07-04 (first Tint-enabled native run) — three yawgpu findings

Status: **OPEN — fixes for findings 1, 2, 3b landed in-tree 2026-07-04** (all Noop gates
green; pending lavapipe + validation-layer re-verification and the native-ANV cluster
re-runs). Finding 3a needs no yawgpu change (harness-side). Finding 4 deferred until
after the finding-1 re-measure. Surfaced by the first full CTS sweep with a working Tint frontend on
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
