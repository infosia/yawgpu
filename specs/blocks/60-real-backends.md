# Block 60 — Real backends (Phase 7)

Phase 7 brings up **real GPU backends** behind the existing
enum-dispatch HAL (`HalInstance/Adapter/Device/Queue/...` +
`HalError`), filling the `cfg(feature = "metal")` / `"vulkan")`
variants that currently return `BackendUnavailable`. Unlike Phases
0–6 this is **not** validation-rule porting — it is execution
bring-up verified by Dawn `end2end` Basic/Compute/Copy ports.

## Scope decisions (authoritative)

- **Metal first → Vulkan** (deliberate reorder of the SPEC roadmap's
  "Vulkan→Metal"). Rationale: the development platform is macOS, where
  Metal is native (no MoltenVK / Vulkan SDK / ICD setup) so real-GPU
  verification is possible on this machine immediately. Vulkan follows,
  reusing the same HAL contract. Recorded as a roadmap divergence in
  `tracking/phase-7.md` and annotated in `SPEC.md`.
- **Gating: compile-gated + `#[ignore]` / runtime adapter-probe
  skip.** Real backends live behind cargo features (`metal`,
  `vulkan`); they are never in `default` (= `noop`). end2end Rust
  ports are `#[ignore]`d (or self-skip when no real adapter is
  present) so `cargo test --workspace` (codex/CI) stays **Noop-only,
  build-only for real backends** — CLAUDE.md core principle 2 is
  preserved. Real-GPU runs are performed **manually by the user**
  (`cargo test --features metal -- --ignored`), reported back, and
  logged in `tracking/phase-7.md`.
- **Permanent gate unchanged**: `cargo test --workspace` +
  `cargo clippy --workspace --all-targets -- -D warnings` green on
  Noop. Additionally each slice must **build** with its backend
  feature on (`cargo build -p yawgpu --features metal`, clippy too).
- No-panic principle still holds in `yawgpu-hal`: backend FFI/driver
  errors map to `HalError`, surfaced as device errors — never panic
  in library code (Objective-C/`ash` boundaries may `expect` only
  where a null/!success is a true programming error, mirroring the C
  FFI-boundary exception).
- Out of scope: **D3D** (permanent); Dawn `wire/`; multi-adapter
  selection beyond what Basic/Compute/Copy need; swapchain/surface
  (Phase 8); robustness/zero-init/advanced end2end suites (revisit).
  *(GLES was previously listed here; it is now Tier 2 / experimental
  brought up in Phase 15 — see `67-gles-backend.md`.)*

## HAL contract the real backends must satisfy

The `yawgpu-core` ↔ `yawgpu-hal` seam is already exercised by Noop.
Real backends implement the same enum arms; **no `dyn Trait`** — add
`cfg`-gated arms to the existing `HalInstance/Adapter/Device/Queue`
+ resource/command/pipeline enums. The surface a backend must provide
(derive the exact signatures from `yawgpu-hal/src/noop` + how
`yawgpu-core` calls it):

- Instance: backend create + `enumerate_adapters` (real physical
  device); Adapter: `create_device` → real device + queue, report
  limits/features the core layer already validates against.
- Resources: Buffer (alloc, map/unmap or staging, destroy), Texture
  + TextureView, Sampler — backed by `MTL*` / `Vk*`.
- Commands: command encoder → buffer/texture copies, render pass
  (load/store, draw), compute pass (dispatch), submit + work-done.
- Pipelines: WGSL → backend shader (naga **MSL** backend for Metal,
  **SPIR-V** for Vulkan), bind-group/layout binding, render + compute
  pipeline objects.

Validation stays in `yawgpu-core` (Phases 0–6); the backend only
**executes** already-validated work. A backend op failing at the
driver level → `HalError` → device error (no panic).

### Minimum Vulkan version

The Vulkan HAL targets **Vulkan 1.1** as its minimum. This is declared via
`pApplicationInfo.apiVersion = VK_API_VERSION_1_1` at `vkCreateInstance`,
and `VulkanAdapter::new` rejects physical devices whose
`VkPhysicalDeviceProperties.apiVersion` is below 1.1 (returning `None` so
the adapter is silently dropped from `enumerate_adapters`).

Rationale: naga's SPIR-V output declares `SPV_KHR_storage_buffer_storage_class`,
which is core in Vulkan 1.1 (and would otherwise require enabling
`VK_KHR_storage_buffer_storage_class` as a device extension). 1.1 also
unblocks subgroup operations, 16-bit storage, and variable pointers — all
features naga may emit lazily. The 1.1 baseline is universally available
across the drivers yawgpu targets (MoltenVK ≥ 1.1.0, native desktop Vulkan,
Android mobile Vulkan since 2018). Follow-up tracking lives in
`specs/tracking/vulkan-buffer-texture-usage-vuids.md` § F3.

## Instance backend selection (`YaWGPUInstanceBackendSelect`)

The `YaWGPUInstanceBackendSelect` chain entry pins the HAL backend chosen
by `wgpuCreateInstance`. The rules below govern its behaviour and apply
uniformly across Tier 1 (Metal, Vulkan) and Tier 2 (GLES) backends.

- **IB1** No chain entry present ⇒ `wgpuCreateInstance` returns a Noop
  instance (unchanged). This is the lenient default that keeps "I just
  want a WebGPU instance for unit tests" working with no per-call
  configuration.
- **IB2** Chain entry with `backend == YAWGPU_INSTANCE_BACKEND_NOOP` ⇒
  returns a Noop instance (unchanged). This preserves an opt-in "I want
  Noop explicitly" route.
- **IB3** Chain entry with `backend == YAWGPU_INSTANCE_BACKEND_{METAL,
  VULKAN, GLES}` is **strict**: `wgpuCreateInstance` returns `NULL`
  if any of the following hold:
  - the matching cargo feature (`metal` / `vulkan` / `gles`) was not
    compiled into this `yawgpu` build;
  - the backend's `HalInstance::new` returns `Err`;
  - the backend's `enumerate_adapters()` returns empty.

  A best-effort diagnostic line is written to `stderr` via `eprintln!`
  (matching the existing `YaWGPUGlesContextBackend` diagnostic style)
  identifying which of the three causes fired. The caller's only in-band
  signal is the `NULL` return — there is no error callback on
  `wgpuCreateInstance` per webgpu.h.

  **Rationale:** the silent-Noop fallback that previously fired here
  caused test code that explicitly requested Metal/Vulkan/GLES to
  produce false-positive passes against a Noop instance. Tests can now
  fail-fast on backend-availability mismatches.

- **IB4** Chain entry with an unrecognised `backend` value (anything
  outside the four `YAWGPU_INSTANCE_BACKEND_*` constants) is treated as
  if no chain were present (returns Noop). This is the only remaining
  lenient case, kept so that an older `yawgpu` build reading a
  descriptor produced by a newer header (which may define additional
  backend constants) does not immediately fail. Callers wanting strict
  detection can verify the chosen backend via
  `wgpuAdapterGetInfo().backendType` after `wgpuInstanceRequestAdapter`.

The same `NULL`-on-strict-failure contract applies regardless of how the
yawgpu instance descriptor is constructed (with or without the
`YaWGPUGlesContextBackend` companion chain).

### Acceptance tests (Noop gate)

Direct unit tests on `wgpuCreateInstance` covering each rule, in
`yawgpu/src/ffi/instance.rs` (or `yawgpu/src/ffi/mod.rs` where the
existing `instance_backend_selection` tests live):

- **IB1**: descriptor with `nextInChain == NULL` ⇒ non-NULL handle.
- **IB2**: chain `backend = NOOP` ⇒ non-NULL handle.
- **IB3-no-feature**: chain `backend = METAL/VULKAN/GLES` on a build
  whose matching cargo feature is **not** enabled ⇒ NULL handle. (The
  Noop gate runs without any of these features, so the chain entries
  for Metal/Vulkan/GLES all hit this path.)
- **IB4**: chain `backend = 0x42` (or any unrecognised value) ⇒
  non-NULL handle (Noop fallback). 

The existing chain-routing tests at `yawgpu/src/ffi/mod.rs` must be
audited and any that depended on the silent-Noop fallback for
Metal/Vulkan/GLES updated to either (a) drop the chain entry, (b)
switch to `backend = NOOP`, or (c) be moved behind the matching
backend feature flag.

## Slices → end2end port targets

Dawn `dawn/src/dawn/tests/end2end/`. Port the **minimal**
Basic/Compute/Copy subset to `yawgpu/tests/e2e_*` (gated). Each slice:
Red (ported end2end test, `#[ignore]`, fails / unimplemented) → Green
(backend impl) → user runs `--ignored` on real GPU, reports, logged.

- **P7.0** Bring-up scaffolding + gating harness (de-risk; no GPU
  code path executed in CI). `metal` dep wiring, gpu-gated test
  helper in `yawgpu-test` (adapter-probe / `#[ignore]`), backend
  selection in `wgpuCreateInstance`. Acceptance: builds with
  `--features metal`; Noop gate unchanged; harness skips cleanly with
  no adapter.
- **P7.1** Metal Instance/Adapter/Device/Queue. Port: `BasicTests`
  (device/queue creation, empty submit). 
- **P7.2** Metal Buffer + Queue writeBuffer/submit + B2B copy. Port:
  `BufferTests` / `CopyTests` (buffer subset).
- **P7.3** Metal Texture/Sampler + B2T/T2B/T2T. Port: `CopyTests`
  (texture subset).
- **P7.4** Metal Shader (naga→MSL) + compute pipeline + dispatch.
  Port: `ComputeDispatchTests` (basic).
- **P7.5** Metal render pipeline + render pass draw. Port:
  `BasicTests` render / a minimal draw end2end.
- **P7.6** Vulkan bring-up mirroring P7.1–P7.5 over the same HAL
  contract (`ash` + MoltenVK on macOS), reusing the ported end2end
  tests parametrized by backend feature.
- **Phase 7 Review** (mandatory Clean Review Then Fix) → COMPLETE.

## Open questions (resolve per slice, record divergences)

- Metal crate choice (`metal` vs `objc2-metal`) — decide in P7.0,
  record.
- Buffer mapping model on Metal (shared storage vs staging blit) —
  decide in P7.2.
- naga MSL/SPIR-V backend options (bindings model, entry-point
  remap) vs the bind-group layout core already derives — P7.4.
- end2end readback (map-after-submit) needed to assert results;
  scope the minimal readback path in P7.2.

## CTS finding F-069 — Metal threadgroup memory (2026-06-11)

The MSL backend emits `var<workgroup>` globals as `[[threadgroup(N)]]`
entry-point arguments; Metal requires the compute encoder to size each slot
via `setThreadgroupMemoryLength:atIndex:` before dispatch (unallocated slots
read zeros). Rule: MSL generation reports per-entry-point workgroup-variable
sizes, rounded up to a multiple of 16 (Metal requirement); the Metal compute
pipeline stores them and the encoder sets each slot length right after
`setComputePipelineState`. Vulkan/GLES ignore the field (workgroup storage is
declared in the module). **Tint migration (2026-06-27):** the per-index sizes
now come from Tint's `msl::writer::Output::workgroup_allocations` (surfaced
through the `yawgpu-tint` shim). The first Tint cut stubbed this to empty and
**regressed** every `atomics:*_workgroup` case to read zeros — re-fixed and
re-verified (atomics 1445/0 on Metal); see
`specs/tracking/tint-migration-plan.md` → "Post-migration CTS regression audit"
(which also covers the sample-mask and frag-depth-clamp regressions from the
same class).

## CTS finding F-068 — vertex OOB robustness (2026-06-11)

WebGPU requires OOB vertex-attribute fetches (including via indirect draw
params) to be clamped/zeroed. yawgpu does NOT implement wgpu's
indirect-validation compute prepass; instead: **Vulkan** enables the
`robustBufferAccess` device feature when supported (hardware bounds vertex
fetches; MoltenVK cannot honor it — documented translation artifact, native
Vulkan authoritative). **Metal** bounds-guards attribute fetches in the vertex
shader against buffer sizes. GLES (Tier 2): unhandled, catalogue in block 67 if
bring-up reaches this. **Tint migration (2026-06-27):** Metal no longer uses
naga's `vertex_pulling_transform` — Tint emits `[[stage_in]]` vertex MSL driven
by an `MTLVertexDescriptor` (the HAL detects `[[stage_in]]` and builds the
descriptor), and OOB robustness is handled by Tint's own robustness transform
(`disable_robustness=false`). Re-verified clean on Metal:
`rendering,robust_access_index` passes under Tint (no regression — unlike the
three transform behaviors in F-069 above, this path migrated cleanly).

## CTS finding F-112 — workgroup-atomic coherence vs SPIR-V buffer bounds policy (2026-06-16)

`shader,execution,memory_model,coherence:corr` (the `atomic_workgroup;
intra_workgroup` non-RMW subcase) recorded the WebGPU-disallowed weak outcome
`r0==1 && r1==0` (single-location read-read coherence violation) on native
Vulkan (NVIDIA RTX 5060 Ti). Root-caused on hardware (full diagnostic ledger:
webgpu-native-cts `docs/FINDINGS.md` F-112):

- **Not a naga/SPIR-V atomic-semantics defect.** yawgpu and wgpu-native emit
  byte-identical workgroup-atomic SPIR-V (`OpMemoryModel GLSL450`,
  `scope=Workgroup`, `semantics=0`, no `Coherent` decoration; both post naga
  PR #8391). Verified by reassembling wgpu-native's captured SPIR-V.
- **Cause: the SPIR-V `buffer` bounds-check policy.** yawgpu compiled with
  `BoundsCheckPolicy::Restrict` for `buffer`, emitting a software bounds clamp
  (`OpArrayLength`+`OpISub`+`OpExtInst UMin`) on every runtime-sized
  storage-buffer access. On this NVIDIA driver that clamp breaks the
  workgroup-atomic read-read coherence guarantee in the stress shader.
  Verified: flipping **only** the `buffer` policy to `Unchecked` makes
  `coherence:corr` pass 6/6; SPIR-V version (1.0/1.3/1.6), Vulkan API version
  (1.1/1.3), and workgroup zero-init mode were all ruled out as non-causal.

**Decision (mirrors wgpu).** Detect `VK_EXT_robustness2` / `robustBufferAccess2`
on the adapter; when present, enable the extension + feature and compile the
SPIR-V `buffer` bounds-check policy as `Unchecked` (hardware robustness bounds
OOB buffer access). When absent, keep `Restrict` as the safe fallback. The
`index` and `image_load` policies stay `Restrict` (the `index` clamp on the
workgroup array is coherence-neutral and confirmed not the cause). Minimum
Vulkan version is unchanged (1.1; `VK_EXT_robustness2` is available as an
extension on 1.1+). yawgpu already enables Vulkan-1.0 `robustBufferAccess`, so
OOB writes remain bounded regardless. No naga change is required.

## CTS finding F-127 — Tint-era robustness via the Vulkan Memory Model (2026-06-28)

**Supersedes the F-112 `VK_EXT_robustness2` decision above** (now retired). The
naga→Tint frontend migration invalidated F-112's mechanism: naga had a
**per-address-space** `BoundsCheckPolicy`, so F-112 could set `buffer = Unchecked`
in isolation. Tint's SPIR-V writer
(`third_party/dawn/src/tint/lang/spirv/writer/common/options.h`) exposes only a
**single whole-shader** `disable_robustness` flag (no per-address-space toggle).
yawgpu had been driving `disable_robustness = robust_buffer_access2()`, which on a
robustBufferAccess2 device (NVIDIA) disabled robustness for the **entire** shader —
so uniform (sub-`robustUniformBufferAccessSizeAlignment`), workgroup, function, and
private out-of-bounds accesses, and OOB writes, all lost clamping
(`shader,execution,robust_access:linear_memory` → 216 fail; webgpu-native-cts
`docs/FINDINGS.md` F-127).

The original F-112 workaround (disable buffer robustness to dodge the NVIDIA
workgroup-atomic coherence violation that the software `OpArrayLength` clamp
triggers) is fundamentally incompatible with Tint's all-or-nothing robustness:
clamping cannot be kept for uniform/workgroup/private while removed for storage.

**Decision (mirrors Dawn).** Enable the **Vulkan Memory Model** and keep SPIR-V
robustness fully **ON**. Dawn does exactly this — `PhysicalDeviceVk.cpp` defaults
`Toggle::UseVulkanMemoryModel` on when the extension is available and passes
`tintOptions.extensions.use_vulkan_memory_model` to Tint (`ShaderModuleVk.cpp`),
with full robustness. yawgpu now:

- **yawgpu-hal** enables `VK_KHR_vulkan_memory_model` + `vulkanMemoryModel`
  (`vulkanMemoryModelDeviceScope` when reported), available from Vulkan-1.2 core or
  the extension on 1.1; exposes `VulkanDevice::vulkan_memory_model()`. The
  `VK_EXT_robustness2` / `robustBufferAccess2` enablement and the
  `robust_buffer_access2()` accessor are **removed** (the Vulkan-1.0
  `robustBufferAccess` vertex-robustness enablement, F-068, is unaffected).
- **yawgpu-tint** threads a `use_vulkan_memory_model` flag through
  `generate_spirv` (shim + binding) → `options.extensions.use_vulkan_memory_model`
  on the SPIR-V path.
- **yawgpu-core** `ReflectedModule::generate_spirv` always passes `robust = true`
  and forwards the device's `vulkan_memory_model()` bit through
  `select_{compute,render}_shader_source`. MSL/Metal is untouched.

**Verification (native NVIDIA RTX 5060 Ti, Vulkan 1.4).**
`robust_access:linear_memory:*` → **pass=1626 fail=0** (was 216 fail). The change
also *improves* coherence: `coherence:corr:atomic_workgroup;intra_workgroup` (the
original F-112 subcase) now **passes** under VMM + full robustness, where the old
robustness-off config failed it deterministically (~1600 weak behaviors/run).

**`coherence:corr:atomic_storage;intra_workgroup` is a driver limitation, not a
yawgpu defect.** It fails on this GPU under **every** configuration, including the
**Dawn oracle** (3/3 runs, ~800 disallowed weak behaviors each) and the pre-fix
yawgpu config — i.e. the RTX 5060 Ti's memory subsystem exhibits the WebGPU-
disallowed weak read-read behavior regardless of the implementation. Per the
project's oracle-cross-check rule (cf. F-128), a case the Dawn oracle fails
identically is not attributable to yawgpu; it is carried as an `xfail` in
webgpu-native-cts `expectations/yawgpu-vulkan.txt`, not chased in the library. The
F-112 finding's historical "`coherence:* 27/27`" no longer reproduces on the
current driver/CTS.

## CTS finding F-138 — `bgra8unorm` storage-texture view format mismatch (2026-06-28)

`textureStore` to a `bgra8unorm` write-only storage texture wrote wrong/zero bytes
on native Vulkan (`expected 51, got 0`); every other store format passed. Dawn (same
Tint) passes. Root cause is **not** in Tint or the VkFormat map: Tint's
`Bgra8UnormPolyfill` (`third_party/dawn/src/tint/lang/spirv/.../bgra8unorm_polyfill.cc`)
already rewrites a `bgra8unorm` storage texture to an **`rgba8unorm`** storage image
plus a `(2,1,0,3)` channel swizzle, so the emitted SPIR-V writes RGBA-ordered bytes
through an image **declared `rgba8unorm`**. yawgpu bound that storage texture through
a VkImageView created with the **BGRA** VkFormat, and the backing image lacked
`MUTABLE_FORMAT`, so Vulkan reinterpreted the shader's RGBA bytes as BGRA.

**Decision (mirrors Dawn `TextureVk.cpp` `mHandleForBGRA8UnormStorage`).** For a
`bgra8unorm` texture with `STORAGE_BINDING` usage, create the backing image with
`VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` (plus a `VkImageFormatListCreateInfo` listing
`B8G8R8A8_UNORM` + `R8G8B8A8_UNORM` when `VK_KHR_image_format_list` / Vulkan 1.2 is
available) and bind storage through a dedicated `R8G8B8A8_UNORM` view
(`yawgpu-hal/src/vulkan/texture.rs` caches one for the canonical whole-resource range
on `VulkanTextureInner`; `pipeline.rs` `create_storage_texture_image_view` reuses it
or builds a remapped on-the-fly view for non-canonical subresource ranges).
HAL-only — no core or shim change. Non-BGRA storage textures are unaffected.

Verification (native NVIDIA Vulkan): the `bgra8unorm` cases of
`shader,execution,expression,call,builtin,textureStore:*` → `fail=0`; plus an
in-repo `e2e_vulkan` regression that stores to a `bgra8unorm` storage texture and
reads back the expected byte order.

## CTS finding F-135 — Vulkan entry loaded per `VulkanInstance` → device-creation churn leak (2026-06-23)

webgpu-native-cts `docs/FINDINGS.md` F-135 (`specs/investigate-yawgpu-device-create-leak.md`)
localized a yawgpu-specific HAL leak: CTS families that own and churn their **own**
`instance+adapter+device` every case — `api,validation,capability_checks,limits,*`
(`LimitTest`, `limit_utils.h:330-432`) and `state,device_lost,destroy`
(`destroy.spec.cpp`) — fail in a single process after ~150 device creations with
`requestDevice failed: HAL device creation failed: vulkan` (onset at result ~#151,
then success/failure intermixed — a creation *ceiling*, not a hard cliff). The
tight repro `…capability_checks,limits,maxStorageBuffersPerShaderStage:*` (700
cases) is **fail=318** single-process (`--workers 1`) vs **fail=0** under
`--isolate` (fresh process per case). Whole-area `api,validation` (39,349) shows
the leak manifesting nondeterministically as either cascade `requestDevice`
failures or hard `0xC0000005` access violations in exactly these churning
families. The CTS harness `device-recycle` cannot help — it rebuilds only the
*cached* device, which these tests never touch (wrong layer).

**Root cause (yawgpu HAL).** `VulkanInstance::new` (`yawgpu-hal/src/vulkan/mod.rs`)
calls `ash::Entry::load()` on **every** `WGPUInstance` creation, and the loaded
entry is owned per-instance (`VulkanInstanceInner._entry`), so the last Arc drop
runs `vkDestroyInstance` **and** `FreeLibrary("vulkan-1.dll")`. Each `WGPUInstance`
lifecycle is therefore a full `LoadLibrary`/`FreeLibrary` round-trip of the Vulkan
loader + NVIDIA ICD. The Windows loader/ICD does not fully reclaim per-load state
(TLS slots, ICD process state) across repeated load/unload cycles; after ~150
cycles the process hits the ceiling and `vkCreateInstance`/`vkCreateDevice` starts
failing (or access-violates). The rest of the HAL teardown is correct — every
`impl Drop` (`VulkanDeviceInner`, `VulkanInstanceInner`, queue/buffer/texture/…)
destroys its handle, and there is no global registry or Arc cycle retaining
devices (`VulkanQueueInner → Arc<VulkanDeviceInner>`, no back-edge). The leak is
purely the per-instance library churn, not unreleased WebGPU objects (consistent
with F-126, which found no per-test resource leak on the *cached* device path).

**Decision (mirrors wgpu/Dawn).** Load the Vulkan library **once per process** and
share it across every `VulkanInstance`; never `FreeLibrary` it. Concretely: a
process-global `static VULKAN_ENTRY: OnceLock<ash::Entry>` in
`yawgpu-hal/src/vulkan/mod.rs`; `VulkanInstance::new` obtains the entry via
`get_or_try_init(|| unsafe { ash::Entry::load() })` (mapping load failure to
`HalError::BackendUnavailable { backend: "vulkan" }`, as today). `VulkanInstanceInner`
holds a shared handle to the cached entry instead of an owned one, so instance
`Drop` runs **only** `destroy_instance` — the loader DLL stays resident for the
process lifetime. This eliminates the load/unload churn so repeated
`WGPUInstance`/`WGPUDevice` creation no longer self-poisons; `--isolate` remains a
valid containment but is no longer required for these families. Minimum Vulkan
version, extension/feature logic, and all existing Drop semantics are unchanged.

Verification: a CTS-test-free standalone churn loop
(`createInstance → requestAdapter → requestDevice → release` ×700) must reproduce
`HAL device creation failed` within ~150 iterations **before** the fix and run
fail-free for 700 iterations **after** it, on the Windows native-Vulkan host. The
CTS-side carry (run device-churning families under `--isolate`; no `expectations/`
xfail) is recorded in webgpu-native-cts and is orthogonal to this library fix.
Implementation handoff: `specs/tracking/f135-vulkan-entry-leak-handoff.md`.

## CTS finding (pending F-number) — Vulkan cumulative queue-submission degradation (2026-07-03)

webgpu-native-cts `REPORT.md` (2026-07-03, yawgpu `490743e`, Windows/NVIDIA RTX
5060 Ti): during sustained in-process `--workers` sweeps of `api,validation`, the
Vulkan backend starts failing queue submissions intermittently mid-run (onset
~44% into a fresh worker process, then a continuous stochastic cascade —
6,202–19,323 `uncaptured error: HAL queue submission failed: vulkan` fails per
shard). The device is **not** lost (the same worker keeps passing 160k+ cases
interleaved), the same cases pass standalone, and a back-to-back Dawn/Vulkan
control run on the identical harness path was clean — yawgpu-side confirmed,
harness exonerated. The primary root cause (what makes the *first* submit fail)
is **unconfirmed** because the submit path discards every underlying `VkResult`.

### Contract 1 — submit-path error transparency

`HalError::QueueSubmissionFailed` carries a `message: String` naming the failing
call and the underlying backend error:

```rust
QueueSubmissionFailed { backend: &'static str, message: String }
// Display: "HAL queue submission failed: {backend}: {message}"
// e.g.     "HAL queue submission failed: vulkan: vkQueueSubmit failed: ERROR_OUT_OF_DEVICE_MEMORY"
```

Every construction site populates it: Vulkan submit path (`encode.rs`,
`queue.rs`, `surface.rs`) includes the `vk::Result` debug form; Metal / GLES /
Noop sites use a static description of the failing step. Rationale: the error
string is what the CTS harness prints as the uncaptured device error — the
`yawgpu` cdylib initializes no logger, so the message field (not `log::error!`)
is the only channel that reaches the sweep logs. This is a breaking change to a
public enum variant; all in-workspace matchers are updated in the same commit.

### Contract 2 — retire-ring error path must not leak or shrink

`RetireRing` (`yawgpu-hal/src/vulkan/surface.rs`) invariants on a fence-wait
failure (previously violated via `std::mem::forget` at surface.rs:127–131 and
151–154, which leaked the frame's fence/command pool/descriptor pools/
framebuffers/image views/render passes **and** left the ring slot `None`
forever):

- A failed `vkWaitForFences` must **not** leak the waited frame and must **not**
  reduce the ring's usable depth. The slot's frame is waited **by reference**
  and only `.take()`n after a successful wait.
- When the slot cannot be freed, the incoming frame is parked on an
  `overflow: Vec<InFlightFrame>` drained at the start of the next
  `retire`/`wait_all` — never forgotten.
- Mutex-poison paths on the submit path recover the guard via
  `PoisonError::into_inner` instead of `std::mem::forget`ing cleanup lists.

### Contract 3 — per-submit objects are destroyed on a failed submit

`record_and_submit_copies` creates per-submit objects (TRANSIENT command pool,
command buffer, fence, descriptor pools, framebuffers, image views, render
passes). On **any** error before the retire-ring registration succeeds, all
objects created so far are destroyed immediately instead of leaked. This is
safe: a failed `vkQueueSubmit` enqueues nothing (Vulkan spec), and earlier
failures never reached submit, so no GPU work references them. (Previously each
failed submit leaked the command pool + temps, turning the first failure into a
self-amplifying cascade — consistent with the observed continuous fail tail.)

### Root-cause stage (deferred until the VkResult is visible)

With Contracts 1–3 in place, the degrading load is re-run and the surfaced
`VkResult` pins the failure class (`OUT_OF_DEVICE_MEMORY` / `OUT_OF_POOL_MEMORY`
/ `OUT_OF_HOST_MEMORY` / `DEVICE_LOST`); the corresponding fix (allocation
strategy, pool/fence reuse, or device-loss signalling to core) is specified
then. Expected end state: full `api,validation` runs `fail=0` (modulo the 4
known non-defects) under plain `--workers` with no cumulative HAL degradation.
