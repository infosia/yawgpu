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

naga's MSL backend emits `var<workgroup>` globals as `[[threadgroup(N)]]`
entry-point arguments; Metal requires the compute encoder to size each slot
via `setThreadgroupMemoryLength:atIndex:` before dispatch (unallocated slots
read zeros). Rule: MSL generation (`generate_msl`) reports per-entry-point
workgroup-variable sizes in naga emission order, rounded up to a multiple of
16 (Metal requirement; mirrors wgpu-hal `load_shader`); the Metal compute
pipeline stores them and the encoder sets each slot length right after
`setComputePipelineState`. Vulkan/GLES ignore the field (workgroup storage is
declared in the module). CTS: `shader,execution,memory_layout` Metal residue
is naga-lineage only (F-070).

## CTS finding F-068 — vertex OOB robustness (2026-06-11)

WebGPU requires OOB vertex-attribute fetches (including via indirect draw
params) to be clamped/zeroed. yawgpu does NOT implement wgpu's
indirect-validation compute prepass; instead: **Vulkan** enables the
`robustBufferAccess` device feature when supported (hardware bounds vertex
fetches; MoltenVK cannot honor it — documented translation artifact, native
Vulkan authoritative). **Metal** enables naga's `vertex_pulling_transform`:
vertex shaders bounds-guard every attribute fetch against `_mslBufferSizes`;
the slot is forced whenever vertex buffers exist, and the HAL writes
storage-array sizes followed by per-vertex-buffer `size − offset` (mapping
order) before every draw. GLES (Tier 2): unhandled, catalogue in block 67 if
bring-up reaches this.

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
