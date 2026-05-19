# Phase 7 — Real backends

Status: **in progress** (P7.0 active). Rules/plan:
`../blocks/60-real-backends.md`. Roles/loop:
`../reference/workflow.md`.

**Roadmap divergence (approved):** SPEC roadmap lists Phase 7 as
"Vulkan→Metal"; we bring up **Metal first, then Vulkan** because the
dev platform is macOS (Metal native; no MoltenVK/Vulkan-SDK needed for
on-machine real-GPU verification). Vulkan (P7.6) reuses the identical
HAL contract. SPEC.md Phase-7 row annotated accordingly.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` green on **Noop**
(real-backend code is build-only in CI). Per slice **also**: `cargo
build -p yawgpu --features metal` (later `--features vulkan`) +
clippy with the feature. **Real-GPU end2end** (`cargo test --features
metal -- --ignored`) is run **by Claude directly** — the Bash tool
executes on this Apple Silicon and the sandbox permits Metal device access
(confirmed P7.1) — and logged here per slice (no manual user step for
Metal). Vulkan (P7.6) via MoltenVK is also machine-runnable
(`$VULKAN_SDK` = `$VULKAN_SDK`; Apple Silicon enumerates).
**Phase ends with the mandatory Phase Review**
(`tracking/phase-7-review.md`).

Methodology shift vs Phases 0–6: not validation-rule porting —
execution bring-up verified by gated Dawn `end2end` Basic/Compute/Copy
ports. Validation stays in `yawgpu-core`; backends only execute
already-validated work; driver failure → `HalError` → device error,
never panic.

## P7.0 — Bring-up scaffolding + gating harness  *(☑ DONE)*

Done: `metal` crate (0.33.0, recorded in `dependencies.md`) wired as
an **optional** `yawgpu-hal` dep behind `metal = ["dep:metal"]`
(`default = ["noop"]` unchanged); `yawgpu` gained a `metal` feature
forwarding to `yawgpu-hal/metal`. Inline `metal` HAL placeholder
moved to `yawgpu-hal/src/metal/mod.rs` mirroring the Noop contract
(`MetalInstance/Adapter/Device/Queue/Buffer/Texture/Sampler`): every
fallible entry (`*::new`, `MetalAdapter::create_device`) returns
`HalError::BackendUnavailable`, `enumerate_adapters()` is empty (so
the `HalInstance::Metal` arm is unreachable), infallible creators are
allocation-counting no-ops; `use metal as _;` proves link with **zero
Objective-C/MTL calls**. `yawgpu-test` gained `RealBackend` +
`real_backend_available` (→ false in P7.0) + `real_backend_skip_
reason`; one `#[ignore]` `yawgpu/tests/e2e_metal_smoke.rs` asserting
unavailability (proves the harness shape). `wgpuCreateInstance`
backend *selection* intentionally deferred to P7.1 (nothing real to
select yet; Noop remains the only reachable backend). Gate: Noop
`cargo test --workspace` 43 binaries green + `clippy --workspace
--all-targets -D warnings` clean (smoke ignored, not run); `cargo
build -p yawgpu --features metal` + `clippy -p yawgpu --features
metal --all-targets -D warnings` clean; smoke passes on `--features
metal -- --ignored`. Committed `phase-7: P7.0`.

## P7.1 — Metal Instance/Adapter/Device/Queue  *(☑ DONE — real-GPU-verified)*

Done (codex + Noop/feature gate): `metal` module real for objects —
`MetalInstance::new` ok; `enumerate_adapters` via `metal::Device::
all()` (name from `device.name()`); `MetalAdapter::create_device`
builds a `metal::CommandQueue`; `MetalDevice` retains device+queue;
`MetalQueue::submit_empty` = new command buffer → `commit()` →
`wait_until_completed()`; buffer/texture/sampler stay P7.0 counter-
only stubs (`// P7.2/P7.3`). No panics (`system_default`/`all` →
`Option`→`HalError`). `HalError` gained `DeviceCreationFailed`/
`QueueSubmissionFailed`; `HalAdapter::{name,backend}` + `HalBackend`;
`HalQueue::submit_empty` (Noop/Vulkan = `Ok(())` no-op — Noop
byte-for-byte unchanged; Metal real). `core::Queue::submit` returns
`Option<DeviceError>`; **only zero-CB submits** call
`hal.submit_empty()` (`HalError`→`DeviceError::internal`); validation
path unchanged. `DeviceError::{validation,internal}` ctors. FFI:
yawgpu vendor `WGPUYawgpuInstanceBackendSelect` chained struct
(SType `0x7000_0001`, backend Noop=0/Metal=1/Vulkan=2);
`wgpuCreateInstance` selects Metal only when the struct requests it
**and** `cfg(feature="metal")` **and** ≥1 adapter — else exact
`new_noop()` fallback; `WGPUInstanceImpl::from_core`;
`wgpuAdapterGetInfo`/`wgpuAdapterInfoFreeMembers` (for the name
assertion); `dispatch_optional_device_error`. `yawgpu-test` gained an
optional `metal` feature; `real_backend_available(Metal)` probes
`metal::Device::system_default()`. Tests: `e2e_metal_basic.rs` (3:
adapter name, device+queue+empty-submit, default-instance-is-Noop) +
`e2e_metal_smoke.rs` updated to match the probe — all `#[ignore]` +
`cfg(feature="metal")` self-skip. Gate: Noop `cargo test --workspace`
44 binaries green + `clippy --workspace --all-targets -D warnings`
clean; `cargo build -p yawgpu --features metal` + `clippy -p yawgpu
--features metal --all-targets -D warnings` clean; e2e tests ignored
(not run — no GPU in codex/CI). Committed `phase-7: P7.1`.
Real-GPU verified by Claude directly (the Bash tool runs on this
Apple Silicon; the seatbelt sandbox permits Metal device access — no
manual user step needed for Metal slices).

### P7.1 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_basic --test e2e_metal_smoke -- --ignored`:
  **e2e_metal_basic 3/3 pass** (adapter name, device+queue+empty
  submit, default-instance-Noop) + **e2e_metal_smoke 1/1 pass**.
  P7.1 hardware-confirmed.

## P7.2 — Metal Buffer + writeBuffer/submit + B2B  *(☑ DONE — real-GPU-verified)*

Done: `metal` module — real `metal::Buffer`
(`MTLResourceStorageModeShared`, `inner: Option` so alloc-fail errors
instead of panicking); bounds-checked `write`/`read`/`validate_range`
(`checked_add`, `contents().is_null()` guarded, no panic);
`MetalQueue::submit_buffer_copies` = blit encoder
`copy_from_buffer` per copy → `commit`+`wait` (range-validated,
non-Metal source/dest → `HalError`). HAL: `HalBuffer` is `Clone` +
`size/write/read` dispatch (Noop/Vulkan = no-op / zero-fill),
`HalBufferCopy`, `HalQueue::submit_buffer_copies`,
`HalError::BufferOperationFailed`. core: `BufferCopyCommand` recorded
on successful `copy_buffer_to_buffer`, carried in `CommandBuffer`
(empty on error/`finish`-fail); `Queue::submit` translates each CB's
copies via `Buffer::hal()` (skips buffers with no real HAL ⇒ Noop
stays a no-op) → `hal.submit_buffer_copies` (`HalError`→
`DeviceError::internal`); `Queue::write_buffer` → `Buffer::
write_from_queue` (validate then `HalBuffer::write`; Noop no-op);
read-map readback wired in `resolve_pending_map` (Read map ⇒
`hal.read` → fill `HostBuffer`, so the standard
`wgpuBufferMapAsync`+`GetMappedRange` path returns real Metal bytes;
Noop fills zeros — validation-only, no ported test depends on Noop
content, gate confirms). FFI: `wgpuQueueWriteBuffer` threads real
`data` (null-guarded) to `core.write_buffer`. Tests
`e2e_metal_buffer.rs` (3, `#[ignore]`/cfg-gated): write→B2B→map-read
round-trip, partial-range (non-zero offsets), Noop path no-error.

Gate: Noop `cargo test --workspace` 45 binaries green (buffer_map
9/9, buffer_mapped_range 9/9, buffer_creation 8/8 — unchanged) +
`clippy --workspace --all-targets -D warnings` clean; `cargo build/
clippy -p yawgpu --features metal` clean. Committed `phase-7: P7.2`.

### P7.2 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_buffer -- --ignored`: **3/3 pass**
  (`metal_write_copy_readback_round_trip`,
  `metal_partial_buffer_copy_round_trip`,
  `default_noop_write_copy_readback_path_has_no_device_error`). P7.1
  e2e re-run: e2e_metal_basic 3/3 + e2e_metal_smoke 1/1 (no
  regression). **Real CPU→MTLBuffer→blit-B2B→map-readback confirmed.**

## P7.3 — Metal Texture/Sampler + B2T/T2B/T2T  *(☑ DONE — real-GPU-verified)*

Done: P7.2 copy seam generalized to `HalQueue::submit_copies(&[HalCopy])`
with `HalCopy::{Buffer,BufferToTexture,TextureToBuffer,
TextureToTexture}` + HAL-local descriptor/format/usage/sampler types
(`HalTextureDescriptor/Format/Usage`, `HalSamplerDescriptor`, origin/
extent/layout) — buffer-copy path behaviorally identical.
`HalDevice::create_texture/create_sampler` now take the descriptor
(Noop/Vulkan ignore ⇒ unchanged; `HalTexture`/`HalSampler` made
`Clone`). metal: real `metal::Texture` (2D, mapped `MTLPixelFormat`,
`StorageModeShared`, usage flags; `inner: Option` so unsupported/
failed → error not panic; rejects depth/array/mip/sample ≠ 1) +
real `metal::SamplerState`; per-variant blit encoders
(`encode_buffer/buffer_to_texture/texture_to_buffer/texture_to_
texture`) with `checked_add` origin/extent validation; no
`unwrap`/`expect`/panic. core: `TextureCopyCommand{B2T,T2B,T2T}`
recorded on *successfully validated* copies (P6.3 validation
unchanged), carried in `CommandBuffer.texture_copies`; `Texture::
hal()` + `hal_texture_descriptor()`; `Queue::submit` translates
buffer+texture copies to `HalCopy`, skipping any with no real HAL
object (Noop stays a pure no-op) → `submit_copies`. Bounded subset:
2D / 1 layer / mip0 / color formats (R8/RGBA8/BGRA8 Unorm). Tests
`e2e_metal_texture.rs` (4, `#[ignore]`/cfg-gated). Gate: Noop `cargo
test --workspace` 46 binaries green (`command_texture_copy_
validation` 4/4 unchanged) + clippy clean; `cargo build/clippy
-p yawgpu --features metal` clean. Committed `phase-7: P7.3`.

### P7.3 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_texture -- --ignored`: **4/4 pass**
  (`metal_buffer_texture_buffer_round_trip`,
  `metal_texture_texture_round_trip`,
  `metal_sampler_creation_has_no_device_error`,
  `default_noop_texture_and_sampler_path_has_no_device_error`).
  Regression re-run: e2e_metal_basic 3/3 + e2e_metal_buffer 3/3 +
  e2e_metal_smoke 1/1 (the `submit_copies` rename did not regress
  P7.1/P7.2). **Real B2T→T2B & T2T pixel round-trip confirmed.**

## P7.4 — Metal Shader (naga→MSL) + compute dispatch  *(☑ DONE — real-GPU-verified)*

Done: `naga` gains `msl-out`. `shader_naga::generate_msl(entry,
&MslBindingMap) -> GeneratedMsl{source, entry_point}` via
`naga::back::msl` (per-entry-point resource map; resolves the emitted
Metal fn name via `info.entry_point_names`; all errors `Result`, no
naga leak/panic). Deterministic binding map: core sorts pipeline
bind-group buffer bindings by `(group,binding)` → `metal_index =
sorted position`, the *single* source feeding both MSL codegen and
runtime `set_buffer` (contract holds). HAL: `HalComputePipeline`
enum + `HalDevice::create_compute_pipeline(msl,entry,wg)` (Noop/
Vulkan stub; Metal `new_library_with_source`→`get_function`→
`new_compute_pipeline_state_with_function`, failure →
`HalError::ShaderCompilationFailed`); `HalCopy::ComputePass(HalCompute
Pass{pipeline,bind_buffers,workgroups})` executed in recorded order
with copies; metal `new_compute_command_encoder`→
`set_compute_pipeline_state`→`set_buffer` per binding→
`dispatch_thread_groups`. Each copy/pass now gets its own
encoder (blit-vs-compute correctness fix; no panic, offset/Metal-
backing checked). core: `ComputePipeline` holds `Option<HalCompute
Pipeline>` + binding map (naga/Metal fail ⇒ existing error-pipeline
path, not panic); `ComputePassEncoder` records exec ops after P6.5
validation; `Queue::submit` translates → `HalCopy::ComputePass`,
skips no-HAL (Noop pure no-op). Bounded: compute only, uniform/
storage buffers, no textures/samplers/indirect/render; P5/P6
validation unchanged. Tests `e2e_metal_compute.rs` (3,
`#[ignore]`/cfg). Gate: Noop `cargo test --workspace` 47 binaries
green (`compute_pipeline_validation` 7/7, `pass_state_validation`
9/9 unchanged) + clippy clean; `cargo build/clippy -p yawgpu
--features metal` clean. Committed `phase-7: P7.4`.

### P7.4 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_compute -- --ignored`: **3/3 pass**
  (`metal_compute_fills_storage_buffer`,
  `metal_compute_reads_input_and_writes_output_storage_buffers`,
  `default_noop_compute_path_has_no_device_error`). Regression
  re-run: basic 3/3 + buffer 3/3 + texture 4/4 + smoke 1/1 (no
  regression from the per-op-encoder refactor / `HalCopy::ComputePass`).
  **Real WGSL→MSL→MTLComputePipelineState→dispatch→storage-buffer
  readback confirmed.**

## P7.5 — Metal render pipeline + render pass draw  *(☑ DONE — real-GPU-verified)*

Done: `shader_naga::generate_render_msl` emits one MSL module covering
vertex+fragment entry points (per-entry-point resource map; naga
`vertex_buffer_mappings`; emitted names via shared
`emitted_entry_point_name`; `Result`, no panic/leak). Vertex-buffer
index contract (extends P7.4): `metal_vertex_buffer_binding_map`
assigns vertex-buffer `metal_index = bind_group_buffer_count + slot`
— **collision-free** (strictly above all bind-group buffer indices),
single `vertex_buffer_bindings` source feeding both MSL codegen and
runtime `set_vertex_buffer`. HAL: `HalRenderPipeline` +
`HalDevice::create_render_pipeline(msl,vtx,frag,
&HalRenderPipelineDescriptor)` (HAL-local color-format/vertex-layout/
topology types; Metal library→vtx+frag function→
`MTLRenderPipelineDescriptor`+`MTLVertexDescriptor`→
`new_render_pipeline_state`, failure→`HalError`, no panic);
`HalCopy::RenderPass(HalRenderPass{pipeline,color_target(tex,load/
clear/store),vertex_buffers,bind_buffers,draw})` executed in recorded
order; metal `MTLRenderPassDescriptor`→`new_render_command_encoder`→
`set_render_pipeline_state`→`set_vertex_buffer`(+bind buffers in the
collision-free index space)→`draw_primitives`. core: `RenderPipeline`
holds `Option<HalRenderPipeline>`+maps (naga/Metal fail ⇒ existing
error-pipeline path, not panic); `RenderPassEncoder` records
`RenderPassCommand` after P6.4/P6.5 validation; `Queue::submit`
translates → `HalCopy::RenderPass`, skips no-HAL (Noop pure no-op).
Bounded: 1 `RGBA8Unorm` color target, non-indexed `draw`, vertex
buffer + uniform, no depth/MSAA/index/indirect/shader-tex; P5/P6
validation unchanged. Tests `e2e_metal_render.rs` (3,
`#[ignore]`/cfg). Gate: Noop `cargo test --workspace` 48 binaries
green (`render_pipeline_validation` 14, `render_pass_descriptor_
validation` 5, `pass_state_validation` 9 — all unchanged) + clippy
clean; `cargo build/clippy -p yawgpu --features metal` clean.
Committed `phase-7: P7.5`. **Metal backend bring-up complete
(P7.1–P7.5).**

### P7.5 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_render -- --ignored`: **3/3 pass**
  (`metal_render_constant_color_triangle_readback`,
  `metal_render_uniform_color_triangle_readback`,
  `default_noop_render_path_has_no_device_error`). Full regression
  re-run: basic 3/3 + buffer 3/3 + compute 3/3 + texture 4/4 +
  smoke 1/1 (no regression). **Real WGSL vtx+frag → MSL →
  MTLRenderPipelineState → render-pass draw → texture → T2B →
  pixel readback confirmed.**

## P7.6 — Vulkan bring-up (mirror P7.1–P7.5)  *(NEXT)*
`ash` + MoltenVK on macOS; naga→SPIR-V; same HAL contract; reuse the
ported end2end tests parametrized by backend feature.

## Phase 7 exit criteria

- Metal + Vulkan fill their HAL enum arms; `yawgpu-core` validation
  unchanged & still green on Noop; per-slice `--features` build +
  clippy clean.
- Ported `end2end` Basic/Compute/Copy pass on real GPU (Metal on this
  machine; Vulkan as available) — user-run, logged here per slice.
- One commit per slice (`phase-7: <slice> — <short>`).
- **Mandatory Phase 7 Review** before COMPLETE; logged in
  `tracking/phase-7-review.md`.
