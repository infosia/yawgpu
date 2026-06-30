# TBDR re-home onto Tint — integration plan

> **Status:** PLAN (not started). Authored 2026-06-30.
> Supersedes the "removed" disposition of [Block 55](../blocks/55-tiled-rendering.md)
> with a Tint-era re-introduction path. The naga-fork TBDR feature was deleted in
> commit `78cdf48` (2026-06-26, Tint-migration Phase 0); this plan re-homes the
> HAL/core/FFI surface onto the **Tint** shader frontend.

## 1. Framing — two halves of one feature

TBDR (tile-based deferred rendering) splits cleanly along the shader-compiler
boundary:

- **Shader-compiler half** — making the WGSL→{MSL, SPIR-V} compiler emit
  input-attachment / framebuffer-fetch. In yawgpu this is **Tint** (the vendored
  `third_party/dawn` submodule, driven via `yawgpu-tint`). The design for the
  Tint-side work lives in the sibling Dawn clone's `TILED.md`
  (`dawn`, branch `feature/tiled`).
- **HAL/core/FFI half** — transient attachments, subpass passes, subpass
  pipelines, the vendor C ABI. yawgpu had this (naga-era) and deleted it in
  `78cdf48`. Block 55 is the retained historical spec.

**Consequence:** `git revert 78cdf48` does **not** work. The deleted code drove a
naga fork that emitted `[[color(N)]]` for `subpass_input`/`subpassLoad` and SPIR-V
`SubpassData`. That naga capability is gone. The logic must be **ported** to
Tint's WGSL surface, not reverted.

### WGSL surface change (user-facing)

The authored WGSL changes from the naga/wgpu-tiled surface to Tint's:

| Concept | naga-era (deleted) | Tint surface (target) |
|---|---|---|
| input-attachment type | `subpass_input<T>` | `input_attachment<T>` (`enable chromium_internal_input_attachments`) |
| load builtin | `subpassLoad(x)` | `inputAttachmentLoad(x)` |
| index attribute | (binding) | `@input_attachment_index(N)` |
| framebuffer fetch | `@color(N)` | `@color(N)` (`enable chromium_experimental_framebuffer_fetch`) — same |

Block 55, when un-removed, must be rewritten to the Tint surface.

## 2. Confirmed decisions (user, 2026-06-30)

1. **Phase from framebuffer-fetch first.** Slice 1 uses the `@color(N)`
   framebuffer-fetch surface, which **already works on the currently-pinned
   upstream Tint** for both MSL and SPIR-V — no Dawn fork required. Multi-subpass
   `input_attachment` (deferred rendering) comes in later slices.
2. **Dawn-fork dependency — channel pinned (2026-06-30), pin bumped (2026-06-30).**
   `third_party/dawn` was re-pointed from upstream `dawn.googlesource.com` to the
   **`infosia/dawn` fork's `feature/tiled` branch** (commit `f586cbd`; `.gitmodules`
   `branch = feature/tiled`). **The Tint-side TBDR work has now landed on the fork**
   (the user implemented it; 7 commits `c8f5ca3 → a05085e54f`) and the pin was bumped
   to `a05085e54f` (commit `0d53746`, clean `yawgpu-tint --features tiled` rebuild
   green). What landed:
   - **MSL** `input_attachment` → `[[color(N)]]` via a fork-only raise transform,
     driven by the new additive `Options::input_attachment_to_color_index`
     (`unordered_map<BindingPoint, uint32_t>`) — **the §5 color-slot map is now a
     real Tint Options field** yawgpu must populate from the pass layout.
   - **SPIR-V MSAA** `input_attachment` via `Options::multisampled_input_attachment`
     (module-wide flag) + a 2-arg `inputAttachmentLoad(ia, sample_index)` overload.
   - Fork conventions / divergence map: `docs/tint/tbdr-fork-conventions.md`
     (`// tint-tbdr:` markers; `git grep tint-tbdr:` enumerates the rebase surface).

   **Slice 2 and Slice 4 are now unblocked on the Tint side.** Remaining work is the
   yawgpu HAL/core/FFI port (this plan, §6).
3. **Metal + Vulkan only.** GLES Tier 2 (framebuffer-fetch / FBO-rebind) is
   deferred to a later slice — matches Dawn `TILED.md` scope (GLES/HLSL excluded).
   Color aspect only (no depth/stencil subpass inputs).

## 3. Capability baseline on the pinned upstream Tint

Verified from the Dawn `TILED.md` current-state grounding (applies to the pinned
`c8f5ca3`):

- `@color(N)` framebuffer fetch: **works on SPIR-V AND MSL upstream.** ← Slice 1 rides this.
- `input_attachment<T>` + `inputAttachmentLoad()`: works on **SPIR-V single-sampled
  only**; **MSL rejects it** (`msl/writer/writer.cc:65-67`). ← needs Dawn fork (Slice 2).
- SPIR-V MSAA `input_attachment`: not present. ← needs Dawn fork (Slice 4).
- `yawgpu-tint` already exposes `ResourceType::InputAttachment = 14` in reflection,
  but threads **no** `@color`/input-attachment `Options`.

## 4. Layered architecture (where each change lands)

| Layer | File(s) | Change |
|---|---|---|
| **Tint (Dawn submodule)** | `third_party/dawn` | Slice 1: none (upstream `@color` suffices). Slice 2+: per Dawn `TILED.md` (deferred). |
| **shim** | `yawgpu-tint/shim/{tint_shim.h,tint_shim.cpp}`, `src/lib.rs` | Enable `@color` framebuffer-fetch codegen path + reflect which fragment outputs are `@color`; Slice 2+ adds the `input_attachment_to_color_index` (MSL) map + `multisampled_input_attachment` (SPIR-V) `Options`. |
| **HAL** | `yawgpu-hal/src/{command,shader,descriptors,lib}.rs`, `metal/encode.rs`, `vulkan/{encode,pipeline}.rs` | `HalDescriptorBindingKind::InputAttachment`, input-attachment slot metadata on `HalShaderSource`, Metal `[[color(N)]]` bind + Vulkan input-attachment descriptors / multi-subpass `VkRenderPass`. Port from `78cdf48`. |
| **core** | `yawgpu-core/src/{subpass,transient_attachment,bind_group_layout,render_pipeline,device,adapter,lib}.rs` | Restore `subpass.rs` / `transient_attachment.rs`, `BindingLayoutKind::InputAttachment`, subpass pipeline + the **color-slot map computed from the pass layout** at pipeline-compile time (the naga `subpass_color_slots` analog). Port from `78cdf48`. |
| **FFI** | `yawgpu/ffi/webgpu-headers/yawgpu.h`, `yawgpu/src/{ffi,conv}/*`, `src/lib.rs` | Restore vendor entry points + SType `0x7000_0010–1F` (reserved). Template = external-texture (`ffi/external_texture.rs`). |
| **verify** | `yawgpu/tests/e2e_{metal,vulkan}_tiled.rs`, CTS, Block 55 rewrite | Real-GPU e2e on the M2; rewrite Block 55 to Tint surface; fork-conventions note when Dawn fork lands. |

## 5. The one hard data dependency (flag in every review)

Metal has no subpass-input texture type — both naga and Tint surface it as a
`[[color(N)]]` fragment argument. So **yawgpu-core must compute the
WGSL-binding → Metal-color-slot map from the render-pass / subpass-pass layout at
pipeline-compile time** and feed it through the shim into Tint's MSL `Options`.
Tint cannot infer it. This is exactly the deleted code's `subpass_color_slots`
role. For Slice 1 (framebuffer-fetch) the `@color(N)` index *is* the color slot,
so the map is identity and this dependency is trivial; it becomes real in Slice 2.

## 6. Slice plan

### Slice 1 — framebuffer-fetch `@color(N)`, single pass (no Dawn fork)
De-risks the whole yawgpu plumbing against capability that already exists.
1. shim: enable + reflect the `@color(N)` framebuffer-fetch path (MSL + SPIR-V);
   surface per-output `@color` usage in reflection.
2. core: `BindingLayoutKind`/fragment-output validation for `@color`; advertise a
   feature bit (`tiled` cargo feature, default off — matches Block 55).
3. HAL Metal: bind prior color slot as `[[color(N)]]` programmable-blend input.
4. HAL Vulkan: self-dependency / tile-image (or single-subpass input attachment)
   for `@color`.
5. FFI: minimal vendor surface to drive a framebuffer-fetch pipeline.
6. e2e: `e2e_{metal,vulkan}_tiled.rs` framebuffer-fetch smoke (real GPU, M2).
7. Rewrite Block 55 (un-remove) to the Tint `@color` surface for this slice.

### Slice 1 status (2026-06-30)
- **1.1 shim** (commit `1aee426`): `tiled` feature gates `@color` parse; reflection
  exposes `@color(N)`; SPIR-V framebuffer-fetch descriptor set threaded from core.
- **1.2 core** (`79c117c`): `tiled` propagation + `fragment_color_inputs` +
  `@color(N)`-vs-color-target validation. Noop-verified.
- **1.3 Metal** (`f7ae68d`): **zero HAL changes** — once core emits `[[color(N)]]`
  MSL, Metal programmable blending works. Real-Metal e2e green on the M2.
- **1.4 Vulkan**: full input-attachment infra built (dedicated set
  `S = bind_group_count`, `binding = N`; VkRenderPass self-read input refs in
  GENERAL layout + by-region self-dependency; INPUT_ATTACHMENT descriptor; shared
  render-pass construction between pipeline-time and draw-time passes).
  **Spec-valid (Vulkan validation clean).**
  - **MoltenVK GAP (finding):** MoltenVK does not map the input-attachment self-read
    (color attachment == input attachment) to Metal's tile read — it returns zero.
    The setup is validation-clean and the native-Metal `[[color(N)]]` analog (1.3)
    is verified, so the Vulkan path is **correct-by-construction but pixel-unverified
    on this Mac** (no native Vulkan HW). `e2e_vulkan_framebuffer_fetch` **skips on
    macOS/MoltenVK** and runs+verifies on native Vulkan (Linux/Windows). Consistent
    with [[moltenvk-shader-execution-limits]] (MoltenVK is not an authoritative oracle).

### Slice 2 — `input_attachment<T>` multi-subpass deferred (Dawn fork landed 2026-06-30)
**UNBLOCKED** (pin at `a05085e54f`). Restores transient attachments, multi-subpass
passes, subpass pipelines, the 3-subpass deferred example. Port HAL/core from
`78cdf48`, adapt to Tint `input_attachment` + the color-slot map (§5, now the real
`Options::input_attachment_to_color_index` Tint field). Too large for one handoff —
broken into sub-slices, mirroring how Slice 1 ran (shim → core → HAL → FFI → e2e):

- **2.1 shim** — thread the two new Tint Options through the C ABI:
  `input_attachment_to_color_index` (MSL, per-`BindingPoint` → color slot) and
  `multisampled_input_attachment` (SPIR-V, module flag). Reflect `input_attachment<T>`
  module-scope vars + their `@input_attachment_index(N)` so core can build the map.
  Mirrors Slice 1.1; replaces the FB-fetch placeholder path for the handle surface.
  **DONE** (commit `cd12955`): `YawgpuTintInputAttachmentColorIndex` array on the
  bindings struct → `Options::input_attachment_to_color_index`; `multisampled_input_attachment`
  bool on `generate_spirv` (yawgpu-core single-sampled call sites pass `false`);
  `ResourceBinding::input_attachment_index` reflected. Tests: MSL `[[color(0)]]`
  lowering via the map, missing-slot → clean `Err`, single-sampled SPIR-V, MSAA-flag
  reaches the writer. All workspace + tiled gates green.
- **2.2 core** — restore `BindingLayoutKind::InputAttachment` (handle binding, distinct
  from the `@color` FB-fetch path); bind-group-layout validation; compute the
  WGSL-binding → Metal color-slot map from the subpass/pass layout at pipeline-compile
  (the §5 seam) and feed it into the shim. Port from `78cdf48` core diff.
- **2.3 core — subpass/transient** — restore `subpass.rs` (1775 deleted lines) +
  `transient_attachment.rs`. Too large for one handoff; split:
  - **2.3a** — subpass *layout types* + validation (`SubpassPassLayout`,
    `SubpassPassLayoutDescriptor`, `SubpassInputAttachment`/`SubpassDependency`,
    `validate_subpass_pass_layout_descriptor`, `TiledCapabilities`). Static data +
    validation only; Noop-testable; no encoder, no HAL, no codegen.
  - **2.3b** — subpass *render pipeline* + the **color-slot map** computation
    (`new_subpass`, `SubpassPipelineCompatibility`, the deleted `subpass_color_slots`
    analog) + codegen wiring: feed the WGSL-binding → color-slot map into the shim's
    `input_attachment_to_color_index` (§5 seam). Needs 2.3a.
    **RE-PLAN (2026-06-30, after reading the 78cdf48 source):** the color-slot map
    is computed *inside* `create_hal_subpass_render_pipeline`, the core→HAL seam —
    `subpass_color_slots: Vec<((group,binding), source_attachment)>` derives from the
    subpass's `input_attachments` (Metal slot = `input.source_attachment`, NOT
    identity), is threaded through `select_render_shader_source` into MSL codegen,
    AND drives `create_subpass_render_pipeline` (HAL). So the map + codegen + HAL
    pipeline object are **one cohesive unit and do not split cleanly into a core-only
    2.3b + HAL-only 2.4**. Revised approach: do **2.3b as a Metal-first vertical**
    (core map + codegen + Metal HAL subpass pipeline + real-Metal e2e), mirroring how
    Slice 1.3 landed Metal FB-fetch with near-zero HAL; then a **Vulkan vertical**
    (multi-subpass `VkRenderPass` + INPUT_ATTACHMENT descriptors, adapting the new
    Slice-1.4 `HalDescriptorBindingKind::InputAttachment { color_slot }` shape — note
    the deleted code used the older `HalBufferBindingKind::InputAttachment`, which no
    longer exists). The map computation can still be unit-tested in isolation
    (`compute_subpass_color_slots`) before the HAL hookup.
    - **2.3b (core map + codegen) DONE** (commit `f07ce2d`): `compute_subpass_color_slots`
      + `ReflectedModule::generate_render_fragment_msl` threads the map into
      `Bindings.input_attachment_color_index`; `select_render_shader_source` takes
      `subpass_color_slots` (existing callers pass `&[]`). Tests prove NON-identity
      application (`source_attachment=1` → MSL `[[color(1)]]`, not `[[color(0)]]`;
      empty map → clean `Err`). Noop-tested.
    - **2.3b-2 (subpass pipeline object + device entry) DONE** (commit `47d18b2`):
      `SubpassRenderPipelineDescriptor` / `RenderPipeline::new_subpass` /
      `SubpassPipelineCompatibility` + `Device::create_subpass_pass_layout` +
      `Device::create_subpass_render_pipeline`. **Confirmed Metal needs NO new HAL** —
      the subpass path reuses the regular `create_render_pipeline` via a shared
      `create_hal_render_pipeline_with_subpass_color_slots` (regular path passes `&[]`,
      subpass path passes the computed slots). `resolve` gained an `Option<&[u32]>`
      subpass color-attachment-indices param to validate fragment outputs. Noop-tested.
  - **2.3c** — `SubpassRenderPass` runtime encoder (begin / `next_subpass` / draw /
    `set_bind_group` / `end` + `resolve_subpass_render_pass_resources`) + the device
    entry points that create pass layouts / subpass passes. Needs 2.3a+b.
  - **transient_attachment.rs** (146 lines) folds in where the attachment type is
    needed (likely 2.3a data + 2.4/2.5 HAL storage modes).
- **2.4 HAL Metal** — `[[color(N)]]` bind via the map; `MTLStorageMode::Memoryless`
  transient attachments. Port from `78cdf48` metal diff.
- **2.5 HAL Vulkan** — multi-subpass `VkRenderPass`, input-attachment descriptors
  (Slice 1.4 already built the single self-read infra — extend, don't duplicate),
  `LAZILY_ALLOCATED` transient attachments. Port from `78cdf48` vulkan diff.
- **2.6 FFI** — restore the vendor entry points / SType range for declaring subpasses
  + transient attachments + input-attachment bind-group entries. Template =
  external-texture (`ffi/external_texture.rs`). Port from `78cdf48` ffi diff.
- **2.7 e2e + Block 55** — real-GPU `e2e_metal_tiled.rs` (3-subpass deferred) on the
  M2; `e2e_vulkan_tiled.rs` (skips MoltenVK per Slice 1.4 finding); rewrite Block 55
  to the full Tint `input_attachment` surface.

### Slice 3 — transient / memoryless attachments
Vulkan `LAZILY_ALLOCATED` + `TRANSIENT_ATTACHMENT|INPUT_ATTACHMENT`; Metal
`MTLStorageMode::Memoryless`. The bandwidth-saving payoff. (May fold into Slice 2.)

### Slice 4 — SPIR-V MSAA input attachment (most divergent Dawn work)
Per Dawn `TILED.md` Phase 3 (`core.def` overload + `builtin_polyfill.cc` edits).
Land last. MSL MSAA subpass input stays deferred (Metal uses `[[sample_id]]`).

**RESOLVED 2026-06-30 (pin `8dce6b2387`, commit `2b817dd`).** Root cause was exactly
as diagnosed: `inputAttachmentLoad` is declared in **both** `core.def` (core IR table)
and `wgsl.def` (the WGSL frontend/resolver table); the 2-arg MSAA overload had been
added only to `core.def`, so the resolver — which uses the `wgsl.def`-derived table —
rejected it (the builtin-polyfill unit tests build IR directly and bypass the resolver,
which is why they didn't catch it). The fork's `e99bd175e0` + `8dce6b2387` mirror the
overload into `wgsl.def` + regenerate. **End-to-end verified in yawgpu-tint** after a
clean rebuild: the 2-arg shader parses and `generate_spirv(multisampled_input_attachment
=true)` emits valid SPIR-V (186 words, magic `0x07230203`) carrying a Sample image-operand
(multisampled SubpassData path). Slice 4 is now unblocked. (Historical detail of the
original finding below, kept for the diagnosis trail.)

**Original finding (now RESOLVED above).** The
2-arg `inputAttachmentLoad(ia, sample_index)` overload did **not resolve in
yawgpu-tint's WGSL frontend** at the pinned commit `a05085e54f`, even though the
generated core intrinsic table looked correct. Repro: a fragment shader with
`enable chromium_internal_input_attachments;` calling `inputAttachmentLoad(ia, sid)`
fails `Program::parse` with *"no matching call … 1 candidate function"* (only the
1-arg overload is listed). Verified on a guaranteed-clean rebuild
(`cargo clean -p yawgpu-tint`), so it is **not** a stale-build artifact. Static
audit of the pinned `third_party/dawn` tree shows the table is *well-formed*:
`core.def:1599` has the overload; the regen commit `e716902f99` is an ancestor of
the pin; `core/intrinsic/data.cc` builtin `[110] inputAttachmentLoad` has
`num overloads = 2` at `OverloadIndex(548)`; overload `[549]` has
`num_parameters = 2` with `kParameters[445]=kInputAttachment` +
`[446]=kSampleIndex` and `kSupportsFragmentPipeline`. So the table the resolver
links *contains* a valid 2-arg overload, yet the resolver surfaces only one — a
**Dawn-side resolver/table issue for the user to investigate in their Dawn checkout**
(e.g. via `tint_unittests --gtest_filter='*InputAttachment*'`), per the
do-not-drive-Dawn boundary. Slice 2.1's `multisampled_input_attachment` plumbing is
verified at the *flag-reaches-the-writer* level (the writer's arity/option-mismatch
`Failure`); the end-to-end 2-arg MSAA path stays parked until this resolves.

### Deferred (documented)
GLES Tier A/B; depth/stencil-aspect subpass inputs; MSL MSAA subpass input;
programmable tile dispatch; HLSL/D3D (permanently out of scope).

## 7. Risks

1. **naga→Tint WGSL surface change** is user-facing; Block 55 rewrite + any
   example WGSL must move to `input_attachment`/`inputAttachmentLoad`/`@color`.
2. **No git-revert.** Port logic from `78cdf48`; the diff won't apply to the
   Tint-era tree.
3. **Color-slot map** (§5) is the integration seam for Metal — same trap as the
   old `subpass_color_slots`; verify on real GPU (Nv12 e2e–style luck-vs-correct).
4. **Dawn fork maintenance.** Re-pinning the submodule to a `feature/tiled` fork
   raises the rebase cost vs upstream chromium/N; decide mechanism at Slice 2.
5. **Tint MSL `ModuleScopeVars` ordering** (Dawn `TILED.md` Option A) is the Dawn
   fork's risk, but yawgpu's reflection must agree with whatever slot model it picks.

## 8. Verification

- Noop-first (CLAUDE.md principle 2) for all core validation; inline unit tests
  per public fn (principle 1).
- Real-GPU e2e on the M2 (Metal directly; Vulkan/MoltenVK) — Claude runs these
  (see [[claude-runs-real-gpu-tests]]).
- CTS: framebuffer-fetch / input-attachment are vendor/outside-WebGPU-core, so no
  CTS regression risk; confirm the standard api trees stay green.
- Phase Review (Clean-Review-Then-Fix) closes each slice.
