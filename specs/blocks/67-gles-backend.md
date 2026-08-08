# Block 67 — GLES backend (Phase 15, Tier 2 / experimental)

Phase 15 introduces an **OpenGL ES 3.1+ backend** behind the existing
enum-dispatch HAL, targeted at **Android** (native EGL +
`libEGL.so` / `libGLESv3.so`) and **Windows ANGLE**
(`libEGL.dll` / `libGLESv2.dll`). It is positioned as a **Tier 2 /
experimental** backend (see `CLAUDE.md` "Backend support tiers"):
shipped behind the opt-in `gles` cargo feature, with WebGPU semantics
mapped on a **best-effort** basis. Core validation (`yawgpu-core`) is
Tier-independent — it never relaxes a rule for GLES. When a validated
WebGPU operation cannot be cleanly mapped to GLES 3.1, the GLES HAL
arm returns `HalError`, which `yawgpu-core` surfaces as a device error
(no panic). Unmapped paths are catalogued in the **mapping matrix**
below and refined as P15.x slices land.

## Scope decisions (authoritative)

- **Platforms: Android + Windows ANGLE only.** Linux EGL desktop, X11,
  Wayland, WGL, WebGL, and Emscripten are explicitly **out of scope**
  for Phase 15. The EGL code path is a subset of wgpu-hal/src/gles/egl.rs
  trimmed to the two target platforms.
- **Tier 2 / experimental.** `--features gles` is opt-in; never in
  `default`. No runtime marker is added (no `AdapterInfo` suffix, no
  `log::warn!`, no C `#define`) — the cargo feature is the experimental
  signal. Docs (this file, `CLAUDE.md`, `DESIGN.md`, `SPEC.md`,
  `README.md`) carry the Tier 2 wording.
- **`yawgpu.h` vendor extensions are NOT implemented for GLES.**
  `tiled` (Phase 14) and `shader-passthrough` (Phase 13) feature
  surfaces are absent on the GLES adapter; the relevant features
  are not advertised and the corresponding extension FFI calls return
  the existing "feature not enabled" / "backend unavailable" device
  errors when called against a GLES device.
- **CI policy unchanged: Noop-only.** `cargo test --workspace` and
  `cargo clippy --workspace --all-targets -- -D warnings` stay green on
  Noop. Each slice must also **build** with `--features gles`
  (`cargo build -p yawgpu --features gles`, clippy too). Real-GPU
  verification follows the Phase 7 pattern: e2e tests are `#[ignore]`d
  (or self-skip when no GLES adapter is present); **the user runs
  `cargo test --features gles -- --ignored` manually and logs results in
  the relevant `specs/tracking/<topic>.md`** (per-phase `phase-N.md` logs
  are no longer written — see `CLAUDE.md`). In practice the current
  real-GPU route on Windows is **WGL against the native driver**
  (`YAWGPU_GLES_BACKEND=wgl`), not ANGLE: see the "Context backend
  (Windows)" matrix row. The GLES-specific ledger is
  `tracking/cts-gles-sweep-0705.md`.
- **No-panic principle still holds.** EGL / GL driver errors
  (`eglGetError`, `glGetError`) map to `HalError`, surfaced as device
  errors. The FFI-boundary `expect` exception (CLAUDE.md core principle 3)
  does not extend to GLES bring-up code.
- **Out of scope for Phase 15:** D3D backends (permanent); desktop GL
  (4.x) / Wayland / X11 / WebGL / Emscripten; multi-context
  threading beyond single-shared-context serialization; persistent
  buffer mapping fallback emulation when `GL_EXT_buffer_storage` is
  absent (use per-call `glMapBufferRange`); ANGLE backend selection
  (D3D11 vs Vulkan inside ANGLE — leave to ANGLE defaults).
  **WGL** was originally out-of-scope but was added post-COMPLETE
  (2026-05-25) as a Windows-only opt-in verification path; see the
  "Context backend (Windows)" matrix row. A later post-COMPLETE addition
  exposed `YaWGPUGlesContextBackend` /
  `YAWGPU_STYPE_GLES_CONTEXT_BACKEND` so applications can force
  EGL or WGL from the instance descriptor; a non-default chain value
  wins over `YAWGPU_GLES_BACKEND`.

### Minimum GLES version

The GLES HAL targets **OpenGL ES 3.1** as its minimum. Context creation
asks for `EGL_CONTEXT_MAJOR_VERSION = 3` + `EGL_CONTEXT_MINOR_VERSION = 1`
and falls back to plain `EGL_CONTEXT_CLIENT_VERSION = 3` when the driver
rejects the minor attribute, so the real floor is the subsequent
`glGetString(GL_VERSION)` parse: below 3.1 → `HalError::DeviceCreationFailed`.
The adapter constructors are `GlesAdapter::new_egl` / `new_wgl` (both
`-> Result<Self, HalError>`; there is no `GlesAdapter::new`, and neither
returns `Option`), and the version check lives on the **device-creation**
path — `create_egl_device` and the WGL equivalent in `gles/wgl.rs` — not in
adapter construction. `enumerate_adapters` does drop an adapter whose
capability probe fails, but logs the reason via `eprintln!` first rather
than dropping it silently. GLES 3.2 features (e.g.
`glDrawElementsBaseVertex`, broader compute / storage-texture format
support) are **opportunistically used when reported** but never
required.

Rationale: GLES 3.1 is the floor needed for WebGPU's compute path
(compute shaders + SSBOs + image load/store + indirect dispatch) and
is the de-facto Android baseline for hardware shipped since ~2016
(Mali-T7xx, Adreno 4xx, PowerVR Series 6XT and later). ANGLE on
Windows targets ES 3.1 unconditionally. Targeting 3.0 would lose
compute entirely; targeting 3.2 would unnecessarily exclude
mid-range Android devices still in active use.

## HAL contract the GLES backend must satisfy

The `yawgpu-core` ↔ `yawgpu-hal` seam is already exercised by Noop
and proven on Vulkan/Metal. The GLES backend implements the same enum
arms; **no `dyn Trait`** — add `cfg(feature = "gles")` arms to the
existing `HalInstance/Adapter/Device/Queue` + resource / command /
pipeline enums. New surface entry points:

- `HalInstance::create_surface_from_android_native_window(window: *mut c_void)`
  — Android `ANativeWindow*` → EGL window surface.
- The existing `HalInstance::create_surface_from_windows_hwnd` gains
  a GLES arm that calls `eglCreateWindowSurface` against ANGLE.

Per resource:

- **Instance**: owns the EGL display + dynamically loaded
  `libEGL`/`libGLES*` handles (via `khronos-egl` + `libloading`).
  `enumerate_adapters` returns one adapter per usable `EGLConfig`
  (typically one default RGBA8 config).
- **Adapter**: holds the `EGLConfig` + parsed GL_VERSION /
  GL_RENDERER / extension set; `create_device` creates the shared
  `EGLContext` and a `glow::Context` wrapper.
- **Device / Queue**: a single shared GL context per `HalDevice`,
  serialized by a parking-lot `Mutex<()>` ("`AdapterContextLock`"
  pattern from wgpu-hal/gles). `HalQueue` make-current's the context
  before issuing GL calls.
- **Buffer**: `GLuint` BUF + size + usage; `write` via
  `glBufferSubData` or persistent-mapped pointer when
  `GL_EXT_buffer_storage` is present; `read` via `glGetBufferSubData`
  or pixel-pack buffer fence.
- **Texture**: `GLuint` texture object created with immutable storage
  (`glTexStorage2D` / `glTexStorage3D`); descriptor stored alongside
  for view resolution.
- **TextureView**: not a separate GL object. Stored as
  `{parent_tex: HalTexture, base_mip, mip_count, base_layer,
  layer_count, aspect}` and resolved at bind/attach time.
- **Sampler**: `GLuint` sampler object (`glGenSamplers`).
- **Shader / Pipeline**: WGSL → GLSL ES 3.10 via the WGSL frontend's
  GLSL writer; compiled into a `GLuint` program. Bind-group layout + a
  derived linear-binding remap table are stored on the pipeline.
- **Compute pipeline**: program object + workgroup size.

> **Tint migration (2026-06-27; revised 2026-07-02).** The frontend is now
> Tint's `glsl::writer` (was naga `glsl-out`). GLES is Tier-2 and its
> real-GPU re-verification on ANGLE is **deferred**. The Tint-integration
> refactor slice R6 + Phase Review M2 (`specs/tracking/tint-integration-refactor.md`)
> already re-aligned the two load-bearing runtime contracts to Tint's real
> output: first-instance (`tint_immediates[0]` + instance-step attribute
> offsets — the naga `naga_vs_first_instance` uniform was a silent no-op)
> and buffer binding numbers (an explicit `BindingRemap` replacing the naga
> `_block_N` name-parse remap — R6 supplied an identity mapping; it has
> since become the dense per-class, group-collapsing remap described in the
> "GLSL binding numbers" matrix row). See the matrix rows below.
> The load-bearing naga-era names in the **technical-decisions table** and
> the **mapping matrix** were corrected on 2026-08-08 (`SamplerBindMap` →
> Tint's `CombinedSampler` list; `naga_vs_first_instance` →
> `tint_immediates[0]`). Remaining naga mentions live only in the
> historical P15.x slice list, which is explicitly marked as a record of
> the original plan.
- **Render pipeline**: program + draw state (topology, depth/stencil,
  blend, vertex attrib layout) + vertex array object cache.
- **Surface**: `EGLSurface` wrapping `ANativeWindow*` (Android) or
  HWND-via-ANGLE (Windows); `acquire_next_texture` returns a virtual
  texture that resolves to the default framebuffer; `present` calls
  `eglSwapBuffers`.

Validation stays in `yawgpu-core` (Phases 0–8); the GLES backend only
**executes** already-validated work. A GL/EGL op failing at the driver
level → `HalError` → device error (no panic, no core-rule relaxation).

## Slices → bring-up targets

Real-backend e2e tests already ported in Phase 7 are reused unchanged.
Each slice: Red (run the existing `e2e_*` test under
`--features gles -- --ignored`, fails / unimplemented) → Green
(backend impl) → user runs `--ignored` on real GLES hardware, reports,
logged in the relevant `specs/tracking/<topic>.md`.

> **Historical.** The P15.x list below is the original bring-up plan as
> written in 2026-05; it is kept as a record and is **not** a current-state
> description. naga references in it are pre-Tint archaeology (see the Tint
> migration note under "HAL contract"). For current state, read the mapping
> matrix.

- **P15.0** Scaffolding + gating harness. Add `gles` feature, deps
  (`glow`, `khronos-egl`, `libloading`), `HalBackend::Gles`, every HAL
  enum's `Gles` arm returning `HalError::BackendUnavailable` (or
  equivalent). `naga` `glsl-out` feature enabled in workspace.
  Documentation edits (`CLAUDE.md`, `DESIGN.md`, `SPEC.md`,
  `blocks/60-real-backends.md`, `README.md`) for Tier 2 wording.
  Acceptance: `cargo build -p yawgpu --features gles` clean; Noop +
  Vulkan + Metal gates unchanged; clippy `-D warnings` clean with the
  feature on.
- **P15.1** EGL display + adapter enumeration + shared GL context +
  empty `submit_empty`. ANGLE bring-up on Windows. Adapter-probe test
  helper in `yawgpu-test` (skip when no GLES adapter found). Reuses
  `e2e_basic` device/queue creation portion.
- **P15.2** Buffer create / write / read + Queue writeBuffer +
  buffer-to-buffer copy (`glCopyBufferSubData`). Reuses `e2e_buffer`.
- **P15.3** Texture (immutable storage) + Sampler + B2T / T2B / T2T
  copies (`glTexSubImage*`, pixel pack/unpack buffers). View
  resolution at bind time. Reuses `e2e_copy` texture subset.
- **P15.4** Shader (naga WGSL → GLSL ES 3.10) + bind-group → linear
  binding remap + compute pipeline + dispatch + indirect dispatch.
  Reuses `e2e_compute_dispatch`.
- **P15.5** Render pipeline + FBO + render pass + vertex attribs +
  draw / drawIndexed / drawIndirect. `first_instance` handled via
  naga-injected `naga_vs_first_instance` uniform set per draw.
  Reuses `e2e_basic` draw portion.
- **P15.6** Surface: Android `ANativeWindow*` and Windows HWND
  (ANGLE) → EGL window surface; `acquire_next_texture` /
  `eglSwapBuffers`. `examples/triangle` runs under
  `--features gles` on ANGLE.
- **Phase 15 Review** (mandatory Clean Review Then Fix) → COMPLETE.

## Technical decisions

| Topic | Decision | Rationale |
|---|---|---|
| GL context model | One shared `EGLContext` per `HalDevice`. `HalQueue` make-current's the context behind a `Mutex<()>`. | wgpu-hal/gles `AdapterContext` pattern. yawgpu HAL calls are currently serial; no need for multi-context complexity. |
| EGL loader | `khronos-egl` (dynamic) + `libloading`. `libEGL.so` on Android, `libEGL.dll` on Windows (ANGLE). Path resolution defers to the OS; an optional `YAWGPU_ANGLE_PATH` env var can preload from a specific directory before instance creation. | Avoids NDK / ANGLE-as-build-dep coupling. Matches wgpu-hal/gles. |
| GL function loader | `glow` over `eglGetProcAddress`. | Standard Rust GL binding; same as wgpu. |
| Buffer mapping | Per-call `glMapBufferRange` by default. Persistent mapping (`GL_MAP_PERSISTENT_BIT`) when `GL_EXT_buffer_storage` is advertised — never required. | Persistent path is common on Adreno / desktop ANGLE; Mali support varies by generation. |
| Texture views | Stored as `{parent, base_mip, mip_count, base_layer, layer_count, aspect}`. Resolved at bind / attach. | GLES has no view object. |
| Sampler / texture combining | **Tint** returns a `CombinedSampler { glsl_uniform_name, texture_group/binding, sampler_group/binding, uses_placeholder_sampler }` list; the pipeline resolves each to a `glGetUniformLocation` (`resolve_combined_samplers`) and the queue assigns texture units per draw/dispatch (`bind_combined_samplers`), substituting a placeholder sampler where Tint asks for one. There is no `SamplerBindMap` — that was the naga-era design and does not exist in the code. | WebGPU's separate sampler is the principal semantic gap; Tint's combined-sampler list is the same mechanism Dawn's GL backend uses. |
| Storage textures | GLES 3.1 `glBindImageTexture` + `image2D` shader qualifiers. Format coverage validated against the GLES image-format table; unsupported formats → `HalError::FormatUnsupported`-class. **Layer-subrange views (2026-07-08):** a 2d-array/3d storage view of a layer subrange (`base_array_layer>0` or `array_layer_count<full`, count>1) — which `glBindImageTexture` alone cannot express (only whole-layered or single-layer) — is bound by aliasing `[base, base+count)` with a transient `glTextureView` (reusing the sampled-view machinery `create_transient_texture_view`) then `glBindImageTexture(view, layered=GL_TRUE)`; the view has exactly `count` layers so view-relative layers and `imageSize().z`/`textureNumLayers` are correct. Requires glTextureView (GLES 3.2); the ES-3.1 fallback keeps the `HalError` (catalogued). Fixed 482 CTS storage_textures_2d_array / textureNumLayers / out_of_bounds_array fails. | GLES 3.1 image format coverage is narrow (R32F, Rgba8, Rgba32F, …). |
| `first_instance` | **Tint** emits `tint_immediates[0]` (via `Options::first_instance_offset`); the HAL sets it per draw with `glUniform1ui` and additionally offsets every `Instance`-stepped vertex attribute pointer by `first_instance * array_stride`. The naga-era `naga_vs_first_instance` uniform is gone — Tint never emitted it, so it was a silent no-op after the migration. `INDIRECT_FIRST_INSTANCE` is **not** advertised on GLES. | GLES lacks `gl_BaseInstance`; the attribute-offset half matches Dawn's GL backend (`CommandBufferGL.cpp`). |
| Compute / SSBO | Native GLES 3.1 compute + SSBO + indirect dispatch path. | Full WebGPU compute surface available. |
| Memory barriers | `glMemoryBarrier(...)` issued by HAL between hazard-prone ops based on the recorded HalCopy / pass structure. | GLES has no fine-grained barrier API; coarse-grained mask is sufficient for the e2e set. |
| WGSL → GLSL version | Target **GLES 3.10**. Use higher (3.20) only when an emitted feature demands it and the driver reports it. | Matches the 3.1 minimum; broadest device coverage. |
| ANGLE platform selection | Leave ANGLE to its default backend choice (typically D3D11 on Windows). Do not expose `EGL_ANGLE_platform_angle` controls from yawgpu. For environments where the locally available ANGLE caps at ES 3.0 (Chromium / CEF builds), the **WGL fallback** (`YAWGPU_GLES_BACKEND=wgl`, post-COMPLETE addition — see "Context backend (Windows)" row) bypasses ANGLE entirely and uses the host GL driver via `WGL_EXT_create_context_es2_profile`. | Keeps the surface simple; users wanting a specific ANGLE backend can env-var their own ANGLE build, and users without a workable ANGLE can fall back to WGL. |
| Error mapping | `HalError::BufferOperationFailed { backend: "gles", message }` for buffer-class GL failures plus the existing `backend`-only variants (`BackendUnavailable` / `DeviceCreationFailed` / `QueueSubmissionFailed` / `ShaderCompilationFailed`) and message-carrying surface variants (`AcquireFailed` / `PresentFailed` / `SwapchainCreationFailed`). | Mirrors Vk / Metal arms. *(Corrected from the earlier `BackendOperationFailed` wording — that variant does not exist on `HalError`; P15.1 used the real enum correctly.)* |
| Adapter selection from `wgpuCreateInstance` | yawgpu.h vendor extension `YaWGPUInstanceBackendSelect.backend = YAWGPU_INSTANCE_BACKEND_GLES = 3` pins the primary HAL to GLES at instance creation, mirroring the Metal/Vulkan pattern. The standard webgpu.h `WGPURequestAdapterOptions.backendType` field is accepted but treated identically across all backends (the primary HAL chosen at instance creation is what enumerates) — same behavior Vulkan/Metal exhibit. | One consistent selection path across all backends. |

## WebGPU × GLES mapping matrix

Status: ☑ Supported · ◐ Partial / restricted · ✗ Unsupported (HalError)

**This table is the authoritative current-state description of the GLES
backend** — prefer it over the historical scope/slice sections above.
Last reconciled against the source **2026-08-08**, after the
command-stream move (`88cfe58`). No **?** entries remain: the three that
were open (storage textures, indirect compute dispatch, bundle execution)
are resolved in place below, and any future **?** must be resolved before
the owning slice's review.

| Area | GLES 3.1 mapping | Status |
|---|---|---|
| Adapter / device creation | EGL display + shared context | ☑ (P15.1; ANGLE on Windows verified) |
| Buffer create / map / unmap | `glBufferData(&zeros, size, DYNAMIC_DRAW)` (**zero-initialized** — see below) + `glBufferSubData` (write) + `glMapBufferRange(MAP_READ_BIT)` (read). HostBuffer path in core (`mapped_ptr` returns `None`); persistent map deferred. **Zero-init (2026-07-08):** WebGPU requires buffers to behave as zero-initialized. GL recycles freed-buffer memory within the process, so `glBufferData(NULL)` would expose stale bytes from a destroyed buffer (Vulkan/Metal get zeroed fresh OS pages instead). `allocate_buffer` now uploads a host zero vector, both allocating and zeroing. This is Tier-independent conformance (GLES made to match the zero-init core already assumes); it fixed 15,081 `command_buffer,image_copy` "texture padding mismatch" fails — yawgpu's T2B preserves padding correctly (tight byte-granular row copies), but the preserved bytes were uninitialized garbage that differed between the two fresh buffers the FullCopyT2B check compares — plus 2 `resource_init,buffer` fails. Eager zero-init (a host zero-vec per buffer); large-buffer chunking / lazy-init is a possible future optimization. | ☑ (P15.2 + zero-init; ANGLE round-trip + crocus verified) |
| Buffer-to-buffer copy | `glCopyBufferSubData` via `GL_COPY_READ_BUFFER` / `GL_COPY_WRITE_BUFFER` | ☑ (P15.2; ANGLE round-trip verified, full + partial offsets) |
| Buffer clear | `glBufferSubData` zero-fill chunks via `GL_COPY_WRITE_BUFFER` | ☑ (F-023 follow-up; shares the existing GLES buffer write path) |
| `mappedAtCreation` | Allocate + map immediately; flush on unmap | ☑ (P15.2; transparent via HostBuffer path) |
| Texture: 1D | `GL_TEXTURE_2D` with height=1 (no native 1D in GLES) | ◐ Allocation supported (F-026 follow-up); copies go through the plain-2D path, so a copy addressing layers or a non-zero `z` returns `HalError` |
| Texture: 2D | `GL_TEXTURE_2D` + `glTexStorage2D` (non-multisample; uncompressed color formats known by core map to GLES internal/external/type triplets, including integer, signed-normalized, sRGB, 16/32-bit, and packed formats) | ☑ (P15.3; ANGLE verified; format coverage pass expands uncompressed color mappings) |
| Texture: 2D array | `GL_TEXTURE_2D_ARRAY` + `glTexStorage3D` | ☑ Allocation with layers and mips, plus B2T / T2B / T2T copies (`glTexSubImage3D` / `glCopyImageSubData` / `glCopyTexSubImage3D`) and layered FBO attachment via `glFramebufferTextureLayer` |
| Texture: 3D | `GL_TEXTURE_3D` + `glTexStorage3D` | ☑ Allocation with depth and mips, plus B2T / T2B / T2T copies and z-slice FBO attachment. T2B reads a 3D layer through a client-side staging path rather than a PBO |
| Texture: cube / cube-array | Base storage stays `GL_TEXTURE_2D_ARRAY`; a cube **view** aliases it through a transient `glTextureView` at bind time (see the "Flexible texture views" catalogue entry) — mirroring Dawn's opengl backend, so no `textureBindingViewDimension` hint is needed | ◐ Sampling a cube view works where `glTextureView` is available (ES 3.2 / `OES`/`EXT_texture_view`; cube-array additionally needs `supports_cube_map_array`). Without `glTextureView` the view shapes stay `HalError`, and cube / cube-array **color attachments** are rejected outright. Supersedes the earlier "✗ Deferred" |
| Texture views | Subrange metadata resolved by core; HAL receives `HalTexture` + mip/origin in copy descriptors | ☑ (P15.3; degenerate — no HAL-level view object) |
| Storage textures (read/write) | `glBindImageTexture` with `READ_ONLY` / `WRITE_ONLY` / `READ_WRITE` (`gles/queue.rs` `bind_storage_textures`), bound in **both** the compute and render-draw paths behind a `StorageImageCleanup` Drop guard. Layer-subrange views alias the base storage through a transient `glTextureView` (see the technical-decisions row). Residual Tier-2 `HalError`s: a format outside the GLES image table, a view exposing more than one mip level, a non-zero layer of a plain 2D storage texture, and — on ES 3.1 without `glTextureView` — a view that is neither whole-layered nor a single layer. | ◐ (T-G14 `d0ef874` + layer-subrange views `2ba7e96`; crocus verified. Resolves the former **?**) |
| B2T / T2B / T2T copies | `glTexSubImage2D` (PBO unpack + `UNPACK_ROW_LENGTH`) / transient FBO + `glReadPixels` (PBO pack + `PACK_ROW_LENGTH` + `read_buffer(COLOR_ATTACHMENT0)`) / `glCopyImageSubData` (GLES 3.2 or `GL_EXT_copy_image`) | ☑ (P15.3 + F-026 follow-through; 2D, 2D-array, and 3D all execute. The only shape rejection left is `ensure_2d_target_copy`: a copy addressing layers or a non-zero `z` **against a plain 2D texture** returns `HalError` — a correctness guard, not a missing mapping. Supersedes the earlier "2D only" status) |
| Sampler creation | `glGenSamplers` + `glSamplerParameteri/f` (filter / address / mipmap / compare / anisotropy via `GL_EXT_texture_filter_anisotropic`) | ☑ (P15.3; `ClampToBorder` not supported) |
| Compute shader | Tint WGSL → GLSL ES 3.10 (`use_framebuffer_fetch=false`, `zero_initialize_workgroup_memory=true`) compiled via `glCreateShader(COMPUTE_SHADER)` + `glLinkProgram`; bind-group bindings honored via `bind_buffer_range(SHADER_STORAGE_BUFFER \| UNIFORM_BUFFER)` against the remapped binding number (see the "GLSL binding numbers" row). **Multiple bind groups are supported** — `@group(1..3)` no longer returns `HalError`. **Texture bindings now work in compute** — the dispatch path binds combined samplers (`bind_combined_samplers`), the texture-metadata UBO, and storage images (`bind_storage_textures`) before dispatching. This supersedes the earlier "buffer bindings only" restriction: the `HalError` "GLES compute does not support texture bindings" no longer exists in the source (removed by T-G13b `0c77fa6`; storage images landed in T-G14 `d0ef874`). External texture bindings remain rejected. Threading-audit (group F): `dispatchWorkgroupsIndirect` is applied via `glBindBuffer(DISPATCH_INDIRECT_BUFFER)` + `glDispatchComputeIndirect(offset)` (an indirect-arg offset `> i32::MAX` returns `HalError`). | ☑ (P15.4 + threading-audit + T-G13b/T-G14: buffer, sampled-texture, sampler, and storage-image bindings + direct/indirect dispatch; ANGLE + crocus verified. Only external textures are still a Tier-2 `HalError`) |
| Compute dispatch (direct) | `glDispatchCompute(x, y, z)` + `glMemoryBarrier(ALL_BARRIER_BITS)` | ☑ (P15.4) |
| Compute dispatch (indirect) | `glBindBuffer(DISPATCH_INDIRECT_BUFFER)` + `glDispatchComputeIndirect(offset)` + `glMemoryBarrier(ALL_BARRIER_BITS)`. `HalComputeDispatch::Indirect` exists in core and the GLES arm executes it (`gles/queue.rs` ~`:436`). The direct path skips zero-workgroup dispatches CPU-side but deliberately does **not** pre-read indirect args. | ☑ (threading-audit group F, `edb379d`; an indirect offset `> i32::MAX` is a catalogued Tier-2 `HalError`. **Corrects the earlier "✗ Deferred — no indirect variant in core"**, which contradicted the Compute-shader row above) |
| Vertex shader / fragment shader | Tint GLSL ES 3.10 output (per-stage emission); shared `generate_glsl` accepts Vertex / Fragment / Compute; wrapped as `HalShaderSource::GlslStages { vertex, fragment }` for render | ☑ (P15.5; ANGLE verified) |
| Vertex formats (F-044) | the full `GPUVertexFormat` set maps to GL vertex-attribute metadata: integer `uint*`/`sint*` formats via `glVertexAttribIPointer`, `unorm*`/`snorm*` via `glVertexAttribPointer` + normalized, `float16*`/`float32*` via plain float, `unorm10_10_10_2` via `GL_UNSIGNED_INT_2_10_10_10_REV`. **`unorm8x4-bgra` returns `HalError`** — GLES 3.1 has no clean BGRA vertex-attribute swizzle. | ◐ (F-044: full format set mapped; `unorm8x4-bgra` is a catalogued Tier-2 `HalError`. ANGLE verification deferred) |
| Render pipeline state | Cached: GL program + `Vec<HalVertexBufferLayout>` + primitive topology + bindings + `Option<UniformLocation>` for Tint's `tint_immediates[0]` first-instance immediate; supports vertex+fragment color pipelines and vertex-only depth-stencil pipelines where GLES program linking accepts a fragment-less program. F-035 applies the color target `writeMask` (`glColorMask`) and `blend` (`glEnable(GL_BLEND)` + `glBlendFuncSeparate` + `glBlendEquationSeparate`) plus the render-pass blend constant (`glBlendColor`). With MRT (T-G8) these are **global, not per-target**: GLES 3.1 has no `EXT_draw_buffers_indexed`, so a descriptor whose color targets carry **divergent** write masks or blend state is rejected at pipeline creation (`GLES 3.1 cannot apply per-target write masks` / `... per-target blend state`), while uniform state is applied across all targets. **dual-source blend factors (`Src1` / `OneMinusSrc1` / `Src1Alpha` / `OneMinusSrc1Alpha`) return `HalError`** (`dual-source-blending` is not advertised on the GLES 3.1 baseline). Threading-audit (group A/D/E): `primitive.cullMode` + `primitive.frontFace` are applied per draw (`glEnable/Disable(GL_CULL_FACE)` + `glCullFace` + `glFrontFace`); `multisample.alphaToCoverageEnabled` maps to `glEnable/Disable(GL_SAMPLE_ALPHA_TO_COVERAGE)` and `multisample.mask` to `glSampleMaski` (entry point loaded dynamically on both the EGL and WGL paths; a non-default mask when `glSampleMaski` is unavailable returns `HalError`). `sample_count` is validated against `GL_MAX_SAMPLES` rather than rejected outright. `primitive.unclippedDepth = true` still returns `HalError` (no depth-clamp on the GLES 3.1 baseline; core also rejects it pending a `depth-clip-control` feature) | ◐ (P15.5 + F-031 + F-035 + threading-audit + T-G8: multiple color targets, writeMask + blend + blend constant, cullMode/frontFace, alpha-to-coverage and sample mask all applied; depth-only is best-effort Tier-2. Catalogued Tier-2 `HalError`: dual-source blend factors, **divergent per-target** write mask / blend state, unclipped depth, and a non-default sample mask without `glSampleMaski`) |
| Render pass (color + depth/stencil) | Transient FBO + `glFramebufferTexture2D(COLOR_ATTACHMENT0 / DEPTH_ATTACHMENT / STENCIL_ATTACHMENT / DEPTH_STENCIL_ATTACHMENT)` + `glDrawBuffers` + `glViewport` + clear (`glClearColor`/`glClearDepthf`/`glClearStencil`/`glClear`); `RenderPassCleanup` Drop guard ensures VAO + FBO + program-state + memory-barrier cleanup runs regardless of inner error. **Command-stream execution (`88cfe58`):** a WebGPU render pass is now one native GL pass — a single FBO and a single VAO are created per pass and `replay_gles_render_stream` walks the recorded commands — where the previous per-draw path built and tore down an FBO/VAO for every draw and forced intermediate `Load`/`Store`. Consequences: `storeOp: Discard` now actually discards (`glInvalidateFramebuffer` in `discard_render_pass_attachments`; previously stores were forced), a clear-only pass with a resolve target now resolves (the old path returned before the resolve step), and a draw arriving with no pipeline is now an error (`render draw has no GLES pipeline`) instead of silently no-op'ing the pass. F-038 applies the pipeline's depth-stencil **stencil** state per draw: `glEnable(GL_STENCIL_TEST)` + front/back `glStencilFuncSeparate`(compare, **dynamic reference** from `pass.stencil_reference`, readMask) / `glStencilOpSeparate`(fail, depthFail, pass) / `glStencilMaskSeparate`(writeMask); `RenderPassCleanup` also disables `GL_STENCIL_TEST`. The GL `ref` parameter is `GLint`, so a stencil reference `> i32::MAX` returns `HalError` (catalogued Tier-2 limit, no core relaxation). **MRT is implemented (T-G8), superseding the earlier F-040 slice-1 deferral:** `create_render_fbo` resolves every color slot up front, attaches each to `COLOR_ATTACHMENT0 + index`, and issues `glDrawBuffers` over the list (clamped by `GL_MAX_DRAW_BUFFERS`); trailing empty slots are truncated off the list. **MSAA + resolve is implemented, superseding the F-040 slice-2 deferral:** multisample color attachments use `TEXTURE_2D_MULTISAMPLE` and `resolve_render_pass` resolves via `glBlitFramebuffer`. Resolve-side `HalError`s: a single-sample resolve source, a multisample resolve target, or a resolve target that is not a single-sample 2D texture. Threading-audit (group B): `setViewport` (`glViewport` + `glDepthRangef`) and `setScissorRect` (`glEnable(GL_SCISSOR_TEST)` + `glScissor`, `glDisable` when unset) are applied; an unset viewport/scissor keeps the full-attachment default. Threading-audit (group C): a `depthReadOnly` / `stencilReadOnly` aspect maps to `Load` + preserve (no clear), matching Vk/Metal. **Texture and sampler bindings now execute (T-G13b / T-G14), superseding the execution-gap group-A rejection:** each draw binds combined samplers (`bind_combined_samplers`, with a placeholder sampler where Tint asks for one), the texture-metadata UBO, and storage images (`bind_storage_textures`), each behind a Drop guard (`TextureUnitCleanup` / `StorageImageCleanup`) that unbinds afterwards. The `HalError` "GLES render pass does not support texture/sampler bindings" no longer exists in the source. **External** texture bindings are still rejected. **Layered color attachments (2026-07-08):** `create_render_fbo` now accepts `TEXTURE_2D_ARRAY` and `TEXTURE_3D` color targets, attaching the selected `array_layer` (2D-array) or `depth_slice` (3D) via `glFramebufferTextureLayer` (the color analogue of the layered depth-stencil work e185afc, using the `HalRenderColorTarget.array_layer`/`depth_slice` fields); `TEXTURE_CUBE_MAP`/cube-array color targets stay `HalError` (catalogued). Fixed the `rendering,3d_texture_slices` cluster and the non-2D-attachment copyTextureToTexture setup passes. | ◐ (P15.5 + F-031 + F-038 + T-G8 + T-G13b/T-G14 + threading-audit + command stream `88cfe58`: multiple color attachments incl. sparse slots, 2D / 2D-multisample / 2D-array-layer / 3D-slice color + depth/stencil attachment, depth-only, MSAA resolve, occlusion queries, `storeOp: Discard`, texture/sampler/storage-image bindings, dynamic stencil reference, viewport and scissor, read-only depth/stencil preserve. Catalogued Tier-2 `HalError`: cube / cube-array color attachments, external texture bindings, framebuffer fetch, a stencil reference `> i32::MAX`) |
| Sparse color attachments (F-054) | WebGPU allows empty color slots (null view / `Undefined` target) interleaved with real ones, with fragment `@location(N)` targeting slot N. **Now supported at any slot (T-G8), superseding the earlier slot-0-only restriction:** `create_render_fbo` attaches each `Some` slot to `COLOR_ATTACHMENT0 + index` and pushes `GL_NONE` for each sparse slot into the `glDrawBuffers` list, so a hole at slot 0 — or anywhere — is representable and the fragment `@location(N)` output routes correctly (a sparse slot's output is discarded). Trailing empty slots are truncated off the list. Matches core + Metal + Vulkan (Tier-1). | ☑ (F-054 restriction lifted by T-G8; pinned by `validate_render_pipeline_descriptor_accepts_multiple_and_sparse_color_targets`) |
| `draw` / `drawIndexed` | `glDrawArrays` / `glDrawArraysInstanced`; `glDrawElements` / `glDrawElementsInstanced` for indexed draws. `baseVertex != 0` returns `HalError` on the GLES 3.1 baseline because `glDrawElementsBaseVertex*` is not guaranteed. | ◐ (F-034: direct + indexed direct execute when `baseVertex == 0`; nonzero `baseVertex` is catalogued Tier-2 `HalError`) |
| `drawIndirect` / `drawIndexedIndirect` | `glDrawArraysIndirect` / `glDrawElementsIndirect`. Indexed indirect requires `setIndexBuffer` offset 0 because GLES has no separate element-buffer binding offset for indirect draws; nonzero index-buffer offset returns `HalError`. | ◐ (F-034: indirect variants execute; indexed-indirect nonzero index-buffer offset is catalogued Tier-2 `HalError`) |
| `first_instance` direct | Tint `Options::first_instance_offset = 0` (vertex stages) injects `layout(location = 0) uniform uint tint_immediates[1]`; the HAL sets `tint_immediates[0]` via `glUniform1ui` per draw (covers `@builtin(instance_index)`), **and** offsets every `Instance`-stepped vertex buffer's attribute pointers by `first_instance * array_stride` (Dawn GL parity, `CommandBufferGL.cpp:259-261`; no dirty tracking needed — the GLES path re-specifies a fresh VAO per draw). Replaced the naga-era `naga_vs_first_instance` uniform, which Tint never emits (was a silent no-op after the Tint migration). | ☑ (Tint-integration refactor R6 + Phase Review M2; contract pinned by generated-GLSL unit tests; real-ANGLE re-verification pending) |
| `first_instance` indirect | ✗ Unsupported — feature not advertised (`supports_indirect_first_instance()` = false); GLES 3.1 `Draw*IndirectCommand` has no `baseInstance` field, so it genuinely cannot be honored — indirect draws use `first_instance = 0` | locked ✗ |
| GLSL binding numbers (all classes) | `glsl_binding_info` (yawgpu-core) assigns **each resource class its own dense sequence sorted by `(group, binding)`** and forces `dst_group = 0`, so WGSL groups collapse into GL's flat per-class binding spaces without collisions. UBO / SSBO / storage-image remaps are handed to the HAL too, because those GL calls consume the binding number directly; sampled textures and samplers bind through Tint's linked uniform names but still participate in the remap so combined-sampler lowering stays deterministic across groups. This is the **linear binding remap** that multi-bind-group support was waiting on. Without an explicit remap, Tint's `GenerateBindings` renumbers sequentially in declaration order (a `@binding(3)` buffer became `binding = 1`), desyncing GLSL from the HAL. The naga-era `_block_N` name-parse remap was deleted. *(Earlier revisions of this row described an "identity `BindingRemap`" via `tint_bindings_for_glsl`; both the identity property and that function name are obsolete.)* | ☑ (R6 + multi-group remap; pinned by generated-GLSL unit tests in yawgpu-tint/-core) |
| `textureNumLevels` / `textureNumSamples` (`texture_builtins_from_uniform`) | WIRED (T-G17 exposed the polyfill UBO binding through the shim; c06e516 populates each slot from Tint's `ubo_contents` layout, mapping post-remap binding → WGSL group/binding, filled by the queue with mip-level / sample count). **Cross-stage slot assignment fixed (2026-07-08):** the shim generates GLSL per stage, so each stage's `ubo_contents` was packed from offset 0 independently — vertex and fragment then collided at the same UBO offset for *different* textures (core `merge_texture_metadata_slots` raised `unexpected internal error`, 64 CTS fails in `capability_checks,limits,maxSampledTexturesPerShaderStage`). Fix: the shim sets `ubo_contents[i].offset = resolved_binding.binding` (both the remapped and empty-remaps paths), making the offset a deterministic function of the pipeline-stable resolved binding — vertex and fragment independently compute disjoint offsets for different textures and identical offsets for a shared one. Mirrors Dawn's per-pipeline `EmulatedTextureBuiltinRegistrar` (keyed on FlatBindingIndex; `opengl/PipelineGL.cpp:222-246`) without threading a shared registrar across the two per-stage shim calls. | ☑ (T-G17 + c06e516 + cross-stage offset fix; pinned by yawgpu-tint generated-GLSL test + real-EGL cross-stage HAL test) |
| Context backend (Windows) | Default: EGL (`libEGL.dll` ⇒ ANGLE platform-display cascade through Vulkan → D3D11). Opt-in fallback: WGL (`opengl32.dll` + `WGL_EXT_create_context_es2_profile`) selected via `YAWGPU_GLES_BACKEND=wgl`, or programmatically through `YaWGPUGlesContextBackend` (`YAWGPU_STYPE_GLES_CONTEXT_BACKEND`) chained onto `WGPUInstanceDescriptor.nextInChain`. Resolution is chain `EGL`/`WGL` value > env var > default EGL; `DEFAULT` defers to the env var, WGL on non-Windows falls back to EGL, and the chain entry is ignored when the resolved instance backend is not GLES. Both routes converge on the same `glow::Context` API below the make-current seam; `GlesInstanceInner` / `GlesAdapter` / `GlesDeviceInner` / `GlesSurfaceInner` are static enums (`Egl(...)` / `Wgl(...)`) per CLAUDE.md "no `dyn Trait`". WGL surface (HWND): `ChoosePixelFormat`/`SetPixelFormat` with the same descriptor as the helper HWND (shared HGLRC), `wglMakeCurrent(surface.hdc, hglrc)` + glow blit + `SwapBuffers(hdc)` for present; `RestoreCurrent` Drop guard re-binds the helper HDC. | ☑ (P15.6 EGL + post-COMPLETE WGL context + post-COMPLETE WGL surface slices + programmatic override; WGL verified on `OpenGL ES 3.2 NVIDIA 595.95` — **15/15 e2e green, re-run 2026-08-08 after the command-stream move (`88cfe58`)** — + `examples/triangle` runs 60 frames clean) |
| Surface (Android) | `eglCreateWindowSurface(ANativeWindow*)` via `GlesInstance::create_surface_from_android_native_window`. Reuses the existing `choose_config` (RGBA8 + GLES3 + PBUFFER_BIT). | ☑ (P15.6; code path implemented; manual visual verification via Android-side example) |
| Surface (Windows ANGLE) | `eglCreateWindowSurface(HWND)` via `GlesInstance::create_surface_from_windows_hwnd`. ANGLE accepts the pbuffer-capable config for window surfaces too. | ☑ (P15.6; manual visual verification via `examples/triangle`) |
| Present | Back-buffer (`GlesTexture` allocated at `configure()` with `RENDER_ATTACHMENT \| COPY_SRC`) blitted via transient read-FBO + `glBlitFramebuffer` to default FBO, then `eglSwapBuffers`. `RestoreCurrent` Drop guard re-binds the pbuffer after swap (even on error). | ☑ (P15.6) |
| Occlusion queries | `glBeginQuery(ANY_SAMPLES_PASSED)` / `glEndQuery`, with `GlesRenderQueryState` tracking the active query across the pass and `finish()` writing `QUERY_RESULT` back into the query set. `HalError` on: begin with a query already active, no query set, index out of range, end with no active query, and a pass that ends while a query is still open. | ☑ (`88cfe58` — before the command-stream move the fields reached the HAL but no consumer read them, so occlusion queries were silently ignored) |
| Timestamp queries | GLES has `EXT_disjoint_timer_query`; not advertised (`GlesAdapter::supports_timestamp_query()` is hard-`false`) | ✗ (Tier 2, deferred) |
| Bundle execution | Core does **not** flatten bundles: it lowers each into a `HalRenderPassCommand::ExecuteRenderBundle(HalRenderBundle)` wrapper that preserves the execution boundary. The GLES arm recurses into `replay_gles_render_stream` for the bundle's commands, then calls `invalidate_gles_render_bundle_state` (clears pipeline, bind buffers/textures/samplers/external textures, vertex + index buffers) — the WebGPU "a bundle inherits nothing and invalidates state afterwards" rule, matching Tier 1. Capability is exactly that of the GLES render path. | ☑ (`88cfe58`; unit-tested by `render_bundle_invalidation_clears_pipeline_bind_and_buffer_state`. Resolves the former **?**) |
| External texture bindings | No GLES mapping; `reject_external_texture_bindings` returns `HalError` ("GLES does not support external texture bindings") in both the compute and render-draw paths | ✗ (catalogued Tier-2) |
| Render immediates | Not delivered. `SetImmediates` is recorded into the render stream and bounds-checked, but the draw path never reads `state.immediate_data`; only `first_instance` reaches the shader, via the `tint_immediates[0]` uniform. Unchanged by the command-stream move — the per-draw path behaved identically. | ✗ (catalogued Tier-2 gap; silent rather than a `HalError`, so worth a rejection if immediates reach GLES in practice) |
| Framebuffer fetch (`@color(N)`) | `HalError` ("GLES render-pass framebuffer fetch is unsupported") when the stream carries any `framebuffer_fetch_color_slots`. Unreachable through core today because the slots are only populated on the multi-subpass path, which GLES never advertises. Note GLES *does* have `EXT_shader_framebuffer_fetch` / `EXT_shader_pixel_local_storage` — mappable in principle, unimplemented. | ✗ (Tier-2; `88cfe58` turned a silently-ignored field into an explicit rejection) |
| `tiled` feature | Not advertised on GLES, and rejected at three layers: core never inserts the feature (`tiled_features_supported` matches only Metal/Vulkan, so `tiled_capabilities()` is all-zero), `HalDevice::create_subpass_render_pipeline`'s GLES arm returns `BackendUnavailable` unconditionally, and `HalCopy::SubpassRenderPass` submission returns `QueueSubmissionFailed` ("GLES subpass render pass submission is unsupported"). The HAL arm consumes none of its arguments — which is why a `gles`-only build broke until `6f991e7` re-keyed the unused-parameter suppression on `not(any(metal, vulkan))`. | locked ✗ |
| `shader-passthrough` feature | Not advertised on GLES; yawgpu.h passthrough APIs reject GLES device | locked ✗ |

## Open questions (resolve per slice, record divergences)

- ~~**naga `glsl-out` coverage smoke**~~ — OBSOLETE. naga is gone from the
  build; the frontend is Tint's `glsl::writer`, and shader coverage is now
  tracked by the CTS sweep (`tracking/cts-gles-sweep-0705.md`) plus the
  generated-GLSL unit tests in `yawgpu-tint` / `yawgpu-core`.
- ~~**Adapter limit mapping** (P15.1)~~ — RESOLVED (`61dd95b` + `6b105c4`,
  2026-07-06). `GlesAdapterCaps { limits, color_render_caps,
  supports_float32_filterable }` is probed once at adapter-enumeration time
  via a throwaway ES 3.1 context + 1×1 pbuffer, then torn down.
  `query_gles_limits` derives real limits by `min`-ing the per-stage `glGet`
  values (sampled textures, storage buffers, storage textures, uniform
  buffers, `max_color_attachments` = min(`MAX_COLOR_ATTACHMENTS`,
  `MAX_DRAW_BUFFERS`), compute workgroup counts), falling back to
  `HalLimits::DEFAULT` per query on failure. `max_bindings_per_bind_group`
  is the min over all GL binding-point spaces because Tint emits
  `layout(binding = N)` using WGSL binding numbers directly. The stated goal
  is met: the adapter reports real caps and `request_device` declines
  unsatisfiable asks — pinned by a real-EGL test asserting the queried
  binding limit is strictly below the WebGPU default.
- **ANGLE binary distribution** (P15.0): document in `README.md` that
  the user supplies `libEGL.dll` / `libGLESv2.dll`; do not bundle.
- **Buffer mapping fence model** (P15.2): how to expose "map after
  submit" without true CB semantics in HAL. Likely: queue submission
  inserts a fence (`glFenceSync`); map waits on the fence.
- **Storage-texture format gating timing** (P15.3 / P15.4): does core
  validation know enough to reject an unsupported format up front, or
  does HAL surface the rejection as a device error at use time?
  Tier-2 best-effort default is the latter; reconsider per case.
- **Resource hazard barriers** (P15.5 / `e2e_copy`): which
  `glMemoryBarrier` masks to issue between a HalCopy and a subsequent
  bind. Conservative default: `GL_ALL_BARRIER_BITS` after every copy
  the user submits; tighten if profiling demands.

## CTS-confirmed Tier-2 catalogue (2026-07-06, crocus/Mesa sweep)

Gaps surfaced by the api,validation CTS sweep and their disposition.
Feature-advertisement gaps are resolved at the source (the GLES adapter
no longer advertises them, decision 2a); the rest are HalError-rejected
per case.

- **norm16 / snorm / bgra8unorm-srgb color targets, read-write storage
  of the tier2 set** — RESOLVED by not advertising `TextureFormatsTier1`
  / `TextureFormatsTier2` / `Bgra8UnormStorage` on the GLES adapter
  (slice 3b). CTS stops enabling these features, so the cases skip.
- **`unorm8x4-bgra` vertex format** — IMPLEMENTED (slice 5). Accepted
  and mapped; correct B<->R fetch via `glVertexAttribPointer(size=
  GL_BGRA)` when `GL_EXT/ARB_vertex_array_bgra` is present. crocus/Mesa
  does NOT expose that extension, so on this host the format is accepted
  (validation passes) but rendered R/B are swapped — an **execution-only
  divergence** (shader,execution / api,operation), not a validation gap.
- **cube-array textures** (`texture_cube_array`, `samplerCubeArray`) —
  permanent Tier-2 gap: GLES 3.1 has no cube-array. Bindings using it
  return HalError / fail GLSL compile. Note: failures are subcase-
  specific (mixed with passing subcases in the same CTS case), so they
  are NOT expressible in the case-granular expectations file — track
  here only.
- **stencil-aspect texture-to-buffer readback** — GLES cannot
  `glReadPixels` the stencil aspect; returns HalError. A compute-image
  path could lift it later (see the depth compute-fallback, slice 4).
- ~~**>1 bind group** (`@group(1..3)`)~~ — **RESOLVED (2026-08-08 review).**
  The deferred linear binding remap has landed: `glsl_binding_info`
  (yawgpu-core) assigns each resource class a dense sequence sorted by
  `(group, binding)` with `dst_group = 0`, and the GLES HAL resolves each
  bound resource through `flat_binding(binding_remaps, group, binding,
  class)`. `SetBindGroup` tracks per-group slots (state is retained/cleared
  by `binding.group != index`). No `@group`-based rejection remains in the
  GLES source. Was never a hardware gap.
- **maxBindingsPerBindGroup edge** — **still open; mechanism confirmed by
  code read (2026-08-08 review).** `query_gles_limits` computes
  `max_bindings_per_bind_group` as the plain `min` of
  `MAX_UNIFORM_BUFFER_BINDINGS`, `MAX_SHADER_STORAGE_BUFFER_BINDINGS`,
  `MAX_COMBINED_TEXTURE_IMAGE_UNITS`, and `MAX_IMAGE_UNITS`, with **no
  reservation subtracted**. But `glsl_binding_info`'s doc contract states
  that Tint's GLES texture-metadata UBO is placed by the shim "at the next
  free uniform-buffer binding after `bindings.uniform`". So a shader that
  fills the uniform space to the reported limit *and* needs the metadata
  UBO (i.e. uses `textureNumLevels` / `textureNumSamples`) pushes that
  internal block one past `MAX_UNIFORM_BUFFER_BINDINGS` — matching the
  observed compute/fragment compile fails exactly at the limit. Candidate
  fix: subtract one uniform-buffer binding when reporting the limit, or
  place the metadata UBO at a reserved low binding. Not yet fixed.

- **Mesa/crocus driver crash: `textureSize()` on a stencil-mode packed
  depth/stencil texture** — `textureDimensions` on a `stencil-only`
  aspect view of depth24plus-stencil8 / depth32float-stencil8 segfaults
  (signal 11) inside the driver. `texelFetch` on the same stencil-mode
  texture works (T-G18 stencil readback tests pass), so it is
  textureSize-specific and yawgpu cannot distinguish the builtin at
  bind time — a bind-time guard was tried and reverted because it broke
  the working texelFetch path. Suspected Mesa driver defect (a
  hand-written GL repro would upgrade suspected->confirmed, per the
  F-126 / zero-dim precedent). 2 CTS cases; documented, not code-guarded.

- **Flexible texture views via `glTextureView`** (2026-07-06, DONE) —
  cube / cube-array / array-layer subrange / stencil-only / color-format
  reinterpret views. Mirrors Dawn's opengl backend (`TextureGL.cpp`
  `TargetForTextureViewDimension` / `RequiresCreatingNewTextureView` /
  `CreateView`): the WebGPU texture keeps its base GL storage
  (`TEXTURE_2D_ARRAY` for a 2d/6-layer texture) and, when a binding needs
  a different view target/subrange/aspect/format, the bind path creates a
  transient GL texture object aliasing the base storage with
  `glTextureView(view, target, src, internalFormat, minLevel, numLevels,
  minLayer, numLayers)` and binds that. No `textureBindingViewDimension`
  hint is required — this matches the CTS oracle (Dawn uses flexible views,
  so CTS never sets the hint). The capability is detected at adapter time
  (`supports_texture_view` from ES 3.2 / `GL_OES_texture_view` /
  `GL_EXT_texture_view`; `supports_cube_map_array` for cube-array); the
  proc is loaded manually (EGL + WGL paths). glTextureView requires an
  **immutable-format** source, which yawgpu already satisfies (all GLES
  textures are `glTexStorage*`-allocated). Verified on Mesa crocus (Intel
  Haswell, reports ES 3.2): `submit_compute_pass_samples_cube_view_from_2d_array_texture_view`
  samples all 6 faces correctly; array-layer-subrange view verified too.
  **Fallback:** if `glTextureView` is unavailable, the previous
  `HalError` rejection for these view shapes is retained (true ES-3.1
  Tier-2 gap). Supersedes the earlier "cube is a Tier-2 gap" catalogue
  entry and the reverted `textureBindingViewDimension` approach
  (`webgpu-native-cts/transcripts/cube-wip-reverted.patch`).

- **Raw (non-comparison) depth-texture reads** — RESOLVED (P2, 2026-07-08,
  shim-side, no Tint edit). Tint's GLSL printer appends "Shadow" to any
  `core::type::DepthTexture` sampler
  (`third_party/dawn/src/tint/lang/glsl/writer/printer/printer.cc:993`), so a
  `texture_depth_*` read by a **non-comparison** builtin (`textureSample` /
  `textureSampleLevel` / `textureGather` / `textureLoad`) was emitted as a
  `sampler2DShadow` shadow-COMPARE against a dummy ref `0.0` (returns 0/1)
  instead of a raw depth read. Fix: a shim-level Core-IR transform
  (`DepthRawReadTransform` in `yawgpu-tint/shim/tint_shim.cpp`, run on the
  lowered IR right before `glsl::writer::Generate`) rewrites each depth var
  used ONLY by non-comparison builtins to `texture_*<f32>`
  (`ty.sampled_texture(dim, ty.f32())`) — once the IR type is a
  `SampledTexture`, TexturePolyfill's `is_depth` refz injection goes dormant
  and the printer emits `sampler2D`; sample/level/load results are retyped
  `f32`→`vec4<f32>` + `.x` swizzle, gather is left unchanged (already
  `vec4<f32>`). Uses Tint's own machinery as the template
  (`texture_polyfill.cc:345,661-676` + `bgra8unorm_polyfill.cc`); no
  `third_party/dawn` edit and shim-only rebuild (no Tint recompile, no
  host-hang risk). Verified on crocus: **textureSample 885, textureSampleLevel
  2,610, textureGather 3,105 = ~6,600 FAIL→PASS** (incl. cube via the
  glTextureView path), textureLoad depth16unorm/depth32float(-stencil8) pass;
  comparison clusters unchanged (`textureSampleCompare` 16,560 /
  `textureSampleCompareLevel` 49,680 / `textureGatherCompare` 46,800, all
  0-fail).
  - **RESIDUAL (still catalogued Tier-2):** (a) `textureLoad` on
    **depth24plus / depth24plus-stencil8** — 48 fails "expected bits …, got …":
    depth24plus has implementation-defined precision (WebGPU allows ≥24 bits),
    and Mesa/crocus's internal storage bit-count differs from the CTS's
    expected bits; depth16unorm / depth32float / depth32float-stencil8 all
    pass, so this is a format-precision boundary, not the shadow modelling.
    (b) depth handles reached through a **user-function parameter** (a
    `UserCall` before DirectVariableAccess inlines the handle) — the
    eligibility scan marks these ineligible and leaves them as `sampler2DShadow`
    (conservative). (c) Mixed comparison + non-comparison use of ONE depth
    texture — skipped by construction (a comparison use makes the var
    ineligible). (d) Multisampled depth (`DepthMultisampledTexture`) — out of
    scope for this slice.

- **Storage images: vertex-stage + non-required formats (GLES limits)** —
  Tier-2 hardware/spec gaps (2026-07-06, catalogued). (a) GLES 3.1 does not
  guarantee image load/store in the **vertex** stage
  (`GL_MAX_VERTEX_IMAGE_UNIFORMS` is commonly 0, and is 0 on crocus), so a
  render pipeline whose vertex shader does `imageLoad`/`imageStore` cannot
  link — the dominant storage-texture CTS failure (e.g.
  textureLoad:storage_textures_2d_array 768/1056 fails are stage="v"). (b)
  `rg32{uint,sint,float}` are **not** in the GLES 3.1 required
  image-format list, so storage load/store on them is unsupportable
  (~432 CTS fails). Both are real GLES limits, not yawgpu bugs; the ideal
  is a clean HAL rejection rather than a surfaced pipeline-link error, but
  either way the CTS case cannot pass on this hardware. (c) 1D storage
  (`texture_storage_1d`, `HalTextureViewDimension::D1`) is rejected —
  GLES has no `image1D`; correct handling is height-1 2D emulation (a
  separate slice, tied to the general no-1D-textures gap).
