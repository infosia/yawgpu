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
  `cargo test --features gles -- --ignored` manually on Windows ANGLE
  and logs results in `tracking/phase-15.md`**.
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
  "Context backend (Windows)" matrix row and `tracking/phase-15.md`
  → "Post-COMPLETE — WGL fallback". A later post-COMPLETE addition
  exposed `YaWGPUGlesContextBackend` /
  `YAWGPU_STYPE_GLES_CONTEXT_BACKEND` so applications can force
  EGL or WGL from the instance descriptor; a non-default chain value
  wins over `YAWGPU_GLES_BACKEND`.

### Minimum GLES version

The GLES HAL targets **OpenGL ES 3.1** as its minimum. This is
declared via `EGL_CONTEXT_CLIENT_VERSION = 3` +
`EGL_CONTEXT_MINOR_VERSION = 1` at `eglCreateContext`, and
`GlesAdapter::new` rejects contexts whose `glGetString(GL_VERSION)`
parses to less than 3.1 (returning `None` so the adapter is silently
dropped from `enumerate_adapters`). GLES 3.2 features (e.g.
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
- **Shader / Pipeline**: WGSL → GLSL ES 3.10 via naga's `glsl-out`;
  compiled into a `GLuint` program. Bind-group layout + a derived
  linear-binding remap table are stored on the pipeline.
- **Compute pipeline**: program object + workgroup size.
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
(backend impl) → user runs `--ignored` on Windows ANGLE, reports,
logged in `tracking/phase-15.md`.

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
| Sampler / texture combining | naga `glsl-out` emits combined texture-sampler uniforms; pipeline carries a `SamplerBindMap`-style table; rebind on bind-group change. | WebGPU's separate sampler is the principal semantic gap. wgpu's approach is proven. |
| Storage textures | GLES 3.1 `glBindImageTexture` + `image2D` shader qualifiers. Format coverage validated against the GLES image-format table; unsupported formats → `HalError::FormatUnsupported`-class. | GLES 3.1 image format coverage is narrow (R32F, Rgba8, Rgba32F, …). |
| `first_instance` | naga injects a `naga_vs_first_instance` uniform; HAL sets it per draw. `INDIRECT_FIRST_INSTANCE` feature is **not** advertised on GLES. | GLES lacks `gl_BaseInstance`; matches wgpu's GLES & GL 4.1 path. |
| Compute / SSBO | Native GLES 3.1 compute + SSBO + indirect dispatch path. | Full WebGPU compute surface available. |
| Memory barriers | `glMemoryBarrier(...)` issued by HAL between hazard-prone ops based on the recorded HalCopy / pass structure. | GLES has no fine-grained barrier API; coarse-grained mask is sufficient for the e2e set. |
| WGSL → GLSL version | Target **GLES 3.10**. Use higher (3.20) only when an emitted feature demands it and the driver reports it. | Matches the 3.1 minimum; broadest device coverage. |
| ANGLE platform selection | Leave ANGLE to its default backend choice (typically D3D11 on Windows). Do not expose `EGL_ANGLE_platform_angle` controls from yawgpu. For environments where the locally available ANGLE caps at ES 3.0 (Chromium / CEF builds), the **WGL fallback** (`YAWGPU_GLES_BACKEND=wgl`, post-COMPLETE addition — see "Context backend (Windows)" row) bypasses ANGLE entirely and uses the host GL driver via `WGL_EXT_create_context_es2_profile`. | Keeps the surface simple; users wanting a specific ANGLE backend can env-var their own ANGLE build, and users without a workable ANGLE can fall back to WGL. |
| Error mapping | `HalError::BufferOperationFailed { backend: "gles", message }` for buffer-class GL failures plus the existing `backend`-only variants (`BackendUnavailable` / `DeviceCreationFailed` / `QueueSubmissionFailed` / `ShaderCompilationFailed`) and message-carrying surface variants (`AcquireFailed` / `PresentFailed` / `SwapchainCreationFailed`). | Mirrors Vk / Metal arms. *(Corrected from the earlier `BackendOperationFailed` wording — that variant does not exist on `HalError`; P15.1 used the real enum correctly.)* |
| Adapter selection from `wgpuCreateInstance` | yawgpu.h vendor extension `YaWGPUInstanceBackendSelect.backend = YAWGPU_INSTANCE_BACKEND_GLES = 3` pins the primary HAL to GLES at instance creation, mirroring the Metal/Vulkan pattern. The standard webgpu.h `WGPURequestAdapterOptions.backendType` field is accepted but treated identically across all backends (the primary HAL chosen at instance creation is what enumerates) — same behavior Vulkan/Metal exhibit. | One consistent selection path across all backends. |

## WebGPU × GLES mapping matrix (initial)

Status: ☑ Supported · ◐ Partial / restricted · ✗ Unsupported (HalError)

Entries are filled in / refined as P15.x slices land. Anything left as
**?** must be resolved before that slice's review.

| Area | GLES 3.1 mapping | Initial status |
|---|---|---|
| Adapter / device creation | EGL display + shared context | ☑ (P15.1; ANGLE on Windows verified) |
| Buffer create / map / unmap | `glBufferData(NULL, size, DYNAMIC_DRAW)` + `glBufferSubData` (write) + `glMapBufferRange(MAP_READ_BIT)` (read). HostBuffer path in core (`mapped_ptr` returns `None`); persistent map deferred. | ☑ (P15.2; ANGLE round-trip verified) |
| Buffer-to-buffer copy | `glCopyBufferSubData` via `GL_COPY_READ_BUFFER` / `GL_COPY_WRITE_BUFFER` | ☑ (P15.2; ANGLE round-trip verified, full + partial offsets) |
| Buffer clear | `glBufferSubData` zero-fill chunks via `GL_COPY_WRITE_BUFFER` | ☑ (F-023 follow-up; shares the existing GLES buffer write path) |
| `mappedAtCreation` | Allocate + map immediately; flush on unmap | ☑ (P15.2; transparent via HostBuffer path) |
| Texture: 1D | `GL_TEXTURE_2D` with height=1 (no native 1D in GLES) | ◐ Allocation supported (F-026 follow-up); copy execution remains restricted by the 2D-only GLES copy path where unsupported shapes return `HalError` |
| Texture: 2D | `GL_TEXTURE_2D` + `glTexStorage2D` (non-multisample; uncompressed color formats known by core map to GLES internal/external/type triplets, including integer, signed-normalized, sRGB, 16/32-bit, and packed formats) | ☑ (P15.3; ANGLE verified; format coverage pass expands uncompressed color mappings) |
| Texture: 2D array | `GL_TEXTURE_2D_ARRAY` + `glTexStorage3D` | ◐ Allocation supported with layers and mips (F-026 follow-up); B2T / T2B / T2T remain unsupported for array copies and return `HalError` |
| Texture: 3D | `GL_TEXTURE_3D` + `glTexStorage3D` | ◐ Allocation supported with depth and mips (F-026 follow-up); B2T / T2B / T2T remain unsupported for 3D copies and return `HalError` |
| Texture: cube | `GL_TEXTURE_CUBE_MAP` + `glTexStorage2D` | ✗ Deferred |
| Texture views | Subrange metadata resolved by core; HAL receives `HalTexture` + mip/origin in copy descriptors | ☑ (P15.3; degenerate — no HAL-level view object) |
| Storage textures (read/write) | `glBindImageTexture`; restricted format set | ? (P15.4) |
| B2T / T2B / T2T copies | `glTexSubImage2D` (PBO unpack + `UNPACK_ROW_LENGTH`) / transient FBO + `glReadPixels` (PBO pack + `PACK_ROW_LENGTH` + `read_buffer(COLOR_ATTACHMENT0)`) / `glCopyImageSubData` (GLES 3.2 or `GL_EXT_copy_image`) | ◐ (P15.3; ANGLE verified for 2D only; array / 3D copy execution remains `HalError`) |
| Sampler creation | `glGenSamplers` + `glSamplerParameteri/f` (filter / address / mipmap / compare / anisotropy via `GL_EXT_texture_filter_anisotropic`) | ☑ (P15.3; `ClampToBorder` not supported) |
| Compute shader | naga WGSL → GLSL ES 3.10 (`use_framebuffer_fetch=false`, `zero_initialize_workgroup_memory=true`) compiled via `glCreateShader(COMPUTE_SHADER)` + `glLinkProgram`; bind-group bindings honored via `bind_buffer_range(SHADER_STORAGE_BUFFER \| UNIFORM_BUFFER)` against the WGSL `@binding(N)` emitted as `layout(binding=N)`. Single bind group (`@group(0)`) only. | ☑ (P15.4; ANGLE verified) |
| Compute dispatch (direct) | `glDispatchCompute(x, y, z)` + `glMemoryBarrier(ALL_BARRIER_BITS)` | ☑ (P15.4) |
| Compute dispatch (indirect) | `glDispatchComputeIndirect` | ✗ Deferred — `HalComputePass` carries no indirect variant in core; gate first on a core HAL extension |
| Vertex shader / fragment shader | naga GLSL ES 3.10 output (per-stage emission); shared `generate_glsl` accepts Vertex / Fragment / Compute; wrapped as `HalShaderSource::GlslStages { vertex, fragment }` for render | ☑ (P15.5; ANGLE verified) |
| Render pipeline state | Cached: GL program + `Vec<HalVertexBufferLayout>` + primitive topology + bindings + `Option<UniformLocation>` for naga's `naga_vs_first_instance`; supports vertex+fragment color pipelines and vertex-only depth-stencil pipelines where GLES program linking accepts a fragment-less program | ◐ (P15.5 + F-031: at most one RGBA8Unorm/BGRA8Unorm color target; depth-only is best-effort Tier-2) |
| Render pass (color + depth/stencil) | Transient FBO + `glFramebufferTexture2D(COLOR_ATTACHMENT0 / DEPTH_ATTACHMENT / STENCIL_ATTACHMENT / DEPTH_STENCIL_ATTACHMENT)` + `glDrawBuffers` + `glViewport` + clear (`glClearColor`/`glClearDepthf`/`glClearStencil`/`glClear`); `RenderPassCleanup` Drop guard ensures VAO + FBO + program-state + memory-barrier cleanup runs regardless of inner error. Clear-only render passes (`pass.pipeline == None`) take the FBO+clear path and skip VAO+draw (Q1 fix) | ◐ (P15.5 + F-031: 2D color/depth/stencil FBO attachment supported, including depth-only; non-2D render attachments return `HalError`) |
| `draw` / `drawIndexed` | `glDrawArrays` / `glDrawArraysInstanced`; indexed deferred (needs `HalRenderPass`/`HalDraw` core extension) | ◐ (P15.5: drawArrays + drawArraysInstanced only) |
| `drawIndirect` / `drawIndexedIndirect` | `glDrawArraysIndirect` / `glDrawElementsIndirect` | ✗ Deferred — `HalRenderPass`/`HalDraw` carry no indirect variant in core |
| `first_instance` direct | naga injects `uniform uint naga_vs_first_instance`; HAL sets it via `glUniform1ui` per draw before `glDrawArraysInstanced` | ☑ (P15.5; uniform-injection path implemented, unexercised by e2e but code path active) |
| `first_instance` indirect | ✗ Unsupported — feature not advertised | locked ✗ |
| Context backend (Windows) | Default: EGL (`libEGL.dll` ⇒ ANGLE platform-display cascade through Vulkan → D3D11). Opt-in fallback: WGL (`opengl32.dll` + `WGL_EXT_create_context_es2_profile`) selected via `YAWGPU_GLES_BACKEND=wgl`, or programmatically through `YaWGPUGlesContextBackend` (`YAWGPU_STYPE_GLES_CONTEXT_BACKEND`) chained onto `WGPUInstanceDescriptor.nextInChain`. Resolution is chain `EGL`/`WGL` value > env var > default EGL; `DEFAULT` defers to the env var, WGL on non-Windows falls back to EGL, and the chain entry is ignored when the resolved instance backend is not GLES. Both routes converge on the same `glow::Context` API below the make-current seam; `GlesInstanceInner` / `GlesAdapter` / `GlesDeviceInner` / `GlesSurfaceInner` are static enums (`Egl(...)` / `Wgl(...)`) per CLAUDE.md "no `dyn Trait`". WGL surface (HWND): `ChoosePixelFormat`/`SetPixelFormat` with the same descriptor as the helper HWND (shared HGLRC), `wglMakeCurrent(surface.hdc, hglrc)` + glow blit + `SwapBuffers(hdc)` for present; `RestoreCurrent` Drop guard re-binds the helper HDC. | ☑ (P15.6 EGL + post-COMPLETE WGL context + post-COMPLETE WGL surface slices + programmatic override; WGL verified on `OpenGL ES 3.2 NVIDIA 595.95`, 12/12 e2e green + `examples/triangle` runs 60 frames clean) |
| Surface (Android) | `eglCreateWindowSurface(ANativeWindow*)` via `GlesInstance::create_surface_from_android_native_window`. Reuses the existing `choose_config` (RGBA8 + GLES3 + PBUFFER_BIT). | ☑ (P15.6; code path implemented; manual visual verification via Android-side example) |
| Surface (Windows ANGLE) | `eglCreateWindowSurface(HWND)` via `GlesInstance::create_surface_from_windows_hwnd`. ANGLE accepts the pbuffer-capable config for window surfaces too. | ☑ (P15.6; manual visual verification via `examples/triangle`) |
| Present | Back-buffer (`GlesTexture` allocated at `configure()` with `RENDER_ATTACHMENT \| COPY_SRC`) blitted via transient read-FBO + `glBlitFramebuffer` to default FBO, then `eglSwapBuffers`. `RestoreCurrent` Drop guard re-binds the pbuffer after swap (even on error). | ☑ (P15.6) |
| Timestamp / occlusion queries | GLES has `EXT_disjoint_timer_query`; not advertised initially | ✗ (Tier 2, deferred) |
| Bundle execution | Reuses pass-level GL calls | ? (P15.5) |
| `tiled` feature | Not advertised on GLES; yawgpu.h tiled APIs reject GLES device | locked ✗ |
| `shader-passthrough` feature | Not advertised on GLES; yawgpu.h passthrough APIs reject GLES device | locked ✗ |

## Open questions (resolve per slice, record divergences)

- **naga `glsl-out` coverage smoke** (P15.0/P15.1): pick the shaders
  used by the reused Phase 7 e2e tests and confirm they compile to GLES
  ES 3.10 cleanly. Failures may pull WGSL feature gating forward into
  P15.4 scope.
- **Adapter limit mapping** (P15.1): how the existing core
  `RequiredLimits` validation reconciles with GLES `glGet*`-derived
  caps (`GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS`,
  `GL_MAX_VERTEX_UNIFORM_VECTORS`, etc.). Goal: avoid widening the
  default-limit envelope just to satisfy a Tier 2 adapter — instead
  the GLES adapter reports its real caps and `request_device` declines
  unsatisfiable limit asks.
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
