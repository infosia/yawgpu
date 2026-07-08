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
> and buffer binding numbers (explicit identity `BindingRemap`; the naga
> `_block_N` name-parse remap was deleted). See the matrix rows below.
> Remaining naga-era mechanism names elsewhere in this block (`glsl-out`
> flags, `SamplerBindMap`) are historical and pending revision when GLES
> bring-up resumes.
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
| Storage textures | GLES 3.1 `glBindImageTexture` + `image2D` shader qualifiers. Format coverage validated against the GLES image-format table; unsupported formats → `HalError::FormatUnsupported`-class. **Layer-subrange views (2026-07-08):** a 2d-array/3d storage view of a layer subrange (`base_array_layer>0` or `array_layer_count<full`, count>1) — which `glBindImageTexture` alone cannot express (only whole-layered or single-layer) — is bound by aliasing `[base, base+count)` with a transient `glTextureView` (reusing the sampled-view machinery `create_transient_texture_view`) then `glBindImageTexture(view, layered=GL_TRUE)`; the view has exactly `count` layers so view-relative layers and `imageSize().z`/`textureNumLayers` are correct. Requires glTextureView (GLES 3.2); the ES-3.1 fallback keeps the `HalError` (catalogued). Fixed 482 CTS storage_textures_2d_array / textureNumLayers / out_of_bounds_array fails. | GLES 3.1 image format coverage is narrow (R32F, Rgba8, Rgba32F, …). |
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
| Buffer create / map / unmap | `glBufferData(&zeros, size, DYNAMIC_DRAW)` (**zero-initialized** — see below) + `glBufferSubData` (write) + `glMapBufferRange(MAP_READ_BIT)` (read). HostBuffer path in core (`mapped_ptr` returns `None`); persistent map deferred. **Zero-init (2026-07-08):** WebGPU requires buffers to behave as zero-initialized. GL recycles freed-buffer memory within the process, so `glBufferData(NULL)` would expose stale bytes from a destroyed buffer (Vulkan/Metal get zeroed fresh OS pages instead). `allocate_buffer` now uploads a host zero vector, both allocating and zeroing. This is Tier-independent conformance (GLES made to match the zero-init core already assumes); it fixed 15,081 `command_buffer,image_copy` "texture padding mismatch" fails — yawgpu's T2B preserves padding correctly (tight byte-granular row copies), but the preserved bytes were uninitialized garbage that differed between the two fresh buffers the FullCopyT2B check compares — plus 2 `resource_init,buffer` fails. Eager zero-init (a host zero-vec per buffer); large-buffer chunking / lazy-init is a possible future optimization. | ☑ (P15.2 + zero-init; ANGLE round-trip + crocus verified) |
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
| Compute shader | Tint WGSL → GLSL ES 3.10 (`use_framebuffer_fetch=false`, `zero_initialize_workgroup_memory=true`) compiled via `glCreateShader(COMPUTE_SHADER)` + `glLinkProgram`; bind-group bindings honored via `bind_buffer_range(SHADER_STORAGE_BUFFER \| UNIFORM_BUFFER)` against the WGSL `@binding(N)` emitted as `layout(binding=N)`. Single bind group (`@group(0)`) only. **Buffer bindings only** — F-040/F-041: the compute path binds `pass.bind_buffers`; a compute pass with any **texture binding** (sampled or storage texture, e.g. `texture_storage_2d<…, read>`) returns `HalError` ("GLES compute does not support texture bindings") rather than dispatching with the texture unbound. GLES 3.1 `glBindImageTexture` storage-image binding is mappable but unimplemented; core (Tier-1 read-only storage textures via F-041) is unchanged. Threading-audit (group F): `dispatchWorkgroupsIndirect` is applied via `glBindBuffer(DISPATCH_INDIRECT_BUFFER)` + `glDispatchComputeIndirect(offset)` (an indirect-arg offset `> i32::MAX` returns `HalError`). | ☑ (P15.4 + threading-audit: ANGLE verified — buffer bindings + direct/indirect dispatch; texture bindings in compute are catalogued Tier-2 `HalError`) |
| Compute dispatch (direct) | `glDispatchCompute(x, y, z)` + `glMemoryBarrier(ALL_BARRIER_BITS)` | ☑ (P15.4) |
| Compute dispatch (indirect) | `glDispatchComputeIndirect` | ✗ Deferred — `HalComputePass` carries no indirect variant in core; gate first on a core HAL extension |
| Vertex shader / fragment shader | Tint GLSL ES 3.10 output (per-stage emission); shared `generate_glsl` accepts Vertex / Fragment / Compute; wrapped as `HalShaderSource::GlslStages { vertex, fragment }` for render | ☑ (P15.5; ANGLE verified) |
| Vertex formats (F-044) | the full `GPUVertexFormat` set maps to GL vertex-attribute metadata: integer `uint*`/`sint*` formats via `glVertexAttribIPointer`, `unorm*`/`snorm*` via `glVertexAttribPointer` + normalized, `float16*`/`float32*` via plain float, `unorm10_10_10_2` via `GL_UNSIGNED_INT_2_10_10_10_REV`. **`unorm8x4-bgra` returns `HalError`** — GLES 3.1 has no clean BGRA vertex-attribute swizzle. | ◐ (F-044: full format set mapped; `unorm8x4-bgra` is a catalogued Tier-2 `HalError`. ANGLE verification deferred) |
| Render pipeline state | Cached: GL program + `Vec<HalVertexBufferLayout>` + primitive topology + bindings + `Option<UniformLocation>` for Tint's `tint_immediates[0]` first-instance immediate; supports vertex+fragment color pipelines and vertex-only depth-stencil pipelines where GLES program linking accepts a fragment-less program. F-035 applies the color target `writeMask` (`glColorMask`) and `blend` (`glEnable(GL_BLEND)` + `glBlendFuncSeparate` + `glBlendEquationSeparate`) plus the render-pass blend constant (`glBlendColor`) on the single color target; **dual-source blend factors (`Src1` / `OneMinusSrc1` / `Src1Alpha` / `OneMinusSrc1Alpha`) return `HalError`** (`dual-source-blending` is not advertised on the GLES 3.1 baseline). Threading-audit (group A/D/E): `primitive.cullMode` + `primitive.frontFace` are applied per draw (`glEnable/Disable(GL_CULL_FACE)` + `glCullFace` + `glFrontFace`); but `multisample.mask != 0xFFFF_FFFF`, `multisample.alphaToCoverageEnabled = true`, and `primitive.unclippedDepth = true` each return `HalError` (non-default sample mask / alpha-to-coverage / depth-clamp are not supported on the GLES 3.1 baseline — and core already rejects `unclippedDepth` pending a `depth-clip-control` feature) | ◐ (P15.5 + F-031 + F-035 + threading-audit: at most one RGBA8Unorm/BGRA8Unorm color target with writeMask + blend + blend constant + cullMode/frontFace; depth-only is best-effort Tier-2; dual-source blend factors, non-default sample mask, alpha-to-coverage, and unclipped depth are catalogued Tier-2 `HalError`) |
| Render pass (color + depth/stencil) | Transient FBO + `glFramebufferTexture2D(COLOR_ATTACHMENT0 / DEPTH_ATTACHMENT / STENCIL_ATTACHMENT / DEPTH_STENCIL_ATTACHMENT)` + `glDrawBuffers` + `glViewport` + clear (`glClearColor`/`glClearDepthf`/`glClearStencil`/`glClear`); `RenderPassCleanup` Drop guard ensures VAO + FBO + program-state + memory-barrier cleanup runs regardless of inner error. Clear-only render passes (`pass.pipeline == None`) take the FBO+clear path and skip VAO+draw (Q1 fix). F-038 applies the pipeline's depth-stencil **stencil** state per draw: `glEnable(GL_STENCIL_TEST)` + front/back `glStencilFuncSeparate`(compare, **dynamic reference** from `pass.stencil_reference`, readMask) / `glStencilOpSeparate`(fail, depthFail, pass) / `glStencilMaskSeparate`(writeMask); `RenderPassCleanup` also disables `GL_STENCIL_TEST`. The GL `ref` parameter is `GLint`, so a stencil reference `> i32::MAX` returns `HalError` (catalogued Tier-2 limit, no core relaxation). F-040 slice 1 (multiple color attachments) is **Tier-2-deferred on GLES**: the transient FBO path binds a single `COLOR_ATTACHMENT0`, so a regular render pass with **more than one** color target returns `HalError` ("GLES render pass supports at most one color attachment"). GLES 3.1 MRT (`glDrawBuffers` over `COLOR_ATTACHMENT0..N`) is mappable but unimplemented; core validation (which permits N targets on Tier-1) is unchanged. F-040 slice 2 (MSAA + resolve) is also **Tier-2-deferred on GLES**: a multisample render pipeline (`sample_count > 1`, rejected in `gles/pipeline.rs`) or a render pass with any `resolveTarget` (rejected in `gles/queue.rs`) returns `HalError` ("GLES render pass does not support multisample/resolve"). GLES 3.1 has multisample renderbuffers + `glBlitFramebuffer` resolve, mappable but unimplemented; core (Tier-1 MSAA/resolve) is unchanged. Threading-audit (group B): `setViewport` (`glViewport` + `glDepthRangef`) and `setScissorRect` (`glEnable(GL_SCISSOR_TEST)` + `glScissor`, `glDisable` when unset) are applied; an unset viewport/scissor keeps the full-attachment default. Threading-audit (group C): a `depthReadOnly` / `stencilReadOnly` aspect maps to `Load` + preserve (no clear), matching Vk/Metal. Execution-gap audit (group A): render passes with texture or sampler bindings return `HalError` ("GLES render pass does not support texture/sampler bindings") rather than drawing with them unbound; GLES texture/sampler binding remains mappable but unimplemented. **Layered color attachments (2026-07-08):** `create_render_fbo` now accepts `TEXTURE_2D_ARRAY` and `TEXTURE_3D` color targets, attaching the selected `array_layer` (2D-array) or `depth_slice` (3D) via `glFramebufferTextureLayer` (the color analogue of the layered depth-stencil work e185afc, using the `HalRenderColorTarget.array_layer`/`depth_slice` fields); `TEXTURE_CUBE_MAP`/cube-array color targets stay `HalError` (catalogued). Fixed the `rendering,3d_texture_slices` cluster and the non-2D-attachment copyTextureToTexture setup passes. | ◐ (P15.5 + F-031 + F-038 + F-040 + threading-audit + execution-gap audit: 2D / 2D-array-layer / 3D-slice color + depth/stencil FBO attachment supported, including depth-only; dynamic stencil reference, viewport, and scissor applied per draw; read-only depth/stencil preserved; a stencil reference `> i32::MAX`, cube color attachments, and any texture/sampler binding return `HalError`) |
| Sparse color attachments (F-054) | WebGPU allows empty color slots (null view / `Undefined` target) interleaved with real ones, with fragment `@location(N)` targeting slot N. The GLES transient-FBO path binds the single real target to `COLOR_ATTACHMENT0`, so a real target at **color slot 0** (with or without trailing empty slots) is supported, but a real target at a **non-zero slot** (any empty slot/hole before it) returns `HalError` — GLES cannot represent a hole at slot 0, and the GLSL `@location(N)` output would mis-route. Rejected in both `gles/pipeline.rs` (`validate_render_pipeline_descriptor`) and `gles/queue.rs` (`create_render_fbo`). Core + Metal + Vulkan support sparse layouts at any slot (Tier-1). | ◐ (F-054: real color target at slot 0 supported, trailing holes OK; a real target at a non-zero slot is a catalogued Tier-2 `HalError`) |
| `draw` / `drawIndexed` | `glDrawArrays` / `glDrawArraysInstanced`; `glDrawElements` / `glDrawElementsInstanced` for indexed draws. `baseVertex != 0` returns `HalError` on the GLES 3.1 baseline because `glDrawElementsBaseVertex*` is not guaranteed. | ◐ (F-034: direct + indexed direct execute when `baseVertex == 0`; nonzero `baseVertex` is catalogued Tier-2 `HalError`) |
| `drawIndirect` / `drawIndexedIndirect` | `glDrawArraysIndirect` / `glDrawElementsIndirect`. Indexed indirect requires `setIndexBuffer` offset 0 because GLES has no separate element-buffer binding offset for indirect draws; nonzero index-buffer offset returns `HalError`. | ◐ (F-034: indirect variants execute; indexed-indirect nonzero index-buffer offset is catalogued Tier-2 `HalError`) |
| `first_instance` direct | Tint `Options::first_instance_offset = 0` (vertex stages) injects `layout(location = 0) uniform uint tint_immediates[1]`; the HAL sets `tint_immediates[0]` via `glUniform1ui` per draw (covers `@builtin(instance_index)`), **and** offsets every `Instance`-stepped vertex buffer's attribute pointers by `first_instance * array_stride` (Dawn GL parity, `CommandBufferGL.cpp:259-261`; no dirty tracking needed — the GLES path re-specifies a fresh VAO per draw). Replaced the naga-era `naga_vs_first_instance` uniform, which Tint never emits (was a silent no-op after the Tint migration). | ☑ (Tint-integration refactor R6 + Phase Review M2; contract pinned by generated-GLSL unit tests; real-ANGLE re-verification pending) |
| `first_instance` indirect | ✗ Unsupported — feature not advertised (`supports_indirect_first_instance()` = false); GLES 3.1 `Draw*IndirectCommand` has no `baseInstance` field, so it genuinely cannot be honored — indirect draws use `first_instance = 0` | locked ✗ |
| GLSL buffer binding numbers | Core supplies an explicit **identity `BindingRemap`** (`tint_bindings_for_glsl`) so Tint emits `layout(binding = N)` equal to the WGSL `@binding(N)` the HAL binds via `glBindBufferRange`. Without it, Tint's `GenerateBindings` renumbers bindings sequentially in declaration order (a `@binding(3)` buffer became `binding = 1`) — desyncing GLSL from the HAL for non-sequential layouts. The naga-era `_block_N` name-parse remap was deleted (wrong against Tint's declaration-counter block naming). | ☑ (R6; pinned by generated-GLSL unit tests in yawgpu-tint/-core) |
| `textureNumLevels` / `textureNumSamples` (`texture_builtins_from_uniform`) | WIRED (T-G17 exposed the polyfill UBO binding through the shim; c06e516 populates each slot from Tint's `ubo_contents` layout, mapping post-remap binding → WGSL group/binding, filled by the queue with mip-level / sample count). **Cross-stage slot assignment fixed (2026-07-08):** the shim generates GLSL per stage, so each stage's `ubo_contents` was packed from offset 0 independently — vertex and fragment then collided at the same UBO offset for *different* textures (core `merge_texture_metadata_slots` raised `unexpected internal error`, 64 CTS fails in `capability_checks,limits,maxSampledTexturesPerShaderStage`). Fix: the shim sets `ubo_contents[i].offset = resolved_binding.binding` (both the remapped and empty-remaps paths), making the offset a deterministic function of the pipeline-stable resolved binding — vertex and fragment independently compute disjoint offsets for different textures and identical offsets for a shared one. Mirrors Dawn's per-pipeline `EmulatedTextureBuiltinRegistrar` (keyed on FlatBindingIndex; `opengl/PipelineGL.cpp:222-246`) without threading a shared registrar across the two per-stage shim calls. | ☑ (T-G17 + c06e516 + cross-stage offset fix; pinned by yawgpu-tint generated-GLSL test + real-EGL cross-stage HAL test) |
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
- **>1 bind group** (`@group(1..3)`) — the GLES backend implements only
  `@group(0)` today; other groups return HalError. DEFERRED
  implementation (linear binding remap), not a hardware gap.
- **maxBindingsPerBindGroup edge** — a residual cluster of compute/
  fragment shader-compile fails at the reported binding limit suggests
  the GL-queried `max_bindings_per_bind_group` may be 1 too high (a
  reserved binding point such as the T-G17 texture-metadata UBO not
  subtracted). Open follow-up, not catalogued.

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
