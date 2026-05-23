# DONE — refine `examples/tiled_deferred` (3-subpass port)

This handoff was authored against `specs/blocks/55-tiled-rendering.md` →
"Reference example (`examples/tiled_deferred`) — deferred-shading demo"
and would rewrite `examples/tiled_deferred` as a 3-subpass demo
equivalent to `../wgpu/examples/features/src/deferred_rendering`.

**Status:** **DONE 2026-05-23.** Landed across `bd2764b` (initial port) +
`fbd6823` (look_at transpose fix) + `6cd881e` (Windows portability +
windowed-loop). Five library prereqs that this port surfaced landed
between the parking and the final commit:

1. `76aaaac` — mixed input-attachment bind groups (lighting's `[input,
   input, uniform]` @group(0)).
2. `b94d780` — depth-stencil + multi-color subpass pipelines.
3. `e9ebde1` — Metal subpass `stencilAttachment` format-aspect gate.
4. `087c51f` — Rgba16Float HAL format support.
5. `af1bdd2` — Metal no-op `MTLDepthStencilState` fallback.

End-to-end verification:
- Mac Metal (`tiled_deferred --verify`): center pixel `(130, 60, 57, 255)`,
  full 5×5 cube grid with Blinn-Phong lighting in `tiled_deferred.png`.
- Windows native Vulkan (user-verified): renders correctly.
- MoltenVK on macOS self-skips via `adapter_is_moltenvk` (lighting subpass
  needs framebuffer-fetch / input-attachment paths MoltenVK doesn't expose).

The body below is preserved as historical context — it captures the spec
+ guidance the Phase 14.x cycle was driven by. Don't re-dispatch unless
the example breaks against a future API change; see
`specs/tracking/phase-14.md` → "Phase 14.x extensions" for the canonical
final-state log.

---

**Historical parking notes (kept for context):** parked **again** 2026-05-23,
waiting on the Phase 14.x library work that lifts the subpass-pipeline
scaffold's depth-stencil + multi-color restriction (the active `HANDOFF.md`
at that point dispatched that work — long since closed; see #5 above).

**Workflow change (also 2026-05-23):** per user feedback
([[feedback-example-vs-library-workflow]]), example development +
debugging in `examples/` is now Claude's direct responsibility (the
handoff-then-review cycle is too expensive for the trial-and-error
shape of example work). The agent's `examples/tiled_deferred/` WIP
(unstaged on disk: `main.c` 1753-line rewrite, `gbuffer.wgsl`,
`lighting.wgsl`, `composite.wgsl`, `math.h`, updated `CMakeLists.txt`)
becomes Claude's baseline. Claude resumes example verification + debug
directly once the library landing is in.

**Reason for parking (second time):** the wgpu reference's G-Buffer
subpass writes **2 color targets** (albedo + normal) AND has a
**depth-stencil attachment**. `yawgpu-core/src/render_pipeline.rs:755`
today hard-rejects subpass pipelines whose descriptor has
`depth_stencil.is_some()` OR `fragment.target_count != 1`. Additionally
the HAL's `HalRenderPipelineDescriptor` does not yet carry a
`depth_stencil` field at all, and Vulkan's `create_graphics_pipeline` has
a hardcoded length-1 color-blend-attachment array. The deferred demo
cannot proceed until the library supports multi-color + depth subpass
pipelines.

This doc retains the original handoff body (with Revision 1 from the
WGSL-semantics correction round) as a reference for what the example
must do. Re-running the example after library landing should not need
re-dispatching this handoff verbatim — Claude takes over directly.

---

# Task: Refine `examples/tiled_deferred` — port `wgpu deferred_rendering`

Replace the current 2-subpass / 16×16-readback `examples/tiled_deferred`
with a 3-subpass deferred-shading demo equivalent to the wgpu reference at
`../wgpu/examples/features/src/deferred_rendering/`. Visually equivalent
output (same scene values), driven by the same animated camera + 4 lights.
Two run modes: windowed demo (default) and `--verify` (offscreen → PNG +
center-pixel sanity check).

The contract is captured in `specs/blocks/55-tiled-rendering.md` under
**"Reference example (`examples/tiled_deferred`) — deferred-shading demo"**.
Read that first — it is the authoritative description of what the example
must do. This handoff just translates that contract into an
implementation directive.

## Inputs to read

- `specs/blocks/55-tiled-rendering.md` (in particular the new
  "Reference example" sub-section listed above, **and** the existing
  "dual convention accepted" rule — the new example uses the
  subpass-local form so a single WGSL source works on both backends)
- The wgpu reference port (read both the Rust orchestration *and* the
  three WGSL files in full):
  - `../wgpu/examples/features/src/deferred_rendering/mod.rs` (834 lines)
  - `../wgpu/examples/features/src/deferred_rendering/gbuffer.wgsl`
  - `../wgpu/examples/features/src/deferred_rendering/lighting.wgsl`
  - `../wgpu/examples/features/src/deferred_rendering/composite.wgsl`
- yawgpu C example patterns:
  - `examples/triangle/{main.c,CMakeLists.txt,shader.wgsl}` — the
    windowed-example template (window → surface → swapchain → render
    pipeline → main loop, with a CMake POST_BUILD copy of the WGSL).
  - `examples/framework/framework.h` — the helper API surface
    (`yawgpu_context_create`, `yawgpu_window_create`,
    `yawgpu_load_wgsl_shader`, `yawgpu_create_buffer_init`, etc.).
  - `examples/tiled_deferred/main.c` — the *current* tiled_deferred is
    rewritten by this handoff, but its `adapter_is_moltenvk` /
    `TILED_BACKEND_{OK,FAIL,SKIP}` flow is the right pattern to **keep**
    for MoltenVK self-skip.
- `CLAUDE.md`, `specs/reference/naming-conventions.md`.

## Files allowed to change (or create)

- `examples/tiled_deferred/main.c` (rewrite)
- `examples/tiled_deferred/gbuffer.wgsl` (new; copy verbatim from the
  wgpu reference)
- `examples/tiled_deferred/lighting.wgsl` (new; copy verbatim)
- `examples/tiled_deferred/composite.wgsl` (new; copy verbatim)
- `examples/tiled_deferred/math.h` (new, optional — if you'd rather keep
  the Mat4 / Vec3 helpers inline at the top of `main.c`, that's fine
  too; either way, no third-party math library)
- `examples/tiled_deferred/CMakeLists.txt` (update: add the three WGSL
  POST_BUILD copies + ensure GLFW link is picked up via the shared
  framework target the same way `examples/triangle` does it)

## Files NOT allowed to change

- **Any** file outside `examples/tiled_deferred/` — no yawgpu library
  code (yawgpu-core / yawgpu-hal / yawgpu FFI), no framework changes
  (`examples/framework/*`), no spec or tracking files. If something in
  the yawgpu C API or framework looks insufficient or broken, **stop
  and ask** rather than working around it inline.

## What to do

### 1. Three WGSL files — gbuffer verbatim, lighting/composite dual-entry

**This section was revised 2026-05-23 (Revision 1).** The earlier
"verbatim all three" directive was wrong: naga's MSL backend does not
subpass-remap fragment output `@location(N)`, so verbatim
`lighting.wgsl` / `composite.wgsl` with `@location(0)` will write to MTL
slot 0 instead of the lit (flat slot 2) and output (flat slot 3)
attachments on Metal — Metal verify would be black. Validation
accepting both conventions doesn't help: validation only checks "is
there an output for that location", not where it lands.

Do this instead:

- **`gbuffer.wgsl`**: copy from
  `../wgpu/examples/features/src/deferred_rendering/gbuffer.wgsl`
  byte-for-byte. Its fragment writes `@location(0)` (albedo) +
  `@location(1)` (normal), which already match flat MTL slots 0 + 1 in
  subpass 0 — no divergence needed.
- **`lighting.wgsl`**: start from the wgpu reference, then add a
  **second** fragment entry point next to `fs`:
  ```wgsl
  // Vulkan: subpass-local @location(0) is remapped by VkRenderPass to flat
  // attachment slot 2 (the lit HDR target).
  @fragment
  fn fs(in: VertexOutput) -> @location(0) vec4<f32> { ... existing body ... }

  // Metal: naga MSL does not subpass-remap; flat MTL slot 2 must be
  // declared on the WGSL output directly.
  @fragment
  fn fs_metal(in: VertexOutput) -> @location(2) vec4<f32> { ... same body ... }
  ```
  Comment the divergence at the top of the file, citing the
  block-55 "dual convention accepted" rule.
- **`composite.wgsl`**: same pattern, but the flat slot is 3 (the
  output attachment):
  ```wgsl
  @fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> { ... }
  @fragment fn fs_metal(in: VertexOutput) -> @location(3) vec4<f32> { ... }
  ```

In `main.c`, when creating the lighting + composite pipelines, query
`WGPUAdapterInfo.backendType` once at startup and pick the fragment
entry name:

```c
const char *lighting_fs = "fs";
const char *composite_fs = "fs";
WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
if (wgpuAdapterGetInfo(app->context.adapter, &info) == WGPUStatus_Success) {
    if (info.backendType == WGPUBackendType_Metal) {
        lighting_fs = "fs_metal";
        composite_fs = "fs_metal";
    }
    wgpuAdapterInfoFreeMembers(info);
}
```

This is the same dual-entry-point pattern used by
`yawgpu/tests/e2e_metal_tiled.rs` and the prior 2-subpass tiled_deferred.
`gbuffer.wgsl` does not need it (its outputs already match flat slots).

### 2. Math helpers

Implement (in `math.h` or inline at the top of `main.c`):

- `typedef struct { float x, y, z; } Vec3;`
- `typedef struct { float m[16]; } Mat4;`  (column-major, matching glam's
  `to_cols_array_2d()` layout the wgpu reference uses)
- `Vec3` ops: `vec3_make`, `vec3_sub`, `vec3_dot`, `vec3_cross`,
  `vec3_normalize`, `vec3_length`
- `Mat4` ops: `mat4_identity`, `mat4_mul`, `mat4_look_at_rh`,
  `mat4_perspective_rh`, `mat4_inverse`

`mat4_perspective_rh` follows wgpu's convention (D3D/WebGPU NDC z ∈ [0, 1]).
`mat4_look_at_rh` and `mat4_inverse` use the same right-handed,
column-major math as glam. If you're unsure, derive against glam's source
(it's part of the wgpu reference's deps) or — safer — port `glam::Mat4::
look_at_rh / perspective_rh / inverse` by reading the Rust source. Keep
it minimal: only the four matrix functions and ~five vec3 helpers.

### 3. Rewrite `main.c`

Mirror the wgpu reference's structure. Suggested C layout:

- `Vertex` struct: `float position[3]; float normal[3]; float color[3];`.
- `Uniforms`: `float view_proj[16];` (column-major).
- `LightParams`: 4 × `float lights[4]` (xyz=pos, w=intensity) +
  `float camera_pos[3]; float time; float inv_view_proj[16]; float screen_size[2]; float _padding[2];`.
  Layout must match `lighting.wgsl`'s `LightParams` exactly — std140-ish
  uniform layout means the `_padding[2]` is load-bearing; do NOT remove
  it.
- `create_cube_vertices()` returns 24 vertices + 36 `uint16_t` indices,
  values **identical** to the wgpu reference's `create_cube_vertices()`.
- `create_attachment_textures(width, height, surface_format)` allocates
  the four offscreen render-attachment textures (albedo, normal, depth,
  lit) with the formats from the spec:
  - albedo: `WGPUTextureFormat_RGBA8Unorm`
  - normal: `WGPUTextureFormat_RGBA16Float`
  - depth:  `WGPUTextureFormat_Depth32Float`
  - lit:    `WGPUTextureFormat_RGBA16Float`
- `build_subpass_pass_layout(device, surface_format)` constructs the
  `YaWGPUSubpassPassLayout` matching the reference's `build_subpass_target_base`:
  4 color attachments (albedo / normal / lit / output), 1 depth-stencil,
  3 subpasses with `color_attachment_indices` and `input_attachments`
  exactly as in the reference, 2 `ColorToInput` dependencies (0→1, 1→2).
- Three `create_*_pipeline()` helpers, one per subpass; bind groups
  built via **explicit** `wgpuDeviceCreateBindGroupLayout` +
  `wgpuDeviceCreatePipelineLayout` (yawgpu's C API does not yet expose
  `wgpuRenderPipelineGetBindGroupLayout` for the auto-derive path the
  Rust reference uses).
- Main loop: each frame, write camera + light uniforms (using
  `wgpuQueueWriteBuffer`), record the 3-subpass render pass, submit,
  present. In verify mode, render the single time=0 frame to an
  offscreen `Rgba8Unorm` texture instead of presenting, copy it to a
  readback buffer, map, and inspect the center pixel.

### 4. `--verify` mode

Argv-driven: `argc >= 2 && strcmp(argv[1], "--verify") == 0`.

In verify mode:

- Do not open a window. Use an offscreen `Rgba8Unorm` texture as the
  composite subpass's color attachment. Render exactly one frame with
  `time = 0.0`. Copy the texture to a readback buffer, map it, examine
  the center pixel. Pass if `alpha > 0` and `R + G + B > 12` (i.e. not
  the hemisphere-ambient near-black background). Write the frame to
  `tiled_deferred.png` either way.
- Exit `EXIT_SUCCESS` on pass, `EXIT_FAILURE` on fail. The MoltenVK
  self-skip path (`adapter_is_moltenvk` → `TILED_BACKEND_SKIP` →
  `EXIT_SUCCESS` with an explanatory message) is preserved unchanged.

In windowed mode:

- Open a `1024×768` GLFW window via `yawgpu_window_create`. Configure
  the surface with the surface's preferred BGRA8Unorm/RGBA8Unorm format
  (mirror `examples/triangle::choose_surface_format`). Main loop:
  poll events → request `wgpuSurfaceGetCurrentTexture` → use its view
  as the composite subpass's color attachment → submit → present.
  Animate camera + lights against wall-clock seconds since startup
  (the reference's `start_time.elapsed().as_secs_f32()`).

### 5. CMake

Update `examples/tiled_deferred/CMakeLists.txt`:

- Keep `yawgpu_add_example(tiled_deferred)` — that pulls in the
  framework + GLFW link the same way `examples/triangle` does.
- Replace the lone `target_compile_definitions ... STB_IMAGE_WRITE_IMPLEMENTATION`
  with a small block (and the corresponding `target_include_directories
  ... "${CMAKE_SOURCE_DIR}/capture"` to keep `stb_image_write.h` reachable
  for PNG output in verify mode).
- Add three `add_custom_command(TARGET tiled_deferred POST_BUILD COMMAND
  copy_if_different ...)` lines, one per WGSL, mirroring
  `examples/triangle/CMakeLists.txt`.

## Acceptance criteria

- [ ] `gbuffer.wgsl` in `examples/tiled_deferred/` is byte-equal to
      the wgpu reference (`diff -q` against
      `../wgpu/examples/features/src/deferred_rendering/gbuffer.wgsl`
      reports no diff).
- [ ] `lighting.wgsl` and `composite.wgsl` declare BOTH a `fs` (writes
      `@location(0)`) and an `fs_metal` (writes `@location(2)` and
      `@location(3)` respectively) fragment entry point, with a header
      comment citing the block-55 "dual convention accepted" rule.
      The shared body must be identical between the two entry points
      (only the output location differs).
- [ ] `main.c` selects `lighting_fs` / `composite_fs` based on
      `WGPUAdapterInfo.backendType == WGPUBackendType_Metal`.
- [ ] `cmake -S examples -B examples/build-tm -DYAWGPU_FEATURE=metal
      -DYAWGPU_EXTENSIONS=tiled && cmake --build examples/build-tm --target tiled_deferred`
      succeeds (Claude verifies).
- [ ] `cmake -S examples -B examples/build-tv -DYAWGPU_FEATURE=vulkan
      -DYAWGPU_EXTENSIONS=tiled && cmake --build examples/build-tv --target tiled_deferred`
      succeeds (Claude verifies).
- [ ] `examples/build-tm/tiled_deferred/tiled_deferred --verify` on
      Metal: exit 0, prints a center-pixel report showing alpha > 0 and
      R+G+B > 12; writes `tiled_deferred.png` (Claude verifies on this M2).
- [ ] `examples/build-tv/tiled_deferred/tiled_deferred --verify` on
      MoltenVK: exit 0 with the self-skip message (the demo's lighting
      subpass needs framebuffer-fetch / input-attachments that MoltenVK
      doesn't expose; preserve the existing self-skip path).
- [ ] `examples/build-tm/tiled_deferred/tiled_deferred` (windowed,
      Metal): opens a window and renders the animated 5×5 grid for at
      least one second without crashing or emitting uncaptured device
      errors (Claude verifies; can be sanity-checked by closing the
      window after a couple of frames).
- [ ] No new files / changes outside `examples/tiled_deferred/`.
- [ ] No new dependency (vendored or system) — only the existing
      framework + GLFW link picked up via `yawgpu_add_example`.
- [ ] Code style mirrors the existing C examples (4-space indent, snake_case,
      structs `Pascal`-prefixed where the existing code does that, no
      tabs). No emoji. Comments only where the *why* is non-obvious.

## Out of scope

- Any yawgpu library change (FFI, core, HAL).
- Any framework change (`examples/framework/*`). If GLFW window
  creation or surface configuration has a gap on a backend, stop and
  ask — do not patch around it.
- The depth-transient B2 follow-up. The example uses regular DRAM-backed
  textures for albedo / normal / depth / lit, matching the wgpu
  reference. When the yawgpu transient depth path closes, a follow-up
  HANDOFF will switch this example over.
- The `wgpuRenderPipelineGetBindGroupLayout` auto-derive path the wgpu
  reference uses for lighting / composite. yawgpu's C API does not
  expose that; use **explicit** `wgpuDeviceCreateBindGroupLayout`
  declarations in this example (one for the gbuffer uniform-only group,
  one for the lighting group with 2 subpass inputs + 1 uniform, one
  for the composite group with 1 subpass input).

## Report back

- The four files written / rewritten (paths + brief description).
- The two `cmake --build` commands you used and their exit status.
- The `--verify` exit status + center-pixel report on whichever
  backend you tested locally (if any).
- Anything in the spec rule or this handoff that was unclear — **ask**
  rather than improvise. Two prior rounds were rejected for missed
  WGSL semantics; if any acceptance criterion looks impossible without
  touching a non-allowlisted file, STOP and surface that.

## Revision 1 (2026-05-23 — WGSL semantics correction)

The earlier "all three WGSL files verbatim" directive in Section 1 above
was wrong: naga's MSL backend does not subpass-remap fragment output
`@location(N)`, so verbatim `lighting.wgsl` / `composite.wgsl` with
`@location(0)` lands in MTL slot 0 on Metal — never reaching the lit
(flat slot 2) or output (flat slot 3) attachments. Metal verify would
be black. The C1 cascade fix relaxed *validation* to accept either
convention (`outputs.get(&subpass_local).or_else(|| outputs.get(&flat))`),
but validation is a "does the shader write to a location" check, not a
"does the location route to the right MTL slot" routing. The routing is
naga's responsibility, and naga doesn't remap.

Section 1 above has been rewritten in place to require:
- `gbuffer.wgsl`: verbatim (its outputs are already at flat slots 0 + 1).
- `lighting.wgsl` and `composite.wgsl`: keep the wgpu reference's `fs`
  entry point (Vulkan), add a sibling `fs_metal` whose
  `@location(N)` matches the flat MTL slot for that subpass's output
  (2 for lighting, 3 for composite). Identical bodies; only the output
  location differs. Same divergence pattern documented in
  `specs/blocks/55-tiled-rendering.md`'s "dual convention accepted" rule
  and used by `yawgpu/tests/e2e_metal_tiled.rs`.
- `main.c`: pick `lighting_fs` / `composite_fs` from
  `WGPUAdapterInfo.backendType`.

Acceptance criteria in this file have been updated to match.

If your in-progress `main.c` already implements the rest of the
contract (3-subpass pass layout, mixed bind group for lighting, vertex
buffers, etc.), the surgical delta is:
1. Write the two new WGSL files with both entry points (do not
   regenerate the others).
2. Thread the entry-name selection into the lighting + composite
   pipeline-creation calls.
3. Re-run the verify gate on Metal — center pixel should now be lit
   instead of black.

## Revision 2 (2026-05-23 — parked again, library blocker)

Agent reported successful implementation of:
- 3-subpass deferred shading demo in `main.c` (1753-line rewrite from the
  2-subpass version).
- `gbuffer.wgsl` byte-equal to wgpu reference.
- `lighting.wgsl` / `composite.wgsl` with `fs` + `fs_metal` dual entries
  per Revision 1.
- Runtime entry-name selection in `main.c` via `WGPUAdapterInfo.backendType`.
- `math.h` (Mat4/Vec3 helpers, no third-party math).
- `CMakeLists.txt` POST_BUILD copies for the three WGSLs.

Result on Metal `--verify`:

    tiled_deferred: center pixel RGBA=(0,0,0,0), rgb_sum=0 FAILED

Root cause is not in the example. `yawgpu-core/src/render_pipeline.rs:755`'s
scaffold guard rejects the gbuffer subpass pipeline (2 color targets +
depth_stencil = Some). The deferred demo cannot proceed until the library
supports multi-color + depth subpass pipelines.

When the Phase 14.x library work lands, this handoff body re-dispatches
unchanged. Expected verify outcome on Metal: green-lit cube grid center
pixel; on MoltenVK: still self-skip (the lighting subpass needs the
framebuffer-fetch path that MoltenVK doesn't expose).
