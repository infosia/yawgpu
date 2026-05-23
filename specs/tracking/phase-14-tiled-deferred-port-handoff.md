# Parked handoff — refine `examples/tiled_deferred` (3-subpass port)

This handoff was authored against `specs/blocks/55-tiled-rendering.md` →
"Reference example (`examples/tiled_deferred`) — deferred-shading demo"
and would have rewritten `examples/tiled_deferred` as a 3-subpass demo
equivalent to `../wgpu/examples/features/src/deferred_rendering`.

**Status:** parked, waiting on the mixed input-attachment bind-group
library work (next active `HANDOFF.md`, captured here on the same date).

**Reason for parking:** the wgpu reference's `lighting.wgsl` places two
`subpass_input<f32>` bindings (`@group(0) @binding(0..=1)`) and one
`var<uniform> LightParams` (`@group(0) @binding(2)`) into a single bind
group. Today's `yawgpu-core::bind_group::validate_bind_group_descriptor`
rejects a bind group whose `entries[]` count differs from the layout's
entry count, so the caller cannot create a bind group that supplies only
the non-input entry while leaving the two input-attachment slots
auto-wired. Re-dispatch this handoff *unchanged* once that library
change lands; the WGSL stays verbatim with the reference.

When re-dispatching, place the content below back into `HANDOFF.md`
(under `# Task: ...`) and confirm the spec rule it references
("Mixed group" wording added 2026-05-23) is still in place.

---

(The exact handoff body, verbatim from the `HANDOFF.md` it was originally
written into, follows. Do not edit it here — fix the spec or write a
revision instead, the same way other revisions in this repo work.)

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

### 1. Three WGSL files, verbatim

Copy `../wgpu/examples/features/src/deferred_rendering/{gbuffer,lighting,composite}.wgsl`
to `examples/tiled_deferred/` byte-for-byte. Do not adapt them — the
contract says verbatim. The lighting + composite fragments already use
the **subpass-local** `@location(0)` convention, which the cascade's
tolerant `validate_color_targets` accepts on both Vulkan (subpass-local)
and Metal (the fallback flat lookup also matches `@location(0)` when the
subpass writes its local color slot 0). No `fs_metal` second entry point
is needed for this example, unlike the 2-subpass smoke test.

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

- [ ] Three WGSL files in `examples/tiled_deferred/` byte-equal to the
      wgpu reference (`diff -q` against
      `../wgpu/examples/features/src/deferred_rendering/{gbuffer,lighting,composite}.wgsl`
      should report no diff).
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
  rather than improvise. The previous round had a misaligned-handoff
  failure; if any acceptance criterion looks impossible without
  touching a non-allowlisted file, STOP and surface that.
