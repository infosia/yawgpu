# Block 146 — External textures on Vulkan

Status: **COMPLETE (2026-11-05)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; all resolved, see
below. The shared e2e on Metal is owed to a Mac host.
Source: user request (2026-11-05). Lifts the Vulkan rejection of Block 36 R9.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/`):
- `native/BindGroupLayoutInternal.cpp` (the external-texture binding expansion: two
  sampled textures and one uniform buffer per `texture_external`)
- `native/ShaderModule.cpp` / `native/PipelineLayout.cpp` (the expansion map handed to Tint)
- the Tint multiplanar external-texture transform, which every writer, SPIR-V included,
  applies from `tint::Bindings::external_texture`

## Problem

External textures work on Metal only. On Vulkan, `vulkan_external_texture_rejection`
(`yawgpu-core/src/compute_pipeline.rs`, called for compute and render pipelines) fails
every pipeline whose shader uses a `texture_external` with the internal error "external
textures are not supported on the Vulkan backend". The rejection dates from the naga
frontend and was kept because yawgpu never set up the expanded bindings on Vulkan, so
whether the SPIR-V was accepted depended on the driver.

With Tint, the missing piece is only yawgpu's wiring:
- Tint already lowers `texture_external` for SPIR-V when `tint::Bindings::external_texture`
  maps it to two planes and a params buffer. The shim builds that map
  (`make_bindings` in `yawgpu-tint/shim/tint_shim.cpp`) but always places the expansion in
  group 0, because only MSL uses it today.
- Core already builds a backend-independent `HalBoundExternalTexture` (plane0, plane1, the
  params buffer; Blocks 36, 144, 145). The Vulkan HAL only retains it; it writes no
  descriptors for it.

## Rules

R1. **Expansion numbering.** For each bind group layout, every `texture_external` entry at
binding `b` is expanded to:
- plane0 at binding `b` (sampled image)
- plane1 at `base + 2k` (sampled image)
- params at `base + 2k + 1` (uniform buffer)

where `base` is one more than the largest binding number in that bind group layout, and
`k` is the index of the entry among that layout's external-texture entries in ascending
binding order. The numbering is a pure function of the bind group layout, computed in
core, and used identically for SPIR-V generation, descriptor-set layouts, and descriptor
writes. Dawn counts down from `u32::MAX`; yawgpu packs above the largest binding instead,
so that descriptor-set layouts stay dense. The values are internal and never reach WGSL.

R2. **SPIR-V generation.** For Vulkan, the SPIR-V request carries the external-texture map
with R1's numbers in the texture's own group. The shim's remap gains the destination group
(MSL keeps passing group 0, unchanged). `vulkan_external_texture_rejection` is removed for
the normal compute and render paths. The SPIR-V codegen key (Block 95) includes the map.

R3. **Descriptor-set layouts.** The `HalDescriptorBinding` list core gives the Vulkan HAL
for a pipeline contains, for each external texture the pipeline uses, the three expanded
bindings: plane0 and plane1 as `Texture`, params as `Buffer(Uniform)`, with the same
pipeline-wide stage flags the Vulkan HAL gives every other binding. The list comes from
the bind group layout's full entries, not only the bindings the shader uses, so set
layouts and descriptor writes always agree.

R4. **Descriptor writes.** `HalBoundExternalTexture` carries R1's plane1 and params binding
numbers for Vulkan (as it carries Metal slot indices today). At each draw or dispatch the
Vulkan HAL writes plane0's view at `binding`, plane1's view at the plane1 binding, and the
params buffer at the params binding, in the same descriptor set as the group's other
resources, with the same image layouts and barriers as sampled textures and uniform
buffers. The resources stay retained until the submission completes, as today.

R5. **Unchanged.**
- Metal and its slots.
- WebGPU validation of external textures, bind groups and limits (core, Block 36).
- GLES keeps its HAL-level rejection (Tier 2, Block 67 catalogue).
- Tiled subpass pipelines: if one uses an external texture, it fails with an internal
  error that names the subpass path, on every backend; supporting it is out of scope. On
  Metal this replaces a silent failure: the subpass encoder passed no external-texture
  bindings, so the planes were left unbound.
- `shader-passthrough` SPIR-V: no change.

## Tests

- Unit (core): R1 numbering for one and several external textures, unsorted bindings,
  a layout whose largest binding is the external texture itself, and two groups; the
  SPIR-V request map; the descriptor list (R3); codegen keys differ when the map differs;
  the removed rejection (a Vulkan pipeline with an external texture is created).
- Unit (shim, real Tint): with the group-aware map, the SPIR-V declares the two planes and
  the params block at the expected set and bindings.
- Unit (Vulkan HAL): the descriptor writes for a bound external texture (R4), where
  testable without a GPU.
- e2e (Claude): the Metal external-texture scenarios move to a shared
  `yawgpu/tests/common/external_texture.rs` and run on Metal and Vulkan. The Vulkan
  rejection e2e is replaced by the real scenarios. Under the Khronos layer, 0 VUIDs.
- CTS (Claude): `api,validation,render_pipeline,misc:external_texture` now passes on Vulkan
  (today an `xfail` in webgpu-native-cts `expectations/yawgpu-vulkan.txt`, F-111). The
  execution tests that use `texture_external` skip in the C-API port.

## Out of scope

- External textures in tiled subpass pipelines.
- GLES.
- Dawn's static sampler for external textures (`NeedsStaticSamplerForExternalTexture`).

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests.
- **S2 (Claude):** shared e2e, Windows native Vulkan and Metal (Mac host) runs, CTS, Phase
  Review.

## Verification (2026-11-05)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The Khronos
validation layer was on for the e2e and HAL runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,436 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace), `vulkan,tiled` and `gles` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, `yawgpu-tint`), all targets | clean |
| HAL lib tests, `vulkan,tiled` | 385 passed, 0 failed |
| core lib tests, `tiled` | pass |
| `yawgpu-tint` tests, real Tint | pass, including `external_texture_remap_generates_spirv_in_its_own_group` |
| e2e `e2e_vulkan_external_texture`, before (rejection still in place) | 0 of 5 pass: "external textures are not supported on the Vulkan backend" |
| e2e `e2e_vulkan_external_texture`, after | 6 of 6 pass: RGBA passthrough, NV12 two-plane, discarded planes read zero, plane-view subresource, compute with an unreferenced external texture, and two external textures in a sparse group 1 through a render bundle |
| Vulkan e2e, every integration test, `--ignored` | 176 passed, 0 failed, 0 VUIDs |
| Vulkan HAL ignored tests (before the review fixes) | 82 passed, 0 failed, 0 VUIDs |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, before the review fixes | `fail=2 crash=2`, all known |

| Area | pass | skip | fail | crash | Block 136 run pass |
|---|---:|---:|---:|---:|---:|
| `api,validation` | 246,724 | 111,045 | 0 | 0 | 246,666 |
| `api,operation` | 181,451 | 152,147 | 0 | 0 | 180,779 |
| `shader,execution` | 564,770 | 318,448 | 0 | 2 | 562,531 |
| `shader,validation` | 662,599 | 20,235 | 2 | 0 | 662,535 |

- About 3,000 subcases move from skip to pass across the four areas: tests that the CTS
  skipped when a Vulkan pipeline with an external texture failed now run.
- `api,validation,render_pipeline,misc:external_texture` (2 cases, F-111 in the suite) now
  pass and show as `xpass`; their `xfail` entries in webgpu-native-cts
  `expectations/yawgpu-vulkan.txt` can be dropped.
- The 2 failures are the Block 119 R9 cases; the 2 crashes are F-165.

The review fixes (m2 and the HAL test) change no Vulkan behaviour on this host; the gates,
the HAL lib tests and the full Vulkan e2e ran after them.

Block 141's int64-atomics HAL unit test failed on Windows: it compared extension names by
pointer, and a Rust constant does not have one guaranteed address. It now compares the
string contents. It passed on macOS by chance.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | The tiled-subpass rejection fires on every backend, which R5's "Metal unchanged" contradicted | R5 reworded: on Metal it replaces a silent failure, since the subpass encoder bound no external-texture planes |
| m2 | Building `HalBoundExternalTexture` on every backend depended on the Vulkan expansion numbers; a missing number would drop Metal's bindings | The numbers are `Option<u32>`; only the Vulkan HAL requires them and returns a `HalError` when absent; unit tests both ways |
| m3 | README and Blocks 36 and 144 still said external textures are Metal-only | Updated |
| m4 | The shared e2e only covered group 0 with one external texture | New scenario: group 1, external textures at bindings 1 and 4, a uniform at 9, drawn from a render bundle |
| m5 | R3 said the expanded bindings use the stages that use the texture; the HAL uses pipeline-wide flags | R3 states it |
