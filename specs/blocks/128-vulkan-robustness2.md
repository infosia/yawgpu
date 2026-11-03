# Block 128 — Vulkan robustness2 and pipeline robustness, as Dawn

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; all
resolved, see below. Native Vulkan (NVIDIA, buffer and image paths) ran on 2026-11-03;
see "Windows native Vulkan" below.
Source: the Block 124 survey's remaining items (backlog "Recommended order" 1), started
by user request on 2026-11-03.
Oracle: Dawn at the pinned `third_party/dawn`:
- `native/vulkan/DeviceVk.cpp` (feature enables under `IsRobustnessEnabled()`)
- `native/vulkan/PhysicalDeviceVk.cpp` (`VulkanUseImageRobustAccess2`,
  `VulkanUseBufferRobustAccess2` defaults)
- `native/vulkan/ShaderModuleVk.cpp` (Tint `extensions.disable_*` options)
- `native/vulkan/RenderPipelineVk.cpp` (`VkPipelineRobustnessCreateInfo`)
- `native/vulkan/BufferVk.cpp` (uniform allocation rounding, which yawgpu does not do)

## Problem

yawgpu enables Vulkan 1.0 `robustBufferAccess` and runs Tint's full robustness
transform on every SPIR-V shader. Dawn also uses the driver's robustness where the
device offers it, and skips the matching shader-side clamps:

1. With `VK_EXT_robustness2`, Dawn enables the extension and chains every supported
   `VkPhysicalDeviceRobustness2FeaturesEXT` bit.
2. If `robustImageAccess2` is on, Dawn sets Tint's `disable_image_robustness`.
3. If `robustBufferAccess2` is on, Dawn sets
   `disable_storage_runtime_sized_array_index_clamping`. There is one exception: when
   descriptor indexing is available but `robustBufferAccessUpdateAfterBind` is false.
4. With `VK_EXT_pipeline_robustness`, Dawn enables `pipelineRobustness`. Each render
   pipeline then chains `VkPipelineRobustnessCreateInfoEXT`: vertex inputs use
   `ROBUST_BUFFER_ACCESS`, and uniform buffers, storage buffers and images use
   `DEVICE_DEFAULT`.

yawgpu does none of this. Its shaders carry clamps that Dawn's do not, and its
OOB storage-array accesses behave differently from Dawn's on the same device. Both
behaviours are allowed by WebGPU, so the CTS cannot see the difference.

F-112 tried to drop buffer clamps when naga was the compiler. F-127 then removed that,
because Tint at the time had only a whole-shader `disable_robustness`. The pinned Tint now
has the per-class options above, so the same goal can be met the way Dawn does it.

## Rules

R1. **Device creation.** When `VK_EXT_robustness2` is available, enable it and chain
the supported `VkPhysicalDeviceRobustness2FeaturesEXT` bits unchanged, as Dawn does.
When `VK_EXT_pipeline_robustness` is available and `pipelineRobustness` is supported,
enable the extension and the feature. `robustBufferAccess` stays as it is today.

R2. **Decisions.** Each is a pure function of device facts, with a unit test:
- `use_image_robust_access2` = robustness2 is enabled && `robustImageAccess2`.
- `use_buffer_robust_access2` = robustness2 is enabled && `robustBufferAccess` is
  enabled && `robustBufferAccess2` && !(descriptor indexing is available &&
  !`robustBufferAccessUpdateAfterBind`). "Available" follows Dawn's
  `DeviceExt::DescriptorIndexing`, which is set only when `VK_EXT_descriptor_indexing`
  is advertised. Dawn does no core-version promotion for it
  (`VulkanExtensions.cpp` `EnsureDependencies`). Query
  `VkPhysicalDeviceDescriptorIndexingProperties` only when it is available.
- `robustBufferAccess2` is chained only when `robustBufferAccess` is enabled
  (VUID-VkPhysicalDeviceRobustness2FeaturesEXT-robustBufferAccess2-04000). Dawn
  requires `robustBufferAccess` of every adapter. yawgpu enables it only when it is
  supported, so it clears the bit instead.

R3. **Tint options.** SPIR-V generation sets the following two options. Both are in
the SPIR-V codegen key (Block 95):
- `extensions.disable_image_robustness = use_image_robust_access2`
- `extensions.disable_storage_runtime_sized_array_index_clamping = use_buffer_robust_access2`

`disable_uniform_runtime_sized_array_index_clamping` stays false. This is a recorded
deviation, for two reasons. Dawn's condition for it relies on rounding uniform
allocations up to 16 bytes, which yawgpu does not do. And WebGPU without Chromium's
`buffer_view` extension has no runtime-sized uniform arrays. The
subgroup-matrix option is not applicable.

R4. **Pipeline robustness.** When R1 enables pipeline robustness, every Vulkan render
pipeline chains `VkPipelineRobustnessCreateInfoEXT` with Dawn's values. This covers
the normal path and the tiled subpass path. Compute pipelines do not chain it, as in
Dawn.

An out-of-bounds atomic on a runtime-sized storage array, with the clamp removed,
may return an undefined value under `robustBufferAccess2`. WGSL asks for a value
from the variable or zero. Dawn behaves the same way, so the gap is recorded here
rather than worked around.

R5. Keep `disable_robustness` false and the rest of the robustness transform
unchanged. This still covers uniform, workgroup, function and private accesses, and
fixed-size storage arrays.

## Tests

- Unit tests:
  - R2: every branch, including the descriptor-indexing exception and a missing extension.
  - R3: the codegen key differs when either flag differs.
  - Shim: with `disable_image_robustness`, a `textureLoad` shader loses its coordinate
    clamp. With `disable_storage_runtime_sized_array_index_clamping`, a runtime-sized
    storage array index loses its clamp.
- Vulkan HAL `#[ignore]` test: device creation reports the enabled robustness2 bits and
  pipeline robustness, and a render pipeline is created with 0 VUIDs.
- e2e on real hardware (Claude): OOB `textureLoad`, and OOB runtime-sized storage reads
  and writes, give WebGPU-allowed results.
  - MoltenVK takes the image path: `robustImageAccess2 = true`, `robustBufferAccess2 = false`.
  - Native NVIDIA takes the buffer path.
- CTS: `shader,execution,robust_access*` and `api,operation` on MoltenVK. On Metal,
  `webgpu:*` must stay `fail=2 crash=0`. Native Vulkan `webgpu:*` is owed. Record F-141
  (NVIDIA `corr`) there, because F-112's original target was the storage clamp.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests.
- **S2 (Claude):** e2e, MoltenVK and Metal CTS, Phase Review.

## Verification (2026-11-03)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which
is reference-only. MoltenVK reports `robustImageAccess2 = true`,
`robustBufferAccess2 = false`, and no `VK_EXT_pipeline_robustness`. So this host takes
the image path only. The HAL test logs `imageDecision=true, bufferDecision=false`.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,324 passed, 0 failed |
| fmt, clippy on 8 feature sets | clean |
| Vulkan HAL ignored tests | 76 passed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled` | 149 passed, 1 failed (the known `imageView2DOn3DImage`). Every VUID comes from that test and its leaked objects |
| `e2e_vulkan_robust_access`, 3 cases | pass |
| Metal e2e, `metal,tiled` | 154 passed, 0 failed; `e2e_metal_robust_access` 3/3 |
| webgpu-native-cts, MoltenVK, `robust_access`, `robust_access_vertex`, `textureLoad`, `arrayLength` | same failing cases with the Block 126 library and this one |
| webgpu-native-cts, MoltenVK, `textureStore`, `textureDimensions`, `textureNumLayers`, `textureNumLevels`, `api,operation` | pass 307,453; same failing cases before and after |

The e2e shows the path change. These are the out-of-bounds results:

| Case | MoltenVK, driver image robustness | Metal, Tint clamps |
|---|---|---|
| `textureLoad` on a sampled texture | `(0, 0, 0, 0)` | the clamped in-bounds texel |
| `textureLoad` on a read-write storage texture | `0` | the clamped in-bounds texel |
| Out-of-bounds `textureStore` | dropped | dropped |

All of these are values WGSL allows.

The MoltenVK failures that exist before and after are known MoltenVK artifacts:
- `robust_access:linear_memory`: 216 failures, all f16 matrices in storage, where MoltenVK fails to create the pipeline.
- `robust_access_vertex`: the failing subcases change from run to run. This is the MoltenVK-only
  residual noted under F-068 in the CTS suite.
- `copyTextureToTexture` into sRGB destinations.
- `3d_texture_slices`.

Metal CTS was not rerun, because the change touches only SPIR-V options and the Vulkan
HAL. A native Vulkan run on NVIDIA is owed. It is the only host here that takes the
`robustBufferAccess2` path. Record F-141 (`memory_model,coherence:corr`) in that run.

### Windows native Vulkan (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`5ecaafc`. The Khronos validation layer was on for the e2e and HAL runs. This host
takes every path: the HAL test logs `robustBufferAccess2=1, robustImageAccess2=1,
nullDescriptor=1, descriptorIndexing=true, robustBufferAccessUpdateAfterBind=1,
pipelineRobustness=true, imageDecision=true, bufferDecision=true`.

| Gate | Result |
|---|---|
| Vulkan HAL ignored tests | 76 passed, 0 failed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 160 passed, 0 failed, 0 VUIDs |
| `e2e_vulkan_robust_access`, 3 cases | pass |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt` | `fail=3 crash=1`; none caused by this block, see below |

The out-of-bounds results on the buffer path:

| Case | Native NVIDIA, driver buffer and image robustness |
|---|---|
| Out-of-bounds runtime-sized storage reads | `[0, 0, 0, 0]` |
| Out-of-bounds runtime-sized storage writes | dropped; the binding keeps `[1040, 1041, 1042, 1043]` |
| `textureLoad` on a sampled texture | in-bounds `[3, 2, 2, 7]`, every out-of-bounds load `(0, 0, 0, 0)` |
| `textureLoad` on a read-write storage texture | `0` |
| Out-of-bounds `textureStore` | dropped |

All of these are values WGSL allows.

The CTS ran as four per-area runs with `--workers 8` and took 29 minutes.

| Area | pass | skip | fail | crash | xfail | Block 125 run |
|---|---:|---:|---:|---:|---:|---:|
| `api,validation` | 246,484 | 111,283 | 0 | 0 | 4 | 246,484 |
| `api,operation` | 180,778 | 152,819 | 1 | 0 | 0 | 180,779 |
| `shader,execution` | 562,543 | 320,680 | 0 | 1 | 25 | 562,548 |
| `shader,validation` | 662,535 | 20,299 | 2 | 0 | 0 | 662,535 |
| **total** | **1,652,340** | **605,081** | **3** | **1** | **29** | **1,652,346** |

- The 2 `shader,validation` failures are the Block 119 R9 cases.
- The crash is F-165, the intermittent NVIDIA driver crash on `bufferArrayView`
  `mixed_types_aliasing` (workgroup, `vec4i_vec4h`). It accounts for the 5 missing
  `shader,execution` passes. The Block 125 run did not hit it; the 2026-11-01 baseline did.
- F-141: `memory_model,coherence:corr` on `atomic_storage` observed the disallowed
  weak behaviour again (`testResults[3] == 4`), so it stays `xfail`. The
  `atomic_workgroup` variant (F-152) passed and shows as `xpass`. The storage index
  clamp that F-112 targeted is now off on this host, and the result is unchanged.
- The `api,operation` failure is
  `copyTextureToTexture:color_textures,non_compressed,array` for `r8snorm` 3d,
  "texture padding mismatch". **It is not caused by this block.** Running
  `copyTextureToTexture:*` alone gives 2 or 3 such failures (`r8snorm`, `r8uint`,
  `r8sint`, all 3d) on every run, with this block's library, with the `6264d93`
  library (before Block 128), and with the `2b3f4a7` library (before Block 121). The
  single case run by itself passes 5 of 5. The cause is that Vulkan buffer memory is
  not zero-initialized. `create_buffer` (`yawgpu-hal/src/vulkan/buffer.rs`) allocates
  and maps each buffer and never clears it. Vulkan does not guarantee that new memory
  reads as zero, and the driver hands back memory freed earlier in the process. A
  diagnostic build that zero-filled the mapped memory, not committed, gave 0 failures
  in 4 of 4 runs of `copyTextureToTexture:*`. This is the same failure GLES had before
  its zero-init fix (`specs/blocks/67-gles-backend.md`, buffer row). Backlog A18.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | Descriptor-indexing availability promoted Vulkan 1.2, but Dawn checks only the extension name | Extension-only, as Dawn; the spec text was wrong and is corrected |
| m2 | `robustBufferAccess2` could be chained without `robustBufferAccess` (VUID-04000) | The bit is cleared, and the decision is false, when `robustBufferAccess` is not enabled |
| m3 | The e2e passes on either path | The path is logged by the HAL ignored test, and the e2e prints the values. They differ by path, as the table above shows |
| m4 | No storage-texture case; atomics | `storage_texture_out_of_bounds` added. The atomic gap is shared with Dawn and recorded under R3 |
| m5 | HAL test creates only a normal-path pipeline | The tiled e2e ran under the Khronos layer with no new VUID |
