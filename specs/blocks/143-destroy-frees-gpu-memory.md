# Block 143 — `destroy()` frees GPU memory

Status: **COMPLETE (2026-11-05)**. Phase Review: 0 CRITICAL, 0 MAJOR, 7 MINOR; six
resolved, one declined with a reason, see below.
Source: backlog A24, started by user request on 2026-11-05.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`metal/BufferMTL.mm` (`Buffer::DestroyImpl`), `vulkan/BufferVk.cpp`
(`Buffer::DestroyImpl`), `metal/TextureMTL.mm` (`Texture::DestroyImpl`),
`vulkan/TextureVk.cpp` (`InternalTexture::DestroyImpl`), `Texture.cpp`
(`TextureBase::Destroy`).

## Problem

`Buffer::destroy`, `Texture::destroy`, and `QuerySet::destroy` in `yawgpu-core` mark the
object destroyed. `Buffer::destroy` also clears its map state and host bytes. None of them
release the HAL object: `inner.hal` is set at creation and lives until the last handle
drops. Until then the GPU allocation stays in use.

Dawn releases the backend object at `destroy()`. On Metal it drops its reference to the
`MTLBuffer` or `MTLTexture`; Metal command buffers keep their own references, so the
memory goes once the GPU is done with it. On Vulkan it hands the `VkBuffer`, `VkImage`,
and their memory to the fenced deleter, which frees them after the last submission that
used them completes.

Where handles are released late, for example a binding whose garbage collector releases
them, `destroy()` is the only way an application can return memory promptly. yawgpu
ignores it.

yawgpu already keeps resources alive for the GPU, without help from core:

- The Metal HAL keeps every submission's `HalCopy` list, and with it every HAL buffer,
  texture, and query set the submission uses, until the command buffer completes
  (`yawgpu-hal/src/metal/queue.rs`). Metal command buffers also retain what they use.
- The Vulkan HAL collects the HAL resources of every submission and keeps them until the
  submission's fence signals (`collect_retained_resources` in
  `yawgpu-hal/src/vulkan/encode.rs`).
- Core `TextureView` and `BindGroup` do not hold HAL objects. The HAL views and descriptor
  bindings are built from `Texture::hal()` and `Buffer::hal()` when a submission is
  lowered.

So core can drop its reference at `destroy()`, and the HAL frees the object after the
last submission that uses it.

## Rules

R1. **Destroy releases the HAL object.** `Buffer::destroy`, `Texture::destroy`, and
`QuerySet::destroy` drop core's reference to the HAL object. Afterwards `hal()` returns
`None` for that object. The HAL object is freed when the last reference to it goes,
which is when every submission that used it has completed. A second `destroy()` does
nothing.

R2. **Nothing else keeps it.** After `destroy()`, and after every submission that used
the object completes, no core or HAL structure holds a reference to the HAL object. The
coding agent audits every place a `HalBuffer`, `HalTexture`, or `HalQuerySet` (or a
backend object inside one) is cloned and kept beyond one call: caches (views,
framebuffers, render passes, bind groups, pipelines, staging), the queue's pending
writes, lazy-initialization state, map state, and surface code. A structure that keeps
the object past the submissions that use it is a defect of this block. Fix it, for
example by evicting on destroy or by keying on an identity instead of the object. The
audit and its outcome go in the handoff report.

R3. **Work recorded before destroy still runs.** A queue write (`writeBuffer`,
`writeTexture`) issued before `destroy()` and not yet flushed still executes against the
object, as today (`queue_pending_write_keeps_destroyed_destination_alive_until_flush`
and the texture variant). The pending write holds its own reference. Submissions already
made keep theirs.

R4. **No panics, no new errors.** Every core path that reads `hal()` must treat `None`
on a destroyed object as it treats a destroyed object today. Validation rejects using a
destroyed object before the HAL is reached. If another thread destroys the object
between validation and lowering, the path returns an error or skips the work. It must
not panic or touch freed memory. Error messages and the validation order do not change.

R5. **Mapped buffers.** `destroy()` on a mapped buffer, or one with a pending map,
behaves as today: the map is aborted or ended and mapped ranges are no longer
available. Then the HAL buffer is released. Nothing reads or writes through a pointer
into the released buffer afterwards, including the write-back of a `MapWrite` mapping
and `mappedAtCreation` scratch.

R6. **Unchanged.**
- The last-handle-drop path.
- Device destruction and loss.
- Surface textures: `destroy()` on the texture from `getCurrentTexture` does not free
  the swapchain image. It may drop core's reference only if the surface keeps its own.
- External texture objects. Their plane textures follow R8.
- Every backend's behaviour except when memory is freed.
- C API.

R7. **Performance.** `hal()` is on the encode and submit hot paths. Making `inner.hal`
releasable must not add measurable cost there. Use an uncontended lock or an atomic
swap; do not add an allocation per access. Claude checks it with the bench.

R8. **External texture planes.** A bound external texture's plane textures take part in
the pass usage scope as sampled reads, and in the submit-time destroyed check, as Dawn
does: Dawn expands an external-texture binding into plane texture bindings
(`BindGroup.cpp`, `PassResourceUsageTracker.cpp` `AddBindGroup`) and rejects a destroyed
plane at submit (`ExternalTexture.cpp`, `ValidateCanUseInSubmitNow`). A destroyed plane
texture fails submit with "queue submit cannot use a destroyed texture". Without the
check, a destroyed plane would reach lowering with no HAL texture.

## Tests

- Unit (core, Noop):
  - after `destroy()`, `hal()` returns `None` for a buffer, a texture, and a query set;
  - a second `destroy()` is harmless;
  - destroying a mapped buffer, a buffer with a pending map, and a `mappedAtCreation`
    buffer then calling `unmap` and `getMappedRange` does not panic and returns what it
    returns today;
  - a pending queue write to a destroyed buffer and texture still flushes (R3);
  - submitting a command buffer that uses a destroyed object is still the same
    validation error.
- Unit (core, Noop): a submission made before `destroy()` keeps the HAL object alive
  until it completes and frees it afterwards. Observe it through a Noop HAL liveness
  counter or a `Weak`, whichever fits the Noop types.
- HAL (Vulkan, Metal, pure or `#[ignore]`): any cache fixed under R2 has a unit test
  showing that the entry goes when the resource does.
- Claude, Metal e2e: in a loop, create a large buffer and a large texture, use each in
  a submission, `destroy()` them without releasing the handles, and wait. The device's
  `currentAllocatedSize` stays bounded, where today it grows with every iteration.
  Release the handles at the end. Also: destroy while a submission is in flight, then
  read back an unrelated resource from the same submission.
- Claude, Vulkan e2e (MoltenVK under the Khronos layer): the in-flight scenario runs
  with no validation errors.
- Claude: the full Metal e2e and every other ignored test. CTS on Metal:
  - targeted `webgpu:api,validation,buffer,destroy:*`,
    `webgpu:api,validation,buffer,mapping:*`, `webgpu:api,validation,queue,destroyed,*`,
    `webgpu:api,validation,query_set,destroy:*`, `webgpu:api,operation,buffers,*`,
    `webgpu:api,operation,device,lost:*`;
  - then `webgpu:api,*` and `webgpu:shader,execution,*` must not regress.
- Claude: a bench comparison against `b81d1b4` (R7).

## Out of scope

- Freeing memory at device destruction or loss.
- Surface and external textures.
- A memory budget or allocator change.
- Releasing the core-side state of a destroyed object other than the HAL object.

## Slices

- **S1 (coding agent):** R1-R7 with the unit tests and the R2 audit.
- **S2 (Claude):** e2e, CTS, bench, Phase Review.

## Verification (2026-11-05)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| R2 audit (coding agent; confirmed by the Phase Review) | Core holds HAL objects only in `BufferInner`, `TextureInner`, `QuerySetInner`, and the queue's pending writes. The Metal and Vulkan HALs retain every resource a submission references until it completes, and keep no cache keyed on a resource. The one structure that kept storage past completion was the Noop queue's command history; it now holds weak references once a submission completes |
| `cargo test --workspace` | 1,426 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` | clean |
| HAL Metal ignored tests | 66 passed |
| `e2e_metal_destroy_memory` (new) | 2 of 2. `currentAllocatedSize` grows by 80 MiB for the first iteration and by 0 MiB after 16 destroyed iterations. The same test against `8122944` (before this block) keeps 1,280 MiB and fails |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 171 passed, 0 failed |
| All `e2e_vulkan_*`, MoltenVK, Khronos layer | 154 passed, 1 failed (the known `imageView2DOn3DImage` portability case). `e2e_vulkan_destroy_memory` 2 of 2 with no validation output |
| webgpu-native-cts, Metal, targeted | `buffer,destroy` 31 / fail 0; `buffer,mapping` 289 / 0; `queue,destroyed,*` 60 / 0; `query_set,destroy` 2 / 0; `operation,buffers,*` 940 / 0; `device,lost` 1 / 0 |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 589,574, skip 101,793, fail 0, crash 0, on the library before and after the Phase Review fixes; identical to Block 142 |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 857,326, skip 25,860, fail 0, crash 0, xfail 67, xpass 5 (F-160 cases that happened to read zero); pass + xfail identical to Block 142 |
| Bench against `b81d1b4`, three interleaved runs each, then five interleaved runs of the `encode/` cases | every case within run-to-run noise (R7) |

`webgpu:shader,execution,*` and the targeted queries ran before the Phase Review fixes.
Those fixes touch the occlusion-query lowering race, comments, the mapped-range copy in
the C API, and the Noop queue; `webgpu:api,*` and `webgpu:api,validation,buffer,*`
(497 / fail 0) ran again after them.

### Windows native Vulkan (2026-11-05)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`c7d9acd` plus the Windows test-build fix (see below). The Khronos validation layer was on.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,426 passed, 0 failed |
| Vulkan e2e (every integration test, `--ignored`) | 171 passed, 0 failed, 0 VUIDs |
| `e2e_vulkan_destroy_memory` | `vulkan_destroy_releases_memory` and `vulkan_destroy_while_in_flight` pass with no validation output |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | A destroy racing submit could lower a render pass with `BeginOcclusionQuery` and no query set | Lowering fails the pass with "render pass lowering failed" |
| m2 | A queue-write comment said a missing HAL buffer means an error buffer only | Comments state both cases |
| m3 | `wgpuBufferReadMappedRange` and `wgpuBufferWriteMappedRange` copied after dropping the buffer state lock, so a concurrent destroy could free the allocation mid-copy | The copy runs in core under the state lock (`Buffer::copy_from_mapped_range`, `copy_to_mapped_range`) |
| m4 | The spec said external textures and validation were unchanged, but plane textures now join the usage scope and the destroyed check | R8 added |
| m5 | Test gaps: no test of destroy between validation and lowering; the in-flight e2e cannot prove the submission was still running; the Vulkan loop has no memory bound | Race test added through the submit test hook (it fails without m1). The in-flight e2e is documented as a safety regression test. The Vulkan memory bound is declined: there is no portable allocated-size query, and MoltenVK is reference-only |
| m6 | The Noop queue walked its whole command history on every completion check | A retired-up-to cursor per history list |
| m7 | External texture planes are not lazily zero-initialized (not caused by this block) | Backlog A27 |
