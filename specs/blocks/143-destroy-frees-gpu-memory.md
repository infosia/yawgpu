# Block 143 — `destroy()` frees GPU memory

Status: **IN PROGRESS**.
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
- External textures.
- Every backend's behaviour except when memory is freed.
- C API.

R7. **Performance.** `hal()` is on the encode and submit hot paths. Making `inner.hal`
releasable must not add measurable cost there. Use an uncontended lock or an atomic
swap; do not add an allocation per access. Claude checks it with the bench.

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
  - targeted `webgpu:api,validation,state,device_lost,destroy:*`,
    `webgpu:api,validation,buffer,destroy:*`, `webgpu:api,validation,texture,destroy:*`,
    `webgpu:api,validation,query_set,destroy:*`, `webgpu:api,operation,buffers,map:*`;
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
