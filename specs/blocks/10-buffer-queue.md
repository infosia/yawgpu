# Block 10 — Buffer + Queue

Phase 2. Rules extracted from Dawn `BufferValidationTests`,
`QueueWriteBufferValidationTests`, `QueueOnSubmittedWorkDoneValidationTests`,
`QueueSubmitValidationTests`, `WriteBufferTests`,
`MinimumBufferSizeValidationTests` (under
`dawn/src/dawn/tests/unittests/validation/`). Status: ☐ ◐ ☑ ✗(N/A).
"Defer→Px" = needs a later-phase resource.

## Surface (webgpu.h)

- `WGPUBufferDescriptor` (1911: nextInChain,label,usage,size,
  mappedAtCreation), `WGPUBufferUsage` (1270: MapRead1 MapWrite2 CopySrc4
  CopyDst8 Index16 Vertex32 Uniform64 Storage128 Indirect256 QueryResolve512),
  `WGPUMapMode` (1338: None0 Read1 Write2), `WGPUBufferMapState` (446:
  Unmapped1 Pending2 Mapped3), `WGPUMapAsyncStatus` (727: Success1
  CallbackCancelled2 Error3 Aborted4), `WGPUQueueWorkDoneStatus` (842:
  Success1 CallbackCancelled2 Error3).
- `wgpuDeviceCreateBuffer` 6281; `wgpuBufferMapAsync` 6119
  (+`WGPUBufferMapCallbackInfo` 1533) → `WGPUFuture`;
  `wgpuBufferGetMappedRange` 6104 / `wgpuBufferGetConstMappedRange` 6087;
  `wgpuBufferUnmap` 6142; `wgpuBufferDestroy` 6069; `wgpuBufferGetSize`
  6106; `wgpuBufferGetUsage` 6107; `wgpuBufferGetMapState` 6105.
- `wgpuQueueSubmit` 6464; `wgpuQueueWriteBuffer` 6469;
  `wgpuQueueOnSubmittedWorkDone` 6462 (+`WGPUQueueWorkDoneCallbackInfo`
  1677) → `WGPUFuture`.

## Buffer map state machine

`Unmapped → MapAsync(sync-validate ok) → Pending → (callback Success) →
Mapped → Unmap → Unmapped`. `Unmap`/`Destroy`/last-ref-drop while
`Pending` ⇒ callback fires `Aborted`, state → `Unmapped`/destroyed.
`mappedAtCreation=true` ⇒ starts `Mapped` (skips Pending; any usage).
`MapAsync` on `Mapped`/`Pending`/destroyed ⇒ validation error + callback
`Error`. Map callback uses the **Phase-1 future/`PendingCallback`
machinery** (callback mode A2–A6); reuse it, do not fork dispatch. Callback
fires **exactly once**; `Unmap`/`Destroy` inside the callback must not
re-fire.

## Divergences (recorded)

- Dawn callback message substrings ("unmapped"/"destroyed") are
  Dawn-internal. yawgpu ported tests assert **status codes**
  (`WGPUMapAsyncStatus_*` / `WGPUQueueWorkDoneStatus_*`) and empty-vs-
  nonempty message only — never exact Dawn strings.
- `GetMappedRange` misuse returns **NULL** (no panic, no device-error);
  matches canonical pointer-returning semantics (Dawn additionally emits a
  device log we do not require).
- `WGPU_WHOLE_MAP_SIZE` sentinel handled per webgpu.h.
- **Buffer descriptor validation is first-match-wins** (B1–B6): a
  multi-rule-violating descriptor dispatches exactly **one** device error
  (Dawn does too). Tests assert "exactly one error" (not a rule-specific
  message, per the status-code divergence); a multi-violation vector
  proves the single-error contract independent of check order. Order is an
  implementation detail, not a spec guarantee.

## Rules

### Buffer creation / reflection / lifetime — Phase 2 (P2.1)

- **B1** usage must be non-zero. `CreationMapUsageNotZero` :115. ☑ (P2.1)
- **B2** `MapRead` may only combine with `CopyDst`. `CreationMapUsage
  Restrictions` :127. ☑ (P2.1)
- **B3** `MapWrite` may only combine with `CopySrc`. same :146. ☑ (P2.1)
- **B4** `size > device maxBufferSize` ⇒ error. `CreationMaxBufferSize`
  :93. ☑ (P2.1) (reuse P1.2a `Limits.max_buffer_size`)
- **B5** `mappedAtCreation` requires `size` 4-byte aligned.
  `MappedAtCreationSizeAlignment` :622. ☑ (P2.1)
- **B6** `mappedAtCreation=true` ⇒ buffer starts `Mapped`, any usage ok.
  `NonMappableMappedAtCreationSuccess` :618. ☑ (P2.1)
- **B32** `Destroy` valid on any buffer incl. error buffer.
  `DestroyErrorBuffer` :663. ☑ (P2.1)
- **B33** `Destroy` idempotent. `DestroyDestroyedBuffer` :680. ☑ (P2.1)
- **B34** `Unmap` on unmapped buffer = safe no-op. `UnmapUnmappedBuffer`
  :834. ☑ (P2.1)
- **B35** `Unmap` on error buffer = safe. `UnmapErrorBuffer` :687. ☑ (P2.1)
- **B36** `Unmap` on destroyed buffer = safe. `UnmapDestroyedBuffer` :698.
  ☑ (P2.1)
- **B37** `MapAsync` after `Destroy`+`Unmap` ⇒ error + callback `Error`.
  `MapDestroyedBufferAfterUnmap` :706. ☑ (P2.2) (callback part may land in P2.2)
- **B38** `GetSize`/`GetUsage` reflect descriptor even on error buffer;
  `GetMapState` queryable. `CreationParameterReflectionForErrorBuffer`
  :1135. ☑ (P2.1)

### Buffer map async — Phase 2 (P2.2)

- **B7/B8** `MapAsync` mode must match usage (Read⇒MapRead, Write⇒
  MapWrite). `MapAsync_WrongUsage` :339. ☑ (P2.2)
- **B9** offset 8-byte aligned. `MapAsync_OffsetSizeAlignment` :262. ☑ (P2.2)
- **B10** size 4-byte aligned. same :281. ☑ (P2.2)
- **B11** offset(+size) within bounds incl. overflow.
  `MapAsync_OffsetSizeOOB` :289. ☑ (P2.2)
- **B12** error if already Mapped (incl. mappedAtCreation).
  `MapAsync_AlreadyMapped` :368. ☑ (P2.2)
- **B13** mode must be exactly Read xor Write (not None, not both).
  `MapAsync_WrongMode` :356. ☑ (P2.2)
- **B14** unsupported mode bits ⇒ error. `MapAsync_UnsupportedMode` :250.
  ☑ (P2.2)
- **B15** on destroyed buffer ⇒ error. `MapAsync_Destroy` :422. ☑ (P2.2)
- **B16** second `MapAsync` while Pending ⇒ error (overlapping AND
  non-overlapping). `MapAsync_PendingMap` :387. ☑ (P2.2)
- **B17** valid `MapAsync` ⇒ state Pending, sync call ok. `MapAsync_
  Success` :228. ☑ (P2.2)
- **B18** completion ⇒ callback `Success`, state Mapped. same. ☑ (P2.2)
- **B19** `Unmap` before result ⇒ callback `Aborted`.
  `MapAsync_UnmapBeforeResult` :429. ☑ (P2.2)
- **B20** `Destroy` before result ⇒ callback `Aborted`.
  `MapAsync_DestroyBeforeResult` :461. ☑ (P2.2)
- **B21** last-ref drop before result ⇒ callback `Aborted`.
  `MapAsync_DroppedBeforeResult` :473. ☑ (P2.2)
- **B22** validation failure ⇒ callback `Error` (retry from callback ok).
  `MapAsync_RetryInErrorCallback` :524. ☑ (P2.2)
- **B23** `Unmap`/`Destroy` inside callback ⇒ no double-fire.
  `MapAsync_UnmapCalledInCallback` :485. ☑ (P2.2)
- **B24** callback honours `WGPUCallbackMode` (A2–A6 machinery). same
  :234. ☑ (P2.2)

### GetMappedRange — Phase 2 (P2.3)

- **B25** NULL on unmapped (incl. after unmap of mappedAtCreation).
  `GetMappedRange_OnUnmappedBuffer` :850. ☑ (P2.3)
- **B26** non-const `GetMappedRange` NULL when mapped read-only;
  `GetConstMappedRange` ok. `..._NonConstOnMappedForReading` :929. ☑ (P2.3)
- **B27** offset beyond mapped range ⇒ NULL. `..._OffsetSizeOOB` :996. ☑ (P2.3)
- **B28** offset+size beyond mapped range (incl. overflow) ⇒ NULL. same
  :1067. ☑ (P2.3)
- **B29** `WGPU_WHOLE_MAP_SIZE` uses mapped size not buffer size. same
  :1087. ☑ (P2.3)
- **B30** offset before mapped-range start ⇒ NULL. same :1096. ☑ (P2.3)
- **B31** NULL on destroyed buffer. `GetMappedRange_OnDestroyedBuffer`
  :914. ☑ (P2.3)

### Queue — Phase 2 (P2.4)

- **B42** `wgpuQueueWriteBuffer` requires `CopyDst`. `WrongUsage` :79. ☑ (P2.4)
- **B43** size 4-byte aligned. `UnalignedSize` :90. ☑ (P2.4)
- **B44** offset 4-byte aligned. `UnalignedOffset` :98. ☑ (P2.4)
- **B45** offset+size within bounds. `OutOfBounds` :58. ☑ (P2.4)
- **B46** overflow detection. `OutOfBoundsOverflow` :66. ☑ (P2.4)
- **B47** fails if buffer mapped (mappedAtCreation or MapAsync).
  `MappedBuffer` :115. ☑ (P2.4)
- **B48** fails if buffer destroyed. `DestroyedBuffer` :106. ☑ (P2.4)
- **B49** success path. `Success` :50. ☑ (P2.4)
- **B50/B51** `wgpuQueueOnSubmittedWorkDone` callback fires `Success`
  (even before any submit). `CallBeforeSubmits` :46. ☑ (P2.4)
- **B52** OnSubmittedWorkDone honours `WGPUCallbackMode`. same. ☑ (P2.4)
- `wgpuQueueSubmit` **top-level pointer** validation only in P2.4
  (`commandCount>0 && commands==NULL` ⇒ device error; `commandCount==0`
  ⇒ no-op). Per-element/`CommandBuffer` validation requires the
  `CommandBuffer` type — Defer→P6 (`QueueSubmitValidationTests` full port
  is P6).

### Deferred

- **B39/B40/B41** submit with mapped/destroyed buffer — needs
  `CommandBuffer`. Defer→P6.
- **B53–B57** `CommandEncoder.WriteBuffer` — needs `CommandEncoder`.
  Defer→P6.
- **B58** `MinimumBufferSizeValidationTests` — needs `BindGroupLayout`.
  Defer→P4.

## Open questions

- Noop buffer storage: back `GetMappedRange` with a real host
  `Vec<u8>`/allocation so pointer/bounds semantics are testable; track map
  range [offset,size) per map.
- `WGPUBufferMapState` mapping; `wgpuBufferGetMapState` returns live state.
- Reuse `Limits.max_buffer_size` (P1.2a) for B4.

## CTS findings — buffer mapping (2026-06-10, batch Y-2)

- **F-072 — Metal zero-size buffers/maps.** Mapping a zero-size buffer or a
  zero-length range is valid WebGPU (Dawn and yawgpu-Vulkan accept it). Metal's
  `newBufferWithLength(0)` returns **nil**, which (post-F-065) surfaces as a
  spurious `OutOfMemory`. Rule: the Metal HAL allocates `max(size, 1)` bytes and
  keeps the **logical** size at the requested value (mirrors wgpu); all
  map/read/write bounds validate against the logical size. CTS:
  `api,operation,buffers,map` Metal `fail=0`.
- **F-073 — mappedAtCreation OOM must not abort.** `wgpuDeviceCreateBuffer`
  with `mappedAtCreation=true` and an impossible size (~9 PB) must not panic:
  the host-side mapping allocation is **fallible** (`try_reserve`-style, never
  an infallible `Vec::with_capacity(size)`), an error/unallocatable buffer
  performs **no** host allocation, and `getMappedRange` on it returns null.
  No process abort under any descriptor. CTS: `api,operation,buffers,map_oom`
  both HALs, no crash.
