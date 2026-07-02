# Block 94 — Immediate data (`SetImmediates`, `maxImmediateSize > 0`)

Full immediates support, closing the remaining Dawn-parity gap found while
attributing the CTS skip delta (Block 93 recorded the parser/validation-only
posture at `maxImmediateSize = 0`; this block supersedes its "out of scope"
section). Dawn baseline: `maxImmediateSize = 64` (`kMaxImmediateDataBytes`,
`Limits.cpp` v1 base tier), delivered to shaders as an *immediates block*
composed of user data plus pipeline-internal constants
(`dawn/native/ImmediatesLayout.{h,cpp}`, `ImmediatesTracker.h`).

## CTS target (Dawn oracle, real Metal M2)

| tree | Dawn | yawgpu today | after |
|---|---|---|---|
| `api,validation,createPipelineLayout` | 115 / 3 | 107 / 11 | 115 / 3 |
| `api,validation,pipeline,immediates` | 30 / 0 | 0 / 30 | 30 / 0 |
| `api,validation,encoding,cmds,setImmediates` | 0 / 3 (port stub) | 0 / 3 | un-stub → both run |
| `api,operation,command_buffer,programmable,immediate` | 0 / 282 (port stub) | 0 / 282 | un-stub → both run |

The two stubbed trees skip on *both* backends today ("setImmediates not
implemented in any backend" — stale: Dawn now ships it); slice 4 un-stubs the
port so the execution path is CTS-verified with Dawn as oracle.

## API surface

Already declared in the vendored `webgpu.h` (nothing to add to the header):

```c
void wgpuComputePassEncoderSetImmediates(enc, uint32_t offset, void const* data, size_t size);
void wgpuRenderPassEncoderSetImmediates(enc, uint32_t offset, void const* data, size_t size);
void wgpuRenderBundleEncoderSetImmediates(enc, uint32_t offset, void const* data, size_t size);
```

## Behaviour contract

- **Limit**: `maxImmediateSize = 64` on Metal and Vulkan adapters and on Noop;
  `Limits::DEFAULT.max_immediate_size` becomes 64 (Dawn's base tier — the
  existing tests asserting 0 update accordingly). GLES (Tier 2) stays 0 and
  the deviation is catalogued in `specs/blocks/67-gles-backend.md` (its
  `tint_immediates` uniform slot is reserved for internal `first_instance`;
  `SetImmediates` on GLES surfaces the usual Tier-2 `HalError` if ever
  reached).
- **Validation of `SetImmediates(offset, data, size)`** — authoritative
  source: Dawn's encoder validation (`ProgrammableEncoder::APISetImmediates`
  / its `ValidateSetImmediates`; read it, do not guess). Expected rules:
  `offset % 4 == 0`, `size % 4 == 0`, no `uint32_t` overflow of
  `offset + size`, and `offset + size <= device.limits.maxImmediateSize`.
  Violations are captured validation errors that invalidate the encoder
  (same routing as the neighbouring encoder-command validation). Zero-size
  writes follow Dawn's behaviour.
- **Semantics**: the pass keeps a 64-byte user-immediate scratch, zero at
  pass begin (Dawn semantics — verify in ImmediatesTracker; if Dawn instead
  makes unwritten ranges undefined, mirror Dawn). `SetImmediates` overwrites
  `[offset, offset+size)`. Contents persist across pipeline changes within
  the pass. Draw/dispatch delivers the *pipeline's* view: user bytes
  `[0, layout.immediate_size)` + internal constants appended after (see
  layout below). Render bundles record the command and replay it into the
  outer pass state, Dawn-equivalent.
- **Pipeline layout / shader rule** stays as Block 93 (entry-point
  `immediate_data_size <= layout.immediate_size`) — now non-vacuous for
  `immediate_size` up to 64, and `createPipelineLayout` enforces
  `immediateSize <= maxImmediateSize` (already implemented; the 8
  `immediate_data_size` CTS cases exercise it once the limit is 64).

## Immediates block layout (per pipeline)

Follow Dawn (`ImmediatesLayout.h`): user immediates first —
`[0, layout.immediate_size)` — internal constants appended after, 4-byte
units. yawgpu's only current internal immediate is the fragment frag-depth
clamp range, which today hardcodes `depth_range_offsets = {0, 4}` in the
MSL path (`tint_shim.cpp` generate_msl) and delivers it via
`setFragmentBytes` at the Tint-chosen `immediate_binding_point` slot
(`encode_render_frag_depth_clamp`). With user immediates the clamp range
moves to `{user_size, user_size + 4}` — the offsets are already computed at
pipeline creation (layout-dependent codegen is the existing pattern), and
the HAL must then deliver ONE combined block (user scratch prefix + clamp
values) instead of the bare 8-byte range. Compute pipelines have no
internal immediates today: the block is just the user prefix.

## HAL contract

- New pass-scoped HAL command (enum-dispatch, all four backends):
  `SetImmediates { offset, data }` recorded into the pass command stream;
  plus the pipeline's immediate metadata (block size, MSL slot / stage
  visibility) threaded from shader codegen exactly like
  `fragment_frag_depth_clamp_slot` is today.
- **Noop**: validates/records, executes as no-op.
- **Metal**: on draw/dispatch (when dirty — pipeline change or
  `SetImmediates` since last delivery), `set{Vertex,Fragment}Bytes` /
  `setBytes` (compute) of the combined block at the pipeline's immediate
  slot, per stage that uses immediates. Replaces (absorbs)
  `encode_render_frag_depth_clamp`'s standalone delivery for pipelines that
  also use user immediates; pipelines with clamp-only keep working.
- **Vulkan**: user+internal immediates become the push-constant block. The
  SPIR-V writer's immediate/push-constant offsets must match the same
  layout (read how Dawn drives `tint::spirv::writer::Options` immediates;
  note whether yawgpu currently wires the Vulkan frag-depth clamp at all —
  F-139 suggests the clamp is Metal-only today — and keep that behaviour
  unchanged apart from the user prefix). `VkPipelineLayout` gains a
  `VkPushConstantRange` covering the block (stage flags per usage);
  `vkCmdPushConstants` on dirty at draw/dispatch. `maxPushConstantsSize >=
  128` (Vulkan minimum) covers 64 user + internal.
- **GLES**: not implemented (limit 0; Tier-2 catalogue row in Block 67).

## Slices

1. **S1 — core + FFI + Noop (Noop-first, no GPU)**: limits (core DEFAULT /
   Noop / Metal / Vulkan HalLimits → 64), `SetImmediates` FFI ×3, core
   encoder state + validation + render-bundle record/replay, HAL command
   enum + Noop no-op, unit tests at every layer. CTS check (validation-only
   trees): `createPipelineLayout` 115/3, `pipeline,immediates` 30/0.
2. **S2 — Metal execution**: block composition (user prefix + clamp),
   MSL `depth_range_offsets` rebase, encoder delivery + dirty tracking,
   real-Metal e2e (compute readback of `var<immediate>` + render clamp
   regression).
3. **S3 — Vulkan execution**: SPIR-V/push-constant path,
   `VkPushConstantRange`, MoltenVK e2e + clamp/F-139 posture unchanged.
4. **S4 — CTS port un-stub** (webgpu-native-cts repo): implement the stubbed
   bodies of `encoding/cmds/setImmediates.spec.cpp` (3) and
   `command_buffer/programmable/immediate.spec.cpp` (282) per the upstream
   `.spec.ts`; verify Dawn green first (oracle), then yawgpu parity.
5. Phase Review (whole block diff), tracking + docs updates.

## Verification

- Unit tests per Block 90 at each public fn; all Noop-green.
- Real-GPU: Metal e2e (S2), MoltenVK e2e (S3); Windows native-Vulkan
  deferred to the next user-run sweep.
- CTS (Metal, vs Dawn same queries): the four trees above + regression
  spots `compute_pipeline:*`, `render_pipeline,misc:*`,
  `rendering,depth_clip_clamp:*` (clamp path touched), `limits,*`
  (`maxImmediateSize` now non-zero).
