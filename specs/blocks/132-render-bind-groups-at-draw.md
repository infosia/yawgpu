# Block 132 — Render bind groups are resolved at draw time

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 5 MINOR; m1-m4 fixed,
m5 needed no change, see below. Metal is owed on a Mac host.
Source: backlog A22, found in the Windows native Vulkan CTS run of Blocks 130 and 131.
Oracle: Dawn at the pinned `third_party/dawn`: `native/BindGroupTracker.h`
(`BindGroupTrackerBase::OnSetBindGroup`, `OnSetPipeline`, `BeforeApply`: bind groups
are recorded as dirty and applied before each draw or dispatch) and
`native/vulkan/CommandBufferVk.cpp` (the tracker's `Apply` before each draw).

## Problem

Render lowering (`hal_render_commands` in `yawgpu-core/src/queue.rs`) resolves each
bind group against the current pipeline as soon as it sees the command:
- `SetBindGroup` with a pipeline set calls `hal_set_bind_group(pipeline, ..)`.
- `SetPipeline` calls `hal_set_bind_group` again for every bind group bound so far.

`hal_set_bind_group` goes through `hal_bind_group_resources`, which returns `None` when
a binding the pipeline declares is missing from the group, or when the resource kinds
do not match. That `None` fails the whole pass.

That pairing is legal WebGPU. Validation only requires compatibility at a draw.
Two valid command orders reach it:
1. `setPipeline(A)`, `setBindGroup(0, groupForB)`, `setPipeline(B)`, `draw`. The group
   is resolved against A before B replaces it.
2. `setPipeline(A)`, `setBindGroup(0, incompatibleGroup)`, no draw, `end`.

Before Block 130 the failed pass was dropped silently, so its draws never ran. Since
Block 130 R1 the whole submit fails with "render pass lowering failed". On Windows native
Vulkan, CTS `api,validation,encoding,programmable,pipeline_bind_group_compat:*` gives 432
failures (`bgl_binding_mismatch` 64, `bgl_resource_type_mismatch` 368), all render pass
and render bundle cases. The code is in core, so every backend is affected.

Compute lowering already does the right thing. Each dispatch carries its own pipeline
and bind-group snapshot, and `hal_compute_pass_execution` resolves them together at the
dispatch. Compute passes pass the same CTS cases.

## Rules

R1. **Bind groups are resolved at draws only.** In `hal_render_commands`, `SetBindGroup`
with a group and `SetPipeline` record lowering state; neither resolves a bind group.
Before each `Draw` command is lowered, every bound group index that is dirty is resolved
against the pipeline current at that draw. Each resolution emits one
`HalRenderPassCommand::SetBindGroup`, after the latest `SetPipeline` and before the
`Draw`. A group index is dirty when its group changed since the last draw, or when the
pipeline changed since the last draw. A draw with no current pipeline fails lowering
(`None`); before this block it lowered without error. Validation rejects such a draw,
so it is unreachable for valid command buffers.

R2. **A pairing that never reaches a draw is never resolved.** Such a pass lowers
successfully. With R1, a resolution failure at a draw is still `None`, and Block 130 R1
still aborts the submit. Validation makes that unreachable for valid objects.

R3. **Unchanged commands.**
- `SetPipeline` is still emitted where it occurs.
- `SetBindGroup` with no group (unbind) still emits the empty `SetBindGroup` where it
  occurs when a pipeline is set, as before, and clears that index's dirty mark.
- Vertex buffers, index buffers, and every other command are unchanged.
- A draw with no change since the previous draw emits no `SetBindGroup`.

R4. **Render bundles.** `ExecuteRenderBundle` lowers the bundle's commands with the
same rules and its own fresh state, as today. After the bundle, the pass's state is
reset as today: pipeline, bind groups, and vertex buffers are cleared.

R5. **Compute and the tiled subpass path are unchanged.** Compute already resolves per
dispatch. The subpass path carries bind groups per draw.

R6. **HAL backends are unchanged.** Every backend already accepts `SetBindGroup` at any
point after `SetPipeline`. Moving it next to the draw changes only where it is
recorded, not what is bound at the draw.

## Tests

- Unit (core), each through `hal_render_pass_execution` or `queue.submit`:
  - Order 2 from the Problem: lowering succeeds, the stream has the `SetPipeline` and no
    `SetBindGroup`, and the submit returns no error. Red before the fix.
  - Order 1: lowering succeeds, and the stream is `SetPipeline(A)`, `SetPipeline(B)`,
    `SetBindGroup` with B's resources, `Draw`. Red before the fix.
  - Two draws with no change between them: one `SetBindGroup` per bound group, before
    the first draw only.
  - A `SetBindGroup` between two draws: re-emitted before the second draw only for that
    index.
  - A pipeline change between two draws: every bound group is re-emitted against the new
    pipeline.
  - A render bundle containing order 1 lowers the same way inside the bundle.
  - The existing render-lowering and Block 130 tests still pass. Update any that assert
    the old emission order, and say which in the report.
- FFI (`yawgpu` crate): order 2 through the C API submits with no device error.
- e2e (Claude), Metal and Vulkan: order 1 renders with group B's data, read back from
  the target.
- CTS red check (Claude): `pipeline_bind_group_compat:*` on Windows native Vulkan. Before:
  432 failures. After: 0.

## Out of scope

- Skipping `SetBindGroup` for groups that did not change across a pipeline change with a
  compatible layout prefix, as Dawn does. That is a later performance change.
- Deferring vertex buffer resolution.

## Slices

- **S1 (coding agent):** R1-R6 with the core and FFI unit tests.
- **S2 (Claude):** e2e, red check, Vulkan e2e and CTS on Windows native Vulkan, Phase
  Review.

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The
Khronos validation layer was on for the e2e and HAL runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,363 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace, all targets) and `vulkan` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, all targets) | clean |
| Core and FFI red tests (coding agent) | the two Problem orders and the FFI order-2 test failed before the change: lowering `None`, and "render pass lowering failed" at submit |
| e2e `group_set_under_another_pipeline_draws_with_the_next`, Vulkan | before: fails with "render pass lowering failed"; after: passes, both halves read back with their group's colour |
| Vulkan HAL ignored tests | 78 passed, 0 failed, 0 VUIDs |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 163 passed, 0 failed, 0 VUIDs |
| Red check: CTS `api,validation,encoding,programmable,pipeline_bind_group_compat:*` | before (`f3a63e9`): 432 failures; after: pass 2,520, fail 0, twice |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt` | `fail=3 crash=0`; see below |

| Area | pass | skip | fail | xfail |
|---|---:|---:|---:|---:|
| `api,validation` | 246,484 | 111,283 | 0 | 4 |
| `api,operation` | 180,779 | 152,819 | 0 | 0 |
| `shader,execution` | 562,547 | 320,680 | 1 | 25 |
| `shader,validation` | 662,535 | 20,299 | 2 | 0 |

`api,validation` is back to the `019bd3b` counts. The 2 `shader,validation` failures are
the Block 119 R9 cases. The `shader,execution` failure is
`memory_model,weak:store_buffer` on `atomic_storage` (3 weak results). It is the
intermittent storage-atomics behaviour of this driver recorded in Block 120:
`memory_model,weak:*` then passed 12/12 in each of 3 runs.

The GPU runs and the CTS used the library before the m2 and m4 test fixes. Those change
only `#[cfg(test)]` code; `cargo test --workspace` above ran after them.

Not exercised: Metal. `e2e_metal_render_bind_groups_at_draw` and the CTS red check are
owed to a Mac host.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| — | Every HAL backend keeps bind state independent of the pipeline and applies it at the draw, so moving `SetBindGroup` next to the draw is safe | Confirmed by the review: Vulkan writes descriptors per draw, Metal re-emits every group after a pipeline change, GLES accumulates state |
| m1 | R3 said every other command is unchanged, but a draw with no pipeline now fails lowering | R1 states it |
| m2 | The bundle test did not prove the outer bind groups are cleared after the bundle | A draw after the final `SetPipeline` now asserts no `SetBindGroup` is emitted |
| m3 | R3's unbind wording omitted "when a pipeline is set" | Corrected |
| m4 | Tests indexed `buffers[0]` inside `matches!` guards | Length checked first |
| m5 | The e2e is sufficient; an indexed draw is optional | No change |
