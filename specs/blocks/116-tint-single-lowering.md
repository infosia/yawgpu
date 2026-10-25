# Block 116 — Tint: one IR lowering per resolved workgroup query

Status: **COMPLETE (2026-10-25)** — implementation + Phase Review fix in one
commit.
Source: Block 111 Phase Review m3 (deferred), `specs/tracking/code-review-2026-10.md`
("Tint double IR lowering (perf)").

## Problem

`ShaderModule::resolved_compute_workgroup_size` (`yawgpu-core/src/shader_tint.rs`)
handles a module that declares overrides with two shim calls for the same
`(entry point, overrides)`:

1. `Program::resolved_workgroup_info` → `yawgpu_tint_resolved_workgroup_size`
2. `Program::workgroup_storage_size` → `yawgpu_tint_workgroup_storage_size`

Each call runs `lower_ir` → `SingleEntryPoint(ep)` →
`SubstituteOverrides(ov)` on a fresh IR module, so every compute-pipeline
creation that misses the memoisation cache lowers the whole program to IR
twice. It is compile-time cost only (results are memoised per entry point and
constants), with no correctness impact.

## Rule

- `yawgpu_tint_resolved_workgroup_size` gains a `uint64_t* storage_size` out
  parameter: after reading the workgroup / subgroup size off the
  override-substituted IR, it runs `tint::core::ir::GetWorkgroupInfo` on the
  **same** IR and reports `storage_size`. Failure of `GetWorkgroupInfo` fails
  the call with `*err` (the two-call path already failed in that case, so the
  observable result does not change). The pointer must be non-NULL, like the
  other out parameters; it is zeroed on entry.
- `yawgpu_tint::ResolvedWorkgroupInfo` gains `workgroup_storage_size: u64`;
  the stub (non-Tint) build stays `Err(Unavailable)`.
- `resolved_compute_workgroup_size` takes the storage size from that info and
  no longer calls `workgroup_storage_size` on the override path. The
  override-free fast path (`compute_workgroup_size`) is unchanged: it lowers
  once already.
- `Program::workgroup_storage_size` and its shim entry point stay (used by the
  fast path and their unit tests).

## Tests

- `yawgpu-tint`: `resolved_workgroup_info` reports the storage size, matching
  `workgroup_storage_size`, for (a) a literal-size module, (b) a module whose
  `var<workgroup>` array length is an override, with and without an override
  value, (c) a two-entry-point module (per-entry-point measurement, Block 111
  R7 shape).
- `yawgpu-core`: existing `resolves_workgroup_storage_size_from_overrides_*`
  and R7 limit tests stay green unchanged.

## Gates

`cargo test --workspace`, strict clippy (default + `metal` + `vulkan` +
`gles`), fmt; Metal compute e2e on the M2 (Claude).

## Phase Review (2026-10-25)

Fresh reviewer over the working-tree diff: **0 CRITICAL / 0 MAJOR / 3 MINOR**.
Behavioural equivalence confirmed: the merged shim runs the same transforms in
the same order as the old two calls, and `GetWorkgroupInfo`'s only failure
(no entry point with a constant workgroup size) is unreachable once
`WorkgroupSizeAsConst` has succeeded on the single-entry-point IR, so no new
error path exists. Gates: `cargo test --workspace` 1231/0, strict clippy
(default / `metal` / `vulkan` / `gles` / `metal,tiled`) + fmt clean; Metal
compute / subgroups / f16 / immediates e2e on the M2 16/0.

| ID | Finding | Disposition |
|---|---|---|
| m1 | `resolved_workgroup_size` (size-only wrapper, test-only callers) now also runs `GetWorkgroupInfo` | accepted: cannot fail after the size read, read-only |
| m2 | `ResolvedWorkgroupInfo` lacks `#[non_exhaustive]` (CLAUDE.md convention) | fixed |
| m3 | no test proves the override path makes one shim call | accepted: the core path has a single call site by construction; no shim-call counter exists to assert against |
