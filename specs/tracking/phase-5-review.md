# Phase 5 Review — Clean Review Then Fix

Process: `../reference/workflow.md` → "Phase Review". Fresh no-context
reviewer over the cumulative Phase-5 diff (`3394593..79d1364`) + block
40 + the W4/W5 carried notes. Result: **0 CRITICAL, 6 MAJOR, 5 MINOR**.

Reviewer verified sound (no finding): no memory-unsafety / FFI ABI
mismatch / refcount double-free / use-after-free / dangling-cache-handle;
`Arc::into_raw`/`from_raw` + permanent-Arc cache sound; async
`PendingCallback` once-only with correct Arc/`'static`-message
lifetimes; new `unsafe extern "C"` signatures match `webgpu.h`.

## Triage + disposition

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| J1 | MAJOR | `stage_resource_bindings` returns *all* module bindings tagged with the pipeline stage ⇒ single combined vertex+fragment module mis-attributes visibility (wrong P39/P42 auto-layout) & wrongly requires Vertex for fragment-only bindings (S35). Vacuous (tests use separate modules). | **FIX (codex)**: filter bindings to those used by the *selected entry point* (per-entry-point reflection); add a combined-module test. |
| J2 | MAJOR | `validate_shader_binding_compat` accepts any Texture/Sampler/StorageTexture pairing — S35 type check only real for buffers. | **FIX (codex)**: compare reflected sample-usage/view-dimension/multisampled/storage-format/access vs the layout entry kind; add tests. |
| J3 | MAJOR | `Create*PipelineAsync` runs sync validation which `dispatch_error`s a device error; Dawn reports async failure **only** via callback `ValidationError`, no device error. Test enshrines the divergence. | **FIX (codex code+test)**: async path validates WITHOUT dispatching a device error (deliver solely via callback); + Claude records the canonical behavior in block 40. |
| J4 | MAJOR | `workgroup_storage_size` `u32 .sum()` panics(debug)/wraps(release) on large valid WGSL (P3 vacuous) and sums whole module not the entry. | **FIX (codex)**: `u64` checked accumulation, error on overflow, scope to entry point. |
| J5 | MAJOR | `sampled_texture_usage` scans only the entry function ⇒ texture sampled via a helper fn ⇒ wrong `UnfilterableFloat` (P42 wrong default BGL). | **FIX (codex)**: also scan `module.functions[*]` (call graph) for `ImageSample`. |
| J6 | MAJOR | `TextureFormat::caps()` `_ =>` returns fully renderable/blendable for any unknown defined format ⇒ P29/P30/P32 vacuous for invalid color formats (carried W5 now a real Phase-5 gap). | **FIX (codex)**: unknown defined format ⇒ `None` (reject) so P29/P32 enforce; **resolves the W5 carried note** (Claude updates the note). |
| m1 | MINOR | `*Pipeline::new` re-resolves with `Limits::DEFAULT` not device limits (latent; equal for Noop). | **FIX (codex)**: thread device `Limits` into `new()`. |
| m2 | MINOR | block 40 design text claims `BindGroupLayout` dedup; not implemented (no P-rule requires it). | **SPEC (Claude)**: amend block 40 to record BGL-dedup is intentionally NOT implemented. |
| m3 | MINOR | cache keys `f64::to_bits` distinguish ±0.0 / NaN encodings ⇒ benign missed cache hit. | **FIX (codex)**: normalize `0.0`/`-0.0` (canonicalize) before `to_bits`. |
| m4 | MINOR | `SHADER_FLOAT16` enabled unconditionally (no device-feature gate) — divergence. | **SPEC (Claude)**: record the divergence (yawgpu has no canonical ShaderF16-feature toggle path; P5.0/W4). |
| m5 | MINOR | `get_pipeline_bind_group_layout` `usize::try_from` failure returns error handle without `dispatch_error` (unreachable on 64-bit). | **FIX (codex)**: dispatch the same validation error in both branches. |

No false positives. **Gate: Phase 5 cannot be COMPLETE while any of
J1–J6 (MAJOR) is open.** m1/m3/m5 fixed in the same codex pass; m2/m4
are Claude spec-side (this commit).

## Resolution log

- _filled as fixes land_
