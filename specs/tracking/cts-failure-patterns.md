# CTS failure-pattern retrospective

A meta-analysis of every resolved WebGPU-CTS finding (F-005..F-043) plus the two proactive audits
([[threading-audit]], [[narrowing-audit]]), cross-checking each finding's stated diagnosis (the user's
`FINDINGS.md`) against the actual fix commit's diff. Goal: name the recurring root-cause patterns so the
remaining instances can be swept proactively and recurrence prevented.

## The dominant pattern: "validate but don't act" (silently-wrong family)

~half of all findings. A WebGPU op/field passes `yawgpu-core` validation but the execution layer drops or
degrades it. The unifying root cause is a gap at the **core→HAL→backend seam** (the single biggest fix
locus: ~16 of 33 fix commits). Three sub-forms:

| Sub-form | Meaning | Findings |
|---|---|---|
| **1a validated-but-not-executed** | op recorded but emits NO HAL command → silent no-op | F-023 (clearBuffer), F-025 (writeTexture), F-034 (indexed/indirect draw), F-042 (render bundle) |
| **1b validated-but-not-threaded** | field accepted but value never reaches the backend → default used | F-035 (blend/writeMask), F-038 (stencil ref), F-041 (storage-texture binding), F-043 (depthSlice) + threading audit (10) |
| **1c lossy-narrowing** | value reaches the backend but degraded to a less-informative representation | F-037 (point size), F-032 (copy size) + narrowing audit (5, incl. Vulkan integer clear) |

**Why they all passed the gates:** the tests exercise only DEFAULT values (cull=none, blend off, slice 0,
float formats, no draw), so both validation and the Noop "execution" pass while recording the default. The
CTS caught them because it exercises NON-DEFAULT values on REAL hardware. Systemic mitigation: assert the
GPU effect of a non-default value on a real backend — which is exactly what the audits + e2e probes added.

## Other recurring patterns

- **Over-strict validation (false-reject of spec-legal ops)** — ~9: F-005, F-009, F-011, F-016, F-018,
  F-022, F-039, F-040, F-042. Amplified by **encoder poisoning**: one `record_first_error` marks the whole
  command buffer as error → `finish()` errors → `submit` drops everything, so the symptom looks like
  "nothing ran" rather than "one op rejected" (F-039, F-042). Sub-themes: validation TIMING (F-022,
  minBindingSize deferred to bind time) and over-aggressive feature gates (F-040, MSAA/MRT).
- **Backend doesn't implement a validated path** — ~5: F-026, F-031, F-032, F-040, F-041. Backend hardcodes
  a default / has no path (depth-stencil render path, depth/stencil aspect copies, MSAA resolve,
  multi-layer). Often Metal-first, then Vulkan needs the same fix.
- **Format-table gaps** — ~5: F-006, F-016, F-024 (+ components). A format missing/misclassified in the
  capability table → `Unsupported` → silently dropped copy or false-reject.
- **Sync / lifetime** — 2: F-029 (in-flight resources freed before fence), F-030 (MAP_READ before copy
  completes).
- **Crash/abort on valid input** — 2: F-005 (Depth24PlusStencil8 → nil Metal format), F-023 (0-size blit
  abort).

## Cross-cutting meta-observations

- **Cross-HAL identical symptom ⇒ shared core→HAL seam.** Every 1b finding manifested identically on Metal
  AND Vulkan — the diagnostic signature that the bug is in the shared layer, not a backend. A single-backend
  symptom (F-023, F-037, F-031) points to a backend-specific gap.
- **Symptom ≠ root cause (recurring).** 6 findings had a misleading symptom: F-037 ("race/sync" → MSL
  codegen), F-031 ("copy bug" → whole depth render path missing), F-039 ("contamination" → encoder
  poisoning), F-025 ("bytesPerRow" → Metal swallowed an Err into a null husk), F-023 ("optimization" → Metal
  abort), `e56f30a` (`WHOLE_SIZE` sentinel unresolved at the FFI layer). Localize via the device-error sink +
  code-path instrumentation, not the symptom.
- **Maturity drift.** Early findings (F-005..F-022, the createTexture/View/BGL phases) were VALIDATION bugs
  (both over- and under-strict); later findings (F-025..F-043) shifted to EXECUTION bugs (threading /
  backend). Once validation became conformant, the gaps moved downstream.

## Audit status — ALL SWEPT

- **1b** threading audit — commit `de4a99f` / `f82c2d6`.
- **1c** narrowing audit — commit `73dbf38`. Ledger `narrowing-audit.md`.
- **A — execution-gap (1a)** — commit `7f42d19`. Ledger `execution-gap-audit.md`. Found the SEVERE multi-draw
  re-clear (a render pass with ≥2 draws re-cleared/discarded between draws — only the last survived) + GLES
  render texture/sampler bindings silently ignored. (Documented-deferred: `wgpuCommandEncoderWriteBuffer`
  stub, occlusion/timestamp queries.)
- **B — over-strict validation** — commit `adc6b97`. Ledger `overstrict-audit.md`. Fixed vertex-buffer draw
  OOB (`lastStride`), storage `2d-array` layer false-reject, 3D same-texture disjoint-z copy false-reject.
  (Encoder-poisoning confirmed spec-correct.)
- **C — format-table completeness** — commit `6e288d6`. Ledger `format-completeness-audit.md`. Table was
  complete (0 missing); fixed rgb10a2unorm `has_alpha`; IMPLEMENTED BC/ETC2/ASTC compressed formats with
  per-device feature gating (was advertised-but-unbacked). (Deferred: GLES + sliced-3d compressed
  advertisement; tier2 read-write-storage breadth.)

All real-GPU verified on Metal + Vulkan/MoltenVK; each had a Clean Review with no open CRITICAL/MAJOR.
