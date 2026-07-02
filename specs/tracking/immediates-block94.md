# Block 94 — full immediates support: tracking

Spec: `specs/blocks/94-immediates.md`. Goal: close the last known CTS
coverage delta vs Dawn (`createPipelineLayout` −8, then the discovered
`pipeline,immediates` −30) by implementing `wgpu*SetImmediates` +
`maxImmediateSize = 64` end-to-end, and CTS-verify the execution path by
un-stubbing the port's immediates trees.

| Slice | Scope | Status |
|---|---|---|
| S1 | core + FFI + Noop: SetImmediates ×3, pass scratch + validation (Dawn ProgrammableEncoder parity), bundle record/replay, Noop limit 64 | **DONE** `4db12e3` |
| S2 | Metal execution: limit 64, MSL user_immediate_size + clamp-offset rebase, combined-block setBytes delivery, e2e ×3 | **DONE** `75272ba` |
| S3 | Vulkan execution: limit 64 + core `Limits::DEFAULT` 64, SPIR-V rebase, push-constant combined block, MoltenVK e2e ×3 | **DONE** `9eeeba7` |
| S4 | CTS port un-stub (webgpu-native-cts): setImmediates 378 + programmable/immediate 282 | **DONE** cts `0b8ea7f` |
| Review | Phase Review of the cumulative yawgpu diff — 1 CRITICAL + 1 MAJOR + 5 MINOR, all resolved | **DONE** (fix commit follows this file) |

## Phase Review outcomes (2026-07-03)

- **CRITICAL — GLES default device creation broken by `Limits::DEFAULT = 64`**:
  `validate_required_limits(None)` substituted DEFAULT and the FFI mapped
  `WGPU_LIMIT_U32_UNDEFINED` to DEFAULT, turning "no descriptor" into an
  implicit *ask* of 64 that a Tier-2 GLES adapter (supported 0) must
  reject. Invisible on Noop (supports 64) and GLES e2e is manual-only —
  exactly the kind of gap the fresh-eyes review exists for. Fixed:
  unspecified `maxImmediateSize` is NOT a requirement (`None` skips the
  over-ask check; FFI maps undefined → 0); explicit over-asks still
  reject; effective device value remains the adapter's supported value.
  Regression tests at both layers.
- **MAJOR — tiled subpass path silently zero-filled user immediates**:
  the subpass vendor extension has no `SetImmediates` surface, so
  subpass pipelines whose stages statically use `var<immediate>` are now
  **rejected at creation** (deterministic validation error + unit test)
  instead of reading zeroes — documented tiled-feature limitation.
- MINORs: stale `HalComputePass`/`HalRenderPass` doc comments updated;
  Vulkan `maxPushConstantsSize >= 72` debug assertion added (spec min is
  128); compute auto-budget resolution deduplicated to mirror the render
  path's shared-resolver split; spec text updated (unconditional per-draw
  delivery is the documented S2/S3 simplification; F-139/polyfill
  framing corrected).
- Review explicitly verified finding-free: SetImmediates validation ==
  Dawn `ProgrammableEncoder` exactly; scratch/mask machinery panic-free;
  bundle overlay semantics; auto-budget == Dawn `CreateDefault`; the
  72-byte combined block being > 64 is correct per Dawn's design; shim
  ABI static_asserts consistent; GLES unreachability holds on all paths
  (incl. bundles + indirect); passthrough/tiled feature-matrix
  consistency.

## Findings caught by review/CTS during the block

- **S1 review (MAJOR, fixed pre-commit):** bundle immediates initially
  wholesale-replaced the outer pass scratch on `ExecuteBundles`; Dawn's
  shared-tracker semantics is OVERLAY (bundle draws inherit outer content;
  only written words land back). Fixed with a per-4-byte-word written mask.
  Later validated for real by the S4 CTS execution cases
  (`render_pass_and_bundle_mix`, `render_bundle_isolation`).
- **S2 CTS (8 false rejections, fixed):** Block 93's "auto layout budgets
  0 bytes" posture was wrong — Dawn's `PipelineLayoutBase::CreateDefault`
  budgets the max of the stages' reflected immediate usage, bounded by
  `maxImmediateSize`. Unobservable until the limit became non-zero; specs
  93/94 corrected.
- **S4 port bug (caught on the Dawn oracle):** `sv(const std::string&)`
  dangled a `WGPUStringView` into a temporary for entry-point names →
  every pipeline creation failed → all 252 operation cases invalid on
  every backend. Fixed to `const char*` + `WGPU_STRLEN`.

## Verification (real GPU, Apple M2, 2026-07-02)

- e2e: `e2e_metal_immediates.rs` 3/3, `e2e_vulkan_immediates.rs`
  (MoltenVK) 3/3 — immediate readback, partial-update overlay between
  dispatches, frag_depth-driven-by-immediate composing with the internal
  clamp block.
- CTS, all three backends byte-identical:

| tree | Dawn | yawgpu-Metal | yawgpu-Vulkan (MoltenVK) |
|---|---|---|---|
| `validation,createPipelineLayout` | 115/3 | 115/3 | 115/3 |
| `validation,pipeline,immediates` | 30/0 | 30/0 | 30/0 |
| `validation,encoding,cmds,setImmediates` | 378/0 | 378/0 | 378/0 |
| `operation,...,programmable,immediate` | 252/30/0 | 252/30/0 | 252/30/0 |

- Regression spots green: `compute_pipeline` 11842/0 (Dawn-equal),
  `render_pipeline,misc` 744/0, `rendering,depth_clip_clamp` 4/0,
  `rendering,draw` 744/0, render-bundle validation 113/0.
- Gates per slice: workspace suites, feature-matrix HAL tests
  (metal/vulkan/gles/noop), clippy `-D warnings`, fmt.

## Outstanding

- Native-Vulkan (Windows RTX) confirmation on the next user-run sweep
  (MoltenVK is non-authoritative Vulkan coverage).
- GLES (Tier 2) stays `maxImmediateSize = 0`; `tint_immediates[0]` remains
  reserved for internal `first_instance` (Block 67). Revisit only if the
  GLES uniform-array delivery path is ever wanted.
- Whole-suite sweep tables in the CTS repo README/FINDINGS predate Blocks
  92-94; refresh at the next full sweep.

## Post-S4 native-Vulkan CTS defect (2026-07-03)

- **Defect:** `webgpu:api,validation,encoding,programmable,pipeline_immediate`
  on native Windows/Vulkan reported 138/43 while Dawn reported 181/0. All
  failures were missing expected validation errors.
- **Root cause:** yawgpu tracked only the encoder-written immediate words and
  the pipeline's scalar `immediate_size`; it did not retain the Dawn/Tint
  per-word required mask (`GetImmediateBlockInfo`). Draw/dispatch therefore
  accepted partial writes. `executeBundles` also preserved the outer render
  pass written-state mask instead of invalidating it.
- **Fix:** pipelines now store a required user-immediate word mask reflected
  from Tint and OR'd across render stages. Draw/dispatch validate
  `(written & required) == required`. `executeBundles` still overlays bundle
  bytes into the outer scratch but resets the outer written-state mask.
- **Local verification:** see `REPORT.md` for the exact cargo commands and
  environment limitation encountered on this machine.
