# Block 168 — Multisampled input attachments per binding (SPIR-V)

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 1 MAJOR, 6 MINOR; the MAJOR and
five MINORs fixed, one declined, see below. Fork commit `9bba94346` on
`feature/tiled-v20261030`.
Source: `specs/tracking/threading-audit.md` item 09, reproduced on Windows native Vulkan
(2026-11-28). Started by user request.
Repositories: the Tint fork (`third_party/dawn`, branch `feature/tiled-v20261030`) and
yawgpu (`yawgpu-tint`, `yawgpu-core`).
Oracle: none. Upstream Dawn and Tint have no multisampled input attachments: Tint's
`chromium_internal_input_attachments` is internal-only, its `inputAttachmentLoad` takes
one argument, and Dawn's `InputAttachmentBindingInfo` carries only a sample type. The
two-argument `inputAttachmentLoad(ia, sample_index)`, the SPIR-V option
`multisampled_input_attachment`, and its lowering are fork additions (`tint-tbdr` markers;
the fork's `TILED.md` and `docs/tint/tbdr-fork-conventions.md`).

## Problem

The fork's SPIR-V option `Options::multisampled_input_attachment` is one module-wide
`bool`. When it is set, every `input_attachment<T>` in the module is lowered to a
multisampled `SubpassData` image, and the writer's `CanGenerate` requires every
`inputAttachmentLoad` in the module to pass a sample index. yawgpu sets it for a subpass
pipeline whenever any bind group layout entry is a multisampled input attachment
(`pipeline_has_multisampled_input_attachment`, `yawgpu-core/src/render_pipeline.rs`).

So a valid subpass pipeline that reads one 4x input per sample and one 1x input with the
one-argument load fails at creation with the internal error "inputAttachmentLoad on a
multisampled input_attachment requires an explicit sample index". Core validation
accepts the layout: subpass attachments that a subpass writes must share a sample count,
but inputs read from earlier subpasses may differ. The probe
`vulkan_tiled_mixed_multisampled_and_single_input_attachments` in
`yawgpu/tests/e2e_vulkan_tiled.rs` reproduces it on Windows NVIDIA: pipeline creation
fails, and the readback is zero. The fork documents this as a known constraint ("a
per-attachment (rather than global) multisampled flag on SPIR-V" is deferred).

## Rules

### Tint fork

F1. **Option.** Add a fork-only SPIR-V writer option listing the input attachments that
are multisampled, by binding point (for example
`std::vector<BindingPoint> multisampled_input_attachments`), in
`src/tint/lang/spirv/writer/common/options.h`, in its `TINT_REFLECT`, and threaded through
`raise.cc` into the builtin polyfill config. It is additive and marker-fenced. The
existing `bool multisampled_input_attachment` stays and keeps its meaning: when it is
true, every input attachment is multisampled. An input attachment is multisampled when
the bool is true or its binding point is in the list. State in a comment which binding
points the list uses (the WGSL source ones, or the ones after the writer's binding
remap) and key the lookup to match where the polyfill runs.

F2. **Lowering per variable.** Each module-scope `input_attachment<T>` variable is lowered
to a multisampled `SubpassData` image exactly when it is multisampled under F1, and to a
single-sampled one otherwise, even when both kinds share the same `T`. Its loads and the
`inputAttachmentLoad` calls on it use that image type. The two-argument call adds the
`Sample` image operand, as today.

F3. **Validation.** `CanGenerate` checks each `inputAttachmentLoad` against the input
attachment it reads: two arguments require a multisampled attachment, one argument a
single-sampled one, with clean `Failure` messages naming the mismatch. A shape that cannot
be lowered per variable (for example, an input attachment value passed through a function
parameter when the module has both kinds and the root variable cannot be determined)
returns a clean `Failure`, never an ICE.

F4. **Tests.** Writer and polyfill unit tests: only the bool (today's behaviour,
unchanged); only the list; one module with a multisampled and a single-sampled attachment
of the same `T` (two `SubpassData` image types, `Sample` operand only on the
multisampled load); each mismatch failure; the unsupported shape of F3. If a resolver or
intrinsic change is needed, follow the fork rules (both `.def` files, generated output in
a separate commit).

F5. **Fork documentation.** `TILED.md` replaces the module-wide constraint with the
per-binding rule; `docs/tint/tbdr-fork-conventions.md` removes the deferred
"per-attachment multisampled flag" item and lists any new `tint-tbdr` markers.

### yawgpu

Y1. **Shim.** `yawgpu_tint_generate_spirv` takes the list of multisampled input
attachment binding points (pointer and count) and sets F1's option. The bool is no longer
set by yawgpu.

Y2. **Core.** For a subpass pipeline, core collects the group and binding of every bind
group layout entry that is a multisampled input attachment and passes that list to every
stage's SPIR-V generation (the vertex stage too: Tint validates the whole module). It
replaces `pipeline_has_multisampled_input_attachment`. The list is part of the SPIR-V
codegen key (Block 95).

Y3. **Unchanged.** Pipelines whose input attachments are all multisampled or all
single-sampled produce the same SPIR-V as before. Metal (no MSAA subpass input), GLES and
non-tiled builds are unchanged.

## Tests

- Fork: F4, run with the fork's Tint unit tests (`*InputAttachment*`, `*Spirv*`).
- yawgpu shim (real Tint): a mixed module generates SPIR-V with both image kinds; the
  existing module-wide tests keep passing with the list form.
- yawgpu core: the list is built from the layouts, reaches both stages, and changes the
  codegen key.
- e2e (Claude): the mixed probe passes on Windows native Vulkan under the Khronos layer
  with no VUID, and the existing `vulkan_tiled_msaa_*` cases still pass.

## Process

The fork change lands as commits on `feature/tiled-v20261030` (hand-written and generated
separately, per the fork rules), pushed by the user. yawgpu then bumps the
`third_party/dawn` pin to that commit with the yawgpu-side change and records the bump in
`specs/reference/dependencies.md`.

## Slices

- **S1 (coding agent):** F1-F4 in the fork; Y1-Y3 in yawgpu, with unit tests.
- **S2 (Claude):** F5, Tint unit tests and yawgpu gates on this host, e2e, Phase Review,
  commits in both repositories, the pin bump.

## Implementation notes

- The list carries WGSL source binding points. `MultisampledInputAttachmentVars`
  (fork, `builtin_polyfill`) turns it into the set of module-scope input attachment
  variables whose source point is listed. `raise.cc` calls it before `BindingRemapper`
  and gives the polyfill the variable set, so a remap cannot move another variable onto
  a listed point. `CanGenerate` builds the same set from the same points.
- `ResolveInputAttachmentSampling` follows each variable's uses (loads, lets, accesses,
  user-call arguments into parameters, nested calls) and decides the image type per
  value; a parameter that receives both kinds is a clean `Failure`. `CanGenerate` skips
  values no variable reaches (dead helpers, which `SingleEntryPoint` removes before
  lowering); the polyfill checks them and fails cleanly.
- yawgpu: `YawgpuTintBindingPoint` / `yawgpu_tint::BindingPoint` (8 bytes, layout
  asserted on both sides). `pipeline_multisampled_input_attachments` lists the group
  (the layout's index in the pipeline layout) and binding of every multisampled
  input-attachment entry; both stages get the list, and `SpirvCodegenKey` holds it.
- `yawgpu-tint/build.rs` reruns only when the shim files change, so after a fork-only
  edit a file in `yawgpu-tint/shim/` must be touched (as
  `specs/reference/dependencies.md` already says for pin bumps).

## Verification (2026-11-28)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351.

| Gate | Result |
|---|---|
| Fork Tint unit tests (separate CMake build, `TINT_BUILD_TESTS=ON`) | `*InputAttachment*` 73 passed; `*Spirv*` 1,360 passed; full suite except `*MaxExpressionDepth*` 53,253 passed |
| `cargo test -p yawgpu-tint --features tiled` (real Tint, rebuilt from the fork change) | 135 + 20 passed |
| `cargo test --workspace` | 1,513 passed, 0 failed; with `yawgpu/vulkan,yawgpu/tiled` 1,873 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default (workspace), `vulkan,tiled`, `vulkan,shader-passthrough` and `gles`, all targets | clean |
| Probe `vulkan_tiled_mixed_multisampled_and_single_input_attachments`, before | pipeline creation fails: "inputAttachmentLoad on a multisampled input_attachment requires an explicit sample index"; readback (0, 0, 0, 0) |
| The probe, after, Khronos layer | passes, no VUID |
| Vulkan e2e (`e2e_vulkan_*`, `vulkan,tiled,shader-passthrough`, `--ignored`), Khronos layer | 178 passed, 0 failed, 0 VUIDs; `vulkan_tiled_msaa_*` unchanged |

`ResolverTest.MaxExpressionDepth_Pass` crashes the Windows Release test binary (exit
127), most likely its stack depth; this change does not touch the resolver.

Run on macOS, Apple M2, at `ea4d24e` with the fork at `9bba943462` (2026-11-28):

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,514 passed, 0 failed; with `yawgpu/vulkan,yawgpu/tiled` 1,874 passed, 0 failed |
| clippy `-D warnings` on default, `vulkan,tiled` and `metal,tiled`, all targets | clean |
| `yawgpu-hal --lib` | 395 passed (`vulkan,tiled`); 218 passed (`metal,tiled`) |
| Metal e2e (`metal,tiled`, `--ignored`) | 176 passed, 0 failed |
| Vulkan e2e (`vulkan,tiled`, `--ignored`), MoltenVK, Khronos layer | 175 passed, 1 failed (the known `imageView2DOn3DImage` portability case; every VUID comes from that test). The mixed probe and both `vulkan_tiled_msaa_*` tests pass |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | The list was translated through the binding remap point by point; a listed point that this module does not declare could match another variable remapped onto it, so the polyfill marked a single-sampled variable multisampled and reached `TINT_IR_ASSERT` from valid WGSL | Fixed: multisampling is decided per variable before `BindingRemapper` (`MultisampledInputAttachmentVars`); writer tests for a remap onto an undeclared listed point and onto a listed source point |
| m1 | A stray blank line in `writer.cc` sat outside any `tint-tbdr` fence | Removed |
| m2 | `CanGenerate` rejected a dead helper read with the 2-arg load under list-only use | Values no variable reaches are skipped in `CanGenerate`; the polyfill still fails cleanly; tests both ways |
| m3 | `HoldsInputAttachment` did not unwrap `ResourceTable` | Unwraps it, as `ReplacementType` does |
| m4 | The probe's comment described the old module-wide option | Rewritten as the Block 168 regression guard |
| m5 | The probe's bind group layout helper computed `multisampled` backwards | Maps over `[1, 0]` directly |
| m6 | The `…-per-binding-include` tag also fences a forward declaration | Declined: harmless; the conventions table names it |

