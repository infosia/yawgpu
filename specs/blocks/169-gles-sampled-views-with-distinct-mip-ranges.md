# Block 169 — GLES: sampled views of one texture with distinct mip ranges

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR after three rounds.
Source: `specs/tracking/threading-audit.md` item 08, reproduced on Linux native GLES
(NVIDIA EGL, 2026-11-28). Started by user request.
Tier: 2 (GLES). HAL-only; `yawgpu-core` is unchanged.
Oracle: WebGPU semantics (a texture view selects its own mip range); Tier 1 backends
already honour them. Dawn's OpenGL backend is not a reference here: it applies
`BASE_LEVEL`/`MAX_LEVEL` to the bound texture in the same way.

## Problem

GLES has no view object. `bind_combined_samplers` (`yawgpu-hal/src/gles/queue.rs`) expresses
a sampled view's mip range by setting `GL_TEXTURE_BASE_LEVEL` / `GL_TEXTURE_MAX_LEVEL` on
the texture it binds. A transient `glTextureView` is created only when
`requires_texture_view_for` says the view differs from the texture in target, layer range,
format or stencil aspect — the mip range is not part of that predicate.

`BASE_LEVEL` / `MAX_LEVEL` are texture-object state, not texture-unit state. When one draw
or dispatch samples two views of the same texture that differ only in mip range, both
units bind the same GL name and the second bind's parameters replace the first's. Both
views then read the same levels.

`e2e_gles_compute` `gles_compute_reads_two_mip_views_of_one_texture` reproduces it: a
mip-0 view and a mip-1 view of a 2x2 two-level texture read red `[10, 10]` and report
widths `[2, 2]`; the expected values are `[10, 200]` and `[2, 1]`.

## Rules

R1. **Per-view mip range within one bind set.** Within one call to
`bind_combined_samplers` (one draw or one dispatch), every sampled binding reads the mip
range of its own view, including when several bindings resolve to the same GL texture.

R2. **Conflict detection.** A *mip-range conflict* exists when two sampled bindings in the
bind set would both bind the same raw GL texture name (neither needs a transient view
under `requires_texture_view_for`) with a different `(base_mip_level, mip_level_count)`.
The first such binding in bind order keeps the raw name and its `BASE_LEVEL`/`MAX_LEVEL`;
every later binding of that raw texture whose range differs from the first's is bound
through a transient `glTextureView` covering exactly its own mip range and the full layer
range. Bindings with the same range as the first share the raw name. A texture binding
that is paired with several samplers (several combined units) is one view and never
conflicts with itself.

R3. **No cost without a conflict.** A bind set without a mip-range conflict issues the
same GL calls as before this block: no transient view is created for a mip-subrange view
that is the only range bound for its texture. Views that already need a transient view
(target, layer range, format, stencil aspect) are unaffected and do not take part in
conflict detection, since each owns its GL name.

R4. **Tier 2 fallback.** When a conflict needs a transient view and `glTextureView` is
unavailable (ES 3.1 without `OES`/`EXT_texture_view`), the bind returns `HalError` — the
existing "GLES cannot bind this texture view without glTextureView" class — instead of
sampling the wrong level. This is catalogued in `specs/blocks/67-gles-backend.md`.

R5. **Cleanup.** Transient views created for a conflict are released with the existing
`TextureUnitCleanup` / `TransientTextureView` mechanism, on success and on error paths.

R6. **Decision is a pure function.** The choice of which bindings need a conflict view is
computed by a function that takes no GL context (inputs: per-binding raw texture identity,
mip range, and whether the binding already needs a view) and has a direct inline unit
test. The GL binding code consumes its result.

### Storage images (S2, from Phase Review M1)

`glBindImageTexture` binds one level of a texture, and GL treats an image unit as
inaccessible — stores are discarded and loads return zero, with no error — when that level
lies outside the texture's `[BASE_LEVEL, MAX_LEVEL]`. `bind_storage_textures` binds the raw
GL name without touching those parameters, so a sampled bind of the same texture narrows
the range under it. `e2e_gles_compute`
`gles_compute_stores_to_mip_outside_sampled_view_range` reproduces it through the C API:
a dispatch that samples a mip-0 view and stores to a mip-1 storage view of one texture
leaves mip 1 unchanged (`[200, 0, 0, 255]`, expected `[10, 0, 0, 255]`). The same happens
when an earlier dispatch sampled mip 0 and a later one only stores to mip 1.

R7. **A storage image bound by raw name sees the full mip range.** Before
`glBindImageTexture` binds a level of a raw texture name (the `StorageImageBinding::Base`
path), the texture's `BASE_LEVEL` / `MAX_LEVEL` are reset to its full range, as
`normalize_texture_mip_bounds` does for copies. This must not change which texture is
bound to any texture unit that the same draw or dispatch samples from.

R8. **A sampled view must not hide a storage level of the same texture.** In a bind set
where a raw texture is also bound as a storage image by its raw name at level `L`, a
sampled binding of that texture may keep the raw name only if its mip range contains `L`
(for every such storage level); a sampled binding whose range excludes a storage level is
bound through a transient `glTextureView`, in any bind order. Such a binding does not
claim R2's "first range". Storage images are bound (and R7 applied) before the sampled
bindings, so a raw sampled bind that follows narrows the range only to one that still
contains every storage level. The decision stays in the R6 function, which takes the
storage-bound `(texture, level)` pairs as input. R4 applies when `glTextureView` is
unavailable. A storage binding that goes through its own transient view (the
layer-subrange path) is not a raw-name binding and adds no constraint.

(The first S2 implementation reserved the texture's full mip range for any raw storage
binding. Phase Review round 2 m2: that forced a view — and, without `glTextureView`, a
`HalError` — for a read-only storage binding and a single-mip sampled view of the same
level, which worked before this block. S3 narrows it to the rule above.)

R9. **Storage transient views are released on error.** A failure part-way through
`bind_storage_textures` deletes the transient views it has already created.

## Slices

S1: R1–R6 in `yawgpu-hal/src/gles/queue.rs`, inline tests.

S2: R7–R9 in the same file, inline tests. Applies to both the compute and the render-draw
bind paths.

S3 (Phase Review round 2): R8 narrowed as above (m2); a real-GL HAL test through the
render-draw path that fails if `run_render_draw` stops passing the storage bindings to the
sampled bind (M1).

## Verification

- Inline unit tests (no GPU) for the R6 function: no conflict for a single binding; no
  conflict for two bindings of one texture with the same range; a conflict for two
  different ranges of one texture marks only the later one; three bindings (A, B, A range)
  mark only B; different textures never conflict; a binding that already needs a view is
  ignored and does not become the "first" range.
- HAL real-GL test in `gles/queue.rs` (skipped at runtime without a device), next to
  `submit_compute_pass_samples_array_layer_subrange_texture_view`: one compute dispatch
  samples a mip-0 view and a mip-1 view of one texture and reads each level's colour;
  skipped when `glTextureView` is unavailable.
- S2/S3 inline unit tests for the R6 function: a sampled binding whose range excludes the
  storage level is marked; a sampled binding whose range contains the storage level
  (single-mip at that level, or the full range) is not marked; a storage-bound texture
  does not affect sampled bindings of other textures; with a storage level and two
  sampled bindings of different ranges that both contain it, the first keeps the raw name
  and the second is marked (R2); a marked binding that excludes the storage level does not
  claim the first range for a later binding that contains it.
- S3 HAL real-GL test: one render draw whose fragment stage samples mip 0 and stores to
  mip 1 of one texture; the store lands.
- S2 HAL real-GL tests: one dispatch samples mip 0 and stores to mip 1 of one texture and
  the store lands; a dispatch that samples mip 0 followed by a dispatch that only stores to
  mip 1 lands the store. Both skipped when `glTextureView` is unavailable where they need
  it (the second does not).
- Agent gates: `cargo test -p yawgpu-hal --features gles --lib`,
  `cargo clippy -p yawgpu-hal --features gles --all-targets -- -D warnings`, `cargo fmt`.
- Claude: workspace test; `e2e_gles_*` on Linux native GLES, where
  `gles_compute_reads_two_mip_views_of_one_texture` must pass in both binding orders and
  `gles_compute_stores_to_mip_outside_sampled_view_range` in both halves; GLES HAL tests.

## Verification record

Linux, NVIDIA GeForce RTX 5060 Ti, native GLES through EGL (`EGL_PLATFORM_DEVICE_EXT`),
2026-11-28, on the S3 tree.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,516 passed, 0 failed |
| GLES HAL lib tests (`yawgpu-hal`, `gles`) | 303 passed, 0 failed; the new real-GL tests ran on the device |
| GLES HAL ignored tests | 2 passed, 0 failed |
| GLES e2e (`yawgpu`, `--features gles -- --ignored`) | 38 passed, 0 failed, none skipped |
| clippy `-D warnings`, default (workspace) and `gles`, all targets; fmt | clean |

Before the fix, on the same host: `gles_compute_reads_two_mip_views_of_one_texture` read
`[10, 10, 2, 2]` (expected `[10, 200, 2, 1]`), and with S1 only
`gles_compute_stores_to_mip_outside_sampled_view_range` left mip 1 at `[200, 0, 0, 255]`
(expected `[10, 0, 0, 255]`).

The implementer's sandbox could not reach the NVIDIA device; its S2 and S3 runs were on
Mesa llvmpipe, where the new HAL tests fail without the fix and pass with it. No host
without `glTextureView` was run: R4 is covered by the round 3 reviewer's probe with the
capability forced off, not by a committed test.

## Phase Review

Three fresh-agent rounds.

Round 1 (S1): 0 CRITICAL, 1 MAJOR, 4 MINOR.
- MAJOR: a storage-image store to a mip level outside a sampled view's range was dropped,
  in the same dispatch or after an earlier one (pre-existing, same root cause) — fixed by
  S2 (R7, R8).
- The R4 fallback was not catalogued in Block 67 — fixed.
- The e2e test did not skip without `glTextureView` — fixed: both e2e tests skip on the
  R4 device error. The C API cannot query the capability, so the skip keys on the error
  message.
- No render-draw test — fixed in S3 (round 2 MAJOR).
- `bind_storage_textures` leaked transient views on a failure part-way — fixed by R9.

Round 2 (S1+S2): 0 CRITICAL, 1 MAJOR, 5 MINOR.
- MAJOR: no test reached `run_render_draw`; dropping its storage argument left every test
  green — fixed in S3 (`render_draw_samples_mip_zero_and_stores_to_mip_one`).
- R8 reserved the full range for any raw storage binding, turning a previously working
  bind set into a `HalError` without `glTextureView` — fixed in S3 (R8 narrowed).
- Block 67 and the tracking row were stale — fixed.
- The e2e skip returned before the half that needs no `glTextureView` — fixed: that half
  runs first.
- `create_transient_texture_view` attributes any pending GL error to `glTextureView`, and
  storage binds now precede it — declined: the unchecked-error pattern predates the block
  and the case was not reproduced.

Round 3 (S1–S3): 0 CRITICAL, 0 MAJOR, 3 MINOR. The reviewer's exhaustive probe (a
three-level texture, every storage/sampled combination valid in WebGPU, 2,477 dispatches)
found no mismatch, and with `glTextureView` forced off returned `HalError` exactly for
the bind sets the R6 function marks.
- No test fails if the sampled-view guard stops releasing units and views on a failure
  part-way (R5) — deferred: the storage counterpart (R9) has one; a sampled-side test
  needs a failure injected after a conflict view exists.
- No committed real-GL test for a sampled range that contains the storage level (the
  case R8 was narrowed for) — deferred: covered by the pure-function tests and by the
  reviewer's probe.
- Wording in Block 67 ("first sampled range") and "ignored" for the HAL tests in this
  spec — fixed.

Observation, not a finding: render-pass attachments do not reset the mip range, so a pass
can attach a level outside a range left by an earlier dispatch. It works on NVIDIA
(reviewer probe); other drivers are untested.
