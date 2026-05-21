# Phase 11 — Phase Review (mandatory)

Status: in progress. Per `../reference/workflow.md` ("Phase
Review"): a fresh no-context subagent reviewed the cumulative
Phase 11 diff (`7457e48..HEAD` — P11.1 yawgpu-hal split + P11.2
yawgpu-core split + P11.3 yawgpu split, a pure modularization
refactor). CRITICAL/MAJOR must be fixed before Phase 11 is
COMPLETE.

## Review headline

**Phase 11 review: 0 critical, 1 major, 3 minor** (subagent), but
the MAJOR is **deeper than first reported** — see M1 below. The
core refactor invariant held excellently:

Positive findings (verified, no action):
- Public API byte-for-byte preserved across all three crates
  (yawgpu-hal 47/47, metal+vulkan 10/10 each, yawgpu-core
  104/104, yawgpu 203/203 top-level pub items; the only new
  top-level pub lines are `pub use` re-export shims).
- C ABI intact: exactly 169 `#[no_mangle] pub unsafe extern "C"
  fn wgpu*` in both old and new, identical symbol sets, none
  stripped of `#[no_mangle]`/`extern "C"`.
- No behavior drift: ~12 spot-checked moved bodies are
  byte-identical modulo visibility raises (`fn` →
  `pub(crate)`/`pub(super)`, none reaching `pub`).
- Tests preserved: Noop 502/58, Metal `--ignored` 26, Vulkan
  `--ignored` 22, all unchanged.

## MAJOR finding — must fix

### M1 — Blanket `#![allow(dead_code, private_interfaces, unused_imports)]` masks pervasive duplicated dead code introduced by the split

The split introduced crate-/module-level blanket allows that did
not exist before:
- `yawgpu-core/src/lib.rs:1` — `#![allow(dead_code,
  private_interfaces, unused_imports)]`
- `yawgpu/src/ffi/mod.rs:1` — `#![allow(dead_code,
  private_interfaces, unused_imports)]`
- `yawgpu/src/conv/mod.rs:1` — `#![allow(dead_code,
  unused_imports)]`

(The `pub mod native { #![allow(...)] }` block in `yawgpu/src/
lib.rs` is the **pre-existing** bindgen-output allow — legitimate,
leave it.)

Removing the yawgpu-core allow surfaces **13 dead duplicate
functions + 1 dead duplicate struct + ~40 unused imports** that
the split created by copy-pasting whole-file import headers and
duplicating helper functions into multiple modules. Each dead
item has an identical **live** copy elsewhere (the one actually
called). Confirmed duplications in yawgpu-core (dead copy →
canonical/live copy):

- `format.rs` `VertexFormat` + `VertexFormatInfo`
  → canonical in `render_pipeline.rs` (lib.rs re-exports from
  render_pipeline).
- `shader.rs` `validate_bind_group_layout_descriptor`
  → canonical in `bind_group_layout.rs` (called by device.rs +
  compute_pipeline.rs via
  `crate::bind_group_layout::validate_bind_group_layout_
  descriptor`).
- `render_pass.rs` (11 fns) `validate_render_pass_descriptor`,
  `render_pass_color_execution`, `validate_color_attachment`,
  `validate_depth_stencil_attachment`,
  `validate_render_attachment_common`, `validate_resolve_target`,
  `render_pass_attachment_signature`,
  `render_pass_attachment_textures`,
  `validate_render_pass_timestamp_writes`,
  `validate_timestamp_query_set`, `validate_resolve_query_set`
  → canonical in `command_encoder.rs`.
- `pass.rs` `validate_compute_dispatch_state`
  → canonical in `compute_pass.rs`.
- `copy.rs` `hal_buffer_texture_layout`
  → canonical in `queue.rs`.

The yawgpu crate's `ffi/mod.rs` + `conv/mod.rs` allows almost
certainly mask the same class of issue (copy-pasted import
headers + possibly duplicated helpers); they must be removed and
the surfaced warnings fixed too.

**Why it matters:** the whole point of Phase 11 was clean
modularization. Shipping ~15 dead duplicate functions hidden
behind a blanket `dead_code` allow is the opposite — it bloats
the tree and risks the two copies silently diverging. The
blanket `unused_imports` allow likewise hides that every new
module carries the original monolith's full (mostly-unused)
import list.

**Fix:** remove all three split-introduced blanket allows
(`yawgpu-core/src/lib.rs`, `yawgpu/src/ffi/mod.rs`,
`yawgpu/src/conv/mod.rs`) and resolve every warning they were
masking — delete the dead duplicate items (keep the live
canonical copy, do NOT relocate), and trim the per-file imports
to what each file actually uses (careful with `#[cfg(test)]`-only
imports — those belong inside the test module). Verify the live
copies are byte-identical to the deleted dead copies (pure
dedup, no behavior change).

## MINOR findings

### m1 — (folded into M1)
The subagent listed the yawgpu `ffi/mod.rs` + `conv/mod.rs`
blanket allows as a separate MINOR; in the fix they are handled
together with M1 (same root cause).

### m2 — Empty orphan file `yawgpu/src/conv/test_helpers.rs`
0 bytes, not declared as a `mod` anywhere. Dead weight. **Fix:**
`git rm yawgpu/src/conv/test_helpers.rs`.

### m3 — `yawgpu/src/ffi/mod.rs` is ~4,483 lines (deferred)
Houses the 22 `WGPU*Impl` handle structs + error-sink machinery
+ FFI helpers. A reasonable stopping point (it is the handle hub,
not FFI fns — those were correctly sharded). Could be split
further (`handles.rs` + `error_sink.rs` + `helpers.rs`) in a
follow-up; **deferred**, does not block COMPLETE.

## Fix log

- M1 fixed: removed the three split-introduced blanket allows from
  `yawgpu-core/src/lib.rs`, `yawgpu/src/ffi/mod.rs`, and
  `yawgpu/src/conv/mod.rs`.
- M1 fixed: deleted yawgpu-core dead duplicate items after comparing
  them against their canonical live copies. All were byte-identical
  except `copy.rs` `hal_buffer_texture_layout`, which differed only
  by visibility and `div_ceil_u32` path qualification; the dead copy
  was deleted with the canonical `queue.rs` copy kept.
- M1 fixed: trimmed unused imports surfaced by the allow removal in
  yawgpu-core and yawgpu. No `private_interfaces` warnings remain.
- m2 fixed: deleted empty orphan file
  `yawgpu/src/conv/test_helpers.rs`.

## Status

**Phase 11 COMPLETE** (2026-05-21). The three split-introduced
blanket allows are gone, the ~15 dead duplicate items + ~40
unused imports they masked are removed (-1598 lines net), m2's
orphan file deleted, and all gates are green: Noop 502/58
byte-for-byte unchanged, workspace + --features metal/vulkan
clippy clean with NO blanket allows, Metal --ignored 26/26 +
Vulkan --ignored 22/22, Phase-9 examples exit 0. m3 (ffi/mod.rs
size) deferred as a non-blocking follow-up.
