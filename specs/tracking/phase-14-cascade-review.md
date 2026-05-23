# Phase 14 — cascade-diff review (post-COMPLETE re-open)

## Why this exists

Phase 14 was marked COMPLETE at commit `5616518`. Honest re-verification with
Claude Code's Bash sandbox disabled (commit `f9bf265`, "silent-skip → real
green") then discovered that the Metal subpass-input lowering had been broken
and that the test runner's "0 ran" had been reported as a pass. The Metal fix
landed across commits `f9bf265` / `28d4ae8` / `03cb073`. A fresh Phase Review
on that **cascade diff** (`5616518..HEAD`) surfaced the findings below; this
file is the workflow-mandated review record + the fix handoff.

## Cascade scope

Range: `5616518..HEAD` (currently `5616518..03cb073`)
Touched files:

- `yawgpu-core/src/render_pipeline.rs`
- `yawgpu-core/src/shader_naga.rs`
- `yawgpu-core/src/subpass.rs`
- `yawgpu-hal/src/metal/device.rs`
- `yawgpu-hal/src/metal/pipeline.rs`
- `yawgpu/tests/e2e_metal_tiled.rs`
- `examples/tiled_deferred/main.c`
- `specs/tracking/phase-14.md`

## Findings

### CRITICAL

**[C1] `validate_color_targets` Vulkan regression + 2 failing Noop unit tests** — `yawgpu-core/src/render_pipeline.rs:1647-1650` (function `validate_color_targets`, the `if !skip_shader_outputs { ... }` block).

The cascade replaced the old `outputs.get(&(index as u32))` lookup with:

```rust
let location = subpass_color_attachment_indices
    .and_then(|indices| indices.get(index).copied())
    .unwrap_or(index as u32);
match outputs.get(&location) { ... }
```

This forces fragment `@location(N)` to use the **flat layout slot index** for
all backends. On Vulkan that's wrong: WGSL writes the **subpass-local**
location (which `VkRenderPass` remaps), so for a subpass with
`color_attachment_indices = [1]` the WGSL author writes `@location(0)` and
`outputs` is keyed by `0`. The new lookup asks for `outputs.get(&1)`,
finds `None`, and rejects the pipeline because `target.write_mask != 0`.

Confirmed by directly running the existing Noop subpass unit tests:

```
$ cargo test -p yawgpu-core --features tiled --lib subpass::
...
test subpass::tests::subpass_render_pass_draw_auto_wires_input_attachment_bind_group ... FAILED
  panicked at yawgpu-core/src/subpass.rs:1507:9: assertion failed: !reader.is_error()
test subpass::tests::subpass_render_pipeline_validates_layout_formats_and_subpass_match ... FAILED
  left:  Some("subpass render pass requires a valid render pipeline")
  right: Some("subpass render pipeline is not compatible with the active subpass")
```

Both tests use `test_helpers::render_shader_module` (WGSL `@location(0)`)
on a subpass with `color_attachment_indices = vec![1]`
(`yawgpu-core/src/subpass.rs:985`), so the pipeline becomes an error pipeline
the moment the cascade is applied. The native-Vulkan
`vulkan_two_subpass_draw_subpass_load_readback` e2e would fail identically —
on this M2 it is masked by `adapter_is_moltenvk()` self-skip; the prior
Windows native-Vulkan run (`1cd0a0c`) predates the cascade so it does not
cover the new validation path.

### MAJOR

**[M1] `subpass_input_shader_generates_spirv_and_msl_status_is_known` silently passes on naga error** — `yawgpu-core/src/render_pipeline.rs:2104-2109`.

The MSL `[[color(` assertion only runs inside `Ok(msl)`; the `else` arm just
has a stale comment ("B4 supplies that pass-local map"). Now that B4 *does*
supply the map, an `Err` should be a test failure, not a silent pass.

### MINOR

**[m1] phase-14.md "MoltenVK 4/4 (no regression)" overstates coverage** — `specs/tracking/phase-14.md`, the "Honest re-verification" section.
The 4/4 includes `vulkan_two_subpass_draw_subpass_load_readback`, which
self-skips on MoltenVK via `adapter_is_moltenvk`. The same honesty applied
to the Metal silent-skip story should apply here.

**[m2] `28d4ae8` "cdylib + Metal enumerate_adapters empty-return resolved" claim is Metal-only** — `specs/tracking/phase-14.md`, the strikethrough item.
Today's MoltenVK run showed the cdylib also needs `DYLD_LIBRARY_PATH=${VULKAN_SDK}/lib`
or `wgpuCreateInstance` silently selects Noop (`backendType = Null`). The
`03cb073` follow-up note covers the Vulkan side but the `28d4ae8` wording
reads as if all silent-fallback paths are gone.

**[m3] `adapter_is_moltenvk` heuristic duplicated, no cross-link** — `yawgpu/tests/e2e_vulkan_tiled.rs:185-200` and `examples/tiled_deferred/main.c:439-462`.
The C side comments "mirrors `e2e_vulkan_tiled.rs::adapter_is_moltenvk`" but
the Rust side has no reciprocal comment, so a future edit to the Rust
heuristic will silently drift from the example.

## Triage

All five findings are kept (no false positives).

## Spec updates (Claude — done in this commit)

- `specs/blocks/55-tiled-rendering.md`: new rule **"Fragment `@location(N)`
  on subpass pipelines: dual convention accepted"** documenting the Vulkan
  subpass-local vs Metal flat-slot semantics and the validation contract
  ("the shader is valid if it writes either `@location(i)` *or*
  `@location(layout.color_attachment_indices[i])`").

The corresponding `validate_color_targets` logic + tests are the coding
agent's work (handoff below). Tracking-doc minors **m1 and m2 are Claude's**
(they live in `specs/tracking/phase-14.md`). **m3 (reciprocal cross-link
comment in `yawgpu/tests/e2e_vulkan_tiled.rs`) is the coding agent's** —
it touches test code; bundled into the same handoff.

## Handoff — coding agent (CRITICAL + MAJOR fix)

### Task: cascade-fix-C1 — tolerant subpass `@location` validation

Goal: accept WGSL fragment `@location(N)` written in **either** the
subpass-local convention (Vulkan) **or** the flat-MTL-slot convention
(Metal), as specified in `specs/blocks/55-tiled-rendering.md`'s new "dual
convention" rule. Get the two failing Noop unit tests green without
regressing Phase 14's Metal cascade fix.

Inputs to read:

- `specs/blocks/55-tiled-rendering.md` (in particular the new "dual
  convention accepted" rule)
- `yawgpu-core/src/render_pipeline.rs` — `validate_color_targets` (the only
  place to change) and `select_render_shader_source` (do **not** change its
  `subpass_color_slots` parameter; the MSL slot map is still required)
- `yawgpu-core/src/subpass.rs` — `validate_subpass_render_pipeline_descriptor`
  (do **not** revert its call to `resolve_render_pipeline_descriptor`; the
  `Some(&subpass.color_attachment_indices)` plumbing stays, only the lookup
  policy inside `validate_color_targets` changes)
- `CLAUDE.md`, `specs/reference/naming-conventions.md`

Produce (production code):

1. `validate_color_targets` (`yawgpu-core/src/render_pipeline.rs`): change the
   shader-output lookup to accept either convention. Concretely (sketch — the
   coding agent owns the exact form):

   ```rust
   let subpass_local = index as u32;
   let flat = subpass_color_attachment_indices
       .and_then(|indices| indices.get(index).copied())
       .unwrap_or(subpass_local);
   let output = outputs
       .get(&subpass_local)
       .or_else(|| outputs.get(&flat));
   match output { Some(o) => validate_fragment_output_compat(*o, caps)?, ... }
   ```

   The "no shader output → write_mask must be 0" branch must still fire when
   **neither** lookup matches and `target.write_mask != 0`. Update the
   surrounding comment to point at the block-55 rule rather than the
   "flat MTL only" framing.

2. Tighten the existing M1 test
   (`subpass_input_shader_generates_spirv_and_msl_status_is_known`,
   `yawgpu-core/src/render_pipeline.rs:2086`) so the MSL `[[color(`
   assertion runs unconditionally: `let msl = ...expect("..."); assert!(msl.source.contains("[[color("));`.
   Delete the stale `else` comment.

3. **New** inline unit test (same module, gated on `feature = "tiled"`)
   asserting BOTH conventions resolve to `Ok` and exercise the same
   `RenderPipelineDescriptor` shape used by the failing subpass tests.
   Suggested name: `validate_color_targets_subpass_accepts_both_location_conventions`.
   It should call `resolve_render_pipeline_descriptor` (or
   `validate_color_targets` directly with a constructed descriptor) twice —
   once with WGSL `@location(0)` against `color_attachment_indices=[1]` (Vulkan
   convention), once with WGSL `@location(1)` against the same
   `color_attachment_indices=[1]` (Metal convention) — and assert both `Ok`.

4. **m3 cross-link comment** in `yawgpu/tests/e2e_vulkan_tiled.rs:185-200`
   (`adapter_is_moltenvk`): add a one-line comment noting the heuristic is
   intentionally mirrored in `examples/tiled_deferred/main.c::adapter_is_moltenvk`
   so future edits stay in sync. Doc-only; no behaviour change.

Out of scope: HAL changes, real-backend changes, spec edits, commits, any
change to the cascade's commit history.

Acceptance criteria:

- [ ] `cargo test -p yawgpu-core --features tiled --lib subpass::` is fully
      green (the two currently-failing tests pass without their assertions
      being relaxed).
- [ ] `cargo test --workspace --features yawgpu/tiled` is fully green.
- [ ] `cargo clippy --workspace --all-targets --features yawgpu/tiled -- -D warnings` clean.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` (default
      features, no `tiled`) clean.
- [ ] New `validate_color_targets_subpass_accepts_both_location_conventions`
      test exists and passes.
- [ ] M1 test no longer has a silent-pass branch.
- [ ] No change to the Metal-side cascade (the e2e
      `metal_two_subpass_draw_subpass_load_readback` must still pass —
      Claude verifies on the GPU after the diff lands).
- [ ] No production code added outside the two `yawgpu-core/src/render_pipeline.rs`
      changes above (no HAL changes, no FFI changes, no new modules).
- [ ] No `panic!`/`unwrap`/`expect` introduced into library code outside
      `#[cfg(test)]` (CLAUDE.md principle 3).

Report back: files changed, test counts before/after, anything ambiguous about
the spec rule.

## Gate

Phase 14 cannot be re-declared COMPLETE while C1 or M1 is open. After the
coding agent reports back, Claude:

1. Reads the diff against the acceptance criteria above.
2. Runs the Noop gates.
3. Runs real-GPU re-verification on this M2 (Metal e2e with sandbox off;
   MoltenVK e2e knowing the 2-subpass test self-skips); flags that native
   Vulkan re-verification is still owed and must be done on a real driver
   before re-declaring COMPLETE.
4. Updates `specs/tracking/phase-14.md` minors m1 + m2 and the
   `e2e_vulkan_tiled.rs` reciprocal comment (m3) — all spec/doc edits.
5. Commits the coding agent's fix + Claude's doc edits.

After the fix lands and native-Vulkan re-verification is done, this review
record gains a "Re-COMPLETE" footer with the verification log.
