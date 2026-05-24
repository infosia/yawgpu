# Phase 15 — GLES backend (Tier 2 / experimental)

Status: **PLANNED** (not started). Rules / plan:
`../blocks/67-gles-backend.md`. Roles / loop:
`../reference/workflow.md`.

**Tier:** Tier 2 (best-effort, experimental). The `gles` cargo
feature is the sole experimental signal — no runtime markers
(`AdapterInfo` suffix, `log::warn!`, C `#define`) are added.
`yawgpu.h` vendor extensions (`tiled`, `shader-passthrough`) are
**not** implemented for GLES; the relevant features are not
advertised and the corresponding FFI calls reject GLES devices via
the existing "feature not enabled" / "backend unavailable" paths.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` green on **Noop**.
Per slice **also**: `cargo build -p yawgpu --features gles` +
clippy with the feature on. **Real-GPU end2end**
(`cargo test --features gles -- --ignored`) is run **manually by
the user on Windows ANGLE** and logged here per slice (the dev
machine has ANGLE; see memory `[[windows-native-vulkan-driver]]`
for the analogous Vulkan flow).

**Reused e2e ports:** Phase 7 already ported Dawn end2end
Basic / Compute / Copy targets (`e2e_basic`, `e2e_buffer`,
`e2e_copy`, `e2e_compute_dispatch`). Phase 15 adds **no new
Dawn-derived tests** — it parametrizes the existing tests over
`--features gles`. New GLES-specific direct unit tests (per
CLAUDE.md core principle 1) are added in `yawgpu-hal/src/gles/`.

**Phase ends with the mandatory Phase Review**
(`tracking/phase-15-review.md`, to be created at slice-completion
time).

Methodology: identical to Phase 7 — execution bring-up, not
validation-rule porting. Validation stays in `yawgpu-core`;
backend only executes already-validated work; driver failure →
`HalError` → device error, never panic; **no core-rule relaxation
for Tier 2**.

## P15.0 — Scaffolding + gating harness  *(☐ PLANNED)*

Goal: add the `gles` cargo feature and `Gles` enum arms to
`yawgpu-hal` returning `HalError::BackendUnavailable` so every
crate builds clean with `--features gles`. Land all Tier 2
documentation edits. No GLES code path executed.

Deliverables:

- `yawgpu-hal/Cargo.toml`: `gles = ["dep:glow", "dep:khronos-egl",
  "dep:libloading"]` + workspace deps.
- `yawgpu/Cargo.toml`: `gles = ["yawgpu-hal/gles",
  "yawgpu-test/gles"]`.
- Workspace `naga` features add `glsl-out`.
- `yawgpu-hal/src/lib.rs`: `HalBackend::Gles`; every HAL enum
  (`HalInstance`, `HalAdapter`, `HalDevice`, `HalQueue`, `HalBuffer`,
  `HalTexture`, `HalSampler`, `HalSurface`, `HalComputePipeline`,
  `HalRenderPipeline`) gains a `#[cfg(feature = "gles")] Gles(...)`
  arm. Inline `gles` placeholder module at
  `yawgpu-hal/src/gles/mod.rs` mirroring the P7.0 Metal placeholder
  shape (every fallible entry → `HalError::BackendUnavailable`;
  `enumerate_adapters()` empty; infallible creators are no-ops).
- `HalInstance::create_surface_from_android_native_window`
  introduced (Noop / Vulkan / Metal arms reject; Gles arm = stub).
- `yawgpu-test`: `RealBackend::Gles` + `real_backend_available(Gles)`
  → false in P15.0; `real_backend_skip_reason` updated.
- One `#[ignore]` `yawgpu/tests/e2e_gles_smoke.rs` asserting
  unavailability (proves the harness).
- Documentation: `CLAUDE.md` (Backend support tiers section),
  `DESIGN.md` (Tier section + HAL paragraph), `SPEC.md` (Phase 15
  row + Out of scope update), `specs/blocks/60-real-backends.md`
  (drop GL from out-of-scope), `README.md` (Tier 2 disclaimer +
  ANGLE binary distribution note).

Acceptance:

- Noop `cargo test --workspace` + `clippy --workspace
  --all-targets -D warnings` byte-for-byte unchanged.
- `cargo build -p yawgpu --features gles` clean.
- `cargo clippy -p yawgpu --features gles --all-targets -D warnings`
  clean.
- Smoke test passes under `--features gles -- --ignored`.
- Vulkan + Metal feature builds unchanged.

Coding-agent handoff template (to be issued at slice start):

```
## Task: gles — P15.0 scaffolding + Tier 2 docs

Goal: add a build-only Gles HAL arm + Tier 2 documentation; CI
stays Noop-green; --features gles compiles.

Inputs to read:
- specs/blocks/67-gles-backend.md (this slice)
- specs/tracking/phase-15.md (P15.0 section)
- yawgpu-hal/src/metal/mod.rs (placeholder shape to mirror)
- yawgpu-hal/src/lib.rs (HAL enum dispatch pattern)
- CLAUDE.md / DESIGN.md / SPEC.md (Tier section to add)

Produce:
- yawgpu-hal: feature + Gles arms + stub module
- yawgpu / yawgpu-test: feature forwarding + RealBackend::Gles
- yawgpu/tests/e2e_gles_smoke.rs (#[ignore])
- Documentation edits per the deliverables list

Out of scope: any real EGL / GL code path; surface implementation;
yawgpu.h extension integration; binding to glow API calls.

Acceptance criteria:
- [ ] cargo build -p yawgpu --features gles clean
- [ ] cargo clippy -p yawgpu --features gles --all-targets
  -D warnings clean
- [ ] Noop cargo test --workspace byte-for-byte unchanged
- [ ] e2e_gles_smoke passes under --features gles -- --ignored
- [ ] CLAUDE.md / DESIGN.md / SPEC.md / blocks/60 / README updated
- [ ] no panics in yawgpu-core / yawgpu-hal; CLAUDE.md conventions met

Report back: files changed, any planned deliverables intentionally
deferred (+why).
```

## P15.1 — EGL display + Instance/Adapter/Device/Queue  *(☐ PLANNED)*

Goal: real EGL bring-up via `khronos-egl` dynamic loading; one
adapter from a default RGBA8 `EGLConfig`; shared GL context per
`HalDevice`; `submit_empty` issues `glFlush`.

Real-GPU verification: `cargo test --features gles -- --ignored`
on Windows ANGLE; user logs results here.

(Detailed deliverables / handoff to be drafted when P15.0 lands.)

## P15.2 — Buffer + Queue write/read + B2B copy  *(☐ PLANNED)*

Reuses `e2e_buffer`. Decision required: buffer-mapping fence model
(see `blocks/67` open questions).

## P15.3 — Texture/Sampler + B2T/T2B/T2T  *(☐ PLANNED)*

Reuses `e2e_copy` texture subset. Decision required:
storage-texture format gating timing.

## P15.4 — Shader (naga→GLSL ES 3.10) + compute  *(☐ PLANNED)*

Reuses `e2e_compute_dispatch`. Naga `glsl-out` smoke confirmed in
P15.0/P15.1; any uncovered WGSL constructs flow into this slice's
scope.

## P15.5 — Render pipeline + draw  *(☐ PLANNED)*

Reuses `e2e_basic` draw portion. `first_instance` via naga uniform
injection.

## P15.6 — Surface (Android ANativeWindow + Windows ANGLE HWND)  *(☐ PLANNED)*

`examples/triangle` runs under `--features gles` on ANGLE.

## Phase 15 Review  *(☐ PLANNED)*

Mandatory Clean Review Then Fix per `specs/reference/workflow.md`.
Logged in `tracking/phase-15-review.md` at slice-completion time.
Phase 15 cannot be marked COMPLETE with any open CRITICAL/MAJOR
finding. MINORs may defer with explicit rationale.

## Open follow-ups (carried from `blocks/67-gles-backend.md`)

- naga `glsl-out` coverage smoke for Phase 7 e2e shaders.
- Adapter limit mapping reconciliation with core
  `RequiredLimits` validation.
- ANGLE binary distribution wording in `README.md`.
- Buffer mapping fence model definition.
- Storage-texture format gating timing.
- Resource hazard barrier mask defaults.
