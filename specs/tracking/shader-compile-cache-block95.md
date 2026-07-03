# Block 95 tracking — shader-compile caching (2026-07-03)

Spec: `specs/blocks/95-shader-compile-cache.md`. Trigger: external CTS
(`webgpu-native-cts` `specs/compile-canary.md`) reported full-suite Metal
compile-scaling collapse (`AGX: exceeded compiled variants footprint limit`
serial; `MTLCompilerService` death + tens-of-thousands fake-fail cascades
under `--workers 4/6/8`) and attributed it to Tint-refactor R4 "dropping the
lowered-IR cache". Investigation corrected the attribution — the F4 cache
never existed (see the F4 follow-up note in
`specs/tracking/tint-integration-refactor.md`); yawgpu had never skipped a
duplicate compile at any layer.

## Slices

- **S0 DONE** — spec authored; churn probe
  `yawgpu/tests/e2e_metal_compile_churn.rs` (#[ignore], canary-shaped:
  Phase A = one module × N identical auto-layout compute pipelines
  create/release; Phase B = N unique-constant variants; pass contract
  `A_median < 0.30 × B_median` or ≤ 2 ms noise floor). **Pre-fix baseline
  (M2, dev profile): A_median 21.1 ms, B_median 52.4 ms, ratio 0.403 —
  RED** (every Phase A iteration paid a full Tint compile; only the Metal
  driver's content-addressed cache kept it below B).
- **S1 DONE** (coding agent) — codegen memoization on `ReflectedModule`
  (`shader_tint.rs`): four per-generator `Mutex<HashMap<Key, Result<_,
  String>>>` caches (MSL via the `generate_stage_msl` funnel, SPIR-V, GLSL
  `#[cfg(gles)]`, whole-body `resolved_compute_workgroup_size`),
  order-insensitive `CanonicalConstants` (name-sorted, raw `f64::to_bits`),
  `Hash` derives on the Msl* key types + `ShaderStage`, `codegen_misses`
  counter, 10 unit tests keyed on miss-count deltas.
- **S2 DONE** (coding agent) — FFI pre-compile short-circuit:
  `cached_handle` consulted before `device.core.create_*` in
  `wgpuDeviceCreateShaderModule` / `create_compute_pipeline_handle` /
  `create_render_pipeline_handle` (guards: cross-device validation first,
  skip when lost, auto-layout keys stay uncached); `canonical_f64_bits`
  stops folding `-0.0` into `+0.0` (pre-existing wrong-hit bug); three
  always-on creation counters on core `Device` as the test observable;
  9 unit tests.
- **S3 DONE** (Claude, real GPU M2) — gates: `cargo test --workspace`
  84/84 ok, clippy `-D warnings` clean, HAL metal/vulkan `--lib` and core
  gles `--lib` green. **Churn probe post-fix: A_median 60.9 µs, B_median
  16.3 ms, ratio 0.004 — GREEN** (~350× on Phase A; Phase B also 52→16 ms
  because intra-creation duplicate codegen calls now hit the cache). CTS
  serial `api,validation,compute_pipeline` 11842/0 (ledger parity). CTS
  `api,operation` `--workers 6`: 218118 pass, single `MTLCompilerService`
  incident (8 cases, worker recovered) vs the 75k–92k cascades the canary
  spec describes.
- **S4 Phase Review DONE** — fresh-context review of the cumulative diff:
  **C1 (CRITICAL, fixed)** `render_pipeline_cache_key` collapsed an
  unrepresentable fragment state (`targets`/`constants` null with count>0)
  into `fragment: None`, so the new short-circuit returned a cached
  vertex-only pipeline *without* the spec-mandated validation error; fixed
  by propagating `None` to the whole key + regression test
  `malformed_fragment_state_never_short_circuits`. **m1 (MINOR, fixed)**
  lock-rationale doc comments specialized per cache. All other suspicions
  verified clean (key completeness, Err-memoization determinism, ABA,
  lock ordering, counters, async/lost-device semantics).

## Mid-flight note — upstream merge

During S2 the coding agent stash/pull/pop-merged upstream commit `80bab07`
(immediates: required-slot validation + `immediate_data_used_slots_cache`)
into the working tree, leaving one conflict marker inside a doc comment
(caught by rustdoc in the workspace gate, removed). Post-merge full gates
re-run green; Block 95 caches and the upstream `immediate_data_used_slots`
reflection cache coexist on `ReflectedModule`.

## Pre-existing issue surfaced (NOT Block 95, needs follow-up)

`webgpu:api,operation,texture_view,texture_component_swizzle:*` under
`--workers 6` mass-fails nondeterministically (6.8k–9.3k of 52k; every
failure is downstream `queue submit cannot use an error command buffer`,
no compile-degrade signature, all formats affected, serial runs green).
**Reproduced identically on the pre-Block-95 dylib (6840 fails)** — a
parallel-execution failure mode independent of compile caching, likely
resource pressure with 6 concurrent GPU processes. Deserves its own
investigation / CTS finding; the canary spec's runner-side `envdegrade`
detection (their Part 2) would currently misattribute these.

## Deferred (measure-first, spec "out of scope")

Metal `MTLLibrary`-by-source cache, real `VkPipelineCache`, auto-layout FFI
dedup, Dawn-style `strip_all_names` canonicalization. Post-fix measurements
show the driver content cache + Block 95 layers suffice for the canary
contract; revisit only if compiler-service pressure reappears.
