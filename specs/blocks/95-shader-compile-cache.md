# Block 95 — Shader-compile caching (FFI short-circuit + codegen memoization)

## Background

Full-suite CTS runs against yawgpu on Metal collapse environmentally: serial
runs stall with `AGX: exceeded compiled variants footprint limit`; parallel
runs (workers 4/6/8) kill macOS `MTLCompilerService` and cascade into tens of
thousands of fake fails. Conformance is unaffected (per-file serial runs stay
green) — only shader-compile *scaling* is broken. The external CTS suite
(`webgpu-native-cts`, `specs/compile-canary.md` there) is adding a
compile-canary gate; this block fixes the root cause in yawgpu.

Root cause (correcting the CTS spec's attribution — the R4 "dropped lowered-IR
cache" never existed; see the F4 note in
`specs/tracking/tint-integration-refactor.md`): yawgpu has never skipped a
duplicate shader/pipeline compile, at two frontend layers:

1. **FFI dedup caches never short-circuit.** The four per-device Weak+prune
   dedup caches (`yawgpu/src/ffi/mod.rs`) are consulted only *after*
   `device.core.create_*` has run. A cache hit still pays a full Tint compile
   plus (on Metal) a `newLibraryWithSource` — i.e. a new driver
   compiled-variant — and then discards the freshly built object in favour of
   the cached one. Dawn, by contrast, returns its cached pipeline without
   initializing it (`dawn/native/Device.cpp`, `GetCachedComputePipeline`).
2. **No codegen memoization.** `ReflectedModule`
   (`yawgpu-core/src/shader_tint.rs`) memoizes *reflection* (Tint-refactor R3)
   but every `generate_msl` / `generate_spirv` / `generate_glsl` /
   `resolved_compute_workgroup_size` call re-runs Tint `ProgramToLoweredIR` +
   writer from scratch. An identical **auto-layout** compute pipeline created
   N times against a live module pays N full Tint compiles — and auto-layout
   descriptors are exactly the ones the FFI pipeline cache cannot key
   (`compute_pipeline_cache_key` returns `None`), so layer 1 alone cannot fix
   that shape (the canary's Phase A).

Out of scope for this block (deferred pending post-fix measurement, see the
plan): a Metal `MTLLibrary`-by-source cache, a real `VkPipelineCache`,
FFI dedup of auto-layout pipelines, and Dawn-style `strip_all_names` MSL
canonicalization. Dawn itself ships none of the first two in standalone runs;
it survives because Tint's deterministic output lets the Metal driver's
content-addressed cache dedup byte-identical MSL.

## S1 — Core codegen memoization (`yawgpu-core`)

### Contract

For a given live `ReflectedModule`, two codegen calls with equal inputs
return equal outputs while running the underlying Tint codegen **once**.
Equal inputs are defined by the cache keys below. Both `Ok` and `Err`
results are memoized (an error is a deterministic function of
(module, inputs); repeated failing pipeline creation must not re-cross the
FFI). Memoization is per-`ReflectedModule` instance; it lives and dies with
the module. No entry cap (entries are bounded by distinct pipeline configs
created against the module; a clear-at-N escape hatch may be added later if
profiles ever demand it — document this on the fields, do not implement).

### Keys and canonical forms

`PipelineConstants` is a `HashMap<String, f64>`; its canonical, hashable,
order-insensitive form is:

```rust
struct CanonicalConstants(Vec<(String, u64)>); // name-sorted, f64::to_bits()
```

Raw `to_bits()`, **no** −0.0/NaN folding: non-finite values are rejected by
`validate_pipeline_constant_value` before any codegen call, and −0.0 vs +0.0
produce different compiled shaders so they must be distinct keys.
(`override_values` itself needs no ordering change — Tint consumes named
overrides order-independently.)

Private key structs (all `#[derive(Debug, Clone, PartialEq, Eq, Hash)]`),
one cache per generator:

| Cache field on `ReflectedModule` | Key | Value |
|---|---|---|
| `msl_codegen_cache` | `MslCodegenKey` { entry_point, binding_map: `MslBindingMap`, constants, subpass_color_slots: `Vec<((u32,u32),u32)>`, vertex_buffers: `Vec<MslVertexBufferBinding>`, disable_robustness, emit_vertex_point_size, fixed_sample_mask, user_immediate_size } | `Result<GeneratedMsl, String>` |
| `spirv_codegen_cache` | `SpirvCodegenKey` { entry_point, constants, vulkan_memory_model, framebuffer_fetch_descriptor_set, multisampled_input_attachment, polyfill_pixel_center: `Option<u32>`, user_immediate_size } | `Result<Vec<u32>, String>` |
| `glsl_codegen_cache` (`#[cfg(feature = "gles")]`) | `GlslCodegenKey` { entry_point, stage: `ShaderStage`, constants } | `Result<GeneratedGlsl, String>` |
| `workgroup_resolve_cache` | `WorkgroupResolveKey` { entry_point, constants } | `Result<ReflectedWorkgroupSize, String>` |

Notes:
- `generate_stage_msl` is the memoization point for MSL — it is the funnel
  for `generate_msl` (compute), `generate_render_vertex_msl`, and
  `generate_render_fragment_msl`. Its `vertex_buffers` parameter changes from
  `&[yawgpu_tint::VertexBuffer]` to `&[MslVertexBufferBinding]` (crate-local,
  hashable), with the `tint_vertex_buffers` conversion moved inside the miss
  branch. `yawgpu_tint::VertexBuffer` must not grow Eq/Hash.
- `generate_spirv`'s `_stage` parameter is dead (already ignored) and is
  excluded from `SpirvCodegenKey`.
- `resolved_compute_workgroup_size` memoizes its **entire** body: even the
  "fully literal" fast path crosses the FFI (`workgroup_storage_size`,
  IR-lowering-class work) on every call.
- New `Hash` derives required on `MslBindingMap`, `MslResourceBinding`,
  `MslResourceBindingKind`, `MslVertexBufferBinding`, `MslVertexAttribute`,
  `MslVertexStepMode`, `MslVertexFormat`, and `ShaderStage`
  (`yawgpu-core/src/shader_types.rs`; all fields already hashable; adding a
  derive to the `pub enum ShaderStage` is non-breaking).

### Locking

Each cache is `Mutex<HashMap<K, Result<V, String>>>`, and the mutex is held
**across** the Tint call (the `resource_bindings_for_entry` pattern,
including `PoisonError::into_inner` recovery). Rationale, to record in the
field doc comment: holding the lock dedups *concurrent identical* compiles
(the bug this block fixes); codegen on a shared `Program` is documented
concurrent-safe (`yawgpu-tint/src/lib.rs` Send/Sync notes), so this is a
policy choice, not soundness; the only nested cache call
(`resource_bindings_for_entry`) uses a different mutex — no lock-order
cycle; per-generator mutexes keep unrelated generators unserialized. A
future real-async pipeline-creation pool should revisit.

### Observability

`codegen_misses: AtomicUsize` on `ReflectedModule`, incremented once per
actual Tint codegen/resolve execution (i.e. in each miss branch), plus
`pub(crate) fn codegen_miss_count(&self) -> usize`. This is the unit-test
observable.

### `shader-passthrough` / Noop interaction

None: passthrough modules never construct a `ReflectedModule`; Noop pipeline
creation never calls `generate_*` but does call
`resolved_compute_workgroup_size` (covered by `workgroup_resolve_cache`).
All memoization is unit-testable without a GPU (Tint is pure CPU work; the
existing `shader_tint.rs` tests already call the generators directly).

### S1 unit tests (in `shader_tint.rs` `mod tests`, asserting
`codegen_miss_count()` deltas + value equality)

1. `generate_msl_memoizes_identical_requests` — two identical calls → equal
   `GeneratedMsl`, 1 miss; different `user_immediate_size` → 2nd miss.
2. `generate_msl_key_is_constant_order_insensitive` — same constants,
   opposite insertion orders → 1 miss.
3. `generate_msl_distinguishes_constant_values` — 1.0 vs 2.0 → 2 misses,
   different `source`.
4. `generate_msl_memoizes_error_results` — unknown entry point: `Err` twice,
   identical message, 1 miss.
5. `generate_render_vertex_msl_memoizes_with_vertex_buffers` — identical
   vertex-buffer layout twice → 1 miss; changed attribute offset → 2.
6. `generate_render_fragment_msl_distinguishes_sample_mask` —
   `0xFFFF_FFFF` vs `0xF` → 2 misses; identical repeat → no new miss.
7. `generate_spirv_memoizes_identical_requests` — identical twice → 1 miss;
   flip `vulkan_memory_model` → 2.
8. `resolved_compute_workgroup_size_memoizes_per_constants` — same constants
   twice → 1 miss, equal size; `x=8` vs `x=16` → 2 misses, `[8,2,1]` /
   `[16,2,1]`.
9. `resolved_compute_workgroup_size_memoizes_const_eval_errors` — repeated
   failing resolve → 1 miss, cached `Err`; sibling passing entry point still
   resolves (separate key).
10. `generate_glsl_memoizes_identical_requests` (`#[cfg(feature = "gles")]`)
    — identical twice → 1 miss; `Vertex` vs `Compute` stage → separate
    entries.

Existing shader_tint tests must pass unchanged through the new caches.

## S2 — FFI pre-compile short-circuit (`yawgpu` + core counters)

### Contract

`wgpuDeviceCreateShaderModule`, `wgpuDeviceCreateComputePipeline`(+Async),
and `wgpuDeviceCreateRenderPipeline`(+Async) return the live cached handle
**without running the core compile** when all of the following hold:

- the descriptor produces a cache key (`Some`; auto-layout pipelines and
  unrepresentable descriptors stay uncached and always take the slow path);
- cross-device validation (`validate_compute_pipeline_devices` /
  `validate_render_pipeline_devices`) reports no error — validation runs
  **before** the cache lookup (defensive: a live per-device hit already
  implies a previously validated triple, but the ordering keeps the
  invariant explicit);
- the device is not lost (`device.core.is_lost()` false). A lost device must
  keep returning a fresh error-shaped object exactly as today, never a
  cached healthy one.

On a hit the existing handle `Arc` is returned (one strong ref added — the
same object identity the post-compile dedup already produced). Async
variants need no special casing: a hit flows into
`PendingCallback::Create*PipelineAsync` with `is_error() == false` and fires
the Success callback as today.

### Mechanics

- New `fn cached_handle<T: CacheKey>(cache: &Mutex<HashMap<T, Weak<T::Handle>>>, key: &T) -> Option<Arc<T::Handle>>`
  next to `cache_handle` (`yawgpu/src/ffi/mod.rs`): lock (same `.expect`
  style) → `get(key).and_then(Weak::upgrade)`. **No pruning** on the lookup
  path — pruning stays in `cache_handle`'s insert path so existing prune
  semantics/tests are unchanged. A dead `Weak` simply misses; the slow path
  then prunes + reinserts via `cache_handle`.
- Short-circuit sites: `create_compute_pipeline_handle`,
  `create_render_pipeline_handle` (`ffi/mod.rs`), and
  `wgpuDeviceCreateShaderModule` (`ffi/device.rs`, after
  `shader_module_cache_key` is computed from the mapped source).

### `canonical_f64_bits` fix (pre-existing wrong-hit bug)

`canonical_f64_bits` (`ffi/mod.rs`) folds `-0.0` into `+0.0`, so a pipeline
created with constant `-0.0` after one with `+0.0` dedups to the
`+0.0`-compiled pipeline — wrong today, and the short-circuit would make the
wrong hit also skip recompilation. Fix: drop the zero-folding (keep keys as
raw `to_bits()`; NaN never reaches the cache past validation — if a NaN
folding branch is kept for defensive reasons it must fold NaNs only, never
signed zeros). Regression test:
`pipeline_cache_distinguishes_negative_zero_constants`.

### Observability — core creation counters

Model: `Device::allocation_count()` (`yawgpu-core/src/device.rs`). Three
always-on `AtomicU64` counters on core `Device` with `pub` accessors +
`///` docs + direct unit tests (CLAUDE.md principle 1):

- `shader_module_creation_count` — bumped at the top of
  `create_shader_module` (every call, including error paths);
- `compute_pipeline_creation_count` — `create_compute_pipeline` +
  `create_compute_pipeline_without_error_dispatch`;
- `render_pipeline_creation_count` — `create_render_pipeline` +
  `create_render_pipeline_without_error_dispatch`.

`create_subpass_render_pipeline` is **not** counted (not routed through the
FFI cache; keep scope tight).

### S2 unit tests

Core (`device.rs` tests):
1. `device_creation_counts_increment_per_create` — each `create_*` bumps its
   counter by 1, including error-descriptor and lost-device paths.

FFI (Noop, `ffi/mod.rs` tests, next to the existing cache tests):
2. `shader_module_cache_hit_skips_core_compile` — identical WGSL twice →
   same handle, `shader_module_creation_count() == 1`; release both, create
   again → count 2 (dead entry recompiles).
3. `compute_pipeline_cache_hit_skips_core_compile` — explicit layout,
   identical twice → same handle, count 1; different entry point or constant
   → count 2.
4. `render_pipeline_cache_hit_skips_core_compile` — same shape (vertex-only
   pipeline suffices).
5. `auto_layout_pipeline_creates_are_never_short_circuited` — `layout =
   null`, identical twice → count 2, nothing cached.
6. `pipeline_cache_short_circuit_still_validates_device_mismatch` —
   module+layout from device B used on device A twice → validation error
   both times (pop error scope twice), count 2, nothing cached.
7. `compute_pipeline_async_cache_hit_completes_with_success_callback` —
   sync create (cached) then identical Async → Success callback, non-null
   pipeline, count still 1.
8. `shader_module_cache_hit_skipped_when_device_lost` — lose device after a
   cached create; repeat create takes the slow path (count increments) and
   returns the lost/error-shaped module as today.
9. `pipeline_cache_distinguishes_negative_zero_constants` — `+0.0` then
   `-0.0` constant → two distinct pipelines, count 2.

Existing cache tests (`shader_module_cache_dedups_live_and_reclaims_released`,
`cache_handle_*`) must pass unchanged.

## Verification (S3)

- Noop gates: `cargo test --workspace`, clippy `-D warnings`, feature-gated
  HAL/lib runs (`-p yawgpu-hal --features metal --lib`, gles lib build).
- Real Metal (M2): `yawgpu/tests/e2e_metal_compile_churn.rs` (#[ignore],
  canary-shaped — Phase A: one module, N identical auto-layout compute
  pipelines created/released in a loop, timed; Phase B: N unique-constant
  variants). Pre-fix baseline recorded; post-fix expectation: Phase A
  per-iteration time collapses after iteration 1 and
  `A_median < 0.30 × B_median` (the external canary's exit-0 contract).
- CTS spot-check on Metal with parallel workers: no `MTLCompilerService` /
  AGX degrade signatures; pass counts match the ledger (no conformance
  drift).
- The deferred HAL layer (MTLLibrary-by-source cache, `VkPipelineCache`) is
  reconsidered only if these measurements still show compiler-service
  pressure.
