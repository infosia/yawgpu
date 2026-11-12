# Block 151 — Whole-code review 2026-11-13: small P1 fixes

Status: **COMPLETE (2026-11-13)**.
Source: whole-code review at `92e38c7` (`REPORT-review.md`), findings R02, R03, R04, R09, R10,
and R12. These are the review's localized P1 findings, plus R10 (same validation function as
R04) and R12 (needed so that R09's new allocation error reaches the right error scope).
Oracle: pinned Dawn — `native/ExternalTexture.cpp` (`ValidateExternalTextureDescriptor`,
plane ownership and component counts), `native/Device.cpp` (`CreateQuerySet` error kinds).

## Rules

R1. **Error textures allocate no initialization state (review R02).** `Texture::new` with
`is_error == true` (validation failure, lost device, or HAL allocation failure) builds a
disabled, empty `TextureInitState`: no per-subresource vector is allocated, whatever the
descriptor's mip and layer counts are. The getters still return the descriptor values.
Valid textures keep their current tracking, sized from validated dimensions. A descriptor
with `mip_level_count == u32::MAX` and `depth_or_array_layers == u32::MAX` produces an
error texture with no panic and no large allocation on both the live-device and the
lost-device paths. On a live device it reports one validation error; on a lost device
it reports nothing (`dispatch_error` stays silent after loss, unchanged).

R2. **No safe public function follows caller-supplied raw chain pointers (review R03).**
`yawgpu::conv::map_texture_view_descriptor` becomes `unsafe fn` with a `# Safety` section:
`value.nextInChain`, when non-null, must point to a valid chain of `WGPUChainedStruct`s
that stays live for the call. Callers are updated. Any other `pub` safe function in
`yawgpu::conv` that dereferences a raw pointer read from its argument (a chain pointer, an
array pointer, a string pointer) gets the same treatment; list each one in the report.

R3. **External-texture planes must belong to the creating device (review R04).**
`yawgpuDeviceCreateExternalTexture` rejects a `plane0` or `plane1` view whose owning
device is not the creating device (`core::Device::same` on the FFI view's `_device` and the
device's core). The result is a validation error on the creating device and a null
return, like the existing descriptor errors. Message: `external texture plane0 must
belong to the same device` (and `plane1`).

R4. **External-texture plane component counts (review R10).**
`validate_external_texture_descriptor` checks the plane view format's
`FormatCaps::color_components`: 4 for an RGBA external texture's `plane0`; 1 for an NV12
`plane0` and 2 for an NV12 `plane1`. The check sits beside the existing filterable-float
rule. Message names the plane, the component count, and the required count (Dawn: "The
format (%s) component count (%u) is not %u."). Existing tests and examples that build
external textures from valid formats keep passing.

R5. **A missing Metal occlusion buffer is an allocation failure (review R09).**
`MetalQuerySet::new` maps a `None` result of `newBufferWithLength_options` for the
occlusion visibility buffer to `HalError::OutOfMemory`, as the timestamp branch already
does. The query set is not constructed.

R6. **Query-set creation keeps the error kind (review R12).**
`Device::create_query_set` returns its error as a typed `DeviceError` (kind + message)
instead of `Option<String>`. Descriptor validation failures are `Validation`;
`HalError::OutOfMemory` is `OutOfMemory`; every other HAL error is `Internal` (the same
mapping as `dispatch_hal_allocation_error`). `wgpuDeviceCreateQuerySet` dispatches with
that kind. The lost-device path stays silent.

## Slices

S1: R1–R6, with inline unit tests next to each changed function. One slice: each rule is a
small local change.

## Verification

- Inline unit tests: R1 (extreme mip × layer descriptor on a live Noop device: one
  validation error; on a lost one: no error; both an error texture with disabled, empty
  init state and no panic), R2 (existing conv
  tests updated to call through `unsafe`), R3 (plane0 and plane1 from another Noop device
  rejected; same-device accepted), R4 (RGBA with an `r8unorm` plane0 rejected; NV12 with
  `rgba8unorm` plane0 or `r8unorm` plane1 rejected; the valid layouts accepted), R6 (a
  HAL query-set failure maps to `OutOfMemory`/`Internal`, validation stays
  `Validation`). R5 is a one-line mapping; it is covered by compiling the Metal HAL and
  by the existing Metal occlusion e2e tests.
- Agent gates: `cargo test -p yawgpu-core --lib`, `cargo test -p yawgpu --lib`,
  the `yawgpu` integration tests that cover external textures and query sets,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo clippy -p yawgpu-hal --features metal --all-targets -- -D warnings`.
- Claude: workspace test, `cargo test -p yawgpu-hal --features metal --lib`, and the
  Metal external-texture and occlusion-query e2e suites on the M2.

## Verification record

S1: workspace test 1,451/0; `yawgpu-core --lib` 725/0; `yawgpu --lib` 307/0; clippy clean
(workspace and `yawgpu-hal --features metal`). Real GPU on the M2: `yawgpu-hal --features
metal --lib` 201/0 plus the 7 ignored query-set tests; `e2e_metal_external_texture` 6/6,
`e2e_metal_timestamp_query` 4/4, `e2e_metal_depth` 20/20, `e2e_metal_immediates` 3/3,
`e2e_metal_draw` 3/3. The R2 audit found no other safe `pub` function in `yawgpu::conv`
that dereferences a caller-supplied raw pointer.

## Phase Review (2026-11-13, fresh-context reviewer over `92e38c7..1855795`)

0 CRITICAL, 0 MAJOR, 2 MINOR.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| m1 | MINOR | R3 compares the outer `Arc<core::Device>` with `Arc::ptr_eq`; every other FFI cross-device check uses `core::Device::same` | R3 now reads "the same device (`core::Device::same`)"; the code change rides with the Block 152 fix handoff |
| m2 | MINOR | No FFI-level test that `wgpuDeviceCreateQuerySet` passes `OutOfMemory`/`Internal` through | Accepted: the FFI line only forwards `error.kind`; the core test covers the mapping, and the FFI test module has no HAL fault injection |
