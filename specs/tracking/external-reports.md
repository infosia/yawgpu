# External bug reports — ledger

Findings reported by external clients (via the repo-root `HANDOFF.md`
drop, which is gitignored) that are not CTS findings (`F-xxx`, ledger
`specs/tracking/cts-coverage.md`) and not perf findings (`P-xxx`,
ledger `specs/tracking/perf-dawn-baseline.md`). Each entry records the
report, the root cause, the fix, and how it was verified.

## 2026-08-09 — surface configure rejects the header's zero sentinels

**Reported by:** subscript-gpu (windowed triangle example), against a
`4e961e6`-era tree, `metal` release build.

**Symptom.** A surface configured from `WGPU_SURFACE_CONFIGURATION_INIT`
(only device/format/usage/size set) never produces a texture:
`wgpuSurfaceConfigure` silently dispatches "surface configuration
present mode is not supported" to the device error sink, `configured`
stays `None`, and every `wgpuSurfaceGetCurrentTexture` reports
`Error` with a null texture.

**Root cause.** `surface_configuration_error`
(`yawgpu/src/ffi/mod.rs`) validated `presentMode` / `alphaMode` by
membership in `SURFACE_PRESENT_MODES = [Fifo]` /
`SURFACE_ALPHA_MODES = [Opaque]`, but the pinned `webgpu.h` defines
both zero values as valid configure inputs:
`WGPUPresentMode_Undefined` (0) "defaults to Fifo", and
`WGPUCompositeAlphaMode_Auto` (0) is "an alias for the first element"
of the reported `alphaModes`. The `INIT` macro sets exactly these two
zeros, so the header's own defaults were unconfigurable. Downstream
`hal_present_mode` already mapped `Undefined → Fifo`; only the
validation ahead of it was wrong.

**Fix.** Resolve the sentinels before the membership check
(`resolved_present_mode`: `Undefined → Fifo`; `resolved_alpha_mode`:
`Auto → SURFACE_ALPHA_MODES[0]`, i.e. first-capability alias, not a
hardcoded constant), store and forward the *resolved* modes.
Membership checks, error messages, and `wgpuSurfaceGetCapabilities`
(concrete modes only, never sentinels) are unchanged; explicit
unsupported modes (`Immediate`, `Mailbox`, `FifoRelaxed`,
`Premultiplied`, `Unpremultiplied`, `Inherit`, out-of-range) still
fail with the existing messages. Contract recorded in
`specs/blocks/70-finalize.md` → Surface → "Sentinel resolution".

**Verification.**

- Inline unit tests: resolver behaviour + `surface_configuration_error`
  accepts the sentinel pair and still rejects each explicit
  unsupported mode with the exact existing message.
- Noop integration (`yawgpu/tests/surface_validation.rs`): INIT-default
  configure dispatches no error and reaches the configured-Noop
  `Lost` boundary; explicit `Immediate` / `Premultiplied` still error.
- Real-Metal e2e (`yawgpu/tests/e2e_metal_surface.rs`, new): the
  report's repro as a test — standalone `CAMetalLayer`, configure with
  both sentinels, `getCurrentTexture` returns `SuccessOptimal` with a
  non-null texture, `wgpuSurfacePresent` succeeds; companion test
  keeps explicit unsupported modes erroring on real Metal. Full
  real-Metal e2e suite: 105 passed / 0 failed.
- CTS: the harness never calls `wgpuSurface*` (verified by grep over
  `webgpu-native-cts/src`), so the changed functions are unreachable
  from CTS; the Run-4 Metal tree set (8 `api,operation` subtrees,
  `--workers 6`) was still re-run as a before/after failure-set diff:
  `pass=175738 skip=47 warn=0 fail=0 crash=0` — identical to the
  baseline, failure set empty before and after.

## 2026-08-10 — `libyawgpu.so` cannot resolve its sibling `libtint_shim.so`

**Reported by:** external consumer (HANDOFF drop), against `96b380c`,
x86_64 Linux, release build, glibc loader.

**Symptom.** Any consumer process that loads `libyawgpu.so` as a shared
object fails at startup: `error while loading shared libraries:
libtint_shim.so: cannot open shared object file`. Workaround required
`LD_LIBRARY_PATH` on every developer machine and CI job. yawgpu's own
test suite never showed it — test binaries link `tint_shim` directly,
so their own `DT_RUNPATH` resolves it; the failure needs the shim as a
**transitive** `NEEDED` of the cdylib.

**Root cause.** `DT_RUNPATH` is not transitive: it resolves only the
`NEEDED` entries of the object carrying it, so the consumer's own
`RUNPATH` never applies to `libyawgpu.so`'s `NEEDED libtint_shim.so`.
`yawgpu/build.rs` handled Apple (`@rpath` install name; the shim's
`@loader_path` name) and `yawgpu-tint`'s `copy_runtime_shim` handled
Windows (image-directory DLL copy) and already places
`libtint_shim.so` beside the cdylib on ELF too — but no ELF image
pointed at that directory: `readelf -d target/release/libyawgpu.so`
showed `NEEDED libtint_shim.so` and no `RUNPATH` line. The
`yawgpu-tint/build.rs` rpath cannot fix it: `cargo:rustc-*link-arg`
does not propagate from a dependency's build script to the dependent's
cdylib link.

**Fix.** `yawgpu/build.rs`: beside the Apple `install_name` branch, an
ELF branch (`CARGO_CFG_TARGET_FAMILY == "unix"` and vendor not
`apple`, so Linux + Android; never Windows/wasm — target detection,
not host) emits `cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN`
(`$ORIGIN` written literally; cargo passes the arg to rustc with no
shell in between). The `yawgpu-tint/build.rs` rpath and the
Apple/Windows paths are untouched. Contract recorded in
`specs/reference/dependencies.md` → "ELF consumers — `$ORIGIN` rpath
on the cdylib".

**Verification.**

- New Linux-only integration tests (`yawgpu/tests/linkage_elf.rs`):
  `cdylib_carries_origin_runpath` parses the built cdylib's ELF64
  `.dynamic` section directly (no `readelf` dependency) and asserts a
  `$ORIGIN` component in `DT_RUNPATH`/`DT_RPATH`; it reads
  `<profile>/deps/libyawgpu.so` first because the profile-root file is
  an uplifted hardlink that test-only builds do not refresh.
  `consumer_loads_cdylib_without_ld_library_path` copies the cdylib
  (+ shim when present, absent in Dawn-less stub builds) into a fresh
  `CARGO_TARGET_TMPDIR` dir and loads it from a fresh python3 child
  process with `LD_LIBRARY_PATH` removed — a fresh process because the
  test binary itself links the shim, so an in-process `dlopen` would
  falsely succeed.
- Red-first at `96b380c`: both tests failed before the build.rs change
  (`no $ORIGIN component in DT_RUNPATH or DT_RPATH (found [])`;
  child `OSError: libtint_shim.so: cannot open shared object file`),
  and passed after.
- `readelf -d` on the rebuilt cdylib now reports
  `(RUNPATH) Library runpath: [$ORIGIN]`.
- Full Noop gate on Linux: `cargo test --workspace` 987 passed /
  0 failed; `cargo clippy --workspace --all-targets -- -D warnings`
  clean.
- macOS: unchanged by inspection (the new branch is unreachable for
  `vendor == apple`); the report's criterion "re-confirm the
  `@loader_path` install name on Apple" is deferred to the next run on
  a macOS host, as this fix was produced on a Linux-only host.

## 2026-08-24 — `texture-formats-tier2` advertised without a device query

**Reported by:** subscript-typegpu (downstream GPU project), against
`13ac0b4`, Metal. Stated from the code paths — the reporter's machines
report tier 2, so the consequence could not be reproduced downstream.

**Symptom.** `MetalAdapter::supports_texture_formats_tier2()` returned
`true` unconditionally, so on a Metal device below read-write texture
tier 2 yawgpu advertised the feature, accepted a `read-write` bind group
layout on a tier-2 format such as `rgba8unorm`, and handed Tint MSL to a
compiler whose device cannot execute it — the failure surfaced late
(Metal compiler / wrong results) instead of at the feature query. Dawn
gates the feature on `[device readWriteTextureSupport] ==
MTLReadWriteTextureTier2` (`PhysicalDeviceMTL.mm`); `readWriteTextureSupport`
appeared nowhere in the yawgpu tree.

**Root cause.** The Metal feature table (`yawgpu-hal/src/metal/mod.rs`)
hard-coded tier 2 as supported; the core `read-write` rejection message
also did not name the format or the feature, so a client on a tier-1
device would have had no pointer to the missing feature.

**Fix.** Contract: `specs/blocks/72-texture-formats-tier2.md`.
`MetalAdapter::new` queries `readWriteTextureSupport` once and caches
the `MTLReadWriteTextureTier`; `supports_texture_formats_tier2()` is
`tier == Tier2` (Dawn's rule). Core: when a `read-write` storage-texture
layout entry is rejected and the format *would* be read-write capable
with `TextureFormatsTier2` (evaluated via `caps()` against
`features ∪ {Tier2}`, no second table), the message names both —
`storage texture binding format rgba8unorm supports read-write storage
access only with the texture-formats-tier2 feature, which this device
does not have`; formats that never support read-write (`rg32float`)
keep the existing generic message. `TextureFormat::name()` (WebGPU IDL
names, all 102 header formats) was added for the message. Tier 1,
Noop, Vulkan and GLES advertisement are untouched.

**Vulkan analogue (Slice 2, fixed in the same session).**
`VulkanAdapter::supports_texture_formats_tier{1,2}()` were also unconditional
`true`, and `create_device` never enabled `shaderStorageImageExtendedFormats`
even though the tier 1/2 storage formats (`r8unorm`, `r16float`, `rg8*`, …)
are Vulkan extended storage-image formats. Now both tiers return one
`supports_texture_formats_tiers()` implementing Dawn's `PhysicalDeviceVk`
rule verbatim (feature bit + 10 formats colour-attachment-renderable and
blendable in optimal tiling), and device creation enables the feature
whenever the physical device supports it.

**Verification.**

- Inline unit tests: `TextureFormat::name()` sample + completeness
  (every `caps()`-recognised raw has a name); BGL message names
  `rgba8unorm` + `texture-formats-tier2` without the feature, generic
  message for `rg32float` with it, no error for `rgba8unorm` with it.
- Metal HAL unit (`#[ignore]`, real GPU): cached tier equals a fresh
  `readWriteTextureSupport()` query and `supports_texture_formats_tier2()`
  is `tier == Tier2` — 42/42 ignored Metal HAL tests green on the M2.
- Noop integration (`yawgpu/tests/bind_group_validation.rs`): device
  without the feature → error naming format + tier; with it → no error;
  `rg32float` with it → generic message.
- Real-Metal e2e (`yawgpu/tests/e2e_metal_texture_formats_tier2.rs`, new,
  3/3): advertisement equals a direct `MTLDevice.readWriteTextureSupport`
  oracle; a device without the feature rejects the `rgba8unorm`
  `read-write` layout with the tier-naming message; a device with the
  feature executes a `texture_storage_2d<rgba8unorm, read_write>` compute
  pass and the readback matches. Full real-Metal e2e suite: 104 passed / 0 failed.
- Noop workspace suite (88 binaries) green; clippy `-D warnings` clean on
  default / `metal` / `vulkan` / `gles,tiled`; `cargo fmt --check` clean.
- CTS (Metal, M2 — reports tier 2, so no behaviour change expected):
  `capability_checks,features,texture_formats{,_tier1,_tier2}`,
  `createBindGroupLayout`, `storage_texture,{read_only,read_write}` —
  `pass=2056 skip=263 warn=0 fail=0 crash=0` before and after, non-pass
  set diff empty.
- Slice 2 (Vulkan): real-Vulkan HAL unit test recomputes Dawn's rule from raw
  `ash` queries and asserts both tiers equal it; the create-device unit test
  covers `shader_storage_image_extended_formats` forwarding; real-Vulkan e2e
  (`yawgpu/tests/e2e_vulkan_texture_formats_tier2.rs`, MoltenVK on the M2):
  3/3 — device without the feature rejects the `rgba8unorm` `read-write` layout with the tier-naming message, and a device with the feature executes `read_write` compute passes on `rgba8unorm` and on the extended format `r8unorm` with matching readback (proves the device-level feature enablement); real-Vulkan HAL ignored tests 39/39, full real-Vulkan e2e suite 67 passed / 0 failed. CTS (MoltenVK, same tree set):
  `pass=2014 skip=305 warn=0 fail=0 crash=0` before and after, non-pass set
  diff empty (MoltenVK on the M2 satisfies the rule, so the advertisement is
  unchanged there).
