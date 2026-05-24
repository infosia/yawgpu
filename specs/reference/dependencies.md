# Dependencies & pinned references

## Reference projects (read-only, not vendored)

| Path (from `yawgpu/`) | Role |
|---|---|
| `webgpu-headers/webgpu.h` | Canonical WebGPU C header — the spec we implement (6766 lines) |
| `dawn` | Behaviour spec + ported test source (`src/dawn/tests/unittests/validation`) |
| `wgpu-native` | C ABI structure inspiration (bindgen rename trick, Arc handles, conv.rs) |
| `mgpu` | Project layout + enum-dispatch HAL template + conventions |
| `wgpu` | `infosia/wgpu` fork (gfx-rs/wgpu `v29.0.3` + `tiled-fork` patches); source of `naga` — **git+rev dependency**, local `path` overlay for dev (decided) |

## Crate dependencies

- **naga**: consumed from the `infosia/wgpu` fork as a **git dependency
  pinned by commit SHA** (not crates.io — the fork carries `tiled-fork`
  patches the reference stack needs; not a bare `path` dep — CI must build
  without a sibling `wgpu` checkout). Needed from Phase 4 (shaders).
  ```toml
  [workspace.dependencies]
  naga = { git = "https://github.com/infosia/wgpu.git", rev = "<SHA>" }
  ```
  Local fast-iteration only (never committed): overlay with
  `[patch."https://github.com/infosia/wgpu.git"] naga = { path = "wgpu/naga" }`.
- **bindgen** (build-dep): generate `webgpu.h` bindings.
- HAL backends: `ash` (Vulkan), `objc2`/`objc2-metal`/`block2` (Metal) —
  added Phase 7, feature-gated. Noop has no GPU deps. **DirectX is the only
  permanently out-of-scope backend family.** GLES is **Tier 2 /
  experimental** (`glow` / `khronos-egl` / `libloading`, added Phase 15,
  feature-gated; see "Phase 15 / GLES backend" below).
- Workspace-wide: `thiserror`, `parking_lot`, `bitflags`, `smallvec`,
  `raw-window-handle` (mirrors wgpu-native/mgpu).

## Pinning

Record the `webgpu.h` upstream commit/version here once Phase 0 vendors or
references it, so bindgen output is reproducible:

- `webgpu.h`: vendored at `yawgpu/ffi/webgpu-headers/webgpu.h` (6766 lines),
  byte-identical copy from `webgpu-headers` commit
  `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603` (2026-05-07). Bound via
  `bindgen 0.72` in `yawgpu/build.rs`.
  bindgen workaround for Windows MSVC builds (added 2026-05-21):
  `.clang_macro_fallback()` plus blocklist+raw_line overrides for the four
  `(SIZE_MAX)`/`(UINT64_MAX)`-derived constants (`WGPU_WHOLE_MAP_SIZE`,
  `WGPU_WHOLE_SIZE`, `WGPU_LIMIT_U64_UNDEFINED`, `WGPU_STRLEN`). Without the
  overrides bindgen emits them as `i32 = -1` regardless of the C-side type,
  which silently works on macOS but breaks Windows MSVC with `cannot find
  value`. The `native::WGPU*` conversion paths in `yawgpu/src/conv.rs` use the
  primitive `From<u32>`/`From<i32>` impls in `yawgpu-core` to handle the
  separate enum underlying-type platform difference (MSVC `c_int`, macOS clang
  `c_uint`) by bit-preserving `as` casts at the FFI↔core boundary.
- `naga`: pinned to `infosia/wgpu` rev
  **`216627076a7b22ad09fa566de53d1f0f74b59de3`** (`git describe`:
  `v29.0.3-44-g216627076`; remote `https://github.com/infosia/wgpu.git`).
  Wired in Phase 4 (P4.0): `[workspace.dependencies] naga = { git =
  "https://github.com/infosia/wgpu.git", rev =
  "216627076a7b22ad09fa566de53d1f0f74b59de3" }`. Cargo.lock records the
  SHA. Local dev overlay (never committed):
  `[patch."https://github.com/infosia/wgpu.git"] naga = { path =
  "wgpu/naga" }`. naga is `edition 2021`; License MIT OR Apache-2.0
  (`wgpu/naga/LICENSE.{MIT,APACHE}`), compatible with this repo.

## naga upstream-tracking model (two tiers)

Tracking is split so each tier has one owner:

| Tier | Direction | Owner | Mechanism |
|---|---|---|---|
| 1 | `infosia/wgpu` ← `gfx-rs/wgpu` | the `wgpu` fork repo | rebase/merge `tiled-fork` onto upstream tags; **out of yawgpu's scope** |
| 2 | `yawgpu` ← `infosia/wgpu` | yawgpu (this repo) | `naga` git dep pinned by `rev` (commit SHA) |

Rules:

- yawgpu never follows a moving branch — always a fixed `rev`.
- **During the yawgpu development phases the fork is NOT tagged**
  (decided); Tier 2 stays on `rev` (SHA) pins. Cargo.lock records the SHA,
  so reproducibility equals tag pinning.
- A `naga` bump is its own slice: bump `rev` → run the full Noop
  validation suite → log in `specs/tracking/phase-N.md`. The ported Dawn
  tests are the regression gate.
- Each bump records here: new SHA, the gfx-rs upstream version it sits on,
  date, and reason.

### Future task (post-development)

After the yawgpu development phases complete: cut a stable tag on
`infosia/wgpu` (e.g. `yawgpu-base-v29.0.3`) and migrate the Tier 2 pin
from `rev = "<SHA>"` to `tag = "<tag>"`. Track as a one-off migration
slice; not done during active development.

## Phase 7 / real backends

- **objc2 0.6 / objc2-core-foundation 0.3 / objc2-foundation 0.3 /
  objc2-metal 0.3 / objc2-quartz-core 0.3**: the Metal
  backend was migrated from the deprecated `metal 0.33.0` crate during
  the P9.0 review fixes (2026-05-20), matching the working `mgpu`
  Metal backend's objc2 family. This resolves the Phase-7 Review MINOR
  about the deprecated `objc` ecosystem and keeps yawgpu on a
  maintained binding set. The Phase-7 Metal HAL surface is preserved
  unchanged — all `e2e_metal_*` tests pass on the host after migration
  (basic/buffer/texture/compute/render/smoke). Dependencies remain
  optional and are enabled only by the `metal` cargo feature, so
  default Noop builds do not compile or link Objective-C/Metal crates.
  P9.2 adds `objc2-core-foundation` + `objc2-quartz-core` for the
  real CAMetalLayer swapchain path (`WGPUSurfaceSourceMetalLayer`);
  they are also `metal`-feature-only.
  **Architectural follow-up (resolved 2026-05-20):** `wgpuBuffer
  GetMappedRange` was switched to return the real backend's
  persistently-mapped pointer (Metal `MTLBuffer.contents()` on
  Shared storage; Vulkan persistent `vkMapMemory` HOST_VISIBLE|
  COHERENT) via a new `HalBuffer::mapped_ptr()` accessor, with
  `Buffer::unmap` write-through and `resolve_pending_map` read-copy
  skipped when `mapped_ptr.is_some()` — mirroring `mgpu`'s
  direct-mapping model. Noop falls back to the core `HostBuffer`
  unchanged. After this fix the C `examples/compute` reads the real
  Collatz `[0,1,7,2]` on Metal AND Vulkan/MoltenVK; full Phase-7
  e2e regression remains green on the host.
  *New known issue (separate, tracked as Phase-9 follow-up):*
  yawgpu's naga MSL backend does not emit the "sizes buffer" slot
  required for storage buffers declared as **runtime-sized arrays**
  (`var<storage> values: array<u32>;`). Such shaders fail compute-
  pipeline creation on Metal with `mapping for sizes buffer is
  missing`. Fixed-size arrays (`array<u32, N>`) compile cleanly on
  Metal (and Vulkan/Noop). `examples/compute/shader.wgsl`
  consequently uses `array<u32, 4>` matching the input length; the
  same restriction applies to `mgpu`'s `hello_compute` shader
  (`array<u32, 256>`). Supporting runtime-sized storage arrays on
  Metal requires extending the binding map with a sizes-buffer
  argument and wiring it from compute-pipeline reflection through
  to dispatch.
- **ash 0.38.0+1.3.281**: selected in P7.6a as the latest stable
  `ash` crate release visible on docs.rs/crates.io at implementation
  time (2026-05-19). It is wired as optional `yawgpu-hal` and
  `yawgpu-test` dependencies and enabled only by the `vulkan` cargo
  feature. P7.6a also adds naga's `spv-out` feature to `yawgpu-core`
  so the Vulkan SPIR-V shader path can be implemented in P7.6d without
  another dependency-graph change.

## Phase 15 / GLES backend (Tier 2 / experimental)

- **glow 0.14**: GL function loader, wired as an optional
  `yawgpu-hal` dependency behind the `gles` feature (P15.0,
  2026-05-24). Selected to match wgpu-hal's workspace pin. No
  function pointers are loaded in P15.0; the dep exists so the
  scaffold module can `use glow as _;` to prove linkage. Real
  loader bring-up lands in P15.1.
- **khronos-egl 6** (`features = ["dynamic"]`): EGL dynamic
  binding, wired as an optional `yawgpu-hal` dependency behind
  the `gles` feature (P15.0). The `dynamic` feature selects the
  `libloading`-backed runtime EGL resolver — required for both
  Android (`libEGL.so`) and Windows ANGLE (`libEGL.dll`) where
  the EGL library is not link-time available. Static linking is
  explicitly avoided.
- **libloading 0.8**: cross-platform `dlopen` shim used by
  `khronos-egl`'s dynamic backend and by the GLES HAL to resolve
  ANGLE-on-Windows EGL/GLES libraries via the
  `YAWGPU_ANGLE_PATH` env var (P15.1). Optional `yawgpu-hal` dep
  gated on `gles` (P15.0).
- **parking_lot** (workspace version, no version bump): added as
  an optional `yawgpu-hal` dep behind the `gles` feature in
  P15.1 to provide the `Mutex<()>` that serializes
  `eglMakeCurrent` + GL command issuance on the shared
  per-`HalDevice` GL context. Already present as a workspace
  dep for other crates; gating it on `gles` keeps the default
  Noop build's dep graph unchanged.

These four crates are pulled **only** with `--features gles` —
default Noop builds, Vulkan-feature builds, and Metal-feature
builds do not see them. Real EGL/GL calls land in P15.1
(`GlesInstance::new` initializes the display; `GlesAdapter::
create_device` creates a 3.1+ context + 1×1 pbuffer and loads
GL via `glow`; `GlesQueue::submit_empty` make-currents + flushes).
Verified on Windows ANGLE — `libEGL.dll` / `libGLESv2.dll`
discovered via the default Windows DLL search path (or
`YAWGPU_ANGLE_PATH` for an explicit directory).
