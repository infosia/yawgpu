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
  added Phase 7, feature-gated. Noop has no GPU deps. **Vulkan and Metal are
  the only real backends; OpenGL/GLES and DirectX are out of scope** (no GL/
  D3D crates pulled in).
- Workspace-wide: `thiserror`, `parking_lot`, `bitflags`, `smallvec`,
  `raw-window-handle` (mirrors wgpu-native/mgpu).

## Pinning

Record the `webgpu.h` upstream commit/version here once Phase 0 vendors or
references it, so bindgen output is reproducible:

- `webgpu.h`: vendored at `yawgpu/ffi/webgpu-headers/webgpu.h` (6766 lines),
  byte-identical copy from `webgpu-headers` commit
  `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603` (2026-05-07). Bound via
  `bindgen 0.72` in `yawgpu/build.rs`.
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

- **objc2 0.6 / objc2-foundation 0.3 / objc2-metal 0.3**: the Metal
  backend was migrated from the deprecated `metal 0.33.0` crate during
  the P9.0 review fixes (2026-05-20), matching the working `mgpu`
  Metal backend's objc2 family. This resolves the Phase-7 Review MINOR
  about the deprecated `objc` ecosystem and keeps yawgpu on a
  maintained binding set. The Phase-7 Metal HAL surface is preserved
  unchanged — all `e2e_metal_*` tests pass on the host after migration
  (basic/buffer/texture/compute/render/smoke). Dependencies remain
  optional and are enabled only by the `metal` cargo feature, so
  default Noop builds do not compile or link Objective-C/Metal crates.
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
