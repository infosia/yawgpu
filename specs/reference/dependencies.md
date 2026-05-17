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
- `naga`: pinned by `infosia/wgpu` commit SHA. Initial pin candidate
  `216627076` (`wgpu` HEAD, describes as `v29.0.3-44-g216627076`).
  Set the concrete `rev` when naga is first wired in (Phase 4).
  License: MIT OR Apache-2.0 (`wgpu/naga/LICENSE.{MIT,APACHE}`),
  compatible with this repo.

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
