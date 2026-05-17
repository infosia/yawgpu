# Dependencies & pinned references

## Reference projects (read-only, not vendored)

| Path (from `yawgpu/`) | Role |
|---|---|
| `webgpu-headers/webgpu.h` | Canonical WebGPU C header — the spec we implement (6766 lines) |
| `dawn` | Behaviour spec + ported test source (`src/dawn/tests/unittests/validation`) |
| `wgpu-native` | C ABI structure inspiration (bindgen rename trick, Arc handles, conv.rs) |
| `mgpu` | Project layout + enum-dispatch HAL template + conventions |
| `wgpu/naga` | WGSL compiler — **path dependency** (decided) |

## Crate dependencies

- **naga**: `path = "wgpu/naga"`. Decided over crates.io to track the same
  source the reference stack uses. Needed from Phase 4 (shaders).
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
- `naga` revision: _TBD at Phase 4_ (path dep tracks `wgpu` working tree)
