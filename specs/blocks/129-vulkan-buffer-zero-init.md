# Block 129 — Vulkan buffers read as zero

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 4 MINOR; all fixed,
see below. Metal is owed on a Mac host (R3).
Source: backlog A18, found in the Block 128 Windows native Vulkan CTS run.
Oracle: Dawn at the pinned `third_party/dawn`: `native/vulkan/BufferVk.cpp`
(`Buffer::Initialize`, `Buffer::EnsureDataInitialized`, `Buffer::ClearBuffer`).

## Problem

WebGPU requires a new buffer to read as zero. The Vulkan HAL's `create_buffer`
(`yawgpu-hal/src/vulkan/buffer.rs`) allocates memory, binds it, maps it persistently,
and never clears it. Vulkan does not guarantee that `vkAllocateMemory` returns zeroed
memory, and drivers hand back memory freed earlier in the same process.

On Windows NVIDIA (RTX 5060 Ti, driver 617.14), webgpu-native-cts
`api,operation,command_buffer,copyTextureToTexture:*` gives 2 or 3 "texture padding
mismatch" failures on every run (`r8snorm`, `r8uint`, `r8sint`, 3d): bytes the copy
never writes hold stale data. This reproduces with libraries back to at least
`2b3f4a7`. A diagnostic build that zero-filled the mapped memory in `create_buffer`
gave 0 failures in 4 of 4 runs. GLES had the same defect and fixed it eagerly
(`specs/blocks/67-gles-backend.md`, buffer row, 2026-07-08).

Dawn clears the user-visible range lazily. A Vulkan buffer records whether its data is
initialized, and the first use that reads it without overwriting it all records a
`vkCmdFillBuffer` (`InitializeToZero`, `ClearBuffer`). The allocation padding past the
requested size is cleared eagerly in `Buffer::Initialize`, on the host through the
mapping when the memory is host-visible. A buffer mapped at creation is initialized in
`BufferBase::MapAtCreation`. Dawn's Metal backend follows the same split
(`native/metal/BufferMTL.mm`): padding cleared at creation with a GPU clear, the data
range cleared lazily.

## Decision

Clear eagerly, on the host, in the HAL. Every Vulkan buffer yawgpu creates is
host-visible, host-coherent, and persistently mapped, so a host fill of the mapping
needs no command buffer, no barrier, and no initialized-state tracking in core. It is
the same choice GLES made. The cost is one host fill per created buffer, proportional
to its size. Dawn's lazy clear pays a GPU fill instead, and only for buffers that are
read before they are fully written. If that cost shows up in a bench, lazy clearing is
a later block; it is not needed for correctness. The fill also touches every page of
every allocation at creation. On a driver that commits memory lazily, such as
MoltenVK (see the heap-size guard in `create_buffer`), a large buffer now commits its
memory when it is created rather than when it is first used. A later change in CTS
memory use or run time on MoltenVK may trace back here.

## Rules

R1. Vulkan `create_buffer` writes zero over the whole mapped allocation
(`VkMemoryRequirements::size` bytes, which covers any allocation padding past the
requested size) after mapping succeeds and before it returns. The fill is on the host,
through the mapped pointer. No error path changes.

R2. Every buffer that goes through `create_buffer` is cleared, including core's staging
buffers and the HAL's internal temporaries. No caller opts out. The internal
compressed-clear buffer keeps its `vkCmdFillBuffer`; it stays correct, and removing it
is out of scope.

R3. Metal, GLES, and Noop are unchanged. GLES already clears (Block 67). Metal is not
cleared by yawgpu, and Dawn does clear it, so Metal may have the same defect. The Metal
e2e is only a regression guard, because a pass proves nothing if the driver did not
reuse memory. The check on a Mac host is the CTS `copyTextureToTexture:*` and
`resource_init,buffer` queries, run several times. A failure there opens a Metal block.

## Tests

- Unit (Vulkan HAL `#[ignore]`, real GPU): a new buffer's mapped bytes are all zero,
  for a size that is not a multiple of the allocation granularity, across the whole
  `VkMemoryRequirements::size`. Write a non-zero pattern through the mapping, drop the
  buffer, create buffers of the same size again, and assert each reads as zero.
- e2e (Claude): `yawgpu/tests/e2e_{vulkan,metal}_buffer_zero_init.rs`
  (`common/buffer_zero_init.rs`). Fresh buffers read as zero through `mapAsync` and
  through `mappedAtCreation`, after earlier rounds filled and released buffers of the
  same sizes. Driver memory reuse is the driver's heuristic, so this is a regression
  guard. A heavier variant failed once in four runs before the fix.
- Red check (Claude): CTS `api,operation,command_buffer,copyTextureToTexture:*` on
  Windows native Vulkan. Before: 2 or 3 failures on every run. After: 0.

## Out of scope

- Lazy, GPU-side clearing with initialized-state tracking, as Dawn does.
- Texture zero-initialization, which Block 104 covers.
- Device-local buffer memory. yawgpu has none today; a block that adds it must clear
  that memory on the GPU.

## Slices

- **S1 (coding agent):** R1-R3 with the unit test.
- **S2 (Claude):** e2e, red check, Vulkan e2e and CTS on Windows native Vulkan, Phase
  Review.

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351. The
Khronos validation layer was on for the e2e and HAL runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,324 passed, 0 failed |
| `cargo fmt --all -- --check`; clippy `-D warnings` on default and on `vulkan` (`yawgpu`, `yawgpu-hal`, all targets) | clean |
| Vulkan HAL ignored tests | 77 passed, 0 failed, 0 VUIDs, including `vulkan_buffer_create_zeroes_whole_allocation_after_reuse` |
| Vulkan e2e, `vulkan,tiled,shader-passthrough`, `--ignored` | 162 passed, 0 failed, 0 VUIDs, including both `e2e_vulkan_buffer_zero_init` cases |
| Red check: CTS `api,operation,command_buffer,copyTextureToTexture:*` | before: 2 or 3 failures in each of 3 runs (`5ecaafc`); after: pass 43,100, fail 0 in each of 3 runs |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, `expectations/yawgpu-vulkan.txt` | `fail=2 crash=0` |

The CTS ran as four per-area runs with `--workers 8` and took 28 minutes:

| Area | pass | skip | fail | xfail |
|---|---:|---:|---:|---:|
| `api,validation` | 246,484 | 111,283 | 0 | 4 |
| `api,operation` | 180,779 | 152,819 | 0 | 0 |
| `shader,execution` | 562,548 | 320,680 | 0 | 25 |
| `shader,validation` | 662,535 | 20,299 | 2 | 0 |
| **total** | **1,652,346** | **605,081** | **2** | **29** |

The 2 failures are the Block 119 R9 cases. The F-165 driver crash did not fire this run.
The totals equal the Block 125 run, which did not hit the `copyTextureToTexture`
failure either. That failure is intermittent at full-suite scale.

The heavier e2e variant described in `common/buffer_zero_init.rs` failed once in four
runs before the fix. The committed e2e is the lighter variant, as a regression guard.

Not exercised: Metal (R3). The Metal e2e and the Mac CTS check in R3 are owed to a Mac
host.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | The spec said Dawn clears lazily; Dawn clears the allocation padding eagerly, on the host for host-visible memory | Problem section corrected |
| m2 | Dawn's Metal backend was described as plainly lazy; it clears padding at creation | Corrected |
| m3 | The cost note omitted that the fill commits memory at creation on lazily committing drivers (MoltenVK) | Added to Decision |
| m4 | R3 said the Metal e2e checks Metal; a pass proves nothing without driver reuse | R3 names the Mac CTS queries as the check |
