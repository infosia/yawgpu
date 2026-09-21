# Pass-encoder resource retention (Block 50 post-COMPLETE addition)

Ledger for "Pass-encoder resource retention on `end()`" in
`specs/blocks/50-commands.md`. Opened 2026-09-21.

## Origin

webgpu-native-cts could not run its `shader/execution` area in one process
against yawgpu: a single `cts` process grew ~45 MB/s with no plateau, was
OOM-killed at a 4 GB cap and again at a 10 GB cap, and with `--workers 4`
took the 32 GB host down twice. The brief handed over blamed
"~9 MB per distinct compiled shader retained for the process lifetime" and
recorded one open question — whether the memory was retained by yawgpu or by
the NVIDIA driver — with the stated discriminator being "build Dawn on this
host and A/B it".

That question was answerable without Dawn, and the answer is yawgpu.

## Host

Linux, NVIDIA GeForce RTX 5060 Ti (driver 595.91.07, Vulkan 1.4), Ryzen 7
7800X3D, 29 GB RAM. yawgpu `d6f865b` built `--release --features vulkan`;
webgpu-native-cts `efc9edd`.

## Method

Every measurement below was taken **without modifying either repo**, using
`LD_PRELOAD` shims, until the final refcount step. The shims are worth
keeping; they answer this class of question cheaply:

| shim | what it answers |
|---|---|
| Tint create/destroy counter (interposes `yawgpu_tint_program_create` / `…_destroy`) | are compiled programs destroyed at all |
| C-API event tracer (interposes create/release for modules, pipelines, device) | does the caller release its handles, and in what order |
| allocation tracker (interposes `malloc`/`free`, records ≥16 KB with backtraces, reports survivors grouped by frame) | which code path allocated the memory that is still live |
| RSS + `mallinfo2` sampler (20 ms, reports peak of each) | is the growth inside the process heap or outside it |

## What the measurements showed

1. **Tint itself is clean.** Create + destroy of 48 programs in isolation:
   round 1 `+3.7 MB` (heap high-water), rounds 2 and 3 **`+0 KB`** — the
   memory is fully reused. Skipping the destroy instead grows 0.30 MB per
   program. `yawgpu_tint_program_destroy` returns everything.
2. **The frontend and FFI are clean.** Through the full C API on Noop,
   module + pipeline create/release for 200 distinct shaders: rounds 2-3
   `+0 KB`.
3. **Not the driver.** At the CTS's peak RSS of 629 MB, **528 MB (84%) was
   inside the process malloc heap** (`mallinfo2`), in `libtint_shim.so`.
   Separately, 200 distinct compiles on Vulkan plateau: non-malloc RSS flat
   from round 6 to round 19.
4. **The allocation tracker named the owner's data:** 449 MB in 7,263
   allocations under
   `tint::core::constant::Manager::Get` ← `Resolver::Literal` ←
   `Resolver::Structure`, **live at process exit**. That arena belongs to a
   `tint::Program`, moved in by `yawgpu_tint_program_create`.
5. **The decisive count:** in the CTS run, `program_create=442`,
   `program_destroy=0`. Zero. In the same build driven by a standalone
   program, 144 created / 132 destroyed — the 12 undestroyed being exactly
   the 12 the probe deliberately leaked.
6. **The caller was not leaking the obvious handles:** shader modules
   `create=476 release=476 addRef=0`, compute pipelines `476/476`.

## Ruled out, each by experiment

Shader content (a real CTS shader run standalone destroys its program);
shader size (the repro shader is **405 bytes**, 18 lines — the arena is big
because of `array<T, 1031>`, not because the source is big); GPU submission;
batched-vs-immediate release; scale (476 modules standalone: all destroyed);
Noop vs Vulkan; the device-lost callback; duplicate sources;
`GetCompilationInfo` (instrumented, `compinfo_registered=0`); glibc
allocator behaviour (`MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_` /
`MALLOC_ARENA_MAX` together move the peak only 636 → 558 MB).

The standalone control and the CTS were reduced to the **same four API
calls** against the **same** shader — `CreateShaderModule`,
`CreateComputePipeline`, `PipelineRelease`, `ShaderModuleRelease` — with
opposite outcomes. That is what forced the refcount instrumentation.

## Root cause

Temporary refcount instrumentation inside yawgpu (env-gated, reverted; the
diff is kept out of tree) produced the discriminator, one line apart:

```
CTS        DROP WGPUShaderModuleImpl handles_live=0 core_arc_strong=2   <- no ReflectedModule drop
standalone DROP WGPUShaderModuleImpl handles_live=0 core_arc_strong=1   <- DROP ReflectedModule follows
```

The FFI handle dies in both. The second core reference is the pipeline, held
by a **`WGPUComputePassEncoder` the CTS never releases**
(`expression.cpp:987` begins a pass, ends it at 991, never releases; the
harness tracks modules/pipelines/buffers/encoders but not pass encoders).
`PassEncoderInner::end()` sets `ended = true` and takes `render_commands`,
but never clears `state.compute_pipeline`, so:

```
leaked pass encoder -> compute_pipeline -> shader module -> ReflectedModule -> tint::Program
```

At scale: 476 pass encoders created, 0 dropped, 442 programs retained. One
leaked encoder per case, one retained program per case.

**Shared defect, two independent fixes.** The CTS must release its pass
encoders (174 sites; handed off in that repo). yawgpu must stop amplifying
it: an ended pass encoder has no use for its bindings and should not pin
them. Only the second is in this repo's control, and it is the one that
makes the library robust against any caller with the same bug.

## Corrections to the original brief

- The open question ("yawgpu or the NVIDIA driver?") is **answered: yawgpu**,
  without building Dawn. Facts 3 and 5 above are each sufficient.
- "~9 MB per distinct compiled shader" is right in magnitude but misleading
  in cause: the shaders are tiny and the arena is driven by array extents.
- The brief's proposed acceptance test — a Noop retention test that must fail
  before the fix — **would not have failed**: standalone Noop create/release
  does not retain (fact 2). A regression test must reproduce the *ended pass
  encoder still held* shape, not just module churn. This is what
  `50-commands.md` slice C1 specifies.
- An intermediate claim of this investigation, that the CTS harness was
  "exonerated by measurement", was **wrong**: the audit covered shader
  modules and compute pipelines but not pass encoders. The harness does leak,
  at 174 sites.

## Result (C1-C3 landed 2026-09-21)

`PassEncoderInner::end()` now calls `PassEncoderState::clear_ended_resources()`
— 15 field resets — on every path that sets `ended = true`. Measured against
webgpu-native-cts **`efc9edd`, binary md5 `b160eab4…`, verified identical
before and after each run**, so the pair isolates this change; the CTS side
had not yet been rebuilt with its own fix.

| query | cases | before | after |
|---|---:|---:|---:|
| `…binary,bitwise:bitwise_or:*` | 48 | 635,708 KB / destroy 0 of 442 | **294,876 KB / 442 of 442** |
| `…binary,bitwise:*` | 240 | ~1,768 MB / destroy 0 of 1842 | **311,732 KB / 1842 of 1842** |

Both summary lines byte-identical (`pass=36 skip=12` and `pass=204 skip=36`,
`fail=0 crash=0`). Every Tint program is now destroyed, not merely most.

The shape matters more than either number: **48 cases 288 MB vs 240 cases
304 MB**. Peak memory no longer scales with case count. That is the plateau
the investigation was looking for, and it is what makes the area runnable in
one process.

Gates: `cargo test --workspace` 1023 passed / 0 failed (1020 baseline + 3 new
tests, **no pre-existing test edited**); both clippy gates clean;
`cargo fmt -p yawgpu-core --check` clean.

### Note on later measurements

After the above, the webgpu-native-cts binary was rebuilt by separate work on
that repo's own fix (md5 `6c605d5b…`). Any number taken from that binary
onward reflects **both** fixes and cannot be attributed to this change; a run
of `bitwise:*` against it gave 336,100 KB / 1842 destroys. Cite the table
above, not that, for this change's effect.

### Spec corrections this work forced

- **`scope_usage_index` was missing from D1.** `LenientUsageScopeIndex`
  stores clones of `TextureScopeUse`, each owning a `Texture` handle, so the
  original fourteen fields left a leaked ended *render* pass encoder still
  pinning its attachment textures — which acceptance criterion 1 explicitly
  named. Found by the implementer, who followed the list as written and
  reported the gap instead of improvising; D1 and criterion 1 were amended and
  the field added. Clearing it also restores the index's own documented
  invariant, which clearing only the scope vectors would have falsified.
- **D3 said "three early returns".** Only two are constructible from the
  public API with bindings still in place; a pass that is not the active pass
  has necessarily already been ended. Wording fixed.

### Not fixed here

A pass encoder leaked **without** being ended still retains everything, by
design (D4) — it is legitimately still encoding. The library's obligation is
to stop amplifying a caller's handle leak, and that is what this delivers.

## Phase Review (2026-09-21)

Clean Review per `specs/reference/workflow.md`: a fresh no-context agent given
the cumulative diff `d6f865b..HEAD`, this block's section, `CLAUDE.md`, the
naming conventions and `blocks/90-unit-tests.md`. It re-derived D2
independently — every reader of all 15 cleared fields, across the workspace —
rather than trusting the diff's comments or the spec's assertion, and
confirmed the gate results itself.

**0 CRITICAL, 2 MAJOR, 5 MINOR.** Both MAJORs were defects in **this
section's own field list**, not in the implementation, which had followed the
list as written.

| id | severity | finding | disposition |
|---|---|---|---|
| M1 | MAJOR | `render_commands` was excluded on the stated ground that `end()` always `mem::take`s it. It does not: the take sits inside the `if` for attachments, so a render pass with zero color attachments and no depth-stencil left its recorded commands — and the `Arc`s they own — behind. Reachable from the C ABI, since `begin_render_pass` records the descriptor error and still returns a usable encoder. | **Fixed** — cleared unconditionally; new test `render_pass_end_releases_recorded_commands_without_attachments`, red without the line |
| M2 | MAJOR | `immediate_data` was listed as "unbounded caller data". It is a fixed 64-byte scratch that `record_set_immediates` / `overlay_written_immediates` index **unchecked** under the documented precondition that it is never reset for the pass's lifetime, which this change silently falsified — arming a latent panic to recover 64 bytes. | **Fixed** — removed from the clear; D1 now excludes it with the reason |
| m1 | MINOR | The section still opened "SPEC ONLY — not implemented" three commits after it landed, contradicting this ledger. | **Fixed** (spec) |
| m2 | MINOR | `command_referenced_buffers.clear()` is dead on the pass path — its only writers are the four `push` sites in `render_bundle.rs`. | **No change** — free and correct if a pass-side writer ever appears; recorded so the field count is not read as coverage |
| m3 | MINOR | `clear_ended_resources` was `pub(crate)` although calling it from the other owner of `PassEncoderState` (the bundle encoder, which has no `ended` flag and reads three of the cleared fields back at `finish()`) would silently produce an empty `RenderBundle`. | **Fixed** — narrowed to a private `fn`, so the invariant is enforced rather than documented |
| m4 | MINOR | The doc comment justified retaining the scalars as "still read by later validation". None is: each is consumed above the clear point or never read again once `ended`. | **Fixed** — accurate reason |
| m5 | MINOR | D3's positive half (soft-error paths still clear) had no test; it held only because the clear sits above those branches. | **Fixed** — `pass_end_soft_error_still_releases_bindings`, red when the clear is made conditional on no soft error |

No finding was dropped as a false positive. Both MAJORs were confirmed
independently against the source before being accepted.

### Final gates

`cargo test --workspace` **1025 passed / 0 failed** (1020 pre-block + 5 new
tests, **no pre-existing test edited**); `cargo clippy --workspace
--all-targets -- -D warnings` clean; the same with `--features vulkan
--keep-going` clean; `cargo fmt -p yawgpu-core --check` clean.

**Block 50's "Pass-encoder resource retention on `end()`" is COMPLETE**: no
open CRITICAL or MAJOR, the one deferred MINOR (m2) carries its written
rationale above.

### What the review caught that the implementation could not

Both MAJORs came from the spec asserting something about the code that was
false — that `end()` always drains `render_commands`, and that
`immediate_data` was unbounded. The implementer had been told to follow D1's
list exactly and report gaps rather than improvise, and did; a fresh reader
checking the spec's claims against the source is what turned them up. Worth
repeating: the review's value here was in treating the spec as a claim to
verify, not as ground truth.
