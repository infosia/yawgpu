# Toolchain — clippy 1.98 gate breakage (`chunks_exact_to_as_chunks`)

Filed 2026-09-20 during the Linux clean-install bring-up. **Independent of
Block 98 and of Block 80 / P9.5**, found while running their gates. Its own
commit.

## Finding

The repo's mandatory gate

```sh
cargo clippy --workspace --all-targets -- -D warnings
```

**fails on an unmodified checkout** with the toolchain installed on this host
(rustc / clippy 1.98.1, `48a229cea`, 2026-09-01):

```
error: using `chunks_exact` with a constant chunk size
    --> yawgpu-tint/src/lib.rs:1708:14
     |
1708 |             .chunks_exact(2)
     |              ^^^^^^^^^^^^^^^ help: consider using `as_chunks` instead: `as_chunks::<2>().0.iter()`
     |
     = note: `-D clippy::chunks-exact-to-as-chunks` implied by `-D warnings`

error: could not compile `yawgpu-tint` (lib) due to 1 previous error
error: could not compile `yawgpu-tint` (lib test) due to 1 previous error
```

Verified pre-existing: `git diff -- yawgpu-tint/src/lib.rs` is empty and the
`chunks_exact(2)` call is present at `HEAD` (`f240940`). Nothing in this
session's work introduced it.

Cause: `clippy::chunks_exact_to_as_chunks` newly fires in this clippy release
(it depends on `slice::as_chunks`, stabilized upstream). The project was
developed against an older toolchain where the gate was green, so this is a
toolchain-drift breakage, not a code regression. Any contributor on a
current toolchain hits it on a clean clone.

## Site

`yawgpu-tint/src/lib.rs`, `msl_buffer_size_bindings` — pairs of `u32` read out
of the shim's flat word buffer:

```rust
Ok(guard
    .as_slice(word_len)
    .chunks_exact(2)
    .map(|pair| MslBufferSizeBinding {
        group: pair[0],
        binding: pair[1],
    })
    .collect())
```

## Rules

- **R1 — Gate green on the current toolchain.**
  `cargo clippy --workspace --all-targets -- -D warnings` passes with no
  `-A` overrides on rustc 1.98.1. An `#[allow(...)]` is **not** the accepted
  fix here: the lint is correct and the suggested form is strictly better
  typed (`&[u32; 2]` instead of an unbounded `&[u32]`, so the `pair[0]` /
  `pair[1]` indexing stops being a runtime bounds check).

- **R2 — Behaviour identical.** `as_chunks::<2>().0` yields only the complete
  pairs, exactly as `chunks_exact(2)` did; the trailing remainder (`.1`) is
  discarded, matching the current semantics. `word_len` is already
  `len * 2` (checked), so a remainder cannot occur in practice — but the
  handling must stay equivalent regardless, not "improved".

- **R3 — No other lint suppressed.** The fix touches only this call site. If
  the same toolchain surfaces further pre-existing lints elsewhere, each is
  recorded here and fixed on its merits, never blanket-allowed.

## Verification

- `cargo clippy --workspace --all-targets -- -D warnings` — clean, no overrides.
- `cargo test --workspace` — Noop gate green; baseline
  **1013 passed / 0 failed / 10 ignored** across 90 binaries. The MSL
  sizes-buffer reflection path is covered by the `yawgpu-tint` unit tests
  (`generate_msl_*` / runtime-array cases), so a behaviour change would show up
  there.

## Note for CI

If CI pins an older toolchain, it will not see this failure, and the reverse
drift (a future lint) will keep landing on contributors first. Pinning the
gate's toolchain, or running clippy on both the pinned and the current stable,
is worth considering — recorded as an open question, not scope here.

## Second round — feature-gated targets (2026-09-20)

R3 anticipated this: more pre-existing lints, found only because a slice
happened to run the gate with `--features vulkan`. All are red on HEAD
independently of any local change.

**This list was first recorded as two findings. That was wrong** — see
"Enumerating the true set" below. The complete set is the two here plus the
seven in the third round.

1. `yawgpu-hal/src/vulkan/device.rs:249` — `use super::super::*;` inside the
   `#[cfg(test)] mod tests` is unused (`unused-imports`). Delete the line.
2. `yawgpu/tests/e2e_vulkan_texture_compression.rs:638` —
   `pixels.chunks_exact(4).enumerate()` trips the same
   `chunks_exact_to_as_chunks` lint already fixed in `yawgpu-tint`. Same
   treatment per R1 (no `#[allow]`): `as_chunks::<4>().0.iter().enumerate()`,
   adjusting the binding for the `&[u8; 4]` element type.

### Process finding — the gate command does not cover what it claims

`cargo clippy --workspace --all-targets -- -D warnings` compiles only the
default feature set, so every `#[cfg(feature = "vulkan")]` module, every
`--features vulkan` test target, and the same for `metal` / `gles` /
`shader-passthrough` / `tiled`, are **never linted by it**. A "clippy gate
clean" claim made with that command alone is therefore narrower than it sounds
— which is exactly how (2) above survived the first round of this fix.

- **R4 — The gate is per-feature.** A clippy-clean claim states which feature
  sets it covers. At minimum the default set and `vulkan` are both run on a host
  that can build them; `metal` on macOS. Backends the host cannot build are
  named as not covered rather than silently omitted.

Not in scope here: changing CI to run the per-feature gates. Recorded as an open
question — CI currently runs the default gate only, so feature-gated lint rot
reaches contributors first, the same drift this document already describes.

### Enumerating the true set

`cargo clippy` stops scheduling new targets once one fails ("build failed,
waiting for other jobs to finish..."), so a plain run reports a **partial,
nondeterministic** subset of the failing targets — three identical runs produced
three different subsets. The two findings first recorded above were one such
subset, not the whole.

- **R5 — Enumerate with `--keep-going`.** Any claim about *which* lints remain
  is made from
  `cargo clippy --workspace --all-targets --features <f> --keep-going --message-format=short -- -D warnings`.
  A plain run answers only "clean / not clean", never "these are the findings".

## Third round — the remaining `--features vulkan` sites (2026-09-20)

Enumerated with `--keep-going`; 7 sites, all `chunks_exact_to_as_chunks`, all in
`--features vulkan` test targets that the default gate never compiles:

| file:line | chunk size |
|---|---|
| `yawgpu/tests/e2e_vulkan_compute.rs:533` | `std::mem::size_of::<u32>()` |
| `yawgpu/tests/e2e_vulkan_f16.rs:150` | `2` |
| `yawgpu/tests/e2e_vulkan_immediates.rs:607` | `std::mem::size_of::<u32>()` |
| `yawgpu/tests/e2e_vulkan_render.rs:907` | `BYTES_PER_PIXEL` |
| `yawgpu/tests/e2e_vulkan_render.rs:917` | `BYTES_PER_PIXEL` |
| `yawgpu/tests/e2e_vulkan_subgroups.rs:126` | `4` |
| `yawgpu/tests/e2e_vulkan_texture_formats_tier2.rs:67` | `4` |

R1 applies (no `#[allow]`), R2 applies (`as_chunks::<N>().0` discards the
remainder exactly as `chunks_exact(N)` did), R3 applies (only these sites).

- **R6 — `size_of` chunk sizes need a named const.** Clippy prints
  `as_chunks::<std::mem::size_of::<u32>()>()`, which is not valid const-generic
  argument syntax. Those two sites introduce a named `const` (or use the literal
  width) instead of pasting the suggestion.

Each rewrite changes the iterator element from `&[T]` to `&[T; N]`, so
comparisons against an array may need a deref (`*chunk == expected`); the
assertion's meaning and message must not change.

### Knock-on lint (third round)

Rewriting `e2e_vulkan_render.rs:907` surfaced a lint the original code did not
trip: with the element type now `&[u8; 4]`, `.iter().any(|p| *p == rgba)` trips
`clippy::manual_contains`. Fixed on the merits per R1 —
`pixels.as_chunks::<BYTES_PER_PIXEL>().0.contains(&rgba)`, same predicate.

Generalisation worth remembering: a lint fix can change an expression's type and
so expose a *different* lint at the same site. The R5 `--keep-going` enumeration
is therefore re-run **after** applying fixes, not only before — the set is not
static.
