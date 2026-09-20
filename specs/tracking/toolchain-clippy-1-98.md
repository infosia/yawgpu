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
