# Uniformity analysis (F-120 residual) — investigated → deferred → RESOLVED (naga#1744 = gfx-rs/wgpu#4369)

> **SUPERSEDED 2026-06-20**: this ledger records the *initial* investigation that
> deferred uniformity to naga#1744. The user then chose to implement it, and it was
> DONE — see `uniformity-naga1744-design.md` (naga `783ced3bf`, yawgpu `0320944`,
> green Metal+MoltenVK). Issue ref: **gfx-rs/wgpu#4369** "Implement WGSL uniformity
> analysis" (the naga source cited the old number `gfx-rs/naga#1744`, renumbered to
> #4369 after the wgpu monorepo merge). Note: `wgpu#6458` below is a *separate*
> issue (warning-level diagnostic items), not the uniformity-analysis issue.

Investigation 2026-06-20 of the only remaining open chunk of **F-120**
(webgpu-native-cts finding): the `shader,validation,uniformity,uniformity` CTS
tree (~22467 cases where naga diverges from Tint/Dawn). The structural F-120
slices (shader_io, decl, functions, types, const_assert ~330 cases) are all
RESOLVED and pin-bumped (`c80b32f` / naga `4d5500a`); uniformity is the residual.

**Outcome: investigated end-to-end, a candidate fork fix was built, measured, and
REVERTED.** Uniformity is classified as an **upstream-naga limitation
(naga#1744 / wgpu#6458)** — same class as F-085 (`sample_mask`): yawgpu's failing
set == wgpu-native's (yawgpu-only = 0), Dawn-green, so it is NOT a yawgpu defect.
No fork change landed; branch `feature/tiled` is clean, yawgpu pin unchanged.

## CTS scope (current pin, Metal)

| g.test | cases | fail (baseline) | note |
|---|---:|---:|---|
| `basics` | 135 | ~151/case (≈20385) | constant 191 pass / 151 fail per statement (if/switch/loop/for) → inner (source × op) cross-product, not control-flow shape |
| `basics,subgroups` | 135 | — | skips on Metal (no `subgroups` feature) |
| `uniform_subgroup_ops` | 52 | — | skips on Metal |
| `binary_expressions` | — | 768 | |
| `function_variables` | — | 130 | |
| `pointers` | — | 37 | |

## Root-cause taxonomy (all measured via local `[patch]` → release-relink → CTS)

1. **Derivative builtins don't concretize an abstract-int literal arg**
   (OVER-reject, ~9 per basics case). `dpdx(0)` rejected ("Derivatives can only
   be taken from scalar and vector floats"); `dpdx(0.0)` and `sqrt(0)` OK —
   verified with `naga-cli ... --validate`. Derivative-family-specific (general
   builtins already concretize). Candidate fix: `front/wgsl/lower/mod.rs`
   derivative lowering via `try_automatic_conversion_for_leaf_scalar(.., F32, ..)`.
   Correct in isolation, but on its own it makes the uniformity CTS count WORSE by
   unmasking #2 (the derivative type-error was accidentally satisfying the
   "expected an error" oracle in non-uniform cases). See [[cts-failure-patterns]]
   (crash/error-masks-behavior).

2. **`DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE = true`** (`naga/src/valid/
   analyzer.rs:21`) blanket-zeros the `DERIVATIVE` (0x2) / `IMPLICIT_LEVEL` (0x4)
   requirement bits, so fragment derivative/implicit-LOD-`textureSample`
   uniformity is never enforced (UNDER-reject — the bulk). The diagnostic-filter
   emission is ALREADY complete and correct: `analyzer.rs:1010-1026` routes the
   requirement through `StandardFilterableTriggeringRule::DerivativeUniformity`
   (default `Severity::Error`, honours `diagnostic(off, derivative_uniformity)`).
   So enabling enforcement is a 1-line flag flip — the directive machinery the CTS
   relies on (`expectCompileResult(true, 'diagnostic(off,...)' + code)`) is in place.

3. **Value-INSENSITIVE analysis is the real blocker (naga#1744).** Flipping the
   flag to `false` (+ #1) was measured:
   - `binary_expressions` 768 → **0** (clean win).
   - `basics:if/switch` 151 → 88 fail; `function_variables` 130 → 89 fail.
   - BUT it introduces **FALSE POSITIVES** (naga rejects shaders Dawn accepts):
     `basics:if` 48 over-reject, `function_variables` **89/89** over-reject.
   - `basics:loop-*` stays ~160 under-reject (loop convergence not modelled);
     `pointers` stays 37 under-reject (pointer propagation gap).

   Root: naga marks **every `LocalVariable` load non-uniform** (`analyzer.rs:766`),
   likewise pointers / function args, and does not model loop reconvergence — far
   coarser than the WGSL-standard algorithm Tint implements. Confirmed false
   positive (Dawn accepts, naga rejects after the flip):

   ```wgsl
   @fragment
   fn main() {
     var x : u32;                 // uninitialised local → value is uniform (0)
     if x > 0 {                   // naga: x is "non-uniform" → non-uniform CF
       let tmp = textureSample(t, s, vec2f(0,0));  // → wrongly rejected
     }
   }
   ```

   Rejecting VALID shaders is worse for real users than the existing leniency —
   which is exactly why upstream naga keeps the flag `true`.

## Decision

REVERTED #1 (derivative concretize) + #2/#3 (flag flip + naga tests); fork branch
clean, yawgpu untouched, NO pin bump, nothing committed. A proper fix requires
porting **naga#1744** (WGSL-standard value-sensitive uniformity: track the
uniformity of values stored in locals / loaded through pointers / passed as
function args, and model loop reconvergence). Only worth doing if we commit to
that rewrite; the 1-line flag flip alone is a net regression (false positives).

**Re-verify trigger:** when a naga uniformity-analysis fix lands upstream (or we
port naga#1744 in-fork), re-run
`webgpu:shader,validation,uniformity,uniformity:{basics,binary_expressions,function_variables,pointers}:*`
on Metal and re-measure over/under-reject split before bumping the pin.

## Lesson

Measure before landing. The flag flip LOOKED like a 1-line win (and the
diagnostic-filter infra was already built), but local-`[patch]` CTS measurement
exposed naga's coarse analysis (false positives) and the #1→#2 unmasking.
Related: [[cts-failure-patterns]], [[cts-coverage]].
