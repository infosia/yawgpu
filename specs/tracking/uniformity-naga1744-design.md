# naga#1744 (= gfx-rs/wgpu#4369) — WGSL-standard uniformity analysis: design & slice plan

> Tracking issue: **gfx-rs/wgpu#4369** "Implement WGSL uniformity analysis"
> (`https://github.com/gfx-rs/wgpu/issues/4369`). The naga source historically
> cited this as `gfx-rs/naga#1744`; after naga merged into the wgpu monorepo that
> issue was renumbered to **#4369** (the old `naga#1744` URL redirects to it). This
> doc uses "naga#1744" and "wgpu#4369" interchangeably.

Implementing the WGSL-spec uniformity analysis (graph-reachability, as Tint does)
in yawgpu's naga fork (`<wgpu-fork>` `feature/tiled`), to replace naga's coarse
**value-insensitive** analysis and close F-120 uniformity (~22467 CTS cases). See
[[uniformity-naga1744]] for the investigation that motivated this; canonical Tint
source `src/tint/lang/wgsl/resolver/uniformity.cc` (in the `third_party/dawn` submodule).

**STATUS 2026-06-20: slices 1-4 DONE — F-120 uniformity RESOLVED on Metal.** The
whole `shader,validation` tree is fail=0 (minus subgroup feature-skips): uniformity
`basics` **46170/0** (was ~20385 fail), `function_variables`/`binary_expressions`/
`pointers` all 0, and structural `shader_io`/`decl`/`functions`/`types`/
`const_assert` stay 0. No shader/execution regression (textureSample 47920/0,
memory_model 83/0). New module `naga/src/valid/uniformity_graph.rs` (graph +
flow-sensitive variable tracking + loop/switch convergence + inter-procedural
summaries + pointer dual-tracking). **Slices 1-5 DONE** (slice 5 cutover removed the
dead `disruptor` path + the `USE_GRAPH_UNIFORMITY`/`DISABLE_UNIFORMITY_REQ_FOR_
FRAGMENT_STAGE` consts; graph is the only path). **MoltenVK re-verified green**
(basics:if 342/0, function_variables 219/0, binary_expressions 1024/0, pointers
58/0 — shared frontend matches Metal). yawgpu workspace tests pass. naga fork was
`783ced3bf` (pin `0320944`), then **13 EDGE CASES** found by the CTS
`uniformity:functions` (12) + `function_pointer_parameters` (1) g.tests (which the
initial pass hadn't measured — they're methods of the `uniformity,uniformity` file,
query `uniformity,uniformity:functions:*`) were fixed in **`4065fd824`**: (a)
derivative + implicit-LOD textureSample RESULTS are non-uniform (seed
may_be_non_uniform, not just require-uniform-CF) so a value derived from a
derivative driving control flow is caught + a fn returning it gets
ReturnValueMayBeNonUniform; (b) pointer-param codependency (store to `*q` under CF
that depends on `*p` taints q's output contents). Re-verified Metal: the WHOLE
non-subgroup uniformity tree fail=0 (functions 40/0, function_pointer_parameters
22/0, basics 46170/0, …) + structural 0. **F-120 uniformity now FULLY resolved**
(was "nearly"). REMAINING: (1) user pushes fork `4065fd824`; (2) bump yawgpu pin to
`4065fd824` + remove `[patch]`; (3) commit yawgpu; (4) slice 6 (subgroup uniformity
— deferred; skips on Metal, do when yawgpu advertises `subgroups`). **MoltenVK run gotcha**: invoke the cts binary DIRECTLY (no
`perl -e exec` wrapper) — SIP strips `DYLD_LIBRARY_PATH` for /usr/bin/perl children,
making the MoltenVK loader fail `BackendUnavailable`.

## Why the current naga model fails

`naga/src/valid/analyzer.rs` threads a single `disruptor: Option<UniformityDisruptor>`
(is the current control flow non-uniform + one cause) and a per-expression
`non_uniform_result: Option<Handle<Expression>>`. It is NOT flow-sensitive per
variable: `Expression::LocalVariable` loads are treated **always non-uniform**
(false positives — e.g. `var x:u32; if x>0 { textureSample }` wrongly rejected),
and loop reconvergence is not modelled (false negatives). The fix is a real
**graph** with flow-sensitive variable value-tracking.

## Target algorithm (Tint, condensed)

Per function, build a directed graph; edge `A → B` ≡ "A's uniformity depends on B"
(so reachability **from** a RequiredToBeUniform sink that reaches the
MayBeNonUniform source = violation). Special nodes per function:
`required_to_be_uniform{_error,_warning,_info}`, `may_be_non_uniform`, `cf_start`,
`value_return`. A `variables` map (var → current value-node), scope-managed, makes
it flow-sensitive. Statements thread a `cf` (current control-flow node) and update
`variables`; if/switch/loop merge via exit-nodes; loops add input-nodes +
backedges for convergence. Functions are summarized (callsite_tag / function_tag /
per-parameter tags) and spliced at call sites (processed in dependency order).
Violation = `Traverse(required_to_be_uniform[sev])` reaches `may_be_non_uniform`.
Full algorithm spec captured from `uniformity.cc` in the design session (node
struct, statement rules, call summary, reachability) — reproduce from there.

## naga IR mapping (the port)

New module `naga/src/valid/uniformity_graph.rs` building the graph over naga IR,
invoked from the existing analyzer per function. Key mappings:

| Tint AST concept | naga IR |
|---|---|
| graph `Node*` + edges | `struct UNode { kind, edges: Vec<NodeId> }` in a `Vec<UNode>`; `NodeId = u32` |
| `variables` (var → value node) | `FastHashMap<Handle<LocalVariable>, NodeId>`, snapshot/restore for scopes |
| expr value node | `FastHashMap<Handle<Expression>, NodeId>` (lazily; arena is SSA-ish) |
| assignment `x = v` | `S::Store { pointer, value }` → resolve pointer root `LocalVariable` (walk `Access`/`AccessIndex`); new node ← value's node; `variables[local] = new`. Partial (Access) → also edge old value. |
| read variable | `Expression::Load { pointer }` → root local → `variables[local]`; `Expression::GlobalVariable` read-write → edge `may_be_non_uniform`; builtins via `FunctionArgument`/entry IO |
| `cf` threading | pass `cf: NodeId` through `process_*`, return updated cf |
| `S::If{condition,accept,reject}` | cf-node ← condition value; recurse both blocks with snapshots; merge locals via exit-nodes (slice 1) |
| `S::Switch{selector,cases}` | like if, N branches + fall_through (slice 2) |
| `S::Loop{body,continuing,break_if}` | input-nodes + backedge + exit-nodes (slice 2) |
| `S::Call{function,arguments,result}` | splice callee summary (slice 3) |
| barrier (`S::Barrier`/ControlBarrier) | `required_to_be_uniform → cf` |
| derivative `Expression::Derivative`, implicit-LOD `ImageSample` | `required_to_be_uniform → cf` (its Emit point) + result seeds nothing extra; severity via `DiagnosticFilter` `DerivativeUniformity` |
| read-write storage load, `local_invocation_id`/`global_invocation_id` etc., `AtomicResult` | edge `→ may_be_non_uniform` |

Notes: naga materializes expressions via `S::Emit(range)`; evaluate an expr's
uniformity node at its Emit point (or lazily on first use), matching the
disruptor-at-Emit model the current code uses (`analyzer.rs:1001-1037`). Reuse the
existing `DiagnosticFilterNode::search(... DerivativeUniformity ...)` +
`severity.report_diag(...)` path (analyzer.rs:1010-1026) for emission, so
`diagnostic(off, derivative_uniformity)` keeps working. Process functions in the
existing dependency order (callees first) so summaries are ready at call sites.

## Slice plan (each: naga tests + local CTS verify via yawgpu [patch])

1. **Graph infra + intra-procedural straight-line/if** — **DONE 2026-06-20**
   (`uniformity_graph.rs`, `USE_GRAPH_UNIFORMITY=true` at analyzer.rs:22, wired at
   analyzer.rs:1410). Verified on Metal: `basics:if` **151→0 fail (fully green)**,
   `binary_expressions` 0, `function_variables` 130→20 (residual 9 over + 11 under
   ALL loop cases = slice 2), structural trees (shader_io/decl/functions/types)
   stay **fail=0** (no regression). Two refinements folded in: read-only storage
   texture/buffer `ImageLoad` is uniform (only read_write seeds MayBeNonUniform);
   re-applied the derivative-arg abstract→f32 concretization (`lower/mod.rs`, basics
   feeds `dpdx(0)`). The new path enforces fragment derivative/implicit-LOD
   uniformity WITHOUT flipping analyzer.rs:21 (slice 5 removes the old path). NO
   genuine slice-1 false positives remain.
2. **Switch + loop convergence** — **DONE 2026-06-20**. `switch` (+fall_through),
   `loop`/`break_if`, `break`/`continue`, `continuing`, input/exit/phi nodes +
   backedges, block-level dead-code reachability, **loop control-flow reconvergence**
   (post-loop cf = incoming cf for normal exit; = divergent latch cf when the loop
   body has a reachable `return`/`kill`, matching Tint `behaviors.Contains(Return)?
   cf1:cf`), and **non-uniform break/break_if exit-value taint** (live-out vars
   exited via a non-uniform condition become non-uniform). Verified: full `basics`
   **~20385→0 fail** (over 0 / under 0 across all 135 statements incl. all
   loop/switch/for/while/continuing/return-end-op variants), `function_variables`
   130→0, `binary_expressions` 0, structural trees stay fail=0. Required 3 codex
   iterations (slice 2 → reconvergence → return-case + break-taint).
3+4. **Inter-procedural + pointers** — **DONE 2026-06-20** (combined; the
   `pointers` test exercises both). Per-function summary (callsite_tag/function_tag/
   param tag_direct+tag_retval/pointer-param effects) computed by reachability,
   stored on FunctionInfo, spliced at `S::Call` in dependency order (callee-first);
   replaced the slice-1 call stopgap. Pointer dual-tracking: load-rule edges to both
   root value and pointer-value; pointer-`Access` propagates a non-uniform index
   into the pointer ADDRESS; `workgroupUniformLoad` requires uniform CF AND a uniform
   pointer, result uniform. KEY FIX: a non-entry **function parameter is NOT a
   non-uniform source** — a requirement reaching a param sets the param's tag (for
   the caller), not a self-violation (this was why `fn needs_uniform(v){if v==0
   {workgroupBarrier();}}` was wrongly rejected). Verified: `pointers` **37/53→0
   (58/0 green)**, function_variables/binary_expressions stay 0. **⚠ STALE-NAGA
   GOTCHA**: cargo did NOT always recompile the [patch] path-dep naga on yawgpu
   rebuild — a slice looked like it "didn't work" (pointers stuck at 53) until
   `find <wgpu-fork>/naga/src -name '*.rs' -exec touch {} +` forced recompile. ALWAYS
   touch naga src before the yawgpu build when verifying a fork edit, and confirm
   `grep -c "Compiling naga"` in the build log. Also: naga-cli `--validate 27` omits
   CONTROL_FLOW_UNIFORMITY (0x4) — use `--validate 31` to exercise uniformity.
5. **Cutover** — replace the old uniformity path with the graph; flip
   `DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE = false`; match diagnostic notes to
   CTS expectations; update naga's own uniformity tests. Target: whole
   `shader,validation,uniformity` tree fail→0 (minus subgroup feature-skips).
6. **(optional) subgroup uniformity** — the subgroup_uniformity second graph;
   low priority (subgroup g.tests skip on Metal).

## Verification per slice

`cargo test -p naga`; then yawgpu local `[patch]` → `cargo build --release -p
yawgpu --features metal` → `cmake --build build-yawgpu --target cts` → run the
targeted `webgpu:shader,validation,uniformity,uniformity:<slice>:*`, plus the
**structural** regression (`shader_io`/`decl`/`functions`/`types` must stay
fail=0) and over-reject watch (the `unexpected validation error for valid shader`
count must trend to 0). Ledger: this file + [[uniformity-naga1744]].

## Risks

- Large/subtle: naga#1744 has been open for years. Slice-gate strictly; never
  relax CTS. - Over-rejection (false positives) is worse than leniency for users —
  watch the over-reject count every slice. - naga IR's Emit/Load/Store + pointer
  model differs from Tint's AST; the `variables`-map-over-LocalVariable mapping is
  the crux — get slice 1 right before building on it.
