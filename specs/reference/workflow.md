# Workflow & roles

Implementation is performed by a **separate coding agent**. Claude acts as
**planner and orchestrator**, not implementer.

## Role split

| Actor | Responsibilities |
|---|---|
| **Claude** (planner/orchestrator) | Author & maintain `specs/` (SPEC, blocks, tracking, this doc); decompose each block into self-contained **task handoffs**; review the coding agent's diff against acceptance criteria; run/inspect `cargo build` & `cargo test`; manage version control (`git add`, `git commit`); update the area's `tracking/<topic>.md` doc; decide go/no-go for the next slice. Also writes `examples/` and the real-GPU `e2e_{metal,vulkan}_*.rs` tests directly (the coding agent's sandbox has no GPU). |
| **Coding agent** (implementer) | Read `HANDOFF.md` + the referenced block spec; write the library code and its inline unit tests; make the targeted Noop gates green; write `REPORT.md`. Does **not** edit `specs/`, commit, or change scope. |

Work is logged in **per-topic** tracking docs, `tracking/<topic>.md`
(e.g. `adapter-limits.md`, `cts-coverage.md`).

## Per-slice loop

A "slice" is one independently reviewable step of a block
(`specs/blocks/<NN>-<area>.md`, slices S1..Sn).

1. **Plan (Claude)** — the block spec states the rules and the slice's
   behaviour contract; write the slice's task handoff to `HANDOFF.md`
   (template below).
2. **Implement (coding agent)** — library code + inline unit tests
   (CLAUDE.md principle 1); targeted gates green; completion report in
   `REPORT.md` (`blocks/91-cts-conformance.md` → "Completion report").
3. **Review (Claude)** — verify against the handoff's acceptance criteria:
   - every new/changed public fn has its inline unit test,
   - validation routes through the device error sink (no panics in
     `yawgpu-core`/`yawgpu-hal`; FFI-boundary `expect` only where allowed),
   - conventions in `CLAUDE.md` honoured,
   - `cargo test --workspace` + clippy clean on Noop; feature-gated HAL
     tests and real-GPU e2e run where the slice touches a backend.
   On failure: return a revision handoff. Do not fix it inline.
4. **Integrate (Claude)** — update the area's `tracking/<topic>.md`;
   `git add` + `git commit` (see "Version control").

## Task handoff template

Illustrative shape of `HANDOFF.md`; adapt the sections to the slice.

```
## Task: Block <NN> S<k> — <short>

Goal: <one line>

Inputs to read:
- specs/blocks/<NN>-<area>.md  (rules R1..Rn, slice S<k>)
- CLAUDE.md, specs/reference/naming-conventions.md
- <oracle source when relevant, e.g. the Dawn file that implements the rule>

Produce:
- <library changes by crate>
- inline #[cfg(test)] unit tests for every new/changed public fn

Out of scope: spec edits, commits, <anything the slice must not touch>.

Acceptance criteria:
- [ ] rules R1..Rn each exercised by at least one test
- [ ] targeted Noop gates + clippy green (commands redirected to a file)
- [ ] no panics in yawgpu-core/yawgpu-hal; CLAUDE.md conventions met

Report: REPORT.md — files changed, commands + exit codes, anything deferred (+why).
```

## Coding-agent command execution (codex output-polling constraint)

The coding agent runs in **codex**, whose `exec_command` is asynchronous:
it launches the process, then drains stdout via `write_stdin` in **30-second
polling windows** with limited output per chunk. A command that streams a
burst of output fills the stdout pipe buffer and **blocks on `write()` until
codex reads it 30 s later** (pipe back-pressure). This throttles throughput
~100×: a `cargo test --workspace` that runs in ~25 s when executed freely
(Claude's Bash, the user's terminal) took **30–73 min** inside codex purely
from this drain — root-caused 2026-06-17 from `~/.codex/sessions` receipts
(the build "Finished" at +2.5 min; the remaining ~70 min was 30 s polls
trickling test output). It is **not** build/link time (a full cold build is
~17 s here) and **not** cargo lock contention (all actors share `./target`
with identical rustc fingerprints; no `Blocking … file lock` ever observed).

**Rule:** in a codex handoff, any long-running or verbose command (test
suites, full builds) must redirect output to a file and report the exit
code, never stream to the console:

```
cargo test --workspace > /tmp/out.log 2>&1; echo "EXIT=$?"; tail -n 40 /tmp/out.log
```

This lets the process run at full speed while codex reads only the small
tail. A test-name **filter does not avoid the cost**: `cargo test -p yawgpu
<filter>` still spawns every integration-test binary in the package (each
prints `running 0 tests`), so codex polls through 50+ output flushes — a
single-test run was observed taking 45 min. Use `--test <binary> <filter>`
to run one binary, or (preferred) just redirect to a file. The agent's
targeted gates and the workspace-test ban are specified
in `blocks/91-cts-conformance.md` → "Completion report → Verification".
**Claude** runs the full `cargo test --workspace` on review directly via its
own Bash (no polling harness — ~25 s), so it remains the backstop.

## Phase Review (mandatory — "Clean Review Then Fix")

Every phase ends with a **mandatory Phase Review** before it can be marked
COMPLETE. Per-slice review (Claude, full session context) catches
slice-local issues; the Phase Review catches **accumulated / cross-slice**
issues that a context-primed reviewer rationalizes away.

1. **Clean Review (fresh agent, no session context).** Claude spawns a
   subagent that has **no conversation history**. It is given only:
   the phase's cumulative `git diff` (the block's commit range, e.g.
   `aaef70c..3f809b1`), the
   phase's `blocks/<area>.md`, `CLAUDE.md`,
   `specs/reference/naming-conventions.md`, and the phase exit criteria.
   It does **not** see this conversation or prior rationale. It produces
   **severity-tagged findings**, each with `file:line` + rationale:
   - **CRITICAL** — memory unsafety/UB, soundness, FFI ABI mismatch,
     a panic reachable from the C ABI on valid input, a spec rule
     silently wrong, data loss.
   - **MAJOR** — a ported rule not actually enforced, missing/empty
     test coverage for a rule, convention breach with real impact,
     resource/refcount leak.
   - **MINOR** — naming, dead code, redundant work, doc/comment gaps,
     non-idiomatic but correct code.
2. **Triage (Claude).** Drop false positives with a one-line written
   reason; keep the rest. Anything dropped is recorded in the area's
   `tracking/<topic>.md`.
3. **Fix in severity order.** CRITICAL first, then MAJOR, then MINOR.
   Production-code fixes go to the **coding agent** via a fix handoff
   (Claude does not write production code); spec fixes are Claude's.
   Re-run the full gate (`cargo test --workspace` +
   `cargo clippy --workspace --all-targets -- -D warnings`) after each
   severity tier.
4. **Gate.** Phase cannot be marked COMPLETE while any **CRITICAL** or
   **MAJOR** finding is open. **MINOR** may be deferred only with an
   explicit written rationale logged in the area's `tracking/<topic>.md`
   (and a rule/Defer marker if it maps to one).
5. **Log.** The area's `tracking/<topic>.md` records: the finding list
   with severities + file:line, triage decisions, the fix commits, and
   the final gate result. Commit per "Version control", naming the review
   in the subject, e.g. `fix(hal/vulkan): Block 107 Phase Review — <fixes>`.

The Clean Review reviewer is a throwaway subagent per phase (no memory of
previous phases beyond what the diff shows); this is deliberate.

## Version control

Claude commits per slice on the current branch (`main`; no automatic
branching). The coding agent never commits. Commit message convention is
Conventional-Commits style, `type(scope): <short> — <detail>`, where `type`
is `feat` / `fix` / `test` / `docs` / `refactor` / `cts` / `build` and
`scope` names the crate or layer (`core`, `hal/vulkan`, `hal/metal`, `ffi`,
`e2e`, `specs`, …); the block and slice go in the subject, e.g.
`feat(hal/vulkan): Block 105 S2 — render passes transition attachments and
bound views per subresource`.
