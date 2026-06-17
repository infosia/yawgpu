# Workflow & roles

Implementation is performed by a **separate coding agent**. Claude acts as
**planner and orchestrator**, not implementer.

## Role split

| Actor | Responsibilities |
|---|---|
| **Claude** (planner/orchestrator) | Author & maintain `specs/` (SPEC, blocks, tracking, this doc); decompose each phase into a self-contained **task handoff**; review the coding agent's diff against acceptance criteria; run/inspect `cargo build` & `cargo test`; manage version control (`git init`, `git add`, `git commit`); update `tracking/phase-N.md` and the `dawn-test-mapping.md` status column; decide go/no-go for the next slice. |
| **Coding agent** (implementer) | Read the assigned task handoff + referenced block spec + Dawn source; write the code and the ported tests; make `cargo test` green on Noop; report what it changed. Does **not** edit `specs/`, commit, or change scope. |

Claude does **not** write production code itself; it writes specs, reviews,
and integrates. The coding agent does **not** plan or commit.

## Per-slice loop

A "slice" is one Dawn test file (or, for Phase 0, one scaffold deliverable).

1. **Plan (Claude)** — ensure the relevant `blocks/<area>.md` has the rules
   extracted from the Dawn source; emit a task handoff (template below).
2. **Implement (coding agent)** — produce code + ported test; make Noop
   `cargo test` green; report.
3. **Review (Claude)** — verify against the handoff's acceptance criteria:
   - test file faithfully ports the Dawn cases (no silently dropped cases),
   - validation routes through the device error sink (no panics in
     `yawgpu-core`/`yawgpu-hal`; FFI-boundary `expect` only where allowed),
   - conventions in `CLAUDE.md` honoured,
   - `cargo build` + `cargo test` clean on Noop.
   On failure: return a revision handoff. Do not fix it inline.
4. **Integrate (Claude)** — update `tracking/phase-N.md` and the status
   column in `dawn-test-mapping.md`; `git add` + `git commit` with a message
   referencing the phase and Dawn file.

## Task handoff template

Claude produces one of these per slice (kept in `tracking/phase-N.md` or
inline when dispatching the coding agent):

```
## Task: <area> — port <DawnFile>ValidationTests

Goal: <one line>

Inputs to read:
- specs/blocks/<block>.md  (rules R1..Rn)
- dawn/.../<DawnFile>ValidationTests.cpp
- specs/reference/naming-conventions.md, CLAUDE.md

Produce:
- yawgpu/tests/<area>_validation.rs  (port every TEST_F case; map
  ASSERT_DEVICE_ERROR -> assert_device_error!)
- minimal impl in yawgpu-core (+ yawgpu FFI fns) to make them pass on Noop

Out of scope: real backends, unrelated APIs, spec edits, commits.

Acceptance criteria:
- [ ] every Dawn TEST_F case in the file has a corresponding #[test]
- [ ] cargo test green on Noop, no GPU
- [ ] no panics in yawgpu-core/yawgpu-hal; CLAUDE.md conventions met
- [ ] rules R1..Rn each exercised by at least one test

Report back: files changed, any Dawn cases intentionally deferred (+why).
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
   the phase's cumulative `git diff` (the `phase-N` commit range), the
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
   reason; keep the rest. Anything dropped is recorded in
   `tracking/phase-N.md`.
3. **Fix in severity order.** CRITICAL first, then MAJOR, then MINOR.
   Production-code fixes go to the **coding agent** via a fix handoff
   (Claude does not write production code); spec fixes are Claude's.
   Re-run the full gate (`cargo test --workspace` +
   `cargo clippy --workspace --all-targets -- -D warnings`) after each
   severity tier.
4. **Gate.** Phase cannot be marked COMPLETE while any **CRITICAL** or
   **MAJOR** finding is open. **MINOR** may be deferred only with an
   explicit written rationale logged in `tracking/phase-N.md` (and a
   rule/Defer marker if it maps to one).
5. **Log.** `tracking/phase-N.md` records: the finding list with
   severities + file:line, triage decisions, the fix commits, and the
   final gate result. Commit: `phase-N: phase review — <n> findings
   (<c> CRITICAL / <m> MAJOR / <k> MINOR) fixed`.

The Clean Review reviewer is a throwaway subagent per phase (no memory of
previous phases beyond what the diff shows); this is deliberate.

## Version control

The repo is not yet a git repository. Claude runs `git init` during Phase 0
integration and commits per slice. The coding agent never commits. Commit
message convention: `phase-N: <area> — <short>` (e.g.
`phase-2: buffer — port BufferValidationTests`).
