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

## Version control

The repo is not yet a git repository. Claude runs `git init` during Phase 0
integration and commits per slice. The coding agent never commits. Commit
message convention: `phase-N: <area> — <short>` (e.g.
`phase-2: buffer — port BufferValidationTests`).
