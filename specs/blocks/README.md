# Block specs

One file per API area, authored by **Claude** (the coding agent does not
edit `specs/`). Filled at the **start of its phase** by reading the
corresponding Dawn validation test(s) and extracting the rules under test.
Each rule gets an ID (R1, R2, …) so task handoffs can reference
"rules R1..Rn each exercised by a test" as acceptance criteria.
Format per block:

- **Surface**: `webgpu.h` functions/structs covered.
- **Rules**: enumerated validation rules (each traceable to a Dawn test case).
- **Async**: futures/callbacks involved, Noop completion behaviour.
- **Open questions**.

Planned blocks (created lazily, tracked in `../SPEC.md`):

| File | Phase |
|---|---|
| `00-foundation.md` | 0–1 (instance/adapter/device/future/error sink) |
| `10-buffer-queue.md` | 2 |
| `20-texture-sampler.md` | 3 |
| `30-shader-binding.md` | 4 |
| `40-pipeline.md` | 5 |
| `50-commands.md` | 6 |
| `60-backends.md` | 7 |
| `70-surface-query-errorscope.md` | 8 |
