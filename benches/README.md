# benches — cross-implementation CPU-overhead benchmark

A single C++ translation unit written against canonical `webgpu.h`, compiled
twice: once linked against `libyawgpu`, once against Dawn's `libwebgpu_dawn`.
Because both link the same C ABI, every measurement crosses identical entry
points and the difference is attributable to the implementation behind them.

This measures **CPU cost** — validation, object construction, command
recording, submission bookkeeping. GPU execution time is excluded except in the
cases whose names end in `_wait`, which are latency figures and labelled as such
in `specs/tracking/perf-dawn-baseline.md`.

Dawn is the same oracle used for conformance (see `CLAUDE.md` → "CTS
conformance"), so a performance delta here is measured against the
implementation yawgpu is already held to for behaviour.

## Method

Each case runs `iters/10` untimed warmup iterations, then `reps` batches of
`iters` timed iterations. The reported figure is the **minimum** per-op time
across batches — the least noise-contaminated estimate of true cost — with the
median alongside, so a large min/median gap flags an unstable case.

The device is created with an uncaptured-error callback that aborts the run. An
operation that silently fails validation is cheap, and would otherwise show up
as a benchmark win.

## Build

yawgpu (requires the backend feature to be compiled in):

```sh
cargo build -p yawgpu --release --features metal
cmake -S benches -B benches/build-yawgpu -DBENCH_BACKEND=yawgpu
cmake --build benches/build-yawgpu -j8
```

Dawn (a Dawn checkout with a built `libwebgpu_dawn` is a local prerequisite;
no path to it is recorded in this repository):

```sh
cmake -S benches -B benches/build-dawn -DBENCH_BACKEND=dawn \
  -DBENCH_DAWN_DIR=/path/to/dawn \
  -DBENCH_DAWN_BUILD_DIR=/path/to/dawn/out/Release
cmake --build benches/build-dawn -j8
```

## Run

```sh
./benches/build-yawgpu/bench                     # aligned table
./benches/build-yawgpu/bench --tsv               # machine-readable
./benches/build-yawgpu/bench --filter submit/    # subset
./benches/build-yawgpu/bench --scale 0.02        # quick smoke
./benches/build-yawgpu/bench --reps 15           # more batches
```

`YAWGPU_BENCH_BACKEND=metal|vulkan|gles` selects the backend for either binary,
mirroring the CTS harness's `CTS_YAWGPU_BACKEND` / `CTS_DAWN_BACKEND`.

To compare two TSV runs:

```sh
awk -F'\t' 'NR==FNR{if(FNR>1)y[$2]=$4;next} FNR>1 && ($2 in y){
  printf "%-36s %12.1f %12.1f %8.2fx\n",$2,y[$2],$4,y[$2]/$4}' y.tsv d.tsv
```

## Reading the results

Two classes of case are **not** like-for-like and must not be quoted as a
speedup:

- `submit/empty` — yawgpu short-circuits a submit with no recorded work
  (`MetalQueue::submit_copies` returns early on an empty slice), so it is not
  doing the work Dawn does.
- `queue/write_buffer_4kb` — an implementation may defer the upload into the
  next submit, moving cost outside the timed region.

`frame/10writes_dispatch_submit_wait` exists for exactly that reason: it drives
uploads, a dispatch, a submit and a queue drain, so no work can be deferred past
the measurement. It is the case to quote for an end-to-end comparison.

Recorded results and root causes: `specs/tracking/perf-dawn-baseline.md`.
