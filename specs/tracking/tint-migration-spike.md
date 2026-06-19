# Tint-vs-naga migration — feasibility spike (2026-06-20)

Motivation: the F-120 `uniformity` investigation ([[uniformity-naga1744]]) showed
naga's WGSL analysis is structurally weaker than Tint's (value-insensitive
uniformity → both false negatives AND false positives; the real fix is the
naga#1744 rewrite). That raised the question: **how hard is it to replace naga
with Tint** (Dawn's WGSL compiler, the CTS oracle's compiler) as yawgpu's shader
frontend? This spike de-risks the *feasibility* unknowns. It does NOT commit to
the migration — that's a separate A-vs-B decision (below).

Spike lives OUTSIDE the yawgpu repo at `../tint-spike/` (throwaway, not committed).
Dawn/Tint source: `../../C/dawn`. Built on this M2.

## What the spike proved (all measured)

| Concern | Result |
|---|---|
| **Standalone CMake build of Tint** (WGSL reader + SPV/MSL writer + inspector only; Dawn backends / HLSL / GLSL / tintd / tests / protobuf all OFF) | ✅ **73 s** clean (`-j`), Makefiles (no ninja), third_party pre-vendored → **no network**. Tint static libs ~14 MB; abseil+SPIRV-Tools actually-linked ~256 KB. |
| **C++ shim** (`tint_shim.cpp`, ~100 LOC): C ABI `wgsl_to_msl` / `wgsl_to_spirv` mirroring `cmd/tint/main.cc` (`Initialize` → `Parse` → `ProgramToLoweredIR` → `{msl,spirv}::writer::Generate`, bindings via `tint::GenerateBindings`) | ✅ compiles. **Note: Tint requires C++20** (concepts, `std::span`). Link needs `tint_api` **AND** `tint_api_helpers` (GenerateBindings lives there). |
| **Self-contained dylib** (SHARED lib linking tint_api PRIVATE → bundles all needed Tint code behind one library; sidesteps 56-static-lib link-order pain) | ✅ **5.6 MB** `libtint_shim.dylib`, exports the 4 C symbols. |
| **Rust → Tint FFI** (`tint-spike` crate; build.rs emits link-search + `-l dylib=tint_shim` + rpath + `-lc++`; objc2-metal 0.3) | ✅ Rust calls the shim, gets MSL string + SPIR-V `Vec<u32>`. |
| **Tint codegen correctness** | ✅ `fs_main`→MSL 1556 B (incl. `dpdx(0)` abstract-int, which **naga over-rejects**); `cs_main`→SPIR-V 204 words, magic `0x07230203`; `cs_main`→MSL 1192 B. |
| **Metal runs Tint's MSL** (objc2-metal: `newLibraryWithSource` → `newFunctionWithName` → `newComputePipelineStateWithFunction`) | ✅ on Apple M2: compiled to a compute pipeline, `maxTotalThreadsPerThreadgroup = 8` (= `@workgroup_size(8,1,1)`). |
| **iOS cross-build** (`-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0`) | ✅ tint+shim cross-compiled with **zero extra env**; `libtint_shim.dylib` = Mach-O arm64, `LC_BUILD_VERSION platform 2` (iOS), minos 14.0. Matches [[ios-cross-build-setup]] (iOS is the easy mobile target) — now confirmed for the C++ Tint dep too. |
| **Android cross-build** (NDK r30 `android.toolchain.cmake`, `-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26`) | ✅ **zero source changes**; `libtint_shim.so` = ELF AArch64, NEEDED only `libm/libdl/libc` (system libs; libc++ statically bundled), **5.9 MB stripped** (≈ the iOS dylib). Notably CLEANER than naga's Android cross-build, which needed the load-bearing `BINDGEN_EXTRA_CLANG_ARGS` sysroot hack ([[android-cross-build-setup]]) — a native CMake/NDK build has no bindgen-into-C-headers step to break. |

## De-risked vs still-unknown

**De-risked (feasibility confirmed):** Tint is standalone-buildable (fast, light,
offline), drivable from Rust via a tiny C shim + one dylib, emits MSL the Metal
driver accepts at runtime, and cross-builds cleanly to **both** iOS arm64 (zero
extra env) **and** Android arm64-v8a (zero source changes, NDK toolchain only,
~6 MB stripped .so, system-lib deps only). The original "55 MB / hard to embed /
GN-only / mobile" fears were overstated — built artifact is ~14 MB, CMake works,
deps vendored, both mobile targets cross-compile without hacks.

**Still unknown / not in this spike (the real integration body of work):**
1. **Reflection wiring** — the spike used `GenerateBindings` auto-flatten, NOT
   yawgpu's real group/binding model nor the `Inspector` → BGL-auto-generation
   path that `shader_naga.rs`'s `ReflectedModule` feeds. Replicating yawgpu's
   binding map + reflection is real work.
2. **Full CTS re-verification** — Tint emits different MSL/SPIR-V than naga; every
   currently-green CTS case (Metal whole suite + native Vulkan) must be
   re-confirmed against the HAL's acceptance of Tint output (argument buffers,
   entry-point naming, robustness, workgroup allocations…).
3. **build.rs productionization** — drive CMake from build.rs (e.g. `cmake` crate)
   against `$OUT_DIR` with caching; vendor/submodule the dawn checkout.
4. The accumulated naga-fork features (f16, clamp_frag_depth, sample_mask, external
   textures, unrestricted_pointer_parameters, all the F-120 validator rules) —
   most become unnecessary (Tint does them), but the yawgpu-specific ones
   (shader-passthrough MSL/SPIRV variants, external-texture honest-rejection)
   need re-homing.

## Integration surface (yawgpu side)

Replacement target is `yawgpu-core/src/shader_naga.rs` (**3393 LOC**, owns
WGSL→MSL/SPIRV + reflection, exposes `ReflectedModule`). But naga types leak past
that abstraction into the HAL: `metal/{device,pipeline,encode}.rs` build
`naga::back::msl::{EntryPointResources,BindTarget,MslBindingMap,MslResourceBinding}`
directly, and `naga::ReflectedResourceBinding` (12 sites) is consumed in pipeline
+ BGL-auto-gen. A swap re-homes all of these onto Tint's `Bindings` + `Inspector`.

## Effort estimate

Feasibility = confirmed (incl. **both** mobile targets). Full production parity
(shim+FFI + reflection rewiring + CTS re-verify of the whole green surface +
build.rs productionization) is a **~1.5–3 month** effort, now dominated by **CTS
re-verification + reflection wiring**, NOT by the (proven) build/FFI/mobile
mechanics.

## Decision (open — A vs B)

- **A — stay on naga**, port specific Tint algorithms (uniformity = naga#1744)
  incrementally. Pure-Rust, simple build, easy mobile; but chases Tint's
  correctness forever and some gaps (uniformity) are large per-fix.
- **B — migrate to Tint.** Conformance-by-construction (same compiler as the Dawn
  oracle → a whole class of naga-divergence findings, incl. uniformity & F-085,
  evaporate) + drops the naga-fork maintenance tax; cost is the integration body
  above + a permanent C++ dependency (acceptable — yawgpu already links
  MoltenVK/objc2) tracking Dawn's unversioned release cycle.

Spike verdict: **B is technically de-risked and viable.** The call is a product
priority decision (maximal conformance + C++ dep vs pure-Rust simplicity), not a
feasibility one. Related: [[uniformity-naga1744]], [[cts-coverage]].
