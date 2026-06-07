# Format-table completeness audit (Audit C)

Proactive sweep (2 parallel agents: core `FormatCaps` table vs the WebGPU spec; HAL backend mappers vs the
core-advertised set) for the F-006/F-016/F-018/F-024 family — a format MISSING / mis-classified / advertised
but unmapped. Part of [[cts-failure-patterns]] (pattern 4). Coverage tally: **0 missing formats** — every
WebGPU plain-color, depth/stencil, and BC/ETC2/EAC/ASTC format is present in the core table with correct
block size/aspect; all 48 `HalTextureFormat` variants map to a real Vulkan + Metal native format (GLES maps
43 color formats, rejects 5 depth/stencil with a catalogued `HalError`). The previously-templated bug classes
(wrong multisample, missing read-write storage, over-gated storage, silent-zero copy) have no remaining
instances.

## Findings

| # | Site | Issue | Severity | Status |
|---|---|---|---|---|
| **1** | `yawgpu-core/src/format.rs:345` `RGB10A2_UNORM` caps | `float_color(4,4).blendable().renderable().multisample()` — never calls `.alpha()`, so `has_alpha = false` (every other 4-component color format sets it). The sole consumer is the alpha-to-coverage check (`render_pipeline.rs:2046`), so a render pipeline whose only color target is `rgb10a2unorm` with `multisample.alphaToCoverageEnabled = true` is **wrongly rejected** ("alphaToCoverage requires an alpha blendable color target"), while the identical `rgba8unorm` pipeline is accepted. (`rgb10a2uint` also lacks `.alpha()` but is uint → not blendable → no consequence.) | over-strict / wrong-cap (narrow) | **FIX (this slice)** |
| **2** | `yawgpu-core/src/adapter.rs:206` `supported_features()` | advertises `TextureCompressionBc`/`BcSliced3d`/`Etc2`/`Astc`/`AstcSliced3d` **unconditionally**, and the core `caps()` table accepts compressed 2D textures once the feature is enabled — but NO compressed format has a `HalTextureFormat` variant (`hal_texture_format` → `Unsupported`), so Vulkan + Metal **hard-error at `create_texture`**. The device advertises 5 features it cannot fulfill. WebGPU requires an advertised feature to be fully usable. Invisible on the Noop CTS gate (Noop doesn't allocate). `cts-coverage.md:345` notes the HAL compressed-format deferral but not that core still over-advertises. | conformance (feature-set) | **PENDING USER DECISION** (stop-advertise / implement / document) |

## Ambiguous (needs spec confirmation — not fixed)
- `texture-formats-tier2` read-write-storage set: yawgpu grants it to `{R32Float, RGBA16Float, RGBA32Float}`
  (+ always-on `R32Uint/Sint`); the impl and its CTS port (`tier2_read_write_formats`) agree, but the current
  WebGPU `texture-formats-tier2` spec may extend read-write storage to a broader set (rg32*, rgba32*,
  rgba16*, rgba8*). Impl+test self-consistent → low confidence; confirm against current spec before changing.

## RESOLVED (commit pending push)
- **#1 rgb10a2unorm** now sets `.alpha()`; alpha-to-coverage pipeline targeting it accepted (Noop unit).
- **#2 compressed formats IMPLEMENTED** (user chose implement): `HalTextureFormat` variants for BC1–BC7 /
  ETC2+EAC / ASTC LDR (+sRGB); `hal_texture_format` arms; Vulkan + Metal native mappings (correct block byte
  sizes); GLES constant mappings (advertises nothing yet → deferred). **Per-device feature gating** (the
  conformance fix): `supported_features()` no longer advertises compression unconditionally;
  `Adapter::features()` adds each family iff the device supports it AND it's implemented (Vulkan
  `vkGetPhysicalDeviceFeatures`; Metal `supportsBCTextureCompression`+`supportsFamily(Apple2)`; GLES false;
  Noop true). Compressed B2T/T2B copies use block sizing (incl. the Round-2 Vulkan `bufferImageHeight`
  block→texel fix for multi-layer copies). Sliced-3d + GLES advertisement deferred.
- **Verified real-GPU Metal + Vulkan/MoltenVK:** BC1/ETC2/ASTC single-block create+writeTexture+T2B
  round-trip, and a **2-layer** BC1 round-trip (the multi-layer probe that caught the Round-2
  `bufferImageHeight` MAJOR — per-layer block correct after the fix). 13→14 e2e probes per backend green; no
  regression. Noop unit tests (rgb10a2 alpha, gating, copy-block, Vulkan layout conversion). Clean Review:
  no CRITICAL/MAJOR after the Round-2 fix.
- **Ambiguous (still deferred):** tier2 read-write-storage breadth (impl + CTS port agree; needs spec
  confirmation).
