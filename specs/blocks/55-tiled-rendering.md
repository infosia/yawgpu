# Block 55 — Tiled rendering (TBDR mobile extension)

Phase 14. **Vendor extension** (not a Dawn port): tile-based deferred
rendering primitives for mobile GPUs — transient/memoryless attachments,
multi-subpass render passes, subpass-input / framebuffer fetch, subpass-aware
render pipelines. (Programmable tile dispatch was an earlier scaffold; it was
removed before Phase 14 closed — see "Programmable tile dispatch — removed".)
Rules are
exercised by **direct unit tests** (CLAUDE.md principle 1) plus real-backend
`#[ignore]` e2e. Status legend: ☐ todo ◐ partial ☑ done.

Gated by the cargo feature **`tiled`** (default **off**). Purely additive: no
existing public signature/struct changes. **Vulkan + Metal only** (no GL/D3D,
per CLAUDE.md "Out of scope"); Noop accepts handles/descriptors and performs no
GPU work.

## Motivation

On tile-based deferred renderers (Apple GPUs; mobile Vulkan) a multi-pass
G-buffer pipeline can keep intermediate attachments **in tile memory** instead
of round-tripping to system RAM, saving bandwidth and power. WebGPU's core API
has no concept of subpasses or transient attachments; this block exposes them
as additive yawgpu vendor entry points, mirroring the semantics of the
reference mobile extension while staying within yawgpu's enum-dispatch HAL and
C-ABI conventions.

## Surface (yawgpu.h — vendor additions)

> All descriptor structs ship with a `YAWGPU_*_INIT` zero/sentinel
> initializer macro (matching `webgpu.h` ergonomics), since several carry
> nullable pointers, counts, labels, and chained structs.

### Capabilities + features (B1)

```c
typedef struct YaWGPUTiledCapabilities {
    WGPUChainedStruct const* nextInChain;
    uint32_t maxSubpasses;
    uint32_t maxSubpassColorAttachments;
    uint32_t maxInputAttachments;
    uint32_t estimatedTileMemoryBytes;
} YaWGPUTiledCapabilities;

WGPUStatus yawgpuAdapterGetTiledCapabilities(
    WGPUAdapter adapter, YaWGPUTiledCapabilities* capabilities);

/* vendor feature names (0x7001_xxxx; usable via standard
   wgpuAdapterHasFeature / WGPUDeviceDescriptor.requiredFeatures) */
#define YaWGPUFeatureName_MultiSubpass             ((WGPUFeatureName)0x70010001)
#define YaWGPUFeatureName_TransientAttachments     ((WGPUFeatureName)0x70010002)
#define YaWGPUFeatureName_ShaderFramebufferFetch   ((WGPUFeatureName)0x70010003)
/* 0x70010004 reserved — see "Programmable tile dispatch — removed". */
```

### Transient attachment — first-class Arc resource (B2)

```c
typedef struct YaWGPUTransientAttachmentImpl* YaWGPUTransientAttachment;

typedef enum YaWGPUTransientSizeMode {
    YaWGPUTransientSizeMode_MatchTarget = 0x00000000, /* follow render target */
    YaWGPUTransientSizeMode_Explicit    = 0x00000001,
    YaWGPUTransientSizeMode_Force32      = 0x7FFFFFFF
} YaWGPUTransientSizeMode;

typedef struct YaWGPUTransientAttachmentDescriptor {
    WGPUChainedStruct const*    nextInChain;
    WGPUStringView              label;
    WGPUTextureFormat           format;
    YaWGPUTransientSizeMode sizeMode;
    uint32_t                    width;   /* Explicit only */
    uint32_t                    height;  /* Explicit only */
    uint32_t                    sampleCount;
} YaWGPUTransientAttachmentDescriptor;

YaWGPUTransientAttachment yawgpuDeviceCreateTransientAttachment(
    WGPUDevice device, YaWGPUTransientAttachmentDescriptor const* descriptor);
void yawgpuTransientAttachmentAddRef(YaWGPUTransientAttachment attachment);
void yawgpuTransientAttachmentRelease(YaWGPUTransientAttachment attachment);
```

A transient attachment is **only** usable inside a subpass render pass — as a
color/depth slot resource or as an input-attachment source. It is never bound
by the caller through a bind group, so it does not expose a `WGPUTextureView`.

### Subpass-input binding layout — chained on the BGL entry (B3)

```c
#define YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT ((WGPUSType)0x70000010)

typedef struct YaWGPUInputAttachmentBindingLayout {
    WGPUChainedStruct     chain;        /* on WGPUBindGroupLayoutEntry.nextInChain */
    WGPUTextureSampleType sampleType;
    WGPUBool              multisampled;
} YaWGPUInputAttachmentBindingLayout;
```

This chained layout marks a `(group, binding)` in a bind-group/pipeline layout
as an **input attachment** so the descriptor-set/pipeline layout is correct.
The *resource* feeding it is wired **automatically** by yawgpu from the pass
layout's input-attachment source mapping (pass-local model, below) — the caller
does **not** create a bind group or supply a view for input attachments.

### Subpass pass layout — reusable compatibility object (B4)

The render-pass *shape* (attachment formats/sample counts, per-subpass
attachment usage, input-attachment source mapping, dependencies) is described
once in a `YaWGPUSubpassPassLayout`, then referenced by **both** pipeline
creation and pass begin. This is the single source of truth for Vulkan
`VkRenderPass` compatibility; actual views/transient resources are attached only
at pass begin.

```c
typedef struct YaWGPUSubpassPassLayoutImpl* YaWGPUSubpassPassLayout;

/* static description of one attachment slot (shape only, no resource) */
typedef struct YaWGPUAttachmentLayout {
    WGPUTextureFormat format;       /* WGPUTextureFormat_Undefined = unused slot */
    uint32_t          sampleCount;
} YaWGPUAttachmentLayout;

typedef enum YaWGPUSubpassDependencyType {
    YaWGPUSubpassDependencyType_ColorToInput      = 0x00000000,
    YaWGPUSubpassDependencyType_DepthToInput      = 0x00000001,
    YaWGPUSubpassDependencyType_ColorDepthToInput = 0x00000002,
    YaWGPUSubpassDependencyType_Force32           = 0x7FFFFFFF
} YaWGPUSubpassDependencyType;

typedef struct YaWGPUSubpassDependency {
    uint32_t                        srcSubpass;
    uint32_t                        dstSubpass;
    YaWGPUSubpassDependencyType dependencyType;
    WGPUBool                        byRegion;
} YaWGPUSubpassDependency;

/* DEPTH_STENCIL sentinel for an input-attachment source that reads depth */
#define YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX 0xFFFFFFFFu

/* a prior subpass attachment feeding a shader binding (group + binding) */
typedef struct YaWGPUSubpassInputAttachment {
    uint32_t group;
    uint32_t binding;
    uint32_t sourceSubpass;
    uint32_t sourceAttachment; /* color slot index, or the depth-stencil sentinel */
} YaWGPUSubpassInputAttachment;

/* per-subpass shape: which slots it writes, whether it uses depth/stencil,
   which prior attachments it reads as inputs */
typedef struct YaWGPUSubpassLayoutDesc {
    uint32_t const*                         colorAttachmentIndices;
    size_t                                  colorAttachmentIndexCount;
    WGPUBool                                usesDepthStencil;
    YaWGPUSubpassInputAttachment const* inputAttachments;
    size_t                                  inputAttachmentCount;
} YaWGPUSubpassLayoutDesc;

typedef struct YaWGPUSubpassPassLayoutDescriptor {
    WGPUChainedStruct const*           nextInChain;
    WGPUStringView                     label;
    YaWGPUAttachmentLayout const*  colorAttachments;     /* indexed by slot */
    size_t                             colorAttachmentCount;
    YaWGPUAttachmentLayout         depthStencilAttachment;/* format Undefined = none */
    YaWGPUSubpassLayoutDesc const* subpasses;
    size_t                             subpassCount;
    YaWGPUSubpassDependency const* dependencies;
    size_t                             dependencyCount;
} YaWGPUSubpassPassLayoutDescriptor;

YaWGPUSubpassPassLayout yawgpuDeviceCreateSubpassPassLayout(
    WGPUDevice device, YaWGPUSubpassPassLayoutDescriptor const* descriptor);
void yawgpuSubpassPassLayoutAddRef(YaWGPUSubpassPassLayout layout);
void yawgpuSubpassPassLayoutRelease(YaWGPUSubpassPassLayout layout);
```

### Subpass render pass — attaches resources to a layout (B4)

```c
typedef struct YaWGPUSubpassRenderPassEncoderImpl* YaWGPUSubpassRenderPassEncoder;

typedef enum YaWGPUSubpassAttachmentKind {
    YaWGPUSubpassAttachmentKind_Persistent = 0x00000000, /* view */
    YaWGPUSubpassAttachmentKind_Transient  = 0x00000001, /* transient handle */
    YaWGPUSubpassAttachmentKind_Force32     = 0x7FFFFFFF
} YaWGPUSubpassAttachmentKind;

/* one bound color slot: Persistent carries a view + resolve target,
   Transient carries a transient-attachment handle directly. */
typedef struct YaWGPUColorAttachmentBinding {
    YaWGPUSubpassAttachmentKind kind;
    WGPUTextureView                 view;          /* Persistent */
    WGPUTextureView                 resolveTarget;  /* Persistent, optional */
    YaWGPUTransientAttachment   transient;     /* Transient */
    WGPULoadOp                      loadOp;
    WGPUStoreOp                     storeOp;
    WGPUColor                       clearValue;
} YaWGPUColorAttachmentBinding;

typedef struct YaWGPUDepthStencilAttachmentBinding {
    YaWGPUSubpassAttachmentKind kind;
    WGPUTextureView                 view;        /* Persistent */
    YaWGPUTransientAttachment   transient;   /* Transient */
    WGPULoadOp                      depthLoadOp;
    WGPUStoreOp                     depthStoreOp;
    float                           depthClearValue;
    WGPULoadOp                      stencilLoadOp;
    WGPUStoreOp                     stencilStoreOp;
    uint32_t                        stencilClearValue;
} YaWGPUDepthStencilAttachmentBinding;

typedef struct YaWGPUSubpassRenderPassDescriptor {
    WGPUChainedStruct const*                       nextInChain;
    WGPUStringView                                 label;
    YaWGPUSubpassPassLayout                    passLayout;  /* the shape */
    WGPUExtent3D                                   extent;
    YaWGPUColorAttachmentBinding const*        colorAttachments;     /* indexed by slot */
    size_t                                         colorAttachmentCount;
    YaWGPUDepthStencilAttachmentBinding const* depthStencilAttachment;/* nullable */
} YaWGPUSubpassRenderPassDescriptor;

YaWGPUSubpassRenderPassEncoder yawgpuCommandEncoderBeginSubpassRenderPass(
    WGPUCommandEncoder encoder,
    YaWGPUSubpassRenderPassDescriptor const* descriptor);
void yawgpuSubpassRenderPassEncoderNextSubpass(YaWGPUSubpassRenderPassEncoder e);
void yawgpuSubpassRenderPassEncoderEnd(YaWGPUSubpassRenderPassEncoder e);
void yawgpuSubpassRenderPassEncoderAddRef(YaWGPUSubpassRenderPassEncoder e);
void yawgpuSubpassRenderPassEncoderRelease(YaWGPUSubpassRenderPassEncoder e);
```

**Lifetime:** `BeginSubpassRenderPass` retains the pass layout and every
referenced view / transient attachment until the pass ends (and until queue
submission completes). The caller may `Release` its own references right after
`BeginSubpassRenderPass` returns.

### Per-subpass draw machinery — dedicated encoder (B5)

```c
void yawgpuSubpassRenderPassEncoderSetPipeline(YaWGPUSubpassRenderPassEncoder e, WGPURenderPipeline p);
void yawgpuSubpassRenderPassEncoderSetBindGroup(YaWGPUSubpassRenderPassEncoder e, uint32_t group, WGPUBindGroup bg, size_t dynamicOffsetCount, uint32_t const* dynamicOffsets);
void yawgpuSubpassRenderPassEncoderSetVertexBuffer(YaWGPUSubpassRenderPassEncoder e, uint32_t slot, WGPUBuffer buf, uint64_t offset, uint64_t size);
void yawgpuSubpassRenderPassEncoderSetIndexBuffer(YaWGPUSubpassRenderPassEncoder e, WGPUBuffer buf, WGPUIndexFormat format, uint64_t offset, uint64_t size);
void yawgpuSubpassRenderPassEncoderDraw(YaWGPUSubpassRenderPassEncoder e, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
void yawgpuSubpassRenderPassEncoderDrawIndexed(YaWGPUSubpassRenderPassEncoder e, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
void yawgpuSubpassRenderPassEncoderSetViewport(YaWGPUSubpassRenderPassEncoder e, float x, float y, float w, float h, float minDepth, float maxDepth);
void yawgpuSubpassRenderPassEncoderSetScissorRect(YaWGPUSubpassRenderPassEncoder e, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
```

### Subpass-aware render pipeline (B5)

```c
typedef struct YaWGPUSubpassRenderPipelineDescriptor {
    WGPUChainedStruct const*     nextInChain;
    WGPURenderPipelineDescriptor base;         /* standard descriptor, embedded */
    YaWGPUSubpassPassLayout  passLayout;    /* SAME layout used at pass begin */
    uint32_t                     subpassIndex;  /* which subpass this pipeline targets */
} YaWGPUSubpassRenderPipelineDescriptor;

WGPURenderPipeline yawgpuDeviceCreateSubpassRenderPipeline(
    WGPUDevice device,
    YaWGPUSubpassRenderPipelineDescriptor const* descriptor);
```

Because both the pipeline and the pass reference the **same**
`YaWGPUSubpassPassLayout`, compatibility information is described once and
cannot drift between the two call sites.

### Programmable tile dispatch — removed (was scaffold-only)

Removed in Phase 14 (post-B6). It had no implementation on any backend and no
implementation plan, so shipping the surface ahead of a real impl only locked us
into an API shape that wasn't driven by anything. The numeric IDs
(`YaWGPUFeatureName_ProgrammableTileDispatch == 0x70010004`, and any future
tile-dispatch SType / C entry-point name) are **reserved** by a comment in
`yawgpu.h` so they aren't reused for unrelated features. The API shape will be
defined when a real backend implementation lands.

## Design decisions

- **enum-dispatch, no extension traits.** TBDR folds into the existing
  `HalDevice`/`HalCommandEncoder`/resource enums with `cfg(feature="tiled")`
  arms; per-backend bodies are further `cfg`-gated on `vulkan`/`metal`. No
  `dyn` trait, matching CLAUDE.md.
- **Layout vs resources separation (`YaWGPUSubpassPassLayout`).** The
  pass *shape* (formats, per-subpass usage, input-attachment source mapping,
  dependencies) is created once as an Arc resource and referenced by both
  `create_subpass_render_pipeline` and `begin_subpass_render_pass`. Vulkan builds
  (and caches) the compatible `VkRenderPass` on the layout; the pass begin only
  needs a `VkFramebuffer` over the supplied views/transients. This removes the
  duplicate-compat-info hazard and is the analogue of Vulkan
  `VkRenderPass` (layout) vs `VkFramebuffer`/begin-info (resources).
- **Transient attachment = first-class Arc resource**, supplied directly in the
  `Transient` branch of a color/depth attachment binding (no separate index
  table; the handle *is* the resolution). Usable only inside a subpass pass.
  - Vulkan: `VkImage` with `TRANSIENT_ATTACHMENT | INPUT_ATTACHMENT` usage +
    `LAZILY_ALLOCATED` memory (fallback to a normal allocation if unavailable).
  - Metal: `MTLTexture` with `MTLStorageMode::Memoryless` (fallback `Private`
    with a logged warning).
- **Input attachments are pass-local, auto-wired.** The pass layout declares,
  per subpass, which prior attachment feeds which `(group, binding)`. yawgpu
  binds it automatically (Vulkan `INPUT_ATTACHMENT` descriptor from the layout;
  Metal implicit `[[color(N)]]`). The caller never supplies a view for an
  input-attachment binding slot — regardless of whether the slot lives in an
  input-attachment-**only** group or a **mixed** group that also carries
  other resources. The `(group, binding)` pair uniquely identifies the shader
  binding (resolves the earlier ambiguity). Concretely:
  - **Input-only group** (every slot is `InputAttachment`): the caller does
    not call `wgpuRenderPassEncoderSetBindGroup` for that group at all; the
    subpass pass auto-wires every slot.
  - **Mixed group** (input + non-input slots): the caller creates a
    `WGPUBindGroup` whose `entries[]` covers **only the non-input slots**,
    then calls `setBindGroup` normally. `wgpuDeviceCreateBindGroup` accepts
    a descriptor with the input-slot entries omitted (validation allows the
    entry-count and per-binding-coverage gaps for `InputAttachment` kinds
    only); the pass auto-wires those slots at submit time the same way it
    does for input-only groups. This is the pattern used by wgpu's
    `deferred_rendering` lighting bind group (2 subpass inputs + 1 uniform
    in `@group(0)`).
- **Fragment `@location(N)` on subpass pipelines: dual convention accepted.**
  naga lowers `@location` differently per backend, so the WGSL author has two
  valid choices and yawgpu's validation must accept either (the HAL is the
  authority on routing):
  - **Vulkan (subpass-local).** WGSL writes the *subpass-local* index
    starting from 0 (e.g. a subpass with `color_attachment_indices = [1]`
    accepts `@location(0)`). naga's SPIR-V backend emits `Location 0` and
    `VkRenderPass` remaps it to the flat attachment slot.
  - **Metal (flat slot).** naga's MSL backend does **not** subpass-remap; the
    WGSL must write the *flat MTL slot* directly (the same subpass accepts
    `@location(1)`).
  - **Validation rule.** For a subpass pipeline's fragment target at
    subpass-local index `i`, the shader is valid if it writes either
    `@location(i)` *or* `@location(layout.color_attachment_indices[i])`. A
    pipeline that needs to support both backends from a single WGSL source
    therefore needs two fragment entry points (cf. `mgpu/examples/hello_deferred`
    and `yawgpu/tests/e2e_metal_tiled.rs` — `fs` for Vulkan, `fs_metal` for
    Metal). The example `examples/tiled_deferred` picks the entry by
    `WGPUAdapterInfo.backendType` at runtime.
- **Reference example (`examples/tiled_deferred`) — deferred-shading demo.**
  yawgpu's flagship tiled-rendering demo mirrors the wgpu reference example
  `wgpu-tiled/examples/features/src/deferred_rendering` (in
  `../wgpu/examples/features/src/deferred_rendering` for this fork).
  Target shape:
  - **Three subpasses** in one render pass:
    1. **G-Buffer** — instanced 5×5 cube grid, writes `albedo` (`Rgba8Unorm`)
       + `world_normal+depth` (`Rgba16Float`) with depth testing
       (`Depth32Float`, `LessEqual`, write-enabled).
    2. **Lighting** — fullscreen triangle, reads `albedo` + `normal` as
       subpass inputs, Blinn-Phong with 4 orbiting point lights +
       hemisphere ambient, reconstructs world position from depth + inverse
       view-proj, writes HDR result to `Rgba16Float`.
    3. **Composite** — fullscreen triangle, reads HDR via subpass input,
       Reinhard tonemap, writes to the swapchain (sRGB / linear converted
       in hardware).
  - **WGSL in three separate files** `gbuffer.wgsl`, `lighting.wgsl`,
    `composite.wgsl`, copied verbatim from the wgpu reference (CMake's
    `add_custom_command` POST_BUILD copies them next to the binary, mirroring
    `examples/triangle`). The lighting + composite fragments use the
    **subpass-local convention** (`@location(0)`) so a single WGSL source
    works for both Vulkan and Metal — the cascade `validate_color_targets`
    accepts either convention; the WGSL author picks the subpass-local form
    for portability.
  - **Two run modes** controlled by the `--verify` CLI flag:
    - **Default (windowed demo).** Open a GLFW window (Metal `CAMetalLayer`
      or Vulkan `VK_KHR_*_surface`), configure a swapchain on the surface,
      run the main loop with animated camera + lights (orbiting on
      sub-second-of-day clock) until the user closes the window.
    - **`--verify`** (CI/regression). Render exactly one frame at `time = 0.0`
      to an offscreen `Rgba8Unorm` texture (skip the window), copy to a
      readback buffer, and inspect the center pixel: assert it represents
      a lit cube (alpha > 0, non-zero RGB) rather than the cleared
      background. Also writes the frame to `tiled_deferred.png` for visual
      inspection. The verify mode is the load-bearing
      "deferred-pipeline-actually-ran" check; it is what cargo-style CI
      gates would call once the C example becomes part of an automated
      gate.
  - **Scene values are verbatim from the wgpu reference** so the two
    examples produce visually equivalent output: 5×5 cube grid with spacing
    3.0, eye orbits at radius 12 + offset (0, 8, 15), `frac_pi_4` fovy,
    near/far 0.1/100.0, light positions/intensities exactly as in the
    reference's `LightParams` write.
  - **No depth-format transient attachments yet** (B2 follow-up;
    `examples/tiled_deferred` allocates a regular `RENDER_ATTACHMENT` depth
    texture, like the wgpu reference). When B2's depth-transient gap closes,
    the example switches to `transient` for the depth + g-buffer + HDR slots
    and only the swapchain target stays persistent.
  - **Math** (Mat4 / Vec3 / look_at / perspective / inverse) lives in a
    tiny `examples/tiled_deferred/math.h` (or inline at the top of `main.c`)
    — minimal C helpers, no third-party math dependency.
- **Multi-subpass execution.** Vulkan: `vkCmdBeginRenderPass` (multi-subpass
  `VkRenderPass` from the layout) → `vkCmdNextSubpass` → `vkCmdEndRenderPass`.
  Metal: a single `MTLRenderCommandEncoder` state machine; `next_subpass`
  advances internal state; a `tile_memory_check` rejects passes whose memoryless
  footprint exceeds the device tile budget.
- **Capabilities as a query** (`yawgpuAdapterGetTiledCapabilities`); not
  folded into `WGPULimits` (additive, no struct change).
- **Eager-dispatch ordering constraint (documented).** A subpass render pass
  must be the **first** operation on its command encoder; yawgpu validates this
  (error if the encoder already recorded commands) rather than reconciling
  deferred replay in v1.
- **Error-object model** as elsewhere: invalid descriptors emit a device error;
  resource/encoder creators return a `Release`-safe error handle.
- **naga subpass IR is already present** in the pinned naga (see
  `reference/dependencies.md`); B3 is *enablement + wiring*, not naga
  implementation.

## Core / HAL model (sketch)

- `yawgpu-hal`: `HalTransientAttachment` enum (Vulkan/Metal/Noop); render-pass
  layout types (`HalSubpassDescription`, `HalSubpassDependency`,
  `HalInputAttachmentReference`); `HalCommandEncoder::{begin_subpass_render_pass,
  next_subpass, end_subpass_render_pass}` + per-subpass draw forwarding;
  Vulkan `FramebufferFetchPath { Disabled, TileImage, RasterOrderAttachmentAccess }`.
- `yawgpu-core`: `SubpassPassLayout` + `TransientAttachment` resource wrappers
  (Arc); `SubpassRenderPass` handle holding the layout + active-subpass state;
  subpass-pipeline validation (the pipeline's `passLayout`+`subpassIndex` must
  match the pass at begin); `BindingType::InputAttachment` through
  bind-group-layout / pipeline-layout.
- `yawgpu` (FFI): the `yawgpu*` entry points + `yawgpu.h` declarations +
  `*_INIT` macros + `conv` for the tagged attachment structs.

## Rules

### Capabilities + features (P14.1)

- **T1** `yawgpuAdapterGetTiledCapabilities` returns real per-backend values
  (Vulkan device limits / Metal tile budget); Noop returns zeros + `Success`. ☐ (UT)
- **T2** vendor feature names report via `wgpuAdapterHasFeature` and are accepted
  in `requiredFeatures`; advertised only on a backend that supports them. ☐ (UT noop + e2e)

### Transient attachments (P14.2)

- **T3** `yawgpuDeviceCreateTransientAttachment` returns an Arc handle;
  `AddRef`/`Release` refcount; `Drop` releases the backend image. ☐ (UT)
- **T4** Vulkan allocates `LAZILY_ALLOCATED`+`TRANSIENT_ATTACHMENT|INPUT_ATTACHMENT`
  (fallback normal); Metal `Memoryless` (fallback `Private`+warn). ☐ (e2e)
- **T5** `MatchTarget` resolves size from the pass extent at begin; `Explicit`
  uses width/height; zero-size Explicit ⇒ error. ☐ (UT)
- **T6** Metal `tile_memory_check`: a pass whose memoryless footprint exceeds the
  tile budget ⇒ error. ☐ (e2e Metal)

### Subpass input binding (P14.3)

- **T7** `YaWGPUInputAttachmentBindingLayout` chained on a BGL entry marks a
  `(group, binding)` as input-attachment; the resource is auto-wired from the
  pass layout (Vulkan `INPUT_ATTACHMENT`, Metal implicit). Caller binds no view. ☐ (UT noop + e2e)
- **T8** WGSL using `subpass_input<T>` + `subpassLoad` compiles on both backends
  (entry-point `@color(N)` on Metal; global form Vulkan-only). ☐ (e2e)
- **T9** a `subpass_input<i32>` bound against an `f32` attachment ⇒ pipeline
  layout / derivation error. ☐ (UT)

### Subpass pass layout (P14.4a)

- **T10** `yawgpuDeviceCreateSubpassPassLayout` builds the layout (Vulkan
  caches a compatible `VkRenderPass`); invalid shape (input source out of range,
  subpass/attachment counts beyond `TiledCapabilities`) ⇒ error. ☐ (UT)

### Multi-subpass render pass (P14.4b)

- **T11** `BeginSubpassRenderPass` attaches the supplied views/transients to the
  layout; `NextSubpass` advances; `End` finishes; Drop without `End` still
  releases the HAL pass. ☐ (UT noop + e2e)
- **T12** `Persistent` (view) and `Transient` (handle) color/depth slots both
  work; transient handles are retained for the pass lifetime. ☐ (UT noop + e2e)
- **T13** a subpass render pass that is **not** the first encoder operation ⇒
  error (eager-dispatch ordering). ☐ (UT)
- **T14** attachment kinds / counts inconsistent with `passLayout` (e.g. a slot
  the layout marks unused, count mismatch) ⇒ error. ☐ (UT)

### Subpass pipeline + draw (P14.5)

- **T15** `yawgpuDeviceCreateSubpassRenderPipeline` builds a pipeline
  compatible with `(passLayout, subpassIndex)`; format/subpass mismatch ⇒ error;
  using a pipeline whose layout differs from the pass's layout at draw ⇒ error. ☐ (UT noop + e2e)
- **T16** the dedicated draw encoder methods record into the active subpass and
  register resource tracking (pipeline/bind-group/buffers). ☐ (UT noop + e2e)

### Framebuffer fetch (P14.6)

- **T17** Vulkan `FramebufferFetchPath` is detected (`TileImage` /
  `RasterOrderAttachmentAccess` / `Disabled`); Metal implicit; the
  `ShaderFramebufferFetch` feature is advertised accordingly. ☐ (e2e)

## Async

No new async surface. Submission/work-done reuse block 50's queue machinery.

## Feature gating

- Cargo feature **`tiled`** on `yawgpu` forwards to `yawgpu-core/tiled` +
  `yawgpu-hal/tiled`. Default off. Umbrella `mobile = ["shader-passthrough",
  "tiled"]`.
- When **off**: the `yawgpu*` tiled entry points are not compiled; `yawgpu.h`
  still declares them + a `YAWGPU_HAS_TILED` macro for `#ifdef` guards.
- Orthogonal to `metal`/`vulkan`: core types/validation/Noop compile without a
  backend; real per-backend bodies are `cfg(all(feature="tiled", feature="<bk>"))`.
- **Gates run in both configs** (default + `tiled` on, and with each backend),
  including `clippy -D warnings` + `missing_docs`; feature-gated `pub fn`s carry
  their unit tests under the same `#[cfg]`.

## Slices (1 handoff + 1 commit each)

- **B1** features + `TiledCapabilities` query (T1/T2).
- **B2** transient attachment resource + Vulkan/Metal alloc + `tile_memory_check`
  (T3/T4/T5/T6).
- **B3** naga subpass IR enablement + `BindingType::InputAttachment` plumbing,
  pass-local input wiring (T7/T8/T9).
- **B4** subpass pass layout object (T10) + multi-subpass render pass + encoder
  (T11/T12/T13/T14).
- **B5** subpass render pipeline (references layout) + dedicated draw encoder
  (T15/T16).
- **B6** framebuffer fetch path detection + feature advertise (T17).
- **B7** ~~programmable tile dispatch scaffold~~ — REMOVED (no implementation on
  any backend; numeric ID reserved in `yawgpu.h`).
- **B8** examples (Metal+Vulkan deferred-shading) + e2e (`#[ignore]`) +
  **Phase Review**.

## Known limitations (carried into v1, documented)

1. **Eager-dispatch ordering** — subpass pass must be the first encoder op (T13).
2. **MSL subpass-input globals unsupported** — `@color(N)` entry-point form only;
   the global `var g: subpass_input<f32>;` form works on Vulkan only.
3. **Timestamp / occlusion queries inside subpass passes are out of scope for
   v1** (the standard render pass covers the common case).
4. **GLES / DX12** — not applicable (yawgpu has no such backends).

## Open questions

- `YaWGPUSubpassPassLayout` ↔ Vulkan `VkRenderPass` cache key derivation
  (decide during B4; the layout owns the compat `VkRenderPass`, the pass owns
  the `VkFramebuffer` keyed on the supplied views/transients).
- Whether `MatchTarget` transient attachments are validated against the pass
  extent eagerly at `BeginSubpassRenderPass` or lazily at first use.
- Framebuffer-cache eviction on transient-attachment / texture-view destruction
  (mirror block 50's view lifetime handling).
