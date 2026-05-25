// triangle_passthrough — same RGB-corner gradient triangle as
// `examples/triangle`, but fed to the GPU through yawgpu's vendor
// SPIR-V / MSL shader-passthrough APIs instead of the WGSL path.
//
// Why this exists
// ---------------
// WebGPU's portable shader language is WGSL. yawgpu's default path
// compiles a WGSL source to whatever the backend wants (SPIR-V for
// Vulkan, MSL for Metal) via the bundled `naga` compiler. That covers
// the vast majority of WebGPU usage.
//
// Some applications can't or don't want to go through WGSL:
//   * Engines that already have a SPIR-V/MSL toolchain (hand-tuned
//     GLSL, offline-compiled HLSL, custom shader graphs that emit
//     SPIR-V directly).
//   * Apps porting an existing Vulkan/Metal codebase that wants to
//     keep its native shader assets.
//   * Cases where WGSL doesn't expose a backend feature yet (subgroup
//     intrinsics, etc.) and the workaround is to drop down to native
//     shader bytecode.
//
// For these cases yawgpu exposes two vendor entry points (gated by the
// `shader-passthrough` cargo feature; `YAWGPU_HAS_SHADER_PASSTHROUGH`
// is defined when the C ABI is built with that feature):
//
//   * `yawgpuDeviceCreateShaderModuleSpirV(device, &desc)`
//     - `desc.code` is a `uint32_t*` array of SPIR-V words (NOT bytes).
//     - `desc.codeSize` is the WORD count, not the byte count.
//     - yawgpu does NOT reflect the SPIR-V; the entry-point name
//       written in the SPIR-V must match what the pipeline descriptor
//       passes at `vertex.entryPoint` / `fragment.entryPoint`.
//     - A SPIR-V module typically contains one stage (vertex or
//       fragment), so two `.spv` files become two `WGPUShaderModule`
//       handles which the pipeline references separately.
//
//   * `yawgpuDeviceCreateShaderModuleMsl(device, &desc)`
//     - `desc.code` is a `WGPUStringView` of plain MSL text.
//     - `desc.entryPoints` is a `YaWGPUMslEntryPoint[]` declaring each
//       function the caller intends to use (name, stage, optional
//       compute workgroup size). The caller must enumerate them
//       because yawgpu doesn't reflect MSL either.
//     - A single MSL source can host vertex + fragment + compute, so
//       this example uses one `WGPUShaderModule` for both stages.
//
// Buffer-binding-index rule (MSL only, see yawgpu.h comment): yawgpu
// assigns dense `[[buffer(N)]]` Metal indices in (group, binding) order
// derived from the explicit pipeline layout. Textures and samplers
// keep the WebGPU binding number as-is. This example has zero
// resource bindings so the rule is moot here, but production MSL
// shaders must respect it (see the worked example in yawgpu.h).
//
// Backend selection
// -----------------
// At startup we query `WGPUAdapterInfo.backendType` and pick the
// matching shader source. The example self-skips on Noop (the
// passthrough APIs are real-backend-only — Noop has no shader compiler
// to feed). On non-tiled/non-passthrough builds the
// `YAWGPU_HAS_SHADER_PASSTHROUGH` guard at the top of `main` exits
// politely.
//
// Visual equivalence across backends
// ----------------------------------
// Vulkan's clip space has +Y pointing down, while WGSL/Metal both
// have +Y up. The hand-written GLSL vertex shader negates Y to
// compensate; the hand-written MSL vertex shader does not. As a
// result the rendered triangle points up on all three backends.

#include "framework.h"

#if !defined(YAWGPU_HAS_SHADER_PASSTHROUGH)

int main(void) {
    puts("shader-passthrough feature not enabled");
    return EXIT_SUCCESS;
}

#else

#include <stdlib.h>

// Shared per-app state, same shape as `examples/triangle::TriangleApp`
// plus an optional second shader module for the SPIR-V path.
typedef struct PassthroughApp {
    YawgpuContext context;
    WGPUQueue queue;
    YawgpuWindow *window;
    WGPUSurface surface;
    char shader_prefix[1024];
    // SPIR-V uses two modules (one per stage); MSL uses one module
    // hosting both `vs_main` and `fs_main`. We store both fields and
    // only populate the ones the active backend needs.
    WGPUShaderModule msl_module;
    WGPUShaderModule spirv_vs_module;
    WGPUShaderModule spirv_fs_module;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
} PassthroughApp;

// Picks the directory `argv[0]` was launched from so the binary can
// fopen its sibling .spv / .msl files regardless of cwd. Mirrors
// `tiled_deferred::set_shader_prefix` (handles `/` and `\`).
static void set_shader_prefix(PassthroughApp *app, const char *argv0) {
    app->shader_prefix[0] = '\0';
    if (!argv0) {
        return;
    }
    const char *fwd = strrchr(argv0, '/');
    const char *bwd = strrchr(argv0, '\\');
    const char *slash = (fwd && bwd) ? (fwd > bwd ? fwd : bwd) : (fwd ? fwd : bwd);
    if (!slash) {
        return;
    }
    size_t length = (size_t)(slash - argv0 + 1);
    if (length >= sizeof(app->shader_prefix)) {
        length = sizeof(app->shader_prefix) - 1;
    }
    memcpy(app->shader_prefix, argv0, length);
    app->shader_prefix[length] = '\0';
}

// Reads an entire file into a heap-allocated buffer. `out_size` receives
// the byte count. Returns NULL on any error. The caller owns the buffer
// and must `free()` it. Works for both binary (.spv) and text (.msl) —
// .msl is read as-is (no NUL terminator added; the caller knows the
// length).
static uint8_t *read_file_bytes(const char *path, size_t *out_size) {
    *out_size = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    uint8_t *buffer = (uint8_t *)malloc((size_t)size);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, fp) != (size_t)size) {
        free(buffer);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)size;
    return buffer;
}

// Resolves a relative shader filename against the binary's directory.
// Returns malloc'd path the caller must `free`.
static char *resolve_shader_path(const PassthroughApp *app, const char *name) {
    size_t prefix_len = strlen(app->shader_prefix);
    size_t name_len = strlen(name);
    char *out = (char *)malloc(prefix_len + name_len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, app->shader_prefix, prefix_len);
    memcpy(out + prefix_len, name, name_len + 1);
    return out;
}

// Loads a .spv file and feeds the SPIR-V words to
// `yawgpuDeviceCreateShaderModuleSpirV`. Important details:
//   * `.spv` files are little-endian 32-bit-word streams. The descriptor
//     wants a `uint32_t*` and a WORD count (not byte count).
//   * The file size must be a multiple of 4. We reject non-multiples
//     defensively — a misaligned .spv usually means a copy-paste
//     truncation rather than a yawgpu bug.
//   * yawgpu does NOT compile or validate the SPIR-V at module-creation
//     time; defects surface later at pipeline creation as device errors.
static WGPUShaderModule create_spirv_module(PassthroughApp *app, const char *name) {
    char *path = resolve_shader_path(app, name);
    if (!path) {
        return NULL;
    }
    size_t byte_size = 0;
    uint8_t *bytes = read_file_bytes(path, &byte_size);
    free(path);
    if (!bytes) {
        fprintf(stderr, "failed to read shader file: %s\n", name);
        return NULL;
    }
    if (byte_size == 0 || byte_size % sizeof(uint32_t) != 0) {
        fprintf(stderr, "%s is not a valid SPIR-V byte stream (size=%zu)\n", name, byte_size);
        free(bytes);
        return NULL;
    }
    YaWGPUShaderModuleSpirVDescriptor spirv = YAWGPU_SHADER_MODULE_SPIRV_DESCRIPTOR_INIT;
    spirv.label = yawgpu_string_view(name);
    spirv.codeSize = (uint32_t)(byte_size / sizeof(uint32_t));
    spirv.code = (const uint32_t *)bytes;
    WGPUShaderModule module = yawgpuDeviceCreateShaderModuleSpirV(app->context.device, &spirv);
    free(bytes);
    if (!module) {
        fprintf(stderr, "yawgpuDeviceCreateShaderModuleSpirV failed for %s\n", name);
    }
    return module;
}

// Loads a .msl file as a `WGPUStringView` and feeds it to
// `yawgpuDeviceCreateShaderModuleMsl` along with the explicit entry-point
// list. yawgpu does NOT parse the MSL to discover stages — the entry
// list must match what the source actually declares (function name +
// `[[vertex]]`/`[[fragment]]`/`[[kernel]]` qualifier).
static WGPUShaderModule create_msl_module(PassthroughApp *app, const char *name) {
    char *path = resolve_shader_path(app, name);
    if (!path) {
        return NULL;
    }
    size_t byte_size = 0;
    uint8_t *bytes = read_file_bytes(path, &byte_size);
    free(path);
    if (!bytes) {
        fprintf(stderr, "failed to read shader file: %s\n", name);
        return NULL;
    }

    // Two entry points for this example's gradient triangle: vertex
    // `vs_main` and fragment `fs_main`. `workgroupSize` is only meaningful for
    // compute (`WGPUShaderStage_Compute`); the `_INIT` macro leaves it
    // zeroed which is exactly what we want for graphics stages.
    YaWGPUMslEntryPoint entries[2] = {
        YAWGPU_MSL_ENTRY_POINT_INIT,
        YAWGPU_MSL_ENTRY_POINT_INIT,
    };
    entries[0].name = yawgpu_string_view("vs_main");
    entries[0].stage = WGPUShaderStage_Vertex;
    entries[1].name = yawgpu_string_view("fs_main");
    entries[1].stage = WGPUShaderStage_Fragment;

    YaWGPUShaderModuleMslDescriptor msl = YAWGPU_SHADER_MODULE_MSL_DESCRIPTOR_INIT;
    msl.label = yawgpu_string_view(name);
    msl.code = (WGPUStringView){.data = (const char *)bytes, .length = byte_size};
    msl.entryPointCount = 2;
    msl.entryPoints = entries;

    WGPUShaderModule module = yawgpuDeviceCreateShaderModuleMsl(app->context.device, &msl);
    free(bytes);
    if (!module) {
        fprintf(stderr, "yawgpuDeviceCreateShaderModuleMsl failed for %s\n", name);
    }
    return module;
}

// Same surface-format heuristic as `examples/triangle`.
static bool choose_surface_format(WGPUSurface surface,
                                  WGPUAdapter adapter,
                                  WGPUTextureFormat *format) {
    WGPUSurfaceCapabilities capabilities = {0};
    if (wgpuSurfaceGetCapabilities(surface, adapter, &capabilities) != WGPUStatus_Success) {
        fprintf(stderr, "failed to get surface capabilities\n");
        return false;
    }
    bool found = false;
    *format = WGPUTextureFormat_BGRA8Unorm;
    for (size_t i = 0; i < capabilities.formatCount; ++i) {
        if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm) {
            *format = WGPUTextureFormat_BGRA8Unorm;
            found = true;
            break;
        }
        if (capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            *format = WGPUTextureFormat_RGBA8Unorm;
            found = true;
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    if (!found) {
        fprintf(stderr, "no supported surface format found\n");
    }
    return found;
}

static void app_destroy(PassthroughApp *app) {
    if (app->pipeline) wgpuRenderPipelineRelease(app->pipeline);
    if (app->pipeline_layout) wgpuPipelineLayoutRelease(app->pipeline_layout);
    if (app->msl_module) wgpuShaderModuleRelease(app->msl_module);
    if (app->spirv_vs_module) wgpuShaderModuleRelease(app->spirv_vs_module);
    if (app->spirv_fs_module) wgpuShaderModuleRelease(app->spirv_fs_module);
    if (app->surface) {
        wgpuSurfaceUnconfigure(app->surface);
        wgpuSurfaceRelease(app->surface);
    }
    if (app->window) yawgpu_window_destroy(app->window);
    if (app->queue) wgpuQueueRelease(app->queue);
    yawgpu_context_release(&app->context);
    *app = (PassthroughApp){0};
}

// Build the pipeline. The two backends produce different `WGPUVertexState`
// / `WGPUFragmentState.module` references:
//   * SPIR-V — one module per stage (two separate .spv files), each
//     declares the SPIR-V default entry point "main".
//   * MSL — one module shared by both stages, with stage-specific entry
//     names "vs_main" and "fs_main".
// Everything else (pipeline layout, primitive state, color target,
// multisample) is identical to the WGSL `examples/triangle`.
static bool create_pipeline(PassthroughApp *app, WGPUTextureFormat format, bool use_msl) {
    app->pipeline_layout = wgpuDeviceCreatePipelineLayout(
        app->context.device,
        &(WGPUPipelineLayoutDescriptor){
            .label = yawgpu_string_view("triangle passthrough layout"),
            .bindGroupLayoutCount = 0,
            .bindGroupLayouts = NULL,
        });
    if (!app->pipeline_layout) {
        fprintf(stderr, "failed to create pipeline layout\n");
        return false;
    }

    WGPUShaderModule vs_module;
    WGPUShaderModule fs_module;
    const char *vs_entry;
    const char *fs_entry;
    if (use_msl) {
        vs_module = app->msl_module;
        fs_module = app->msl_module;
        vs_entry = "vs_main";
        fs_entry = "fs_main";
    } else {
        vs_module = app->spirv_vs_module;
        fs_module = app->spirv_fs_module;
        // glslangValidator names the SPIR-V entry point "main" by default;
        // see the triangle.{vert,frag}.glsl headers.
        vs_entry = "main";
        fs_entry = "main";
    }

    app->pipeline = wgpuDeviceCreateRenderPipeline(
        app->context.device,
        &(WGPURenderPipelineDescriptor){
            .label = yawgpu_string_view("triangle passthrough pipeline"),
            .layout = app->pipeline_layout,
            .vertex = {
                .module = vs_module,
                .entryPoint = yawgpu_string_view(vs_entry),
                .bufferCount = 0,
                .buffers = NULL,
            },
            .primitive = {
                .topology = WGPUPrimitiveTopology_TriangleList,
                .stripIndexFormat = WGPUIndexFormat_Undefined,
                .frontFace = WGPUFrontFace_CCW,
                .cullMode = WGPUCullMode_None,
            },
            .depthStencil = NULL,
            .multisample = {
                .count = 1,
                .mask = 0xFFFFFFFFu,
                .alphaToCoverageEnabled = false,
            },
            .fragment = &(WGPUFragmentState){
                .module = fs_module,
                .entryPoint = yawgpu_string_view(fs_entry),
                .targetCount = 1,
                .targets = (WGPUColorTargetState[]){
                    {
                        .format = format,
                        .blend = NULL,
                        .writeMask = WGPUColorWriteMask_All,
                    },
                },
            },
        });
    if (!app->pipeline) {
        fprintf(stderr, "failed to create render pipeline\n");
        return false;
    }
    return true;
}

static bool app_init(PassthroughApp *app, const char *argv0) {
    *app = (PassthroughApp){0};
    set_shader_prefix(app, argv0);
    app->context = yawgpu_context_create();
    if (!app->context.instance || !app->context.adapter || !app->context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return false;
    }
    app->queue = wgpuDeviceGetQueue(app->context.device);
    if (!app->queue) {
        fprintf(stderr, "failed to get queue\n");
        return false;
    }

    // Backend pick. SPIR-V passthrough on Vulkan, MSL passthrough on
    // Metal. Noop / other backends have no native shader compiler and
    // self-skip with exit 0 so CI gates don't fail on a missing real
    // adapter.
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(app->context.adapter, &info) != WGPUStatus_Success) {
        fprintf(stderr, "failed to query adapter info\n");
        return false;
    }
    WGPUBackendType backend = info.backendType;
    wgpuAdapterInfoFreeMembers(info);

    bool use_msl;
    if (backend == WGPUBackendType_Metal) {
        use_msl = true;
    } else if (backend == WGPUBackendType_Vulkan) {
        use_msl = false;
    } else {
        printf("triangle_passthrough skipped: backend type=%u is not Metal or Vulkan\n",
               (unsigned int)backend);
        return false;
    }

    // Build the shader modules BEFORE opening the window so a missing
    // .spv / .msl file aborts with a clear error.
    if (use_msl) {
        app->msl_module = create_msl_module(app, "triangle.msl");
        if (!app->msl_module) {
            return false;
        }
    } else {
        app->spirv_vs_module = create_spirv_module(app, "triangle.vert.spv");
        app->spirv_fs_module = create_spirv_module(app, "triangle.frag.spv");
        if (!app->spirv_vs_module || !app->spirv_fs_module) {
            return false;
        }
    }

    app->window = yawgpu_window_create(800, 600, "yawgpu triangle passthrough");
    if (!app->window) {
        fprintf(stderr, "failed to create window\n");
        return false;
    }
    app->surface = yawgpu_window_create_surface(app->context.instance,
                                                app->window,
                                                "triangle passthrough surface");
    if (!app->surface) {
        fprintf(stderr, "failed to create surface\n");
        return false;
    }
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    if (!choose_surface_format(app->surface, app->context.adapter, &format)) {
        return false;
    }
    if (!create_pipeline(app, format, use_msl)) {
        return false;
    }

    int width = 0;
    int height = 0;
    yawgpu_window_framebuffer_size(app->window, &width, &height);
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "invalid framebuffer size\n");
        return false;
    }
    wgpuSurfaceConfigure(
        app->surface,
        &(WGPUSurfaceConfiguration){
            .device = app->context.device,
            .format = format,
            .usage = WGPUTextureUsage_RenderAttachment,
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .alphaMode = WGPUCompositeAlphaMode_Opaque,
            .presentMode = WGPUPresentMode_Fifo,
        });
    return true;
}

// Same per-frame render loop as `examples/triangle`. Acquire texture →
// begin render pass clearing to black → bind pipeline → draw 3 verts →
// submit → present.
static bool render_frame(const PassthroughApp *app) {
    WGPUSurfaceTexture current = {0};
    wgpuSurfaceGetCurrentTexture(app->surface, &current);
    if (current.status == WGPUSurfaceGetCurrentTextureStatus_Lost && !current.texture) {
        return true;
    }
    if (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
        !current.texture) {
        fprintf(stderr, "failed to acquire surface texture, status=%u\n", current.status);
        return false;
    }
    WGPUTextureView view = wgpuTextureCreateView(current.texture, NULL);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(app->context.device, NULL);
    bool ok = view && encoder;
    if (ok) {
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
            encoder,
            &(WGPURenderPassDescriptor){
                .label = yawgpu_string_view("triangle passthrough pass"),
                .colorAttachmentCount = 1,
                .colorAttachments = (WGPURenderPassColorAttachment[]){
                    {
                        .view = view,
                        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                        .loadOp = WGPULoadOp_Clear,
                        .storeOp = WGPUStoreOp_Store,
                        .clearValue = {.r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0},
                    },
                },
            });
        if (!pass) {
            ok = false;
        } else {
            wgpuRenderPassEncoderSetPipeline(pass, app->pipeline);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
        }
    }
    WGPUCommandBuffer commands = NULL;
    if (ok) {
        commands = wgpuCommandEncoderFinish(encoder, NULL);
        ok = commands != NULL;
    }
    if (ok) {
        wgpuQueueSubmit(app->queue, 1, &commands);
    }
    if (commands) wgpuCommandBufferRelease(commands);
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (ok) {
        if (wgpuSurfacePresent(app->surface) != WGPUStatus_Success) {
            fprintf(stderr, "surface present failed\n");
            ok = false;
        }
    }
    if (view) wgpuTextureViewRelease(view);
    wgpuTextureRelease(current.texture);
    return ok;
}

int main(int argc, char **argv) {
    PassthroughApp app = {0};
    if (!app_init(&app, argc > 0 ? argv[0] : NULL)) {
        app_destroy(&app);
        // app_init returning false on Metal/Vulkan is a real failure; on
        // Noop or other backends it's a clean self-skip. Distinguish by
        // checking whether we even got to the point of creating a window:
        // if not, treat it as a skip.
        return app.window ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    while (!yawgpu_window_should_close(app.window)) {
        if (!render_frame(&app)) {
            app_destroy(&app);
            return EXIT_FAILURE;
        }
        yawgpu_window_poll_events();
    }
    app_destroy(&app);
    return EXIT_SUCCESS;
}

#endif // YAWGPU_HAS_SHADER_PASSTHROUGH
