// tiled_deferred - three-subpass deferred shading demo for yawgpu's tiled API.

#include "framework.h"

#if !defined(YAWGPU_HAS_TILED)

int main(void) {
    puts("tiled extension not enabled");
    return EXIT_SUCCESS;
}

#else

#include "math.h"
#include "stb_image_write.h"

#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
#define strncasecmp _strnicmp
#else
#include <strings.h>
#endif

enum {
    WINDOW_WIDTH = 1024,
    WINDOW_HEIGHT = 768,
    GRID_SIZE = 5,
    INSTANCE_COUNT = GRID_SIZE * GRID_SIZE,
    VERTEX_COUNT = 24,
    INDEX_COUNT = 36,
    COPY_BYTES_PER_ROW_ALIGNMENT = 256,
    BYTES_PER_PIXEL = 4,
};

typedef struct Vertex {
    float position[3];
    float normal[3];
    float color[3];
} Vertex;

typedef struct Uniforms {
    float view_proj[16];
} Uniforms;

typedef struct LightParams {
    float lights[4][4];
    float camera_pos[3];
    float time;
    float inv_view_proj[16];
    float screen_size[2];
    float _padding[2];
} LightParams;

typedef struct BufferDimensions {
    uint32_t width;
    uint32_t height;
    uint32_t unpadded_bytes_per_row;
    uint32_t padded_bytes_per_row;
} BufferDimensions;

typedef struct MapState {
    WGPUMapAsyncStatus status;
    bool called;
} MapState;

typedef struct QueueWorkDoneState {
    WGPUQueueWorkDoneStatus status;
    bool called;
} QueueWorkDoneState;

typedef enum {
    TILED_BACKEND_OK = 0,
    TILED_BACKEND_FAIL = 1,
    TILED_BACKEND_SKIP = 2,
} TiledBackendStatus;

typedef struct AttachmentResources {
    WGPUTexture albedo_texture;
    WGPUTexture normal_texture;
    WGPUTexture depth_texture;
    WGPUTexture lit_texture;
    WGPUTexture output_texture;
    WGPUTextureView albedo_view;
    WGPUTextureView normal_view;
    WGPUTextureView depth_view;
    WGPUTextureView lit_view;
    WGPUTextureView output_view;
} AttachmentResources;

typedef struct TiledDeferredApp {
    YawgpuContext context;
    WGPUQueue queue;
    YawgpuWindow *window;
    WGPUSurface surface;
    WGPUTextureFormat output_format;
    uint32_t width;
    uint32_t height;
    bool verify;
    char shader_prefix[1024];
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
    WGPUBuffer uniform_buffer;
    WGPUBuffer light_buffer;
    WGPUBuffer output_buffer;
    bool output_buffer_mapped;
    BufferDimensions dimensions;
    uint64_t output_buffer_size;
    WGPUShaderModule gbuffer_module;
    WGPUShaderModule lighting_module;
    WGPUShaderModule composite_module;
    WGPUBindGroupLayout gbuffer_bgl;
    WGPUBindGroupLayout lighting_input_bgl;
    WGPUBindGroupLayout lighting_uniform_bgl;
    WGPUBindGroupLayout composite_bgl;
    WGPUPipelineLayout gbuffer_layout;
    WGPUPipelineLayout lighting_layout;
    WGPUPipelineLayout composite_layout;
    WGPUBindGroup uniform_bind_group;
    WGPUBindGroup lighting_bind_group;
    WGPURenderPipeline gbuffer_pipeline;
    WGPURenderPipeline lighting_pipeline;
    WGPURenderPipeline composite_pipeline;
    YaWGPUSubpassPassLayout pass_layout;
    unsigned int initial_error_count;
    clock_t start_clock;
} TiledDeferredApp;

static uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    uint32_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

static BufferDimensions buffer_dimensions_create(uint32_t width, uint32_t height) {
    return (BufferDimensions){
        .width = width,
        .height = height,
        .unpadded_bytes_per_row = width * BYTES_PER_PIXEL,
        .padded_bytes_per_row = align_up_u32(width * BYTES_PER_PIXEL,
                                             COPY_BYTES_PER_ROW_ALIGNMENT),
    };
}

static void map_callback(WGPUMapAsyncStatus status,
                         WGPUStringView message,
                         void *userdata1,
                         void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    MapState *state = (MapState *)userdata1;
    state->status = status;
    state->called = true;
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "buffer map failed: ");
        yawgpu_print_string_view(message);
        fprintf(stderr, "\n");
    }
}

static void queue_work_done_callback(WGPUQueueWorkDoneStatus status,
                                     WGPUStringView message,
                                     void *userdata1,
                                     void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    QueueWorkDoneState *state = (QueueWorkDoneState *)userdata1;
    state->status = status;
    state->called = true;
    if (status != WGPUQueueWorkDoneStatus_Success) {
        fprintf(stderr, "queue work failed: ");
        yawgpu_print_string_view(message);
        fprintf(stderr, "\n");
    }
}

static void create_cube_vertices(Vertex vertices[VERTEX_COUNT], uint16_t indices[INDEX_COUNT]) {
    const float positions[VERTEX_COUNT][3] = {
        {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},   {1.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f},   {-1.0f, -1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
        {1.0f, 1.0f, -1.0f},   {1.0f, -1.0f, -1.0f},  {-1.0f, 1.0f, -1.0f},
        {-1.0f, 1.0f, 1.0f},   {1.0f, 1.0f, 1.0f},    {1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f},  {1.0f, -1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, -1.0f},  {1.0f, 1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},    {1.0f, -1.0f, 1.0f},   {-1.0f, -1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f},  {-1.0f, 1.0f, 1.0f},   {-1.0f, 1.0f, -1.0f},
    };
    const float normals[VERTEX_COUNT][3] = {
        {0.0f, 0.0f, 1.0f},   {0.0f, 0.0f, 1.0f},   {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},   {0.0f, 0.0f, -1.0f},  {0.0f, 0.0f, -1.0f},
        {0.0f, 0.0f, -1.0f},  {0.0f, 0.0f, -1.0f},  {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},   {0.0f, 1.0f, 0.0f},   {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},  {0.0f, -1.0f, 0.0f},  {0.0f, -1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},  {1.0f, 0.0f, 0.0f},   {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},   {1.0f, 0.0f, 0.0f},   {-1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},  {-1.0f, 0.0f, 0.0f},  {-1.0f, 0.0f, 0.0f},
    };
    const float colors[6][3] = {
        {1.0f, 0.3f, 0.3f}, {0.3f, 1.0f, 0.3f}, {0.3f, 0.3f, 1.0f},
        {1.0f, 1.0f, 0.3f}, {1.0f, 0.3f, 1.0f}, {0.3f, 1.0f, 1.0f},
    };
    for (uint32_t i = 0; i < VERTEX_COUNT; ++i) {
        memcpy(vertices[i].position, positions[i], sizeof(vertices[i].position));
        memcpy(vertices[i].normal, normals[i], sizeof(vertices[i].normal));
        memcpy(vertices[i].color, colors[i / 4], sizeof(vertices[i].color));
    }
    for (uint16_t face = 0; face < 6; ++face) {
        uint16_t base = (uint16_t)(face * 4);
        uint32_t offset = face * 6;
        indices[offset + 0] = base;
        indices[offset + 1] = (uint16_t)(base + 1);
        indices[offset + 2] = (uint16_t)(base + 2);
        indices[offset + 3] = base;
        indices[offset + 4] = (uint16_t)(base + 2);
        indices[offset + 5] = (uint16_t)(base + 3);
    }
}

static void set_shader_prefix(TiledDeferredApp *app, const char *argv0) {
    app->shader_prefix[0] = '\0';
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

static WGPUShaderModule load_shader(TiledDeferredApp *app, const char *name) {
    char path[1200];
    if (app->shader_prefix[0] != '\0') {
        snprintf(path, sizeof(path), "%s%s", app->shader_prefix, name);
        WGPUShaderModule module = yawgpu_load_wgsl_shader(app->context.device, path);
        if (module) {
            return module;
        }
    }
    return yawgpu_load_wgsl_shader(app->context.device, name);
}

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
            found = true;
            *format = WGPUTextureFormat_BGRA8Unorm;
            break;
        }
        if (capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            found = true;
            *format = WGPUTextureFormat_RGBA8Unorm;
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    if (!found) {
        fprintf(stderr, "no supported surface format found\n");
    }
    return found;
}

static bool adapter_is_moltenvk(WGPUAdapter adapter) {
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
        return false;
    }
    const WGPUStringView fields[3] = {info.vendor, info.device, info.description};
    bool found = false;
    for (size_t i = 0; i < 3 && !found; ++i) {
        const char *s = fields[i].data;
        size_t n = fields[i].length;
        for (size_t j = 0; s && j < n; ++j) {
            if (j + 6 <= n && strncasecmp(s + j, "molten", 6) == 0) {
                found = true;
                break;
            }
            if (j + 5 <= n && strncasecmp(s + j, "apple", 5) == 0) {
                found = true;
                break;
            }
        }
    }
    wgpuAdapterInfoFreeMembers(info);
    return found;
}

static TiledBackendStatus require_tiled_backend(WGPUAdapter adapter) {
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
        fprintf(stderr, "failed to query adapter info\n");
        return TILED_BACKEND_FAIL;
    }
    WGPUBackendType backend = info.backendType;
    wgpuAdapterInfoFreeMembers(info);
    if (backend != WGPUBackendType_Metal && backend != WGPUBackendType_Vulkan) {
        const char *requested_backend = getenv("YAWGPU_BACKEND");
        if (requested_backend && strcmp(requested_backend, "vulkan") == 0) {
            fprintf(stderr,
                    "skipping tiled_deferred: requested Vulkan but selected backend type=%u\n",
                    (unsigned int)backend);
            return TILED_BACKEND_SKIP;
        }
        fprintf(stderr, "tiled_deferred requires Metal or native Vulkan; selected backend type=%u\n",
                (unsigned int)backend);
        return TILED_BACKEND_FAIL;
    }
    if (backend == WGPUBackendType_Vulkan && adapter_is_moltenvk(adapter)) {
        fprintf(stderr,
                "skipping tiled_deferred on MoltenVK; subpass-input read requires "
                "a native Vulkan driver\n");
        return TILED_BACKEND_SKIP;
    }
    if (!wgpuAdapterHasFeature(adapter, YaWGPUFeatureName_MultiSubpass)) {
        fprintf(stderr, "tiled_deferred requires YaWGPUFeatureName_MultiSubpass\n");
        return TILED_BACKEND_FAIL;
    }
    return TILED_BACKEND_OK;
}

static YaWGPUSubpassPassLayout create_pass_layout(TiledDeferredApp *app) {
    YaWGPUAttachmentLayout color_layouts[4] = {
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
        {.format = WGPUTextureFormat_RGBA16Float, .sampleCount = 1},
        {.format = WGPUTextureFormat_RGBA16Float, .sampleCount = 1},
        {.format = app->output_format, .sampleCount = 1},
    };
    uint32_t subpass0_colors[2] = {0, 1};
    uint32_t subpass1_colors[1] = {2};
    uint32_t subpass2_colors[1] = {3};
    YaWGPUSubpassInputAttachment subpass1_inputs[2] = {
        {.group = 0, .binding = 0, .sourceSubpass = 0, .sourceAttachment = 0},
        {.group = 0, .binding = 1, .sourceSubpass = 0, .sourceAttachment = 1},
    };
    YaWGPUSubpassInputAttachment subpass2_inputs[1] = {
        {.group = 0, .binding = 0, .sourceSubpass = 1, .sourceAttachment = 2},
    };
    YaWGPUSubpassLayoutDesc subpasses[3] = {
        {.colorAttachmentIndices = subpass0_colors,
         .colorAttachmentIndexCount = 2,
         .usesDepthStencil = true,
         .inputAttachments = NULL,
         .inputAttachmentCount = 0},
        {.colorAttachmentIndices = subpass1_colors,
         .colorAttachmentIndexCount = 1,
         .usesDepthStencil = false,
         .inputAttachments = subpass1_inputs,
         .inputAttachmentCount = 2},
        {.colorAttachmentIndices = subpass2_colors,
         .colorAttachmentIndexCount = 1,
         .usesDepthStencil = false,
         .inputAttachments = subpass2_inputs,
         .inputAttachmentCount = 1},
    };
    YaWGPUSubpassDependency dependencies[2] = {
        {.srcSubpass = 0,
         .dstSubpass = 1,
         .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
         .byRegion = true},
        {.srcSubpass = 1,
         .dstSubpass = 2,
         .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
         .byRegion = true},
    };
    YaWGPUSubpassPassLayoutDescriptor descriptor = {
        .label = yawgpu_string_view("tiled deferred pass layout"),
        .colorAttachments = color_layouts,
        .colorAttachmentCount = 4,
        .depthStencilAttachment = {.format = WGPUTextureFormat_Depth32Float, .sampleCount = 1},
        .subpasses = subpasses,
        .subpassCount = 3,
        .dependencies = dependencies,
        .dependencyCount = 2,
    };
    return yawgpuDeviceCreateSubpassPassLayout(app->context.device, &descriptor);
}

static bool create_buffers(TiledDeferredApp *app) {
    Vertex vertices[VERTEX_COUNT];
    uint16_t indices[INDEX_COUNT];
    Vertex draw_vertices[INDEX_COUNT];
    create_cube_vertices(vertices, indices);
    for (uint32_t i = 0; i < INDEX_COUNT; ++i) {
        draw_vertices[i] = vertices[indices[i]];
    }
    app->vertex_buffer = yawgpu_create_buffer_init(
        app->context.device,
        &(YawgpuBufferInitDescriptor){.label = "tiled deferred vertices",
                                      .usage = WGPUBufferUsage_Vertex,
                                      .contents = draw_vertices,
                                      .size = sizeof(draw_vertices)});
    app->index_buffer = yawgpu_create_buffer_init(
        app->context.device,
        &(YawgpuBufferInitDescriptor){.label = "tiled deferred indices",
                                      .usage = WGPUBufferUsage_Index,
                                      .contents = indices,
                                      .size = sizeof(indices)});
    Uniforms uniforms = {0};
    LightParams lights = {0};
    app->uniform_buffer = yawgpu_create_buffer_init(
        app->context.device,
        &(YawgpuBufferInitDescriptor){.label = "tiled deferred uniforms",
                                      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                      .contents = &uniforms,
                                      .size = sizeof(uniforms)});
    app->light_buffer = yawgpu_create_buffer_init(
        app->context.device,
        &(YawgpuBufferInitDescriptor){.label = "tiled deferred lights",
                                      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                      .contents = &lights,
                                      .size = sizeof(lights)});
    if (!app->vertex_buffer || !app->index_buffer || !app->uniform_buffer ||
        !app->light_buffer) {
        fprintf(stderr, "failed to create buffers\n");
        return false;
    }
    if (app->verify) {
        app->dimensions = buffer_dimensions_create(app->width, app->height);
        app->output_buffer_size =
            (uint64_t)app->dimensions.padded_bytes_per_row * app->dimensions.height;
        WGPUBufferDescriptor descriptor = {
            .label = yawgpu_string_view("tiled deferred readback"),
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            .size = app->output_buffer_size,
        };
        app->output_buffer = wgpuDeviceCreateBuffer(app->context.device, &descriptor);
        if (!app->output_buffer) {
            fprintf(stderr, "failed to create readback buffer\n");
            return false;
        }
    }
    return true;
}

static WGPUBindGroupLayout create_uniform_bgl(TiledDeferredApp *app, const char *label) {
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Vertex;
    entry.buffer.type = WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = sizeof(Uniforms);
    WGPUBindGroupLayoutDescriptor descriptor = {
        .label = yawgpu_string_view(label),
        .entryCount = 1,
        .entries = &entry,
    };
    return wgpuDeviceCreateBindGroupLayout(app->context.device, &descriptor);
}

static WGPUBindGroupLayout create_lighting_bgl(TiledDeferredApp *app) {
    YaWGPUInputAttachmentBindingLayout input0 = {
        .chain = {.next = NULL, .sType = YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT},
        .sampleType = WGPUTextureSampleType_Float,
        .multisampled = false,
    };
    YaWGPUInputAttachmentBindingLayout input1 = input0;
    WGPUBindGroupLayoutEntry entries[2] = {
        WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
        WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
    };
    entries[0].nextInChain = &input0.chain;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[1].nextInChain = &input1.chain;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    WGPUBindGroupLayoutDescriptor descriptor = {
        .label = yawgpu_string_view("tiled deferred lighting input bgl"),
        .entryCount = 2,
        .entries = entries,
    };
    return wgpuDeviceCreateBindGroupLayout(app->context.device, &descriptor);
}

static WGPUBindGroupLayout create_lighting_uniform_bgl(TiledDeferredApp *app) {
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Fragment;
    entry.buffer.type = WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = sizeof(LightParams);
    WGPUBindGroupLayoutDescriptor descriptor = {
        .label = yawgpu_string_view("tiled deferred lighting uniform bgl"),
        .entryCount = 1,
        .entries = &entry,
    };
    return wgpuDeviceCreateBindGroupLayout(app->context.device, &descriptor);
}

static WGPUBindGroupLayout create_composite_bgl(TiledDeferredApp *app) {
    YaWGPUInputAttachmentBindingLayout input = {
        .chain = {.next = NULL, .sType = YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT},
        .sampleType = WGPUTextureSampleType_Float,
        .multisampled = false,
    };
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.nextInChain = &input.chain;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Fragment;
    WGPUBindGroupLayoutDescriptor descriptor = {
        .label = yawgpu_string_view("tiled deferred composite bgl"),
        .entryCount = 1,
        .entries = &entry,
    };
    return wgpuDeviceCreateBindGroupLayout(app->context.device, &descriptor);
}

static WGPUPipelineLayout create_pipeline_layout(TiledDeferredApp *app,
                                                 const char *label,
                                                 WGPUBindGroupLayout bgl) {
    WGPUPipelineLayoutDescriptor descriptor = {
        .label = yawgpu_string_view(label),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgl,
    };
    return wgpuDeviceCreatePipelineLayout(app->context.device, &descriptor);
}

static WGPUPipelineLayout create_pipeline_layout2(TiledDeferredApp *app,
                                                  const char *label,
                                                  WGPUBindGroupLayout bgl0,
                                                  WGPUBindGroupLayout bgl1) {
    WGPUBindGroupLayout layouts[2] = {bgl0, bgl1};
    WGPUPipelineLayoutDescriptor descriptor = {
        .label = yawgpu_string_view(label),
        .bindGroupLayoutCount = 2,
        .bindGroupLayouts = layouts,
    };
    return wgpuDeviceCreatePipelineLayout(app->context.device, &descriptor);
}

static bool create_bind_groups(TiledDeferredApp *app) {
    WGPUBindGroupEntry uniform_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    uniform_entry.binding = 0;
    uniform_entry.buffer = app->uniform_buffer;
    uniform_entry.size = sizeof(Uniforms);
    WGPUBindGroupDescriptor uniform_descriptor = {
        .label = yawgpu_string_view("tiled deferred uniform bind group"),
        .layout = app->gbuffer_bgl,
        .entryCount = 1,
        .entries = &uniform_entry,
    };
    app->uniform_bind_group =
        wgpuDeviceCreateBindGroup(app->context.device, &uniform_descriptor);

    WGPUBindGroupEntry light_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    light_entry.binding = 0;
    light_entry.buffer = app->light_buffer;
    light_entry.size = sizeof(LightParams);
    WGPUBindGroupDescriptor lighting_descriptor = {
        .label = yawgpu_string_view("tiled deferred lighting bind group"),
        .layout = app->lighting_uniform_bgl,
        .entryCount = 1,
        .entries = &light_entry,
    };
    app->lighting_bind_group =
        wgpuDeviceCreateBindGroup(app->context.device, &lighting_descriptor);
    return app->uniform_bind_group && app->lighting_bind_group;
}

static WGPURenderPipeline create_pipeline(TiledDeferredApp *app,
                                          const char *label,
                                          WGPUShaderModule module,
                                          const char *fragment_entry,
                                          WGPUPipelineLayout layout,
                                          uint32_t subpass,
                                          WGPUColorTargetState *targets,
                                          size_t target_count,
                                          const WGPUVertexBufferLayout *vertex_layout,
                                          WGPUDepthStencilState *depth_stencil) {
    WGPUFragmentState fragment = {
        .module = module,
        .entryPoint = yawgpu_string_view(fragment_entry),
        .targetCount = target_count,
        .targets = targets,
    };
    WGPURenderPipelineDescriptor base = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    base.label = yawgpu_string_view(label);
    base.layout = layout;
    base.vertex.module = module;
    base.vertex.entryPoint = yawgpu_string_view("vs_main");
    base.vertex.bufferCount = vertex_layout ? 1 : 0;
    base.vertex.buffers = vertex_layout;
    base.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    base.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    base.primitive.frontFace = WGPUFrontFace_CCW;
    base.primitive.cullMode = vertex_layout ? WGPUCullMode_Back : WGPUCullMode_None;
    base.depthStencil = depth_stencil;
    base.multisample.count = 1;
    base.multisample.mask = 0xFFFFFFFFu;
    base.fragment = &fragment;
    YaWGPUSubpassRenderPipelineDescriptor descriptor = {
        .base = base,
        .passLayout = app->pass_layout,
        .subpassIndex = subpass,
    };
    return yawgpuDeviceCreateSubpassRenderPipeline(app->context.device, &descriptor);
}

static bool create_pipelines(TiledDeferredApp *app) {
    const char *lighting_fs = "fs";
    const char *composite_fs = "fs";
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(app->context.adapter, &info) == WGPUStatus_Success) {
        if (info.backendType == WGPUBackendType_Metal) {
            lighting_fs = "fs_metal";
            composite_fs = "fs_metal";
        }
        wgpuAdapterInfoFreeMembers(info);
    }

    app->gbuffer_bgl = create_uniform_bgl(app, "tiled deferred gbuffer bgl");
    app->lighting_input_bgl = create_lighting_bgl(app);
    app->lighting_uniform_bgl = create_lighting_uniform_bgl(app);
    app->composite_bgl = create_composite_bgl(app);
    app->gbuffer_layout =
        create_pipeline_layout(app, "tiled deferred gbuffer layout", app->gbuffer_bgl);
    app->lighting_layout =
        create_pipeline_layout2(app,
                                "tiled deferred lighting layout",
                                app->lighting_input_bgl,
                                app->lighting_uniform_bgl);
    app->composite_layout =
        create_pipeline_layout(app, "tiled deferred composite layout", app->composite_bgl);
    if (!app->gbuffer_bgl || !app->lighting_input_bgl || !app->lighting_uniform_bgl ||
        !app->composite_bgl ||
        !app->gbuffer_layout || !app->lighting_layout || !app->composite_layout) {
        fprintf(stderr, "failed to create bind group or pipeline layouts\n");
        return false;
    }
    if (!create_bind_groups(app)) {
        fprintf(stderr, "failed to create bind groups\n");
        return false;
    }

    WGPUVertexAttribute attributes[3] = {
        {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
        {.format = WGPUVertexFormat_Float32x3, .offset = 3 * sizeof(float), .shaderLocation = 1},
        {.format = WGPUVertexFormat_Float32x3, .offset = 6 * sizeof(float), .shaderLocation = 2},
    };
    WGPUVertexBufferLayout vertex_layout = {
        .arrayStride = sizeof(Vertex),
        .stepMode = WGPUVertexStepMode_Vertex,
        .attributeCount = 3,
        .attributes = attributes,
    };
    WGPUDepthStencilState depth_stencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    depth_stencil.format = WGPUTextureFormat_Depth32Float;
    depth_stencil.depthWriteEnabled = WGPUOptionalBool_True;
    depth_stencil.depthCompare = WGPUCompareFunction_Less;

    WGPUColorTargetState gbuffer_targets[2] = {
        {.format = WGPUTextureFormat_RGBA8Unorm, .writeMask = WGPUColorWriteMask_All},
        {.format = WGPUTextureFormat_RGBA16Float, .writeMask = WGPUColorWriteMask_All},
    };
    WGPUColorTargetState lighting_target = {
        .format = WGPUTextureFormat_RGBA16Float,
        .writeMask = WGPUColorWriteMask_All,
    };
    WGPUColorTargetState composite_target = {
        .format = app->output_format,
        .writeMask = WGPUColorWriteMask_All,
    };
    app->gbuffer_pipeline = create_pipeline(app,
                                            "tiled deferred gbuffer pipeline",
                                            app->gbuffer_module,
                                            "fs_main",
                                            app->gbuffer_layout,
                                            0,
                                            gbuffer_targets,
                                            2,
                                            &vertex_layout,
                                            &depth_stencil);
    app->lighting_pipeline = create_pipeline(app,
                                             "tiled deferred lighting pipeline",
                                             app->lighting_module,
                                             lighting_fs,
                                             app->lighting_layout,
                                             1,
                                             &lighting_target,
                                             1,
                                             NULL,
                                             NULL);
    app->composite_pipeline = create_pipeline(app,
                                              "tiled deferred composite pipeline",
                                              app->composite_module,
                                              composite_fs,
                                              app->composite_layout,
                                              2,
                                              &composite_target,
                                              1,
                                              NULL,
                                              NULL);
    return app->gbuffer_pipeline && app->lighting_pipeline && app->composite_pipeline;
}

static WGPUTexture create_attachment_texture(TiledDeferredApp *app,
                                             const char *label,
                                             WGPUTextureFormat format,
                                             WGPUTextureUsage usage) {
    WGPUTextureDescriptor descriptor = {
        .label = yawgpu_string_view(label),
        .usage = usage,
        .dimension = WGPUTextureDimension_2D,
        .size = {.width = app->width, .height = app->height, .depthOrArrayLayers = 1},
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    return wgpuDeviceCreateTexture(app->context.device, &descriptor);
}

static bool create_attachments(TiledDeferredApp *app,
                               WGPUTextureView output_view,
                               AttachmentResources *attachments) {
    *attachments = (AttachmentResources){0};
    attachments->albedo_texture = create_attachment_texture(app,
                                                            "tiled deferred albedo",
                                                            WGPUTextureFormat_RGBA8Unorm,
                                                            WGPUTextureUsage_RenderAttachment);
    attachments->normal_texture = create_attachment_texture(app,
                                                            "tiled deferred normal",
                                                            WGPUTextureFormat_RGBA16Float,
                                                            WGPUTextureUsage_RenderAttachment);
    attachments->depth_texture = create_attachment_texture(app,
                                                           "tiled deferred depth",
                                                           WGPUTextureFormat_Depth32Float,
                                                           WGPUTextureUsage_RenderAttachment);
    attachments->lit_texture = create_attachment_texture(app,
                                                         "tiled deferred lit",
                                                         WGPUTextureFormat_RGBA16Float,
                                                         WGPUTextureUsage_RenderAttachment);
    if (app->verify) {
        attachments->output_texture =
            create_attachment_texture(app,
                                      "tiled deferred output",
                                      app->output_format,
                                      WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
        attachments->output_view = wgpuTextureCreateView(attachments->output_texture, NULL);
    } else {
        attachments->output_view = output_view;
    }
    if (!attachments->albedo_texture || !attachments->normal_texture ||
        !attachments->depth_texture || !attachments->lit_texture || !attachments->output_view) {
        fprintf(stderr, "failed to create attachment textures\n");
        return false;
    }
    attachments->albedo_view = wgpuTextureCreateView(attachments->albedo_texture, NULL);
    attachments->normal_view = wgpuTextureCreateView(attachments->normal_texture, NULL);
    attachments->depth_view = wgpuTextureCreateView(attachments->depth_texture, NULL);
    attachments->lit_view = wgpuTextureCreateView(attachments->lit_texture, NULL);
    if (!attachments->albedo_view || !attachments->normal_view || !attachments->depth_view ||
        !attachments->lit_view) {
        fprintf(stderr, "failed to create attachment views\n");
        return false;
    }
    return true;
}

static void destroy_attachments(AttachmentResources *attachments, bool release_output_view) {
    if (release_output_view && attachments->output_view) {
        wgpuTextureViewRelease(attachments->output_view);
    }
    if (attachments->lit_view) wgpuTextureViewRelease(attachments->lit_view);
    if (attachments->depth_view) wgpuTextureViewRelease(attachments->depth_view);
    if (attachments->normal_view) wgpuTextureViewRelease(attachments->normal_view);
    if (attachments->albedo_view) wgpuTextureViewRelease(attachments->albedo_view);
    if (attachments->output_texture) wgpuTextureRelease(attachments->output_texture);
    if (attachments->lit_texture) wgpuTextureRelease(attachments->lit_texture);
    if (attachments->depth_texture) wgpuTextureRelease(attachments->depth_texture);
    if (attachments->normal_texture) wgpuTextureRelease(attachments->normal_texture);
    if (attachments->albedo_texture) wgpuTextureRelease(attachments->albedo_texture);
    *attachments = (AttachmentResources){0};
}

static void write_frame_uniforms(TiledDeferredApp *app, float time_seconds) {
    Vec3 eye = vec3_make(12.0f * cosf(time_seconds * 0.3f),
                         8.0f,
                         12.0f * sinf(time_seconds * 0.3f) + 15.0f);
    Mat4 view = mat4_look_at_rh(eye, vec3_make(0.0f, 0.0f, 0.0f), vec3_make(0.0f, 1.0f, 0.0f));
    Mat4 projection =
        mat4_perspective_rh(45.0f * 3.14159265358979323846f / 180.0f,
                            (float)app->width / (float)app->height,
                            0.1f,
                            100.0f);
    Mat4 view_proj = mat4_mul(projection, view);
    Mat4 inv_view_proj = mat4_identity();
    mat4_inverse(view_proj, &inv_view_proj);

    Uniforms uniforms = {0};
    memcpy(uniforms.view_proj, view_proj.m, sizeof(uniforms.view_proj));
    wgpuQueueWriteBuffer(app->queue, app->uniform_buffer, 0, &uniforms, sizeof(uniforms));

    LightParams lights = {0};
    lights.lights[0][0] = 10.0f * cosf(time_seconds * 0.7f);
    lights.lights[0][1] = 8.0f;
    lights.lights[0][2] = 10.0f * sinf(time_seconds * 0.7f);
    lights.lights[0][3] = 50.0f;
    lights.lights[1][0] = -8.0f * cosf(time_seconds * 0.5f);
    lights.lights[1][1] = 6.0f;
    lights.lights[1][2] = -8.0f * sinf(time_seconds * 0.5f);
    lights.lights[1][3] = 40.0f;
    lights.lights[2][0] = 6.0f * sinf(time_seconds * 1.1f);
    lights.lights[2][1] = 4.0f;
    lights.lights[2][2] = 6.0f * cosf(time_seconds * 1.1f);
    lights.lights[2][3] = 35.0f;
    lights.lights[3][0] = -5.0f;
    lights.lights[3][1] = 10.0f + 3.0f * sinf(time_seconds * 0.3f);
    lights.lights[3][2] = 5.0f;
    lights.lights[3][3] = 45.0f;
    lights.camera_pos[0] = eye.x;
    lights.camera_pos[1] = eye.y;
    lights.camera_pos[2] = eye.z;
    lights.time = time_seconds;
    memcpy(lights.inv_view_proj, inv_view_proj.m, sizeof(lights.inv_view_proj));
    lights.screen_size[0] = (float)app->width;
    lights.screen_size[1] = (float)app->height;
    wgpuQueueWriteBuffer(app->queue, app->light_buffer, 0, &lights, sizeof(lights));
}

static bool record_tiled_pass(TiledDeferredApp *app,
                              WGPUCommandEncoder encoder,
                              const AttachmentResources *attachments) {
    YaWGPUColorAttachmentBinding colors[4] = {
        {.kind = YaWGPUSubpassAttachmentKind_Persistent,
         .view = attachments->albedo_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0.0, 0.0, 0.0, 0.0}},
        {.kind = YaWGPUSubpassAttachmentKind_Persistent,
         .view = attachments->normal_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0.0, 0.0, 0.0, 0.0}},
        {.kind = YaWGPUSubpassAttachmentKind_Persistent,
         .view = attachments->lit_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0.0, 0.0, 0.0, 0.0}},
        {.kind = YaWGPUSubpassAttachmentKind_Persistent,
         .view = attachments->output_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0.0, 0.0, 0.0, 1.0}},
    };
    YaWGPUDepthStencilAttachmentBinding depth = {
        .kind = YaWGPUSubpassAttachmentKind_Persistent,
        .view = attachments->depth_view,
        .depthLoadOp = WGPULoadOp_Clear,
        .depthStoreOp = WGPUStoreOp_Discard,
        .depthClearValue = 1.0f,
        .stencilLoadOp = WGPULoadOp_Clear,
        .stencilStoreOp = WGPUStoreOp_Discard,
        .stencilClearValue = 0,
    };
    YaWGPUSubpassRenderPassDescriptor descriptor = {
        .label = yawgpu_string_view("tiled deferred subpass render pass"),
        .passLayout = app->pass_layout,
        .extent = {.width = app->width, .height = app->height, .depthOrArrayLayers = 1},
        .colorAttachments = colors,
        .colorAttachmentCount = 4,
        .depthStencilAttachment = &depth,
    };
    YaWGPUSubpassRenderPassEncoder pass =
        yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
    if (!pass) {
        fprintf(stderr, "failed to begin subpass render pass\n");
        return false;
    }
    yawgpuSubpassRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, (float)app->width,
                                              (float)app->height, 0.0f, 1.0f);
    yawgpuSubpassRenderPassEncoderSetScissorRect(pass, 0, 0, app->width, app->height);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, app->gbuffer_pipeline);
    yawgpuSubpassRenderPassEncoderSetBindGroup(pass, 0, app->uniform_bind_group, 0, NULL);
    yawgpuSubpassRenderPassEncoderSetVertexBuffer(pass,
                                                  0,
                                                  app->vertex_buffer,
                                                  0,
                                                  sizeof(Vertex) * INDEX_COUNT);
    yawgpuSubpassRenderPassEncoderDraw(pass, INDEX_COUNT, INSTANCE_COUNT, 0, 0);

    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, app->lighting_pipeline);
    yawgpuSubpassRenderPassEncoderSetBindGroup(pass, 1, app->lighting_bind_group, 0, NULL);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, app->composite_pipeline);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpuSubpassRenderPassEncoderRelease(pass);
    return true;
}

static bool wait_for_queue(TiledDeferredApp *app) {
    QueueWorkDoneState queue_state = {0};
    WGPUFuture queue_future = wgpuQueueOnSubmittedWorkDone(
        app->queue,
        (WGPUQueueWorkDoneCallbackInfo){.mode = WGPUCallbackMode_AllowProcessEvents,
                                        .callback = queue_work_done_callback,
                                        .userdata1 = &queue_state});
    yawgpu_wait_for_future(app->context.instance, queue_future);
    if (!queue_state.called || queue_state.status != WGPUQueueWorkDoneStatus_Success) {
        fprintf(stderr, "submitted work did not complete successfully\n");
        return false;
    }
    return true;
}

static bool submit_encoder(TiledDeferredApp *app, WGPUCommandEncoder encoder, bool wait) {
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){.label = yawgpu_string_view("tiled deferred commands")});
    if (!commands) {
        fprintf(stderr, "failed to finish command encoder\n");
        return false;
    }
    wgpuQueueSubmit(app->queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    return !wait || wait_for_queue(app);
}

static bool verify_center_pixel(const TiledDeferredApp *app, const uint8_t *pixels) {
    uint32_t x = app->dimensions.width / 2;
    uint32_t y = app->dimensions.height / 2;
    size_t offset = (size_t)y * app->dimensions.padded_bytes_per_row +
                    (size_t)x * BYTES_PER_PIXEL;
    uint8_t r = pixels[offset + 0];
    uint8_t g = pixels[offset + 1];
    uint8_t b = pixels[offset + 2];
    uint8_t a = pixels[offset + 3];
    unsigned int rgb_sum = (unsigned int)r + (unsigned int)g + (unsigned int)b;
    bool ok = a > 0 && rgb_sum > 12;
    printf("tiled_deferred: center pixel RGBA=(%u,%u,%u,%u), rgb_sum=%u %s\n",
           (unsigned int)r,
           (unsigned int)g,
           (unsigned int)b,
           (unsigned int)a,
           rgb_sum,
           ok ? "OK" : "FAILED");
    return ok;
}

static bool readback_verify_and_write_png(TiledDeferredApp *app) {
    MapState map_state = {0};
    WGPUFuture map_future = wgpuBufferMapAsync(
        app->output_buffer,
        WGPUMapMode_Read,
        0,
        app->output_buffer_size,
        (WGPUBufferMapCallbackInfo){.mode = WGPUCallbackMode_AllowProcessEvents,
                                    .callback = map_callback,
                                    .userdata1 = &map_state});
    yawgpu_wait_for_future(app->context.instance, map_future);
    if (!map_state.called || map_state.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "readback map did not complete successfully\n");
        return false;
    }
    app->output_buffer_mapped = true;
    const uint8_t *pixels =
        wgpuBufferGetConstMappedRange(app->output_buffer, 0, app->output_buffer_size);
    if (!pixels) {
        fprintf(stderr, "readback mapped range is null\n");
        return false;
    }
    bool pixel_ok = verify_center_pixel(app, pixels);
    bool wrote_png = stbi_write_png("tiled_deferred.png",
                                    (int)app->dimensions.width,
                                    (int)app->dimensions.height,
                                    BYTES_PER_PIXEL,
                                    pixels,
                                    (int)app->dimensions.padded_bytes_per_row) != 0;
    if (!wrote_png) {
        fprintf(stderr, "failed to write tiled_deferred.png\n");
        return false;
    }
    printf("wrote tiled_deferred.png (%ux%u, bytesPerRow=%u padded to %u)\n",
           app->dimensions.width,
           app->dimensions.height,
           app->dimensions.unpadded_bytes_per_row,
           app->dimensions.padded_bytes_per_row);
    return pixel_ok;
}

static bool render_verify(TiledDeferredApp *app) {
    write_frame_uniforms(app, 0.0f);
    AttachmentResources attachments = {0};
    if (!create_attachments(app, NULL, &attachments)) {
        destroy_attachments(&attachments, true);
        return false;
    }
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        app->context.device,
        &(WGPUCommandEncoderDescriptor){.label = yawgpu_string_view("tiled deferred encoder")});
    if (!encoder) {
        destroy_attachments(&attachments, true);
        return false;
    }
    bool ok = record_tiled_pass(app, encoder, &attachments);
    if (ok) {
        // Pass + texture-to-buffer copy in a single encoder so they submit as
        // one ordered command buffer.
        WGPUExtent3D copy_size = {.width = app->width, .height = app->height, .depthOrArrayLayers = 1};
        wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &(WGPUTexelCopyTextureInfo){.texture = attachments.output_texture,
                                        .mipLevel = 0,
                                        .origin = {0, 0, 0},
                                        .aspect = WGPUTextureAspect_All},
            &(WGPUTexelCopyBufferInfo){.buffer = app->output_buffer,
                                       .layout = {.offset = 0,
                                                  .bytesPerRow = app->dimensions.padded_bytes_per_row,
                                                  .rowsPerImage = app->dimensions.height}},
            &copy_size);
        ok = submit_encoder(app, encoder, true);
    }
    wgpuCommandEncoderRelease(encoder);
    destroy_attachments(&attachments, true);
    if (!ok) {
        return false;
    }
    if (yawgpu_uncaptured_error_count() != app->initial_error_count) {
        fprintf(stderr, "tiled_deferred: FAILED due to uncaptured device error\n");
        return false;
    }
    if (!readback_verify_and_write_png(app)) {
        return false;
    }
    return true;
}

static bool render_window_frame(TiledDeferredApp *app) {
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
    WGPUTextureView output_view = wgpuTextureCreateView(current.texture, NULL);
    AttachmentResources attachments = {0};
    bool ok = output_view && create_attachments(app, output_view, &attachments);
    WGPUCommandEncoder encoder = NULL;
    if (ok) {
        float seconds = (float)(clock() - app->start_clock) / (float)CLOCKS_PER_SEC;
        write_frame_uniforms(app, seconds);
        encoder = wgpuDeviceCreateCommandEncoder(app->context.device, NULL);
        ok = encoder && record_tiled_pass(app, encoder, &attachments) &&
             submit_encoder(app, encoder, false);
    }
    if (encoder) {
        wgpuCommandEncoderRelease(encoder);
    }
    if (ok && wgpuSurfacePresent(app->surface) != WGPUStatus_Success) {
        fprintf(stderr, "surface present failed\n");
        ok = false;
    }
    destroy_attachments(&attachments, false);
    if (output_view) {
        wgpuTextureViewRelease(output_view);
    }
    wgpuTextureRelease(current.texture);
    if (yawgpu_uncaptured_error_count() != app->initial_error_count) {
        fprintf(stderr, "tiled_deferred: FAILED due to uncaptured device error\n");
        ok = false;
    }
    return ok;
}

static void tiled_deferred_app_destroy(TiledDeferredApp *app) {
    if (app->output_buffer_mapped) wgpuBufferUnmap(app->output_buffer);
    if (app->composite_pipeline) wgpuRenderPipelineRelease(app->composite_pipeline);
    if (app->lighting_pipeline) wgpuRenderPipelineRelease(app->lighting_pipeline);
    if (app->gbuffer_pipeline) wgpuRenderPipelineRelease(app->gbuffer_pipeline);
    if (app->lighting_bind_group) wgpuBindGroupRelease(app->lighting_bind_group);
    if (app->uniform_bind_group) wgpuBindGroupRelease(app->uniform_bind_group);
    if (app->composite_layout) wgpuPipelineLayoutRelease(app->composite_layout);
    if (app->lighting_layout) wgpuPipelineLayoutRelease(app->lighting_layout);
    if (app->gbuffer_layout) wgpuPipelineLayoutRelease(app->gbuffer_layout);
    if (app->composite_bgl) wgpuBindGroupLayoutRelease(app->composite_bgl);
    if (app->lighting_uniform_bgl) wgpuBindGroupLayoutRelease(app->lighting_uniform_bgl);
    if (app->lighting_input_bgl) wgpuBindGroupLayoutRelease(app->lighting_input_bgl);
    if (app->gbuffer_bgl) wgpuBindGroupLayoutRelease(app->gbuffer_bgl);
    if (app->pass_layout) yawgpuSubpassPassLayoutRelease(app->pass_layout);
    if (app->composite_module) wgpuShaderModuleRelease(app->composite_module);
    if (app->lighting_module) wgpuShaderModuleRelease(app->lighting_module);
    if (app->gbuffer_module) wgpuShaderModuleRelease(app->gbuffer_module);
    if (app->output_buffer) wgpuBufferRelease(app->output_buffer);
    if (app->light_buffer) wgpuBufferRelease(app->light_buffer);
    if (app->uniform_buffer) wgpuBufferRelease(app->uniform_buffer);
    if (app->index_buffer) wgpuBufferRelease(app->index_buffer);
    if (app->vertex_buffer) wgpuBufferRelease(app->vertex_buffer);
    if (app->surface) {
        wgpuSurfaceUnconfigure(app->surface);
        wgpuSurfaceRelease(app->surface);
    }
    if (app->window) yawgpu_window_destroy(app->window);
    if (app->queue) wgpuQueueRelease(app->queue);
    yawgpu_context_release(&app->context);
    *app = (TiledDeferredApp){0};
}

static TiledBackendStatus tiled_deferred_app_init(TiledDeferredApp *app,
                                                  bool verify,
                                                  const char *argv0) {
    *app = (TiledDeferredApp){0};
    app->verify = verify;
    app->width = WINDOW_WIDTH;
    app->height = WINDOW_HEIGHT;
    app->output_format = verify ? WGPUTextureFormat_RGBA8Unorm : WGPUTextureFormat_Undefined;
    set_shader_prefix(app, argv0);
    app->context = yawgpu_context_create();
    if (!app->context.instance || !app->context.adapter || !app->context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return TILED_BACKEND_FAIL;
    }
    TiledBackendStatus status = require_tiled_backend(app->context.adapter);
    if (status != TILED_BACKEND_OK) {
        return status;
    }
    app->queue = wgpuDeviceGetQueue(app->context.device);
    if (!app->queue) {
        fprintf(stderr, "failed to get queue\n");
        return TILED_BACKEND_FAIL;
    }
    if (!verify) {
        app->window = yawgpu_window_create(WINDOW_WIDTH, WINDOW_HEIGHT, "yawgpu tiled deferred");
        if (!app->window) {
            fprintf(stderr, "failed to create window\n");
            return TILED_BACKEND_FAIL;
        }
        app->surface = yawgpu_window_create_surface(app->context.instance,
                                                    app->window,
                                                    "tiled deferred surface");
        if (!app->surface ||
            !choose_surface_format(app->surface, app->context.adapter, &app->output_format)) {
            return TILED_BACKEND_FAIL;
        }
        int width = 0;
        int height = 0;
        yawgpu_window_framebuffer_size(app->window, &width, &height);
        if (width <= 0 || height <= 0) {
            fprintf(stderr, "invalid framebuffer size\n");
            return TILED_BACKEND_FAIL;
        }
        app->width = (uint32_t)width;
        app->height = (uint32_t)height;
        wgpuSurfaceConfigure(app->surface,
                             &(WGPUSurfaceConfiguration){.device = app->context.device,
                                                         .format = app->output_format,
                                                         .usage =
                                                             WGPUTextureUsage_RenderAttachment,
                                                         .width = app->width,
                                                         .height = app->height,
                                                         .alphaMode =
                                                             WGPUCompositeAlphaMode_Opaque,
                                                         .presentMode = WGPUPresentMode_Fifo});
    }
    app->pass_layout = create_pass_layout(app);
    app->gbuffer_module = load_shader(app, "gbuffer.wgsl");
    app->lighting_module = load_shader(app, "lighting.wgsl");
    app->composite_module = load_shader(app, "composite.wgsl");
    if (!app->pass_layout || !app->gbuffer_module || !app->lighting_module ||
        !app->composite_module || !create_buffers(app) || !create_pipelines(app)) {
        return TILED_BACKEND_FAIL;
    }
    app->initial_error_count = yawgpu_uncaptured_error_count();
    app->start_clock = clock();
    return TILED_BACKEND_OK;
}

int main(int argc, char **argv) {
    bool verify = argc >= 2 && strcmp(argv[1], "--verify") == 0;
    TiledDeferredApp app = {0};
    TiledBackendStatus status = tiled_deferred_app_init(&app, verify, argv[0]);
    if (status == TILED_BACKEND_SKIP) {
        tiled_deferred_app_destroy(&app);
        return EXIT_SUCCESS;
    }
    if (status != TILED_BACKEND_OK) {
        tiled_deferred_app_destroy(&app);
        return EXIT_FAILURE;
    }
    bool ok = false;
    if (verify) {
        ok = render_verify(&app);
    } else {
        ok = true;
        while (!yawgpu_window_should_close(app.window)) {
            if (!render_window_frame(&app)) {
                ok = false;
                break;
            }
            yawgpu_window_poll_events();
        }
    }
    tiled_deferred_app_destroy(&app);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif
