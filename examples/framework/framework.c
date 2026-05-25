// framework.c — implementation of the shared example helpers declared in
// framework.h. The interesting recurring pattern here is how WebGPU's
// asynchronous requests (adapter, device) are turned into simple blocking
// calls: register a callback that stores its result into a small state
// struct, then pump the instance event loop until the callback fires.

#include "framework.h"

// Captures the result of an async wgpuInstanceRequestAdapter callback.
typedef struct RequestAdapterState {
    WGPUAdapter adapter;
    WGPURequestAdapterStatus status;
} RequestAdapterState;

// Captures the result of an async wgpuAdapterRequestDevice callback.
typedef struct RequestDeviceState {
    WGPUDevice device;
    WGPURequestDeviceStatus status;
} RequestDeviceState;

static unsigned int g_uncaptured_error_count = 0;

static void print_callback_message(const char *prefix, WGPUStringView message) {
    if (!message.data || message.length == 0) {
        return;
    }
    fputs(prefix, stderr);
    if (message.length == (size_t)WGPU_STRLEN) {
        fputs(message.data, stderr);
    } else {
        fprintf(stderr, "%.*s", (int)message.length, message.data);
    }
    fputc('\n', stderr);
}

// Installed on the device so any validation/internal error that is not
// caught by an explicit error scope is printed instead of silently ignored.
static void uncaptured_error_callback(const WGPUDevice *device,
                                      WGPUErrorType type,
                                      WGPUStringView message,
                                      void *userdata1,
                                      void *userdata2) {
    YAWGPU_UNUSED(device);
    YAWGPU_UNUSED(userdata1);
    YAWGPU_UNUSED(userdata2);
    g_uncaptured_error_count += 1;
    fprintf(stderr, "[yawgpu] uncaptured error type=%u: ", type);
    yawgpu_print_string_view(message);
    fputc('\n', stderr);
}

static void request_adapter_callback(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     WGPUStringView message,
                                     void *userdata1,
                                     void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    RequestAdapterState *state = (RequestAdapterState *)userdata1;
    state->status = status;
    state->adapter = adapter;
    if (status != WGPURequestAdapterStatus_Success) {
        print_callback_message("[yawgpu] request adapter failed: ", message);
    }
}

static void request_device_callback(WGPURequestDeviceStatus status,
                                    WGPUDevice device,
                                    WGPUStringView message,
                                    void *userdata1,
                                    void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    RequestDeviceState *state = (RequestDeviceState *)userdata1;
    state->status = status;
    state->device = device;
    if (status != WGPURequestDeviceStatus_Success) {
        print_callback_message("[yawgpu] request device failed: ", message);
    }
}

WGPUStringView yawgpu_string_view(const char *value) {
    WGPUStringView view = {0};
    view.data = value;
    view.length = value ? (size_t)WGPU_STRLEN : 0;
    return view;
}

void yawgpu_print_string_view(WGPUStringView value) {
    if (!value.data || value.length == 0) {
        return;
    }
    if (value.length == (size_t)WGPU_STRLEN) {
        fputs(value.data, stdout);
    } else {
        printf("%.*s", (int)value.length, value.data);
    }
}

unsigned int yawgpu_uncaptured_error_count(void) {
    return g_uncaptured_error_count;
}

// Maps the YAWGPU_BACKEND environment variable to a yawgpu.h backend enum
// value, defaulting to Noop for unset/empty/unknown values.
static uint32_t backend_from_environment(void) {
    const char *backend = getenv("YAWGPU_BACKEND");
    if (!backend || strcmp(backend, "") == 0 || strcmp(backend, "noop") == 0) {
        return YAWGPU_INSTANCE_BACKEND_NOOP;
    }
    if (strcmp(backend, "metal") == 0) {
        return YAWGPU_INSTANCE_BACKEND_METAL;
    }
    if (strcmp(backend, "vulkan") == 0) {
        return YAWGPU_INSTANCE_BACKEND_VULKAN;
    }
    if (strcmp(backend, "gles") == 0) {
        return YAWGPU_INSTANCE_BACKEND_GLES;
    }
    fprintf(stderr, "unknown YAWGPU_BACKEND=%s, using noop\n", backend);
    return YAWGPU_INSTANCE_BACKEND_NOOP;
}

// Maps the YAWGPU_GLES_CONTEXT_BACKEND environment variable to a yawgpu.h
// GLES context-backend enum value, defaulting to DEFAULT (= defer to the
// library's own YAWGPU_GLES_BACKEND parser) for unset/empty/unknown values.
// Lets examples force EGL or WGL from the chain without polluting
// YAWGPU_GLES_BACKEND for the rest of the process.
static uint32_t gles_context_backend_from_environment(void) {
    const char *value = getenv("YAWGPU_GLES_CONTEXT_BACKEND");
    if (!value || strcmp(value, "") == 0 || strcmp(value, "default") == 0) {
        return YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT;
    }
    if (strcmp(value, "egl") == 0) {
        return YAWGPU_GLES_CONTEXT_BACKEND_EGL;
    }
    if (strcmp(value, "wgl") == 0) {
        return YAWGPU_GLES_CONTEXT_BACKEND_WGL;
    }
    fprintf(stderr,
            "unknown YAWGPU_GLES_CONTEXT_BACKEND=%s, deferring to library default\n",
            value);
    return YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT;
}

WGPUInstance yawgpu_instance_create(void) {
    // Optional second chain entry: only consumed when the resolved instance
    // backend is GLES, ignored otherwise. The library treats DEFAULT as
    // "fall through to YAWGPU_GLES_BACKEND", so leaving the env var unset
    // gives the existing behaviour byte-for-byte.
    YaWGPUGlesContextBackend gles_context = YAWGPU_GLES_CONTEXT_BACKEND_INIT;
    gles_context.contextBackend = gles_context_backend_from_environment();

    // Chain the vendor backend-select struct onto the instance descriptor
    // so the chosen backend is applied at creation time.
    YaWGPUInstanceBackendSelect backend = {
        .chain = {
            .next = &gles_context.chain,
            .sType = YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        .backend = backend_from_environment(),
    };
    WGPUInstanceDescriptor descriptor = {
        .nextInChain = &backend.chain,
    };
    return wgpuCreateInstance(&descriptor);
}

// Blocks until `future` is done by processing queued events and then waiting
// on it. This is how the examples turn async operations into synchronous code.
void yawgpu_wait_for_future(WGPUInstance instance, WGPUFuture future) {
    wgpuInstanceProcessEvents(instance);
    WGPUFutureWaitInfo wait_info = {
        .future = future,
        .completed = 0,
    };
    (void)wgpuInstanceWaitAny(instance, 1, &wait_info, 0);
}

WGPUAdapter yawgpu_request_adapter(WGPUInstance instance) {
    RequestAdapterState state = {0};
    WGPURequestAdapterCallbackInfo callback_info = {
        .nextInChain = NULL,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = request_adapter_callback,
        .userdata1 = &state,
        .userdata2 = NULL,
    };
    WGPUFuture future = wgpuInstanceRequestAdapter(instance, NULL, callback_info);
    yawgpu_wait_for_future(instance, future);
    return state.adapter;
}

WGPUDevice yawgpu_request_device(WGPUInstance instance, WGPUAdapter adapter) {
    RequestDeviceState state = {0};
    // Request a device with default limits/features, wiring up the
    // uncaptured-error callback so mistakes surface on stderr.
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.label = yawgpu_string_view("yawgpu example device");
    descriptor.uncapturedErrorCallbackInfo.callback = uncaptured_error_callback;
    WGPURequestDeviceCallbackInfo callback_info = {
        .nextInChain = NULL,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = request_device_callback,
        .userdata1 = &state,
        .userdata2 = NULL,
    };
    WGPUFuture future = wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    yawgpu_wait_for_future(instance, future);
    return state.device;
}

// Acquires instance → adapter → device in sequence; on any failure the
// remaining fields stay NULL and the caller releases what was obtained.
// Logs the selected backend so callers can tell at a glance when a
// requested backend (e.g. YAWGPU_BACKEND=gles) silently fell back to Noop
// because of a missing dynamic library (ANGLE on Windows, libEGL.so on
// Android/Linux) or an unavailable driver. WGPUBackendType numeric values
// come from webgpu.h.
static const char *backend_type_name(uint32_t backend_type) {
    switch (backend_type) {
        case 1: return "Null (Noop)";
        case 5: return "Metal";
        case 6: return "Vulkan";
        case 7: return "OpenGL";
        case 8: return "OpenGLES";
        default: return "Other";
    }
}

YawgpuContext yawgpu_context_create(void) {
    YawgpuContext context = {0};
    context.instance = yawgpu_instance_create();
    if (!context.instance) {
        return context;
    }
    context.adapter = yawgpu_request_adapter(context.instance);
    if (!context.adapter) {
        return context;
    }
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(context.adapter, &info) == WGPUStatus_Success) {
        const char *requested = getenv("YAWGPU_BACKEND");
        printf("yawgpu: backend=%s (requested YAWGPU_BACKEND=%s)\n",
               backend_type_name(info.backendType),
               requested && *requested ? requested : "<unset>");
        wgpuAdapterInfoFreeMembers(info);
    }
    context.device = yawgpu_request_device(context.instance, context.adapter);
    return context;
}

void yawgpu_context_release(YawgpuContext *context) {
    if (!context) {
        return;
    }
    if (context->device) {
        wgpuDeviceRelease(context->device);
    }
    if (context->adapter) {
        wgpuAdapterRelease(context->adapter);
    }
    if (context->instance) {
        wgpuInstanceRelease(context->instance);
    }
    context->device = NULL;
    context->adapter = NULL;
    context->instance = NULL;
}

// Directory part of argv[0] (with trailing separator), or "" when the
// caller hasn't called yawgpu_set_argv0 — in which case yawgpu_load_wgsl_shader
// falls back to the original cwd-relative behaviour.
static char g_binary_dir_prefix[1024] = "";

void yawgpu_set_argv0(const char *argv0) {
    g_binary_dir_prefix[0] = '\0';
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
    if (length >= sizeof(g_binary_dir_prefix)) {
        return;
    }
    memcpy(g_binary_dir_prefix, argv0, length);
    g_binary_dir_prefix[length] = '\0';
}

// Returns a malloc'd path: when the prefix is set and `path` is relative,
// `prefix + path`; otherwise a copy of `path`. NULL on alloc failure.
static char *resolve_shader_path(const char *path) {
    bool is_absolute = path[0] == '/' || path[0] == '\\' ||
                       (path[0] != '\0' && path[1] == ':');
    if (g_binary_dir_prefix[0] == '\0' || is_absolute) {
        size_t length = strlen(path);
        char *out = (char *)malloc(length + 1);
        if (out) {
            memcpy(out, path, length + 1);
        }
        return out;
    }
    size_t prefix_len = strlen(g_binary_dir_prefix);
    size_t path_len = strlen(path);
    char *out = (char *)malloc(prefix_len + path_len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, g_binary_dir_prefix, prefix_len);
    memcpy(out + prefix_len, path, path_len + 1);
    return out;
}

// Reads an entire file into a newly-allocated, NUL-terminated buffer.
// Returns NULL (after printing the reason) on any I/O failure; the caller
// frees the result.
static char *read_file(const char *path, size_t *length_out) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0) {
        perror("ftell");
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        perror("fseek");
        fclose(file);
        return NULL;
    }
    char *contents = (char *)malloc((size_t)length + 1);
    if (!contents) {
        fclose(file);
        return NULL;
    }
    size_t read = fread(contents, 1, (size_t)length, file);
    fclose(file);
    if (read != (size_t)length) {
        free(contents);
        return NULL;
    }
    contents[length] = '\0';
    if (length_out) {
        *length_out = (size_t)length;
    }
    return contents;
}

// Loads a WGSL file and compiles it. The WGSL source is supplied through a
// WGPUShaderSourceWGSL struct chained onto the shader-module descriptor —
// the standard way to pass shader text across the C ABI.
//
// Path resolution: absolute paths are used as-is. Relative paths are
// resolved against the directory prefix set by yawgpu_set_argv0() if any;
// otherwise they remain cwd-relative (matches pre-existing behaviour for
// callers that don't forward argv[0]). CMake stages each example's WGSL
// next to its binary, so the prefix lets binaries run from any cwd.
WGPUShaderModule yawgpu_load_wgsl_shader(WGPUDevice device, const char *path) {
    char *resolved = resolve_shader_path(path);
    if (!resolved) {
        return NULL;
    }
    size_t length = 0;
    char *source = read_file(resolved, &length);
    free(resolved);
    if (!source) {
        return NULL;
    }
    WGPUShaderSourceWGSL wgsl = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = {
            .data = source,
            .length = length,
        },
    };
    WGPUShaderModuleDescriptor descriptor = {
        .nextInChain = &wgsl.chain,
        .label = yawgpu_string_view(path),
    };
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &descriptor);
    free(source);
    return module;
}

// Creates a buffer and uploads `contents` into it. Uses the
// `mappedAtCreation` shortcut: a buffer created mapped can be written
// immediately via GetMappedRange + memcpy, with no queue or copy involved;
// Unmap then hands it to the GPU.
WGPUBuffer yawgpu_create_buffer_init(WGPUDevice device,
                                     const YawgpuBufferInitDescriptor *descriptor) {
    if (!descriptor) {
        return NULL;
    }
    // Buffers must be non-empty; use a 4-byte minimum for a zero-size request.
    size_t size = descriptor->size == 0 ? 4 : descriptor->size;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
        .nextInChain = NULL,
        .label = yawgpu_string_view(descriptor->label),
        .usage = descriptor->usage,
        .size = size,
        .mappedAtCreation = descriptor->size > 0,
    });
    if (buffer && descriptor->contents && descriptor->size > 0) {
        void *mapped = wgpuBufferGetMappedRange(buffer, 0, descriptor->size);
        if (mapped) {
            memcpy(mapped, descriptor->contents, descriptor->size);
        }
        wgpuBufferUnmap(buffer);
    }
    return buffer;
}

void yawgpu_print_adapter_info(WGPUAdapter adapter) {
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
        printf("Adapter info unavailable\n");
        return;
    }
    printf("AdapterInfo\n");
    printf("  vendor: ");
    yawgpu_print_string_view(info.vendor);
    printf("\n  architecture: ");
    yawgpu_print_string_view(info.architecture);
    printf("\n  device: ");
    yawgpu_print_string_view(info.device);
    printf("\n  description: ");
    yawgpu_print_string_view(info.description);
    printf("\n  backendType: %u\n", info.backendType);
    printf("  adapterType: %u\n", info.adapterType);
    printf("  vendorID: %u\n", info.vendorID);
    printf("  deviceID: %u\n", info.deviceID);
    wgpuAdapterInfoFreeMembers(info);
}
