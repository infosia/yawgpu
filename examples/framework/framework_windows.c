// framework_windows.c — Windows implementation of the windowing helpers.
//
// Rather than depend on GLFW, the Windows examples talk to the Win32 API
// directly: register a window class, create an HWND, run a minimal message
// loop, and hand the HWND + HINSTANCE to yawgpu as the native surface source
// (which the Vulkan backend turns into a VkSurfaceKHR via VK_KHR_win32_surface).

#include "framework.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Concrete definition of the opaque YawgpuWindow from framework.h.
struct YawgpuWindow {
    HWND hwnd;
    HINSTANCE hinstance;
    bool should_close; // set when the user closes the window
};

// A window class must be registered once before any window of that class is
// created. We register lazily on first use and reference-count live windows
// so the class can be unregistered again when the last one closes.
static const wchar_t YAWGPU_WINDOW_CLASS_NAME[] = L"YawgpuExampleWindow";
static ATOM yawgpu_window_class = 0;
static unsigned int yawgpu_window_count = 0;

static void yawgpu_unregister_window_class_if_unused(void) {
    if (yawgpu_window_count == 0 && yawgpu_window_class) {
        UnregisterClassW(YAWGPU_WINDOW_CLASS_NAME, GetModuleHandleW(NULL));
        yawgpu_window_class = 0;
    }
}

// Win32 wants wide (UTF-16) strings; the example titles are plain ASCII, so
// a byte-to-wchar widen is all we need (truncated to fit dest).
static void widen_ascii(const char *source, wchar_t *dest, size_t dest_len) {
    if (!dest || dest_len == 0) {
        return;
    }
    size_t i = 0;
    if (source) {
        for (; source[i] && i + 1 < dest_len; ++i) {
            dest[i] = (wchar_t)(unsigned char)source[i];
        }
    }
    dest[i] = L'\0';
}

// Window procedure: handles close/destroy by flagging should_close (which
// yawgpu_window_should_close reports), and defers everything else to the OS.
static LRESULT CALLBACK yawgpu_window_proc(HWND hwnd,
                                           UINT message,
                                           WPARAM wparam,
                                           LPARAM lparam) {
    // The HWND stores a back-pointer to our YawgpuWindow in its USERDATA
    // slot. On the very first message (WM_NCCREATE) that slot is still empty,
    // so grab the pointer from the creation params and stash it for later.
    YawgpuWindow *window = (YawgpuWindow *)GetWindowLongPtrW(hwnd, GWLP_USERDATA);
    if (message == WM_NCCREATE) {
        CREATESTRUCTW *create = (CREATESTRUCTW *)lparam;
        window = (YawgpuWindow *)create->lpCreateParams;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)window);
    }

    switch (message) {
    case WM_CLOSE:
        if (window) {
            window->should_close = true;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        if (window) {
            window->should_close = true;
        }
        return 0;
    default:
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }
}

static bool yawgpu_register_window_class(HINSTANCE hinstance) {
    if (yawgpu_window_class) {
        return true;
    }

    WNDCLASSEXW window_class = {
        .cbSize = sizeof(WNDCLASSEXW),
        .style = CS_HREDRAW | CS_VREDRAW,
        .lpfnWndProc = yawgpu_window_proc,
        .hInstance = hinstance,
        .hCursor = LoadCursorW(NULL, MAKEINTRESOURCEW(32512)),
        .lpszClassName = YAWGPU_WINDOW_CLASS_NAME,
    };
    yawgpu_window_class = RegisterClassExW(&window_class);
    if (!yawgpu_window_class) {
        fprintf(stderr, "failed to register Win32 window class\n");
        return false;
    }
    return true;
}

YawgpuWindow *yawgpu_window_create(int width, int height, const char *title) {
    HINSTANCE hinstance = GetModuleHandleW(NULL);
    if (!yawgpu_register_window_class(hinstance)) {
        return NULL;
    }

    // The YawgpuWindow is allocated up front so its address can be passed as
    // CreateWindowExW's last argument and recovered in the window procedure.

    YawgpuWindow *window = (YawgpuWindow *)calloc(1, sizeof(YawgpuWindow));
    if (!window) {
        return NULL;
    }
    window->hinstance = hinstance;

    // `width`/`height` are the desired client (drawable) area. Win32 sizes
    // include the title bar and borders, so AdjustWindowRectEx grows the rect
    // to the full window size that yields exactly that client area.
    DWORD style = WS_OVERLAPPEDWINDOW;
    DWORD ex_style = 0;
    RECT rect = {
        .left = 0,
        .top = 0,
        .right = width,
        .bottom = height,
    };
    if (!AdjustWindowRectEx(&rect, style, FALSE, ex_style)) {
        fprintf(stderr, "failed to adjust Win32 window rect\n");
        free(window);
        yawgpu_unregister_window_class_if_unused();
        return NULL;
    }

    wchar_t wide_title[256];
    widen_ascii(title, wide_title, sizeof(wide_title) / sizeof(wide_title[0]));
    HWND hwnd = CreateWindowExW(ex_style,
                                YAWGPU_WINDOW_CLASS_NAME,
                                wide_title,
                                style,
                                CW_USEDEFAULT,
                                CW_USEDEFAULT,
                                rect.right - rect.left,
                                rect.bottom - rect.top,
                                NULL,
                                NULL,
                                hinstance,
                                window);
    if (!hwnd) {
        fprintf(stderr, "failed to create Win32 window\n");
        free(window);
        yawgpu_unregister_window_class_if_unused();
        return NULL;
    }

    window->hwnd = hwnd;
    ++yawgpu_window_count;
    ShowWindow(hwnd, SW_SHOW); // make it visible
    UpdateWindow(hwnd);        // force an initial paint
    return window;
}

void yawgpu_window_destroy(YawgpuWindow *window) {
    if (!window) {
        return;
    }
    if (window->hwnd) {
        // Clear the back-pointer first so any late WM_DESTROY won't touch the
        // memory we're about to free.
        SetWindowLongPtrW(window->hwnd, GWLP_USERDATA, 0);
        DestroyWindow(window->hwnd);
    }
    free(window);

    if (yawgpu_window_count > 0) {
        --yawgpu_window_count;
    }
    yawgpu_unregister_window_class_if_unused();
}

bool yawgpu_window_should_close(YawgpuWindow *window) {
    return !window || window->should_close;
}

// Drains the Win32 message queue, dispatching each message to the window
// procedure above. Called once per frame, like glfwPollEvents on macOS.
void yawgpu_window_poll_events(void) {
    MSG message;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

// Wraps the Win32 HWND as a WebGPU surface. The native handle is passed via
// a WGPUSurfaceSourceWindowsHWND struct chained onto the surface descriptor
// (the Windows analogue of the CAMetalLayer source used on macOS).
WGPUSurface yawgpu_window_create_surface(WGPUInstance instance,
                                         YawgpuWindow *window,
                                         const char *label) {
    if (!window || !window->hwnd) {
        fprintf(stderr, "failed to get Win32 HWND\n");
        return NULL;
    }

    WGPUSurfaceSourceWindowsHWND hwnd_source = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_SurfaceSourceWindowsHWND,
        },
        .hinstance = (void *)window->hinstance,
        .hwnd = (void *)window->hwnd,
    };
    WGPUSurface surface = wgpuInstanceCreateSurface(
        instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &hwnd_source.chain,
            .label = yawgpu_string_view(label),
        });
    if (!surface) {
        fprintf(stderr, "failed to create surface\n");
    }
    return surface;
}

// macOS-only concept; there is no CAMetalLayer on Windows.
void *yawgpu_window_metal_layer(YawgpuWindow *window) {
    YAWGPU_UNUSED(window);
    return NULL;
}

// Reports the framebuffer size in physical pixels. The Win32 client rect is
// already in pixels, so (unlike macOS) no drawable-size sync step is needed.
void yawgpu_window_framebuffer_size(YawgpuWindow *window, int *width, int *height) {
    int local_width = 0;
    int local_height = 0;
    if (window && window->hwnd) {
        RECT rect;
        if (GetClientRect(window->hwnd, &rect)) {
            local_width = rect.right - rect.left;
            local_height = rect.bottom - rect.top;
        }
    }
    if (width) {
        *width = local_width;
    }
    if (height) {
        *height = local_height;
    }
}
