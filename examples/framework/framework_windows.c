#include "framework.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

struct YawgpuWindow {
    HWND hwnd;
    HINSTANCE hinstance;
    bool should_close;
};

static const wchar_t YAWGPU_WINDOW_CLASS_NAME[] = L"YawgpuExampleWindow";
static ATOM yawgpu_window_class = 0;
static unsigned int yawgpu_window_count = 0;

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

static LRESULT CALLBACK yawgpu_window_proc(HWND hwnd,
                                           UINT message,
                                           WPARAM wparam,
                                           LPARAM lparam) {
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

    YawgpuWindow *window = (YawgpuWindow *)calloc(1, sizeof(YawgpuWindow));
    if (!window) {
        return NULL;
    }
    window->hinstance = hinstance;

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
        return NULL;
    }

    window->hwnd = hwnd;
    ++yawgpu_window_count;
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    return window;
}

void yawgpu_window_destroy(YawgpuWindow *window) {
    if (!window) {
        return;
    }
    if (window->hwnd) {
        DestroyWindow(window->hwnd);
    }
    free(window);

    if (yawgpu_window_count > 0) {
        --yawgpu_window_count;
    }
    if (yawgpu_window_count == 0 && yawgpu_window_class) {
        UnregisterClassW(YAWGPU_WINDOW_CLASS_NAME, GetModuleHandleW(NULL));
        yawgpu_window_class = 0;
    }
}

bool yawgpu_window_should_close(YawgpuWindow *window) {
    return !window || window->should_close;
}

void yawgpu_window_poll_events(void) {
    MSG message;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

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

void *yawgpu_window_metal_layer(YawgpuWindow *window) {
    YAWGPU_UNUSED(window);
    return NULL;
}

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
