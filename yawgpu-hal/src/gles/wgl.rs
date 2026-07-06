//! WGL backend (Windows OpenGL via opengl32.dll).
//!
//! This path is selected with `YAWGPU_GLES_BACKEND=wgl` on Windows. It creates
//! an OpenGL ES profile context through `WGL_EXT_create_context_es2_profile`
//! so the GLES e2e tests can run against the host GL driver without ANGLE.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use glow::HasContext;
use parking_lot::{Mutex, MutexGuard};
use windows_sys::Win32::Foundation::{HMODULE, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
use windows_sys::Win32::Graphics::OpenGL::{
    wglCreateContext, wglDeleteContext, wglGetProcAddress, wglMakeCurrent, HGLRC,
};
use windows_sys::Win32::Graphics::OpenGL::{
    ChoosePixelFormat, SetPixelFormat, SwapBuffers, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW,
    PFD_MAIN_PLANE, PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassExW, UnregisterClassW,
    CW_USEDEFAULT, WNDCLASSEXW, WS_OVERLAPPEDWINDOW,
};

use super::adapter::{
    detect_base_vertex_support, detect_color_render_caps, parse_gles_version,
    query_gles_adapter_caps,
};
use super::device::GlesSampleMaskIFn;
use super::format::GlesColorRenderCaps;
use super::sampler::create_nearest_placeholder_sampler;
use super::BACKEND;
use crate::HalError;

const WGL_CONTEXT_MAJOR_VERSION_ARB: i32 = 0x2091;
const WGL_CONTEXT_MINOR_VERSION_ARB: i32 = 0x2092;
const WGL_CONTEXT_PROFILE_MASK_ARB: i32 = 0x9126;
const WGL_CONTEXT_ES2_PROFILE_BIT_EXT: i32 = 0x0000_0004;
static NEXT_CLASS_ID: AtomicU64 = AtomicU64::new(1);
static REPORTED_MISSING_SWAP_INTERVAL: AtomicBool = AtomicBool::new(false);

type WglCreateContextAttribsArbFn =
    unsafe extern "system" fn(hdc: HDC, share_context: HGLRC, attribs: *const i32) -> HGLRC;
type WglGetExtensionsStringArbFn = unsafe extern "system" fn(hdc: HDC) -> *const i8;
type WglSwapIntervalExtFn = unsafe extern "system" fn(interval: i32) -> i32;

pub(super) struct WglInstanceState {
    opengl32: HMODULE,
    hinstance: HMODULE,
    class_name: Vec<u16>,
}

// SAFETY: The handles are immutable process/window-class handles. Context use
// is synchronized at the device level.
unsafe impl Send for WglInstanceState {}
// SAFETY: See the `Send` impl.
unsafe impl Sync for WglInstanceState {}

impl Drop for WglInstanceState {
    fn drop(&mut self) {
        unsafe {
            let _ = UnregisterClassW(self.class_name.as_ptr(), self.hinstance);
        }
    }
}

impl WglInstanceState {
    pub(super) fn new() -> Result<Self, HalError> {
        let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) };
        if hinstance.is_null() {
            eprintln!("yawgpu-gles: GetModuleHandleW failed");
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        }

        let opengl32 = unsafe {
            let name: Vec<u16> = "opengl32.dll\0".encode_utf16().collect();
            LoadLibraryW(name.as_ptr())
        };
        if opengl32.is_null() {
            eprintln!("yawgpu-gles: LoadLibrary(opengl32.dll) failed");
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        }

        let class_id = NEXT_CLASS_ID.fetch_add(1, Ordering::Relaxed);
        let class_name: Vec<u16> = format!("yawgpu_wgl_helper_{class_id}\0")
            .encode_utf16()
            .collect();
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: Some(default_wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinstance,
            hIcon: std::ptr::null_mut(),
            hCursor: std::ptr::null_mut(),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: class_name.as_ptr(),
            hIconSm: std::ptr::null_mut(),
        };
        if unsafe { RegisterClassExW(&wc) } == 0 {
            eprintln!("yawgpu-gles: RegisterClassExW failed");
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        }

        Ok(Self {
            opengl32,
            hinstance,
            class_name,
        })
    }

    pub(super) fn create_window_surface(&self, hwnd: HWND) -> Result<WglSurfaceState, HalError> {
        let hdc = unsafe { GetDC(hwnd) };
        if hdc.is_null() {
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "GetDC on user HWND failed",
            });
        }
        if !set_pixel_format(hdc) {
            unsafe {
                let _ = ReleaseDC(hwnd, hdc);
            }
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "ChoosePixelFormat / SetPixelFormat on user HWND failed",
            });
        }
        Ok(WglSurfaceState { hwnd, hdc })
    }
}

pub(super) fn query_adapter_caps(
    instance: Arc<super::instance::GlesInstanceInner>,
    wgl_state: &WglInstanceState,
) -> Result<super::adapter::GlesAdapterCaps, HalError> {
    let state = WglDeviceState::create(instance, wgl_state)?;
    Ok(query_gles_adapter_caps(
        &state.gl,
        state.gl.supported_extensions(),
    ))
}

unsafe extern "system" fn default_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

pub(super) struct WglDeviceState {
    hwnd: HWND,
    hdc: HDC,
    hglrc: HGLRC,
    pub(super) gl: glow::Context,
    current_lock: Mutex<()>,
    pub(super) allocations: AtomicU64,
    /// Whether the context supports the base-vertex indexed-draw entry
    /// points (GLES 3.2 core or `GL_OES/EXT_draw_elements_base_vertex`);
    /// detected once at device creation (T-G11).
    pub(super) supports_base_vertex: bool,
    /// Extension-gated float color-renderability caps
    /// (`GL_EXT_color_buffer_float` / `GL_EXT_color_buffer_half_float`);
    /// detected once at device creation (T-G12).
    pub(super) color_render_caps: GlesColorRenderCaps,
    /// Maximum sample count reported by `GL_MAX_SAMPLES`.
    pub(super) max_samples: i32,
    /// GLES 3.1 core `glSampleMaski`; cached because glow 0.14 does not expose
    /// a public wrapper on `HasContext`.
    pub(super) sample_mask_i: Option<GlesSampleMaskIFn>,
    pub(super) placeholder_sampler: Result<glow::Sampler, HalError>,
}

// SAFETY: All GL access goes through `with_current_context`, which serializes
// `wglMakeCurrent` and the GL operation behind `current_lock`.
unsafe impl Send for WglDeviceState {}
// SAFETY: See the `Send` impl.
unsafe impl Sync for WglDeviceState {}

impl Drop for WglDeviceState {
    fn drop(&mut self) {
        unsafe {
            let _ = wglMakeCurrent(self.hdc, self.hglrc);
            if let Ok(sampler) = self.placeholder_sampler.as_ref() {
                self.gl.delete_sampler(*sampler);
            }
            let _ = wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
            if !self.hglrc.is_null() {
                let _ = wglDeleteContext(self.hglrc);
            }
            if !self.hdc.is_null() && !self.hwnd.is_null() {
                let _ = ReleaseDC(self.hwnd, self.hdc);
            }
            if !self.hwnd.is_null() {
                let _ = DestroyWindow(self.hwnd);
            }
        }
    }
}

impl WglDeviceState {
    pub(super) fn create(
        _instance: Arc<super::instance::GlesInstanceInner>,
        wgl_state: &WglInstanceState,
    ) -> Result<Self, HalError> {
        let hwnd = create_helper_window(wgl_state)?;
        let hdc = unsafe { GetDC(hwnd) };
        if hdc.is_null() {
            unsafe {
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        if !set_pixel_format(hdc) {
            unsafe {
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let dummy_context = unsafe { wglCreateContext(hdc) };
        if dummy_context.is_null() {
            unsafe {
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }
        if unsafe { wglMakeCurrent(hdc, dummy_context) } == 0 {
            unsafe {
                let _ = wglDeleteContext(dummy_context);
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let Some(create_context_attribs) =
            load_wgl_proc::<WglCreateContextAttribsArbFn>("wglCreateContextAttribsARB")
        else {
            eprintln!(
                "yawgpu-gles: wglCreateContextAttribsARB not found; host GL driver lacks WGL_ARB_create_context"
            );
            destroy_dummy_and_window(dummy_context, hwnd, hdc);
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        };

        let es_profile_supported =
            load_wgl_proc::<WglGetExtensionsStringArbFn>("wglGetExtensionsStringARB")
                .and_then(|get_extensions| {
                    let ptr = unsafe { get_extensions(hdc) };
                    if ptr.is_null() {
                        None
                    } else {
                        Some(
                            unsafe { std::ffi::CStr::from_ptr(ptr) }
                                .to_string_lossy()
                                .into_owned(),
                        )
                    }
                })
                .is_some_and(|extensions| {
                    extensions
                        .split_whitespace()
                        .any(|ext| ext == "WGL_EXT_create_context_es2_profile")
                });
        if !es_profile_supported {
            eprintln!("yawgpu-gles: ES profile not supported by the host GL driver");
            destroy_dummy_and_window(dummy_context, hwnd, hdc);
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let attribs = [
            WGL_CONTEXT_MAJOR_VERSION_ARB,
            3,
            WGL_CONTEXT_MINOR_VERSION_ARB,
            1,
            WGL_CONTEXT_PROFILE_MASK_ARB,
            WGL_CONTEXT_ES2_PROFILE_BIT_EXT,
            0,
        ];
        let hglrc = unsafe { create_context_attribs(hdc, std::ptr::null_mut(), attribs.as_ptr()) };
        unsafe {
            let _ = wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
            let _ = wglDeleteContext(dummy_context);
        }
        if hglrc.is_null() {
            eprintln!(
                "yawgpu-gles: wglCreateContextAttribsARB(ES 3.1) failed; host driver may lack WGL_EXT_create_context_es2_profile"
            );
            unsafe {
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        if unsafe { wglMakeCurrent(hdc, hglrc) } == 0 {
            unsafe {
                let _ = wglDeleteContext(hglrc);
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let opengl32 = wgl_state.opengl32;
        let gl =
            unsafe { glow::Context::from_loader_function(|name| load_gl_func(opengl32, name)) };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        eprintln!("yawgpu-gles: WGL GL_VERSION={version:?}");
        let Some((major, minor)) = parse_gles_version(&version) else {
            eprintln!("yawgpu-gles: unable to parse GL_VERSION={version:?}");
            unsafe {
                let _ = wglDeleteContext(hglrc);
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        };
        if (major, minor) < (3, 1) {
            eprintln!(
                "yawgpu-gles: GLES {major}.{minor} below the required 3.1 (GL_VERSION={version:?})"
            );
            unsafe {
                let _ = wglDeleteContext(hglrc);
                let _ = ReleaseDC(hwnd, hdc);
                let _ = DestroyWindow(hwnd);
            }
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let supports_base_vertex =
            detect_base_vertex_support((major, minor), gl.supported_extensions());
        let color_render_caps = detect_color_render_caps(gl.supported_extensions());
        let max_samples = unsafe { gl.get_parameter_i32(glow::MAX_SAMPLES) };
        let sample_mask_i = load_wgl_proc::<GlesSampleMaskIFn>("glSampleMaski");
        let placeholder_sampler = unsafe { create_nearest_placeholder_sampler(&gl) };

        Ok(Self {
            hwnd,
            hdc,
            hglrc,
            gl,
            current_lock: Mutex::new(()),
            allocations: AtomicU64::new(0),
            supports_base_vertex,
            color_render_caps,
            max_samples,
            sample_mask_i,
            placeholder_sampler,
        })
    }

    pub(super) fn current_lock_acquire(&self) -> MutexGuard<'_, ()> {
        self.current_lock.lock()
    }

    pub(super) fn gl(&self) -> &glow::Context {
        &self.gl
    }

    pub(super) fn make_current_on_hdc(&self, hdc: HDC) -> Result<(), HalError> {
        if unsafe { wglMakeCurrent(hdc, self.hglrc) } == 0 {
            return Err(HalError::PresentFailed {
                backend: BACKEND,
                message: "wglMakeCurrent(window) failed",
            });
        }
        Ok(())
    }

    pub(super) fn restore_current(&self) {
        unsafe {
            let _ = wglMakeCurrent(self.hdc, self.hglrc);
        }
    }

    pub(super) fn with_current_context<R>(
        &self,
        f: impl FnOnce(&glow::Context) -> R,
    ) -> Result<R, HalError> {
        let _guard = self.current_lock.lock();
        if unsafe { wglMakeCurrent(self.hdc, self.hglrc) } == 0 {
            return Err(HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: "wglMakeCurrent failed".to_string(),
            });
        }
        Ok(f(&self.gl))
    }
}

pub(super) struct WglSurfaceState {
    hwnd: HWND,
    hdc: HDC,
}

// SAFETY: The HWND/HDC are immutable handles. GL access is serialized by the
// configured device's context lock in `surface.rs`.
unsafe impl Send for WglSurfaceState {}
// SAFETY: See the `Send` impl.
unsafe impl Sync for WglSurfaceState {}

impl WglSurfaceState {
    pub(super) fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub(super) fn hdc(&self) -> HDC {
        self.hdc
    }
}

pub(super) fn release_surface_dc(surface: &WglSurfaceState) {
    unsafe {
        let _ = wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
        let _ = ReleaseDC(surface.hwnd(), surface.hdc());
    }
}

pub(super) fn swap_buffers(hdc: HDC) -> Result<(), HalError> {
    if unsafe { SwapBuffers(hdc) } == 0 {
        return Err(HalError::PresentFailed {
            backend: BACKEND,
            message: "SwapBuffers failed",
        });
    }
    Ok(())
}

pub(super) fn swap_interval(interval: i32) {
    let Some(wgl_swap_interval) = load_wgl_proc::<WglSwapIntervalExtFn>("wglSwapIntervalEXT")
    else {
        if !REPORTED_MISSING_SWAP_INTERVAL.swap(true, Ordering::Relaxed) {
            eprintln!(
                "yawgpu-gles: wglSwapIntervalEXT not found; present mode interval is a no-op"
            );
        }
        return;
    };
    unsafe {
        let _ = wgl_swap_interval(interval);
    }
}

fn create_helper_window(wgl_state: &WglInstanceState) -> Result<HWND, HalError> {
    let window_name: Vec<u16> = "yawgpu wgl helper\0".encode_utf16().collect();
    let hwnd = unsafe {
        CreateWindowExW(
            0,
            wgl_state.class_name.as_ptr(),
            window_name.as_ptr(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            wgl_state.hinstance,
            std::ptr::null(),
        )
    };
    if hwnd.is_null() {
        Err(HalError::DeviceCreationFailed { backend: BACKEND })
    } else {
        Ok(hwnd)
    }
}

fn set_pixel_format(hdc: HDC) -> bool {
    let pfd = build_pixel_format_descriptor();
    let pixel_format = unsafe { ChoosePixelFormat(hdc, &pfd) };
    pixel_format != 0 && unsafe { SetPixelFormat(hdc, pixel_format, &pfd) } != 0
}

fn build_pixel_format_descriptor() -> PIXELFORMATDESCRIPTOR {
    let mut pfd: PIXELFORMATDESCRIPTOR = unsafe { std::mem::zeroed() };
    pfd.nSize = std::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16;
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE as u8;
    pfd
}

fn destroy_dummy_and_window(dummy_context: HGLRC, hwnd: HWND, hdc: HDC) {
    unsafe {
        let _ = wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
        let _ = wglDeleteContext(dummy_context);
        let _ = ReleaseDC(hwnd, hdc);
        let _ = DestroyWindow(hwnd);
    }
}

fn load_gl_func(opengl32: HMODULE, name: &str) -> *const std::ffi::c_void {
    let Ok(cname) = CString::new(name) else {
        return std::ptr::null();
    };
    let proc = unsafe { wglGetProcAddress(cname.as_ptr() as *const u8) };
    if let Some(proc) = proc {
        let ptr = proc as *const std::ffi::c_void;
        let value = ptr as usize;
        if value > 3 && value != usize::MAX {
            return ptr;
        }
    }
    unsafe { GetProcAddress(opengl32, cname.as_ptr() as *const u8) }
        .map(|proc| proc as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null())
}

fn load_wgl_proc<T>(name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let proc = unsafe { wglGetProcAddress(cname.as_ptr() as *const u8) }?;
    Some(unsafe { std::mem::transmute_copy(&proc) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_pixel_format_descriptor_matches_wgl_surface_contract() {
        let pfd = build_pixel_format_descriptor();
        assert_eq!(
            pfd.nSize,
            std::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16
        );
        assert_eq!(pfd.nVersion, 1);
        assert_eq!(
            pfd.dwFlags,
            PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        );
        assert_eq!(pfd.iPixelType, PFD_TYPE_RGBA);
        assert_eq!(pfd.cColorBits, 32);
        assert_eq!(pfd.cAlphaBits, 8);
        assert_eq!(pfd.cDepthBits, 24);
        assert_eq!(pfd.cStencilBits, 8);
        assert_eq!(pfd.iLayerType, PFD_MAIN_PLANE as u8);
    }
}
