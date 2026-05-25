use khronos_egl as egl;

use super::BACKEND;
use crate::HalError;

pub(super) type EglInstance = egl::DynamicInstance<egl::EGL1_4>;
pub(super) type EglDisplay = egl::Display;
pub(super) type EglConfig = egl::Config;
pub(super) type EglContext = egl::Context;
pub(super) type EglSurface = egl::Surface;

// ANGLE platform-selection constants (EGL_ANGLE_platform_angle extension).
// Not exported by khronos-egl; declared here. Values from ANGLE's
// EGL/eglext_angle.h. Only consumed by the Windows ANGLE
// platform-display cascade in `get_and_initialize_display`, so gate the
// declarations to `cfg(windows)` to keep non-Windows builds clean under
// `-D warnings`.
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_ANGLE: egl::Enum = 0x3202;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_TYPE_ANGLE: egl::Attrib = 0x3203;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE: egl::Attrib = 0x3208;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE: egl::Attrib = 0x3450;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE: egl::Attrib = 0x3209;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE: egl::Attrib = 0x320A;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE: egl::Attrib = 0x3210;
#[cfg(windows)]
pub(super) const EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE: egl::Attrib = 0x3211;

/// Acquires the EGL display that gives the best chance of an ES 3.1
/// context. On Windows the loaded EGL is ANGLE (Tier 2 / experimental
/// target); ANGLE's default-display path often picks the OpenGL backend
/// which caps at ES 3.0 on some host drivers, so we explicitly request
/// `EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE` via `eglGetPlatformDisplay`
/// (EGL 1.5) which uniformly exposes ES 3.1 on Feature Level 11.0+
/// hardware. On Android the native EGL implementation already returns a
/// display backed by the device's GPU driver — typically Mali / Adreno /
/// PowerVR — which exposes ES 3.1+ directly, so the default-display path
/// is correct without additional platform attributes. The platform branch
/// always falls back to `eglGetDisplay(EGL_DEFAULT_DISPLAY)` when the
/// preferred selection isn't available, so Tier 2 GLES still loads
/// (just possibly capped at ES 3.0, which then fails the version check
/// in `adapter::create_device` with a clear diagnostic).
pub(super) fn get_and_initialize_display(egl: &EglInstance) -> Option<EglDisplay> {
    #[cfg(windows)]
    {
        if let Some(egl15) = egl.upcast::<egl::EGL1_5>() {
            // Cascade: try the ANGLE backends most likely to expose ES 3.1+
            // first. Vulkan exposes the full ES 3.2 surface natively;
            // D3D11 exposes ES 3.1 on Feature Level 11.0+. Some ANGLE
            // builds (notably Chrome's bundled libGLESv2.dll, which is a
            // WebGL2-targeted build) cap at ES 3.0 on the D3D11 backend;
            // Vulkan-backed ANGLE bypasses that cap when Vulkan drivers
            // are installed (NVIDIA / AMD / Intel ship them on Win10+).
            // Each candidate is fully initialized before being accepted,
            // so a display whose backend can be acquired but not
            // initialized (e.g. ANGLE Vulkan when the host Vulkan ICD
            // refuses) falls through to the next candidate.
            for (kind, type_value) in [
                ("Vulkan", EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE),
                ("D3D11", EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE),
            ] {
                // Explicitly request ES 3.1 via the ANGLE max-version attrs.
                // Some ANGLE builds (Chrome's bundled libGLESv2 in
                // particular) default to ES 3.0 unless told otherwise.
                let attribs: [egl::Attrib; 9] = [
                    EGL_PLATFORM_ANGLE_TYPE_ANGLE,
                    type_value,
                    EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE,
                    EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE,
                    EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE,
                    3,
                    EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE,
                    1,
                    egl::ATTRIB_NONE,
                ];
                // SAFETY: `egl::DEFAULT_DISPLAY` is the platform-default
                // sentinel, valid for `eglGetPlatformDisplay`. The
                // attribute list is a stack-local array kept alive across
                // this call.
                let display = unsafe {
                    egl15.get_platform_display(
                        EGL_PLATFORM_ANGLE_ANGLE,
                        egl::DEFAULT_DISPLAY,
                        &attribs,
                    )
                };
                let Ok(display) = display else {
                    eprintln!(
                        "yawgpu-gles: eglGetPlatformDisplay(ANGLE/{kind}) failed; trying next backend"
                    );
                    continue;
                };
                if egl.initialize(display).is_ok() {
                    eprintln!("yawgpu-gles: using ANGLE {kind} backend");
                    return Some(display);
                }
                eprintln!(
                    "yawgpu-gles: eglInitialize on ANGLE/{kind} display failed; trying next backend"
                );
                // The failed-init display is implicitly released by ANGLE
                // when no further references exist; no eglTerminate needed.
            }
        }
    }
    // Non-Windows (Android / Linux / etc.) and Windows-fallback path: the
    // system EGL's default display is the right choice. Android's native
    // EGL maps it to the device's GPU driver (which advertises ES 3.1+
    // directly on hardware shipping since ~2016).
    // SAFETY: `egl::DEFAULT_DISPLAY` is the documented sentinel value for
    // `eglGetDisplay` on every supported platform.
    let display = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }?;
    if egl.initialize(display).is_ok() {
        Some(display)
    } else {
        eprintln!("yawgpu-gles: eglInitialize on default display failed");
        None
    }
}

#[cfg(windows)]
fn preload_angle_from_env() {
    let Some(dir) = std::env::var_os("YAWGPU_ANGLE_PATH") else {
        return;
    };

    for dll in ["libEGL.dll", "libGLESv2.dll"] {
        let mut path = std::path::PathBuf::from(&dir);
        path.push(dll);
        if let Ok(library) = unsafe { libloading::Library::new(&path) } {
            std::mem::forget(library);
        }
    }
}

#[cfg(not(windows))]
fn preload_angle_from_env() {}

pub(super) fn load_egl() -> Result<EglInstance, HalError> {
    preload_angle_from_env();
    #[cfg(windows)]
    let loaded = unsafe { EglInstance::load_required_from_filename("libEGL.dll") };
    #[cfg(not(windows))]
    let loaded = unsafe { EglInstance::load_required() };

    loaded.map_err(|err| {
        // Diagnostic: surface the underlying libloading error so callers can
        // tell ANGLE-DLL-not-found apart from architecture-mismatch / missing
        // dependency / wrong-version cases when GLES silently falls back to
        // Noop. Single eprintln on the failure path; never hit on success.
        eprintln!("yawgpu-gles: load_egl failed: {err}");
        HalError::BackendUnavailable { backend: BACKEND }
    })
}
