use khronos_egl as egl;

#[cfg(target_os = "linux")]
use super::instance::EglDeviceChoice;
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

// ---------------------------------------------------------------------------
// Linux EGL device selection (Block 67, "Linux EGL device selection").
//
// `eglGetDisplay(EGL_DEFAULT_DISPLAY)` is correct on Android — the platform
// EGL maps it to the device's own GPU driver — but on a desktop Linux host
// libglvnd hands it to whichever vendor claims the default display, which
// silently resolves to Mesa's software rasterizer (llvmpipe) when that vendor
// cannot drive the installed GPU. The cascade below prefers a *validated*
// hardware `EGL_PLATFORM_DEVICE_EXT` display and falls back to the default
// display, so the worst case is exactly today's behaviour.
//
// Gated on `target_os = "linux"`, which excludes Android by construction
// (`target_os = "android"` is a distinct value), and leaves the Windows/ANGLE
// arm and the default-display arm untouched on every other target.
//
// An `EGL_PLATFORM_DEVICE_EXT` display is headless and cannot back
// `eglCreateWindowSurface`. Linux windowed presentation is out of scope for
// this block (surface constructors exist for Android and Windows only), so
// nothing here can regress a surface path. If Linux windowed presentation is
// ever added, this cascade must be skipped — or the device display rejected —
// whenever a window surface is requested.
// ---------------------------------------------------------------------------

/// `EGL_PLATFORM_DEVICE_EXT` from `EGL_EXT_platform_device`. Not exported by
/// khronos-egl; declared here like the ANGLE constants above.
#[cfg(target_os = "linux")]
const EGL_PLATFORM_DEVICE_EXT: egl::Enum = 0x313F;

/// `EGLDeviceEXT` from `EGL_EXT_device_query`; an opaque handle.
#[cfg(target_os = "linux")]
type EglDeviceExt = *mut std::ffi::c_void;

#[cfg(target_os = "linux")]
type EglQueryDevicesExtFn =
    unsafe extern "system" fn(egl::Int, *mut EglDeviceExt, *mut egl::Int) -> egl::Boolean;

#[cfg(target_os = "linux")]
type EglQueryDeviceStringExtFn =
    unsafe extern "system" fn(EglDeviceExt, egl::Int) -> *const std::ffi::c_char;

// Note: `eglGetPlatformDisplayEXT` takes `const EGLint *attrib_list`, not the
// EGL 1.5 core `EGLAttrib *`, which is why the EGL 1.5 wrapper in khronos-egl
// cannot be reused here.
#[cfg(target_os = "linux")]
type EglGetPlatformDisplayExtFn =
    unsafe extern "system" fn(egl::Enum, *mut std::ffi::c_void, *const egl::Int) -> egl::EGLDisplay;

/// Returns whether `name` appears as a whole entry in a space-separated EGL
/// extension string. Substring matching would accept `EGL_EXT_device_base`
/// for `EGL_EXT_device_b`, so split on whitespace instead.
#[cfg(target_os = "linux")]
fn extension_present(extensions: &str, name: &str) -> bool {
    extensions.split_whitespace().any(|entry| entry == name)
}

/// How an enumerated EGL device is classified for the `Auto` cascade.
///
/// Three-valued rather than a `bool` because `eglQueryDeviceStringEXT` can
/// return NULL. Folding that case into "hardware" would let the software
/// rasterizer join the `Auto` list and be selected while the diagnostic
/// claimed `(hardware)` — the silent software fallback D1 forbids. Folding it
/// into "software" is no better: a genuine hardware device whose query fails
/// would become unreachable under `Auto`, which then falls through to the
/// default display and lands back on software. So `Unknown` is its own value:
/// still eligible for `Auto`, just ordered after every known-hardware device.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EglDeviceKind {
    /// The device's extension string was read and does not advertise
    /// `EGL_MESA_device_software`.
    Hardware,
    /// The device advertises `EGL_MESA_device_software`.
    Software,
    /// The device's extension string could not be read, so it is neither
    /// confirmed hardware nor confirmed software.
    Unknown,
}

#[cfg(target_os = "linux")]
impl EglDeviceKind {
    /// The word this kind contributes to the `yawgpu-gles:` selection
    /// diagnostic, so a log line never claims more than was actually probed.
    fn label(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
            Self::Unknown => "unknown",
        }
    }
}

/// Maps an [`EglDeviceChoice`] plus the per-device classification onto the
/// enumeration indices the cascade should try, in order.
///
/// Returns `None` when the ask is well-formed but unsatisfiable (index out of
/// range, or no software device present); the caller then diagnoses and falls
/// through to the default display rather than substituting another device.
/// [`EglDeviceChoice::Auto`] returns every [`EglDeviceKind::Hardware`] device
/// in enumeration order, followed by every [`EglDeviceKind::Unknown`] one —
/// possibly empty, which is not an error, just an empty candidate list. A
/// [`EglDeviceKind::Software`] device is never part of the `Auto` list.
/// `Software` and `Index` are unaffected by the classification of the devices
/// they do not name.
#[cfg(target_os = "linux")]
fn resolve_device_candidates(
    choice: EglDeviceChoice,
    kinds: &[EglDeviceKind],
) -> Option<Vec<usize>> {
    match choice {
        // `Default` short-circuits before the cascade ever enumerates.
        EglDeviceChoice::Default => None,
        EglDeviceChoice::Auto => {
            let indices_of = |wanted: EglDeviceKind| {
                kinds
                    .iter()
                    .enumerate()
                    .filter(move |(_, kind)| **kind == wanted)
                    .map(|(index, _)| index)
            };
            Some(
                indices_of(EglDeviceKind::Hardware)
                    .chain(indices_of(EglDeviceKind::Unknown))
                    .collect(),
            )
        }
        EglDeviceChoice::Software => kinds
            .iter()
            .position(|kind| *kind == EglDeviceKind::Software)
            .map(|index| vec![index]),
        EglDeviceChoice::Index(index) => {
            let index = usize::try_from(index).ok()?;
            (index < kinds.len()).then(|| vec![index])
        }
    }
}

/// Resolves a client (no-display) EGL entry point through `eglGetProcAddress`
/// and transmutes it to `T`.
///
/// The device-enumeration entry points must resolve before any display
/// exists, so this works from `&EglInstance` rather than from the
/// `EglInstanceState` that `adapter::load_egl_proc` takes.
///
/// `T` must be a bare function pointer. `transmute_copy` does not size-check,
/// so a wider `T` would read past the source with no compile error; the
/// `debug_assert_eq!` below turns that mis-instantiation into a loud failure
/// in debug builds instead.
#[cfg(target_os = "linux")]
fn load_client_proc<T>(egl: &EglInstance, name: &str) -> Option<T> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<extern "system" fn()>(),
        "load_client_proc::<T> requires T to be a bare function pointer"
    );
    let proc = egl.get_proc_address(name)?;
    // SAFETY: `eglGetProcAddress` returned a non-null address for `name`, and
    // every caller instantiates `T` with the `extern "system" fn` signature
    // the EGL extension specification gives for that exact name. The width of
    // `T` — the part `transmute_copy` itself does not check — is enforced by
    // the `debug_assert_eq!` above; matching the signature to the name stays a
    // review obligation, as in `adapter::load_egl_proc`.
    Some(unsafe { std::mem::transmute_copy(&proc) })
}

/// Calls `eglQueryDevicesEXT` twice — once for the count, once for the
/// handles — and returns the enumerated devices.
#[cfg(target_os = "linux")]
fn query_egl_devices(query_devices: EglQueryDevicesExtFn) -> Option<Vec<EglDeviceExt>> {
    let mut count: egl::Int = 0;
    // SAFETY: a null device array with `max_devices == 0` is the documented
    // count-only form; `count` is a live stack local.
    let ok = unsafe { query_devices(0, std::ptr::null_mut(), &mut count) };
    if ok != egl::TRUE {
        eprintln!("yawgpu-gles: eglQueryDevicesEXT(count) failed; using the default display");
        return None;
    }
    let Ok(count_usize) = usize::try_from(count) else {
        eprintln!(
            "yawgpu-gles: eglQueryDevicesEXT reported {count} devices; using the default display"
        );
        return None;
    };
    if count_usize == 0 {
        eprintln!("yawgpu-gles: eglQueryDevicesEXT reported no devices; using the default display");
        return None;
    }

    let mut devices: Vec<EglDeviceExt> = vec![std::ptr::null_mut(); count_usize];
    let mut written: egl::Int = 0;
    // SAFETY: `devices` has exactly `count` elements and outlives the call;
    // `written` is a live stack local.
    let ok = unsafe { query_devices(count, devices.as_mut_ptr(), &mut written) };
    if ok != egl::TRUE {
        eprintln!("yawgpu-gles: eglQueryDevicesEXT(fill) failed; using the default display");
        return None;
    }
    let written = usize::try_from(written).unwrap_or(0).min(count_usize);
    devices.truncate(written);
    Some(devices)
}

/// Classifies an enumerated device by whether it advertises
/// `EGL_MESA_device_software`, i.e. is a software rasterizer.
///
/// A NULL return from `eglQueryDeviceStringEXT` is [`EglDeviceKind::Unknown`],
/// not a default of either polarity — see the [`EglDeviceKind`] doc comment
/// for why neither default is safe. The caller emits the diagnostic, since it
/// is the one that knows the device's enumeration index.
#[cfg(target_os = "linux")]
fn classify_device(
    query_device_string: EglQueryDeviceStringExtFn,
    device: EglDeviceExt,
) -> EglDeviceKind {
    // SAFETY: `device` came from `eglQueryDevicesEXT` and stays valid for the
    // life of the process; `EGL_EXTENSIONS` is a valid `name` for
    // `eglQueryDeviceStringEXT`.
    let raw = unsafe { query_device_string(device, egl::EXTENSIONS) };
    if raw.is_null() {
        return EglDeviceKind::Unknown;
    }
    // SAFETY: EGL owns the returned string and guarantees it is
    // NUL-terminated and valid for the life of the device handle.
    let extensions = unsafe { std::ffi::CStr::from_ptr(raw) };
    if extension_present(&extensions.to_string_lossy(), "EGL_MESA_device_software") {
        EglDeviceKind::Software
    } else {
        EglDeviceKind::Hardware
    }
}

/// Opens, initializes and validates one candidate device display (D2), and on
/// success returns it together with the driver strings the probe reported.
///
/// Validation is `eglInitialize` → `eglBindAPI(EGL_OPENGL_ES_API)` →
/// `choose_config` → the existing throwaway ES 3.1 context + 1×1 pbuffer caps
/// probe, because a successful `eglInitialize` alone is not proof of a usable
/// headless ES 3.1 context. A display that initialized but failed validation
/// is left alone: EGL displays are process-global and yawgpu never calls
/// `eglTerminate` (see the `EglInstanceState` comment in `instance.rs`).
#[cfg(target_os = "linux")]
fn try_device_display(
    egl: &EglInstance,
    get_platform_display: EglGetPlatformDisplayExtFn,
    device: EglDeviceExt,
    index: usize,
) -> Option<(EglDisplay, super::adapter::GlesDriverInfo)> {
    // SAFETY: `device` came from `eglQueryDevicesEXT`, and a null attribute
    // list is the documented "no attributes" form for
    // `eglGetPlatformDisplayEXT`.
    let raw = unsafe { get_platform_display(EGL_PLATFORM_DEVICE_EXT, device, std::ptr::null()) };
    if raw.is_null() {
        eprintln!("yawgpu-gles: eglGetPlatformDisplayEXT failed for EGL device {index}");
        return None;
    }
    // SAFETY: `raw` is the non-null display `eglGetPlatformDisplayEXT` just
    // returned for `device`.
    let display = unsafe { EglDisplay::from_ptr(raw) };

    if let Err(err) = egl.initialize(display) {
        eprintln!("yawgpu-gles: eglInitialize failed for EGL device {index}: {err:?}");
        return None;
    }
    if let Err(err) = egl.bind_api(egl::OPENGL_ES_API) {
        eprintln!("yawgpu-gles: eglBindAPI failed for EGL device {index}: {err:?}");
        return None;
    }
    let config = match super::instance::choose_config(egl, display) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("yawgpu-gles: no usable EGLConfig on EGL device {index}: {err:?}");
            return None;
        }
    };
    let driver = match super::adapter::query_egl_adapter_caps(egl, display, config) {
        Ok((_, driver)) => driver,
        Err(err) => {
            eprintln!("yawgpu-gles: ES 3.1 capability probe failed on EGL device {index}: {err:?}");
            return None;
        }
    };
    Some((display, driver))
}

/// Runs the Linux device cascade and returns the selected display, or `None`
/// to let the caller fall through to `eglGetDisplay(EGL_DEFAULT_DISPLAY)`.
///
/// Every failure is a `yawgpu-gles:` diagnostic plus a fall-through; nothing
/// here panics. Note the two distinct fallbacks: an *unparseable*
/// `YAWGPU_GLES_EGL_DEVICE` degrades to `Auto` at parse time, while a
/// *well-formed but unsatisfiable* ask falls through to the default display
/// rather than silently selecting some other device.
#[cfg(target_os = "linux")]
fn select_linux_device_display(egl: &EglInstance) -> Option<EglDisplay> {
    let choice = super::instance::egl_device_from_env();
    if choice == EglDeviceChoice::Default {
        eprintln!("yawgpu-gles: YAWGPU_GLES_EGL_DEVICE=default; using the default display");
        return None;
    }

    let client_extensions = match egl.query_string(None, egl::EXTENSIONS) {
        Ok(extensions) => extensions.to_string_lossy().into_owned(),
        Err(err) => {
            eprintln!(
                "yawgpu-gles: eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS) failed ({err:?}); using the default display"
            );
            return None;
        }
    };
    let has_enumeration = extension_present(&client_extensions, "EGL_EXT_device_enumeration")
        || extension_present(&client_extensions, "EGL_EXT_device_base");
    if !has_enumeration || !extension_present(&client_extensions, "EGL_EXT_platform_device") {
        eprintln!(
            "yawgpu-gles: EGL device enumeration unavailable (EGL_EXT_device_enumeration / EGL_EXT_platform_device missing); using the default display"
        );
        return None;
    }

    let (Some(query_devices), Some(query_device_string), Some(get_platform_display)) = (
        load_client_proc::<EglQueryDevicesExtFn>(egl, "eglQueryDevicesEXT"),
        load_client_proc::<EglQueryDeviceStringExtFn>(egl, "eglQueryDeviceStringEXT"),
        load_client_proc::<EglGetPlatformDisplayExtFn>(egl, "eglGetPlatformDisplayEXT"),
    ) else {
        eprintln!(
            "yawgpu-gles: EGL device extension entry points did not resolve; using the default display"
        );
        return None;
    };

    let devices = query_egl_devices(query_devices)?;
    let kinds: Vec<EglDeviceKind> = devices
        .iter()
        .enumerate()
        .map(|(index, device)| {
            let kind = classify_device(query_device_string, *device);
            if kind == EglDeviceKind::Unknown {
                eprintln!(
                    "yawgpu-gles: eglQueryDeviceStringEXT(EGL_EXTENSIONS) returned no string for EGL device {index}; cannot tell hardware from software, so it is tried only after every known-hardware device"
                );
            }
            kind
        })
        .collect();

    let Some(candidates) = resolve_device_candidates(choice, &kinds) else {
        eprintln!(
            "yawgpu-gles: YAWGPU_GLES_EGL_DEVICE={choice:?} cannot be satisfied by the {} enumerated EGL device(s); using the default display",
            devices.len()
        );
        return None;
    };

    for index in candidates {
        let (Some(device), Some(kind)) = (devices.get(index), kinds.get(index)) else {
            continue;
        };
        if let Some((display, driver)) =
            try_device_display(egl, get_platform_display, *device, index)
        {
            let kind = kind.label();
            let renderer = &driver.renderer;
            eprintln!(
                "yawgpu-gles: selected EGL device {index} ({kind}) via EGL_PLATFORM_DEVICE_EXT: GL_RENDERER={renderer:?}"
            );
            return Some(display);
        }
    }

    eprintln!(
        "yawgpu-gles: no EGL device satisfied YAWGPU_GLES_EGL_DEVICE={choice:?}; using the default display"
    );
    None
}

/// Acquires the EGL display that gives the best chance of an ES 3.1
/// context. On Windows the loaded EGL is ANGLE (Tier 2 / experimental
/// target); ANGLE's default-display path often picks the OpenGL backend
/// which caps at ES 3.0 on some host drivers, so we explicitly request
/// `EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE` via `eglGetPlatformDisplay`
/// (EGL 1.5) which uniformly exposes ES 3.1 on Feature Level 11.0+
/// hardware. On Android the native EGL implementation already returns a
/// display backed by the device's GPU driver — typically Mali / Adreno /
/// PowerVR — which exposes ES 3.1+ directly, so the default-display path
/// is correct without additional platform attributes. On desktop Linux the
/// default display often resolves to Mesa's software rasterizer, so a
/// `EGL_PLATFORM_DEVICE_EXT` cascade over `eglQueryDevicesEXT` runs first and
/// picks the first *validated* hardware device (overridable through
/// `YAWGPU_GLES_EGL_DEVICE`). The platform branch
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
    #[cfg(target_os = "linux")]
    {
        // Desktop Linux: prefer a validated hardware EGL device over the
        // default display, which libglvnd may resolve to llvmpipe. Any
        // failure falls through to the unchanged default-display path below.
        if let Some(display) = select_linux_device_display(egl) {
            return Some(display);
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use EglDeviceKind::{Hardware, Software, Unknown};

    #[test]
    fn extension_present_matches_whole_entries_only() {
        let extensions = "EGL_EXT_device_base EGL_EXT_device_enumeration EGL_EXT_platform_device";
        assert!(extension_present(extensions, "EGL_EXT_device_base"));
        assert!(extension_present(extensions, "EGL_EXT_platform_device"));
        // A prefix of a listed entry is not a match.
        assert!(!extension_present(extensions, "EGL_EXT_device_b"));
        assert!(!extension_present(extensions, "EGL_MESA_device_software"));
        assert!(!extension_present("", "EGL_EXT_platform_device"));
    }

    #[test]
    fn extension_present_tolerates_irregular_separators() {
        let extensions = "  EGL_EXT_device_base   EGL_MESA_device_software\t";
        assert!(extension_present(extensions, "EGL_MESA_device_software"));
        assert!(extension_present(extensions, "EGL_EXT_device_base"));
    }

    #[test]
    fn resolve_device_candidates_auto_lists_hardware_devices_in_enumeration_order() {
        let kinds = [Hardware, Hardware, Hardware, Software];
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Auto, &kinds),
            Some(vec![0, 1, 2])
        );
    }

    #[test]
    fn resolve_device_candidates_auto_orders_unknown_devices_after_hardware() {
        // A device whose extension string could not be read stays eligible —
        // excluding it would make a real GPU unreachable and send `Auto` back
        // to the default display — but it is only tried once every confirmed
        // hardware device has failed.
        let kinds = [Unknown, Software, Hardware, Unknown, Hardware];
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Auto, &kinds),
            Some(vec![2, 4, 0, 3])
        );
        // Unknown-only enumeration: still a candidate list, not an empty one.
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Auto, &[Unknown, Unknown]),
            Some(vec![0, 1])
        );
    }

    #[test]
    fn resolve_device_candidates_auto_yields_an_empty_list_when_every_device_is_software() {
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Auto, &[Software]),
            Some(Vec::new())
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Auto, &[]),
            Some(Vec::new())
        );
    }

    #[test]
    fn resolve_device_candidates_software_picks_the_first_software_device() {
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Software, &[Hardware, Software, Software]),
            Some(vec![1])
        );
        // An unknown device is never mistaken for the software one.
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Software, &[Unknown, Software]),
            Some(vec![1])
        );
    }

    #[test]
    fn resolve_device_candidates_software_is_unsatisfiable_without_a_software_device() {
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Software, &[Hardware, Hardware]),
            None
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Software, &[Unknown, Unknown]),
            None
        );
    }

    #[test]
    fn resolve_device_candidates_index_pins_the_requested_device() {
        // Including the software one, and including an unknown one: an
        // explicit index overrides the classification entirely.
        let kinds = [Hardware, Unknown, Hardware, Software];
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Index(0), &kinds),
            Some(vec![0])
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Index(1), &kinds),
            Some(vec![1])
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Index(3), &kinds),
            Some(vec![3])
        );
    }

    #[test]
    fn resolve_device_candidates_index_out_of_range_is_unsatisfiable() {
        assert_eq!(
            resolve_device_candidates(
                EglDeviceChoice::Index(4),
                &[Hardware, Hardware, Hardware, Software]
            ),
            None
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Index(0), &[]),
            None
        );
    }

    #[test]
    fn resolve_device_candidates_default_never_produces_candidates() {
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Default, &[Hardware, Software]),
            None
        );
        assert_eq!(
            resolve_device_candidates(EglDeviceChoice::Default, &[Unknown]),
            None
        );
    }

    #[test]
    fn label_names_each_device_kind() {
        assert_eq!(Hardware.label(), "hardware");
        assert_eq!(Software.label(), "software");
        assert_eq!(Unknown.label(), "unknown");
    }
}
