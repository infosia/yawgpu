use std::sync::Arc;

use glow::HasContext;
use khronos_egl as egl;

use super::device::GlesDevice;
use super::egl::{EglConfig, EglContext, EglSurface};
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::BACKEND;
use crate::HalError;

/// Stores GLES adapter data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesAdapter {
    inner: GlesAdapterInner,
}

#[derive(Clone)]
enum GlesAdapterInner {
    Egl {
        instance: Arc<GlesInstanceInner>,
        config: EglConfig,
    },
    #[cfg(windows)]
    Wgl { instance: Arc<GlesInstanceInner> },
}

// SAFETY: The adapter is an immutable handle to an EGL config plus the shared
// instance. Context creation uses EGL calls and returns errors on failure; no
// Rust-managed mutable state is shared through this type.
unsafe impl Send for GlesAdapter {}
// SAFETY: See the `Send` impl; all fields are immutable after construction.
unsafe impl Sync for GlesAdapter {}

impl std::fmt::Debug for GlesAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesAdapter").finish()
    }
}

impl GlesAdapter {
    pub(super) fn new_egl(instance: Arc<GlesInstanceInner>, config: EglConfig) -> Self {
        Self {
            inner: GlesAdapterInner::Egl { instance, config },
        }
    }

    #[cfg(windows)]
    pub(super) fn new_wgl(instance: Arc<GlesInstanceInner>) -> Self {
        Self {
            inner: GlesAdapterInner::Wgl { instance },
        }
    }

    /// Returns the adapter name.
    #[must_use]
    pub fn name(&self) -> &str {
        match &self.inner {
            GlesAdapterInner::Egl { .. } => "yawgpu GLES Adapter (EGL)",
            #[cfg(windows)]
            GlesAdapterInner::Wgl { .. } => "yawgpu GLES Adapter (WGL)",
        }
    }

    /// Returns true when BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc(&self) -> bool {
        false
    }

    /// Returns true when 3D BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        false
    }

    /// Returns true when ETC2/EAC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_etc2(&self) -> bool {
        false
    }

    /// Returns true when ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc(&self) -> bool {
        false
    }

    /// Returns true when 3D ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        false
    }

    /// Returns true when texture view component swizzling is supported.
    #[must_use]
    pub fn supports_texture_component_swizzle(&self) -> bool {
        false
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub(crate) fn supports_shader_float16(&self) -> bool {
        false
    }

    /// Returns true when WGSL `subgroups` is supported.
    #[must_use]
    pub(crate) fn supports_subgroups(&self) -> bool {
        false
    }

    /// Returns true when depth clip control is supported.
    #[must_use]
    pub(crate) fn supports_depth_clip_control(&self) -> bool {
        false
    }

    /// Returns true when float32 color target blending is supported.
    #[must_use]
    pub(crate) fn supports_float32_blendable(&self) -> bool {
        false
    }

    /// Returns true when dual-source blending is supported.
    #[must_use]
    pub(crate) fn supports_dual_source_blending(&self) -> bool {
        false
    }

    /// Returns true when WGSL clip distances are supported.
    #[must_use]
    pub(crate) fn supports_clip_distances(&self) -> bool {
        false
    }

    /// Returns true when WGSL primitive index is supported.
    #[must_use]
    pub(crate) fn supports_primitive_index(&self) -> bool {
        false
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub(crate) fn supports_indirect_first_instance(&self) -> bool {
        false
    }

    /// Returns the supported subgroup size range.
    #[must_use]
    pub(crate) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        None
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<GlesDevice, HalError> {
        match &self.inner {
            GlesAdapterInner::Egl { instance, config } => {
                let GlesInstanceInner::Egl(egl_state) = instance.as_ref() else {
                    return Err(HalError::DeviceCreationFailed { backend: BACKEND });
                };
                create_egl_device(instance, egl_state, *config)
            }
            #[cfg(windows)]
            GlesAdapterInner::Wgl { instance } => {
                let GlesInstanceInner::Wgl(wgl_state) = instance.as_ref() else {
                    return Err(HalError::DeviceCreationFailed { backend: BACKEND });
                };
                let device_state =
                    super::wgl::WglDeviceState::create(Arc::clone(instance), wgl_state)?;
                Ok(GlesDevice::from_wgl(device_state))
            }
        }
    }
}

fn create_egl_device(
    instance: &Arc<GlesInstanceInner>,
    egl_state: &EglInstanceState,
    config: EglConfig,
) -> Result<GlesDevice, HalError> {
    // One-shot EGL display introspection so failures surface what ANGLE /
    // the host EGL stack actually reports. Cheap; called once per device.
    if let Ok(version) = egl_state
        .egl
        .query_string(Some(egl_state.display), egl::VERSION)
    {
        eprintln!("yawgpu-gles: EGL_VERSION={:?}", version.to_string_lossy());
    }
    if let Ok(vendor) = egl_state
        .egl
        .query_string(Some(egl_state.display), egl::VENDOR)
    {
        eprintln!("yawgpu-gles: EGL_VENDOR={:?}", vendor.to_string_lossy());
    }

    // Try ES 3.1 first (CONTEXT_MAJOR_VERSION=3 + CONTEXT_MINOR_VERSION=1).
    // If ANGLE rejects with BadMatch (some configs don't accept MINOR),
    // fall back to ES 3 (CLIENT_VERSION=3, no MINOR). The downstream
    // GL_VERSION check still enforces the >= 3.1 floor.
    let attribs_es31 = [
        egl::CONTEXT_MAJOR_VERSION,
        3,
        egl::CONTEXT_MINOR_VERSION,
        1,
        egl::NONE,
    ];
    let attribs_es3 = [egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = match egl_state
        .egl
        .create_context(egl_state.display, config, None, &attribs_es31)
    {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("yawgpu-gles: eglCreateContext(ES 3.1) failed: {err:?}; retrying with ES 3");
            egl_state
                .egl
                .create_context(egl_state.display, config, None, &attribs_es3)
                .map_err(|err2| {
                    eprintln!("yawgpu-gles: eglCreateContext(ES 3) failed: {err2:?}");
                    HalError::DeviceCreationFailed { backend: BACKEND }
                })?
        }
    };

    let pbuffer_attribs = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
    let surface =
        match egl_state
            .egl
            .create_pbuffer_surface(egl_state.display, config, &pbuffer_attribs)
        {
            Ok(surface) => surface,
            Err(err) => {
                eprintln!("yawgpu-gles: eglCreatePbufferSurface failed: {err:?}");
                destroy_context(egl_state, context);
                return Err(HalError::DeviceCreationFailed { backend: BACKEND });
            }
        };

    if let Err(err) = egl_state.egl.make_current(
        egl_state.display,
        Some(surface),
        Some(surface),
        Some(context),
    ) {
        eprintln!("yawgpu-gles: eglMakeCurrent(pbuffer) failed: {err:?}");
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    }

    let gl = unsafe {
        glow::Context::from_loader_function(|name| {
            egl_state
                .egl
                .get_proc_address(name)
                .map(|proc| proc as *const _)
                .unwrap_or(std::ptr::null())
        })
    };
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };
    let Some((major, minor)) = parse_gles_version(&version) else {
        eprintln!("yawgpu-gles: unable to parse GL_VERSION={version:?}");
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    };
    if (major, minor) < (3, 1) {
        eprintln!(
            "yawgpu-gles: GLES {major}.{minor} below the required 3.1 (GL_VERSION={version:?})"
        );
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    }

    Ok(GlesDevice::from_egl(
        Arc::clone(instance),
        context,
        surface,
        gl,
    ))
}

fn destroy_context(instance: &EglInstanceState, context: EglContext) {
    let _ = instance.egl.destroy_context(instance.display, context);
}

fn destroy_surface(instance: &EglInstanceState, surface: EglSurface) {
    let _ = instance.egl.destroy_surface(instance.display, surface);
}

pub(super) fn parse_gles_version(version: &str) -> Option<(u32, u32)> {
    let rest = version.strip_prefix("OpenGL ES ")?;
    let mut parts = rest.split_whitespace().next()?.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gles_version_accepts_es_versions() {
        assert_eq!(parse_gles_version("OpenGL ES 3.1"), Some((3, 1)));
        assert_eq!(
            parse_gles_version("OpenGL ES 3.2 ANGLE (Vulkan 1.3)"),
            Some((3, 2))
        );
        assert_eq!(parse_gles_version("OpenGL ES 3.0"), Some((3, 0)));
    }

    #[test]
    fn parse_gles_version_rejects_non_es_versions() {
        assert_eq!(parse_gles_version("OpenGL ES-CM 1.1"), None);
        assert_eq!(parse_gles_version(""), None);
        assert_eq!(parse_gles_version("OpenGL 4.5"), None);
    }
}
