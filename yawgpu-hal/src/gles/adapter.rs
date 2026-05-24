use std::sync::Arc;

use glow::HasContext;
use khronos_egl as egl;

use super::device::GlesDevice;
use super::egl::{EglConfig, EglContext, EglSurface};
use super::instance::GlesInstanceInner;
use super::BACKEND;
use crate::HalError;

/// Stores GLES adapter data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesAdapter {
    instance: Arc<GlesInstanceInner>,
    config: EglConfig,
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
    pub(super) fn new(instance: Arc<GlesInstanceInner>, config: EglConfig) -> Self {
        Self { instance, config }
    }

    /// Returns the adapter name.
    #[must_use]
    pub fn name(&self) -> &str {
        "yawgpu GLES Adapter"
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<GlesDevice, HalError> {
        // One-shot EGL display introspection so failures surface what ANGLE /
        // the host EGL stack actually reports. Cheap; called once per device.
        if let Ok(version) = self
            .instance
            .egl
            .query_string(Some(self.instance.display), egl::VERSION)
        {
            eprintln!(
                "yawgpu-gles: EGL_VERSION={:?}",
                version.to_string_lossy()
            );
        }
        if let Ok(vendor) = self
            .instance
            .egl
            .query_string(Some(self.instance.display), egl::VENDOR)
        {
            eprintln!(
                "yawgpu-gles: EGL_VENDOR={:?}",
                vendor.to_string_lossy()
            );
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
        let context = match self.instance.egl.create_context(
            self.instance.display,
            self.config,
            None,
            &attribs_es31,
        ) {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!(
                    "yawgpu-gles: eglCreateContext(ES 3.1) failed: {err:?}; retrying with ES 3"
                );
                self.instance
                    .egl
                    .create_context(self.instance.display, self.config, None, &attribs_es3)
                    .map_err(|err2| {
                        eprintln!("yawgpu-gles: eglCreateContext(ES 3) failed: {err2:?}");
                        HalError::DeviceCreationFailed { backend: BACKEND }
                    })?
            }
        };

        let pbuffer_attribs = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
        let surface = match self.instance.egl.create_pbuffer_surface(
            self.instance.display,
            self.config,
            &pbuffer_attribs,
        ) {
            Ok(surface) => surface,
            Err(err) => {
                eprintln!("yawgpu-gles: eglCreatePbufferSurface failed: {err:?}");
                destroy_context(&self.instance, context);
                return Err(HalError::DeviceCreationFailed { backend: BACKEND });
            }
        };

        if let Err(err) = self.instance.egl.make_current(
            self.instance.display,
            Some(surface),
            Some(surface),
            Some(context),
        ) {
            eprintln!("yawgpu-gles: eglMakeCurrent(pbuffer) failed: {err:?}");
            destroy_surface(&self.instance, surface);
            destroy_context(&self.instance, context);
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                self.instance
                    .egl
                    .get_proc_address(name)
                    .map(|proc| proc as *const _)
                    .unwrap_or(std::ptr::null())
            })
        };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        let Some((major, minor)) = parse_gles_version(&version) else {
            eprintln!("yawgpu-gles: unable to parse GL_VERSION={version:?}");
            destroy_surface(&self.instance, surface);
            destroy_context(&self.instance, context);
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        };
        if (major, minor) < (3, 1) {
            eprintln!("yawgpu-gles: GLES {major}.{minor} below the required 3.1 (GL_VERSION={version:?})");
            destroy_surface(&self.instance, surface);
            destroy_context(&self.instance, context);
            return Err(HalError::DeviceCreationFailed { backend: BACKEND });
        }

        Ok(GlesDevice::from_parts(
            Arc::clone(&self.instance),
            context,
            surface,
            gl,
        ))
    }
}

fn destroy_context(instance: &GlesInstanceInner, context: EglContext) {
    let _ = instance.egl.destroy_context(instance.display, context);
}

fn destroy_surface(instance: &GlesInstanceInner, surface: EglSurface) {
    let _ = instance.egl.destroy_surface(instance.display, surface);
}

fn parse_gles_version(version: &str) -> Option<(u32, u32)> {
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
