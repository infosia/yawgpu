use khronos_egl as egl;

use super::BACKEND;
use crate::HalError;

pub(super) type EglInstance = egl::DynamicInstance<egl::EGL1_4>;
pub(super) type EglDisplay = egl::Display;
pub(super) type EglConfig = egl::Config;
pub(super) type EglContext = egl::Context;
pub(super) type EglSurface = egl::Surface;

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

    loaded.map_err(|_| HalError::BackendUnavailable { backend: BACKEND })
}
