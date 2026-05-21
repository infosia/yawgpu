use std::ffi::c_void;
use std::sync::Arc;

use yawgpu_hal::{HalInstance, HalSurface};

use crate::adapter::*;
use crate::error::*;
use crate::future::*;

/// Stores instance data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

/// Holds shared state for the instance handle.
#[derive(Debug)]
pub(crate) struct InstanceInner {
    pub(crate) hal: HalInstance,
    pub(crate) futures: FutureRegistry,
}

impl Instance {
    /// Returns new noop.
    #[must_use]
    pub fn new_noop() -> Self {
        Self::from_hal(HalInstance::new_noop())
    }

    /// Constructs this object from the backend HAL object.
    #[must_use]
    pub fn from_hal(hal: HalInstance) -> Self {
        Self {
            inner: Arc::new(InstanceInner {
                hal,
                futures: FutureRegistry::new(),
            }),
        }
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.enumerate_adapters_with_feature_level(FeatureLevel::Core)
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters_with_feature_level(
        &self,
        feature_level: FeatureLevel,
    ) -> Vec<Adapter> {
        self.inner
            .hal
            .enumerate_adapters()
            .into_iter()
            .map(|hal| Adapter::from_hal_with_feature_level(hal, feature_level))
            .collect()
    }

    /// Returns the future registry used by asynchronous callbacks.
    #[must_use]
    pub fn future_registry(&self) -> &FutureRegistry {
        &self.inner.futures
    }

    /// Returns the HAL.
    #[must_use]
    pub fn hal(&self) -> &HalInstance {
        &self.inner.hal
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn create_surface_from_metal_layer(
        &self,
        layer: *mut c_void,
    ) -> Result<HalSurface, Error> {
        unsafe {
            self.inner
                .hal
                .create_surface_from_metal_layer(layer)
                .map_err(Error::Hal)
        }
    }

    /// # Safety
    ///
    /// `hwnd` must be a valid Win32 window handle and `hinstance` its owning
    /// module handle or null.
    pub unsafe fn create_surface_from_windows_hwnd(
        &self,
        hinstance: *mut c_void,
        hwnd: *mut c_void,
    ) -> Result<HalSurface, Error> {
        unsafe {
            self.inner
                .hal
                .create_surface_from_windows_hwnd(hinstance, hwnd)
                .map_err(Error::Hal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_from_hal_wraps_noop_hal() {
        let instance = Instance::from_hal(yawgpu_hal::HalInstance::new_noop());

        assert_eq!(instance.enumerate_adapters().len(), 1);
    }

    #[test]
    fn instance_enumerate_adapters_with_feature_level_sets_adapter_feature_level() {
        let instance = Instance::new_noop();
        let core = instance.enumerate_adapters_with_feature_level(FeatureLevel::Core);
        let compatibility =
            instance.enumerate_adapters_with_feature_level(FeatureLevel::Compatibility);

        assert_eq!(core.len(), 1);
        assert_eq!(core[0].feature_level(), FeatureLevel::Core);
        assert_eq!(compatibility.len(), 1);
        assert_eq!(
            compatibility[0].feature_level(),
            FeatureLevel::Compatibility
        );
    }

    #[test]
    fn instance_future_registry_process_events_is_empty_without_futures() {
        let instance = Instance::new_noop();

        assert!(instance.future_registry().process_events().is_empty());
    }

    #[test]
    fn instance_hal_returns_noop_hal_instance() {
        let instance = Instance::new_noop();

        assert!(matches!(instance.hal(), yawgpu_hal::HalInstance::Noop(_)));
    }

    #[test]
    fn instance_create_surface_from_metal_layer_noop_returns_noop_surface() {
        let instance = Instance::new_noop();

        let surface = unsafe { instance.create_surface_from_metal_layer(std::ptr::null_mut()) }
            .expect("Noop surface creation should ignore the layer pointer");

        assert!(matches!(surface, yawgpu_hal::HalSurface::Noop));
    }

    #[test]
    fn instance_create_surface_from_windows_hwnd_noop_returns_noop_surface() {
        let instance = Instance::new_noop();
        let hwnd = std::ptr::dangling_mut();

        let surface =
            unsafe { instance.create_surface_from_windows_hwnd(std::ptr::null_mut(), hwnd) }
                .expect("Noop surface creation should ignore HWND pointers");

        assert!(matches!(surface, yawgpu_hal::HalSurface::Noop));
    }
}
