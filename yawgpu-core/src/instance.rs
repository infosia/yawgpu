use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pass::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::*;
use crate::error::*;
use crate::extent::*;
use crate::format::*;
use crate::future::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

#[derive(Debug)]
pub(crate) struct InstanceInner {
    pub(crate) hal: HalInstance,
    pub(crate) futures: FutureRegistry,
}

impl Instance {
    #[must_use]
    pub fn new_noop() -> Self {
        Self::from_hal(HalInstance::new_noop())
    }

    #[must_use]
    pub fn from_hal(hal: HalInstance) -> Self {
        Self {
            inner: Arc::new(InstanceInner {
                hal,
                futures: FutureRegistry::new(),
            }),
        }
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.enumerate_adapters_with_feature_level(FeatureLevel::Core)
    }

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

    #[must_use]
    pub fn future_registry(&self) -> &FutureRegistry {
        &self.inner.futures
    }

    #[must_use]
    pub fn hal(&self) -> &HalInstance {
        &self.inner.hal
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn create_surface_from_metal_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> Result<HalSurface, Error> {
        unsafe {
            self.inner
                .hal
                .create_surface_from_metal_layer(layer)
                .map_err(Error::Hal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
}
