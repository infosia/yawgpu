use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::{NSArray, NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBlitOption,
    MTLBuffer as MTLBufferTrait, MTLClearColor, MTLColorWriteMask, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthClipMode, MTLDepthStencilDescriptor,
    MTLDepthStencilState, MTLDevice, MTLDrawable, MTLFunction, MTLIndexType, MTLLibrary,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineColorAttachmentDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState,
    MTLScissorRect, MTLSize, MTLStencilDescriptor, MTLStencilOperation, MTLStorageMode,
    MTLStoreAction, MTLTexture as MTLTextureTrait, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction, MTLViewport,
    MTLWinding,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::{
    HalAddressMode, HalBlendFactor, HalBlendOperation, HalBoundBuffer, HalBoundSampler,
    HalBoundTexture, HalBuffer, HalBufferClear, HalBufferTextureCopy, HalBufferUsage,
    HalColorTargetState, HalCompareFunction, HalComputeDispatch, HalComputePass, HalCopy,
    HalCullMode, HalDepthStencilState, HalDescriptorBinding, HalDraw, HalError, HalExtent3d,
    HalFilterMode, HalFrontFace, HalIndexFormat, HalMipmapFilterMode, HalMslBufferSizeBinding,
    HalPrimitiveTopology, HalRenderLoadOp, HalRenderPass, HalRenderPipelineDescriptor, HalSampler,
    HalSamplerDescriptor, HalShaderSource, HalStencilFaceState, HalStencilOperation,
    HalSurfaceConfiguration, HalTexture, HalTextureCopy, HalTextureDescriptor, HalTextureFormat,
    HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};
#[cfg(feature = "tiled")]
use crate::{
    HalRenderPipeline, HalSubpassAttachmentResource, HalSubpassDraw, HalSubpassPassLayout,
    HalSubpassRenderPassCommand, HalTransientAttachment, HalTransientAttachmentDescriptor,
};

const BACKEND: &str = "metal";

/// Stores metal instance data used by validation and backend submission.
pub struct MetalInstance;

impl std::fmt::Debug for MetalInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalInstance").finish()
    }
}

impl MetalInstance {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        Ok(Self)
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<MetalAdapter> {
        autoreleasepool(|_| {
            let mut adapters = Vec::new();
            if let Some(device) = MTLCreateSystemDefaultDevice() {
                adapters.push(MetalAdapter::new(device));
            }

            let devices: Retained<NSArray<ProtocolObject<dyn MTLDevice>>> = MTLCopyAllDevices();
            for device in devices {
                let registry_id = device.registryID();
                if adapters
                    .iter()
                    .any(|adapter: &MetalAdapter| adapter.registry_id() == registry_id)
                {
                    continue;
                }
                adapters.push(MetalAdapter::new(device));
            }
            adapters
        })
    }
}

/// Stores metal adapter data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalAdapter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    name: String,
}

impl std::fmt::Debug for MetalAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalAdapter")
            .field("name", &self.name)
            .finish()
    }
}

impl MetalAdapter {
    /// Creates a new instance.
    #[must_use]
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let name = device.name().to_string();
        Self { device, name }
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the IORegistry ID for this adapter's Metal device.
    #[must_use]
    pub fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<MetalDevice, HalError> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or(HalError::DeviceCreationFailed { backend: BACKEND })?;
        Ok(MetalDevice {
            device: self.device.clone(),
            allocations: AtomicU64::new(0),
            queue: MetalQueue { inner: queue },
        })
    }
}

mod buffer;
mod device;
mod encode;
mod format;
mod pipeline;
mod queue;
mod surface;
use self::encode::*;
use self::format::*;
use self::pipeline::*;
use self::texture::*;
#[cfg(test)]
mod test_helpers;
mod texture;

pub use buffer::MetalBuffer;
pub use device::MetalDevice;
pub use pipeline::{MetalComputePipeline, MetalRenderPipeline};
pub use queue::MetalQueue;
pub use surface::MetalSurface;
#[cfg(feature = "tiled")]
pub use texture::MetalTransientAttachment;
pub use texture::{MetalSampler, MetalTexture};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_instance_new_constructs() {
        MetalInstance::new().expect("create Metal instance");
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_instance_enumerate_adapters_returns_devices() {
        let adapters = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters();
        assert!(!adapters.is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_enumerate_adapters_returns_dedup_set_with_registry_id() {
        let adapters = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters();
        assert!(!adapters.is_empty());

        let mut registry_ids = std::collections::BTreeSet::new();
        for adapter in &adapters {
            assert!(
                registry_ids.insert(adapter.registry_id()),
                "duplicate Metal registry ID"
            );
        }
        assert_ne!(adapters[0].registry_id(), 0);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_adapter_new_captures_device_name() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");
        let rebuilt = MetalAdapter::new(adapter.device.clone());
        assert_eq!(rebuilt.name(), adapter.name());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_adapter_name_returns_non_empty_name() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");
        assert!(!adapter.name().is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_adapter_create_device_returns_zero_allocation_device() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");
        let device = adapter.create_device().expect("create Metal device");
        assert_eq!(device.allocation_count(), 0);
    }
}
