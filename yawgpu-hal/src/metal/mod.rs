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
    MTLDepthStencilState, MTLDevice, MTLDrawable, MTLFunction, MTLGPUFamily, MTLIndexType,
    MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineColorAttachmentDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState,
    MTLScissorRect, MTLSize, MTLStencilDescriptor, MTLStencilOperation, MTLStorageMode,
    MTLStoreAction, MTLTexture as MTLTextureTrait, MTLTextureDescriptor, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureType, MTLTextureUsage, MTLVertexDescriptor,
    MTLVertexFormat, MTLVertexStepFunction, MTLViewport, MTLVisibilityResultMode, MTLWinding,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::{
    HalAddressMode, HalBlendFactor, HalBlendOperation, HalBoundBuffer, HalBoundExternalTexture,
    HalBoundSampler, HalBoundTexture, HalBuffer, HalBufferClear, HalBufferTextureCopy,
    HalBufferUsage, HalColorTargetState, HalCompareFunction, HalComputeDispatch, HalComputePass,
    HalCopy, HalCullMode, HalDepthStencilState, HalDescriptorBinding, HalDraw, HalError,
    HalExtent3d, HalFilterMode, HalFrontFace, HalIndexFormat, HalLimits, HalMipmapFilterMode,
    HalMslBufferSizeBinding, HalPresentMode, HalPrimitiveTopology, HalQueryKind, HalQuerySet,
    HalRenderLoadOp, HalRenderPass, HalRenderPipelineDescriptor, HalResolveQuerySet, HalSampler,
    HalSamplerDescriptor, HalShaderSource, HalStencilFaceState, HalStencilOperation,
    HalSurfaceConfiguration, HalTexture, HalTextureClear, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};
#[cfg(feature = "tiled")]
use crate::{HalSubpassAttachmentResource, HalSubpassRenderPassCommand};

const BACKEND: &str = "metal";
const MAX_VERTEX_BUFFERS: u32 = 8;
const RESERVED_BUFFER_LENGTH_SLOT: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalGpuFamily {
    Apple1,
    Apple2,
    Apple3,
    Apple4,
    Apple5,
    Apple6,
    Apple7,
    Apple8,
    Apple9,
    Mac1,
    Mac2,
}

impl MetalGpuFamily {
    fn is_apple(self) -> bool {
        matches!(
            self,
            Self::Apple1
                | Self::Apple2
                | Self::Apple3
                | Self::Apple4
                | Self::Apple5
                | Self::Apple6
                | Self::Apple7
                | Self::Apple8
                | Self::Apple9
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct MetalDeviceLimits {
    max_vertex_attribs_per_descriptor: u32,
    max_buffer_argument_entries_per_func: u32,
    max_sampler_state_argument_entries_per_func: u32,
    max_threads_per_threadgroup: u32,
    max_total_threadgroup_memory: u32,
    max_fragment_inputs: u32,
    max_fragment_input_components: u32,
    max_1d_texture_size: u32,
    max_2d_texture_size: u32,
    max_3d_texture_size: u32,
    max_texture_array_layers: u32,
    min_buffer_offset_alignment: u32,
    max_color_render_targets: u32,
    max_total_render_target_size: u32,
}

impl MetalDeviceLimits {
    fn for_family(family: MetalGpuFamily) -> Self {
        let index = family as usize;
        Self {
            max_vertex_attribs_per_descriptor: [31; 11][index],
            max_buffer_argument_entries_per_func: [31; 11][index],
            max_sampler_state_argument_entries_per_func: [16; 11][index],
            max_threads_per_threadgroup: [
                512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
            ][index],
            max_total_threadgroup_memory: [
                16_352, 16_352, 16_384, 32_768, 32_768, 32_768, 32_768, 32_768, 32_768, 32_768,
                32_768,
            ][index],
            max_fragment_inputs: [60, 60, 60, 124, 124, 124, 124, 124, 124, 32, 32][index],
            max_fragment_input_components: [60, 60, 60, 124, 124, 124, 124, 124, 124, 124, 124]
                [index],
            max_1d_texture_size: [
                8192, 8192, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384,
            ][index],
            max_2d_texture_size: [
                8192, 8192, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384, 16_384,
            ][index],
            max_3d_texture_size: [2048; 11][index],
            max_texture_array_layers: [2048; 11][index],
            min_buffer_offset_alignment: [4, 4, 4, 4, 4, 4, 4, 4, 4, 256, 256][index],
            max_color_render_targets: [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8][index],
            max_total_render_target_size: [16, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128][index],
        }
    }
}

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

    /// Returns the backend-reported supported limits.
    #[must_use]
    pub(crate) fn limits(&self) -> HalLimits {
        let family = self.gpu_family();
        let mtl = MetalDeviceLimits::for_family(family);
        let mut limits = HalLimits::DEFAULT;

        // Ported from Dawn PhysicalDeviceMTL.mm InitializeSupportedLimitsImpl
        // lines 815-973, including the kMTLLimits table and buffer argument
        // split arithmetic.
        limits.max_texture_dimension_1d = mtl.max_1d_texture_size;
        limits.max_texture_dimension_2d = mtl.max_2d_texture_size;
        limits.max_texture_dimension_3d = mtl.max_3d_texture_size;
        limits.max_texture_array_layers = mtl.max_texture_array_layers;
        limits.max_color_attachments = mtl.max_color_render_targets;
        limits.max_color_attachment_bytes_per_sample = mtl.max_total_render_target_size;

        let max_buffers_per_stage =
            mtl.max_buffer_argument_entries_per_func - RESERVED_BUFFER_LENGTH_SLOT;
        let base_max_buffers_per_stage = limits.max_storage_buffers_per_shader_stage
            + limits.max_uniform_buffers_per_shader_stage
            + MAX_VERTEX_BUFFERS;
        if max_buffers_per_stage > base_max_buffers_per_stage {
            limits.max_storage_buffers_per_shader_stage +=
                max_buffers_per_stage - base_max_buffers_per_stage;
        }

        // yawgpu binds Metal textures via the direct MSL argument table, whose
        // per-stage namespace is 31 slots (see yawgpu-core MAX_TEXTURE_SLOT = 30),
        // not Dawn's argument-buffer budget of maxTextureArgumentEntriesPerFunc.
        // Metal additionally caps read_write (storage) textures at 8 per stage.
        const MAX_TEXTURE_SLOTS: u32 = 31;
        const MAX_READ_WRITE_TEXTURES: u32 = 8;
        limits.max_sampled_textures_per_shader_stage = MAX_TEXTURE_SLOTS;
        limits.max_storage_textures_per_shader_stage = MAX_READ_WRITE_TEXTURES;

        limits.max_samplers_per_shader_stage = mtl.max_sampler_state_argument_entries_per_func;
        limits.max_dynamic_uniform_buffers_per_pipeline_layout = 11;
        limits.max_dynamic_storage_buffers_per_pipeline_layout = 11;
        limits.max_vertex_attributes =
            limits.max_vertex_buffers * mtl.max_vertex_attribs_per_descriptor;
        limits.max_inter_stage_shader_variables = if family.is_apple() {
            mtl.max_fragment_inputs
                .min(mtl.max_fragment_input_components / 4)
        } else {
            mtl.max_fragment_inputs.saturating_sub(4)
        };
        limits.max_compute_workgroup_storage_size = mtl.max_total_threadgroup_memory;
        limits.max_compute_invocations_per_workgroup = mtl.max_threads_per_threadgroup;
        limits.max_compute_workgroup_size_x = mtl.max_threads_per_threadgroup;
        limits.max_compute_workgroup_size_y = mtl.max_threads_per_threadgroup;
        limits.max_compute_workgroup_size_z = mtl.max_threads_per_threadgroup;
        limits.min_uniform_buffer_offset_alignment = mtl.min_buffer_offset_alignment;
        limits.min_storage_buffer_offset_alignment = mtl.min_buffer_offset_alignment;

        let max_buffer_length = self.device.maxBufferLength() as u64;
        let max_binding_size = max_buffer_length.min(u64::from(u32::MAX));
        limits.max_buffer_size = max_buffer_length;
        limits.max_uniform_buffer_binding_size = max_binding_size;
        limits.max_storage_buffer_binding_size = max_binding_size;

        limits.max_storage_buffers_in_fragment_stage = limits.max_storage_buffers_per_shader_stage;
        limits.max_storage_textures_in_fragment_stage =
            limits.max_storage_textures_per_shader_stage;
        limits.max_storage_buffers_in_vertex_stage = limits.max_storage_buffers_per_shader_stage;
        limits.max_storage_textures_in_vertex_stage = limits.max_storage_textures_per_shader_stage;

        limits
    }

    fn gpu_family(&self) -> MetalGpuFamily {
        if self.device.supportsFamily(MTLGPUFamily::Apple9) {
            MetalGpuFamily::Apple9
        } else if self.device.supportsFamily(MTLGPUFamily::Apple8) {
            MetalGpuFamily::Apple8
        } else if self.device.supportsFamily(MTLGPUFamily::Apple7) {
            MetalGpuFamily::Apple7
        } else if self.device.supportsFamily(MTLGPUFamily::Apple6) {
            MetalGpuFamily::Apple6
        } else if self.device.supportsFamily(MTLGPUFamily::Apple5) {
            MetalGpuFamily::Apple5
        } else if self.device.supportsFamily(MTLGPUFamily::Apple4) {
            MetalGpuFamily::Apple4
        } else if self.device.supportsFamily(MTLGPUFamily::Apple3) {
            MetalGpuFamily::Apple3
        } else if self.device.supportsFamily(MTLGPUFamily::Apple2) {
            MetalGpuFamily::Apple2
        } else if self.device.supportsFamily(MTLGPUFamily::Apple1) {
            MetalGpuFamily::Apple1
        } else if self.device.supportsFamily(MTLGPUFamily::Mac2) {
            MetalGpuFamily::Mac2
        } else {
            MetalGpuFamily::Mac1
        }
    }

    /// Returns the IORegistry ID for this adapter's Metal device.
    #[must_use]
    pub fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    /// Returns true when BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc(&self) -> bool {
        self.device.supportsBCTextureCompression()
    }

    /// Returns true when 3D BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        self.device.supportsBCTextureCompression()
    }

    /// Returns true when ETC2/EAC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_etc2(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Apple2)
    }

    /// Returns true when ASTC LDR texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Apple3)
    }

    /// Returns true when 3D ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Apple3)
    }

    /// Returns true when texture view component swizzling is supported.
    #[must_use]
    pub fn supports_texture_component_swizzle(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Mac2)
            || self.device.supportsFamily(MTLGPUFamily::Apple2)
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub(super) fn supports_shader_float16(&self) -> bool {
        true
    }

    /// Returns true when WGSL `subgroups` is supported.
    #[must_use]
    pub(super) fn supports_subgroups(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Apple6)
            || self.device.supportsFamily(MTLGPUFamily::Metal3)
    }

    /// Returns true when depth clip control is supported.
    #[must_use]
    pub(super) fn supports_depth_clip_control(&self) -> bool {
        true
    }

    /// Returns true when float32 color target blending is supported.
    #[must_use]
    pub(super) fn supports_float32_blendable(&self) -> bool {
        true
    }

    /// Returns true when dual-source blending is supported.
    #[must_use]
    pub(super) fn supports_dual_source_blending(&self) -> bool {
        true
    }

    /// Returns true when WGSL clip distances are supported.
    #[must_use]
    pub(super) fn supports_clip_distances(&self) -> bool {
        true
    }

    /// Returns true when WGSL primitive index is supported.
    #[must_use]
    pub(super) fn supports_primitive_index(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Apple7)
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub(super) fn supports_indirect_first_instance(&self) -> bool {
        true
    }

    /// Returns the supported subgroup size range.
    #[must_use]
    pub(super) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        self.supports_subgroups().then_some((32, 32))
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
mod query_set;
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
pub use query_set::MetalQuerySet;
pub use queue::MetalQueue;
pub use surface::MetalSurface;
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
    fn metal_adapter_limits_reports_real_device_limits() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");
        let limits = adapter.limits();

        assert!(limits.max_texture_dimension_2d >= 8192);
        assert!(limits.max_compute_invocations_per_workgroup >= 256);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_adapter_supports_texture_component_swizzle_matches_supported_families() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");

        assert_eq!(
            adapter.supports_texture_component_swizzle(),
            adapter.device.supportsFamily(MTLGPUFamily::Mac2)
                || adapter.device.supportsFamily(MTLGPUFamily::Apple2)
        );
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_adapter_supports_astc_compression_on_apple8_m2() {
        let adapter = MetalInstance::new()
            .expect("create Metal instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Metal adapter");

        assert!(adapter.device.supportsFamily(MTLGPUFamily::Apple8));
        assert!(adapter.supports_texture_compression_astc());
        assert!(adapter.supports_texture_compression_astc_sliced_3d());
        assert_eq!(
            adapter.supports_texture_compression_bc_sliced_3d(),
            adapter.supports_texture_compression_bc()
        );
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
