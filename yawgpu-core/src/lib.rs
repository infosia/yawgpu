#![warn(missing_docs)]
//! Core WebGPU validation and object model used by the C API.

mod adapter;
mod bind_group;
mod bind_group_layout;
mod buffer;
mod command_encoder;
mod compute_pass;
mod compute_pipeline;
mod copy;
mod device;
mod error;
mod extent;
mod external_texture;
mod format;
mod future;
mod instance;
mod limits;
mod pass;
mod pipeline_id;
mod pipeline_layout;
mod query_set;
mod queue;
mod render_bundle;
mod render_pass;
mod render_pipeline;
mod sampler;
mod shader;
mod shader_tint;
mod shader_types;
#[cfg(test)]
mod test_helpers;
mod texture;
mod texture_view;
mod wgsl_language_features;

pub(crate) use crate::shader_tint as frontend;

pub use adapter::{Adapter, Feature, FeatureLevel};
pub use bind_group::{BindGroup, BindGroupEntry, BindGroupResource};
pub use bind_group_layout::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingLayoutKind,
    BufferBindingType, SamplerBindingType, StorageTextureAccess, TextureSampleType,
};
pub use buffer::{
    Buffer, BufferDescriptor, BufferMapState, BufferUsage, MapAsyncStatus, MapMode,
    QueueWorkDoneStatus,
};
pub use command_encoder::{validate_compute_pass_timestamp_writes, CommandBuffer, CommandEncoder};
pub use compute_pass::ComputePassEncoder;
pub use compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptor, ComputePipelineLayout, PipelineConstant,
};
pub use copy::{
    Color, LoadOp, StoreOp, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
};
pub use device::{Device, DeviceLostReason, FeatureSet};
pub use error::{DeviceError, Error, ErrorFilter, ErrorKind, PopErrorScopeError};
pub use extent::{Extent3d, Origin3d};
pub use external_texture::{
    ExternalTexture, ExternalTextureDescriptor, ExternalTextureFormat, ExternalTextureParams,
    ExternalTextureRotation, Origin2d,
};
pub use format::{FormatAspects, FormatCaps, FormatOutputClass, TextureFormat};
pub use future::{FutureCallbackMode, FutureId, FutureRegistry, WaitAnyResult, WaitAnyStatus};
pub use instance::Instance;
pub use limits::Limits;
pub use pipeline_layout::{PipelineLayout, PipelineLayoutDescriptor};
pub use query_set::{QuerySet, QuerySetDescriptor, QueryType};
pub use queue::{Queue, QueueBufferWrite, QueueTextureWrite};
pub use render_bundle::{RenderBundle, RenderBundleEncoder, RenderBundleEncoderDescriptor};
pub use render_pass::{
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPassEncoder, RenderPassTimestampWrites,
};
pub use render_pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, CullMode,
    DepthStencilState, FrontFace, IndexFormat, MultisampleState, PrimitiveState, PrimitiveTopology,
    RenderPipeline, RenderPipelineDescriptor, RenderPipelineFragmentState, RenderPipelineLayout,
    RenderPipelineShaderStage, RenderPipelineVertexState, StencilFaceState, StencilOperation,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
pub use sampler::{
    AddressMode, CompareFunction, FilterMode, MipmapFilterMode, ResolvedSamplerDescriptor, Sampler,
    SamplerDescriptor,
};
pub use shader::{CompilationMessage, CompilationSeverity, ShaderModule, ShaderModuleSource};
pub use shader_tint::ReflectedModule;
pub use texture::{Texture, TextureDescriptor, TextureDimension, TextureUsage};
pub use texture_view::{
    ComponentSwizzle, TextureAspect, TextureComponentSwizzle, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};
pub use wgsl_language_features::{
    SUPPORTED_WGSL_LANGUAGE_FEATURES, WGSL_LANGUAGE_FEATURE_LINEAR_INDEXING,
    WGSL_LANGUAGE_FEATURE_PACKED_4X8_INTEGER_DOT_PRODUCT,
    WGSL_LANGUAGE_FEATURE_POINTER_COMPOSITE_ACCESS,
    WGSL_LANGUAGE_FEATURE_READONLY_AND_READWRITE_STORAGE_TEXTURES,
    WGSL_LANGUAGE_FEATURE_TEXTURE_AND_SAMPLER_LET, WGSL_LANGUAGE_FEATURE_TEXTURE_FORMATS_TIER1,
    WGSL_LANGUAGE_FEATURE_UNIFORM_BUFFER_STANDARD_LAYOUT,
    WGSL_LANGUAGE_FEATURE_UNRESTRICTED_POINTER_PARAMETERS,
};

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn creates_noop_device_and_queue() {
        let instance = Instance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0]
            .create_device(None, &[], "", "")
            .expect("Noop device should be created");
        assert_eq!(device.allocation_count(), 0);

        let _queue = device.queue();
    }
}
