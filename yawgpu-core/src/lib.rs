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
mod format;
mod future;
mod instance;
mod limits;
mod pass;
mod pipeline_layout;
mod query_set;
mod queue;
mod render_bundle;
mod render_pass;
mod render_pipeline;
mod sampler;
mod shader;
#[cfg(feature = "tiled")]
mod subpass;
#[cfg(test)]
mod test_helpers;
mod texture;
mod texture_view;
#[cfg(feature = "tiled")]
mod transient_attachment;

/// Shader naga module.
pub(crate) mod shader_naga;

#[cfg(feature = "tiled")]
pub use adapter::TiledCapabilities;
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
pub use command_encoder::{CommandBuffer, CommandEncoder};
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
pub use format::{FormatAspects, FormatCaps, FormatOutputClass, TextureFormat};
pub use future::{FutureCallbackMode, FutureId, FutureRegistry, WaitAnyResult, WaitAnyStatus};
pub use instance::Instance;
pub use limits::Limits;
pub use pipeline_layout::{PipelineLayout, PipelineLayoutDescriptor};
pub use query_set::{QuerySet, QuerySetDescriptor, QueryType};
pub use queue::Queue;
pub use render_bundle::{RenderBundle, RenderBundleEncoder, RenderBundleEncoderDescriptor};
pub use render_pass::{
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPassEncoder, RenderPassTimestampWrites,
};
#[cfg(feature = "tiled")]
pub use render_pipeline::SubpassRenderPipelineDescriptor;
pub use render_pipeline::{
    ColorTargetState, DepthStencilState, IndexFormat, MultisampleState, PrimitiveState,
    PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor, RenderPipelineFragmentState,
    RenderPipelineLayout, RenderPipelineShaderStage, RenderPipelineVertexState, StencilFaceState,
    StencilOperation, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
pub use sampler::{
    AddressMode, CompareFunction, FilterMode, MipmapFilterMode, ResolvedSamplerDescriptor, Sampler,
    SamplerDescriptor,
};
#[cfg(feature = "shader-passthrough")]
pub use shader::{MslEntryPoint, MslReflection};
pub use shader::{ShaderModule, ShaderModuleSource};
#[cfg(feature = "shader-passthrough")]
pub use shader_naga::ReflectedModule;
#[cfg(feature = "tiled")]
pub use subpass::{
    AttachmentLayout, SubpassAttachmentResource, SubpassColorAttachmentBinding, SubpassDependency,
    SubpassDependencyType, SubpassDepthStencilAttachmentBinding, SubpassInputAttachment,
    SubpassLayoutDesc, SubpassPassLayout, SubpassPassLayoutDescriptor, SubpassRenderPass,
    SubpassRenderPassDescriptor, DEPTH_STENCIL_ATTACHMENT_INDEX,
};
pub use texture::{Texture, TextureDescriptor, TextureDimension, TextureUsage};
pub use texture_view::{TextureAspect, TextureView, TextureViewDescriptor, TextureViewDimension};
#[cfg(feature = "tiled")]
pub use transient_attachment::{
    TransientAttachment, TransientAttachmentDescriptor, TransientSizeMode,
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
