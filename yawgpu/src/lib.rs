macro_rules! declare_impl_handles {
    ($($name:ident),* $(,)?) => {
        $(
            pub struct $name;
        )*
    };
}

declare_impl_handles!(
    WGPUAdapterImpl,
    WGPUBindGroupImpl,
    WGPUBindGroupLayoutImpl,
    WGPUBufferImpl,
    WGPUCommandBufferImpl,
    WGPUCommandEncoderImpl,
    WGPUComputePassEncoderImpl,
    WGPUComputePipelineImpl,
    WGPUDeviceImpl,
    WGPUInstanceImpl,
    WGPUPipelineLayoutImpl,
    WGPUQuerySetImpl,
    WGPUQueueImpl,
    WGPURenderBundleImpl,
    WGPURenderBundleEncoderImpl,
    WGPURenderPassEncoderImpl,
    WGPURenderPipelineImpl,
    WGPUSamplerImpl,
    WGPUShaderModuleImpl,
    WGPUSurfaceImpl,
    WGPUTextureImpl,
    WGPUTextureViewImpl,
);

pub mod native {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
