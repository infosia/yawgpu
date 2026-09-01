use super::*;

wgpu_handle_exports!(
    refcount_and_label:
    WGPUCommandBufferImpl,
    native::WGPUCommandBuffer,
    "WGPUCommandBuffer",
    wgpuCommandBufferAddRef,
    wgpuCommandBufferRelease,
    wgpuCommandBufferSetLabel
);
