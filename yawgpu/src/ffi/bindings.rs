use super::*;

wgpu_handle_exports!(
    refcount_and_label:
    WGPUBindGroupImpl,
    native::WGPUBindGroup,
    "WGPUBindGroup",
    wgpuBindGroupAddRef,
    wgpuBindGroupRelease,
    wgpuBindGroupSetLabel
);

wgpu_handle_exports!(
    refcount_and_label:
    WGPUBindGroupLayoutImpl,
    native::WGPUBindGroupLayout,
    "WGPUBindGroupLayout",
    wgpuBindGroupLayoutAddRef,
    wgpuBindGroupLayoutRelease,
    wgpuBindGroupLayoutSetLabel
);
