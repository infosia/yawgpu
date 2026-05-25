use super::*;
use crate::{HalPresentMode, HalRenderPipelineDescriptor, HalSamplerDescriptor, HalTextureUsage};

/// Returns vulkan device.
pub(crate) fn vulkan_device() -> VulkanDevice {
    let instance = VulkanInstance::new().expect("create Vulkan instance");
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("at least one Vulkan adapter");
    adapter.create_device().expect("create Vulkan device")
}

/// Returns texture usage.
pub(crate) fn texture_usage() -> HalTextureUsage {
    HalTextureUsage {
        copy_src: true,
        copy_dst: true,
        texture_binding: true,
        storage_binding: false,
        render_attachment: true,
    }
}

/// Returns texture descriptor.
pub(crate) fn texture_descriptor() -> HalTextureDescriptor {
    HalTextureDescriptor {
        format: HalTextureFormat::Rgba8Unorm,
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
        mip_level_count: 1,
        sample_count: 1,
        usage: texture_usage(),
    }
}

/// Returns sampler descriptor.
pub(crate) fn sampler_descriptor() -> HalSamplerDescriptor {
    HalSamplerDescriptor {
        address_mode_u: HalAddressMode::ClampToEdge,
        address_mode_v: HalAddressMode::ClampToEdge,
        address_mode_w: HalAddressMode::ClampToEdge,
        mag_filter: HalFilterMode::Linear,
        min_filter: HalFilterMode::Linear,
        mipmap_filter: HalMipmapFilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        max_anisotropy: 1,
    }
}

/// Returns render descriptor.
pub(crate) fn render_descriptor() -> HalRenderPipelineDescriptor {
    HalRenderPipelineDescriptor {
        color_formats: vec![HalTextureFormat::Rgba8Unorm],
        depth_stencil: None,
        vertex_buffers: Vec::new(),
        primitive_topology: HalPrimitiveTopology::TriangleList,
    }
}

/// Returns surface config.
pub(crate) fn surface_config() -> HalSurfaceConfiguration {
    HalSurfaceConfiguration::new(
        HalTextureFormat::Bgra8Unorm,
        texture_usage(),
        4,
        4,
        HalPresentMode::Fifo,
    )
}

/// Returns dummy surface.
pub(crate) fn dummy_surface(instance: &VulkanInstance) -> VulkanSurface {
    let surface = vk::SurfaceKHR::null();
    VulkanSurface {
        surface,
        surface_inner: Arc::new(VulkanSurfaceInner::new(
            Arc::clone(&instance.inner),
            surface,
        )),
        swapchain: None,
        config: None,
        current_image_index: None,
        pending_state: Arc::new(Mutex::new(SurfacePendingState::new())),
        image_acquired_semaphores: Vec::new(),
        render_finished_semaphores: Vec::new(),
        present_ready_semaphores: Vec::new(),
        in_flight_fences: Vec::new(),
        next_sync_index: 0,
    }
}

/// Returns compute spirv.
pub(crate) fn compute_spirv() -> Vec<u32> {
    vec![
        119734787, 65536, 524299, 10, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134,
        0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1, 196611, 2, 450,
        262149, 4, 1852399981, 0, 262215, 9, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0,
        262167, 7, 6, 3, 262187, 6, 8, 1, 393260, 7, 9, 8, 8, 8, 327734, 2, 4, 0, 3, 131320, 5,
        65789, 65592,
    ]
}

/// Returns vertex spirv.
pub(crate) fn vertex_spirv() -> Vec<u32> {
    vec![
        119734787, 65536, 524299, 21, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134,
        0, 196622, 0, 1, 393231, 0, 4, 1852399981, 0, 13, 196611, 2, 450, 262149, 4, 1852399981, 0,
        393221, 11, 1348430951, 1700164197, 2019914866, 0, 393222, 11, 0, 1348430951, 1953067887,
        7237481, 458758, 11, 1, 1348430951, 1953393007, 1702521171, 0, 458758, 11, 2, 1130327143,
        1148217708, 1635021673, 6644590, 458758, 11, 3, 1130327143, 1147956341, 1635021673,
        6644590, 196613, 13, 0, 196679, 11, 2, 327752, 11, 0, 11, 0, 327752, 11, 1, 11, 1, 327752,
        11, 2, 11, 3, 327752, 11, 3, 11, 4, 131091, 2, 196641, 3, 2, 196630, 6, 32, 262167, 7, 6,
        4, 262165, 8, 32, 0, 262187, 8, 9, 1, 262172, 10, 6, 9, 393246, 11, 7, 6, 10, 10, 262176,
        12, 3, 11, 262203, 12, 13, 3, 262165, 14, 32, 1, 262187, 14, 15, 0, 262187, 6, 16, 0,
        262187, 6, 17, 1065353216, 458796, 7, 18, 16, 16, 16, 17, 262176, 19, 3, 7, 327734, 2, 4,
        0, 3, 131320, 5, 327745, 19, 20, 13, 15, 196670, 20, 18, 65789, 65592,
    ]
}

/// Returns fragment spirv.
pub(crate) fn fragment_spirv() -> Vec<u32> {
    vec![
        119734787, 65536, 524299, 13, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134,
        0, 196622, 0, 1, 393231, 4, 4, 1852399981, 0, 9, 196624, 4, 7, 196611, 2, 450, 262149, 4,
        1852399981, 0, 327685, 9, 1131705711, 1919904879, 0, 262215, 9, 30, 0, 131091, 2, 196641,
        3, 2, 196630, 6, 32, 262167, 7, 6, 4, 262176, 8, 3, 7, 262203, 8, 9, 3, 262187, 6, 10,
        1065353216, 262187, 6, 11, 0, 458796, 7, 12, 10, 11, 11, 10, 327734, 2, 4, 0, 3, 131320, 5,
        196670, 9, 12, 65789, 65592,
    ]
}
