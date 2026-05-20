use super::*;
use crate::{HalPresentMode, HalRenderPipelineDescriptor, HalSamplerDescriptor, HalTextureUsage};

pub(crate) fn metal_device() -> MetalDevice {
    let instance = MetalInstance::new().expect("create Metal instance");
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("at least one Metal adapter");
    adapter.create_device().expect("create Metal device")
}

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

pub(crate) fn texture_usage() -> HalTextureUsage {
    HalTextureUsage {
        copy_src: true,
        copy_dst: true,
        texture_binding: true,
        storage_binding: false,
        render_attachment: true,
    }
}

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

pub(crate) fn surface_config() -> HalSurfaceConfiguration {
    HalSurfaceConfiguration::new(
        HalTextureFormat::Rgba8Unorm,
        texture_usage(),
        100,
        100,
        HalPresentMode::Fifo,
    )
}

pub(crate) fn render_descriptor() -> HalRenderPipelineDescriptor {
    HalRenderPipelineDescriptor {
        color_formats: vec![HalTextureFormat::Rgba8Unorm],
        vertex_buffers: Vec::new(),
        primitive_topology: HalPrimitiveTopology::TriangleList,
    }
}

pub(crate) fn compute_msl() -> HalShaderSource {
    HalShaderSource::Msl(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void main0() {}
"#
        .to_owned(),
    )
}

pub(crate) fn render_msl() -> HalShaderSource {
    HalShaderSource::Msl(
        r#"
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut vs_main(uint vertex_id [[vertex_id]]) {
    VertexOut out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    return out;
}
fragment float4 fs_main() { return float4(1.0, 0.0, 0.0, 1.0); }
"#
        .to_owned(),
    )
}

pub(crate) fn metal_layer() -> Retained<CAMetalLayer> {
    CAMetalLayer::layer()
}
