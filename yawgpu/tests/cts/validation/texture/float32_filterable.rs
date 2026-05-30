use yawgpu::native;

use crate::{common, feature_common};

#[test]
fn create_bind_group() {
    feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_Float32Filterable);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_Float32Filterable);
    unsafe {
        let texture = common::create_texture(
            test.device(),
            native::WGPUTextureFormat_R32Float,
            native::WGPUTextureUsage_TextureBinding,
            4,
            4,
        );
        let view = common::create_texture_view(texture);
        let entry = texture_entry(native::WGPUTextureSampleType_Float);
        let layout = common::create_bind_group_layout(test.device(), &[entry]);
        let group_entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: view,
        };
        let group = common::create_bind_group(test.device(), layout, &[group_entry]);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

fn texture_entry(sample_type: native::WGPUTextureSampleType) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: sample_type,
            viewDimension: native::WGPUTextureViewDimension_2D,
            multisampled: 0,
        },
        ..feature_common::storage_texture_entry(
            native::WGPUTextureFormat_Undefined,
            native::WGPUStorageTextureAccess_BindingNotUsed,
        )
    }
}
