use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, create_bind_group_layout,
    create_pipeline_layout, uniform_layout,
};

#[test]
fn pipeline_layout_shader_exact_match() {
    let test = ValidationTest::new();
    unsafe {
        compute_resource_cases(&test);
    }
}

unsafe fn compute_resource_cases(test: &ValidationTest) {
    let cases = [
        (
            vec![uniform_layout(0, native::WGPUShaderStage_Compute, 16)],
            "@group(0) @binding(0) var<uniform> u: vec4<f32>;
                 @compute @workgroup_size(1) fn main() { _ = u; }",
            true,
        ),
        (
            vec![],
            "@group(0) @binding(0) var<uniform> u: vec4<f32>;
                 @compute @workgroup_size(1) fn main() { _ = u; }",
            false,
        ),
        (
            vec![uniform_layout(0, native::WGPUShaderStage_Fragment, 16)],
            "@group(0) @binding(0) var<uniform> u: vec4<f32>;
                 @compute @workgroup_size(1) fn main() { _ = u; }",
            false,
        ),
        (
            vec![uniform_layout(1, native::WGPUShaderStage_Compute, 16)],
            "@group(0) @binding(0) var<uniform> u: vec4<f32>;
                 @compute @workgroup_size(1) fn main() { _ = u; }",
            false,
        ),
    ];

    for (entries, source, success) in cases {
        let bgl = create_bind_group_layout(test.device(), &entries);
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        if success {
            assert_compute_pipeline_ok(test, false, source, Some("main"), &[], Some(layout));
        } else {
            assert_compute_pipeline_error(test, false, source, Some("main"), &[], Some(layout));
        }
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

#[test]
fn pipeline_layout_shader_exact_match_full_matrix_gaps() {
    let test = ValidationTest::new();
    unsafe {
        let bgl = create_bind_group_layout(
            test.device(),
            &[storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Sint,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        assert_compute_pipeline_error(
            &test,
            false,
            "@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
             @compute @workgroup_size(1) fn main() { textureStore(tex, vec2i(0), vec4f()); }",
            Some("main"),
            &[],
            Some(layout),
        );
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

fn storage_texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    access: native::WGPUStorageTextureAccess,
    format: native::WGPUTextureFormat,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, visibility);
    entry.storageTexture.access = access;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = view_dimension;
    entry
}

fn default_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    }
}
