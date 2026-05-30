//! CTS: src/webgpu/api/validation/createBindGroupLayout.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::bind_group_common::{
    buffer_layout, dynamic_buffer_layout, expect_bind_group_layout, expect_pipeline_layout,
    release_bind_group_layouts, release_pipeline_layout, sampler_layout, storage_texture_layout,
    texture_layout,
};
use crate::common::device_limits;

#[test]
fn duplicate_bindings() {
    let test = ValidationTest::new();
    unsafe {
        for (entries, success) in [
            (
                vec![
                    buffer_layout(
                        0,
                        native::WGPUShaderStage_Compute,
                        native::WGPUBufferBindingType_Storage,
                    ),
                    buffer_layout(
                        1,
                        native::WGPUShaderStage_Compute,
                        native::WGPUBufferBindingType_Storage,
                    ),
                ],
                true,
            ),
            (
                vec![
                    buffer_layout(
                        0,
                        native::WGPUShaderStage_Compute,
                        native::WGPUBufferBindingType_Storage,
                    ),
                    buffer_layout(
                        0,
                        native::WGPUShaderStage_Compute,
                        native::WGPUBufferBindingType_Storage,
                    ),
                ],
                false,
            ),
        ] {
            let layout = expect_bind_group_layout(&test, success, &entries);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn maximum_binding_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (binding, success) in [
            (0, true),
            (limits.maxBindingsPerBindGroup - 1, true),
            (limits.maxBindingsPerBindGroup, false),
        ] {
            let entry = buffer_layout(
                binding,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Storage,
            );
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn visibility() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                vec![buffer_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUBufferBindingType_Uniform,
                )],
                true,
            ),
            (
                vec![sampler_layout(0, native::WGPUShaderStage_Fragment)],
                true,
            ),
            (
                vec![texture_layout(
                    0,
                    native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Fragment,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                )],
                true,
            ),
        ];
        for (entries, success) in cases {
            let layout = expect_bind_group_layout(&test, success, &entries);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn visibility_vertex_shader_stage_buffer_type() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                buffer_layout(
                    0,
                    native::WGPUShaderStage_Vertex,
                    native::WGPUBufferBindingType_Storage,
                ),
                false,
            ),
            (
                buffer_layout(
                    0,
                    native::WGPUShaderStage_Vertex,
                    native::WGPUBufferBindingType_ReadOnlyStorage,
                ),
                true,
            ),
            (
                buffer_layout(
                    0,
                    native::WGPUShaderStage_Fragment,
                    native::WGPUBufferBindingType_Storage,
                ),
                true,
            ),
        ];
        for (entry, success) in cases {
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn visibility_vertex_shader_stage_storage_texture_access() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                storage_texture_layout(
                    0,
                    native::WGPUShaderStage_Vertex,
                    native::WGPUStorageTextureAccess_ReadOnly,
                    native::WGPUTextureFormat_R32Uint,
                    native::WGPUTextureViewDimension_2D,
                ),
                true,
            ),
            (
                storage_texture_layout(
                    0,
                    native::WGPUShaderStage_Vertex,
                    native::WGPUStorageTextureAccess_WriteOnly,
                    native::WGPUTextureFormat_R32Uint,
                    native::WGPUTextureViewDimension_2D,
                ),
                false,
            ),
        ];
        for (entry, success) in cases {
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn multisampled_validation() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Uint,
                    native::WGPUTextureViewDimension_2D,
                    true,
                ),
                true,
            ),
            (
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    true,
                ),
                false,
            ),
            (
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Uint,
                    native::WGPUTextureViewDimension_2DArray,
                    true,
                ),
                false,
            ),
        ];
        for (entry, success) in cases {
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn max_dynamic_buffers() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (ty, limit) in [
            (
                native::WGPUBufferBindingType_Uniform,
                limits.maxDynamicUniformBuffersPerPipelineLayout,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                limits.maxDynamicStorageBuffersPerPipelineLayout,
            ),
        ] {
            let entries = (0..limit)
                .map(|binding| dynamic_buffer_layout(binding, native::WGPUShaderStage_Compute, ty))
                .collect::<Vec<_>>();
            let layout = expect_bind_group_layout(&test, true, &entries);
            yawgpu::wgpuBindGroupLayoutRelease(layout);

            let entries = (0..=limit)
                .map(|binding| dynamic_buffer_layout(binding, native::WGPUShaderStage_Compute, ty))
                .collect::<Vec<_>>();
            let layout = expect_bind_group_layout(&test, false, &entries);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn max_resources_per_stage_in_bind_group_layout() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let entries = (0..limits.maxSampledTexturesPerShaderStage)
            .map(|binding| {
                texture_layout(
                    binding,
                    native::WGPUShaderStage_Fragment,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let layout = expect_bind_group_layout(&test, true, &entries);
        yawgpu::wgpuBindGroupLayoutRelease(layout);

        let entries = (0..=limits.maxSampledTexturesPerShaderStage)
            .map(|binding| {
                texture_layout(
                    binding,
                    native::WGPUShaderStage_Fragment,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let layout = expect_bind_group_layout(&test, false, &entries);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn max_resources_per_stage_in_pipeline_layout() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let maxed_entries = (0..limits.maxSamplersPerShaderStage)
            .map(|binding| sampler_layout(binding, native::WGPUShaderStage_Fragment))
            .collect::<Vec<_>>();
        let maxed = expect_bind_group_layout(&test, true, &maxed_entries);
        let extra = expect_bind_group_layout(
            &test,
            true,
            &[sampler_layout(0, native::WGPUShaderStage_Fragment)],
        );
        let layout = expect_pipeline_layout(&test, false, &[maxed, extra], 0);
        release_pipeline_layout(layout);
        release_bind_group_layouts(&[extra, maxed]);
    }
}

#[test]
fn storage_texture_layout_dimension() {
    let test = ValidationTest::new();
    unsafe {
        for (dimension, success) in [
            (native::WGPUTextureViewDimension_2D, true),
            (native::WGPUTextureViewDimension_2DArray, true),
            (native::WGPUTextureViewDimension_Cube, false),
            (native::WGPUTextureViewDimension_CubeArray, false),
        ] {
            let entry = storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Unorm,
                dimension,
            );
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn storage_texture_formats() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUStorageTextureAccess_WriteOnly,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUStorageTextureAccess_WriteOnly,
                false,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUStorageTextureAccess_ReadWrite,
                false,
            ),
        ];
        for (format, access, success) in cases {
            let entry = storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                access,
                format,
                native::WGPUTextureViewDimension_2D,
            );
            let layout = expect_bind_group_layout(&test, success, &[entry]);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}
