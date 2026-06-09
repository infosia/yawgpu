//! CTS: src/webgpu/api/validation/createBindGroup.spec.ts
//!
//! The five external_texture,texture_view,* cases are web-only N/A for yawgpu:
//! C webgpu.h has no GPUExternalTexture binding resource analogue.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::bind_group_common::{
    buffer_binding, buffer_layout, create_buffer, create_sampler, create_texture_2d,
    create_texture_view, expect_bind_group, expect_bind_group_layout, release_bind_group,
    release_bind_group_layouts, sampler_binding, sampler_layout_typed, storage_texture_layout,
    texture_binding, texture_layout,
};
use crate::common::{device_limits, empty_string_view, request_device};

#[test]
fn binding_count_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        for (layout_entry_count, bind_group_entry_count) in [(1, 1), (1, 2), (2, 1), (3, 3)] {
            let layout_entries = (0..layout_entry_count)
                .map(|binding| {
                    buffer_layout(
                        binding,
                        native::WGPUShaderStage_Compute,
                        native::WGPUBufferBindingType_Storage,
                    )
                })
                .collect::<Vec<_>>();
            let layout = expect_bind_group_layout(&test, true, &layout_entries);
            let buffers = (0..bind_group_entry_count)
                .map(|_| create_buffer(test.device(), native::WGPUBufferUsage_Storage, 1024))
                .collect::<Vec<_>>();
            let entries = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| buffer_binding(binding as u32, *buffer, 0, 256))
                .collect::<Vec<_>>();

            let bind_group = expect_bind_group(
                &test,
                layout_entry_count == bind_group_entry_count,
                layout,
                &entries,
            );
            release_bind_group(bind_group);
            for buffer in buffers {
                yawgpu::wgpuBufferRelease(buffer);
            }
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn binding_must_be_present_in_layout() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 1024);
        for (layout_binding, binding) in [(0, 0), (0, 1), (2, 1)] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[buffer_layout(
                    layout_binding,
                    native::WGPUShaderStage_Compute,
                    native::WGPUBufferBindingType_Storage,
                )],
            );
            let entry = buffer_binding(binding, buffer, 0, 256);
            let bind_group = expect_bind_group(&test, layout_binding == binding, layout, &[entry]);
            release_bind_group(bind_group);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn binding_must_contain_resource_defined_in_layout() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let sampler = create_sampler(test.device(), native::WGPUCompareFunction_Undefined);
        let texture = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());

        let layouts_and_entries = [
            (
                buffer_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUBufferBindingType_Uniform,
                ),
                vec![
                    (buffer_binding(0, buffer, 0, 256), true),
                    (sampler_binding(0, sampler), false),
                    (texture_binding(0, view), false),
                ],
            ),
            (
                sampler_layout_typed(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUSamplerBindingType_Filtering,
                ),
                vec![
                    (sampler_binding(0, sampler), true),
                    (buffer_binding(0, buffer, 0, 256), false),
                    (texture_binding(0, view), false),
                ],
            ),
            (
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
                vec![
                    (texture_binding(0, view), true),
                    (buffer_binding(0, buffer, 0, 256), false),
                    (sampler_binding(0, sampler), false),
                ],
            ),
        ];

        for (layout_entry, cases) in layouts_and_entries {
            let layout = expect_bind_group_layout(&test, true, &[layout_entry]);
            for (entry, success) in cases {
                let bind_group = expect_bind_group(&test, success, layout, &[entry]);
                release_bind_group(bind_group);
            }
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }

        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn texture_binding_must_have_correct_usage() {
    let test = ValidationTest::new();
    unsafe {
        for (layout_entry, good_usage, bad_usage) in [
            (
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
                native::WGPUTextureUsage_TextureBinding,
                native::WGPUTextureUsage_CopyDst,
            ),
            (
                storage_texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUStorageTextureAccess_WriteOnly,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureViewDimension_2D,
                ),
                native::WGPUTextureUsage_StorageBinding,
                native::WGPUTextureUsage_TextureBinding,
            ),
        ] {
            let layout = expect_bind_group_layout(&test, true, &[layout_entry]);
            for (usage, success) in [(good_usage, true), (bad_usage, false)] {
                let texture = create_texture_2d(
                    test.device(),
                    usage,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    1,
                    1,
                    1,
                );
                let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
                let bind_group =
                    expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
                release_bind_group(bind_group);
                yawgpu::wgpuTextureViewRelease(view);
                yawgpu::wgpuTextureRelease(texture);
            }
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn texture_must_have_correct_component_type() {
    let test = ValidationTest::new();
    unsafe {
        for (sample_type, good_format, bad_format) in [
            (
                native::WGPUTextureSampleType_Float,
                native::WGPUTextureFormat_R8Unorm,
                native::WGPUTextureFormat_R8Uint,
            ),
            (
                native::WGPUTextureSampleType_Sint,
                native::WGPUTextureFormat_R8Sint,
                native::WGPUTextureFormat_R8Unorm,
            ),
            (
                native::WGPUTextureSampleType_Uint,
                native::WGPUTextureFormat_R8Uint,
                native::WGPUTextureFormat_R8Sint,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[texture_layout(
                    0,
                    native::WGPUShaderStage_Fragment,
                    sample_type,
                    native::WGPUTextureViewDimension_2D,
                    false,
                )],
            );
            for (format, success) in [(good_format, true), (bad_format, false)] {
                let texture = create_texture_2d(
                    test.device(),
                    native::WGPUTextureUsage_TextureBinding,
                    format,
                    1,
                    1,
                    1,
                );
                let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
                let bind_group =
                    expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
                release_bind_group(bind_group);
                yawgpu::wgpuTextureViewRelease(view);
                yawgpu::wgpuTextureRelease(texture);
            }
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn texture_must_have_correct_dimension() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                native::WGPUTextureUsage_TextureBinding,
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
                native::WGPUTextureViewDimension_2D,
                true,
            ),
            (
                native::WGPUTextureUsage_TextureBinding,
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
                native::WGPUTextureViewDimension_2DArray,
                false,
            ),
            (
                native::WGPUTextureUsage_StorageBinding,
                storage_texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUStorageTextureAccess_WriteOnly,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureViewDimension_2D,
                ),
                native::WGPUTextureViewDimension_2DArray,
                false,
            ),
        ];
        for (usage, layout_entry, view_dimension, success) in cases {
            let layout = expect_bind_group_layout(&test, true, &[layout_entry]);
            let texture = create_texture_2d(
                test.device(),
                usage,
                native::WGPUTextureFormat_RGBA8Unorm,
                1,
                2,
                1,
            );
            let view = create_texture_view(
                texture,
                view_dimension,
                native::WGPUTextureFormat_Undefined,
                0,
                native::WGPU_MIP_LEVEL_COUNT_UNDEFINED,
                0,
                if view_dimension == native::WGPUTextureViewDimension_2DArray {
                    2
                } else {
                    1
                },
            );
            let bind_group = expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
            yawgpu::wgpuTextureRelease(texture);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn multisampled_validation() {
    let test = ValidationTest::new();
    unsafe {
        for (multisampled, sample_count, success) in [
            (false, 1, true),
            (false, 4, false),
            (true, 1, false),
            (true, 4, true),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_UnfilterableFloat,
                    native::WGPUTextureViewDimension_2D,
                    multisampled,
                )],
            );
            let texture = create_texture_2d(
                test.device(),
                native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                sample_count,
                1,
                1,
            );
            let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
            let bind_group = expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
            yawgpu::wgpuTextureRelease(texture);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn buffer_offset_and_size_for_bind_groups_match() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Storage,
            )],
        );
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 1024);
        for (offset, size, success) in [
            (0, 512, true),
            (256, 256, true),
            (0, u64::MAX, true),
            (0, 0, false),
            (1, 256, false),
            (0, 1280, false),
            (1024, 1, false),
        ] {
            let bind_group = expect_bind_group(
                &test,
                success,
                layout,
                &[buffer_binding(0, buffer, offset, size)],
            );
            release_bind_group(bind_group);
        }
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn min_binding_size() {
    let test = ValidationTest::new();
    unsafe {
        for (min_binding_size, size, success) in
            [(0, 4, true), (8, 4, false), (8, 8, true), (8, 12, true)]
        {
            let mut entry = buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Storage,
            );
            entry.buffer.minBindingSize = min_binding_size;
            let layout = expect_bind_group_layout(&test, true, &[entry]);
            let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, size);
            let bind_group = expect_bind_group(
                &test,
                success,
                layout,
                &[buffer_binding(0, buffer, 0, u64::MAX)],
            );
            release_bind_group(bind_group);
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn buffer_resource_state() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Storage,
            )],
        );
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 256);
        let bind_group = expect_bind_group(&test, true, layout, &[buffer_binding(0, buffer, 0, 4)]);
        release_bind_group(bind_group);
        yawgpu::wgpuBufferDestroy(buffer);
        let bind_group =
            expect_bind_group(&test, false, layout, &[buffer_binding(0, buffer, 0, 4)]);
        release_bind_group(bind_group);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn texture_resource_state() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUTextureSampleType_Float,
                native::WGPUTextureViewDimension_2D,
                false,
            )],
        );
        let texture = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let bind_group = expect_bind_group(&test, true, layout, &[texture_binding(0, view)]);
        release_bind_group(bind_group);
        yawgpu::wgpuTextureDestroy(texture);
        let bind_group = expect_bind_group(&test, false, layout, &[texture_binding(0, view)]);
        release_bind_group(bind_group);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn bind_group_layout_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let foreign_layout = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let local_layout = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let foreign_layout = {
            let entries = [buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )];
            let descriptor = native::WGPUBindGroupLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: crate::common::empty_string_view(),
                entryCount: entries.len(),
                entries: entries.as_ptr(),
            };
            yawgpu::wgpuBindGroupLayoutRelease(foreign_layout);
            yawgpu::wgpuDeviceCreateBindGroupLayout(other, &descriptor)
        };
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 256);
        for (layout, success) in [(local_layout, true), (foreign_layout, false)] {
            let bind_group =
                expect_bind_group(&test, success, layout, &[buffer_binding(0, buffer, 0, 4)]);
            release_bind_group(bind_group);
        }
        release_bind_group_layouts(&[foreign_layout, local_layout]);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn binding_resources_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let local_buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 256);
        let foreign_buffer = create_buffer(other, native::WGPUBufferUsage_Storage, 256);
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[
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
        );
        for (entry0, entry1, success) in [
            (
                buffer_binding(0, local_buffer, 0, 4),
                buffer_binding(1, local_buffer, 0, 4),
                true,
            ),
            (
                buffer_binding(0, foreign_buffer, 0, 4),
                buffer_binding(1, local_buffer, 0, 4),
                false,
            ),
            (
                buffer_binding(0, local_buffer, 0, 4),
                buffer_binding(1, foreign_buffer, 0, 4),
                false,
            ),
        ] {
            let bind_group = expect_bind_group(&test, success, layout, &[entry0, entry1]);
            release_bind_group(bind_group);
        }
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(foreign_buffer);
        yawgpu::wgpuBufferRelease(local_buffer);

        let local_sampler = create_sampler(test.device(), native::WGPUCompareFunction_Undefined);
        let foreign_sampler = create_sampler(other, native::WGPUCompareFunction_Undefined);
        let sampler_layout = expect_bind_group_layout(
            &test,
            true,
            &[
                sampler_layout_typed(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUSamplerBindingType_Filtering,
                ),
                sampler_layout_typed(
                    1,
                    native::WGPUShaderStage_Compute,
                    native::WGPUSamplerBindingType_Filtering,
                ),
            ],
        );
        for (entry0, entry1, success) in [
            (
                sampler_binding(0, local_sampler),
                sampler_binding(1, local_sampler),
                true,
            ),
            (
                sampler_binding(0, foreign_sampler),
                sampler_binding(1, local_sampler),
                false,
            ),
            (
                sampler_binding(0, local_sampler),
                sampler_binding(1, foreign_sampler),
                false,
            ),
        ] {
            let bind_group = expect_bind_group(&test, success, sampler_layout, &[entry0, entry1]);
            release_bind_group(bind_group);
        }
        yawgpu::wgpuBindGroupLayoutRelease(sampler_layout);
        yawgpu::wgpuSamplerRelease(foreign_sampler);
        yawgpu::wgpuSamplerRelease(local_sampler);

        let local_texture = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let foreign_texture = create_texture_2d(
            other,
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let local_view = yawgpu::wgpuTextureCreateView(local_texture, std::ptr::null());
        let foreign_view = yawgpu::wgpuTextureCreateView(foreign_texture, std::ptr::null());
        let texture_layout = expect_bind_group_layout(
            &test,
            true,
            &[
                texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
                texture_layout(
                    1,
                    native::WGPUShaderStage_Compute,
                    native::WGPUTextureSampleType_Float,
                    native::WGPUTextureViewDimension_2D,
                    false,
                ),
            ],
        );
        for (entry0, entry1, success) in [
            (
                texture_binding(0, local_view),
                texture_binding(1, local_view),
                true,
            ),
            (
                texture_binding(0, foreign_view),
                texture_binding(1, local_view),
                false,
            ),
            (
                texture_binding(0, local_view),
                texture_binding(1, foreign_view),
                false,
            ),
        ] {
            let bind_group = expect_bind_group(&test, success, texture_layout, &[entry0, entry1]);
            release_bind_group(bind_group);
        }
        yawgpu::wgpuBindGroupLayoutRelease(texture_layout);
        yawgpu::wgpuTextureViewRelease(foreign_view);
        yawgpu::wgpuTextureViewRelease(local_view);
        yawgpu::wgpuTextureRelease(foreign_texture);
        yawgpu::wgpuTextureRelease(local_texture);

        let local_storage = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let foreign_storage = create_texture_2d(
            other,
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            1,
        );
        let local_storage_view = yawgpu::wgpuTextureCreateView(local_storage, std::ptr::null());
        let foreign_storage_view = yawgpu::wgpuTextureCreateView(foreign_storage, std::ptr::null());
        let storage_layout = expect_bind_group_layout(
            &test,
            true,
            &[
                storage_texture_layout(
                    0,
                    native::WGPUShaderStage_Compute,
                    native::WGPUStorageTextureAccess_WriteOnly,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureViewDimension_2D,
                ),
                storage_texture_layout(
                    1,
                    native::WGPUShaderStage_Compute,
                    native::WGPUStorageTextureAccess_WriteOnly,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureViewDimension_2D,
                ),
            ],
        );
        for (entry0, entry1, success) in [
            (
                texture_binding(0, local_storage_view),
                texture_binding(1, local_storage_view),
                true,
            ),
            (
                texture_binding(0, foreign_storage_view),
                texture_binding(1, local_storage_view),
                false,
            ),
            (
                texture_binding(0, local_storage_view),
                texture_binding(1, foreign_storage_view),
                false,
            ),
        ] {
            let bind_group = expect_bind_group(&test, success, storage_layout, &[entry0, entry1]);
            release_bind_group(bind_group);
        }
        yawgpu::wgpuBindGroupLayoutRelease(storage_layout);
        yawgpu::wgpuTextureViewRelease(foreign_storage_view);
        yawgpu::wgpuTextureViewRelease(local_storage_view);
        yawgpu::wgpuTextureRelease(foreign_storage);
        yawgpu::wgpuTextureRelease(local_storage);

        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn storage_texture_usage() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        for (usage, success) in [
            (native::WGPUTextureUsage_StorageBinding, true),
            (native::WGPUTextureUsage_TextureBinding, false),
        ] {
            let texture = create_texture_2d(
                test.device(),
                usage,
                native::WGPUTextureFormat_RGBA8Unorm,
                1,
                1,
                1,
            );
            let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
            let bind_group = expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn storage_texture_mip_level_count() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        let texture = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
            1,
            4,
        );
        for (mip_level_count, success) in [(1, true), (2, false)] {
            let view = create_texture_view(
                texture,
                native::WGPUTextureViewDimension_2D,
                native::WGPUTextureFormat_Undefined,
                1,
                mip_level_count,
                0,
                1,
            );
            let bind_group = expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
        }
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn storage_texture_format() {
    let test = ValidationTest::new();
    unsafe {
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        for (format, success) in [
            (native::WGPUTextureFormat_RGBA8Unorm, true),
            (native::WGPUTextureFormat_RGBA8Uint, false),
        ] {
            let texture = create_texture_2d(
                test.device(),
                native::WGPUTextureUsage_StorageBinding,
                format,
                1,
                1,
                1,
            );
            let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
            let bind_group = expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for (ty, good_usage, bad_usage) in [
            (
                native::WGPUBufferBindingType_Uniform,
                native::WGPUBufferUsage_Uniform,
                native::WGPUBufferUsage_CopyDst,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferUsage_Storage,
                native::WGPUBufferUsage_CopyDst,
            ),
            (
                native::WGPUBufferBindingType_ReadOnlyStorage,
                native::WGPUBufferUsage_Storage,
                native::WGPUBufferUsage_CopyDst,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[buffer_layout(0, native::WGPUShaderStage_Compute, ty)],
            );
            for (usage, success) in [(good_usage, true), (bad_usage, false)] {
                let buffer = create_buffer(test.device(), usage, 256);
                let bind_group =
                    expect_bind_group(&test, success, layout, &[buffer_binding(0, buffer, 0, 4)]);
                release_bind_group(bind_group);
                yawgpu::wgpuBufferRelease(buffer);
            }
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn buffer_resource_offset() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (ty, usage, alignment) in [
            (
                native::WGPUBufferBindingType_Uniform,
                native::WGPUBufferUsage_Uniform,
                limits.minUniformBufferOffsetAlignment,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferUsage_Storage,
                limits.minStorageBufferOffsetAlignment,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[buffer_layout(0, native::WGPUShaderStage_Compute, ty)],
            );
            let buffer = create_buffer(test.device(), usage, 1024);
            for (offset, success) in [(0, true), (u64::from(alignment), true), (1, false)] {
                let bind_group = expect_bind_group(
                    &test,
                    success,
                    layout,
                    &[buffer_binding(0, buffer, offset, 4)],
                );
                release_bind_group(bind_group);
            }
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn buffer_resource_binding_size() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (ty, usage, max_size) in [
            (
                native::WGPUBufferBindingType_Uniform,
                native::WGPUBufferUsage_Uniform,
                limits.maxUniformBufferBindingSize,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferUsage_Storage,
                limits.maxStorageBufferBindingSize,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[buffer_layout(0, native::WGPUShaderStage_Compute, ty)],
            );
            let buffer = create_buffer(test.device(), usage, max_size + 4);
            for (size, success) in [(4, true), (max_size, true), (max_size + 4, false)] {
                let bind_group = expect_bind_group(
                    &test,
                    success,
                    layout,
                    &[buffer_binding(0, buffer, 0, size)],
                );
                release_bind_group(bind_group);
            }
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn buffer_effective_buffer_binding_size() {
    let test = ValidationTest::new();
    unsafe {
        for (ty, usage, size, success) in [
            (
                native::WGPUBufferBindingType_Uniform,
                native::WGPUBufferUsage_Uniform,
                2,
                true,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferUsage_Storage,
                4,
                true,
            ),
            (
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferUsage_Storage,
                2,
                false,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[buffer_layout(0, native::WGPUShaderStage_Compute, ty)],
            );
            let buffer = create_buffer(test.device(), usage, 256);
            let bind_group = expect_bind_group(
                &test,
                success,
                layout,
                &[buffer_binding(0, buffer, 0, size)],
            );
            release_bind_group(bind_group);
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn sampler_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let layout = expect_bind_group_layout(
            &test,
            true,
            &[sampler_layout_typed(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUSamplerBindingType_Filtering,
            )],
        );
        let local_sampler = create_sampler(test.device(), native::WGPUCompareFunction_Undefined);
        let foreign_sampler = create_sampler(other, native::WGPUCompareFunction_Undefined);
        for (sampler, success) in [(local_sampler, true), (foreign_sampler, false)] {
            let bind_group =
                expect_bind_group(&test, success, layout, &[sampler_binding(0, sampler)]);
            release_bind_group(bind_group);
        }
        yawgpu::wgpuSamplerRelease(foreign_sampler);
        yawgpu::wgpuSamplerRelease(local_sampler);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn sampler_compare_function_with_binding_type() {
    let test = ValidationTest::new();
    unsafe {
        for (binding_type, compare, success) in [
            (
                native::WGPUSamplerBindingType_Filtering,
                native::WGPUCompareFunction_Undefined,
                true,
            ),
            (
                native::WGPUSamplerBindingType_Filtering,
                native::WGPUCompareFunction_Less,
                false,
            ),
            (
                native::WGPUSamplerBindingType_Comparison,
                native::WGPUCompareFunction_Less,
                true,
            ),
            (
                native::WGPUSamplerBindingType_Comparison,
                native::WGPUCompareFunction_Undefined,
                false,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[sampler_layout_typed(
                    0,
                    native::WGPUShaderStage_Compute,
                    binding_type,
                )],
            );
            let sampler = create_sampler(test.device(), compare);
            let bind_group =
                expect_bind_group(&test, success, layout, &[sampler_binding(0, sampler)]);
            release_bind_group(bind_group);
            yawgpu::wgpuSamplerRelease(sampler);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

/// Regression for F-055: the depth/stencil **aspect** of a depth-stencil
/// texture view must be bindable to the sample type its aspect supports.
/// WebGPU sample-type compatibility is aspect-specific: a `DepthOnly` view
/// supports `depth` and `unfilterable-float` (a `texture_2d<f32>` accessed via
/// `textureLoad`), a `StencilOnly` view supports `uint`. yawgpu previously
/// validated only the whole format's output class (which is `None` for a
/// depth-stencil format), wrongly rejecting both — so binding the aspect views
/// the CTS `readonly_depth_stencil:sampling_while_testing` test uses failed,
/// invalidating the command buffer and reading back 0.
#[test]
fn depth_stencil_aspect_sample_type_compat() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture_2d(
            test.device(),
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            1,
            1,
        );

        let make_aspect_view = |aspect: native::WGPUTextureAspect| {
            let descriptor = native::WGPUTextureViewDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                dimension: native::WGPUTextureViewDimension_2D,
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: 1,
                aspect,
                usage: native::WGPUTextureUsage_None,
            };
            yawgpu::wgpuTextureCreateView(texture, &descriptor)
        };

        // (view aspect, layout sample type, accepted?)
        for (aspect, sample_type, success) in [
            // depth aspect → depth (texture_depth_2d) and unfilterable-float
            // (texture_2d<f32> via textureLoad) are both accepted.
            (
                native::WGPUTextureAspect_DepthOnly,
                native::WGPUTextureSampleType_Depth,
                true,
            ),
            (
                native::WGPUTextureAspect_DepthOnly,
                native::WGPUTextureSampleType_UnfilterableFloat,
                true,
            ),
            // depth aspect is not uint/sint.
            (
                native::WGPUTextureAspect_DepthOnly,
                native::WGPUTextureSampleType_Uint,
                false,
            ),
            // stencil aspect → uint (texture_2d<u32>) only.
            (
                native::WGPUTextureAspect_StencilOnly,
                native::WGPUTextureSampleType_Uint,
                true,
            ),
            (
                native::WGPUTextureAspect_StencilOnly,
                native::WGPUTextureSampleType_Depth,
                false,
            ),
            (
                native::WGPUTextureAspect_StencilOnly,
                native::WGPUTextureSampleType_UnfilterableFloat,
                false,
            ),
        ] {
            let layout = expect_bind_group_layout(
                &test,
                true,
                &[texture_layout(
                    0,
                    native::WGPUShaderStage_Fragment,
                    sample_type,
                    native::WGPUTextureViewDimension_2D,
                    false,
                )],
            );
            let view = make_aspect_view(aspect);
            let bind_group =
                expect_bind_group(&test, success, layout, &[texture_binding(0, view)]);
            release_bind_group(bind_group);
            yawgpu::wgpuTextureViewRelease(view);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }

        yawgpu::wgpuTextureRelease(texture);
    }
}
