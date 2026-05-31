//! CTS port of `webgpu/api/validation/encoding/cmds/setBindGroup.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, bundle_descriptor, color_attachment,
    compute_pass_descriptor, create_buffer, create_encoder, create_render_target,
    empty_string_view, expect_command_buffer, render_pass_descriptor, CommandExpectation,
};

#[test]
fn state_and_binding_index() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for encoder_type in ProgrammableEncoderType::all() {
            for index in [1, limits.maxBindGroups - 1, limits.maxBindGroups] {
                let (layout, group, buffer) = create_one_buffer_bind_group(
                    test.device(),
                    stage_for_encoder(encoder_type),
                    false,
                    4,
                );
                let expectation = if index < limits.maxBindGroups {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                };
                expect_set_bind_group(&test, encoder_type, expectation, index, group, &[]);
                release_bind_group_parts(layout, group, buffer);
            }
        }
    }
}

#[test]
fn bind_group_device_mismatch() {
    let test = ValidationTest::new();
    let other = ValidationTest::new();
    unsafe {
        for encoder_type in ProgrammableEncoderType::all() {
            for mismatched in [false, true] {
                let source_device = if mismatched {
                    other.device()
                } else {
                    test.device()
                };
                let (layout, group, buffer) = create_one_buffer_bind_group(
                    source_device,
                    stage_for_encoder(encoder_type),
                    true,
                    4,
                );
                let offsets = [0];
                expect_set_bind_group(
                    &test,
                    encoder_type,
                    if mismatched {
                        CommandExpectation::FinishError
                    } else {
                        CommandExpectation::Success
                    },
                    0,
                    group,
                    &offsets,
                );
                release_bind_group_parts(layout, group, buffer);
            }
        }
    }
}

#[test]
fn dynamic_offsets_passed_but_not_expected() {
    let test = ValidationTest::new();
    unsafe {
        for encoder_type in ProgrammableEncoderType::all() {
            let (layout, group, buffer) = create_one_buffer_bind_group(
                test.device(),
                stage_for_encoder(encoder_type),
                false,
                4,
            );
            expect_set_bind_group(
                &test,
                encoder_type,
                CommandExpectation::FinishError,
                0,
                group,
                &[0],
            );
            release_bind_group_parts(layout, group, buffer);
        }
    }
}

#[test]
fn dynamic_offsets_match_expectations_in_pass_encoder() {
    let test = ValidationTest::new();
    unsafe {
        let cases: &[(&[u32], bool)] = &[
            (&[256, 0], true),
            (&[1, 2], false),
            (&[256, 0, 0], false),
            (&[256], false),
            (&[], false),
            (&[512, 0], false),
            (&[1024, 0], false),
            (&[0xffff_ffff, 0], false),
            (&[0, 512], false),
            (&[0, 1024], false),
            (&[0, 0xffff_ffff], false),
        ];
        for encoder_type in ProgrammableEncoderType::all() {
            for (offsets, success) in cases {
                let (layout, group, first, second) = create_two_dynamic_buffer_bind_group(
                    test.device(),
                    stage_for_encoder(encoder_type),
                    false,
                );
                expect_set_bind_group(
                    &test,
                    encoder_type,
                    if *success {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                    0,
                    group,
                    offsets,
                );
                yawgpu::wgpuBindGroupRelease(group);
                yawgpu::wgpuBindGroupLayoutRelease(layout);
                yawgpu::wgpuBufferRelease(second);
                yawgpu::wgpuBufferRelease(first);
            }
        }
    }
}

#[test]
#[ignore = "the C ABI takes only a raw dynamic_offsets pointer and count; JavaScript Uint32Array start/length RangeError validation is not representable"]
fn u32array_start_and_length() {
    // The portable C surface has no `dynamicOffsetsDataStart` parameter.
}

#[test]
fn buffer_dynamic_offsets() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for encoder_type in ProgrammableEncoderType::all() {
            for ty in [
                native::WGPUBufferBindingType_Uniform,
                native::WGPUBufferBindingType_Storage,
                native::WGPUBufferBindingType_ReadOnlyStorage,
            ] {
                let alignment = if ty == native::WGPUBufferBindingType_Uniform {
                    limits.minUniformBufferOffsetAlignment
                } else {
                    limits.minStorageBufferOffsetAlignment
                };
                for dynamic_offset in [
                    alignment,
                    alignment / 2,
                    alignment + alignment / 2,
                    alignment * 2,
                    alignment + 2,
                ] {
                    let (layout, group, buffer) = create_dynamic_buffer_bind_group(
                        test.device(),
                        stage_for_encoder(encoder_type),
                        ty,
                    );
                    expect_set_bind_group(
                        &test,
                        encoder_type,
                        if dynamic_offset % alignment == 0 {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                        0,
                        group,
                        &[dynamic_offset],
                    );
                    release_bind_group_parts(layout, group, buffer);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ProgrammableEncoderType {
    ComputePass,
    RenderPass,
    RenderBundle,
}

impl ProgrammableEncoderType {
    const fn all() -> [Self; 3] {
        [Self::ComputePass, Self::RenderPass, Self::RenderBundle]
    }
}

unsafe fn expect_set_bind_group(
    test: &ValidationTest,
    encoder_type: ProgrammableEncoderType,
    expectation: CommandExpectation,
    index: u32,
    group: native::WGPUBindGroup,
    offsets: &[u32],
) {
    unsafe {
        match encoder_type {
            ProgrammableEncoderType::ComputePass => {
                let encoder = create_encoder(test.device());
                let descriptor = compute_pass_descriptor(None);
                let pass = begin_compute_pass(encoder, Some(&descriptor));
                yawgpu::wgpuComputePassEncoderSetBindGroup(
                    pass,
                    index,
                    group,
                    offsets.len(),
                    offsets.as_ptr(),
                );
                yawgpu::wgpuComputePassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuComputePassEncoderRelease(pass);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
            ProgrammableEncoderType::RenderPass => {
                let encoder = create_encoder(test.device());
                let target =
                    create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
                let attachment = color_attachment(target.view);
                let attachments = [attachment];
                let descriptor = render_pass_descriptor(&attachments, None);
                let pass = begin_render_pass(encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderSetBindGroup(
                    pass,
                    index,
                    group,
                    offsets.len(),
                    offsets.as_ptr(),
                );
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                yawgpu::wgpuTextureViewRelease(target.view);
                yawgpu::wgpuTextureRelease(target.texture);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
            ProgrammableEncoderType::RenderBundle => {
                let formats = [native::WGPUTextureFormat_RGBA8Unorm];
                let descriptor = bundle_descriptor(&formats);
                let encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                assert!(!encoder.is_null());
                yawgpu::wgpuRenderBundleEncoderSetBindGroup(
                    encoder,
                    index,
                    group,
                    offsets.len(),
                    offsets.as_ptr(),
                );
                match expectation {
                    CommandExpectation::Success => {
                        test.clear_errors();
                        let bundle =
                            yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
                        assert!(!bundle.is_null());
                        assert!(
                            test.errors().is_empty(),
                            "unexpected errors: {:?}",
                            test.errors()
                        );
                        yawgpu::wgpuRenderBundleRelease(bundle);
                    }
                    CommandExpectation::FinishError => {
                        let mut bundle = std::ptr::null();
                        test.assert_device_error_after(
                            || {
                                bundle = yawgpu::wgpuRenderBundleEncoderFinish(
                                    encoder,
                                    std::ptr::null(),
                                );
                            },
                            None,
                        );
                        assert!(!bundle.is_null());
                        yawgpu::wgpuRenderBundleRelease(bundle);
                    }
                    CommandExpectation::SubmitError => unreachable!(),
                }
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
            }
        }
    }
}

unsafe fn create_one_buffer_bind_group(
    device: native::WGPUDevice,
    visibility: native::WGPUShaderStage,
    dynamic: bool,
    size: u64,
) -> (
    native::WGPUBindGroupLayout,
    native::WGPUBindGroup,
    native::WGPUBuffer,
) {
    unsafe {
        let buffer = create_buffer(device, size, native::WGPUBufferUsage_Uniform);
        let layout = create_bind_group_layout(
            device,
            &[buffer_layout(
                0,
                visibility,
                native::WGPUBufferBindingType_Uniform,
                dynamic,
            )],
        );
        let group = create_bind_group(device, layout, &[buffer_binding(0, buffer, 0, size)]);
        (layout, group, buffer)
    }
}

unsafe fn create_dynamic_buffer_bind_group(
    device: native::WGPUDevice,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUBufferBindingType,
) -> (
    native::WGPUBindGroupLayout,
    native::WGPUBindGroup,
    native::WGPUBuffer,
) {
    unsafe {
        let usage = if ty == native::WGPUBufferBindingType_Uniform {
            native::WGPUBufferUsage_Uniform
        } else {
            native::WGPUBufferUsage_Storage
        };
        let buffer = create_buffer(device, 768, usage);
        let layout = create_bind_group_layout(device, &[buffer_layout(0, visibility, ty, true)]);
        let group = create_bind_group(device, layout, &[buffer_binding(0, buffer, 0, 12)]);
        (layout, group, buffer)
    }
}

unsafe fn create_two_dynamic_buffer_bind_group(
    device: native::WGPUDevice,
    visibility: native::WGPUShaderStage,
    use_storage: bool,
) -> (
    native::WGPUBindGroupLayout,
    native::WGPUBindGroup,
    native::WGPUBuffer,
    native::WGPUBuffer,
) {
    unsafe {
        let first = create_buffer(device, 520, native::WGPUBufferUsage_Uniform);
        let second_usage = if use_storage {
            native::WGPUBufferUsage_Storage
        } else {
            native::WGPUBufferUsage_Uniform
        };
        let second = create_buffer(device, 520, second_usage);
        let second_type = if use_storage {
            native::WGPUBufferBindingType_Storage
        } else {
            native::WGPUBufferBindingType_Uniform
        };
        let entries = [
            buffer_layout(0, visibility, native::WGPUBufferBindingType_Uniform, true),
            buffer_layout(1, visibility, second_type, true),
        ];
        let layout = create_bind_group_layout(device, &entries);
        let bindings = [
            buffer_binding(0, first, 0, 12),
            buffer_binding(1, second, 0, 12),
        ];
        let group = create_bind_group(device, layout, &bindings);
        (layout, group, first, second)
    }
}

unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    unsafe {
        let descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: entries.len(),
            entries: entries.as_ptr(),
        };
        let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
        assert!(!layout.is_null());
        layout
    }
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    unsafe {
        let descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: entries.len(),
            entries: entries.as_ptr(),
        };
        let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
        assert!(!group.is_null());
        group
    }
}

fn buffer_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUBufferBindingType,
    dynamic: bool,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: ty,
            hasDynamicOffset: u32::from(dynamic),
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

fn buffer_binding(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn stage_for_encoder(encoder_type: ProgrammableEncoderType) -> native::WGPUShaderStage {
    match encoder_type {
        ProgrammableEncoderType::ComputePass => native::WGPUShaderStage_Compute,
        ProgrammableEncoderType::RenderPass | ProgrammableEncoderType::RenderBundle => {
            native::WGPUShaderStage_Fragment
        }
    }
}

unsafe fn release_bind_group_parts(
    layout: native::WGPUBindGroupLayout,
    group: native::WGPUBindGroup,
    buffer: native::WGPUBuffer,
) {
    unsafe {
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut limits),
            native::WGPUStatus_Success
        );
        limits
    }
}
