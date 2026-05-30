//! CTS port of `webgpu/api/validation/encoding/cmds/compute_pass.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, compute_pass_descriptor, create_buffer, create_encoder, create_wgsl_module,
    empty_string_view, expect_command_buffer, CommandExpectation,
};

#[test]
fn set_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        for valid in [true, false] {
            let pipeline = if valid {
                create_compute_pipeline_on_device(test.device(), None)
            } else {
                create_error_compute_pipeline(&test)
            };
            expect_compute_commands(
                &test,
                if valid {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
                |pass| {
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                },
            );
            yawgpu::wgpuComputePipelineRelease(pipeline);
        }
    }
}

#[test]
fn pipeline_device_mismatch() {
    let test = ValidationTest::new();
    let other = ValidationTest::new();
    unsafe {
        for mismatched in [false, true] {
            let source_device = if mismatched {
                other.device()
            } else {
                test.device()
            };
            let pipeline = create_compute_pipeline_on_device(source_device, None);
            expect_compute_commands(
                &test,
                if mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
                |pass| {
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                },
            );
            yawgpu::wgpuComputePipelineRelease(pipeline);
        }
    }
}

#[test]
fn dispatch_sizes() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let max = limits.maxComputeWorkgroupsPerDimension;
        let variants = [
            max_variant(max, 0, 0),
            max_variant(max, 0, 1),
            max_variant(max, 1, 0),
            max_variant(max, 1, 1),
            0x7fff_ffff,
            0xffff_ffff,
        ];
        for dispatch_type in [DispatchType::Direct, DispatchType::Indirect] {
            for large_index in 0..3 {
                for small in [0, 1] {
                    for large in variants {
                        let mut workgroups = [small, small, small];
                        workgroups[large_index] = large;
                        let should_error = dispatch_type == DispatchType::Direct
                            && workgroups.iter().any(|value| *value > max);
                        expect_compute_dispatch(
                            &test,
                            if should_error {
                                CommandExpectation::FinishError
                            } else {
                                CommandExpectation::Success
                            },
                            dispatch_type,
                            workgroups,
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "core reports destroyed indirect dispatch buffers at finish; CTS expects finish success and queue-submit failure for destroyed buffers"]
fn indirect_dispatch_buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        let offsets = [0, 4, 12, 1, 16];
        for state in [
            ResourceState::Valid,
            ResourceState::Invalid,
            ResourceState::Destroyed,
        ] {
            for offset in offsets {
                let buffer = match state {
                    ResourceState::Valid | ResourceState::Destroyed => {
                        create_buffer(test.device(), 24, native::WGPUBufferUsage_Indirect)
                    }
                    ResourceState::Invalid => create_error_buffer(&test),
                };
                if state == ResourceState::Destroyed {
                    yawgpu::wgpuBufferDestroy(buffer);
                }
                let finish_error =
                    state == ResourceState::Invalid || offset % 4 != 0 || offset + 12 > 24;
                let expectation = if finish_error {
                    CommandExpectation::FinishError
                } else if state == ResourceState::Destroyed {
                    CommandExpectation::SubmitError
                } else {
                    CommandExpectation::Success
                };
                expect_compute_commands(&test, expectation, |pass| {
                    let pipeline = create_compute_pipeline_on_device(test.device(), None);
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                    yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, buffer, offset);
                    yawgpu::wgpuComputePipelineRelease(pipeline);
                });
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn indirect_dispatch_buffer_device_mismatch() {
    let test = ValidationTest::new();
    let other = ValidationTest::new();
    unsafe {
        for mismatched in [false, true] {
            let source_device = if mismatched {
                other.device()
            } else {
                test.device()
            };
            let buffer = create_buffer(source_device, 16, native::WGPUBufferUsage_Indirect);
            expect_compute_commands(
                &test,
                if mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
                |pass| {
                    let pipeline = create_compute_pipeline_on_device(test.device(), None);
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                    yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, buffer, 0);
                    yawgpu::wgpuComputePipelineRelease(pipeline);
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn indirect_dispatch_buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        let usages = [
            native::WGPUBufferUsage_CopySrc,
            native::WGPUBufferUsage_CopyDst,
            native::WGPUBufferUsage_Index,
            native::WGPUBufferUsage_Vertex,
            native::WGPUBufferUsage_Uniform,
            native::WGPUBufferUsage_Storage,
            native::WGPUBufferUsage_Indirect,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_Indirect,
            native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_Uniform,
        ];
        for usage in usages {
            let buffer = create_buffer(test.device(), 16, usage);
            let success = usage & native::WGPUBufferUsage_Indirect != 0;
            expect_compute_commands(
                &test,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
                |pass| {
                    let pipeline = create_compute_pipeline_on_device(test.device(), None);
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                    yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, buffer, 0);
                    yawgpu::wgpuComputePipelineRelease(pipeline);
                },
            );
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DispatchType {
    Direct,
    Indirect,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceState {
    Valid,
    Invalid,
    Destroyed,
}

unsafe fn expect_compute_dispatch(
    test: &ValidationTest,
    expectation: CommandExpectation,
    dispatch_type: DispatchType,
    workgroups: [u32; 3],
) {
    unsafe {
        let indirect = create_buffer(test.device(), 12, native::WGPUBufferUsage_Indirect);
        expect_compute_commands(test, expectation, |pass| {
            let pipeline = create_compute_pipeline_on_device(test.device(), None);
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
            match dispatch_type {
                DispatchType::Direct => yawgpu::wgpuComputePassEncoderDispatchWorkgroups(
                    pass,
                    workgroups[0],
                    workgroups[1],
                    workgroups[2],
                ),
                DispatchType::Indirect => {
                    yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 0);
                }
            }
            yawgpu::wgpuComputePipelineRelease(pipeline);
        });
        yawgpu::wgpuBufferRelease(indirect);
    }
}

unsafe fn expect_compute_commands<F>(
    test: &ValidationTest,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    unsafe {
        let encoder = create_encoder(test.device());
        let descriptor = compute_pass_descriptor(None);
        let pass = begin_compute_pass(encoder, Some(&descriptor));
        commands(pass);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn create_compute_pipeline_on_device(
    device: native::WGPUDevice,
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    unsafe {
        let module = create_wgsl_module(device, "@compute @workgroup_size(1) fn main() {}");
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: layout.unwrap_or(std::ptr::null()),
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: empty_string_view(),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        };
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
        assert!(!pipeline.is_null());
        yawgpu::wgpuShaderModuleRelease(module);
        pipeline
    }
}

unsafe fn create_error_compute_pipeline(test: &ValidationTest) -> native::WGPUComputePipeline {
    unsafe {
        let module = create_wgsl_module(
            test.device(),
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }",
        );
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: empty_string_view(),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        };
        let mut pipeline = std::ptr::null();
        test.assert_device_error_after(
            || {
                pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
            },
            None,
        );
        assert!(!pipeline.is_null());
        yawgpu::wgpuShaderModuleRelease(module);
        pipeline
    }
}

unsafe fn create_error_buffer(test: &ValidationTest) -> native::WGPUBuffer {
    unsafe {
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: 0,
            size: 24,
            mappedAtCreation: 0,
        };
        let mut buffer = std::ptr::null();
        test.assert_device_error_after(
            || {
                buffer = yawgpu::wgpuDeviceCreateBuffer(test.device(), &descriptor);
            },
            None,
        );
        assert!(!buffer.is_null());
        buffer
    }
}

fn max_variant(limit: u32, mult: u32, add: u32) -> u32 {
    limit.saturating_mul(mult).saturating_add(add)
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
