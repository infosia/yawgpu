//! CTS port of `webgpu/api/validation/encoding/queries/resolveQuerySet.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    create_buffer, create_encoder, create_query_set, expect_command_buffer, CommandExpectation,
};

const QUERY_COUNT: u32 = 2;

#[test]
#[ignore = "core reports destroyed query sets and destination buffers at finish; CTS expects destroyed resources to fail at queue submit"]
fn queryset_and_destination_buffer_state() {
    let test = ValidationTest::new();
    unsafe {
        for query_state in ResourceState::all() {
            for destination_state in ResourceState::all() {
                let query_set = create_query_set_with_state(&test, query_state);
                let destination = create_buffer_with_state(
                    &test,
                    destination_state,
                    16,
                    native::WGPUBufferUsage_QueryResolve,
                );
                let finish_success = query_state != ResourceState::Invalid
                    && destination_state != ResourceState::Invalid;
                let submit_success = query_state == ResourceState::Valid
                    && destination_state == ResourceState::Valid;
                expect_resolve(
                    &test,
                    query_set,
                    0,
                    1,
                    destination,
                    0,
                    if finish_success {
                        if submit_success {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::SubmitError
                        }
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuBufferRelease(destination);
                yawgpu::wgpuQuerySetRelease(query_set);
            }
        }
    }
}

#[test]
fn first_query_and_query_count() {
    let test = ValidationTest::new();
    unsafe {
        for (first_query, query_count) in [
            (0, QUERY_COUNT),
            (0, QUERY_COUNT + 1),
            (1, QUERY_COUNT),
            (QUERY_COUNT, 1),
        ] {
            let query_set =
                create_query_set(test.device(), native::WGPUQueryType_Occlusion, QUERY_COUNT);
            let destination =
                create_buffer(test.device(), 16, native::WGPUBufferUsage_QueryResolve);
            expect_resolve(
                &test,
                query_set,
                first_query,
                query_count,
                destination,
                0,
                if first_query + query_count <= QUERY_COUNT {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(destination);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn destination_buffer_usage() {
    let test = ValidationTest::new();
    unsafe {
        for usage in [
            native::WGPUBufferUsage_Storage,
            native::WGPUBufferUsage_QueryResolve,
        ] {
            let query_set =
                create_query_set(test.device(), native::WGPUQueryType_Occlusion, QUERY_COUNT);
            let destination = create_buffer(test.device(), 16, usage);
            expect_resolve(
                &test,
                query_set,
                0,
                QUERY_COUNT,
                destination,
                0,
                if usage == native::WGPUBufferUsage_QueryResolve {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(destination);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn destination_offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for offset in [0, 128, 256, 384] {
            let query_set =
                create_query_set(test.device(), native::WGPUQueryType_Occlusion, QUERY_COUNT);
            let destination =
                create_buffer(test.device(), 512, native::WGPUBufferUsage_QueryResolve);
            expect_resolve(
                &test,
                query_set,
                0,
                QUERY_COUNT,
                destination,
                offset,
                if offset % 256 == 0 {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(destination);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn resolve_buffer_oob() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (2, 16, 0, true),
            (3, 16, 0, false),
            (2, 16, 256, false),
            (2, 272, 256, true),
            (2, 264, 256, false),
        ];
        for (query_count, buffer_size, offset, success) in cases {
            let query_set =
                create_query_set(test.device(), native::WGPUQueryType_Occlusion, query_count);
            let destination = create_buffer(
                test.device(),
                buffer_size,
                native::WGPUBufferUsage_QueryResolve,
            );
            expect_resolve(
                &test,
                query_set,
                0,
                query_count,
                destination,
                offset,
                if success {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuBufferRelease(destination);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn query_set_buffer_device_mismatch() {
    let test = ValidationTest::new();
    let other = ValidationTest::new();
    unsafe {
        for (query_mismatch, buffer_mismatch) in [(false, false), (true, false), (false, true)] {
            let query_device = if query_mismatch {
                other.device()
            } else {
                test.device()
            };
            let buffer_device = if buffer_mismatch {
                other.device()
            } else {
                test.device()
            };
            let query_set = create_query_set(query_device, native::WGPUQueryType_Occlusion, 1);
            let destination = create_buffer(buffer_device, 8, native::WGPUBufferUsage_QueryResolve);
            expect_resolve(
                &test,
                query_set,
                0,
                1,
                destination,
                0,
                if query_mismatch || buffer_mismatch {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
            );
            yawgpu::wgpuBufferRelease(destination);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceState {
    Valid,
    Invalid,
    Destroyed,
}

impl ResourceState {
    const fn all() -> [Self; 3] {
        [Self::Valid, Self::Invalid, Self::Destroyed]
    }
}

unsafe fn expect_resolve(
    test: &ValidationTest,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    expectation: CommandExpectation,
) {
    unsafe {
        let encoder = create_encoder(test.device());
        yawgpu::wgpuCommandEncoderResolveQuerySet(
            encoder,
            query_set,
            first_query,
            query_count,
            destination,
            destination_offset,
        );
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn create_query_set_with_state(
    test: &ValidationTest,
    state: ResourceState,
) -> native::WGPUQuerySet {
    unsafe {
        let query_set = if state == ResourceState::Invalid {
            let descriptor = native::WGPUQuerySetDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: crate::common::empty_string_view(),
                type_: native::WGPUQueryType_Occlusion,
                count: 0,
            };
            let mut query_set = std::ptr::null();
            test.assert_device_error_after(
                || {
                    query_set = yawgpu::wgpuDeviceCreateQuerySet(test.device(), &descriptor);
                },
                None,
            );
            assert!(!query_set.is_null());
            query_set
        } else {
            create_query_set(test.device(), native::WGPUQueryType_Occlusion, QUERY_COUNT)
        };
        if state == ResourceState::Destroyed {
            yawgpu::wgpuQuerySetDestroy(query_set);
        }
        query_set
    }
}

unsafe fn create_buffer_with_state(
    test: &ValidationTest,
    state: ResourceState,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    unsafe {
        let buffer = if state == ResourceState::Invalid {
            let descriptor = native::WGPUBufferDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: crate::common::empty_string_view(),
                usage: 0,
                size,
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
        } else {
            create_buffer(test.device(), size, usage)
        };
        if state == ResourceState::Destroyed {
            yawgpu::wgpuBufferDestroy(buffer);
        }
        buffer
    }
}
