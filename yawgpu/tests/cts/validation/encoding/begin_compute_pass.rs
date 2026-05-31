//! Ports `$CTS/src/webgpu/api/validation/encoding/beginComputePass.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, compute_pass_descriptor, create_encoder, create_query_set, expect_finish,
    timestamp_writes,
};

#[test]
fn timestamp_writes_query_set_type() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        for query_type in [
            native::WGPUQueryType_Occlusion,
            native::WGPUQueryType_Timestamp,
        ] {
            let query_set = create_query_set(test.device(), query_type, 2);
            let writes = timestamp_writes(query_set, 0, 1);
            let descriptor = compute_pass_descriptor(Some(&writes));
            try_compute_pass(
                &test,
                query_type == native::WGPUQueryType_Timestamp,
                &descriptor,
            );
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn timestamp_writes_invalid_query_set() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        for valid in [true, false] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 1);
            if !valid {
                yawgpu::wgpuQuerySetDestroy(query_set);
            }
            let writes = timestamp_writes(query_set, 0, native::WGPU_QUERY_SET_INDEX_UNDEFINED);
            let descriptor = compute_pass_descriptor(Some(&writes));
            try_compute_pass(&test, valid, &descriptor);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn timestamp_writes_query_index() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        for beginning in [native::WGPU_QUERY_SET_INDEX_UNDEFINED, 0, 1, 2, 3] {
            for end in [native::WGPU_QUERY_SET_INDEX_UNDEFINED, 0, 1, 2, 3] {
                let query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 2);
                let writes = timestamp_writes(query_set, beginning, end);
                let descriptor = compute_pass_descriptor(Some(&writes));
                let is_valid = beginning != end
                    && (beginning == native::WGPU_QUERY_SET_INDEX_UNDEFINED || beginning < 2)
                    && (end == native::WGPU_QUERY_SET_INDEX_UNDEFINED || end < 2);
                try_compute_pass(&test, is_valid, &descriptor);
                yawgpu::wgpuQuerySetRelease(query_set);
            }
        }
    }
}

#[test]
fn timestamp_query_set_device_mismatch() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    let foreign = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        for mismatched in [false, true] {
            let query_set = create_query_set(
                if mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUQueryType_Timestamp,
                1,
            );
            let writes = timestamp_writes(query_set, 0, native::WGPU_QUERY_SET_INDEX_UNDEFINED);
            let descriptor = compute_pass_descriptor(Some(&writes));
            try_compute_pass(&test, !mismatched, &descriptor);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

unsafe fn try_compute_pass(
    test: &ValidationTest,
    success: bool,
    descriptor: &native::WGPUComputePassDescriptor,
) {
    let encoder = create_encoder(test.device());
    test.clear_errors();
    let pass = begin_compute_pass(encoder, Some(descriptor));
    assert!(
        test.errors().is_empty(),
        "beginComputePass should defer descriptor validation to finish: {:?}",
        test.errors()
    );
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = expect_finish(test, encoder, success);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}
