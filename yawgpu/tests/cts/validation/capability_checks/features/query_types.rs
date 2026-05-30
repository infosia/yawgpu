use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::{common, feature_common};

#[test]
fn create_query_set() {
    unsafe {
        let default_test = ValidationTest::new();
        let occlusion =
            feature_common::create_query_set_ok(&default_test, native::WGPUQueryType_Occlusion);
        yawgpu::wgpuQuerySetRelease(occlusion);
        feature_common::create_query_set_error(&default_test, native::WGPUQueryType_Timestamp);

        feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_TimestampQuery);
        let timestamp_test =
            feature_common::test_with_feature(native::WGPUFeatureName_TimestampQuery);
        let timestamp =
            feature_common::create_query_set_ok(&timestamp_test, native::WGPUQueryType_Timestamp);
        yawgpu::wgpuQuerySetRelease(timestamp);
    }
}

#[test]
fn timestamp() {
    unsafe {
        feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_TimestampQuery);
        let test = feature_common::test_with_feature(native::WGPUFeatureName_TimestampQuery);
        let query_set = common::create_query_set(test.device(), native::WGPUQueryType_Timestamp, 1);
        let timestamp_writes = native::WGPUPassTimestampWrites {
            nextInChain: std::ptr::null_mut(),
            querySet: query_set,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
        };
        let descriptor = native::WGPUComputePassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: common::empty_string_view(),
            timestampWrites: &timestamp_writes,
        };
        let encoder = common::create_command_encoder(test.device());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuComputePassEncoderEnd(pass);
        let commands = common::finish_command_encoder(encoder);
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}
